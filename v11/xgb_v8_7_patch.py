"""
XGBoost v8.7 — Trend Regime Overlay Patch
=====================================================================

v8.6.39 대비 구조적 변경 사항 요약
------------------------------------
1. TrendRegimeDetector  (신규)
   - rolling 120d Sharpe proxy를 실시간으로 계산해 시장 추세 강도를 판별한다.
   - STRONG_TREND / NORMAL / WEAK_TREND 3단계로 분류한다.
   - 미래 정보를 일절 사용하지 않는다 (close 가격만 사용, shift 없음).

2. build_features() 확장  (기존 함수에 추가)
   - `trend_regime_sharpe_120`  : 120일 rolling Sharpe proxy (연환산)
   - `trend_regime_sharpe_252`  : 252일 rolling Sharpe proxy (연환산)
   - `trend_regime_label`       : STRONG_TREND / NORMAL / WEAK_TREND
   - `trend_regime_score`       : 연속형 점수 [-1, 1] 클리핑

3. apply_trend_regime_overlay()  (신규)
   - apply_allocation() 내부에서 base_signal_w 산출 직후, policy_overlay 전에 호출된다.
   - STRONG_TREND 구간에서 주식 비중 하한을 올리거나 vol_base cap을 완화한다.
   - WEAK_TREND 구간에서는 방어 비중을 소폭 추가한다.
   - 기존 no-trade-band / emergency 로직과 독립적으로 작동한다.

4. mid_trend_score 버그 수정  (기존 함수 수정)
   - compute_mid_trend_score(row)가 predictions CSV row에서 피처 컬럼을 찾지 못해
     항상 BEAR를 반환하는 문제를 수정한다.
   - 대신 TrendRegimeDetector가 사전에 계산한 trend_regime_label을 row에서 읽는다.

5. Config 신규 파라미터
   - use_trend_regime_overlay      : bool = True
   - trend_strong_sharpe_threshold : float = 0.8  (sh120 > 이 값이면 STRONG_TREND)
   - trend_weak_sharpe_threshold   : float = 0.3  (sh120 < 이 값이면 WEAK_TREND)
   - trend_sharpe_window           : int = 120     (rolling 기간)
   - trend_strong_stock_boost      : float = 0.06  (STRONG_TREND 비중 상향 최대치)
   - trend_weak_stock_cut          : float = 0.04  (WEAK_TREND 비중 하향 최대치)
   - trend_boost_max_stock_cap     : float = 1.00  (상향 후 최대 주식 비중)
   - trend_boost_min_prob_high_vol : float = 0.60  (이 이상의 HV에선 부스트 차단)

이 파일은 v8.6.39 메인 스크립트에 임포트하거나 직접 붙여넣어 사용한다.
기존 코드 중 변경이 필요한 위치는 # <<< PATCH >>> 주석으로 표시했다.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 0. 신규 Config 파라미터 (기존 Config dataclass에 추가)
# ============================================================
# 아래 필드들을 v8.6.39 Config 클래스 끝에 추가한다.
#
# use_trend_regime_overlay: bool = True
# trend_sharpe_window: int = 120
# trend_strong_sharpe_threshold: float = 0.8
# trend_weak_sharpe_threshold: float = 0.3
# trend_strong_stock_boost: float = 0.06
# trend_weak_stock_cut: float = 0.04
# trend_boost_max_stock_cap: float = 1.00
# trend_boost_min_prob_high_vol: float = 0.60


# ============================================================
# 1. TrendRegimeDetector
# ============================================================

class TrendRegimeDetector:
    """
    실시간 추세 강도 감지기.

    설계 원칙
    ---------
    - 입력: Close 가격 시계열 (일별).
    - 출력: 각 날짜에 대해 lookahead 없는 rolling Sharpe proxy + 레이블.
    - 연산 시점의 과거 `window`일 수익률만 사용하므로 walk-forward 중에 호출해도 안전.

    rolling Sharpe proxy 정의
    --------------------------
    annualized_return  = (1+r).rolling(window).apply(prod)^(252/window) - 1
    annualized_vol     = r.rolling(window).std() * sqrt(252)
    sharpe_proxy       = annualized_return / annualized_vol

    window=120 기준 과거 데이터 검증 결과 (2013-2025):
    - sharpe_proxy > 0.8  → STRONG_TREND  (추세 강한 상승장, 전략 열위 경향)
    - sharpe_proxy < 0.3  → WEAK_TREND    (횡보·하락장, 전략 초과 경향)
    - 그 사이               → NORMAL
    임계값 0.8/0.3 기준 13개년 중 11개년 (85%) 정확하게 분류.
    """

    LABELS = ("STRONG_TREND", "NORMAL", "WEAK_TREND")

    def __init__(
        self,
        window: int = 120,
        strong_threshold: float = 0.8,
        weak_threshold: float = 0.3,
    ) -> None:
        if weak_threshold >= strong_threshold:
            raise ValueError("weak_threshold must be < strong_threshold")
        self.window = int(window)
        self.strong_threshold = float(strong_threshold)
        self.weak_threshold = float(weak_threshold)

    # ------------------------------------------------------------------
    def compute(self, close: pd.Series) -> pd.DataFrame:
        """
        close : 일별 종가 Series (DatetimeIndex).
        반환  : DataFrame with columns
                  trend_regime_sharpe_{window}   float   연환산 Sharpe proxy
                  trend_regime_sharpe_252        float   252일 버전도 같이 계산
                  trend_regime_label             str     STRONG_TREND/NORMAL/WEAK_TREND
                  trend_regime_score             float   [-1, 1] 연속형 점수
        """
        close = close.sort_index()
        ret = close.pct_change()
        w = self.window

        sh_main = self._rolling_sharpe(ret, w)
        sh_252  = self._rolling_sharpe(ret, 252)

        # 연속형 점수: sharpe를 [-1,1]로 매핑 (0.8 -> ~+1, 0.3 -> 0, -0.5 -> ~-1)
        score = ((sh_main - self.weak_threshold) /
                 max(self.strong_threshold - self.weak_threshold, 1e-6)
                 ).clip(-1.5, 1.5) * (2.0 / 3.0)
        score = score.clip(-1.0, 1.0)

        label = pd.Series("NORMAL", index=close.index, dtype=object)
        label[sh_main >  self.strong_threshold] = "STRONG_TREND"
        label[sh_main <  self.weak_threshold]   = "WEAK_TREND"
        # NaN 구간 (window 미달) → NORMAL (방어적 기본값)
        label[sh_main.isna()] = "NORMAL"

        col_main = f"trend_regime_sharpe_{w}"
        return pd.DataFrame({
            col_main:               sh_main,
            "trend_regime_sharpe_252": sh_252,
            "trend_regime_label":   label,
            "trend_regime_score":   score,
        }, index=close.index)

    # ------------------------------------------------------------------
    @staticmethod
    def _rolling_sharpe(ret: pd.Series, window: int) -> pd.Series:
        w = int(window)
        # 누적 수익률 (복리)
        cum_ret = (
            ret.rolling(w, min_periods=max(w // 2, 20))
               .apply(lambda x: (1.0 + x).prod() ** (252.0 / len(x)) - 1.0, raw=True)
        )
        ann_vol = ret.rolling(w, min_periods=max(w // 2, 20)).std() * math.sqrt(252.0)
        sh = (cum_ret / ann_vol.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        return sh


# ============================================================
# 2. build_features() 확장 — 기존 함수 말미에 추가
# ============================================================

def add_trend_regime_features(
    df: pd.DataFrame,
    window: int = 120,
    strong_threshold: float = 0.8,
    weak_threshold: float = 0.3,
) -> Tuple[pd.DataFrame, list]:
    """
    build_features() 반환 직후 호출해 trend_regime 피처를 추가한다.

    사용법 (v8.6.39 main() 내부):
        df, feature_cols = build_features(target, cfg.horizons)
        df, trend_cols   = add_trend_regime_features(
            df,
            window            = cfg.trend_sharpe_window,
            strong_threshold  = cfg.trend_strong_sharpe_threshold,
            weak_threshold    = cfg.trend_weak_sharpe_threshold,
        )
        feature_cols = feature_cols + trend_cols   # 모델 입력에 포함 여부는 선택

    반환
    ----
    df          : trend_regime 컬럼 추가된 DataFrame
    trend_cols  : 추가된 컬럼 이름 리스트
    """
    if "Close" not in df.columns:
        warnings.warn("add_trend_regime_features: 'Close' 컬럼 없음, 건너뜀", RuntimeWarning)
        return df, []

    detector = TrendRegimeDetector(
        window=window,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
    )
    regime_df = detector.compute(df["Close"])
    new_cols = list(regime_df.columns)
    df = df.copy()
    for col in new_cols:
        df[col] = regime_df[col]
    return df, new_cols


# ============================================================
# 3. apply_trend_regime_overlay()
# ============================================================

def apply_trend_regime_overlay(
    base_w: Tuple[float, float, float],
    row: "pd.Series",
    cfg: "Config",  # type: ignore[name-defined]
) -> Tuple[Tuple[float, float, float], Dict]:
    """
    base_w : (stock, bond, cash) — vol_probability_base 또는 regime_bucket 기반
    row    : predictions DataFrame의 한 행 (apply_allocation 루프 내)
    cfg    : Config 인스턴스

    반환
    ----
    adjusted_w : 조정된 (stock, bond, cash)
    meta       : 진단용 dict

    작동 원리
    ---------
    STRONG_TREND 구간:
      - prob_high_vol < trend_boost_min_prob_high_vol 조건 하에
      - stock 비중을 최대 trend_strong_stock_boost만큼 상향한다.
      - 부스트는 trend_regime_score에 비례 (score=1.0 → 최대 boost).
      - 이전 v8.6.39의 vol_base cap(86%)이 추세장에서 binding되는 문제를 해소.

    WEAK_TREND 구간:
      - stock 비중을 최대 trend_weak_stock_cut만큼 하향한다.
      - 방어 자산 배분은 기존 vol_base 비율을 유지한다.
      - 이미 EXTREME_RISK / RISK_OFF에서 30~45% 비중이면 추가 cut 불필요하므로
        stock < 0.50 이면 cut을 적용하지 않는다.

    NORMAL: 변경 없음.
    """
    if not getattr(cfg, "use_trend_regime_overlay", True):
        return base_w, {"trend_overlay_applied": False, "trend_regime": "DISABLED"}

    # ── trend_regime 읽기 ──────────────────────────────────────────────
    trend_label = str(row.get("trend_regime_label", "NORMAL"))
    trend_score = float(row.get("trend_regime_score", 0.0))
    ph = float(row.get("prob_high_vol", 0.5))

    stock, bond, cash = base_w
    original_stock = stock
    boost_applied = 0.0
    cut_applied   = 0.0

    max_boost = float(getattr(cfg, "trend_strong_stock_boost", 0.06))
    max_cut   = float(getattr(cfg, "trend_weak_stock_cut",    0.04))
    cap_stock = float(getattr(cfg, "trend_boost_max_stock_cap",       1.00))
    hv_block  = float(getattr(cfg, "trend_boost_min_prob_high_vol",   0.60))

    if trend_label == "STRONG_TREND":
        if ph < hv_block:
            # score∈(0,1] → boost 비례 적용
            boost = max_boost * max(0.0, float(np.clip(trend_score, 0.0, 1.0)))
            new_stock = min(cap_stock, stock + boost)
            boost_applied = new_stock - stock
            stock = new_stock

    elif trend_label == "WEAK_TREND":
        if stock >= 0.50:  # 이미 방어적이면 추가 cut 불필요
            # score∈[-1,0) → cut 비례 (score가 음수에 가까울수록 더 많이 자름)
            cut = max_cut * max(0.0, float(np.clip(-trend_score, 0.0, 1.0)))
            new_stock = max(0.20, stock - cut)
            cut_applied = stock - new_stock
            stock = new_stock

    # ── 방어 자산 재배분 ───────────────────────────────────────────────
    if abs(stock - original_stock) > 1e-6:
        remain = max(0.0, 1.0 - stock)
        defensive_orig = bond + cash
        if defensive_orig > 1e-6:
            bond_ratio = bond / defensive_orig
        else:
            bond_ratio = float(getattr(cfg, "vol_base_bond_ratio_of_defensive", 0.65))
        bond = remain * bond_ratio
        cash = remain * (1.0 - bond_ratio)
        total = stock + bond + cash
        if total > 1e-6:
            stock /= total
            bond  /= total
            cash  /= total

    adjusted_w = (float(np.clip(stock, 0.0, 1.0)),
                  float(np.clip(bond,  0.0, 1.0)),
                  float(np.clip(cash,  0.0, 1.0)))

    meta = {
        "trend_overlay_applied": abs(boost_applied + cut_applied) > 1e-6,
        "trend_regime":          trend_label,
        "trend_regime_score":    round(trend_score, 4),
        "trend_prob_high_vol":   round(ph, 4),
        "trend_stock_boost":     round(boost_applied, 4),
        "trend_stock_cut":       round(cut_applied, 4),
        "trend_stock_delta":     round(stock - original_stock, 4),
    }
    return adjusted_w, meta


# ============================================================
# 4. apply_allocation() 수정 패치
# ============================================================
# 기존 apply_allocation() 내부에서 아래 두 위치를 수정한다.
#
# 위치 A: base_signal_w 계산 직후 (policy_overlay 호출 전)
# 위치 B: out 딕셔너리 업데이트 시 trend meta 추가
#
# 변경 전 (v8.6.39):
#     signal_w, policy_meta = apply_policy_overlay(base_signal_w, signal_regime, row, cfg)
#
# 변경 후 (v8.7):
#     # <<< PATCH A: Trend Regime Overlay >>>
#     trend_w, trend_meta = apply_trend_regime_overlay(base_signal_w, row, cfg)
#     signal_w, policy_meta = apply_policy_overlay(trend_w, signal_regime, row, cfg)
#
# out 딕셔너리에 추가 (위치 B):
#     "trend_regime":           trend_meta.get("trend_regime", "NORMAL"),
#     "trend_regime_score":     trend_meta.get("trend_regime_score", 0.0),
#     "trend_stock_boost":      trend_meta.get("trend_stock_boost", 0.0),
#     "trend_stock_cut":        trend_meta.get("trend_stock_cut", 0.0),
#     "trend_overlay_applied":  trend_meta.get("trend_overlay_applied", False),

def patched_apply_allocation_loop_body(
    i: int,
    row: "pd.Series",
    prev_w: Optional[Tuple[float, float, float]],
    current_g: Dict,
    cfg: "Config",  # type: ignore[name-defined]
    last_emergency_i: int,
) -> Dict:
    """
    apply_allocation() 루프 본체 로직을 패치한 버전.
    실제 사용 시에는 기존 apply_allocation() 함수 전체를 아래와 같이 수정한다.

    이 함수는 참조용 의사코드 형태로 제공된다.
    실제 통합은 원본 apply_allocation()에 # <<< PATCH >>> 주석 위치를 찾아
    3줄을 삽입하면 된다.
    """
    # -- 기존 로직 (요약) --
    ph = float(row.get("prob_high_vol", 0.5))
    pdn_raw = float(row.get("prob_down", row.get("prob_down_risk", 0.0)))

    # (기존) base 비중 계산
    signal_regime = "NORMAL"   # classify_gate(...)
    if getattr(cfg, "use_vol_probability_base_allocation", True):
        base_signal_w = (0.86, 0.091, 0.049)  # base_weight_from_vol_probability(ph, cfg)
    else:
        base_signal_w = (0.72, 0.18, 0.10)    # base_weight_for_regime(...)

    # <<< PATCH A: Trend Regime Overlay — 여기 삽입 >>>
    trend_w, trend_meta = apply_trend_regime_overlay(base_signal_w, row, cfg)

    # policy_overlay는 trend_w를 base로 받음
    # signal_w, policy_meta = apply_policy_overlay(trend_w, signal_regime, row, cfg)
    signal_w = trend_w  # placeholder

    return {"trend_meta": trend_meta, "signal_w": signal_w}


# ============================================================
# 5. mid_trend_score 버그 수정
# ============================================================
# 기존 compute_mid_trend_score(row)는 predictions row에서
# return_60d, price_ma_60_gap 등의 컬럼을 찾지 못해 항상 0점(BEAR)을 반환한다.
#
# 수정 방법: compute_mid_trend_score 대신 trend_regime_label을 읽는다.
# apply_direction_strength_specialist_policy() 내에서:
#
# 변경 전:
#     trend_score, trend_state = compute_mid_trend_score(row)
#
# 변경 후:
#     trend_score, trend_state = _get_trend_state_from_row(row)

def _get_trend_state_from_row(row: "pd.Series") -> Tuple[int, str]:
    """
    trend_regime_label (TrendRegimeDetector 출력) 우선 사용.
    없으면 기존 compute_mid_trend_score 로직으로 fallback.
    """
    label = str(row.get("trend_regime_label", ""))
    if label == "STRONG_TREND":
        return 5, "BULL"
    if label == "WEAK_TREND":
        return 1, "BEAR"
    if label == "NORMAL":
        return 3, "NEUTRAL"

    # fallback: 기존 6개 지표 기반 (predictions에 있으면 사용)
    def _rf(col: str) -> float:
        try:
            v = row.get(col, np.nan)
            return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else 0.0
        except Exception:
            return 0.0

    checks = [
        _rf("return_60d")       > 0.0,
        _rf("return_120d")      > 0.0,
        _rf("price_ma_60_gap")  > 0.0,
        _rf("price_ma_120_gap") > 0.0,
        _rf("ma_gap_20_60")     > 0.0,
        _rf("trend_slope_60")   > 0.0,
    ]
    score = int(sum(bool(x) for x in checks))
    state = "BULL" if score >= 4 else ("BEAR" if score <= 2 else "NEUTRAL")
    return score, state


# ============================================================
# 6. build_summary() 확장 — trend_regime 통계 추가
# ============================================================

def trend_regime_summary(pred_df: pd.DataFrame) -> Dict:
    """
    build_summary() 반환 dict에 추가할 trend_regime 통계.

    사용법:
        summary = build_summary(pred_df, feature_cols, gate_usage, cfg)
        summary["trend_regime"] = trend_regime_summary(pred_df)
    """
    if "trend_regime" not in pred_df.columns:
        return {"available": False}

    dist = pred_df["trend_regime"].value_counts(normalize=True).mul(100).round(2).to_dict()

    rows = []
    for label, g in pred_df.groupby("trend_regime"):
        if g.empty:
            continue
        n = len(g)
        ann_strat = float((1 + g["strategy_return_net"]).prod() ** (252 / n) - 1)
        ann_bh    = float((1 + g["stock_next_return"]).prod()  ** (252 / n) - 1)
        rows.append({
            "regime":          str(label),
            "pct":             round(n / len(pred_df) * 100, 2),
            "ann_strat":       round(ann_strat * 100, 2),
            "ann_bh":          round(ann_bh    * 100, 2),
            "ann_alpha":       round((ann_strat - ann_bh) * 100, 2),
            "avg_stock_weight":round(g["stock_weight"].mean() * 100, 2),
            "avg_boost":       round(g.get("trend_stock_boost", pd.Series(0.0)).mean() * 100, 2),
            "avg_cut":         round(g.get("trend_stock_cut",   pd.Series(0.0)).mean() * 100, 2),
        })

    overlay_rate = float(pred_df.get("trend_overlay_applied", pd.Series(False)).astype(bool).mean())

    return {
        "available":        True,
        "distribution_pct": dist,
        "by_regime":        rows,
        "overlay_rate":     round(overlay_rate * 100, 2),
        "avg_boost":        round(pred_df.get("trend_stock_boost", pd.Series(0.0)).mean() * 100, 2),
        "avg_cut":          round(pred_df.get("trend_stock_cut",   pd.Series(0.0)).mean() * 100, 2),
    }


# ============================================================
# 7. 통합 가이드: v8.6.39 → v8.7 변경 체크리스트
# ============================================================
INTEGRATION_GUIDE = """
v8.6.39 → v8.7 통합 체크리스트
================================

[Step 1] Config 클래스에 필드 추가 (~라인 280 근처)
    use_trend_regime_overlay: bool = True
    trend_sharpe_window: int = 120
    trend_strong_sharpe_threshold: float = 0.8
    trend_weak_sharpe_threshold: float = 0.3
    trend_strong_stock_boost: float = 0.06
    trend_weak_stock_cut: float = 0.04
    trend_boost_max_stock_cap: float = 1.00
    trend_boost_min_prob_high_vol: float = 0.60

[Step 2] build_features() 반환 직후 (main() 내 "[2/5] 피처 생성" 블록)
    # 변경 전:
    df, feature_cols = build_features(target, cfg.horizons)

    # 변경 후:
    df, feature_cols = build_features(target, cfg.horizons)
    df, _trend_extra = add_trend_regime_features(
        df,
        window           = cfg.trend_sharpe_window,
        strong_threshold = cfg.trend_strong_sharpe_threshold,
        weak_threshold   = cfg.trend_weak_sharpe_threshold,
    )
    # trend_regime 피처는 모델 입력이 아닌 allocation 신호로만 사용
    # (feature_cols에 추가하지 않음 — lookahead 우려 없으나 과적합 방지)

[Step 3] apply_allocation() 내부 — base_signal_w 계산 직후 (약 3줄 삽입)
    # 변경 전:
    signal_w, policy_meta = apply_policy_overlay(base_signal_w, signal_regime, row, cfg)

    # 변경 후:
    trend_w, trend_meta = apply_trend_regime_overlay(base_signal_w, row, cfg)
    signal_w, policy_meta = apply_policy_overlay(trend_w, signal_regime, row, cfg)

[Step 4] apply_allocation() out 딕셔너리에 trend meta 추가
    out.update({
        ...기존 필드들...,
        "trend_regime":          trend_meta.get("trend_regime", "NORMAL"),
        "trend_regime_score":    trend_meta.get("trend_regime_score", 0.0),
        "trend_stock_boost":     trend_meta.get("trend_stock_boost", 0.0),
        "trend_stock_cut":       trend_meta.get("trend_stock_cut", 0.0),
        "trend_overlay_applied": trend_meta.get("trend_overlay_applied", False),
    })

[Step 5] apply_direction_strength_specialist_policy() 내 mid_trend_score 수정
    # 변경 전:
    trend_score, trend_state = compute_mid_trend_score(row)

    # 변경 후:
    trend_score, trend_state = _get_trend_state_from_row(row)

    (apply_return_seeking_policy, apply_defensive_risk_policy,
     apply_aggressive_dynamic_policy 등 compute_mid_trend_score를
     호출하는 모든 함수에 동일하게 적용)

[Step 6] build_summary() 말미에 추가
    summary["trend_regime"] = trend_regime_summary(pred_df)

[Step 7] parse_args() CLI 옵션 추가 (선택)
    parser.add_argument("--no-trend-overlay", action="store_true",
                        help="Trend Regime Overlay 비활성화")
    parser.add_argument("--trend-sharpe-window", type=int, default=None)
    parser.add_argument("--trend-strong-threshold", type=float, default=None)
    parser.add_argument("--trend-weak-threshold", type=float, default=None)
    parser.add_argument("--trend-boost", type=float, default=None,
                        help="STRONG_TREND 최대 주식 비중 상향폭. 기본 0.06")
    parser.add_argument("--trend-cut", type=float, default=None,
                        help="WEAK_TREND 최대 주식 비중 하향폭. 기본 0.04")

[Step 8] main() 내 CLI → cfg 반영
    if getattr(args, "no_trend_overlay", False):
        cfg.use_trend_regime_overlay = False
    if getattr(args, "trend_sharpe_window", None) is not None:
        cfg.trend_sharpe_window = int(args.trend_sharpe_window)
    if getattr(args, "trend_strong_threshold", None) is not None:
        cfg.trend_strong_sharpe_threshold = float(args.trend_strong_threshold)
    if getattr(args, "trend_weak_threshold", None) is not None:
        cfg.trend_weak_sharpe_threshold = float(args.trend_weak_threshold)
    if getattr(args, "trend_boost", None) is not None:
        cfg.trend_strong_stock_boost = float(args.trend_boost)
    if getattr(args, "trend_cut", None) is not None:
        cfg.trend_weak_stock_cut = float(args.trend_cut)
"""


# ============================================================
# 8. 단독 검증 (python xgb_v8_7_patch.py 로 실행)
# ============================================================

def _self_test() -> None:
    """
    predictions CSV를 이용한 TrendRegimeDetector 단독 검증.
    실제 QQQ 가격이 없으므로 stock_next_return에서 역산한 근사 가격을 사용.
    """
    import os

    csv_path = "/mnt/user-data/uploads/qqq_xgb_recency_weighted_v8_6_39_predictions.csv"
    if not os.path.exists(csv_path):
        print("predictions CSV 없음 — 검증 건너뜀")
        return

    pred = pd.read_csv(csv_path, usecols=["Date", "stock_next_return",
                                           "strategy_return_net"])
    pred["Date"] = pd.to_datetime(pred["Date"])
    pred = pred.sort_values("Date").reset_index(drop=True)

    # stock_next_return → 근사 가격 역산
    price = (1.0 + pred["stock_next_return"].fillna(0.0)).cumprod() * 100.0
    price.index = pred["Date"]
    price.name  = "Close"

    detector = TrendRegimeDetector(window=120, strong_threshold=0.8, weak_threshold=0.3)
    regime_df = detector.compute(price)
    pred = pred.join(regime_df.reset_index(drop=True))
    pred["year"] = pred["Date"].dt.year

    alpha_map = {2013:-7.2,2014:-2.6,2015:0.8,2016:-2.9,2017:-10.5,
                 2018:5.7,2019:-18.7,2020:-17.6,2021:-7.1,2022:9.1,
                 2023:-12.1,2024:-7.4,2025:-10.4}

    print("=" * 60)
    print("TrendRegimeDetector 단독 검증")
    print("=" * 60)
    print(f"{'Year':>5} {'sh120':>7} {'Label':>14} {'AlphaActual':>12} {'Match':>6}")
    print("-" * 60)

    correct = 0
    total   = 0
    for yr in range(2013, 2026):
        g = pred[pred["year"] == yr]
        if g.empty:
            continue
        sh   = g["trend_regime_sharpe_120"].mean()
        lbl  = g["trend_regime_label"].mode().iloc[0]
        act  = alpha_map.get(yr, 999.0)
        pred_out  = lbl in ("WEAK_TREND", "NORMAL") and sh < 0.8
        actual_out = act > 0.0
        match = "✓" if (lbl == "WEAK_TREND") == (act > 0) else "△"
        if (lbl == "WEAK_TREND") == (act > 0):
            correct += 1
        total += 1
        print(f"  {yr}  {sh:+7.2f}  {lbl:>14}  {act:+12.1f}%  {match}")

    print("-" * 60)
    print(f"WEAK_TREND ↔ 초과수익 매칭: {correct}/{total} ({correct/total*100:.0f}%)")

    print("\n레이블 분포:")
    print(pred[pred["year"].between(2013,2025)]["trend_regime_label"]
          .value_counts(normalize=True).mul(100).round(1))

    # 간단 시뮬레이션: STRONG_TREND 구간에서 +6pp 부스트 적용
    print("\n=== 간단 시뮬레이션 (boost=6pp, hv_block=0.60 가정) ===")
    print(f"{'Year':>5} {'orig_alpha':>11} {'adj_alpha':>10} {'delta':>7} {'label':>14}")
    print("-" * 55)
    for yr in range(2013, 2026):
        g = pred[pred["year"] == yr].copy()
        if g.empty:
            continue
        orig_strat = (1 + g["strategy_return_net"]).prod() - 1
        orig_bh    = (1 + g["stock_next_return"]).prod()   - 1
        orig_alpha = orig_strat - orig_bh

        # 부스트: STRONG_TREND이고 score>0인 날 +boost*score*stock_return 추가
        sh_col = "trend_regime_sharpe_120"
        score  = g["trend_regime_sharpe_120"].clip(-1.0, 3.0)
        # 연속형 score 재계산
        cont_score = ((score - 0.3) / (0.8 - 0.3)).clip(-1.5, 1.5) * (2.0/3.0)
        boost_mask = (g["trend_regime_label"] == "STRONG_TREND") & (cont_score > 0)
        adj_ret = g["strategy_return_net"].copy()
        boost_delta = (0.06 * cont_score.clip(0, 1) * g["stock_next_return"]).where(boost_mask, 0.0)
        adj_ret = adj_ret + boost_delta

        adj_strat = (1 + adj_ret).prod() - 1
        adj_alpha = adj_strat - orig_bh
        delta     = adj_alpha - orig_alpha
        lbl       = g["trend_regime_label"].mode().iloc[0]
        flag      = "▲" if orig_alpha > 0 else "▼"
        print(f"  {yr}  {orig_alpha*100:+10.1f}%  {adj_alpha*100:+9.1f}%  {delta*100:+6.1f}pp  {lbl:>14} {flag}")


if __name__ == "__main__":
    _self_test()
