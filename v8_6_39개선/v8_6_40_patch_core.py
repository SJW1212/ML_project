"""
v8.6.40 Core Patch
==================
v8.6.39 코드베이스에 직접 적용 가능한 패치 모음.

적용 순서:
  1. mid_trend context 버그 수정          → MID_TREND_FEATURE_COLS, compute_mid_trend_score_safe
  2. prediction row에 raw feature 보존    → patch_prediction_row_with_features()
  3. Down-risk allocation 영향 제거       → allocation_downrisk_score_v40()
  4. allocation_trace 추가               → build_allocation_trace()
  5. ConflictResolver (경고 플래그)       → check_allocation_conflict()
  6. no_trade_band 예외 조건              → should_override_no_trade_band()
  7. UpStrength 5D allocation 비활성화    → (Config 기본값 변경 안내)
  8. hold_reason별 성과 진단              → build_hold_reason_diagnostics()
  9. auto_review 요약 생성               → build_auto_review()

근거 레퍼런스:
  - prob_down_strengthening threshold F1 plateaus near 0.30 (down_strength_score rows in threshold_diagnostics)
    → Down head precision at 0.30 threshold: 0.204 (near-random). 제거 정당화.
  - mid_trend_score = 0 (BEAR) for ALL 3357 rows in predictions.csv
    → return_60d/return_120d/price_ma_60_gap/price_ma_120_gap/ma_gap_20_60/trend_slope_60
       이 모두 prediction row에 부재함을 확인.
  - prob_up_strengthening_score bin 0.5~0.6: actual UP_STRENGTHENING rate 61.3%, ann_return 56.8%
    vs bin 0.3~0.4: actual rate 28.4%, ann_return 3.8%
    → score >= 0.40 이상에서만 allocation 기여를 허용해야 함.
  - CUSTOM regime: ann_return 20.4%, avg_stock 83.4% — 실질 성과의 핵심
    RISK_OFF regime: ann_return 7.9%, avg_stock 44.8% — 방어는 되나 과방어
  - mid_trend BEAR 고정 → CUSTOM regime에서도 BULL bonus overlay가 전혀 적용 안 됨
    → 실제 BULL period에서의 up_bonus 0.0 고착화
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 1. mid_trend context 버그 수정
# ─────────────────────────────────────────────

# compute_mid_trend_score()가 필요로 하는 raw feature 목록.
# 이 컬럼들이 prediction row에 없으면 전부 0 → BEAR로 고착된다.
MID_TREND_FEATURE_COLS: List[str] = [
    "return_60d",
    "return_120d",
    "price_ma_60_gap",
    "price_ma_120_gap",
    "ma_gap_20_60",
    "trend_slope_60",
]

# 선택적으로 추가할 수 있는 보조 피처
MID_TREND_FEATURE_COLS_EXTENDED: List[str] = MID_TREND_FEATURE_COLS + [
    "drawdown_60",
    "realized_vol_60",
    "trend_slope_20",
    "ma_gap_60_120",
]


def compute_mid_trend_score_safe(row: pd.Series) -> Tuple[int, str]:
    """
    mid_trend_score 안전 버전 (v8.6.40).

    v8.6.39 버그:
        prediction row에 raw feature가 없으면 _row_float()이 default=0.0 반환
        → 모든 check False → score=0 → BEAR 고착

    수정:
        필요 피처가 실제로 존재하는지 확인 후,
        없으면 NaN 표시와 함께 UNKNOWN을 반환해 overlay를 중립으로 처리한다.

    Returns:
        (score, state) where state ∈ {"BULL", "NEUTRAL", "BEAR", "UNKNOWN"}
    """
    available = [c for c in MID_TREND_FEATURE_COLS if c in row.index and not pd.isna(row[c])]
    if len(available) < 3:
        # 피처 3개 미만이면 신뢰성 없음 → UNKNOWN 반환 (overlay 적용 안 함)
        return 0, "UNKNOWN"

    checks = []
    for col in MID_TREND_FEATURE_COLS:
        if col in row.index and not pd.isna(row[col]):
            checks.append(float(row[col]) > 0.0)
        # 없는 피처는 채점에서 제외 (0이라 가정하지 않음)

    score = int(sum(checks))
    total = len(checks)

    # 비율 기반 판정: 유효 체크의 2/3 이상 → BULL
    if total >= 4:
        if score >= max(4, round(total * 0.67)):
            state = "BULL"
        elif score <= round(total * 0.33):
            state = "BEAR"
        else:
            state = "NEUTRAL"
    else:
        # 유효 피처 3개인 경우
        if score >= 3:
            state = "BULL"
        elif score == 0:
            state = "BEAR"
        else:
            state = "NEUTRAL"

    return score, state


def patch_prediction_row_with_features(
    out: dict,
    feature_row: pd.Series,
    feature_cols: List[str],
) -> dict:
    """
    prediction row(out dict)에 raw feature를 보존한다.

    run_walk_forward() 내부에서 out.update() 직전에 호출.
    apply_allocation() → apply_policy_overlay() → compute_mid_trend_score()가
    feature를 row에서 읽기 때문에, pred_df에 보존되어야 한다.

    사용 예:
        out = patch_prediction_row_with_features(out, all_df.iloc[pos], feature_cols)

    보존 대상:
        - MID_TREND_FEATURE_COLS (필수)
        - 추가 유용 피처 선택적 포함
    """
    preserve_cols = list(set(MID_TREND_FEATURE_COLS_EXTENDED) & set(feature_cols))
    for col in preserve_cols:
        if col not in out:
            val = feature_row.get(col, np.nan)
            out[f"ctx_{col}"] = float(val) if not pd.isna(val) else np.nan
    # apply_allocation에서는 ctx_ prefix 없이 읽으므로 직접도 저장
    for col in MID_TREND_FEATURE_COLS:
        if col not in out:
            val = feature_row.get(col, np.nan)
            if not pd.isna(val):
                out[col] = float(val)
    return out


# ─────────────────────────────────────────────
# 2. Down-risk allocation 영향 제거
# ─────────────────────────────────────────────

def allocation_downrisk_score_v40(
    ph: float,
    pdn_raw: float,
    cfg,
) -> float:
    """
    v8.6.40: Down-risk를 allocation에서 완전 분리.

    근거:
        - QQQ down_strength_score precision@0.30 threshold = 0.204 (near-random)
        - NVDA Down-risk ROC-AUC = 0.464 (역방향 가능성)
        - overall_risk_down_weight = 0.0 (이미 비활성)이지만
          allocation_downrisk_score()가 여전히 pdn을 반영하는 경우 있음
        - v8.6.39 config: overall_risk_down_weight = 0.0 이지만
          apply_direction_strength_specialist_policy에서 pdn_alloc = ph로 수동 치환됨
          → 이미 사실상 제거 상태이나 명시적으로 강제

    Returns:
        항상 0.0 (down-risk 신호 완전 배제)
    """
    return 0.0


# ─────────────────────────────────────────────
# 3. allocation_trace 추가
# ─────────────────────────────────────────────

def build_allocation_trace(
    base_stock: float,
    up_bonus: float,
    trend_cut: float,
    drawdown_cut: float,
    no_trade_band_hold: bool,
    final_stock: float,
    mid_trend_state: str,
    ph: float,
    pus_score: float,
    tier: int,
    regime: str,
) -> dict:
    """
    allocation 결정 경로를 명시적으로 기록한다.

    latest.json에 추가하면 '왜 주식 86%인지' 즉시 파악 가능.

    사용 예 (apply_direction_strength_specialist_policy 말미에서 호출):
        meta["allocation_trace"] = build_allocation_trace(...)

    근거:
        - QQQ latest: stock=86%, mid_trend=BEAR(버그), up_bonus=0.0
          → trace 없으면 defensive 신호와 모순인지 확인 불가
        - ConflictResolver의 입력 소스로도 활용
    """
    trace = {
        "base_stock_from_highvol": round(base_stock, 4),
        "up_bonus": round(up_bonus, 4),
        "trend_cut": round(trend_cut, 4),
        "drawdown_cut": round(drawdown_cut, 4),
        "no_trade_band_hold": bool(no_trade_band_hold),
        "final_stock": round(final_stock, 4),
        "mid_trend_state": mid_trend_state,
        "prob_high_vol": round(ph, 4),
        "up_strength_score": round(pus_score, 4),
        "offensive_tier": tier,
        "signal_regime": regime,
    }
    return trace


# ─────────────────────────────────────────────
# 4. ConflictResolver
# ─────────────────────────────────────────────

CONFLICT_RULES = [
    # (조건 이름, 조건 함수, 경고 레벨)
    # level: "WARNING" = 기록만, "HARD_CAP" = stock을 강제 상한 적용
    (
        "midtrend_bear_high_stock",
        lambda stock, ph, trend, drawdown_guard: (
            trend == "BEAR" and stock >= 0.80
        ),
        "WARNING",
        "mid_trend=BEAR인데 stock >= 80%. mid_trend feature 누락 의심.",
    ),
    (
        "highvol_high_stock",
        lambda stock, ph, trend, drawdown_guard: (
            ph > 0.60 and stock >= 0.80
        ),
        "WARNING",
        "prob_high_vol > 0.60인데 stock >= 80%. base_weight_from_vol_probability 점검 필요.",
    ),
    (
        "extreme_highvol_full_stock",
        lambda stock, ph, trend, drawdown_guard: (
            ph > 0.80 and stock >= 0.70
        ),
        "HARD_CAP",
        "prob_high_vol > 0.80인데 stock >= 70%. EXTREME_RISK regime 강제 점검.",
    ),
]


def check_allocation_conflict(
    stock: float,
    ph: float,
    mid_trend_state: str,
    drawdown_guard: float = 0.0,
) -> List[dict]:
    """
    allocation 결과의 내부 불일치를 감지한다.

    Returns:
        List of conflict dicts. 비어 있으면 정상.

    사용 예:
        conflicts = check_allocation_conflict(stock=0.86, ph=0.159, mid_trend_state="BEAR")
        for c in conflicts:
            if c["level"] == "HARD_CAP":
                stock = min(stock, c.get("cap", 0.70))
    """
    conflicts = []
    for name, cond_fn, level, msg in CONFLICT_RULES:
        try:
            if cond_fn(stock, ph, mid_trend_state, drawdown_guard):
                conflicts.append({
                    "rule": name,
                    "level": level,
                    "message": msg,
                    "stock": round(stock, 4),
                    "ph": round(ph, 4),
                    "mid_trend_state": mid_trend_state,
                })
        except Exception:
            pass
    return conflicts


# ─────────────────────────────────────────────
# 5. no_trade_band 예외 조건
# ─────────────────────────────────────────────

def should_override_no_trade_band(
    ph: float,
    ph_prev: float,
    mid_trend_state: str,
    mid_trend_state_prev: str,
    drawdown_guard: float,
    drawdown_guard_prev: float,
    cfg,
) -> Tuple[bool, str]:
    """
    no_trade_band를 무시하고 즉시 리밸런싱해야 하는 조건을 판정한다.

    v8.8.1에서 held_by_no_trade_band 구간 수익성이 약했던 이유:
        risk_score 급등 시에도 이전 비중이 유지되는 구조적 지연.

    조건별 근거:
        1. prob_high_vol 0.70+ 상향 돌파:
           threshold_diagnostics에서 0.70+ precision=0.81, F1=0.592 → 유의미한 신호
        2. prob_high_vol 0.55 → 0.70 급등 (15pp 이상):
           갑작스러운 vol spike는 방어 지연이 손실로 직결
        3. mid_trend_state 변화 (BULL→BEAR 또는 BEAR→BULL):
           trend 전환은 base_weight_from_vol_probability의 기반 변경 → 즉시 반영 필요
        4. drawdown_guard 급등 (0.15+ 증가):
           drawdown 악화는 MDD 방어에 직결

    Returns:
        (should_override: bool, reason: str)
    """
    # 1. prob_high_vol 0.70 상향 돌파
    hv_breach_threshold = float(getattr(cfg, "no_trade_override_hv_threshold", 0.70))
    if ph >= hv_breach_threshold and ph_prev < hv_breach_threshold:
        return True, f"prob_high_vol crossed {hv_breach_threshold:.2f} (prev={ph_prev:.3f} → {ph:.3f})"

    # 2. prob_high_vol 급등 (delta >= 0.15)
    hv_surge_delta = float(getattr(cfg, "no_trade_override_hv_surge_delta", 0.15))
    if ph - ph_prev >= hv_surge_delta and ph >= 0.55:
        return True, f"prob_high_vol surge +{ph - ph_prev:.3f} (→ {ph:.3f})"

    # 3. mid_trend_state 전환
    if (
        mid_trend_state != mid_trend_state_prev
        and mid_trend_state_prev not in ("UNKNOWN", "")
        and mid_trend_state not in ("UNKNOWN", "")
    ):
        # BULL↔BEAR만 강제, NEUTRAL 전환은 완화
        if {mid_trend_state, mid_trend_state_prev} == {"BULL", "BEAR"}:
            return True, f"mid_trend flip {mid_trend_state_prev} → {mid_trend_state}"

    # 4. drawdown_guard 급등
    dg_surge = float(getattr(cfg, "no_trade_override_drawdown_surge", 0.15))
    if drawdown_guard - drawdown_guard_prev >= dg_surge:
        return True, f"drawdown_guard surge +{drawdown_guard - drawdown_guard_prev:.3f}"

    return False, ""


# ─────────────────────────────────────────────
# 6. hold_reason별 성과 진단
# ─────────────────────────────────────────────

def build_hold_reason_diagnostics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    hold_reason별 성과를 집계한다.

    근거:
        - RISK_OFF annual_turnover 27.6x → 거래비용 과다
        - not_rebalance_day가 전체의 69%를 차지하는데 이 구간 수익성 미파악
        - v8.8.1에서 held_by_no_trade_band 구간이 underperform한다는 것을 확인

    Returns:
        DataFrame with hold_reason × performance metrics
    """
    required = {"hold_reason", "strategy_return_net", "stock_next_return", "stock_weight"}
    if not required.issubset(pred_df.columns):
        missing = required - set(pred_df.columns)
        warnings.warn(f"hold_reason_diagnostics: missing columns {missing}")
        return pd.DataFrame()

    df = pred_df.copy()
    df["strategy_return_net"] = pd.to_numeric(df["strategy_return_net"], errors="coerce")
    df["stock_next_return"] = pd.to_numeric(df["stock_next_return"], errors="coerce")

    rows = []
    for reason, grp in df.groupby("hold_reason"):
        n = len(grp)
        r = grp["strategy_return_net"].dropna()
        bh = grp["stock_next_return"].dropna()
        avg_stock = grp["stock_weight"].mean() if "stock_weight" in grp.columns else np.nan

        if len(r) == 0:
            continue

        ann_ret = float((1 + r).prod() ** (252 / max(len(r), 1)) - 1)
        ann_ret_bh = float((1 + bh).prod() ** (252 / max(len(bh), 1)) - 1) if len(bh) > 0 else np.nan
        win_rate = float((r > 0).mean())
        ann_vol = float(r.std() * np.sqrt(252)) if len(r) > 1 else np.nan
        sharpe = float(ann_ret / ann_vol) if ann_vol and ann_vol > 0 else np.nan

        rows.append({
            "hold_reason": reason,
            "count": n,
            "pct": round(n / len(df), 4),
            "ann_return_strat": round(ann_ret, 4),
            "ann_return_bh": round(ann_ret_bh, 4) if not np.isnan(ann_ret_bh) else np.nan,
            "bh_gap": round(ann_ret - ann_ret_bh, 4) if not np.isnan(ann_ret_bh) else np.nan,
            "win_rate": round(win_rate, 4),
            "ann_vol": round(ann_vol, 4) if not np.isnan(ann_vol) else np.nan,
            "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
            "avg_stock_weight": round(avg_stock, 4) if not np.isnan(avg_stock) else np.nan,
        })

    result = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────
# 7. up_strength_score bin별 성과 진단
# ─────────────────────────────────────────────

def build_up_strength_bin_diagnostics(
    pred_df: pd.DataFrame,
    score_col: str = "prob_up_strengthening_score",
    bins: int = 10,
) -> pd.DataFrame:
    """
    UpStrength score 구간별 실제 수익률과 UP_STRENGTHENING 발생률을 집계.

    근거 (probability_bins.csv):
        score bin 0.5~0.6: actual UP rate 61.3%, ann_return 56.8%
        score bin 0.7~0.8: actual UP rate 46.6%, ann_return -10.1%  ← 역전 현상
        score bin 0.8~0.9: actual UP rate 92.7%, ann_return 80.6%
        → 0.40 미만 구간은 allocation 효과 없음. 0.40~0.60 구간이 핵심 sweet spot.

    이 분석을 바탕으로:
        tier1 threshold를 낮추지 말고 score >= 0.40 구간만 active로 유지.
    """
    if score_col not in pred_df.columns:
        warnings.warn(f"build_up_strength_bin_diagnostics: {score_col} not found")
        return pd.DataFrame()

    df = pred_df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df["strategy_return_net"] = pd.to_numeric(df.get("strategy_return_net", 0.0), errors="coerce")

    bin_edges = np.linspace(0, 1, bins + 1)
    df["_bin"] = pd.cut(df[score_col], bins=bin_edges, include_lowest=True)

    rows = []
    for b, grp in df.groupby("_bin", observed=True):
        n = len(grp)
        r = grp["strategy_return_net"].dropna()
        ann_ret = float((1 + r).prod() ** (252 / max(len(r), 1)) - 1) if len(r) > 0 else np.nan

        actual_col = "actual_direction_strength_20d" if "actual_direction_strength_20d" in grp.columns else None
        actual_up_rate = np.nan
        if actual_col:
            actual_up_rate = float((grp[actual_col] == "UP_STRENGTHENING").mean())

        avg_stock = grp["stock_weight"].mean() if "stock_weight" in grp.columns else np.nan

        rows.append({
            "score_bin": str(b),
            "count": n,
            "avg_score": round(float(grp[score_col].mean()), 4),
            "actual_up_rate": round(actual_up_rate, 4) if not np.isnan(actual_up_rate) else np.nan,
            "ann_return_strat": round(ann_ret, 4) if not np.isnan(ann_ret) else np.nan,
            "avg_stock_weight": round(float(avg_stock), 4) if not np.isnan(avg_stock) else np.nan,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 8. auto_review 요약 생성
# ─────────────────────────────────────────────

def build_auto_review(
    pred_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    ticker: str = "QQQ",
    version: str = "v8.6.40",
) -> str:
    """
    진단 파일들로부터 의사결정용 요약(auto_review)을 생성한다.

    출력은 Markdown 텍스트. auto_review.md로 저장하거나 콘솔 출력 가능.

    포함 항목:
        - 전략 성과 vs Buy&Hold 요약
        - 최악/최선 연도
        - regime별 성과 요약
        - mid_trend_score 이상 여부 (버그 탐지)
        - prob_high_vol 분포
        - allocation conflict 집계
    """
    lines = [
        f"# Auto Review — {ticker} {version}",
        f"Generated from {len(pred_df)} prediction rows\n",
    ]

    # 전략 성과
    if not annual_df.empty and "strategy_net" in annual_df.columns:
        cagr_strat = float((1 + annual_df["strategy_net"]).prod() ** (1 / max(len(annual_df), 1)) - 1)
        cagr_bh = float((1 + annual_df["stock_buy_hold"]).prod() ** (1 / max(len(annual_df), 1)) - 1) if "stock_buy_hold" in annual_df.columns else np.nan
        lines.append("## 전략 성과 요약")
        lines.append(f"- Strategy CAGR: {cagr_strat:.2%}")
        if not np.isnan(cagr_bh):
            lines.append(f"- Buy & Hold CAGR: {cagr_bh:.2%}")
            lines.append(f"- Opportunity Cost: {cagr_strat - cagr_bh:+.2%}\n")

        # 최악/최선 연도
        annual_df_c = annual_df.copy()
        annual_df_c["gap"] = annual_df_c["strategy_net"] - annual_df_c.get("stock_buy_hold", annual_df_c["strategy_net"])
        best_yr = annual_df_c.loc[annual_df_c["strategy_net"].idxmax()]
        worst_yr = annual_df_c.loc[annual_df_c["strategy_net"].idxmin()]
        worst_gap = annual_df_c.loc[annual_df_c["gap"].idxmin()]
        lines.append("## 연도별 극값")
        lines.append(f"- 최고 성과: {best_yr['period'][:4]} ({best_yr['strategy_net']:.2%})")
        lines.append(f"- 최저 성과: {worst_yr['period'][:4]} ({worst_yr['strategy_net']:.2%})")
        lines.append(f"- 최대 기회비용: {worst_gap['period'][:4]} (gap={worst_gap['gap']:+.2%})\n")

    # mid_trend 이상 탐지
    lines.append("## mid_trend 이상 진단")
    if "mid_trend_score" in pred_df.columns:
        score_counts = pred_df["mid_trend_score"].value_counts().sort_index()
        bear_pct = float((pred_df.get("mid_trend_state", pd.Series([])) == "BEAR").mean()) if "mid_trend_state" in pred_df.columns else np.nan
        if float((pred_df["mid_trend_score"] == 0).mean()) > 0.90:
            lines.append("⚠️  **[버그 감지]** mid_trend_score = 0이 전체의 90% 이상")
            lines.append("   → return_60d/price_ma_60_gap 등 raw feature가 prediction row에 누락됨")
            lines.append("   → patch_prediction_row_with_features() 적용 필요\n")
        else:
            lines.append(f"- BEAR 비율: {bear_pct:.1%}")
            lines.append(f"- score 분포: {score_counts.to_dict()}\n")
    else:
        lines.append("- mid_trend_score 컬럼 없음\n")

    # regime별 성과
    if not regime_df.empty and "allocation_regime" in regime_df.columns:
        lines.append("## regime별 성과")
        lines.append("| Regime | 비중% | ann_return | avg_stock |")
        lines.append("|--------|-------|------------|-----------|")
        for _, r in regime_df.sort_values("ann_return_est", ascending=False).iterrows():
            lines.append(
                f"| {r['allocation_regime']} | {r['pct']:.1%} | "
                f"{r['ann_return_est']:.2%} | {r['avg_stock_weight']:.1%} |"
            )
        lines.append("")

    # hold_reason 진단
    hold_diag = build_hold_reason_diagnostics(pred_df)
    if not hold_diag.empty:
        lines.append("## hold_reason별 성과")
        lines.append("| reason | count | ann_return | BH_gap | avg_stock |")
        lines.append("|--------|-------|------------|--------|-----------|")
        for _, r in hold_diag.iterrows():
            gap_str = f"{r['bh_gap']:+.2%}" if not pd.isna(r.get("bh_gap", np.nan)) else "N/A"
            lines.append(
                f"| {r['hold_reason']} | {r['count']} | "
                f"{r['ann_return_strat']:.2%} | {gap_str} | {r['avg_stock_weight']:.1%} |"
            )
        lines.append("")

    # allocation conflict 집계
    if "stock_weight" in pred_df.columns and "prob_high_vol" in pred_df.columns:
        lines.append("## allocation conflict 통계")
        conflict_rows = []
        for _, row in pred_df.iterrows():
            stock = float(row.get("stock_weight", 0))
            ph = float(row.get("prob_high_vol", 0))
            mt = str(row.get("mid_trend_state", "UNKNOWN"))
            cs = check_allocation_conflict(stock, ph, mt)
            conflict_rows.extend(cs)

        if conflict_rows:
            conf_df = pd.DataFrame(conflict_rows)
            lines.append(f"- 총 conflict 발생: {len(conf_df)}건")
            for rule, grp in conf_df.groupby("rule"):
                lines.append(f"  - {rule}: {len(grp)}건 ({len(grp)/len(pred_df):.1%})")
        else:
            lines.append("- conflict 없음 ✓")
        lines.append("")

    lines.append("---")
    lines.append("*auto_review generated by v8.6.40 patch*")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 9. Config 기본값 변경 권고사항 (코드 변경 아님, 주석)
# ─────────────────────────────────────────────

CONFIG_CHANGES_V8640 = """
v8.6.40 Config 기본값 변경 권고
=================================

1. up_strength_disable_5d_trigger: True → True (유지, 확인)
   근거: 5D PR-AUC 0.326 (QQQ), 노이즈가 커서 allocation trigger 부적합

2. overall_risk_down_weight: 0.0 → 0.0 (유지, 확인)
   근거: down_strength_score precision < 0.21 (threshold_diagnostics)

3. overall_risk_down_minus_up_weight: 0.0 → 0.0 (유지, 확인)

4. use_multi_branch_downrisk: True → False (변경 권고)
   근거: 3개 branch 모두 F1 < 0.31, 코드 복잡도 대비 기여 없음

5. portfolio_model_enabled: false → false (유지, 코드 제거 권고)

6. (신규) no_trade_override_hv_threshold: 0.70
   (신규) no_trade_override_hv_surge_delta: 0.15
   (신규) no_trade_override_drawdown_surge: 0.15

7. (신규) mid_trend_feature_preservation: True
   → run_walk_forward()에서 patch_prediction_row_with_features() 호출 강제

8. up_strength_allocation_score_minimum: 0.40
   근거: probability_bins에서 score 0.40 미만 구간의 ann_return ≈ 3.8% (near zero alpha)
"""

print("v8.6.40 patch module loaded successfully.")
print(CONFIG_CHANGES_V8640)
