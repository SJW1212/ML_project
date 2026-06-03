"""
XGBoost v8.6.2 - Directional Up/Down + Overall Risk Allocation
====================================================================

목적
- QQQ/IEF/BIL 동적 자산배분 전략
- H10/H20 정상/고변동 Stage1 모델을 앙상블
- H10/H20 하락고변동 Down-risk OVR 모델을 앙상블
- 고변동 내부 상승/하락 Stage2 방향 분류는 제거
- Stage1 고변동 확률을 1차 게이트로 사용하고, Down-risk는 방어 보조 신호로 사용
- v8.4 개선: H10 Down-risk 단독 사용을 기본값으로 채택
- v8.4 개선: no-trade band, RISK_OFF threshold, RISK_OFF 비중, 연속 조정 사용 여부를 validation 기반으로 비교 가능
- v8.4 개선: 조건 선택 근거를 condition_search CSV로 저장
- v8.4 개선: regime/probability bin/threshold/turnover/drawdown/feature optimization diagnostics CSV 추가 저장
- v8.4 개선: v8.3 진단 결과를 반영해 c032 계열 조건을 기본값으로 채택
- v8.4 개선: HIGH_VOL 독립 regime 제거, NORMAL/WATCH/RISK_OFF 중심의 3-regime allocation 적용
- v8.4 개선: 극단 위험 구간에서만 추가 방어하는 EXTREME_RISK sub-regime 추가
- v8.4 개선: condition search는 상위 점수 후보 중 turnover/MDD 안정성을 우선하는 stable-top 선택 로직 적용
- v8.5 개선: 보수적 체결 지연(execution_lag_days), select/holdout 성과 분리, raw/EWMA 확률 분리
- v8.5 개선: classification_report 안정화, 라벨 생성 벡터화, max_train_rows 옵션 추가
- 비용/turnover를 고려해 10거래일 단위 리밸런싱 + 긴급 리밸런싱 제한
- 고변동 라벨 quantile 정책은 고정 또는 adaptive nested validation으로 선택 가능

실행 예시
    py xgb_multi_horizon_stage1_gated_downrisk_v8_4_stable_conditions.py --speed-profile balanced --h10-down-only --condition-search
    py xgb_multi_horizon_stage1_gated_downrisk_v8_4_stable_conditions.py --speed-profile fast --h10-down-only
    py xgb_multi_horizon_stage1_gated_downrisk_v8_4_stable_conditions.py --speed-profile full --adaptive-label --h10-down-only

필요 패키지
    pip install pandas numpy yfinance scikit-learn xgboost

중요
- 미래 수익률/변동성 컬럼은 라벨 생성에만 사용하고, 모델 입력 feature에는 사용하지 않습니다.
- walk-forward 예측 시 max(horizons)만큼 purge gap을 둡니다.
- adaptive label policy는 각 retrain 시점의 과거 train 구간 내부에서만 선택합니다.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("xgboost가 설치되어 있지 않습니다. `pip install xgboost`를 실행하세요.") from exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 0. CONFIG
# ============================================================

@dataclass(frozen=True)
class LabelPolicy:
    name: str
    vol_q: float = 0.80
    down_q: float = 0.20
    up_q: float = 0.80


@dataclass
class Config:
    target_ticker: str = "QQQ"
    bond_ticker: str = "IEF"
    cash_ticker: str = "BIL"

    start_date: str = "1999-03-10"
    backtest_start_date: str = "2013-01-02"
    end_date: Optional[str] = None

    initial_capital: float = 100_000_000
    transaction_cost_rate: float = 0.001
    # 0: 기존 방식. signal[t]로 Close[t] -> Close[t+1] 수익률 반영.
    # 1: 보수적 방식. signal[t] 생성 후 Close[t+1] -> Close[t+2] 수익률 반영.
    execution_lag_days: int = 1
    # False 권장. True면 BIL 다운로드 실패 시 현금 수익률 0으로 대체.
    allow_cash_download_fallback: bool = False

    horizons: Tuple[int, int] = (10, 20)
    primary_horizon: int = 10
    min_train_rows: int = 756
    retrain_every_n_days: int = 10
    # None이면 expanding window 전체 사용. 숫자를 주면 최근 N개 학습 샘플만 사용.
    max_train_rows: Optional[int] = None

    random_state: int = 42
    n_jobs: int = -1

    # XGBoost Stage1: normal vs high-vol
    stage1_n_estimators: int = 150
    stage1_learning_rate: float = 0.025
    stage1_max_depth: int = 3
    stage1_min_child_weight: float = 10.0
    stage1_subsample: float = 0.85
    stage1_colsample_bytree: float = 0.80
    stage1_reg_lambda: float = 8.0
    stage1_reg_alpha: float = 0.1

    # XGBoost Down-risk OVR: down-high-vol vs not down-high-vol
    down_n_estimators: int = 100
    down_learning_rate: float = 0.030
    down_max_depth: int = 2
    down_min_child_weight: float = 6.0
    down_subsample: float = 0.90
    down_colsample_bytree: float = 0.85
    down_reg_lambda: float = 10.0
    down_reg_alpha: float = 0.2

    # v8.6 Multi-branch Down-risk ensemble
    # - price_trend: 가격/추세 붕괴 선행 신호
    # - price_volume: 가격 하락 + 거래량 압력 신호
    # - volatility: 변동성/ATR/Range 확인 신호
    # - high_vol: Stage1 고변동 확률을 final down-risk score의 보조 입력으로 사용
    down_price_trend_weight: float = 0.40
    down_price_volume_weight: float = 0.30
    down_volatility_weight: float = 0.20
    down_highvol_weight: float = 0.00
    use_multi_branch_downrisk: bool = True

    # v8.6.2 Overall risk evaluation
    # - 전체 리스크는 고변동 위험과 하락위험을 함께 본다.
    # - Down-risk만 보면 방어 실패/과잉 방어를 전체 국면 관점에서 해석하기 어렵다.
    overall_risk_high_vol_weight: float = 0.35
    overall_risk_down_weight: float = 0.50
    overall_risk_down_minus_up_weight: float = 0.15
    pred_overall_risk_threshold: float = 0.50

    # v8.6.2 Direction model labels
    # future_return_h가 +threshold보다 크면 상승, -threshold보다 작으면 하락, 그 사이를 중립으로 둔다.
    direction_return_threshold: float = 0.005
    direction_decision_margin: float = 0.05
    direction_min_positive: int = 20

    # Adaptive label policy search
    use_adaptive_label_policy: bool = False
    label_search_valid_rows: int = 252
    label_search_stage1_estimators: int = 60
    label_search_down_estimators: int = 40
    label_search_min_positive: int = 20
    label_policy_candidates: Tuple[LabelPolicy, ...] = (
        LabelPolicy("balanced_q80_d20_u80", 0.80, 0.20, 0.80),
        LabelPolicy("sensitive_q75_d25_u75", 0.75, 0.25, 0.75),
        LabelPolicy("strict_q85_d15_u85", 0.85, 0.15, 0.85),
    )
    fixed_label_policy: LabelPolicy = LabelPolicy("fixed_q80_d20_u80", 0.80, 0.20, 0.80)

    # Ensemble weights
    # v8.1: Stage1은 H10/H20을 거의 균등하게 사용하되, Down-risk는 H10 중심으로 사용
    high_vol_weight_h10: float = 0.55
    high_vol_weight_h20: float = 0.45
    down_risk_weight_h10: float = 1.00
    down_risk_weight_h20: float = 0.00

    # Probability smoothing
    use_prob_ewma: bool = True
    prob_ewma_span: int = 7

    # Prediction thresholds for reporting
    pred_high_vol_threshold: float = 0.50
    pred_down_risk_threshold: float = 0.45

    # Allocation gate thresholds
    gate_normal_high_vol_threshold: float = 0.35
    # v8.1: RISK_OFF 진입을 더 어렵게 만들어 과도한 방어와 turnover를 줄임
    # v8.4: c032 계열 기본값. 0.62/0.52가 v8.3 선택값보다 CAGR/MDD/Calmar/turnover 균형이 좋았음.
    gate_high_vol_threshold: float = 0.62
    gate_riskoff_downrisk_threshold: float = 0.52
    gate_watch_downrisk_threshold: float = 0.64

    # v8.4: HIGH_VOL 독립 regime 제거. NORMAL / WATCH / RISK_OFF 중심으로 단순화.
    use_three_regime_allocation: bool = True

    # v8.4: 극단 위험 구간에서만 추가 방어. 일반 RISK_OFF는 58% 주식 유지.
    use_extreme_risk_cut: bool = True
    extreme_high_vol_threshold: float = 0.75
    extreme_downrisk_threshold: float = 0.65
    extreme_stock_weight: float = 0.45
    extreme_bond_weight: float = 0.35
    extreme_cash_weight: float = 0.20

    # Base bucket allocations
    normal_stock_weight: float = 0.92
    normal_bond_weight: float = 0.06
    normal_cash_weight: float = 0.02

    watch_stock_weight: float = 0.86
    watch_bond_weight: float = 0.10
    watch_cash_weight: float = 0.04

    high_vol_stock_weight: float = 0.75
    high_vol_bond_weight: float = 0.17
    high_vol_cash_weight: float = 0.08

    risk_off_stock_weight: float = 0.58
    risk_off_bond_weight: float = 0.273
    risk_off_cash_weight: float = 0.147

    # Small continuous adjustment within bucket
    use_continuous_adjustment: bool = False
    # v8.1: bucket 내부 연속 조정폭을 축소해 평균 주식 비중과 turnover를 개선
    continuous_high_vol_weight: float = 0.025
    continuous_down_risk_weight: float = 0.035
    max_continuous_stock_cut: float = 0.04

    # Trading rules
    rebalance_every_n_days: int = 10
    no_trade_band: float = 0.09
    emergency_high_vol_threshold: float = 0.80
    emergency_combined_high_vol_threshold: float = 0.65
    emergency_combined_down_threshold: float = 0.55
    emergency_cooldown_days: int = 10

    # Optional small rolling allocation threshold optimization
    # 기본값 False: 속도 문제 방지. 켜도 후보 수는 작게 유지.
    use_rolling_gate_optimization: bool = False
    gate_optimize_every_n_days: int = 120
    gate_rolling_window: int = 504
    gate_min_window: int = 252
    gate_score_cagr_weight: float = 1.30
    gate_score_mdd_weight: float = 0.85
    gate_score_turnover_weight: float = 0.45

    result_dir: str = "results_xgb_v8_6_2_directional_risk"


# ============================================================
# 1. DATA
# ============================================================

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.get_level_values(0).unique()) <= 6:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    return df


def download_ohlcv(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"{ticker} 데이터를 다운로드하지 못했습니다.")
    df = _flatten_yf_columns(df).copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker} 데이터에 필요한 컬럼이 없습니다: {missing}")
    df = df[required].copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def download_close(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"{ticker} 데이터를 다운로드하지 못했습니다.")
    df = _flatten_yf_columns(df).copy()
    if "Close" not in df.columns:
        raise ValueError(f"{ticker} 데이터에 Close 컬럼이 없습니다.")
    s = df["Close"].copy()
    s.name = ticker
    s.index = pd.to_datetime(s.index)
    return s.sort_index()



def build_aligned_forward_returns(
    target_close: pd.Series,
    bond_close: pd.Series,
    cash_close: pd.Series,
    target_index: pd.Index,
    execution_lag_days: int = 1,
) -> pd.DataFrame:
    """
    종가 기반 피처를 사용한 뒤 실제 체결 가능성을 보수적으로 반영하기 위한 forward return 생성.

    execution_lag_days=0:
        기존 방식과 동일하게 signal[t]에 Close[t] -> Close[t+1] 수익률을 연결한다.
    execution_lag_days=1:
        signal[t] 생성 후 다음 거래일부터 체결된다고 보고 Close[t+1] -> Close[t+2] 수익률을 연결한다.

    주의:
    - 마지막 execution_lag_days + 1개 행은 미래 수익률을 알 수 없으므로 NaN이 남는다.
    - 이후 walk-forward 입력 구성에서 dropna되어 백테스트 대상에서 제외된다.
    """
    if execution_lag_days < 0:
        raise ValueError("execution_lag_days는 0 이상의 정수여야 합니다.")

    prices = pd.concat(
        [
            target_close.rename("stock"),
            bond_close.rename("bond"),
            cash_close.rename("cash"),
        ],
        axis=1,
    ).reindex(target_index).ffill()

    shift_n = 1 + int(execution_lag_days)
    out = pd.DataFrame(index=target_index)
    out["stock_next_return"] = prices["stock"].pct_change().shift(-shift_n)
    out["bond_next_return"] = prices["bond"].pct_change().shift(-shift_n)

    if prices["cash"].notna().sum() == 0:
        # 명시적 fallback이 허용된 경우에만 cash_close가 전부 NaN으로 들어온다.
        out["cash_next_return"] = 0.0
    else:
        out["cash_next_return"] = prices["cash"].pct_change().shift(-shift_n)

    return out


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def rolling_rank_last(series: pd.Series, window: int) -> pd.Series:
    def _rank(x: np.ndarray) -> float:
        if np.all(np.isnan(x)):
            return np.nan
        last = x[-1]
        if np.isnan(last):
            return np.nan
        valid = x[~np.isnan(x)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= last).sum() / len(valid))
    return series.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=True)


def calc_trend_slope(close: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        ly = np.log(np.maximum(y, 1e-12))
        return float(np.polyfit(x, ly, 1)[0])
    return close.rolling(window, min_periods=window).apply(_slope, raw=True)


def add_future_targets(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    close = df["Close"]
    ret = df["daily_return"]
    future_cols: Dict[str, pd.Series] = {}
    for h in horizons:
        future_high = close.shift(-1).rolling(h).max().shift(-(h - 1))
        future_low = close.shift(-1).rolling(h).min().shift(-(h - 1))
        future_cols[f"future_volatility_{h}d"] = ret.shift(-1).rolling(h).std().shift(-(h - 1))
        future_cols[f"future_return_{h}d"] = close.shift(-h) / close - 1.0
        future_cols[f"future_max_return_{h}d"] = future_high / close - 1.0
        future_cols[f"future_min_return_{h}d"] = future_low / close - 1.0
    return pd.concat([df, pd.DataFrame(future_cols, index=df.index)], axis=1).copy()


def build_features(ohlcv: pd.DataFrame, horizons: Sequence[int]) -> Tuple[pd.DataFrame, List[str]]:
    df = ohlcv.copy()
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"].replace(0, np.nan)

    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))

    for w in [3, 5, 10, 20, 60, 120]:
        df[f"return_{w}d"] = close / close.shift(w) - 1.0

    for w in [5, 10, 20, 50, 60, 120, 200]:
        ma = close.rolling(w).mean()
        df[f"ma_{w}"] = ma
        df[f"price_ma_{w}_gap"] = close / ma - 1.0

    df["ma_gap_5_20"] = df["ma_5"] / df["ma_20"] - 1.0
    df["ma_gap_20_60"] = df["ma_20"] / df["ma_60"] - 1.0
    df["ma_gap_60_120"] = df["ma_60"] / df["ma_120"] - 1.0
    df["ma_gap_50_200"] = df["ma_50"] / df["ma_200"] - 1.0

    df["trend_slope_20"] = calc_trend_slope(close, 20)
    df["trend_slope_60"] = calc_trend_slope(close, 60)
    df["ma200_slope_60"] = calc_trend_slope(df["ma_200"], 60)

    up = (df["daily_return"] > 0).astype(float)
    large_down = (df["daily_return"] <= -0.02).astype(float)
    large_up = (df["daily_return"] >= 0.02).astype(float)
    for w in [20, 60]:
        df[f"positive_return_ratio_{w}"] = up.rolling(w).mean()
    df["large_down_day_ratio_20"] = large_down.rolling(20).mean()
    df["large_up_day_ratio_20"] = large_up.rolling(20).mean()

    for w in [20, 60, 120]:
        roll_high = close.rolling(w).max()
        roll_low = close.rolling(w).min()
        denom = (roll_high - roll_low).replace(0, np.nan)
        df[f"drawdown_{w}"] = close / roll_high - 1.0
        if w in [20, 60]:
            df[f"price_position_{w}"] = (close - roll_low) / denom
            df[f"close_to_{w}d_high"] = close / roll_high - 1.0

    df["volume_change"] = volume.pct_change()
    volume_ma20 = volume.rolling(20).mean()
    volume_std20 = volume.rolling(20).std()
    df["volume_ratio_20"] = volume / volume_ma20
    df["volume_zscore_20"] = (volume - volume_ma20) / volume_std20.replace(0, np.nan)

    # v8.6: 가격/추세 기반 하락 전조 피처
    for w in [20, 60]:
        prior_low = close.rolling(w).min().shift(1)
        prior_high = close.rolling(w).max().shift(1)
        df[f"breakdown_{w}"] = (close < prior_low).astype(float)
        df[f"failed_rebound_{w}"] = (close < prior_high * 0.97).astype(float)
    df["lower_high_20"] = (close.rolling(5).max() < close.rolling(20).max().shift(5)).astype(float)
    df["trend_consistency_20"] = (df["daily_return"] > 0).astype(float).rolling(20).mean()
    df["trend_consistency_60"] = (df["daily_return"] > 0).astype(float).rolling(60).mean()
    df["bearish_ma_stack"] = ((df["ma_5"] < df["ma_20"]) & (df["ma_20"] < df["ma_60"])).astype(float)

    # v8.6: 가격+거래량 기반 매도 압력 피처
    down_volume = ((df["daily_return"] < 0).astype(float) * volume)
    df["down_volume_ratio_20"] = down_volume.rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    df["high_volume_down_day"] = ((df["daily_return"] < 0) & (df["volume_zscore_20"] > 1.0)).astype(float)
    df["high_volume_down_ratio_20"] = df["high_volume_down_day"].rolling(20).mean()
    df["price_down_volume_up"] = ((df["daily_return"] < 0) & (df["volume_ratio_20"] > 1.2)).astype(float)
    df["weak_rebound_volume"] = ((df["return_5d"] > 0) & (df["volume_ratio_20"] < 0.8)).astype(float)
    df["down_momentum_volume_confirm"] = ((df["return_20d"] < 0) & (df["volume_ratio_20"] > 1.0)).astype(float)
    df["volume_shock_20"] = volume / volume_ma20
    df["volume_shock_rank_252"] = rolling_rank_last(df["volume_shock_20"], 252)
    df["down_volume_shock"] = ((df["daily_return"] < -0.01) & (df["volume_shock_20"] > 1.5)).astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["true_range"] = tr
    df["true_range_pct"] = tr / close
    for w in [14, 20, 60]:
        df[f"atr_{w}"] = tr.rolling(w).mean()
        df[f"atr_pct_{w}"] = df[f"atr_{w}"] / close
    df["atr_ratio_14_60"] = df["atr_14"] / df["atr_60"]
    df["atr_ratio_20_60"] = df["atr_20"] / df["atr_60"]
    df["atr_accel_5"] = df["atr_14"] / df["atr_14"].shift(5) - 1.0
    df["atr_rank_252"] = rolling_rank_last(df["atr_pct_20"], 252)

    log_hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(close / open_).replace([np.inf, -np.inf], np.nan)
    log_oc = np.log(open_ / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    log_ho = np.log(high / open_).replace([np.inf, -np.inf], np.nan)
    log_lo = np.log(low / open_).replace([np.inf, -np.inf], np.nan)

    parkinson_var = (1.0 / (4.0 * np.log(2.0))) * (log_hl ** 2)
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    for w in [20, 60]:
        df[f"realized_vol_{w}"] = df["daily_return"].rolling(w).std()
        df[f"ewma_vol_{w}"] = df["daily_return"].ewm(span=w, adjust=False).std()
        df[f"parkinson_vol_{w}"] = np.sqrt(parkinson_var.rolling(w).mean().clip(lower=0))
        df[f"garman_klass_vol_{w}"] = np.sqrt(gk_var.rolling(w).mean().clip(lower=0))
        df[f"rogers_satchell_vol_{w}"] = np.sqrt(rs_var.rolling(w).mean().clip(lower=0))
        k = 0.34 / (1.34 + (w + 1.0) / max(w - 1.0, 1.0))
        yz_var = log_oc.rolling(w).var() + k * log_co.rolling(w).var() + (1.0 - k) * rs_var.rolling(w).mean()
        df[f"yang_zhang_vol_{w}"] = np.sqrt(yz_var.clip(lower=0))

    df["realized_vol_ratio_20_60"] = df["realized_vol_20"] / df["realized_vol_60"]
    df["parkinson_vol_ratio_20_60"] = df["parkinson_vol_20"] / df["parkinson_vol_60"]
    df["yang_zhang_vol_ratio_20_60"] = df["yang_zhang_vol_20"] / df["yang_zhang_vol_60"]
    df["vol_of_vol_20"] = df["realized_vol_20"].rolling(20).std()

    downside_return = df["daily_return"].clip(upper=0)
    df["downside_vol_20"] = downside_return.rolling(20).std()
    df["downside_vol_60"] = downside_return.rolling(60).std()
    df["semi_vol_20"] = np.sqrt((downside_return ** 2).rolling(20).mean())
    dd20 = close / close.rolling(20).max() - 1.0
    dd60 = close / close.rolling(60).max() - 1.0
    df["ulcer_index_20"] = np.sqrt((dd20 ** 2).rolling(20).mean())
    df["ulcer_index_60"] = np.sqrt((dd60 ** 2).rolling(60).mean())
    df["ulcer_rank_252"] = rolling_rank_last(df["ulcer_index_20"], 252)

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_width_20"] = (4.0 * std20) / ma20
    df["bb_width_rank_252"] = rolling_rank_last(df["bb_width_20"], 252)
    ema20 = close.ewm(span=20, adjust=False).mean()
    df["keltner_width_20"] = (4.0 * df["atr_20"]) / ema20
    squeeze_on = (df["bb_width_20"] < df["keltner_width_20"]).astype(float)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "squeeze_on": squeeze_on,
                    "squeeze_release": ((squeeze_on.shift(1) == 1.0) & (squeeze_on == 0.0)).astype(float),
                },
                index=df.index,
            ),
        ],
        axis=1,
    ).copy()

    df = add_future_targets(df, horizons)

    feature_cols = [
        "daily_return", "log_return",
        "return_3d", "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "price_ma_5_gap", "price_ma_10_gap", "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap",
        "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120",
        "trend_slope_20", "trend_slope_60",
        "positive_return_ratio_20", "positive_return_ratio_60",
        "drawdown_20", "drawdown_60", "drawdown_120",
        "price_position_20", "price_position_60",
        "close_to_20d_high", "close_to_60d_high",
        "large_down_day_ratio_20", "large_up_day_ratio_20",
        "breakdown_20", "breakdown_60", "failed_rebound_20", "failed_rebound_60",
        "lower_high_20", "trend_consistency_20", "trend_consistency_60", "bearish_ma_stack",
        "volume_change", "volume_ratio_20", "volume_zscore_20",
        "down_volume_ratio_20", "high_volume_down_day", "high_volume_down_ratio_20",
        "price_down_volume_up", "weak_rebound_volume", "down_momentum_volume_confirm",
        "volume_shock_20", "volume_shock_rank_252", "down_volume_shock",
        "price_ma_200_gap", "ma_gap_50_200", "ma200_slope_60",
        "true_range_pct",
        "atr_pct_14", "atr_pct_20", "atr_pct_60", "atr_rank_252",
        "atr_ratio_14_60", "atr_ratio_20_60", "atr_accel_5",
        "realized_vol_20", "realized_vol_60", "realized_vol_ratio_20_60",
        "ewma_vol_20", "ewma_vol_60",
        "parkinson_vol_20", "parkinson_vol_60", "parkinson_vol_ratio_20_60",
        "garman_klass_vol_20", "rogers_satchell_vol_20",
        "yang_zhang_vol_20", "yang_zhang_vol_60", "yang_zhang_vol_ratio_20_60",
        "downside_vol_20", "downside_vol_60", "semi_vol_20",
        "ulcer_index_20", "ulcer_index_60", "ulcer_rank_252",
        "bb_width_20", "bb_width_rank_252", "keltner_width_20", "squeeze_on", "squeeze_release",
        "vol_of_vol_20",
    ]
    return df, [c for c in feature_cols if c in df.columns]


def _keep_existing_features(feature_cols: Sequence[str], candidates: Sequence[str]) -> List[str]:
    available = set(feature_cols)
    return [c for c in candidates if c in available]


def build_downrisk_feature_sets(feature_cols: Sequence[str]) -> Dict[str, List[str]]:
    """v8.6 Down-risk 전용 피처셋을 그룹별로 분리한다.

    Stage1 고변동 모델은 전체 피처를 사용한다. Down-risk는 다음 3개 가지를 따로 학습한다.
    - price_trend: 가격/추세 붕괴 신호
    - price_volume: 가격 하락과 거래량 압력 신호
    - volatility: ATR/Range/변동성 기반 확인 신호
    """
    price_trend_candidates = [
        "daily_return", "log_return",
        "return_3d", "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "price_ma_5_gap", "price_ma_10_gap", "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_20", "positive_return_ratio_60",
        "large_down_day_ratio_20", "large_up_day_ratio_20",
        "drawdown_20", "drawdown_60", "drawdown_120",
        "price_position_20", "price_position_60",
        "close_to_20d_high", "close_to_60d_high",
        "breakdown_20", "breakdown_60", "failed_rebound_20", "failed_rebound_60",
        "lower_high_20", "trend_consistency_20", "trend_consistency_60", "bearish_ma_stack",
    ]
    price_volume_candidates = price_trend_candidates + [
        "volume_change", "volume_ratio_20", "volume_zscore_20",
        "down_volume_ratio_20", "high_volume_down_day", "high_volume_down_ratio_20",
        "price_down_volume_up", "weak_rebound_volume", "down_momentum_volume_confirm",
        "volume_shock_20", "volume_shock_rank_252", "down_volume_shock",
    ]
    volatility_candidates = [
        "true_range_pct",
        "atr_pct_14", "atr_pct_20", "atr_pct_60", "atr_rank_252",
        "atr_ratio_14_60", "atr_ratio_20_60", "atr_accel_5",
        "realized_vol_20", "realized_vol_60", "realized_vol_ratio_20_60",
        "ewma_vol_20", "ewma_vol_60",
        "parkinson_vol_20", "parkinson_vol_60", "parkinson_vol_ratio_20_60",
        "garman_klass_vol_20", "rogers_satchell_vol_20",
        "yang_zhang_vol_20", "yang_zhang_vol_60", "yang_zhang_vol_ratio_20_60",
        "downside_vol_20", "downside_vol_60", "semi_vol_20",
        "ulcer_index_20", "ulcer_index_60", "ulcer_rank_252",
        "bb_width_20", "bb_width_rank_252", "keltner_width_20", "squeeze_on", "squeeze_release",
        "vol_of_vol_20",
    ]
    return {
        "price_trend": _keep_existing_features(feature_cols, price_trend_candidates),
        "price_volume": _keep_existing_features(feature_cols, price_volume_candidates),
        "volatility": _keep_existing_features(feature_cols, volatility_candidates),
    }


def normalize_downrisk_branch_weights(cfg: Config) -> Dict[str, float]:
    raw = {
        "price_trend": float(cfg.down_price_trend_weight),
        "price_volume": float(cfg.down_price_volume_weight),
        "volatility": float(cfg.down_volatility_weight),
        "high_vol": float(cfg.down_highvol_weight),
    }
    total = sum(max(0.0, v) for v in raw.values())
    if total <= 0:
        return {"price_trend": 0.45, "price_volume": 0.35, "volatility": 0.20, "high_vol": 0.00}
    return {k: max(0.0, v) / total for k, v in raw.items()}


def compute_overall_risk_prob(
    prob_high_vol: object,
    prob_down: object,
    cfg: Config,
    prob_up: Optional[object] = None,
) -> object:
    """
    v8.6.2 전체 리스크 점수.

    핵심 변경:
    - prob_down은 큰 하락위험이 아니라 방향성 하락 확률로 해석한다.
    - 상승 확률(prob_up)이 있으면 max(prob_down - prob_up, 0)을 추가 위험으로 반영한다.
    - scalar와 pandas Series 모두 처리한다.
    """
    hv_w = max(0.0, float(cfg.overall_risk_high_vol_weight))
    dn_w = max(0.0, float(cfg.overall_risk_down_weight))
    gap_w = max(0.0, float(getattr(cfg, "overall_risk_down_minus_up_weight", 0.0)))

    if prob_up is None:
        gap = 0.0
        total = hv_w + dn_w
        if total <= 0:
            hv_w, dn_w, total = 0.35, 0.50, 0.85
        score = (hv_w / total) * prob_high_vol + (dn_w / total) * prob_down
    else:
        gap = prob_down - prob_up
        if hasattr(gap, "clip"):
            gap = gap.clip(0.0, 1.0)
        else:
            gap = float(np.clip(gap, 0.0, 1.0))
        total = hv_w + dn_w + gap_w
        if total <= 0:
            hv_w, dn_w, gap_w, total = 0.35, 0.50, 0.15, 1.0
        score = (hv_w / total) * prob_high_vol + (dn_w / total) * prob_down + (gap_w / total) * gap

    if hasattr(score, "clip"):
        return score.clip(0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))

def combine_weighted_importance(
    histories: Dict[str, List[Dict[str, float]]],
    weights: Dict[str, float],
) -> Dict[str, float]:
    combined: Dict[str, float] = {}
    for branch, hist in histories.items():
        w = float(weights.get(branch, 0.0))
        mean_imp = mean_importance(hist)
        for feature, imp in mean_imp.items():
            combined[feature] = combined.get(feature, 0.0) + w * float(imp)
    return dict(sorted(combined.items(), key=lambda kv: kv[1], reverse=True))


# ============================================================
# 3. LABEL DESIGN
# ============================================================

def qclip(q: float) -> float:
    return float(np.clip(q, 0.01, 0.99))


def compute_policy_thresholds(train_df: pd.DataFrame, horizon: int, policy: LabelPolicy) -> Dict[str, float]:
    fvol = train_df[f"future_volatility_{horizon}d"]
    fmin = train_df[f"future_min_return_{horizon}d"]
    fmax = train_df[f"future_max_return_{horizon}d"]
    down_loose_q = qclip(policy.down_q + 0.05)
    down_strict_q = qclip(policy.down_q - 0.05)
    up_loose_q = qclip(policy.up_q - 0.05)
    up_strict_q = qclip(policy.up_q + 0.05)
    return {
        "policy_name": policy.name,
        "vol": float(fvol.quantile(policy.vol_q)),
        "down": float(fmin.quantile(policy.down_q)),
        "down_loose": float(fmin.quantile(down_loose_q)),
        "down_strict": float(fmin.quantile(down_strict_q)),
        "up": float(fmax.quantile(policy.up_q)),
        "up_loose": float(fmax.quantile(up_loose_q)),
        "up_strict": float(fmax.quantile(up_strict_q)),
        "vol_q": float(policy.vol_q),
        "down_q": float(policy.down_q),
        "up_q": float(policy.up_q),
    }


def assign_label(row: pd.Series, horizon: int, th: Dict[str, float]) -> str:
    future_vol = row[f"future_volatility_{horizon}d"]
    future_ret = row[f"future_return_{horizon}d"]
    future_max_ret = row[f"future_max_return_{horizon}d"]
    future_min_ret = row[f"future_min_return_{horizon}d"]

    atr_rank = row.get("atr_rank_252", np.nan)
    atr_ratio = row.get("atr_ratio_20_60", np.nan)
    return_20d = row.get("return_20d", np.nan)
    drawdown_60 = row.get("drawdown_60", np.nan)
    price_position_60 = row.get("price_position_60", np.nan)
    positive_ratio_20 = row.get("positive_return_ratio_20", np.nan)
    large_down_ratio_20 = row.get("large_down_day_ratio_20", np.nan)
    ulcer_rank = row.get("ulcer_rank_252", np.nan)
    bb_rank = row.get("bb_width_rank_252", np.nan)

    atr_high = bool(pd.notna(atr_rank) and atr_rank > 0.70)
    atr_extreme = bool(pd.notna(atr_rank) and atr_rank > 0.85)
    atr_expanding = bool(pd.notna(atr_ratio) and atr_ratio > 1.15)
    atr_compressed = bool(pd.notna(atr_rank) and atr_rank < 0.30)

    down_pressure_now = (
        (pd.notna(drawdown_60) and drawdown_60 < -0.08)
        or (pd.notna(return_20d) and return_20d < -0.05)
        or (pd.notna(large_down_ratio_20) and large_down_ratio_20 > 0.20)
        or (pd.notna(ulcer_rank) and ulcer_rank > 0.70)
    )
    up_pressure_now = (
        (pd.notna(return_20d) and return_20d > 0.05)
        and (pd.notna(price_position_60) and price_position_60 > 0.70)
        and (pd.notna(positive_ratio_20) and positive_ratio_20 > 0.55)
        and not (pd.notna(ulcer_rank) and ulcer_rank > 0.70)
    )
    squeeze_or_breakout = (
        (pd.notna(bb_rank) and bb_rank < 0.30)
        and (pd.notna(atr_ratio) and atr_ratio > 1.05)
    )

    down_threshold = th["down"]
    up_threshold = th["up"]

    if atr_high and atr_expanding and down_pressure_now:
        down_threshold = th["down_loose"]
        up_threshold = th["up_loose"]
    elif atr_high and atr_expanding and up_pressure_now:
        down_threshold = th["down_strict"]
        up_threshold = th["up_loose"]
    elif atr_extreme:
        down_threshold = th["down_loose"]
        up_threshold = th["up_loose"]
    elif atr_compressed and squeeze_or_breakout:
        down_threshold = th["down_loose"]
        up_threshold = th["up_loose"]

    is_high_vol = (
        future_vol >= th["vol"]
        or future_min_ret <= down_threshold
        or future_max_ret >= up_threshold
    )
    if not is_high_vol:
        return "정상"

    # Down-risk는 방어 목적상 우선한다.
    severe_down = future_min_ret <= th["down"]
    if severe_down and not up_pressure_now:
        return "하락고변동"

    if future_min_ret <= down_threshold:
        return "하락고변동"

    if future_max_ret >= up_threshold and future_ret > 0:
        return "상승고변동"

    if abs(future_max_ret) >= abs(future_min_ret):
        return "상승고변동"
    return "하락고변동"


def make_labels(df: pd.DataFrame, horizon: int, th: Dict[str, float]) -> pd.Series:
    """
    assign_label(row) 기반 apply를 벡터화한 라벨 생성 함수.

    목적:
    - walk-forward 반복 학습 속도 개선
    - 라벨 로직을 기존 assign_label과 최대한 동일하게 유지

    검증 권장:
    - 변경 직후에는 일부 구간에서 old apply 방식과 crosstab 비교를 수행하는 것이 안전하다.
    """
    idx = df.index

    future_vol = df[f"future_volatility_{horizon}d"]
    future_ret = df[f"future_return_{horizon}d"]
    future_max_ret = df[f"future_max_return_{horizon}d"]
    future_min_ret = df[f"future_min_return_{horizon}d"]

    atr_rank = df.get("atr_rank_252", pd.Series(np.nan, index=idx))
    atr_ratio = df.get("atr_ratio_20_60", pd.Series(np.nan, index=idx))
    return_20d = df.get("return_20d", pd.Series(np.nan, index=idx))
    drawdown_60 = df.get("drawdown_60", pd.Series(np.nan, index=idx))
    price_position_60 = df.get("price_position_60", pd.Series(np.nan, index=idx))
    positive_ratio_20 = df.get("positive_return_ratio_20", pd.Series(np.nan, index=idx))
    large_down_ratio_20 = df.get("large_down_day_ratio_20", pd.Series(np.nan, index=idx))
    ulcer_rank = df.get("ulcer_rank_252", pd.Series(np.nan, index=idx))
    bb_rank = df.get("bb_width_rank_252", pd.Series(np.nan, index=idx))

    atr_high = atr_rank > 0.70
    atr_extreme = atr_rank > 0.85
    atr_expanding = atr_ratio > 1.15
    atr_compressed = atr_rank < 0.30

    down_pressure_now = (
        (drawdown_60 < -0.08)
        | (return_20d < -0.05)
        | (large_down_ratio_20 > 0.20)
        | (ulcer_rank > 0.70)
    )
    up_pressure_now = (
        (return_20d > 0.05)
        & (price_position_60 > 0.70)
        & (positive_ratio_20 > 0.55)
        & ~(ulcer_rank > 0.70)
    )
    squeeze_or_breakout = (bb_rank < 0.30) & (atr_ratio > 1.05)

    cond1 = atr_high & atr_expanding & down_pressure_now
    cond2 = (~cond1) & atr_high & atr_expanding & up_pressure_now
    cond3 = (~cond1) & (~cond2) & atr_extreme
    cond4 = (~cond1) & (~cond2) & (~cond3) & atr_compressed & squeeze_or_breakout

    down_threshold = pd.Series(th["down"], index=idx, dtype=float)
    up_threshold = pd.Series(th["up"], index=idx, dtype=float)

    down_threshold = down_threshold.mask(cond1, th["down_loose"])
    up_threshold = up_threshold.mask(cond1, th["up_loose"])
    down_threshold = down_threshold.mask(cond2, th["down_strict"])
    up_threshold = up_threshold.mask(cond2, th["up_loose"])
    down_threshold = down_threshold.mask(cond3, th["down_loose"])
    up_threshold = up_threshold.mask(cond3, th["up_loose"])
    down_threshold = down_threshold.mask(cond4, th["down_loose"])
    up_threshold = up_threshold.mask(cond4, th["up_loose"])

    is_high_vol = (
        (future_vol >= th["vol"])
        | (future_min_ret <= down_threshold)
        | (future_max_ret >= up_threshold)
    )

    labels = pd.Series("정상", index=idx, dtype=object)

    severe_down = future_min_ret <= th["down"]
    down_first = is_high_vol & severe_down & ~up_pressure_now
    labels.loc[down_first] = "하락고변동"

    down_second = is_high_vol & ~down_first & (future_min_ret <= down_threshold)
    labels.loc[down_second] = "하락고변동"

    up_first = (
        is_high_vol
        & ~down_first
        & ~down_second
        & (future_max_ret >= up_threshold)
        & (future_ret > 0)
    )
    labels.loc[up_first] = "상승고변동"

    remaining = is_high_vol & (labels == "정상")
    up_remaining = remaining & (future_max_ret.abs() >= future_min_ret.abs())
    labels.loc[up_remaining] = "상승고변동"
    labels.loc[remaining & ~up_remaining] = "하락고변동"

    return labels


def make_direction_labels(df: pd.DataFrame, horizon: int, cfg: Config, direction: str) -> pd.Series:
    """방향성 이진 라벨 생성.

    - up: future_return_h > +direction_return_threshold
    - down: future_return_h < -direction_return_threshold
    - 중립 구간은 두 모델 모두 0으로 처리된다.
    """
    ret = df[f"future_return_{horizon}d"].astype(float)
    thr = float(getattr(cfg, "direction_return_threshold", 0.005))
    if direction == "up":
        return (ret > thr).astype(int)
    if direction == "down":
        return (ret < -thr).astype(int)
    raise ValueError(f"unknown direction: {direction}")


def assign_direction_label(row: pd.Series, horizon: int, cfg: Config) -> str:
    ret = float(row[f"future_return_{horizon}d"])
    thr = float(getattr(cfg, "direction_return_threshold", 0.005))
    if ret > thr:
        return "상승"
    if ret < -thr:
        return "하락"
    return "중립"


# ============================================================
# 4. MODEL
# ============================================================

def calc_scale_pos_weight(y_binary: np.ndarray) -> float:
    pos = float(np.sum(y_binary == 1))
    neg = float(np.sum(y_binary == 0))
    if pos <= 0 or neg <= 0:
        return 1.0
    return max(0.1, min(20.0, neg / pos))


def make_xgb_stage1(cfg: Config, scale_pos_weight: float, n_estimators: Optional[int] = None) -> Pipeline:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators or cfg.stage1_n_estimators,
        learning_rate=cfg.stage1_learning_rate,
        max_depth=cfg.stage1_max_depth,
        min_child_weight=cfg.stage1_min_child_weight,
        subsample=cfg.stage1_subsample,
        colsample_bytree=cfg.stage1_colsample_bytree,
        reg_lambda=cfg.stage1_reg_lambda,
        reg_alpha=cfg.stage1_reg_alpha,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def make_xgb_downrisk(cfg: Config, scale_pos_weight: float, n_estimators: Optional[int] = None) -> Pipeline:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators or cfg.down_n_estimators,
        learning_rate=cfg.down_learning_rate,
        max_depth=cfg.down_max_depth,
        min_child_weight=cfg.down_min_child_weight,
        subsample=cfg.down_subsample,
        colsample_bytree=cfg.down_colsample_bytree,
        reg_lambda=cfg.down_reg_lambda,
        reg_alpha=cfg.down_reg_alpha,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def safe_auc(y_true: np.ndarray, p: np.ndarray, kind: str) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    if kind == "roc":
        return float(roc_auc_score(y_true, p))
    if kind == "pr":
        return float(average_precision_score(y_true, p))
    raise ValueError(kind)


def policy_imbalance_penalty(y_high: np.ndarray, y_down: np.ndarray) -> float:
    hv_rate = float(np.mean(y_high)) if len(y_high) else 0.0
    down_rate = float(np.mean(y_down)) if len(y_down) else 0.0
    # 너무 희소하거나 너무 넓은 라벨 정책을 방지
    return abs(hv_rate - 0.33) + 0.75 * abs(down_rate - 0.16)


def select_label_policy(train_df: pd.DataFrame, horizon: int, feature_cols: List[str], cfg: Config) -> Tuple[LabelPolicy, Dict[str, float]]:
    if not cfg.use_adaptive_label_policy:
        th = compute_policy_thresholds(train_df, horizon, cfg.fixed_label_policy)
        return cfg.fixed_label_policy, th

    valid_rows = min(cfg.label_search_valid_rows, max(126, len(train_df) // 4))
    if len(train_df) < cfg.min_train_rows + valid_rows:
        th = compute_policy_thresholds(train_df, horizon, cfg.fixed_label_policy)
        return cfg.fixed_label_policy, th

    inner_train = train_df.iloc[:-valid_rows].copy()
    inner_valid = train_df.iloc[-valid_rows:].copy()

    best_score = -np.inf
    best_policy = cfg.fixed_label_policy
    best_th = compute_policy_thresholds(train_df, horizon, cfg.fixed_label_policy)

    X_inner = inner_train[feature_cols]
    X_valid = inner_valid[feature_cols]

    for policy in cfg.label_policy_candidates:
        th_inner = compute_policy_thresholds(inner_train, horizon, policy)
        labels_inner = make_labels(inner_train, horizon, th_inner)
        labels_valid = make_labels(inner_valid, horizon, th_inner)

        y_high = (labels_inner != "정상").astype(int).values
        y_high_valid = (labels_valid != "정상").astype(int).values
        y_down = (labels_inner == "하락고변동").astype(int).values
        y_down_valid = (labels_valid == "하락고변동").astype(int).values

        if len(np.unique(y_high)) < 2 or int(y_high.sum()) < cfg.label_search_min_positive:
            continue

        try:
            m_high = make_xgb_stage1(cfg, calc_scale_pos_weight(y_high), cfg.label_search_stage1_estimators)
            m_high.fit(X_inner, y_high)
            p_high = m_high.predict_proba(X_valid)[:, 1]
            high_pr = safe_auc(y_high_valid, p_high, "pr") or 0.0
            high_roc = safe_auc(y_high_valid, p_high, "roc") or 0.5
        except Exception:
            continue

        down_pr = 0.0
        down_roc = 0.5
        if len(np.unique(y_down)) == 2 and int(y_down.sum()) >= cfg.label_search_min_positive:
            try:
                m_down = make_xgb_downrisk(cfg, calc_scale_pos_weight(y_down), cfg.label_search_down_estimators)
                m_down.fit(X_inner, y_down)
                p_down = m_down.predict_proba(X_valid)[:, 1]
                down_pr = safe_auc(y_down_valid, p_down, "pr") or 0.0
                down_roc = safe_auc(y_down_valid, p_down, "roc") or 0.5
            except Exception:
                pass

        penalty = policy_imbalance_penalty(y_high, y_down)
        score = 0.35 * high_pr + 0.25 * down_pr + 0.20 * high_roc + 0.10 * down_roc - 0.10 * penalty
        if score > best_score:
            best_score = float(score)
            best_policy = policy
            best_th = compute_policy_thresholds(train_df, horizon, policy)

    return best_policy, best_th


# ============================================================
# 5. WALK-FORWARD PREDICTION
# ============================================================

def extract_model_importance(pipeline: Pipeline, feature_cols: List[str]) -> Dict[str, float]:
    try:
        imp = np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float)
        if len(imp) != len(feature_cols):
            return {}
        return {f: float(v) for f, v in zip(feature_cols, imp)}
    except Exception:
        return {}


def mean_importance(history: List[Dict[str, float]]) -> Dict[str, float]:
    if not history:
        return {}
    imp_df = pd.DataFrame(history).fillna(0.0)
    return imp_df.mean(axis=0).sort_values(ascending=False).to_dict()


def ensemble_weights(cfg: Config) -> Tuple[Dict[int, float], Dict[int, float]]:
    hv = {10: cfg.high_vol_weight_h10, 20: cfg.high_vol_weight_h20}
    dn = {10: cfg.down_risk_weight_h10, 20: cfg.down_risk_weight_h20}
    # 사용자 horizons가 달라져도 정규화되도록 처리
    hv = {h: hv.get(h, 1.0 / len(cfg.horizons)) for h in cfg.horizons}
    dn = {h: dn.get(h, 1.0 / len(cfg.horizons)) for h in cfg.horizons}
    hv_sum = sum(hv.values())
    dn_sum = sum(dn.values())
    return {h: v / hv_sum for h, v in hv.items()}, {h: v / dn_sum for h, v in dn.items()}


def run_walk_forward(df: pd.DataFrame, feature_cols: List[str], cfg: Config) -> pd.DataFrame:
    """Walk-forward prediction with v8.6.2 directional Up/Down + overall risk.

    구조:
    - Stage1: 전체 피처로 정상/고변동 예측
    - Up-model: 가격/추세+거래량 피처로 상승 확률 예측
    - Down-model price_trend: 가격/추세 피처로 하락 확률 예측
    - Down-model price_volume: 가격/추세+거래량 피처로 하락 확률 예측
    - Down-model volatility: 변동성/ATR/Range 피처로 하락 확률 예측
    - Overall-risk: 하락 확률, 고변동 확률, 하락-상승 우위 점수를 종합
    """
    future_cols = []
    for h in cfg.horizons:
        future_cols.extend([
            f"future_volatility_{h}d",
            f"future_return_{h}d",
            f"future_max_return_{h}d",
            f"future_min_return_{h}d",
        ])
    valid_cols = feature_cols + future_cols + ["stock_next_return", "bond_next_return", "cash_next_return"]

    work = df.dropna(subset=valid_cols).copy()
    work = work[work.index >= pd.Timestamp(cfg.backtest_start_date)].copy()
    if len(work) < cfg.min_train_rows:
        raise ValueError("백테스트 가능한 데이터가 부족합니다.")

    all_df = df.copy()
    candidate_positions = [all_df.index.get_loc(idx) for idx in work.index]
    max_gap = max(cfg.horizons)

    models: Dict[int, Dict[str, object]] = {}
    last_retrain_k: Optional[int] = None
    prediction_rows: List[Dict[str, object]] = []
    stage1_imp_hist: List[Dict[str, float]] = []
    up_imp_hist: List[Dict[str, float]] = []
    down_imp_hist_by_branch: Dict[str, List[Dict[str, float]]] = {
        "price_trend": [],
        "price_volume": [],
        "volatility": [],
    }
    policy_usage: Dict[str, int] = {}

    hv_w, dn_w = ensemble_weights(cfg)
    down_feature_sets = build_downrisk_feature_sets(feature_cols)
    direction_cols = down_feature_sets.get("price_volume") or down_feature_sets.get("price_trend") or feature_cols
    branch_weights = normalize_downrisk_branch_weights(cfg)

    for k, pos in enumerate(candidate_positions):
        date = all_df.index[pos]
        train_end_pos = pos - max_gap
        if train_end_pos < cfg.min_train_rows:
            continue

        need_retrain = (not models) or (last_retrain_k is None) or (k - last_retrain_k >= cfg.retrain_every_n_days)
        if need_retrain:
            train_df = all_df.iloc[:train_end_pos].copy().dropna(subset=valid_cols)
            if cfg.max_train_rows is not None:
                train_df = train_df.tail(int(cfg.max_train_rows))
            if len(train_df) < cfg.min_train_rows:
                continue

            models = {}
            X_train_full = train_df[feature_cols]
            for h in cfg.horizons:
                policy, th = select_label_policy(train_df, h, feature_cols, cfg)
                labels = make_labels(train_df, h, th)
                y_high = (labels != "정상").astype(int).values
                y_up = make_direction_labels(train_df, h, cfg, "up").values
                y_down = make_direction_labels(train_df, h, cfg, "down").values

                if len(np.unique(y_high)) < 2:
                    continue

                stage1_model = make_xgb_stage1(cfg, calc_scale_pos_weight(y_high))
                stage1_model.fit(X_train_full, y_high)
                imp1 = extract_model_importance(stage1_model, feature_cols)
                if imp1:
                    stage1_imp_hist.append(imp1)

                up_model: Optional[Pipeline] = None
                up_available = False
                if len(np.unique(y_up)) == 2 and int(y_up.sum()) >= int(getattr(cfg, "direction_min_positive", 20)):
                    up_model = make_xgb_downrisk(cfg, calc_scale_pos_weight(y_up))
                    up_model.fit(train_df[direction_cols], y_up)
                    up_available = True
                    impu = extract_model_importance(up_model, direction_cols)
                    if impu:
                        up_imp_hist.append(impu)

                down_models: Dict[str, Optional[Pipeline]] = {"price_trend": None, "price_volume": None, "volatility": None}
                down_available: Dict[str, bool] = {"price_trend": False, "price_volume": False, "volatility": False}
                if len(np.unique(y_down)) == 2 and int(y_down.sum()) >= int(getattr(cfg, "direction_min_positive", 20)):
                    for branch, cols in down_feature_sets.items():
                        if not cols:
                            continue
                        m_down = make_xgb_downrisk(cfg, calc_scale_pos_weight(y_down))
                        m_down.fit(train_df[cols], y_down)
                        down_models[branch] = m_down
                        down_available[branch] = True
                        impd = extract_model_importance(m_down, cols)
                        if impd:
                            down_imp_hist_by_branch[branch].append(impd)

                models[h] = {
                    "stage1": stage1_model,
                    "up_model": up_model,
                    "up_available": up_available,
                    "down_models": down_models,
                    "down_available": down_available,
                    "thresholds": th,
                    "policy": policy,
                }
                policy_usage[f"H{h}:{policy.name}"] = policy_usage.get(f"H{h}:{policy.name}", 0) + 1

            last_retrain_k = k

        if not models:
            continue

        row_df = all_df.iloc[[pos]]
        X_now_full = row_df[feature_cols]
        out: Dict[str, object] = {"Date": date}

        prob_high_ens = 0.0
        prob_up_ens = 0.0
        prob_down_ens = 0.0
        prob_down_branch_ens: Dict[str, float] = {"price_trend": 0.0, "price_volume": 0.0, "volatility": 0.0}
        actual_primary_label = "정상"
        actual_primary_risk = "정상"
        actual_primary_direction = "중립"

        for h in cfg.horizons:
            if h not in models:
                continue
            m = models[h]
            stage1_model = m["stage1"]
            th = m["thresholds"]
            policy = m["policy"]

            p_high = float(stage1_model.predict_proba(X_now_full)[0, 1])  # type: ignore[union-attr]

            up_model = m.get("up_model")
            if up_model is not None and bool(m.get("up_available", False)):
                p_up = float(up_model.predict_proba(row_df[direction_cols])[0, 1])  # type: ignore[union-attr]
            else:
                p_up = 0.0

            branch_probs: Dict[str, float] = {}
            down_models = m.get("down_models", {})
            down_available = m.get("down_available", {})
            for branch, cols in down_feature_sets.items():
                branch_model = down_models.get(branch) if isinstance(down_models, dict) else None
                branch_ok = bool(down_available.get(branch, False)) if isinstance(down_available, dict) else False
                if branch_model is not None and branch_ok and cols:
                    branch_probs[branch] = float(branch_model.predict_proba(row_df[cols])[0, 1])  # type: ignore[union-attr]
                else:
                    branch_probs[branch] = 0.0

            if bool(cfg.use_multi_branch_downrisk):
                p_down = (
                    branch_weights["price_trend"] * branch_probs["price_trend"]
                    + branch_weights["price_volume"] * branch_probs["price_volume"]
                    + branch_weights["volatility"] * branch_probs["volatility"]
                    + branch_weights["high_vol"] * p_high
                )
            else:
                p_down = branch_probs["price_volume"]
            p_down = float(np.clip(p_down, 0.0, 1.0))

            actual_label_h = assign_label(all_df.iloc[pos], h, th)
            actual_risk_h = "고변동" if actual_label_h != "정상" else "정상"
            actual_direction_h = assign_direction_label(all_df.iloc[pos], h, cfg)

            out[f"prob_high_vol_h{h}"] = p_high
            out[f"prob_up_h{h}"] = p_up
            out[f"prob_down_price_trend_h{h}"] = branch_probs["price_trend"]
            out[f"prob_down_price_volume_h{h}"] = branch_probs["price_volume"]
            out[f"prob_down_volatility_h{h}"] = branch_probs["volatility"]
            out[f"prob_down_h{h}"] = p_down
            out[f"prob_down_risk_h{h}"] = p_down  # 호환 컬럼: v8.6.2에서는 방향성 하락 확률로 해석
            out[f"actual_direction_h{h}"] = actual_direction_h
            out[f"actual_split_vol_h{h}"] = actual_label_h
            out[f"actual_risk_h{h}"] = actual_risk_h
            out[f"label_policy_h{h}"] = policy.name  # type: ignore[union-attr]

            prob_high_ens += hv_w.get(h, 0.0) * p_high
            prob_up_ens += dn_w.get(h, 0.0) * p_up
            prob_down_ens += dn_w.get(h, 0.0) * p_down
            for branch in prob_down_branch_ens:
                prob_down_branch_ens[branch] += dn_w.get(h, 0.0) * branch_probs[branch]

            if h == cfg.primary_horizon:
                actual_primary_label = actual_label_h
                actual_primary_risk = actual_risk_h
                actual_primary_direction = actual_direction_h

        prob_high_ens = float(np.clip(prob_high_ens, 0.0, 1.0))
        prob_up_ens = float(np.clip(prob_up_ens, 0.0, 1.0))
        prob_down_ens = float(np.clip(prob_down_ens, 0.0, 1.0))
        for branch in prob_down_branch_ens:
            prob_down_branch_ens[branch] = float(np.clip(prob_down_branch_ens[branch], 0.0, 1.0))
        prob_down_hv = float(np.clip(min(prob_high_ens, prob_down_ens), 0.0, 1.0))
        prob_up_proxy = prob_up_ens
        prob_overall_risk = compute_overall_risk_prob(prob_high_ens, prob_down_ens, cfg, prob_up=prob_up_ens)
        direction_margin = float(getattr(cfg, "direction_decision_margin", 0.05))
        direction_score = float(prob_up_ens - prob_down_ens)
        if direction_score >= direction_margin:
            pred_direction = "상승"
        elif direction_score <= -direction_margin:
            pred_direction = "하락"
        else:
            pred_direction = "중립"

        out.update({
            "actual_risk": actual_primary_risk,
            "actual_split_vol": actual_primary_label,
            "actual_direction": actual_primary_direction,
            "prob_high_vol": prob_high_ens,
            "prob_up": prob_up_ens,
            "prob_down_price_trend": prob_down_branch_ens["price_trend"],
            "prob_down_price_volume": prob_down_branch_ens["price_volume"],
            "prob_down_volatility": prob_down_branch_ens["volatility"],
            "prob_down": prob_down_ens,
            "prob_down_risk": prob_down_ens,  # 호환 컬럼
            "prob_overall_risk": prob_overall_risk,
            "prob_normal": 1.0 - prob_high_ens,
            "prob_down_high_vol": prob_down_hv,
            "prob_up_proxy": prob_up_proxy,
            "direction_score": direction_score,
            "pred_direction": pred_direction,
            "pred_risk": "고변동" if prob_high_ens >= cfg.pred_high_vol_threshold else "정상",
            "pred_overall_risk": "위험" if prob_overall_risk >= cfg.pred_overall_risk_threshold else "정상",
            "pred_split_vol": "하락고변동" if (prob_high_ens >= cfg.pred_high_vol_threshold and prob_down_ens >= cfg.pred_down_risk_threshold) else ("상승고변동" if prob_high_ens >= cfg.pred_high_vol_threshold else "정상"),
            "stock_next_return": float(all_df.iloc[pos]["stock_next_return"]),
            "bond_next_return": float(all_df.iloc[pos]["bond_next_return"]),
            "cash_next_return": float(all_df.iloc[pos]["cash_next_return"]),
        })
        prediction_rows.append(out)

    pred_df = pd.DataFrame(prediction_rows).sort_values("Date").reset_index(drop=True)
    if pred_df.empty:
        raise ValueError("walk-forward 예측 결과가 비어 있습니다.")

    if cfg.use_prob_ewma:
        prob_cols = [
            c for c in pred_df.columns
            if c.startswith("prob_high_vol")
            or c.startswith("prob_up")
            or c.startswith("prob_down_risk")
            or c.startswith("prob_down_h")
            or c.startswith("prob_down_price_trend")
            or c.startswith("prob_down_price_volume")
            or c.startswith("prob_down_volatility")
        ]
        for col in prob_cols:
            if pred_df[col].dtype.kind in "if" and not col.endswith("_raw"):
                raw_col = f"{col}_raw"
                if raw_col not in pred_df.columns:
                    pred_df[raw_col] = pred_df[col]
                pred_df[col] = pred_df[col].ewm(span=cfg.prob_ewma_span, adjust=False).mean()

        pred_df["prob_high_vol"] = pred_df["prob_high_vol"].clip(0.0, 1.0)
        pred_df["prob_up"] = pred_df["prob_up"].clip(0.0, 1.0)
        pred_df["prob_down_price_trend"] = pred_df["prob_down_price_trend"].clip(0.0, 1.0)
        pred_df["prob_down_price_volume"] = pred_df["prob_down_price_volume"].clip(0.0, 1.0)
        pred_df["prob_down_volatility"] = pred_df["prob_down_volatility"].clip(0.0, 1.0)
        if bool(cfg.use_multi_branch_downrisk):
            pred_df["prob_down"] = (
                branch_weights["price_trend"] * pred_df["prob_down_price_trend"]
                + branch_weights["price_volume"] * pred_df["prob_down_price_volume"]
                + branch_weights["volatility"] * pred_df["prob_down_volatility"]
                + branch_weights["high_vol"] * pred_df["prob_high_vol"]
            ).clip(0.0, 1.0)
        else:
            pred_df["prob_down"] = pred_df["prob_down_price_volume"].clip(0.0, 1.0)
        pred_df["prob_down_risk"] = pred_df["prob_down"]
        pred_df["prob_normal"] = 1.0 - pred_df["prob_high_vol"]
        pred_df["prob_down_high_vol"] = np.minimum(pred_df["prob_high_vol"], pred_df["prob_down"]).clip(0.0, 1.0)
        pred_df["prob_up_proxy"] = pred_df["prob_up"]
        pred_df["direction_score"] = pred_df["prob_up"] - pred_df["prob_down"]
        margin = float(getattr(cfg, "direction_decision_margin", 0.05))
        pred_df["pred_direction"] = np.where(
            pred_df["direction_score"] >= margin,
            "상승",
            np.where(pred_df["direction_score"] <= -margin, "하락", "중립"),
        )
        pred_df["prob_overall_risk"] = compute_overall_risk_prob(
            pred_df["prob_high_vol"], pred_df["prob_down"], cfg, prob_up=pred_df["prob_up"]
        )
        pred_df["pred_risk"] = np.where(pred_df["prob_high_vol"] >= cfg.pred_high_vol_threshold, "고변동", "정상")
        pred_df["pred_overall_risk"] = np.where(pred_df["prob_overall_risk"] >= cfg.pred_overall_risk_threshold, "위험", "정상")
        pred_df["pred_split_vol"] = np.where(
            pred_df["pred_risk"] == "정상",
            "정상",
            np.where(pred_df["prob_down"] >= cfg.pred_down_risk_threshold, "하락고변동", "상승고변동"),
        )

    down_weighted_imp = combine_weighted_importance(down_imp_hist_by_branch, branch_weights)
    pred_df.attrs["stage1_feature_importance_mean"] = mean_importance(stage1_imp_hist)
    pred_df.attrs["up_feature_importance_mean"] = mean_importance(up_imp_hist)
    pred_df.attrs["downrisk_feature_importance_mean"] = down_weighted_imp
    pred_df.attrs["downrisk_price_trend_feature_importance_mean"] = mean_importance(down_imp_hist_by_branch["price_trend"])
    pred_df.attrs["downrisk_price_volume_feature_importance_mean"] = mean_importance(down_imp_hist_by_branch["price_volume"])
    pred_df.attrs["downrisk_volatility_feature_importance_mean"] = mean_importance(down_imp_hist_by_branch["volatility"])
    pred_df.attrs["downrisk_branch_weights"] = branch_weights
    pred_df.attrs["downrisk_feature_sets"] = down_feature_sets
    pred_df.attrs["direction_feature_set"] = direction_cols
    pred_df.attrs["policy_usage"] = policy_usage
    return pred_df


# ============================================================
# 6. ALLOCATION / BACKTEST
# ============================================================

def _normalize_weight_tuple(stock: float, bond: float, cash: float) -> Tuple[float, float, float]:
    vals = np.asarray([stock, bond, cash], dtype=float)
    vals = np.clip(vals, 0.0, 1.0)
    total = float(vals.sum())
    if total <= 0:
        return 1.0, 0.0, 0.0
    vals = vals / total
    return float(vals[0]), float(vals[1]), float(vals[2])


def gate_config_from_cfg(cfg: Config) -> Dict[str, float]:
    return {
        "gate_normal_high_vol_threshold": cfg.gate_normal_high_vol_threshold,
        "gate_high_vol_threshold": cfg.gate_high_vol_threshold,
        "gate_riskoff_downrisk_threshold": cfg.gate_riskoff_downrisk_threshold,
        "gate_watch_downrisk_threshold": cfg.gate_watch_downrisk_threshold,
        "use_three_regime_allocation": cfg.use_three_regime_allocation,
        "use_extreme_risk_cut": cfg.use_extreme_risk_cut,
        "extreme_high_vol_threshold": cfg.extreme_high_vol_threshold,
        "extreme_downrisk_threshold": cfg.extreme_downrisk_threshold,
        "extreme_stock_weight": cfg.extreme_stock_weight,
        "extreme_bond_weight": cfg.extreme_bond_weight,
        "extreme_cash_weight": cfg.extreme_cash_weight,
        "normal_stock_weight": cfg.normal_stock_weight,
        "normal_bond_weight": cfg.normal_bond_weight,
        "normal_cash_weight": cfg.normal_cash_weight,
        "watch_stock_weight": cfg.watch_stock_weight,
        "watch_bond_weight": cfg.watch_bond_weight,
        "watch_cash_weight": cfg.watch_cash_weight,
        "high_vol_stock_weight": cfg.high_vol_stock_weight,
        "high_vol_bond_weight": cfg.high_vol_bond_weight,
        "high_vol_cash_weight": cfg.high_vol_cash_weight,
        "risk_off_stock_weight": cfg.risk_off_stock_weight,
        "risk_off_bond_weight": cfg.risk_off_bond_weight,
        "risk_off_cash_weight": cfg.risk_off_cash_weight,
        "no_trade_band": cfg.no_trade_band,
        "name": "default_v8_4_gate",
    }


def classify_gate(prob_high_vol: float, prob_down_risk: float, g: Dict[str, float]) -> str:
    """
    v8.4 allocation gate.

    기본값은 3-regime 구조다.
    - NORMAL: 고변동 확률이 충분히 낮은 구간
    - WATCH: 정상은 아니지만 RISK_OFF 조건은 아닌 구간
    - RISK_OFF: 고변동 확률과 하락위험 확률이 동시에 높은 구간

    EXTREME_RISK는 RISK_OFF 내부의 추가 방어 sub-regime이다.
    HIGH_VOL은 v8.3 진단에서 표본이 작고 turnover가 높아 기본 구조에서 제거했다.
    """
    ph = float(np.clip(prob_high_vol, 0.0, 1.0))
    pdn = float(np.clip(prob_down_risk, 0.0, 1.0))

    if ph < g["gate_normal_high_vol_threshold"]:
        return "NORMAL"

    if bool(g.get("use_three_regime_allocation", True)):
        if ph >= g["gate_high_vol_threshold"] and pdn >= g["gate_riskoff_downrisk_threshold"]:
            if bool(g.get("use_extreme_risk_cut", True)):
                if ph >= g.get("extreme_high_vol_threshold", 0.75) and pdn >= g.get("extreme_downrisk_threshold", 0.65):
                    return "EXTREME_RISK"
            return "RISK_OFF"
        return "WATCH"

    # 이전 4-regime 구조를 옵션으로 유지
    if ph < g["gate_high_vol_threshold"]:
        if pdn >= g["gate_watch_downrisk_threshold"]:
            return "HIGH_VOL"
        return "WATCH"
    if pdn >= g["gate_riskoff_downrisk_threshold"]:
        return "RISK_OFF"
    return "HIGH_VOL"


def base_weight_for_regime(regime: str, g: Dict[str, float]) -> Tuple[float, float, float]:
    if regime == "NORMAL":
        return _normalize_weight_tuple(g["normal_stock_weight"], g["normal_bond_weight"], g["normal_cash_weight"])
    if regime == "WATCH":
        return _normalize_weight_tuple(g["watch_stock_weight"], g["watch_bond_weight"], g["watch_cash_weight"])
    if regime == "HIGH_VOL":
        return _normalize_weight_tuple(g["high_vol_stock_weight"], g["high_vol_bond_weight"], g["high_vol_cash_weight"])
    if regime == "EXTREME_RISK":
        return _normalize_weight_tuple(g["extreme_stock_weight"], g["extreme_bond_weight"], g["extreme_cash_weight"])
    return _normalize_weight_tuple(g["risk_off_stock_weight"], g["risk_off_bond_weight"], g["risk_off_cash_weight"])


def apply_continuous_adjustment(
    base_w: Tuple[float, float, float],
    prob_high_vol: float,
    prob_down_risk: float,
    cfg: Config,
) -> Tuple[float, float, float]:
    if not cfg.use_continuous_adjustment:
        return base_w
    stock, bond, cash = base_w
    cut = cfg.continuous_high_vol_weight * prob_high_vol + cfg.continuous_down_risk_weight * prob_down_risk
    cut = float(np.clip(cut, 0.0, cfg.max_continuous_stock_cut))
    new_stock = max(0.0, stock - cut)
    defensive_add = stock - new_stock
    defensive_total = bond + cash
    if defensive_total <= 0:
        return _normalize_weight_tuple(new_stock, defensive_add * 0.65, defensive_add * 0.35)
    new_bond = bond + defensive_add * bond / defensive_total
    new_cash = cash + defensive_add * cash / defensive_total
    return _normalize_weight_tuple(new_stock, new_bond, new_cash)


def allocate_from_probs(
    prob_high_vol: float,
    prob_down_risk: float,
    g: Dict[str, float],
    cfg: Config,
    prev_weights: Optional[Tuple[float, float, float]],
) -> Tuple[Tuple[float, float, float], str]:
    regime = classify_gate(prob_high_vol, prob_down_risk, g)
    target = base_weight_for_regime(regime, g)
    target = apply_continuous_adjustment(target, prob_high_vol, prob_down_risk, cfg)

    if prev_weights is not None:
        total_delta = sum(abs(target[i] - prev_weights[i]) for i in range(3))
        if total_delta < g["no_trade_band"]:
            return prev_weights, regime
    return target, regime


def perf_stats(returns: pd.Series, initial_capital: float) -> Dict[str, float]:
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return {"final_capital": initial_capital, "total_return": 0.0, "cagr": 0.0, "mdd": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0}
    equity = initial_capital * (1.0 + r).cumprod()
    final_capital = float(equity.iloc[-1])
    total_return = final_capital / initial_capital - 1.0
    years = len(r) / 252.0
    cagr = (final_capital / initial_capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    vol = float(r.std())
    sharpe = float((r.mean() / vol) * math.sqrt(252)) if vol > 0 else 0.0
    downside = r[r < 0]
    down_std = float(downside.std())
    sortino = float((r.mean() / down_std) * math.sqrt(252)) if down_std > 0 else 0.0
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    return {
        "final_capital": final_capital,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "mdd": mdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
    }


def simulate_gate_config(pred_df: pd.DataFrame, g: Dict[str, float], cfg: Config) -> Dict[str, float]:
    prev_w: Optional[Tuple[float, float, float]] = None
    rets: List[float] = []
    turnovers: List[float] = []
    stock_weights: List[float] = []
    for _, row in pred_df.iterrows():
        w, _ = allocate_from_probs(float(row["prob_high_vol"]), float(row["prob_down_risk"]), g, cfg, prev_w)
        turnover = 0.0 if prev_w is None else sum(abs(w[i] - prev_w[i]) for i in range(3))
        gross = w[0] * row["stock_next_return"] + w[1] * row["bond_next_return"] + w[2] * row["cash_next_return"]
        net = gross - cfg.transaction_cost_rate * turnover
        rets.append(float(net))
        turnovers.append(float(turnover))
        stock_weights.append(float(w[0]))
        prev_w = w
    stats = perf_stats(pd.Series(rets), cfg.initial_capital)
    stats["avg_turnover"] = float(np.mean(turnovers)) if turnovers else 0.0
    stats["avg_stock_weight"] = float(np.mean(stock_weights)) if stock_weights else 0.0
    return stats


def build_small_gate_grid(cfg: Config) -> List[Dict[str, float]]:
    grid: List[Dict[str, float]] = []
    base = gate_config_from_cfg(cfg)
    i = 0
    for nht in [0.35, 0.40]:
        for hht in [0.55, 0.60, 0.65]:
            if hht <= nht:
                continue
            for rdt in [0.50, 0.55]:
                g = dict(base)
                g["gate_normal_high_vol_threshold"] = nht
                g["gate_high_vol_threshold"] = hht
                g["gate_riskoff_downrisk_threshold"] = rdt
                g["gate_watch_downrisk_threshold"] = max(0.60, rdt + 0.10)
                g["name"] = f"gate_{i:03d}_n{nht:.2f}_h{hht:.2f}_d{rdt:.2f}"
                grid.append(g)
                i += 1
    return grid


def gate_score(stats: Dict[str, float], cfg: Config) -> float:
    annual_turnover = stats.get("avg_turnover", 0.0) * 252.0
    return float(
        cfg.gate_score_cagr_weight * stats.get("cagr", 0.0)
        - cfg.gate_score_mdd_weight * abs(stats.get("mdd", 0.0))
        - cfg.gate_score_turnover_weight * annual_turnover
    )




def infer_regime_from_weights(weights: Tuple[float, float, float], g: Dict[str, float]) -> str:
    """실제 실행 비중과 가장 가까운 regime을 역산한다."""
    candidates = ["NORMAL", "WATCH", "HIGH_VOL", "RISK_OFF", "EXTREME_RISK"]
    best_name = "CUSTOM"
    best_dist = float("inf")
    for name in candidates:
        bw = base_weight_for_regime(name, g)
        dist = sum(abs(float(weights[i]) - float(bw[i])) for i in range(3))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name if best_dist <= 0.08 else "CUSTOM"

def apply_allocation(pred_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    pred_df = pred_df.copy().reset_index(drop=True)
    default_g = gate_config_from_cfg(cfg)
    grid = build_small_gate_grid(cfg)
    current_g = default_g
    usage: Dict[str, int] = {}

    prev_w: Optional[Tuple[float, float, float]] = None
    rows: List[Dict[str, object]] = []
    last_emergency_i = -10**9

    for i, row in pred_df.iterrows():
        if cfg.use_rolling_gate_optimization and i >= cfg.gate_min_window and i % cfg.gate_optimize_every_n_days == 0:
            hist = pred_df.iloc[max(0, i - cfg.gate_rolling_window):i].copy()
            best_g = current_g
            best_score = -np.inf
            for cand in grid:
                st = simulate_gate_config(hist, cand, cfg)
                s = gate_score(st, cfg)
                if s > best_score:
                    best_score = s
                    best_g = cand
            current_g = best_g

        ph = float(row["prob_high_vol"])
        pdn = float(row.get("prob_down", row.get("prob_down_risk", 0.0)))
        raw_emergency = (
            ph >= cfg.emergency_high_vol_threshold
            or (ph >= cfg.emergency_combined_high_vol_threshold and pdn >= cfg.emergency_combined_down_threshold)
        )
        emergency = bool(raw_emergency and (i - last_emergency_i >= cfg.emergency_cooldown_days))
        scheduled = (i % cfg.rebalance_every_n_days == 0)
        rebalance_due = prev_w is None or scheduled or emergency

        signal_regime = classify_gate(ph, pdn, current_g)
        signal_w = apply_continuous_adjustment(base_weight_for_regime(signal_regime, current_g), ph, pdn, cfg)

        hold_reason = "rebalanced"
        trade_executed = False
        if prev_w is None:
            w = signal_w
            executed_regime = signal_regime
            hold_reason = "initial"
            trade_executed = True
        elif not rebalance_due:
            w = prev_w
            executed_regime = infer_regime_from_weights(w, current_g)
            hold_reason = "not_rebalance_day"
        else:
            total_delta_to_signal = sum(abs(signal_w[j] - prev_w[j]) for j in range(3))
            if total_delta_to_signal < current_g["no_trade_band"]:
                w = prev_w
                executed_regime = infer_regime_from_weights(w, current_g)
                hold_reason = "no_trade_band"
            else:
                w = signal_w
                executed_regime = signal_regime
                hold_reason = "emergency" if emergency else "scheduled"
                trade_executed = True

        turnover = 0.0 if prev_w is None else sum(abs(w[j] - prev_w[j]) for j in range(3))
        if turnover > 1e-12:
            trade_executed = True
        gross = w[0] * row["stock_next_return"] + w[1] * row["bond_next_return"] + w[2] * row["cash_next_return"]
        cost = cfg.transaction_cost_rate * turnover
        net = gross - cost

        if emergency and rebalance_due:
            last_emergency_i = i

        out = row.to_dict()
        out.update({
            "signal_regime": signal_regime,
            "allocation_regime": executed_regime,
            "executed_regime": executed_regime,
            "hold_reason": hold_reason,
            "held_by_no_trade_band": bool(hold_reason == "no_trade_band"),
            "held_by_schedule": bool(hold_reason == "not_rebalance_day"),
            "signal_stock_weight": float(signal_w[0]),
            "signal_bond_weight": float(signal_w[1]),
            "signal_cash_weight": float(signal_w[2]),
            "stock_weight": float(w[0]),
            "bond_weight": float(w[1]),
            "cash_weight": float(w[2]),
            "turnover": float(turnover),
            "transaction_cost": float(cost),
            "strategy_return_gross": float(gross),
            "strategy_return_net": float(net),
            "rebalance_due": bool(rebalance_due),
            "rebalanced": bool(rebalance_due),
            "trade_executed": bool(trade_executed),
            "emergency_rebalance": bool(emergency and rebalance_due),
            "gate_config": current_g["name"],
        })
        rows.append(out)
        usage[current_g["name"]] = usage.get(current_g["name"], 0) + 1
        prev_w = w

    out_df = pd.DataFrame(rows)
    out_df["strategy_equity_net"] = cfg.initial_capital * (1.0 + out_df["strategy_return_net"]).cumprod()
    out_df["strategy_equity_gross"] = cfg.initial_capital * (1.0 + out_df["strategy_return_gross"]).cumprod()
    return out_df, usage


# ============================================================
# 7. METRICS / SUMMARY
# ============================================================

def binary_cls_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float, pos_name: str) -> Dict[str, object]:
    y_pred = (prob >= threshold).astype(int)
    out: Dict[str, object] = {
        "rows": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, np.clip(prob, 0.0, 1.0))),
        "support_positive": int(np.sum(y_true == 1)),
        "support_negative": int(np.sum(y_true == 0)),
        "pred_positive_ratio": float(np.mean(y_pred)),
        "positive_class": pos_name,
    }
    out["roc_auc"] = safe_auc(y_true, prob, "roc")
    out["pr_auc"] = safe_auc(y_true, prob, "pr")
    return out


def classification_metrics(pred_df: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for h in cfg.horizons:
        if f"actual_risk_h{h}" in pred_df.columns and f"prob_high_vol_h{h}" in pred_df.columns:
            y = (pred_df[f"actual_risk_h{h}"] == "고변동").astype(int).values
            p = pred_df[f"prob_high_vol_h{h}"].astype(float).clip(0.0, 1.0).values
            metrics[f"stage1_h{h}"] = binary_cls_metrics(y, p, cfg.pred_high_vol_threshold, "고변동")
        if f"actual_direction_h{h}" in pred_df.columns and f"prob_up_h{h}" in pred_df.columns:
            y_up = (pred_df[f"actual_direction_h{h}"] == "상승").astype(int).values
            p_up = pred_df[f"prob_up_h{h}"].astype(float).clip(0.0, 1.0).values
            metrics[f"up_h{h}"] = binary_cls_metrics(y_up, p_up, 0.50, "상승")
        if f"actual_direction_h{h}" in pred_df.columns and f"prob_down_h{h}" in pred_df.columns:
            y_down = (pred_df[f"actual_direction_h{h}"] == "하락").astype(int).values
            p_down = pred_df[f"prob_down_h{h}"].astype(float).clip(0.0, 1.0).values
            metrics[f"down_h{h}"] = binary_cls_metrics(y_down, p_down, cfg.pred_down_risk_threshold, "하락")

    y_primary = (pred_df["actual_risk"] == "고변동").astype(int).values
    p_ens = pred_df["prob_high_vol"].astype(float).clip(0.0, 1.0).values
    metrics["stage1_ensemble_vs_primary"] = binary_cls_metrics(y_primary, p_ens, cfg.pred_high_vol_threshold, "고변동")

    y_up_primary = (pred_df["actual_direction"] == "상승").astype(int).values
    y_down_primary = (pred_df["actual_direction"] == "하락").astype(int).values
    if "prob_up" in pred_df.columns:
        p_up = pred_df["prob_up"].astype(float).clip(0.0, 1.0).values
        metrics["up_ensemble_vs_primary"] = binary_cls_metrics(y_up_primary, p_up, 0.50, "상승")
    p_down_ens = pred_df["prob_down"].astype(float).clip(0.0, 1.0).values if "prob_down" in pred_df.columns else pred_df["prob_down_risk"].astype(float).clip(0.0, 1.0).values
    metrics["down_ensemble_vs_primary"] = binary_cls_metrics(y_down_primary, p_down_ens, cfg.pred_down_risk_threshold, "하락")

    if "prob_overall_risk" in pred_df.columns:
        p_overall = pred_df["prob_overall_risk"].astype(float).clip(0.0, 1.0).values
        metrics["overall_risk_vs_highvol_primary"] = binary_cls_metrics(
            y_primary,
            p_overall,
            cfg.pred_overall_risk_threshold,
            "전체위험_by_고변동",
        )
        metrics["overall_risk_vs_down_primary"] = binary_cls_metrics(
            y_down_primary,
            p_overall,
            cfg.pred_overall_risk_threshold,
            "전체위험_by_하락",
        )

    for branch, col in [
        ("price_trend", "prob_down_price_trend"),
        ("price_volume", "prob_down_price_volume"),
        ("volatility", "prob_down_volatility"),
    ]:
        if col in pred_df.columns:
            p_branch = pred_df[col].astype(float).clip(0.0, 1.0).values
            metrics[f"down_{branch}_vs_primary"] = binary_cls_metrics(
                y_down_primary,
                p_branch,
                cfg.pred_down_risk_threshold,
                f"하락_{branch}",
            )

    labels = ["상승", "중립", "하락"]
    if "actual_direction" in pred_df.columns and "pred_direction" in pred_df.columns:
        y_true = pd.Categorical(pred_df["actual_direction"], categories=labels).codes
        y_pred = pd.Categorical(pred_df["pred_direction"], categories=labels).codes
        valid = (y_true >= 0) & (y_pred >= 0)
        metrics["final_direction_3state_vs_primary"] = {
            "rows": int(valid.sum()),
            "accuracy": float(accuracy_score(y_true[valid], y_pred[valid])) if valid.any() else 0.0,
            "macro_f1": float(f1_score(y_true[valid], y_pred[valid], average="macro", zero_division=0)) if valid.any() else 0.0,
            "label_support": pred_df["actual_direction"].value_counts().to_dict(),
            "report": classification_report(y_true[valid], y_pred[valid], labels=[0, 1, 2], target_names=labels, output_dict=True, zero_division=0) if valid.any() else {},
        }
    return metrics

def _pct_weight_dict(row: pd.Series, prefix: str) -> Dict[str, float]:
    return {
        "stock": round(float(row[f"{prefix}stock_weight"]) * 100, 2),
        "bond": round(float(row[f"{prefix}bond_weight"]) * 100, 2),
        "cash": round(float(row[f"{prefix}cash_weight"]) * 100, 2),
    }


def build_summary(pred_df: pd.DataFrame, feature_cols: List[str], gate_usage: Dict[str, int], cfg: Config) -> Dict[str, object]:
    perf = {
        "strategy_after_cost": perf_stats(pred_df["strategy_return_net"], cfg.initial_capital),
        "strategy_gross": perf_stats(pred_df["strategy_return_gross"], cfg.initial_capital),
        "stock_buy_hold": perf_stats(pred_df["stock_next_return"], cfg.initial_capital),
        "benchmark_60_40": perf_stats(0.6 * pred_df["stock_next_return"] + 0.4 * pred_df["bond_next_return"], cfg.initial_capital),
        "static_50_30_20": perf_stats(0.5 * pred_df["stock_next_return"] + 0.3 * pred_df["bond_next_return"] + 0.2 * pred_df["cash_next_return"], cfg.initial_capital),
    }
    latest = pred_df.iloc[-1]
    signal_alloc = {
        "stock": round(float(latest.get("signal_stock_weight", latest["stock_weight"])) * 100, 2),
        "bond": round(float(latest.get("signal_bond_weight", latest["bond_weight"])) * 100, 2),
        "cash": round(float(latest.get("signal_cash_weight", latest["cash_weight"])) * 100, 2),
    }
    executed_alloc = {
        "stock": round(float(latest["stock_weight"]) * 100, 2),
        "bond": round(float(latest["bond_weight"]) * 100, 2),
        "cash": round(float(latest["cash_weight"]) * 100, 2),
    }
    return {
        "model_type": "xgb_multi_branch_directional_risk_v8_6_2_diagnostics",
        "target_ticker": cfg.target_ticker,
        "bond_ticker": cfg.bond_ticker,
        "cash_ticker": cfg.cash_ticker,
        "config": asdict(cfg),
        "period": {"start": str(pred_df["Date"].iloc[0]), "end": str(pred_df["Date"].iloc[-1]), "rows": int(len(pred_df))},
        "feature_count": int(len(feature_cols)),
        "feature_set": "directional_up_down_price_trend_price_volume_volatility_features",
        "feature_cols": feature_cols,
        "stage1_feature_importance_mean": pred_df.attrs.get("stage1_feature_importance_mean", {}),
        "up_feature_importance_mean": pred_df.attrs.get("up_feature_importance_mean", {}),
        "downrisk_feature_importance_mean": pred_df.attrs.get("downrisk_feature_importance_mean", {}),
        "downrisk_price_trend_feature_importance_mean": pred_df.attrs.get("downrisk_price_trend_feature_importance_mean", {}),
        "downrisk_price_volume_feature_importance_mean": pred_df.attrs.get("downrisk_price_volume_feature_importance_mean", {}),
        "downrisk_volatility_feature_importance_mean": pred_df.attrs.get("downrisk_volatility_feature_importance_mean", {}),
        "downrisk_branch_weights": pred_df.attrs.get("downrisk_branch_weights", {}),
        "downrisk_feature_sets": pred_df.attrs.get("downrisk_feature_sets", {}),
        "direction_feature_set": pred_df.attrs.get("direction_feature_set", []),
        "label_policy_usage": pred_df.attrs.get("policy_usage", {}),
        "average_probabilities": {
            "avg_prob_normal": float(pred_df["prob_normal"].mean()),
            "avg_prob_high_vol": float(pred_df["prob_high_vol"].mean()),
            "avg_prob_up": float(pred_df["prob_up"].mean()) if "prob_up" in pred_df.columns else 0.0,
            "avg_prob_down": float(pred_df["prob_down"].mean()) if "prob_down" in pred_df.columns else float(pred_df["prob_down_risk"].mean()),
            "avg_direction_score": float(pred_df["direction_score"].mean()) if "direction_score" in pred_df.columns else 0.0,
            "avg_prob_overall_risk": float(pred_df["prob_overall_risk"].mean()) if "prob_overall_risk" in pred_df.columns else 0.0,
            "avg_prob_down_price_trend": float(pred_df["prob_down_price_trend"].mean()) if "prob_down_price_trend" in pred_df.columns else 0.0,
            "avg_prob_down_price_volume": float(pred_df["prob_down_price_volume"].mean()) if "prob_down_price_volume" in pred_df.columns else 0.0,
            "avg_prob_down_volatility": float(pred_df["prob_down_volatility"].mean()) if "prob_down_volatility" in pred_df.columns else 0.0,
            "avg_prob_down_high_vol": float(pred_df["prob_down_high_vol"].mean()),
        },
        "average_weights": {
            "avg_stock_weight": float(pred_df["stock_weight"].mean()),
            "avg_bond_weight": float(pred_df["bond_weight"].mean()),
            "avg_cash_weight": float(pred_df["cash_weight"].mean()),
            "min_stock_weight": float(pred_df["stock_weight"].min()),
            "max_stock_weight": float(pred_df["stock_weight"].max()),
        },
        "allocation_regime_distribution_pct": pred_df["allocation_regime"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "signal_regime_distribution_pct": pred_df["signal_regime"].value_counts(normalize=True).mul(100).round(2).to_dict() if "signal_regime" in pred_df.columns else {},
        "turnover": {
            "avg_daily_trade_ratio": float(pred_df["turnover"].mean()),
            "annual_turnover_estimate": float(pred_df["turnover"].mean() * 252.0),
            "total_transaction_cost_rate_sum": float(pred_df["transaction_cost"].sum()),
            "rebalance_due_ratio": float(pred_df["rebalance_due"].mean()) if "rebalance_due" in pred_df.columns else float(pred_df["rebalanced"].mean()),
            "trade_executed_ratio": float(pred_df["trade_executed"].mean()) if "trade_executed" in pred_df.columns else float((pred_df["turnover"] > 1e-12).mean()),
            "rebalance_ratio": float(pred_df["rebalanced"].mean()),
            "emergency_rebalance_ratio": float(pred_df["emergency_rebalance"].mean()),
        },
        "performance": perf,
        "classification": classification_metrics(pred_df, cfg),
        "gate_config_usage_top10": dict(sorted(gate_usage.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "latest_prediction": {
            "date": str(latest["Date"]),
            "pred_risk": str(latest["pred_risk"]),
            "pred_direction": str(latest.get("pred_direction", "중립")),
            "pred_overall_risk": str(latest.get("pred_overall_risk", "정상")),
            "prob_normal": round(float(latest["prob_normal"]) * 100, 2),
            "prob_high_vol": round(float(latest["prob_high_vol"]) * 100, 2),
            "prob_up": round(float(latest.get("prob_up", 0.0)) * 100, 2),
            "prob_down": round(float(latest.get("prob_down", latest.get("prob_down_risk", 0.0))) * 100, 2),
            "direction_score": round(float(latest.get("direction_score", 0.0)) * 100, 2),
            "prob_overall_risk": round(float(latest.get("prob_overall_risk", 0.0)) * 100, 2),
            "prob_down_price_trend": round(float(latest.get("prob_down_price_trend", 0.0)) * 100, 2),
            "prob_down_price_volume": round(float(latest.get("prob_down_price_volume", 0.0)) * 100, 2),
            "prob_down_volatility": round(float(latest.get("prob_down_volatility", 0.0)) * 100, 2),
            "prob_down_high_vol": round(float(latest["prob_down_high_vol"]) * 100, 2),
            "signal_regime": str(latest.get("signal_regime", latest["allocation_regime"])),
            "allocation_regime": str(latest["allocation_regime"]),
            "executed_regime": str(latest.get("executed_regime", latest["allocation_regime"])),
            "hold_reason": str(latest.get("hold_reason", "unknown")),
            "held_by_no_trade_band": bool(latest.get("held_by_no_trade_band", False)),
            "held_by_schedule": bool(latest.get("held_by_schedule", False)),
            "signal_allocation": signal_alloc,
            "target_allocation": signal_alloc,
            "executed_allocation": executed_alloc,
        },
    }


def add_condition_period_summary(
    summary: Dict[str, object],
    pred_df: pd.DataFrame,
    cfg: Config,
    split_date: str,
) -> None:
    """
    condition search를 사용했을 때 조건 선택 구간과 holdout 구간의 성과를 분리 저장한다.

    이유:
    - 최종 전체 구간 성과만 보면 조건 선택에 사용된 구간과 사후 검증 구간이 섞인다.
    - holdout 성과를 별도 저장해야 과최적화 여부를 확인할 수 있다.
    """
    if pred_df.empty or "Date" not in pred_df.columns:
        summary["condition_period_performance"] = {"error": "pred_df is empty or Date column is missing"}
        return

    dates = pd.to_datetime(pred_df["Date"])
    split_ts = pd.Timestamp(split_date)
    select_df = pred_df[dates <= split_ts].copy()
    holdout_df = pred_df[dates > split_ts].copy()

    def _period_block(df_part: pd.DataFrame) -> Dict[str, object]:
        if df_part.empty:
            return {
                "start": None,
                "end": None,
                "rows": 0,
                "strategy_after_cost": {},
                "stock_buy_hold": {},
                "benchmark_60_40": {},
                "static_50_30_20": {},
            }
        return {
            "start": str(df_part["Date"].iloc[0]),
            "end": str(df_part["Date"].iloc[-1]),
            "rows": int(len(df_part)),
            "strategy_after_cost": perf_stats(df_part["strategy_return_net"], cfg.initial_capital),
            "stock_buy_hold": perf_stats(df_part["stock_next_return"], cfg.initial_capital),
            "benchmark_60_40": perf_stats(
                0.6 * df_part["stock_next_return"] + 0.4 * df_part["bond_next_return"],
                cfg.initial_capital,
            ),
            "static_50_30_20": perf_stats(
                0.5 * df_part["stock_next_return"]
                + 0.3 * df_part["bond_next_return"]
                + 0.2 * df_part["cash_next_return"],
                cfg.initial_capital,
            ),
        }

    summary["condition_period_performance"] = {
        "split_date": split_date,
        "select_period": _period_block(select_df),
        "holdout_period": _period_block(holdout_df),
    }

def print_summary(summary: Dict[str, object]) -> None:
    p = summary["performance"]
    w = summary["average_weights"]
    t = summary["turnover"]
    cls = summary["classification"]
    print("\n==============================")
    print("XGBoost v8.6.2 Diagnostics 결과 요약")
    print("H10/H20 Stage1 + Up/Down Direction + Overall Risk Allocation")
    print("==============================")
    print(f"기간: {summary['period']['start']} ~ {summary['period']['end']}")
    print(f"거래일 수: {summary['period']['rows']}")
    print(f"피처 수: {summary['feature_count']}")
    print(f"평균 주식 비중: {w['avg_stock_weight'] * 100:.2f}%")
    print(f"평균 채권 비중: {w['avg_bond_weight'] * 100:.2f}%")
    print(f"평균 현금 비중: {w['avg_cash_weight'] * 100:.2f}%")
    print(f"연간 교체율 추정: {t['annual_turnover_estimate'] * 100:.2f}%")
    print(f"리밸런싱 도래 비율: {t['rebalance_ratio'] * 100:.2f}%")
    if 'trade_executed_ratio' in t:
        print(f"실제 거래 발생 비율: {t['trade_executed_ratio'] * 100:.2f}%")
    print(f"긴급 리밸런싱 비율: {t['emergency_rebalance_ratio'] * 100:.2f}%")
    print(f"배분 regime 분포: {summary['allocation_regime_distribution_pct']}")

    for name in ["strategy_after_cost", "strategy_gross", "stock_buy_hold", "benchmark_60_40", "static_50_30_20"]:
        st = p[name]
        print(f"\n[{name}]")
        print(f"최종 자산: {st['final_capital']:,.0f}")
        print(f"총수익률: {st['total_return'] * 100:.2f}%")
        print(f"CAGR: {st['cagr'] * 100:.2f}%")
        print(f"MDD: {st['mdd'] * 100:.2f}%")
        print(f"Sharpe: {st['sharpe']:.4f}")
        print(f"Sortino: {st['sortino']:.6f}")
        print(f"Calmar: {st['calmar']:.6f}")

    print("\n[분류 성능 핵심]")
    for key in [
        "stage1_h10", "stage1_h20", "stage1_ensemble_vs_primary",
        "up_h10", "up_h20", "up_ensemble_vs_primary",
        "down_h10", "down_h20", "down_ensemble_vs_primary",
        "overall_risk_vs_highvol_primary", "overall_risk_vs_down_primary",
        "down_price_trend_vs_primary", "down_price_volume_vs_primary", "down_volatility_vs_primary",
    ]:
        if key in cls:
            m = cls[key]
            print(f"{key:30s} | ROC {m['roc_auc']} | PR {m['pr_auc']} | F1 {m['f1']:.4f} | Recall {m['recall']:.4f}")

    print("\n[최신 예측]")
    print(json.dumps(summary["latest_prediction"], ensure_ascii=False, indent=2))



# ============================================================
# 8. OBJECTIVE CONDITION SEARCH
# ============================================================


def _slice_by_date(df: pd.DataFrame, end_date: Optional[str] = None, start_date: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["Date"])
    if start_date is not None:
        out = out[dates >= pd.Timestamp(start_date)]
        dates = pd.to_datetime(out["Date"])
    if end_date is not None:
        out = out[dates <= pd.Timestamp(end_date)]
    return out.copy()


def allocated_subset_stats(df: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    """성과 지표 + 조건 최적화용 운용 지표를 함께 계산한다."""
    empty = {
        "final_capital": cfg.initial_capital,
        "total_return": 0.0,
        "cagr": 0.0,
        "mdd": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "gross_cagr": 0.0,
        "cost_cagr_drag": 0.0,
        "annual_turnover": 0.0,
        "total_turnover": 0.0,
        "avg_daily_turnover": 0.0,
        "trade_day_ratio": 0.0,
        "avg_trade_size_on_trade": 0.0,
        "max_trade_size": 0.0,
        "total_transaction_cost_rate_sum": 0.0,
        "avg_stock_weight": 0.0,
        "min_stock_weight": 0.0,
        "max_stock_weight": 0.0,
        "std_stock_weight": 0.0,
        "avg_bond_weight": 0.0,
        "avg_cash_weight": 0.0,
        "rebalance_ratio": 0.0,
        "emergency_rebalance_ratio": 0.0,
        "regime_switch_ratio": 0.0,
        "normal_pct": 0.0,
        "watch_pct": 0.0,
        "high_vol_pct": 0.0,
        "risk_off_pct": 0.0,
        "extreme_risk_pct": 0.0,
        "actual_high_vol_rate": 0.0,
        "actual_down_high_vol_rate": 0.0,
        "avg_prob_high_vol": 0.0,
        "avg_prob_down_risk": 0.0,
    }
    if df.empty:
        return empty

    out = perf_stats(df["strategy_return_net"], cfg.initial_capital)
    gross = perf_stats(df["strategy_return_gross"], cfg.initial_capital) if "strategy_return_gross" in df.columns else {}
    out["gross_cagr"] = float(gross.get("cagr", out.get("cagr", 0.0)))
    out["cost_cagr_drag"] = float(out["gross_cagr"] - out.get("cagr", 0.0))

    turnover = df["turnover"].astype(float) if "turnover" in df.columns else pd.Series(index=df.index, data=0.0)
    trade_mask = turnover > 1e-12
    out["annual_turnover"] = float(turnover.mean() * 252.0)
    out["total_turnover"] = float(turnover.sum())
    out["avg_daily_turnover"] = float(turnover.mean())
    out["trade_day_ratio"] = float(trade_mask.mean())
    out["avg_trade_size_on_trade"] = float(turnover[trade_mask].mean()) if trade_mask.any() else 0.0
    out["max_trade_size"] = float(turnover.max()) if len(turnover) else 0.0
    out["total_transaction_cost_rate_sum"] = float(df.get("transaction_cost", pd.Series(index=df.index, data=0.0)).astype(float).sum())

    for col, key in [("stock_weight", "stock"), ("bond_weight", "bond"), ("cash_weight", "cash")]:
        if col in df.columns:
            s = df[col].astype(float)
            out[f"avg_{key}_weight"] = float(s.mean())
            out[f"min_{key}_weight"] = float(s.min())
            out[f"max_{key}_weight"] = float(s.max())
            out[f"std_{key}_weight"] = float(s.std(ddof=0))

    out["rebalance_ratio"] = float(df["rebalanced"].mean()) if "rebalanced" in df.columns else 0.0
    out["emergency_rebalance_ratio"] = float(df["emergency_rebalance"].mean()) if "emergency_rebalance" in df.columns else 0.0
    out["regime_switch_ratio"] = float(df["allocation_regime"].ne(df["allocation_regime"].shift(1)).mean()) if "allocation_regime" in df.columns else 0.0

    if "allocation_regime" in df.columns:
        dist = df["allocation_regime"].value_counts(normalize=True)
        out["normal_pct"] = float(dist.get("NORMAL", 0.0))
        out["watch_pct"] = float(dist.get("WATCH", 0.0))
        out["high_vol_pct"] = float(dist.get("HIGH_VOL", 0.0))
        out["risk_off_pct"] = float(dist.get("RISK_OFF", 0.0))
        out["extreme_risk_pct"] = float(dist.get("EXTREME_RISK", 0.0))

    if "actual_risk" in df.columns:
        out["actual_high_vol_rate"] = float((df["actual_risk"] == "고변동").mean())
    if "actual_split_vol" in df.columns:
        out["actual_down_high_vol_rate"] = float((df["actual_split_vol"] == "하락고변동").mean())
    if "prob_high_vol" in df.columns:
        out["avg_prob_high_vol"] = float(df["prob_high_vol"].astype(float).mean())
    if "prob_down_risk" in df.columns:
        out["avg_prob_down_risk"] = float(df["prob_down_risk"].astype(float).mean())

    return {**empty, **out}


def objective_condition_score(stats: Dict[str, float], score_profile: str = "balanced") -> float:
    cagr = float(stats.get("cagr", 0.0))
    mdd = abs(float(stats.get("mdd", 0.0)))
    sharpe = float(stats.get("sharpe", 0.0))
    calmar = float(stats.get("calmar", 0.0))
    annual_turnover = float(stats.get("annual_turnover", 0.0))

    if score_profile == "cagr":
        return 1.35 * cagr + 0.04 * sharpe - 0.50 * mdd - 0.040 * annual_turnover
    if score_profile == "calmar":
        return 0.70 * cagr + 0.10 * sharpe + 0.35 * calmar - 0.90 * mdd - 0.060 * annual_turnover
    if score_profile == "turnover":
        return 0.80 * cagr + 0.08 * sharpe + 0.12 * calmar - 0.75 * mdd - 0.120 * annual_turnover

    # v8.6.2 balanced: turnover 과다 문제를 줄이기 위해 v8.6.1보다 페널티를 강화한다.
    return 1.00 * cagr + 0.08 * sharpe + 0.14 * calmar - 0.78 * mdd - 0.085 * annual_turnover


def _risk_off_bond_cash_from_stock(stock: float) -> Tuple[float, float]:
    # RISK_OFF에서 방어자산을 IEF:BIL = 2:1 안팎으로 배분
    remain = max(0.0, 1.0 - stock)
    bond = remain * 0.65
    cash = remain * 0.35
    return float(bond), float(cash)


def make_condition_candidate_configs(base_cfg: Config, grid_size: str = "standard") -> List[Tuple[str, Config]]:
    """
    조건을 감으로 하나만 선택하지 않고, 사전에 정의한 후보군을 validation 구간에서 비교한다.
    모델 예측값은 그대로 두고 allocation 조건만 비교하므로 전체 재학습보다 훨씬 빠르다.
    """
    if grid_size == "compact":
        no_trade_list = [0.09, 0.11]
        cont_list = [False]
        riskoff_stock_list = [0.52, 0.58]
        threshold_pairs = [(0.62, 0.52), (0.60, 0.50)]
    elif grid_size == "wide":
        no_trade_list = [0.07, 0.09, 0.11, 0.13]
        cont_list = [False, True]
        riskoff_stock_list = [0.48, 0.52, 0.56, 0.58]
        threshold_pairs = [(0.65, 0.55), (0.62, 0.52), (0.60, 0.50), (0.58, 0.48)]
    else:
        # v8.4 standard: v8.3에서 더 좋았던 c032 계열을 중심으로 탐색
        no_trade_list = [0.09, 0.11, 0.13, 0.15, 0.17]
        cont_list = [False]
        riskoff_stock_list = [0.52, 0.56, 0.58]
        threshold_pairs = [(0.62, 0.52), (0.60, 0.50), (0.58, 0.48)]

    out: List[Tuple[str, Config]] = []
    idx = 0
    for ntb in no_trade_list:
        for cont in cont_list:
            for ro_stock in riskoff_stock_list:
                ro_bond, ro_cash = _risk_off_bond_cash_from_stock(ro_stock)
                for hv_th, dn_th in threshold_pairs:
                    c = replace(base_cfg)
                    c.no_trade_band = float(ntb)
                    c.use_continuous_adjustment = bool(cont)
                    c.risk_off_stock_weight = float(ro_stock)
                    c.risk_off_bond_weight = float(ro_bond)
                    c.risk_off_cash_weight = float(ro_cash)
                    c.gate_high_vol_threshold = float(hv_th)
                    c.gate_riskoff_downrisk_threshold = float(dn_th)
                    c.gate_watch_downrisk_threshold = max(0.60, float(dn_th) + 0.12)
                    c.use_three_regime_allocation = True
                    c.use_extreme_risk_cut = True
                    c.extreme_high_vol_threshold = 0.75
                    c.extreme_downrisk_threshold = 0.65
                    c.extreme_stock_weight = 0.45
                    c.extreme_bond_weight = 0.35
                    c.extreme_cash_weight = 0.20
                    c.result_dir = base_cfg.result_dir
                    name = (
                        f"c{idx:03d}_ntb{ntb:.2f}_cont{int(cont)}_"
                        f"ro{ro_stock:.2f}_hv{hv_th:.2f}_dn{dn_th:.2f}"
                    )
                    out.append((name, c))
                    idx += 1
    return out


def run_condition_search(
    pred_raw: pd.DataFrame,
    base_cfg: Config,
    split_date: str,
    grid_size: str = "standard",
    score_profile: str = "balanced",
) -> Tuple[Config, pd.DataFrame, Dict[str, object]]:
    """
    objective condition search.

    선택 구간: Date <= split_date
    보류/검증 구간: Date > split_date

    주의:
    - 이 함수는 모델을 다시 학습하지 않는다.
    - walk-forward 예측 확률은 이미 각 시점의 과거 데이터만 사용해 만들어진 값이다.
    - 조건 선택은 split_date 이전 구간에서만 수행하고, split_date 이후 성과는 선택 후 확인용으로만 남긴다.
    """
    candidates = make_condition_candidate_configs(base_cfg, grid_size=grid_size)
    rows: List[Dict[str, object]] = []
    best_score = -np.inf
    best_cfg = base_cfg
    best_name = "base"

    for name, cand_cfg in candidates:
        allocated, _usage = apply_allocation(pred_raw, cand_cfg)
        select_df = _slice_by_date(allocated, end_date=split_date)
        holdout_df = _slice_by_date(allocated, start_date=split_date)
        # start_date는 split_date 포함이므로 중복을 피하기 위해 하루 단위 조건 재조정
        holdout_df = holdout_df[pd.to_datetime(holdout_df["Date"]) > pd.Timestamp(split_date)].copy()
        full_stats = allocated_subset_stats(allocated, cand_cfg)
        select_stats = allocated_subset_stats(select_df, cand_cfg)
        holdout_stats = allocated_subset_stats(holdout_df, cand_cfg)
        score = objective_condition_score(select_stats, score_profile=score_profile)

        row: Dict[str, object] = {
            "candidate": name,
            "select_score": score,
            "score_profile": score_profile,
            "split_date": split_date,
            "no_trade_band": cand_cfg.no_trade_band,
            "use_continuous_adjustment": cand_cfg.use_continuous_adjustment,
            "risk_off_stock_weight": cand_cfg.risk_off_stock_weight,
            "risk_off_bond_weight": cand_cfg.risk_off_bond_weight,
            "risk_off_cash_weight": cand_cfg.risk_off_cash_weight,
            "gate_high_vol_threshold": cand_cfg.gate_high_vol_threshold,
            "gate_riskoff_downrisk_threshold": cand_cfg.gate_riskoff_downrisk_threshold,
            "gate_watch_downrisk_threshold": cand_cfg.gate_watch_downrisk_threshold,
        }
        for prefix, stats in [("select", select_stats), ("holdout", holdout_stats), ("full", full_stats)]:
            for key in [
                "cagr", "mdd", "sharpe", "sortino", "calmar", "gross_cagr", "cost_cagr_drag",
                "annual_turnover", "total_turnover", "avg_daily_turnover", "trade_day_ratio",
                "avg_trade_size_on_trade", "max_trade_size", "total_transaction_cost_rate_sum",
                "avg_stock_weight", "min_stock_weight", "max_stock_weight", "std_stock_weight",
                "avg_bond_weight", "avg_cash_weight", "rebalance_ratio", "emergency_rebalance_ratio",
                "regime_switch_ratio", "normal_pct", "watch_pct", "high_vol_pct", "risk_off_pct", "extreme_risk_pct",
                "actual_high_vol_rate", "actual_down_high_vol_rate", "avg_prob_high_vol", "avg_prob_down_risk",
            ]:
                row[f"{prefix}_{key}"] = stats.get(key, 0.0)
        rows.append(row)

        if score > best_score:
            best_score = score
            best_cfg = cand_cfg
            best_name = name

    report_df = pd.DataFrame(rows).sort_values("select_score", ascending=False).reset_index(drop=True)

    # v8.4 stable-top 선택 로직:
    # 1) select_score 상위 후보군을 만든다.
    # 2) 그 안에서 select 기준 turnover/MDD/Calmar가 더 안정적인 후보를 고른다.
    # 3) 전체/holdout 성과는 선택 근거가 아니라 사후 진단으로만 저장한다.
    top_n = max(3, int(math.ceil(len(report_df) * 0.08)))
    pool = report_df.head(top_n).copy()
    # 너무 공격적인 후보를 줄이기 위한 soft constraint. 통과 후보가 없으면 top pool 전체 사용.
    constrained = pool[
        (pool["select_annual_turnover"] <= 2.00) &
        (pool["select_mdd"] >= -0.32) &
        (pool["select_calmar"] >= 0.45)
    ].copy()
    if constrained.empty:
        constrained = pool
    constrained = constrained.sort_values(
        ["select_calmar", "select_annual_turnover", "select_mdd", "select_cagr"],
        ascending=[False, True, False, False],
    )
    selected_name = str(constrained.iloc[0]["candidate"])
    selected_score = float(constrained.iloc[0]["select_score"])
    cfg_map = {name: cfg for name, cfg in candidates}
    selected_cfg = cfg_map[selected_name]

    meta = {
        "selected_candidate": selected_name,
        "selected_score": selected_score,
        "selection_method": "stable_top_select_score_then_select_calmar_turnover_mdd",
        "top_pool_size": int(top_n),
        "constrained_pool_size": int(len(constrained)),
        "raw_best_score_candidate": best_name,
        "raw_best_score": float(best_score),
        "split_date": split_date,
        "grid_size": grid_size,
        "score_profile": score_profile,
        "candidate_count": int(len(candidates)),
    }
    return selected_cfg, report_df, meta

# ============================================================
# 8. OPTIMIZATION DIAGNOSTICS
# ============================================================

def _annualized_return(x: pd.Series) -> float:
    r = x.dropna().astype(float)
    if len(r) == 0:
        return 0.0
    return float((1.0 + r).prod() ** (252.0 / len(r)) - 1.0)


def _annualized_vol(x: pd.Series) -> float:
    r = x.dropna().astype(float)
    return float(r.std() * math.sqrt(252)) if len(r) > 1 else 0.0


def _win_rate(x: pd.Series) -> float:
    r = x.dropna().astype(float)
    return float((r > 0).mean()) if len(r) else 0.0


def build_regime_analysis(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if "allocation_regime" not in pred_df.columns:
        return pd.DataFrame(rows)
    total_n = len(pred_df)
    for regime, g in pred_df.groupby("allocation_regime", dropna=False):
        rr = g["strategy_return_net"].astype(float)
        rows.append({
            "allocation_regime": str(regime),
            "count": int(len(g)),
            "pct": float(len(g) / total_n) if total_n else 0.0,
            "ann_return_est": _annualized_return(rr),
            "ann_vol_est": _annualized_vol(rr),
            "mean_daily_return": float(rr.mean()),
            "win_rate": _win_rate(rr),
            "avg_stock_weight": float(g["stock_weight"].mean()),
            "avg_bond_weight": float(g["bond_weight"].mean()),
            "avg_cash_weight": float(g["cash_weight"].mean()),
            "avg_turnover": float(g["turnover"].mean()),
            "annual_turnover_est": float(g["turnover"].mean() * 252.0),
            "rebalance_ratio": float(g["rebalanced"].mean()),
            "emergency_rebalance_ratio": float(g["emergency_rebalance"].mean()),
            "avg_prob_high_vol": float(g["prob_high_vol"].mean()),
            "avg_prob_down_risk": float(g["prob_down_risk"].mean()),
            "actual_high_vol_rate": float((g["actual_risk"] == "고변동").mean()) if "actual_risk" in g.columns else 0.0,
            "actual_down_high_vol_rate": float((g["actual_split_vol"] == "하락고변동").mean()) if "actual_split_vol" in g.columns else 0.0,
        })
    return pd.DataFrame(rows).sort_values("pct", ascending=False).reset_index(drop=True)


def build_regime_transition_matrix(pred_df: pd.DataFrame) -> pd.DataFrame:
    if "allocation_regime" not in pred_df.columns or len(pred_df) < 2:
        return pd.DataFrame()
    s = pred_df["allocation_regime"].astype(str)
    mat = pd.crosstab(s.shift(1), s, normalize="index").fillna(0.0)
    mat.index.name = "from_regime"
    mat.columns.name = "to_regime"
    return mat.reset_index()


def _bin_series(s: pd.Series, bins: int = 10) -> pd.Series:
    vals = s.astype(float).clip(0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    return pd.cut(vals, bins=edges, include_lowest=True, right=True)


def build_probability_bin_analysis(pred_df: pd.DataFrame, prob_col: str, actual_col: str, positive_value: str, bins: int = 10) -> pd.DataFrame:
    if prob_col not in pred_df.columns or actual_col not in pred_df.columns:
        return pd.DataFrame()
    tmp = pred_df.copy()
    tmp["prob_bin"] = _bin_series(tmp[prob_col], bins=bins)
    tmp["actual_positive"] = (tmp[actual_col] == positive_value).astype(int)
    rows: List[Dict[str, object]] = []
    for b, g in tmp.groupby("prob_bin", observed=False):
        if len(g) == 0:
            continue
        rows.append({
            "prob_col": prob_col,
            "actual_col": actual_col,
            "positive_value": positive_value,
            "prob_bin": str(b),
            "count": int(len(g)),
            "actual_rate": float(g["actual_positive"].mean()),
            "avg_prob": float(g[prob_col].astype(float).mean()),
            "avg_strategy_return_net": float(g["strategy_return_net"].astype(float).mean()),
            "ann_return_est": _annualized_return(g["strategy_return_net"]),
            "avg_stock_weight": float(g["stock_weight"].mean()),
            "avg_turnover": float(g["turnover"].mean()),
        })
    return pd.DataFrame(rows)


def build_threshold_diagnostics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    diagnostics = [
        ("stage1_ensemble", "actual_risk", "고변동", "prob_high_vol"),
        ("overall_risk_highvol", "actual_risk", "고변동", "prob_overall_risk"),
        ("overall_risk_down", "actual_direction", "하락", "prob_overall_risk"),
        ("up_ensemble", "actual_direction", "상승", "prob_up"),
        ("down_ensemble", "actual_direction", "하락", "prob_down"),
        ("down_price_trend", "actual_direction", "하락", "prob_down_price_trend"),
        ("down_price_volume", "actual_direction", "하락", "prob_down_price_volume"),
        ("down_volatility", "actual_direction", "하락", "prob_down_volatility"),
    ]
    for h in [10, 20]:
        if f"prob_high_vol_h{h}" in pred_df.columns:
            diagnostics.append((f"stage1_h{h}", f"actual_risk_h{h}", "고변동", f"prob_high_vol_h{h}"))
        if f"prob_down_h{h}" in pred_df.columns:
            diagnostics.append((f"down_h{h}", f"actual_direction_h{h}", "하락", f"prob_down_h{h}"))
        if f"prob_up_h{h}" in pred_df.columns:
            diagnostics.append((f"up_h{h}", f"actual_direction_h{h}", "상승", f"prob_up_h{h}"))
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    for name, actual_col, positive_value, prob_col in diagnostics:
        if actual_col not in pred_df.columns or prob_col not in pred_df.columns:
            continue
        y_true = (pred_df[actual_col] == positive_value).astype(int).values
        prob = pred_df[prob_col].astype(float).clip(0.0, 1.0).values
        support_pos = int(y_true.sum())
        support_neg = int(len(y_true) - support_pos)
        for th in thresholds:
            y_pred = (prob >= float(th)).astype(int)
            rows.append({
                "model": name,
                "prob_col": prob_col,
                "actual_col": actual_col,
                "positive_value": positive_value,
                "threshold": float(th),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "pred_positive_ratio": float(y_pred.mean()),
                "support_positive": support_pos,
                "support_negative": support_neg,
            })
    return pd.DataFrame(rows)


def build_turnover_diagnostics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    df = pred_df.copy()
    df["trade_occurred"] = df["turnover"].astype(float) > 1e-12
    df["rebalance_type"] = np.where(df["emergency_rebalance"].astype(bool), "emergency", np.where(df["rebalanced"].astype(bool), "scheduled_or_initial", "no_rebalance"))
    for section, group_cols in [
        ("by_rebalance_type", ["rebalance_type"]),
        ("by_regime", ["allocation_regime"]),
        ("by_regime_rebalance_type", ["allocation_regime", "rebalance_type"]),
    ]:
        grouped = df.groupby(group_cols, dropna=False)
        for group_key, g in grouped:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row: Dict[str, object] = {"section": section, "count": int(len(g)), "pct": float(len(g) / len(df)) if len(df) else 0.0}
            for col, val in zip(group_cols, group_key):
                row[col] = str(val)
            row.update({
                "trade_day_ratio": float(g["trade_occurred"].mean()),
                "avg_turnover": float(g["turnover"].mean()),
                "annual_turnover_est": float(g["turnover"].mean() * 252.0),
                "total_turnover": float(g["turnover"].sum()),
                "avg_trade_size_on_trade": float(g.loc[g["trade_occurred"], "turnover"].mean()) if g["trade_occurred"].any() else 0.0,
                "max_turnover": float(g["turnover"].max()),
                "cost_sum": float(g["transaction_cost"].sum()),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def build_drawdown_episodes(pred_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if "strategy_equity_net" not in pred_df.columns or pred_df.empty:
        return pd.DataFrame()
    df = pred_df[["Date", "strategy_equity_net", "allocation_regime", "stock_weight", "prob_high_vol", "prob_down_risk"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    equity = df["strategy_equity_net"].astype(float)
    peak = equity.cummax()
    dd = equity / peak - 1.0
    df["drawdown"] = dd
    episodes: List[Dict[str, object]] = []
    in_dd = False
    start_idx = 0
    trough_idx = 0
    min_dd = 0.0
    for i, val in enumerate(dd.values):
        if not in_dd and val < 0:
            in_dd = True
            start_idx = max(0, i - 1)
            trough_idx = i
            min_dd = float(val)
        elif in_dd:
            if val < min_dd:
                min_dd = float(val)
                trough_idx = i
            if val >= -1e-12:
                end_idx = i
                seg = df.iloc[start_idx:end_idx + 1]
                episodes.append({
                    "start_date": str(df.iloc[start_idx]["Date"].date()),
                    "trough_date": str(df.iloc[trough_idx]["Date"].date()),
                    "recovery_date": str(df.iloc[end_idx]["Date"].date()),
                    "depth": min_dd,
                    "duration_days": int(end_idx - start_idx),
                    "days_to_trough": int(trough_idx - start_idx),
                    "avg_stock_weight": float(seg["stock_weight"].mean()),
                    "avg_prob_high_vol": float(seg["prob_high_vol"].mean()),
                    "avg_prob_down_risk": float(seg["prob_down_risk"].mean()),
                    "trough_regime": str(df.iloc[trough_idx]["allocation_regime"]),
                })
                in_dd = False
    if in_dd:
        end_idx = len(df) - 1
        seg = df.iloc[start_idx:end_idx + 1]
        episodes.append({
            "start_date": str(df.iloc[start_idx]["Date"].date()),
            "trough_date": str(df.iloc[trough_idx]["Date"].date()),
            "recovery_date": "not_recovered",
            "depth": min_dd,
            "duration_days": int(end_idx - start_idx),
            "days_to_trough": int(trough_idx - start_idx),
            "avg_stock_weight": float(seg["stock_weight"].mean()),
            "avg_prob_high_vol": float(seg["prob_high_vol"].mean()),
            "avg_prob_down_risk": float(seg["prob_down_risk"].mean()),
            "trough_regime": str(df.iloc[trough_idx]["allocation_regime"]),
        })
    return pd.DataFrame(episodes).sort_values("depth").head(top_n).reset_index(drop=True)


def build_periodic_returns(pred_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    tmp = pred_df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    tmp = tmp.set_index("Date")
    cols = {
        "strategy_return_net": "strategy_net",
        "strategy_return_gross": "strategy_gross",
        "stock_next_return": "stock_buy_hold",
        "bond_next_return": "bond",
        "cash_next_return": "cash",
    }
    rows: List[Dict[str, object]] = []
    for period, g in tmp.resample(freq):
        if g.empty:
            continue
        row: Dict[str, object] = {"period": str(period.date())}
        for col, name in cols.items():
            if col in g.columns:
                row[name] = float((1.0 + g[col].astype(float)).prod() - 1.0)
        row["avg_stock_weight"] = float(g["stock_weight"].mean())
        row["turnover_sum"] = float(g["turnover"].sum())
        period_equity = (1.0 + g["strategy_return_net"].astype(float)).cumprod()
        row["max_drawdown_inside_period"] = float((period_equity / period_equity.cummax() - 1.0).min()) if len(period_equity) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_optimization_metrics(summary: Dict[str, object]) -> pd.DataFrame:
    s1 = summary.get("stage1_feature_importance_mean", {}) or {}
    dn = summary.get("downrisk_feature_importance_mean", {}) or {}
    features = sorted(set(s1.keys()) | set(dn.keys()))
    rows: List[Dict[str, object]] = []
    for f in features:
        s1_imp = float(s1.get(f, 0.0))
        dn_imp = float(dn.get(f, 0.0))
        rows.append({
            "feature": f,
            "stage1_importance": s1_imp,
            "downrisk_importance": dn_imp,
            "mean_importance": (s1_imp + dn_imp) / 2.0,
            "max_importance": max(s1_imp, dn_imp),
            "importance_gap_stage1_minus_downrisk": s1_imp - dn_imp,
            "is_low_importance_candidate": bool(max(s1_imp, dn_imp) < 0.001),
            "used_more_by": "stage1" if s1_imp > dn_imp else ("downrisk" if dn_imp > s1_imp else "tie"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["stage1_rank"] = df["stage1_importance"].rank(ascending=False, method="min").astype(int)
    df["downrisk_rank"] = df["downrisk_importance"].rank(ascending=False, method="min").astype(int)
    df["mean_rank"] = df["mean_importance"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["mean_importance", "max_importance"], ascending=False).reset_index(drop=True)


def build_optimization_diagnostics(pred_df: pd.DataFrame, summary: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    prob_bins = []
    hv_bins = build_probability_bin_analysis(pred_df, "prob_high_vol", "actual_risk", "고변동")
    if not hv_bins.empty:
        prob_bins.append(hv_bins)
    overall_bins = build_probability_bin_analysis(pred_df, "prob_overall_risk", "actual_risk", "고변동")
    if not overall_bins.empty:
        prob_bins.append(overall_bins)
    dn_bins = build_probability_bin_analysis(pred_df, "prob_down", "actual_direction", "하락") if "prob_down" in pred_df.columns else build_probability_bin_analysis(pred_df, "prob_down_risk", "actual_split_vol", "하락고변동")
    if not dn_bins.empty:
        prob_bins.append(dn_bins)
    up_bins = build_probability_bin_analysis(pred_df, "prob_up", "actual_direction", "상승") if "prob_up" in pred_df.columns else pd.DataFrame()
    if not up_bins.empty:
        prob_bins.append(up_bins)
    return {
        "regime_analysis": build_regime_analysis(pred_df),
        "regime_transition_matrix": build_regime_transition_matrix(pred_df),
        "probability_bins": pd.concat(prob_bins, ignore_index=True) if prob_bins else pd.DataFrame(),
        "threshold_diagnostics": build_threshold_diagnostics(pred_df),
        "turnover_diagnostics": build_turnover_diagnostics(pred_df),
        "drawdown_episodes": build_drawdown_episodes(pred_df),
        "monthly_returns": build_periodic_returns(pred_df, "ME"),
        "annual_returns": build_periodic_returns(pred_df, "YE"),
        "feature_optimization_metrics": build_feature_optimization_metrics(summary),
    }


def diagnostics_summary(diagnostics: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    reg = diagnostics.get("regime_analysis", pd.DataFrame())
    if not reg.empty:
        out["regime_count"] = int(len(reg))
        out["best_regime_by_ann_return"] = str(reg.sort_values("ann_return_est", ascending=False).iloc[0]["allocation_regime"])
        out["highest_turnover_regime"] = str(reg.sort_values("annual_turnover_est", ascending=False).iloc[0]["allocation_regime"])
    dd = diagnostics.get("drawdown_episodes", pd.DataFrame())
    if not dd.empty:
        out["worst_drawdown_episode"] = dd.iloc[0].to_dict()
    feat = diagnostics.get("feature_optimization_metrics", pd.DataFrame())
    if not feat.empty:
        out["low_importance_feature_count"] = int(feat["is_low_importance_candidate"].sum())
        out["top10_mean_importance_features"] = feat.head(10)["feature"].tolist()
    return out

# ============================================================
# 8. CLI / MAIN
# ============================================================

def apply_speed_profile(cfg: Config, profile: str) -> Config:
    if profile == "fast":
        cfg.retrain_every_n_days = 20
        cfg.stage1_n_estimators = 100
        cfg.down_n_estimators = 70
        cfg.use_adaptive_label_policy = False
        cfg.use_rolling_gate_optimization = False
        cfg.result_dir = "results_xgb_v8_6_2_fast"
    elif profile == "balanced":
        cfg.retrain_every_n_days = 10
        cfg.stage1_n_estimators = 150
        cfg.down_n_estimators = 100
        cfg.use_adaptive_label_policy = False
        cfg.use_rolling_gate_optimization = False
        cfg.result_dir = "results_xgb_v8_6_2_balanced"
    elif profile == "full":
        cfg.retrain_every_n_days = 10
        cfg.stage1_n_estimators = 200
        cfg.down_n_estimators = 140
        cfg.use_adaptive_label_policy = True
        cfg.use_rolling_gate_optimization = False
        cfg.result_dir = "results_xgb_v8_6_2_full_adaptive_label"
    else:
        raise ValueError(f"알 수 없는 speed profile: {profile}")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost v8.6.2 Directional Up/Down + Objective Condition Search")
    parser.add_argument("--speed-profile", choices=["fast", "balanced", "full"], default="balanced")
    parser.add_argument("--adaptive-label", action="store_true", help="라벨 quantile 정책을 nested validation으로 선택")
    parser.add_argument("--rolling-gate-opt", action="store_true", help="작은 grid로 allocation gate threshold를 rolling 최적화")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--retrain-every", type=int, default=None)
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--h10-down-only", action="store_true", help="Down-risk ensemble에서 H20을 제거하고 H10만 사용")
    parser.add_argument("--no-trade-band", type=float, default=None, help="거래 무시 band 직접 지정. 예: 0.07")
    parser.add_argument("--condition-search", action="store_true", help="조건을 validation 구간에서 객관식 후보 비교 후 선택")
    parser.add_argument("--condition-split-date", type=str, default="2021-12-31", help="조건 선택 구간의 마지막 날짜. 이후 구간은 holdout 확인용")
    parser.add_argument("--condition-grid-size", choices=["compact", "standard", "wide"], default="standard")
    parser.add_argument("--score-profile", choices=["balanced", "cagr", "calmar", "turnover"], default="balanced")
    parser.add_argument("--four-regime", action="store_true", help="v8.3 방식의 NORMAL/WATCH/HIGH_VOL/RISK_OFF 4-regime 구조 사용")
    parser.add_argument("--no-extreme-risk", action="store_true", help="EXTREME_RISK 추가 방어 규칙 비활성화")
    parser.add_argument("--no-diagnostics", action="store_true", help="추가 최적화 진단 CSV 저장을 생략")
    parser.add_argument("--execution-lag-days", type=int, default=None, help="체결 지연 일수. 0=기존 방식, 1=보수적 다음 거래일 체결 가정")
    parser.add_argument("--max-train-rows", type=int, default=None, help="최근 N개 학습 샘플만 사용. 미지정 시 전체 expanding window")
    parser.add_argument("--allow-cash-download-fallback", action="store_true", help="BIL 다운로드 실패 시 cash return을 0으로 대체")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_speed_profile(Config(), args.speed_profile)
    if args.adaptive_label:
        cfg.use_adaptive_label_policy = True
    if args.rolling_gate_opt:
        cfg.use_rolling_gate_optimization = True
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs
    if args.retrain_every is not None:
        cfg.retrain_every_n_days = args.retrain_every
    if args.result_dir:
        cfg.result_dir = args.result_dir
    if getattr(args, "h10_down_only", False):
        cfg.down_risk_weight_h10 = 1.0
        cfg.down_risk_weight_h20 = 0.0
    if getattr(args, "no_trade_band", None) is not None:
        cfg.no_trade_band = float(args.no_trade_band)
    if getattr(args, "four_regime", False):
        cfg.use_three_regime_allocation = False
    if getattr(args, "no_extreme_risk", False):
        cfg.use_extreme_risk_cut = False
    if getattr(args, "execution_lag_days", None) is not None:
        cfg.execution_lag_days = int(args.execution_lag_days)
    if getattr(args, "max_train_rows", None) is not None:
        cfg.max_train_rows = int(args.max_train_rows)
    if getattr(args, "allow_cash_download_fallback", False):
        cfg.allow_cash_download_fallback = True

    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 데이터 다운로드")
    target = download_ohlcv(cfg.target_ticker, cfg.start_date, cfg.end_date)
    bond_close = download_close(cfg.bond_ticker, cfg.start_date, cfg.end_date)
    try:
        cash_close = download_close(cfg.cash_ticker, cfg.start_date, cfg.end_date)
    except Exception as exc:
        if not cfg.allow_cash_download_fallback:
            raise RuntimeError(
                f"{cfg.cash_ticker} 다운로드 실패. "
                f"현금 수익률을 0으로 대체하면 백테스트가 왜곡될 수 있습니다."
            ) from exc
        warnings.warn(
            f"{cfg.cash_ticker} 다운로드 실패로 cash return을 0으로 대체합니다: {exc}",
            RuntimeWarning,
        )
        cash_close = pd.Series(index=target.index, data=np.nan, name=cfg.cash_ticker)

    print("[2/5] 피처 생성")
    df, feature_cols = build_features(target, cfg.horizons)
    returns_df = build_aligned_forward_returns(
        target_close=df["Close"],
        bond_close=bond_close,
        cash_close=cash_close,
        target_index=df.index,
        execution_lag_days=cfg.execution_lag_days,
    )
    df = pd.concat(
        [
            df,
            returns_df[["stock_next_return", "bond_next_return", "cash_next_return"]],
        ],
        axis=1,
    ).copy()
    print(f"    피처 수: {len(feature_cols)}")
    print(f"    horizons: {cfg.horizons}")
    print(f"    adaptive_label: {cfg.use_adaptive_label_policy}")
    print(f"    rolling_gate_opt: {cfg.use_rolling_gate_optimization}")
    print(f"    execution_lag_days: {cfg.execution_lag_days}")
    print(f"    max_train_rows: {cfg.max_train_rows}")

    print("[3/5] Walk-forward H10/H20 Stage1 + Up/Down 방향성 예측")
    pred_raw = run_walk_forward(df, feature_cols, cfg)

    print("[4/5] 배분/백테스트")
    condition_report_df: Optional[pd.DataFrame] = None
    condition_meta: Dict[str, object] = {}

    if getattr(args, "condition_search", False):
        print("    조건 객관화 탐색 실행")
        print(f"    split_date: {args.condition_split_date}")
        print(f"    grid_size: {args.condition_grid_size}")
        print(f"    score_profile: {args.score_profile}")
        selected_cfg, condition_report_df, condition_meta = run_condition_search(
            pred_raw=pred_raw,
            base_cfg=cfg,
            split_date=args.condition_split_date,
            grid_size=args.condition_grid_size,
            score_profile=args.score_profile,
        )
        # 선택된 조건만 최종 전체 구간에 적용한다. 모델 예측은 pred_raw를 재사용한다.
        selected_cfg.result_dir = cfg.result_dir
        cfg = selected_cfg
        print(f"    selected_candidate: {condition_meta.get('selected_candidate')}")

    pred_df, gate_usage = apply_allocation(pred_raw, cfg)
    pred_df.attrs.update(pred_raw.attrs)

    print("[5/5] 결과 저장")
    summary = build_summary(pred_df, feature_cols, gate_usage, cfg)
    if condition_meta:
        summary["condition_search"] = condition_meta
        add_condition_period_summary(
            summary=summary,
            pred_df=pred_df,
            cfg=cfg,
            split_date=str(condition_meta["split_date"]),
        )

    pred_path = result_dir / "qqq_xgb_v8_6_2_predictions.csv"
    summary_path = result_dir / "qqq_xgb_v8_6_2_summary.json"
    latest_path = result_dir / "qqq_xgb_v8_6_2_latest.json"
    importance_stage1_path = result_dir / "qqq_xgb_v8_6_2_stage1_feature_importance.csv"
    importance_up_path = result_dir / "qqq_xgb_v8_6_2_up_feature_importance.csv"
    importance_down_path = result_dir / "qqq_xgb_v8_6_2_downrisk_feature_importance.csv"
    importance_down_price_trend_path = result_dir / "qqq_xgb_v8_6_2_downrisk_price_trend_feature_importance.csv"
    importance_down_price_volume_path = result_dir / "qqq_xgb_v8_6_2_downrisk_price_volume_feature_importance.csv"
    importance_down_volatility_path = result_dir / "qqq_xgb_v8_6_2_downrisk_volatility_feature_importance.csv"
    condition_search_path = result_dir / "qqq_xgb_v8_6_2_condition_search.csv"

    diagnostics: Dict[str, pd.DataFrame] = {}
    if not getattr(args, "no_diagnostics", False):
        diagnostics = build_optimization_diagnostics(pred_df, summary)
        summary["optimization_diagnostics_summary"] = diagnostics_summary(diagnostics)

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary["latest_prediction"], f, ensure_ascii=False, indent=2)

    pd.Series(summary.get("stage1_feature_importance_mean", {}), name="importance").to_csv(importance_stage1_path, encoding="utf-8-sig")
    pd.Series(summary.get("up_feature_importance_mean", {}), name="importance").to_csv(importance_up_path, encoding="utf-8-sig")
    pd.Series(summary.get("downrisk_feature_importance_mean", {}), name="importance").to_csv(importance_down_path, encoding="utf-8-sig")
    pd.Series(summary.get("downrisk_price_trend_feature_importance_mean", {}), name="importance").to_csv(importance_down_price_trend_path, encoding="utf-8-sig")
    pd.Series(summary.get("downrisk_price_volume_feature_importance_mean", {}), name="importance").to_csv(importance_down_price_volume_path, encoding="utf-8-sig")
    pd.Series(summary.get("downrisk_volatility_feature_importance_mean", {}), name="importance").to_csv(importance_down_volatility_path, encoding="utf-8-sig")
    if condition_report_df is not None:
        condition_report_df.to_csv(condition_search_path, index=False, encoding="utf-8-sig")

    diagnostic_paths: List[Path] = []
    for name, diag_df in diagnostics.items():
        if diag_df is not None and not diag_df.empty:
            pth = result_dir / f"qqq_xgb_v8_6_2_{name}.csv"
            diag_df.to_csv(pth, index=False, encoding="utf-8-sig")
            diagnostic_paths.append(pth)

    print_summary(summary)
    print("\n[저장 완료]")
    print(f"- {pred_path}")
    print(f"- {summary_path}")
    print(f"- {latest_path}")
    print(f"- {importance_stage1_path}")
    print(f"- {importance_up_path}")
    print(f"- {importance_down_path}")
    print(f"- {importance_down_price_trend_path}")
    print(f"- {importance_down_price_volume_path}")
    print(f"- {importance_down_volatility_path}")
    if condition_report_df is not None:
        print(f"- {condition_search_path}")
    for pth in diagnostic_paths:
        print(f"- {pth}")


if __name__ == "__main__":
    main()
