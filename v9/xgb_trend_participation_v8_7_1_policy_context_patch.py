"""
XGBoost v8.7.1 - Policy Context Patch Full Code
================================================

목적
- QQQ/IEF/BIL 동적 자산배분 전략
- v8.7.0의 핵심 구조를 단일 실행 파일 형태로 정리
- v8.7.1 핵심 수정:
  1. run_walk_forward() 예측 결과에 policy-context 과거 피처 저장
  2. mid_trend_score / mid_trend_state 고정 문제 방지
  3. trend participation overlay 실제 작동 가능화
  4. recovery re-risk overlay 실제 작동 가능화
  5. policy_context_diagnostics 저장

실행 예시
    python xgb_trend_participation_v8_7_1_policy_context_patch.py --speed-profile balanced --h10-down-only
    python xgb_trend_participation_v8_7_1_policy_context_patch.py --speed-profile fast --target-ticker QQQ
    python xgb_trend_participation_v8_7_1_policy_context_patch.py --asset-list QQQ,SPY,SOXX

필요 패키지
    pip install pandas numpy yfinance scikit-learn xgboost

주의
- 이 파일은 패치 파일이 아니라 단일 실행 전체 코드입니다.
- 기존 v8.7.0의 모든 실험 옵션을 1:1로 전부 보존한 버전은 아니고,
  사용자가 제공한 v8.7.0 구조 중 실제 성능 검증에 필요한 핵심 파이프라인을 v8.7.1 형태로 재구성한 실행본입니다.
- 미래 컬럼은 라벨 생성에만 사용하고 feature input에는 넣지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
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
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


DIRECTION_STRENGTH_LABELS = [
    "NO_STRENGTH_SIGNAL",
    "UP_STRENGTHENING",
    "DOWN_STRENGTHENING",
]
DIRECTION_STRENGTH_LABEL_TO_ID = {name: i for i, name in enumerate(DIRECTION_STRENGTH_LABELS)}
DIRECTION_STRENGTH_ID_TO_LABEL = {i: name for name, i in DIRECTION_STRENGTH_LABEL_TO_ID.items()}


POLICY_CONTEXT_COLUMNS = [
    # trend overlay
    "return_10d",
    "return_20d",
    "return_60d",
    "return_120d",
    "price_ma_20_gap",
    "price_ma_60_gap",
    "price_ma_120_gap",
    "price_ma_200_gap",
    "ma_gap_20_60",
    "ma_gap_60_120",
    "ma_gap_50_200",
    "trend_slope_60",
    "ma200_slope_60",
    "positive_return_ratio_60",
    "trend_consistency_60",
    "realized_vol_60",
    # recovery / guard
    "drawdown_20",
    "drawdown_60",
    "drawdown_120",
    "price_position_60",
    "close_to_60d_high",
    # diagnostics
    "atr_rank_252",
    "ulcer_rank_252",
    "bb_width_rank_252",
]


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
    execution_lag_days: int = 1
    allow_cash_download_fallback: bool = False

    horizons: Tuple[int, ...] = (5, 10, 20)
    primary_horizon: int = 10
    min_train_rows: int = 756
    retrain_every_n_days: int = 10
    max_train_rows: Optional[int] = None

    random_state: int = 42
    n_jobs: int = -1

    stage1_n_estimators: int = 150
    stage1_learning_rate: float = 0.025
    stage1_max_depth: int = 3
    stage1_min_child_weight: float = 10.0
    stage1_subsample: float = 0.85
    stage1_colsample_bytree: float = 0.80
    stage1_reg_lambda: float = 8.0
    stage1_reg_alpha: float = 0.1

    strength_n_estimators: int = 160
    strength_learning_rate: float = 0.025
    strength_max_depth: int = 2
    strength_min_child_weight: float = 8.0
    strength_subsample: float = 0.85
    strength_colsample_bytree: float = 0.80
    strength_reg_lambda: float = 10.0
    strength_reg_alpha: float = 0.2

    down_n_estimators: int = 100
    down_learning_rate: float = 0.030
    down_max_depth: int = 2
    down_min_child_weight: float = 6.0
    down_subsample: float = 0.90
    down_colsample_bytree: float = 0.85
    down_reg_lambda: float = 10.0
    down_reg_alpha: float = 0.2

    use_prob_ewma: bool = True
    prob_ewma_span: int = 7

    pred_high_vol_threshold: float = 0.50
    pred_overall_risk_threshold: float = 0.50

    high_vol_quantile: float = 0.80
    down_quantile: float = 0.20
    up_quantile: float = 0.80
    direction_strength_ret_eps_k: float = 0.20

    high_vol_weight_h10: float = 0.55
    high_vol_weight_h20: float = 0.45

    # allocation base
    rebalance_every_n_days: int = 5
    no_trade_band: float = 0.12
    emergency_high_vol_threshold: float = 0.88
    emergency_combined_high_vol_threshold: float = 0.78
    emergency_combined_down_threshold: float = 0.78
    emergency_cooldown_days: int = 5

    use_vol_probability_base_allocation: bool = True
    vol_base_stock_lt_25: float = 0.86
    vol_base_stock_lt_35: float = 0.82
    vol_base_stock_lt_50: float = 0.74
    vol_base_stock_lt_65: float = 0.60
    vol_base_stock_lt_75: float = 0.52
    vol_base_stock_lt_86: float = 0.42
    vol_base_stock_ge_86: float = 0.30
    vol_base_bond_ratio_of_defensive: float = 0.65

    # strength policy
    up_strength_weight_5d: float = 0.00
    up_strength_weight_10d: float = 0.20
    up_strength_weight_20d: float = 0.80
    down_strength_weight_5d: float = 0.00
    down_strength_weight_10d: float = 0.20
    down_strength_weight_20d: float = 0.80

    up_strength_bonus_threshold_1: float = 0.30
    up_strength_bonus_threshold_2: float = 0.38
    up_strength_bonus_threshold_3: float = 0.45
    up_strength_confirm_10d_threshold_2: float = 0.32
    up_strength_confirm_20d_threshold_2: float = 0.34
    up_strength_confirm_20d_threshold_3: float = 0.38
    up_strength_low_vol_threshold_1: float = 0.82
    up_strength_low_vol_threshold_2: float = 0.72
    up_strength_low_vol_threshold_3: float = 0.68
    up_strength_single_20d_stock_weight: float = 0.80
    up_strength_pair_10d_20d_stock_weight: float = 0.88
    up_strength_all3_base_stock_weight: float = 0.96
    up_strength_full_stock_score_threshold: float = 0.50
    up_strength_full_stock_10d_threshold: float = 0.38
    up_strength_full_stock_20d_threshold: float = 0.42
    up_strength_full_stock_high_vol_threshold: float = 0.58
    up_strength_all3_strong_stock_weight: float = 1.00
    disable_tier2_signal: bool = False
    force_tier3_rebalance: bool = True
    force_full_stock_rebalance: bool = True

    # v8.7.1 policy context patch
    enable_trend_participation_overlay: bool = True
    trend_sharpe60_threshold: float = 0.50
    trend_sharpe120_threshold: float = 0.45
    trend_positive_ratio_60_threshold: float = 0.55
    trend_min_mid_trend_score: int = 4
    trend_max_high_vol_for_overlay: float = 0.65
    trend_floor_lt25: float = 0.98
    trend_floor_lt35: float = 0.96
    trend_floor_lt50: float = 0.92
    trend_floor_lt65: float = 0.86
    trend_full_stock_when_low_vol: bool = False
    trend_full_stock_high_vol_threshold: float = 0.25
    trend_force_rebalance: bool = False

    enable_recovery_rerisk_overlay: bool = True
    recovery_dd60_threshold: float = -0.12
    recovery_return10_threshold: float = 0.06
    recovery_return20_threshold: float = 0.08
    recovery_price_ma20_gap_threshold: float = 0.00
    recovery_down_strength_max: float = 0.45
    recovery_high_vol_max: float = 0.90
    recovery_stock_floor: float = 0.78
    recovery_strong_up_score_threshold: float = 0.30
    recovery_strong_stock_floor: float = 0.90
    recovery_force_rebalance: bool = True

    enable_soft_drawdown_guard: bool = False
    drawdown_guard_window: int = 60
    drawdown_guard_threshold_1: float = -0.08
    drawdown_guard_threshold_2: float = -0.12
    drawdown_guard_high_vol_min: float = 0.40
    drawdown_guard_cut_1: float = 0.10
    drawdown_guard_cut_2: float = 0.20
    drawdown_guard_stock_cap_2: float = 0.70
    drawdown_guard_min_stock: float = 0.30

    enable_stale_offensive_decay: bool = True
    stale_offensive_stock_gap_threshold: float = 0.12
    stale_offensive_up_strength_reset_threshold: float = 0.20
    stale_offensive_high_vol_threshold: float = 0.72

    result_dir: str = "results_xgb_trend_participation_v8_7_1_policy_context_patch"


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
        out["cash_next_return"] = 0.0
    else:
        out["cash_next_return"] = prices["cash"].pct_change().shift(-shift_n)
    return out


def rolling_rank_last(series: pd.Series, window: int) -> pd.Series:
    def _rank(x: np.ndarray) -> float:
        if len(x) == 0 or not np.isfinite(x[-1]):
            return np.nan
        valid = x[np.isfinite(x)]
        if len(valid) == 0:
            return np.nan
        return float(np.mean(valid <= x[-1]))

    return series.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=True)


def calc_trend_slope(close: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)

    def _slope(y: np.ndarray) -> float:
        if len(y) != window or np.isnan(y).any():
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

    df["return_5d_minus_20d"] = df["return_5d"] - df["return_20d"]
    df["return_10d_minus_20d"] = df["return_10d"] - df["return_20d"]

    for w in [5, 10, 20, 50, 60, 120, 200]:
        ma = close.rolling(w).mean()
        df[f"ma_{w}"] = ma
        df[f"price_ma_{w}_gap"] = close / ma - 1.0

    df["ma_gap_5_20"] = df["ma_5"] / df["ma_20"] - 1.0
    df["ma_gap_20_60"] = df["ma_20"] / df["ma_60"] - 1.0
    df["ma_gap_60_120"] = df["ma_60"] / df["ma_120"] - 1.0
    df["ma_gap_50_200"] = df["ma_50"] / df["ma_200"] - 1.0

    for w in [5, 10, 20, 60]:
        df[f"trend_slope_{w}"] = calc_trend_slope(close, w)
    df["ma200_slope_60"] = calc_trend_slope(df["ma_200"], 60)

    up = (df["daily_return"] > 0).astype(float)
    large_down = (df["daily_return"] <= -0.02).astype(float)
    large_up = (df["daily_return"] >= 0.02).astype(float)
    for w in [5, 10, 20, 60]:
        df[f"positive_return_ratio_{w}"] = up.rolling(w).mean()
        df[f"trend_consistency_{w}"] = df[f"positive_return_ratio_{w}"]
    for w in [5, 10, 20]:
        df[f"large_down_day_ratio_{w}"] = large_down.rolling(w).mean()
        df[f"large_up_day_ratio_{w}"] = large_up.rolling(w).mean()

    for w in [5, 10, 20, 60, 120]:
        roll_high = close.rolling(w).max()
        roll_low = close.rolling(w).min()
        denom = (roll_high - roll_low).replace(0, np.nan)
        df[f"drawdown_{w}"] = close / roll_high - 1.0
        if w in [5, 10, 20, 60]:
            df[f"price_position_{w}"] = (close - roll_low) / denom
            df[f"close_to_{w}d_high"] = close / roll_high - 1.0
    df["price_position_5_minus_20"] = df["price_position_5"] - df["price_position_20"]
    df["price_position_10_minus_20"] = df["price_position_10"] - df["price_position_20"]

    for w in [5, 10, 20]:
        vma = volume.rolling(w).mean()
        vstd = volume.rolling(w).std()
        df[f"volume_ratio_{w}"] = volume / vma
        df[f"volume_zscore_{w}"] = (volume - vma) / vstd.replace(0, np.nan)

    down_volume = ((df["daily_return"] < 0).astype(float) * volume)
    for w in [10, 20]:
        df[f"down_volume_ratio_{w}"] = down_volume.rolling(w).sum() / volume.rolling(w).sum().replace(0, np.nan)

    df["high_volume_down_day"] = ((df["daily_return"] < 0) & (df["volume_zscore_20"] > 1.0)).astype(float)
    for w in [10, 20]:
        df[f"high_volume_down_ratio_{w}"] = df["high_volume_down_day"].rolling(w).mean()
    df["volume_shock_20"] = volume / volume.rolling(20).mean()
    df["volume_shock_rank_252"] = rolling_rank_last(df["volume_shock_20"], 252)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["true_range"] = tr
    df["true_range_pct"] = tr / close
    for w in [5, 10, 14, 20, 60]:
        df[f"atr_{w}"] = tr.rolling(w).mean()
        df[f"atr_pct_{w}"] = df[f"atr_{w}"] / close

    df["atr_ratio_14_60"] = df["atr_14"] / df["atr_60"]
    df["atr_ratio_20_60"] = df["atr_20"] / df["atr_60"]
    df["atr_rank_252"] = rolling_rank_last(df["atr_pct_20"], 252)

    log_hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(close / open_).replace([np.inf, -np.inf], np.nan)
    log_oc = np.log(open_ / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    log_ho = np.log(high / open_).replace([np.inf, -np.inf], np.nan)
    log_lo = np.log(low / open_).replace([np.inf, -np.inf], np.nan)

    parkinson_var = (1.0 / (4.0 * np.log(2.0))) * (log_hl ** 2)
    gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    for w in [5, 10, 20, 60]:
        df[f"realized_vol_{w}"] = df["daily_return"].rolling(w).std()
        df[f"ewma_vol_{w}"] = df["daily_return"].ewm(span=w, adjust=False).std()
        df[f"parkinson_vol_{w}"] = np.sqrt(parkinson_var.rolling(w).mean().clip(lower=0))
        df[f"garman_klass_vol_{w}"] = np.sqrt(gk_var.rolling(w).mean().clip(lower=0))
        df[f"rogers_satchell_vol_{w}"] = np.sqrt(rs_var.rolling(w).mean().clip(lower=0))

    df["realized_vol_ratio_10_20"] = df["realized_vol_10"] / df["realized_vol_20"]
    downside_return = df["daily_return"].clip(upper=0)
    for w in [10, 20, 60]:
        df[f"downside_vol_{w}"] = downside_return.rolling(w).std()
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
    df["vol_of_vol_20"] = df["realized_vol_20"].rolling(20).std()

    df["bearish_ma_stack"] = ((df["ma_5"] < df["ma_20"]) & (df["ma_20"] < df["ma_60"])).astype(float)

    df = add_future_targets(df, horizons)

    feature_cols = [
        "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "return_5d_minus_20d", "return_10d_minus_20d",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_5", "trend_slope_10", "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_10", "positive_return_ratio_20", "positive_return_ratio_60",
        "large_down_day_ratio_10", "large_down_day_ratio_20",
        "large_up_day_ratio_10", "large_up_day_ratio_20",
        "drawdown_5", "drawdown_10", "drawdown_20", "drawdown_60", "drawdown_120",
        "price_position_10", "price_position_20", "price_position_60",
        "price_position_5_minus_20", "price_position_10_minus_20",
        "close_to_10d_high", "close_to_20d_high", "close_to_60d_high",
        "trend_consistency_20", "trend_consistency_60", "bearish_ma_stack",
        "volume_ratio_20", "volume_zscore_20",
        "down_volume_ratio_10", "down_volume_ratio_20",
        "high_volume_down_ratio_10", "high_volume_down_ratio_20",
        "volume_shock_rank_252",
        "true_range_pct",
        "atr_pct_5", "atr_pct_10", "atr_pct_14", "atr_pct_20", "atr_pct_60", "atr_rank_252",
        "atr_ratio_14_60", "atr_ratio_20_60",
        "realized_vol_10", "realized_vol_20", "realized_vol_60", "realized_vol_ratio_10_20",
        "ewma_vol_20", "ewma_vol_60",
        "parkinson_vol_20", "parkinson_vol_60",
        "garman_klass_vol_20", "rogers_satchell_vol_20",
        "downside_vol_10", "downside_vol_20", "downside_vol_60",
        "semi_vol_20",
        "ulcer_index_20", "ulcer_index_60", "ulcer_rank_252",
        "bb_width_20", "bb_width_rank_252",
        "vol_of_vol_20",
    ]
    return df, [c for c in feature_cols if c in df.columns]


def _row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        val = row.get(col, default)
        if pd.isna(val):
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def _finite_row_float(row: pd.Series, col: str) -> Optional[float]:
    try:
        if col not in row.index:
            return None
        val = row.get(col)
        if pd.isna(val):
            return None
        val = float(val)
        if not np.isfinite(val):
            return None
        return val
    except Exception:
        return None


def append_policy_context_to_prediction_row(out: Dict[str, object], source_row: pd.Series) -> None:
    """
    v8.7.1 핵심 패치.
    allocation policy가 쓰는 과거 피처를 pred_raw에 저장한다.
    future_* 컬럼은 저장하지 않으므로 look-ahead leakage는 발생하지 않는다.
    """
    for col in POLICY_CONTEXT_COLUMNS:
        if col in source_row.index:
            val = source_row.get(col)
            if pd.isna(val):
                out[col] = np.nan
            else:
                try:
                    out[col] = float(val)
                except Exception:
                    out[col] = val


def compute_mid_trend_score(row: pd.Series) -> Tuple[int, str]:
    """
    v8.7.1 수정 버전.
    필요한 context 컬럼이 없으면 BEAR로 강제 판정하지 않고 UNKNOWN을 반환한다.
    """
    checks: List[Optional[bool]] = []

    def add_check(col: str, op: str = "gt0") -> None:
        val = _finite_row_float(row, col)
        if val is None:
            checks.append(None)
        elif op == "gt0":
            checks.append(val > 0.0)
        else:
            raise ValueError(op)

    add_check("return_60d")
    add_check("return_120d")
    add_check("price_ma_60_gap")
    add_check("price_ma_120_gap")
    add_check("ma_gap_20_60")
    add_check("trend_slope_60")

    valid = [x for x in checks if x is not None]
    if len(valid) < 4:
        return 0, "UNKNOWN"

    score = int(sum(bool(x) for x in valid))
    if score >= 4:
        state = "BULL"
    elif score <= 2:
        state = "BEAR"
    else:
        state = "NEUTRAL"
    return score, state


def _normalize_weight_tuple(stock: float, bond: float, cash: float) -> Tuple[float, float, float]:
    vals = np.asarray([stock, bond, cash], dtype=float)
    vals = np.clip(vals, 0.0, 1.0)
    total = float(vals.sum())
    if total <= 0:
        return 1.0, 0.0, 0.0
    vals = vals / total
    return float(vals[0]), float(vals[1]), float(vals[2])


def _redistribute_after_stock_change(
    new_stock: float,
    old_w: Tuple[float, float, float],
    cash_ratio: Optional[float] = None,
) -> Tuple[float, float, float]:
    new_stock = float(np.clip(new_stock, 0.0, 1.0))
    remain = max(0.0, 1.0 - new_stock)
    old_bond, old_cash = old_w[1], old_w[2]
    defensive_total = old_bond + old_cash
    if cash_ratio is None:
        if defensive_total <= 0:
            cash_ratio = 0.35
        else:
            cash_ratio = float(np.clip(old_cash / defensive_total, 0.0, 1.0))
    bond_ratio = 1.0 - float(cash_ratio)
    return _normalize_weight_tuple(new_stock, remain * bond_ratio, remain * float(cash_ratio))


def _market_sharpe_proxy(row: pd.Series, horizon: int) -> float:
    h = int(horizon)
    ret = _finite_row_float(row, f"return_{h}d")
    if ret is None:
        return 0.0
    vol_col = f"realized_vol_{h}" if f"realized_vol_{h}" in row.index else "realized_vol_60"
    vol = _finite_row_float(row, vol_col)
    if vol is None or vol <= 1e-12:
        return 0.0
    return float(ret / (vol * math.sqrt(max(h, 1))))


def _trend_floor_from_high_vol(ph: float, cfg: Config) -> float:
    ph = float(np.clip(ph, 0.0, 1.0))
    if ph < 0.25:
        return cfg.trend_floor_lt25
    if ph < 0.35:
        return cfg.trend_floor_lt35
    if ph < 0.50:
        return cfg.trend_floor_lt50
    if ph < 0.65:
        return cfg.trend_floor_lt65
    return 0.0


def apply_trend_participation_overlay(
    signal_w: Tuple[float, float, float],
    row: pd.Series,
    cfg: Config,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    ph = _row_float(row, "prob_high_vol", 0.0)
    pds_score = _row_float(row, "prob_down_strengthening_score", 0.0)
    sharpe60 = _market_sharpe_proxy(row, 60)
    sharpe120 = _market_sharpe_proxy(row, 120)
    pos_ratio60 = _row_float(row, "positive_return_ratio_60", _row_float(row, "trend_consistency_60", 0.0))
    trend_score, trend_state = compute_mid_trend_score(row)

    context_ok = trend_state != "UNKNOWN"
    trend_bull = bool(
        cfg.enable_trend_participation_overlay
        and context_ok
        and ph < cfg.trend_max_high_vol_for_overlay
        and trend_score >= cfg.trend_min_mid_trend_score
        and pos_ratio60 >= cfg.trend_positive_ratio_60_threshold
        and (
            sharpe60 >= cfg.trend_sharpe60_threshold
            or sharpe120 >= cfg.trend_sharpe120_threshold
        )
    )

    action = "off"
    target_stock = float(signal_w[0])
    if trend_bull:
        floor = _trend_floor_from_high_vol(ph, cfg)
        if cfg.trend_full_stock_when_low_vol and ph < cfg.trend_full_stock_high_vol_threshold:
            floor = max(floor, 1.0)
        if pds_score >= 0.50:
            floor = min(floor, max(float(signal_w[0]), 0.90))
        if floor > target_stock + 1e-12:
            target_stock = floor
            action = "trend_floor_upgrade"

    out_w = _redistribute_after_stock_change(target_stock, signal_w)
    meta = {
        "market_sharpe_60": float(sharpe60),
        "market_sharpe_120": float(sharpe120),
        "trend_positive_ratio_60": float(pos_ratio60),
        "trend_context_ok": bool(context_ok),
        "trend_bull_regime": bool(trend_bull),
        "trend_participation_action": action,
        "trend_participation_target_stock": float(target_stock),
        "trend_participation_overlay": float(out_w[0] - signal_w[0]),
        "trend_participation_force_rebalance": bool(
            trend_bull and action != "off" and cfg.trend_force_rebalance
        ),
        "trend_score_for_overlay": int(trend_score),
        "trend_state_for_overlay": trend_state,
    }
    return out_w, meta


def apply_recovery_rerisk_overlay(
    signal_w: Tuple[float, float, float],
    row: pd.Series,
    cfg: Config,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    ph = _row_float(row, "prob_high_vol", 0.0)
    dd60_val = _finite_row_float(row, "drawdown_60")
    ret10_val = _finite_row_float(row, "return_10d")
    ret20_val = _finite_row_float(row, "return_20d")
    ma20_gap_val = _finite_row_float(row, "price_ma_20_gap")

    context_ok = all(v is not None for v in [dd60_val, ret10_val, ret20_val, ma20_gap_val])
    dd60 = float(dd60_val) if dd60_val is not None else 0.0
    ret10 = float(ret10_val) if ret10_val is not None else 0.0
    ret20 = float(ret20_val) if ret20_val is not None else 0.0
    ma20_gap = float(ma20_gap_val) if ma20_gap_val is not None else 0.0

    pds_score = _row_float(row, "prob_down_strengthening_score", 0.0)
    pus_score = _row_float(row, "prob_up_strengthening_score", 0.0)

    recovery = bool(
        cfg.enable_recovery_rerisk_overlay
        and context_ok
        and dd60 <= cfg.recovery_dd60_threshold
        and (ret10 >= cfg.recovery_return10_threshold or ret20 >= cfg.recovery_return20_threshold)
        and ma20_gap >= cfg.recovery_price_ma20_gap_threshold
        and pds_score < cfg.recovery_down_strength_max
        and ph < cfg.recovery_high_vol_max
    )

    action = "off"
    target_stock = float(signal_w[0])
    if recovery:
        floor = cfg.recovery_stock_floor
        if pus_score >= cfg.recovery_strong_up_score_threshold:
            floor = max(floor, cfg.recovery_strong_stock_floor)
        if floor > target_stock + 1e-12:
            target_stock = floor
            action = "recovery_rerisk_upgrade"

    out_w = _redistribute_after_stock_change(target_stock, signal_w)
    meta = {
        "recovery_context_ok": bool(context_ok),
        "recovery_risk_on": bool(recovery),
        "recovery_rerisk_action": action,
        "recovery_rerisk_target_stock": float(target_stock),
        "recovery_rerisk_overlay": float(out_w[0] - signal_w[0]),
        "recovery_rerisk_force_rebalance": bool(
            recovery and action != "off" and cfg.recovery_force_rebalance
        ),
    }
    return out_w, meta


def apply_soft_drawdown_guard(
    signal_w: Tuple[float, float, float],
    row: pd.Series,
    cfg: Config,
    portfolio_drawdown: float,
    *,
    trend_bull: bool,
    recovery_risk_on: bool,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    ph = _row_float(row, "prob_high_vol", 0.0)
    active = False
    action = "off"
    cut = 0.0
    target_stock = float(signal_w[0])

    if cfg.enable_soft_drawdown_guard:
        threshold1 = cfg.drawdown_guard_threshold_1
        threshold2 = cfg.drawdown_guard_threshold_2
        if trend_bull or recovery_risk_on:
            threshold1 -= 0.03
            threshold2 -= 0.03

        if ph >= cfg.drawdown_guard_high_vol_min and target_stock > cfg.drawdown_guard_min_stock:
            if portfolio_drawdown <= threshold2:
                active = True
                cut = cfg.drawdown_guard_cut_2
                target_stock = min(
                    max(cfg.drawdown_guard_min_stock, target_stock - cut),
                    cfg.drawdown_guard_stock_cap_2,
                )
                action = "drawdown_guard_level2"
            elif portfolio_drawdown <= threshold1:
                active = True
                cut = cfg.drawdown_guard_cut_1
                target_stock = max(cfg.drawdown_guard_min_stock, target_stock - cut)
                action = "drawdown_guard_level1"

    out_w = _redistribute_after_stock_change(target_stock, signal_w)
    meta = {
        "portfolio_drawdown_guard_input": float(portfolio_drawdown),
        "drawdown_guard_active": bool(active),
        "drawdown_guard_action": action,
        "drawdown_guard_cut": float(cut),
        "drawdown_guard_target_stock": float(target_stock),
        "drawdown_guard_overlay": float(out_w[0] - signal_w[0]),
        "drawdown_guard_force_rebalance": bool(active),
    }
    return out_w, meta


def calc_scale_pos_weight(y_binary: np.ndarray) -> float:
    pos = float(np.sum(y_binary == 1))
    neg = float(np.sum(y_binary == 0))
    if pos <= 0 or neg <= 0:
        return 1.0
    return max(0.1, min(20.0, neg / pos))


def make_xgb_binary(cfg: Config, scale_pos_weight: float, model_type: str = "stage1") -> Pipeline:
    if model_type == "stage1":
        n_estimators = cfg.stage1_n_estimators
        learning_rate = cfg.stage1_learning_rate
        max_depth = cfg.stage1_max_depth
        min_child_weight = cfg.stage1_min_child_weight
        subsample = cfg.stage1_subsample
        colsample_bytree = cfg.stage1_colsample_bytree
        reg_lambda = cfg.stage1_reg_lambda
        reg_alpha = cfg.stage1_reg_alpha
    else:
        n_estimators = cfg.down_n_estimators
        learning_rate = cfg.down_learning_rate
        max_depth = cfg.down_max_depth
        min_child_weight = cfg.down_min_child_weight
        subsample = cfg.down_subsample
        colsample_bytree = cfg.down_colsample_bytree
        reg_lambda = cfg.down_reg_lambda
        reg_alpha = cfg.down_reg_alpha

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def make_xgb_strength(cfg: Config, n_classes: int) -> Pipeline:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=int(n_classes),
        eval_metric="mlogloss",
        n_estimators=cfg.strength_n_estimators,
        learning_rate=cfg.strength_learning_rate,
        max_depth=cfg.strength_max_depth,
        min_child_weight=cfg.strength_min_child_weight,
        subsample=cfg.strength_subsample,
        colsample_bytree=cfg.strength_colsample_bytree,
        reg_lambda=cfg.strength_reg_lambda,
        reg_alpha=cfg.strength_reg_alpha,
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


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


def compute_policy_thresholds(train_df: pd.DataFrame, horizon: int, cfg: Config) -> Dict[str, float]:
    fvol = train_df[f"future_volatility_{horizon}d"]
    fmin = train_df[f"future_min_return_{horizon}d"]
    fmax = train_df[f"future_max_return_{horizon}d"]
    return {
        "vol": float(fvol.quantile(cfg.high_vol_quantile)),
        "down": float(fmin.quantile(cfg.down_quantile)),
        "up": float(fmax.quantile(cfg.up_quantile)),
    }


def make_high_vol_label(df: pd.DataFrame, horizon: int, th: Dict[str, float]) -> pd.Series:
    return (
        (df[f"future_volatility_{horizon}d"] >= th["vol"])
        | (df[f"future_min_return_{horizon}d"] <= th["down"])
        | (df[f"future_max_return_{horizon}d"] >= th["up"])
    ).astype(int)


def make_down_label(df: pd.DataFrame, horizon: int, th: Dict[str, float]) -> pd.Series:
    return (df[f"future_min_return_{horizon}d"] <= th["down"]).astype(int)


def expected_horizon_vol(df: pd.DataFrame, horizon: int) -> pd.Series:
    vol = df.get("realized_vol_20")
    if vol is None:
        vol = df["Close"].pct_change().rolling(20).std()
    return vol * math.sqrt(max(horizon, 1))


def strength_current_trend_score(df: pd.DataFrame) -> pd.Series:
    comps = pd.DataFrame(index=df.index)
    comps["ret60_pos"] = (df.get("return_60d", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ret120_pos"] = (df.get("return_120d", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ma60_gap_pos"] = (df.get("price_ma_60_gap", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ma120_gap_pos"] = (df.get("price_ma_120_gap", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ma20_60_pos"] = (df.get("ma_gap_20_60", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["slope60_pos"] = (df.get("trend_slope_60", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    return comps.sum(axis=1)


def build_direction_strength_labels(df: pd.DataFrame, cfg: Config, horizon: int) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    ret_col = f"future_return_{horizon}d"
    if ret_col not in df.columns:
        raise KeyError(f"missing {ret_col}")

    r_h = df[ret_col]
    vol_h = expected_horizon_vol(df, horizon).replace(0, np.nan)
    ret_eps = cfg.direction_strength_ret_eps_k * vol_h

    trend_score_t = strength_current_trend_score(df)
    trend_score_f = trend_score_t.shift(-horizon)
    trend_delta = trend_score_f - trend_score_t

    valid = r_h.notna() & vol_h.notna() & trend_delta.notna()
    up_strength = valid & (r_h >= ret_eps) & (trend_delta > 0)
    down_strength = valid & (r_h <= -ret_eps) & (trend_delta < 0)

    y_label = pd.Series("NO_STRENGTH_SIGNAL", index=df.index, dtype=object)
    y_label.loc[up_strength] = "UP_STRENGTHENING"
    y_label.loc[down_strength] = "DOWN_STRENGTHENING"

    aux = pd.DataFrame(
        {
            "direction_strength_label": y_label,
            "direction_strength_ret_eps": ret_eps,
            "direction_strength_trend_delta": trend_delta,
        },
        index=df.index,
    )
    y_id = y_label.map(DIRECTION_STRENGTH_LABEL_TO_ID).astype(float)
    return y_id, valid.astype(bool), aux


def strength_feature_cols(feature_cols: List[str]) -> List[str]:
    preferred = [
        "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "return_5d_minus_20d", "return_10d_minus_20d",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_5", "trend_slope_10", "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_10", "positive_return_ratio_20", "positive_return_ratio_60",
        "large_up_day_ratio_10", "large_up_day_ratio_20",
        "large_down_day_ratio_10", "large_down_day_ratio_20",
        "drawdown_10", "drawdown_20", "drawdown_60", "drawdown_120",
        "price_position_10", "price_position_20", "price_position_60",
        "close_to_10d_high", "close_to_20d_high", "close_to_60d_high",
        "trend_consistency_20", "trend_consistency_60",
        "volume_ratio_20", "volume_zscore_20", "down_volume_ratio_20", "high_volume_down_ratio_20",
        "atr_pct_10", "atr_pct_14", "atr_pct_20", "atr_rank_252",
        "realized_vol_20", "ewma_vol_20", "semi_vol_20", "ulcer_index_20", "bb_width_20",
    ]
    available = set(feature_cols)
    out = [c for c in preferred if c in available]
    return out if out else feature_cols


def _multiclass_sample_weights(y_local: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_local.astype(int))
    total = float(len(y_local))
    k = max(1, len(counts))
    weights = {i: total / (k * c) for i, c in enumerate(counts) if c > 0}
    return np.asarray([weights.get(int(v), 1.0) for v in y_local], dtype=float)


def fit_strength_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    y_strength: pd.Series,
    valid_strength: pd.Series,
    cfg: Config,
) -> Dict[str, object]:
    cols = strength_feature_cols(feature_cols)
    idx = train_df.index[valid_strength.loc[train_df.index].fillna(False).astype(bool).values]
    idx = idx[-1260:]
    if len(idx) < 300:
        return {"model": None, "cols": cols, "available": False, "classes": []}

    y_global = y_strength.loc[idx].astype(int).values
    orig_classes = np.array(sorted(np.unique(y_global).astype(int)))
    if len(orig_classes) < 2:
        return {"model": None, "cols": cols, "available": False, "classes": orig_classes.tolist()}

    local_map = {int(c): i for i, c in enumerate(orig_classes)}
    y_local = np.asarray([local_map[int(v)] for v in y_global], dtype=int)
    model = make_xgb_strength(cfg, n_classes=len(orig_classes))
    sw = _multiclass_sample_weights(y_local)
    model.fit(train_df.loc[idx, cols], y_local, model__sample_weight=sw)
    model.named_steps["model"].original_label_ids_ = orig_classes
    return {"model": model, "cols": cols, "available": True, "classes": orig_classes.tolist()}


def predict_strength_one(spec: Dict[str, object], row_df: pd.DataFrame) -> Dict[str, float]:
    out = {name: 0.0 for name in DIRECTION_STRENGTH_LABELS}
    if not spec or not bool(spec.get("available", False)):
        return out
    model = spec.get("model")
    cols = spec.get("cols", [])
    if model is None or not cols:
        return out

    proba = model.predict_proba(row_df[list(cols)])[0]  # type: ignore[union-attr]
    clf = model.named_steps["model"]  # type: ignore[index]
    classes = getattr(clf, "original_label_ids_", getattr(clf, "classes_", np.arange(len(proba))))
    aligned = np.zeros(len(DIRECTION_STRENGTH_LABELS), dtype=float)
    for j, cls in enumerate(classes):
        if int(cls) in DIRECTION_STRENGTH_ID_TO_LABEL and j < len(proba):
            aligned[int(cls)] = float(proba[j])
    if aligned.sum() <= 0:
        aligned[:] = 1.0 / len(aligned)
    else:
        aligned = aligned / aligned.sum()
    return {name: float(np.clip(aligned[i], 0.0, 1.0)) for i, name in enumerate(DIRECTION_STRENGTH_LABELS)}


def get_up_strength_horizon_weights(cfg: Config) -> Dict[int, float]:
    raw = {
        5: max(0.0, cfg.up_strength_weight_5d),
        10: max(0.0, cfg.up_strength_weight_10d),
        20: max(0.0, cfg.up_strength_weight_20d),
    }
    total = sum(raw.values())
    if total <= 0:
        return {5: 1 / 3, 10: 1 / 3, 20: 1 / 3}
    return {h: v / total for h, v in raw.items()}


def get_down_strength_horizon_weights(cfg: Config) -> Dict[int, float]:
    raw = {
        5: max(0.0, cfg.down_strength_weight_5d),
        10: max(0.0, cfg.down_strength_weight_10d),
        20: max(0.0, cfg.down_strength_weight_20d),
    }
    total = sum(raw.values())
    if total <= 0:
        return {5: 1 / 3, 10: 1 / 3, 20: 1 / 3}
    return {h: v / total for h, v in raw.items()}


def combine_score(values: Dict[int, float], weights: Dict[int, float]) -> float:
    return float(np.clip(sum(weights.get(h, 0.0) * float(values.get(h, 0.0)) for h in weights), 0.0, 1.0))


def safe_auc(y_true: np.ndarray, p: np.ndarray, kind: str) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    if kind == "roc":
        return float(roc_auc_score(y_true, p))
    if kind == "pr":
        return float(average_precision_score(y_true, p))
    raise ValueError(kind)


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


def extract_model_importance_safe(model: Optional[Pipeline], cols: List[str]) -> Dict[str, float]:
    if model is None:
        return {}
    return extract_model_importance(model, cols)


def run_walk_forward(df: pd.DataFrame, feature_cols: List[str], cfg: Config) -> pd.DataFrame:
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

    strength_full_by_h: Dict[int, Tuple[pd.Series, pd.Series, pd.DataFrame]] = {}
    for h in [5, 10, 20]:
        strength_full_by_h[h] = build_direction_strength_labels(all_df, cfg, h)

    hv_weights = {
        5: 0.0,
        10: cfg.high_vol_weight_h10,
        20: cfg.high_vol_weight_h20,
    }
    hv_sum = sum(hv_weights.get(h, 0.0) for h in cfg.horizons)
    if hv_sum <= 0:
        hv_weights = {h: 1.0 / len(cfg.horizons) for h in cfg.horizons}
    else:
        hv_weights = {h: hv_weights.get(h, 0.0) / hv_sum for h in cfg.horizons}

    models: Dict[int, Dict[str, object]] = {}
    strength_models: Dict[int, Dict[str, object]] = {}
    last_retrain_k: Optional[int] = None
    prediction_rows: List[Dict[str, object]] = []
    stage1_imp_hist: List[Dict[str, float]] = []
    down_imp_hist: List[Dict[str, float]] = []
    strength_cols = strength_feature_cols(feature_cols)

    for k, pos in enumerate(candidate_positions):
        date = all_df.index[pos]
        train_end_pos = pos - max_gap
        if train_end_pos < cfg.min_train_rows:
            continue

        need_retrain = (not models) or last_retrain_k is None or (k - last_retrain_k >= cfg.retrain_every_n_days)
        if need_retrain:
            train_df = all_df.iloc[:train_end_pos].copy().dropna(subset=valid_cols)
            if cfg.max_train_rows is not None:
                train_df = train_df.tail(int(cfg.max_train_rows))
            if len(train_df) < cfg.min_train_rows:
                continue

            models = {}
            for h in cfg.horizons:
                train_df_h = train_df.tail(1260 if h == 20 else (1000 if h == 10 else 756))
                if len(train_df_h) < 504:
                    continue

                th = compute_policy_thresholds(train_df_h, h, cfg)
                y_high = make_high_vol_label(train_df_h, h, th).values
                y_down = make_down_label(train_df_h, h, th).values
                if len(np.unique(y_high)) < 2:
                    continue

                stage1_model = make_xgb_binary(cfg, calc_scale_pos_weight(y_high), "stage1")
                stage1_model.fit(train_df_h[feature_cols], y_high)
                imp = extract_model_importance(stage1_model, feature_cols)
                if imp:
                    stage1_imp_hist.append(imp)

                down_model = None
                if len(np.unique(y_down)) == 2 and int(y_down.sum()) >= 20:
                    down_model = make_xgb_binary(cfg, calc_scale_pos_weight(y_down), "down")
                    down_model.fit(train_df_h[strength_cols], y_down)
                    impd = extract_model_importance_safe(down_model, strength_cols)
                    if impd:
                        down_imp_hist.append(impd)

                models[h] = {
                    "stage1": stage1_model,
                    "down": down_model,
                    "thresholds": th,
                    "train_rows": int(len(train_df_h)),
                }

            strength_models = {}
            for h in [5, 10, 20]:
                y_s, valid_s, _aux = build_direction_strength_labels(train_df, cfg, h)
                strength_models[h] = fit_strength_model(train_df, feature_cols, y_s, valid_s, cfg)

            last_retrain_k = k

        if not models:
            continue

        row_df = all_df.iloc[[pos]]
        source_row = all_df.iloc[pos]
        out: Dict[str, object] = {"Date": date}

        # v8.7.1 핵심 패치: 정책용 과거 피처 보존
        append_policy_context_to_prediction_row(out, source_row)

        prob_high_ens = 0.0
        prob_down_ens = 0.0
        actual_primary_risk = "정상"
        actual_primary_split = "정상"

        for h in cfg.horizons:
            if h not in models:
                continue
            m = models[h]
            p_high = float(m["stage1"].predict_proba(row_df[feature_cols])[0, 1])  # type: ignore[index, union-attr]
            down_model = m.get("down")
            p_down = float(down_model.predict_proba(row_df[strength_cols])[0, 1]) if down_model is not None else 0.0  # type: ignore[union-attr]
            th = m["thresholds"]
            y_h = int(make_high_vol_label(row_df, h, th).iloc[0])
            y_d = int(make_down_label(row_df, h, th).iloc[0])

            actual_risk_h = "고변동" if y_h == 1 else "정상"
            actual_split_h = "하락고변동" if y_d == 1 else ("상승고변동" if y_h == 1 else "정상")

            out[f"prob_high_vol_h{h}"] = p_high
            out[f"prob_down_h{h}"] = p_down
            out[f"actual_risk_h{h}"] = actual_risk_h
            out[f"actual_split_vol_h{h}"] = actual_split_h
            out[f"train_rows_h{h}"] = int(m.get("train_rows", 0))

            prob_high_ens += hv_weights.get(h, 0.0) * p_high
            prob_down_ens += hv_weights.get(h, 0.0) * p_down

            if h == cfg.primary_horizon:
                actual_primary_risk = actual_risk_h
                actual_primary_split = actual_split_h

        up_strength_values: Dict[int, float] = {}
        down_strength_values: Dict[int, float] = {}
        for h in [5, 10, 20]:
            probs = predict_strength_one(strength_models.get(h, {}), row_df)
            up_strength_values[h] = probs.get("UP_STRENGTHENING", 0.0)
            down_strength_values[h] = probs.get("DOWN_STRENGTHENING", 0.0)
            out[f"prob_up_strengthening_{h}d"] = up_strength_values[h]
            out[f"prob_down_strengthening_{h}d"] = down_strength_values[h]
            aux = strength_full_by_h[h][2]
            out[f"actual_direction_strength_{h}d"] = str(aux.loc[date, "direction_strength_label"]) if date in aux.index else ""

        up_score = combine_score(up_strength_values, get_up_strength_horizon_weights(cfg))
        down_score = combine_score(down_strength_values, get_down_strength_horizon_weights(cfg))

        prob_high_ens = float(np.clip(prob_high_ens, 0.0, 1.0))
        prob_down_ens = float(np.clip(prob_down_ens, 0.0, 1.0))
        prob_overall_risk = prob_high_ens

        out.update(
            {
                "actual_risk": actual_primary_risk,
                "actual_split_vol": actual_primary_split,
                "prob_high_vol": prob_high_ens,
                "prob_normal": 1.0 - prob_high_ens,
                "prob_down": prob_down_ens,
                "prob_down_risk": prob_down_ens,
                "prob_overall_risk": prob_overall_risk,
                "prob_up_strengthening_score": up_score,
                "prob_down_strengthening_score": down_score,
                "prob_up_strengthening": up_strength_values.get(20, 0.0),
                "prob_down_strengthening": down_strength_values.get(20, 0.0),
                "pred_risk": "고변동" if prob_high_ens >= cfg.pred_high_vol_threshold else "정상",
                "pred_overall_risk": "위험" if prob_overall_risk >= cfg.pred_overall_risk_threshold else "정상",
                "stock_next_return": float(source_row["stock_next_return"]),
                "bond_next_return": float(source_row["bond_next_return"]),
                "cash_next_return": float(source_row["cash_next_return"]),
            }
        )
        prediction_rows.append(out)

    pred_df = pd.DataFrame(prediction_rows).sort_values("Date").reset_index(drop=True)
    if pred_df.empty:
        raise ValueError("walk-forward 예측 결과가 비어 있습니다.")

    if cfg.use_prob_ewma:
        prob_cols = [
            c for c in pred_df.columns
            if c.startswith("prob_high_vol")
            or c.startswith("prob_down")
            or c.startswith("prob_up_strengthening")
        ]
        for col in prob_cols:
            if pred_df[col].dtype.kind in "if" and not col.endswith("_raw"):
                pred_df[f"{col}_raw"] = pred_df[col]
                pred_df[col] = pred_df[col].ewm(span=cfg.prob_ewma_span, adjust=False).mean().clip(0.0, 1.0)

        pred_df["prob_normal"] = 1.0 - pred_df["prob_high_vol"]
        pred_df["prob_overall_risk"] = pred_df["prob_high_vol"]
        pred_df["pred_risk"] = np.where(pred_df["prob_high_vol"] >= cfg.pred_high_vol_threshold, "고변동", "정상")
        pred_df["pred_overall_risk"] = np.where(
            pred_df["prob_overall_risk"] >= cfg.pred_overall_risk_threshold, "위험", "정상"
        )

    pred_df.attrs["stage1_feature_importance_mean"] = mean_importance(stage1_imp_hist)
    pred_df.attrs["downrisk_feature_importance_mean"] = mean_importance(down_imp_hist)
    return pred_df


def base_weight_from_vol_probability(prob_high_vol: float, cfg: Config) -> Tuple[float, float, float]:
    ph = float(np.clip(prob_high_vol, 0.0, 1.0))
    if ph < 0.25:
        stock = cfg.vol_base_stock_lt_25
    elif ph < 0.35:
        stock = cfg.vol_base_stock_lt_35
    elif ph < 0.50:
        stock = cfg.vol_base_stock_lt_50
    elif ph < 0.65:
        stock = cfg.vol_base_stock_lt_65
    elif ph < 0.75:
        stock = cfg.vol_base_stock_lt_75
    elif ph < 0.86:
        stock = cfg.vol_base_stock_lt_86
    else:
        stock = cfg.vol_base_stock_ge_86

    stock = float(np.clip(stock, 0.0, 1.0))
    remain = 1.0 - stock
    bond = remain * cfg.vol_base_bond_ratio_of_defensive
    cash = remain * (1.0 - cfg.vol_base_bond_ratio_of_defensive)
    return _normalize_weight_tuple(stock, bond, cash)


def apply_strength_policy(
    base_w: Tuple[float, float, float],
    row: pd.Series,
    cfg: Config,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    ph = _row_float(row, "prob_high_vol", 0.0)
    p10 = _row_float(row, "prob_up_strengthening_10d", 0.0)
    p20 = _row_float(row, "prob_up_strengthening_20d", 0.0)
    up_score = _row_float(row, "prob_up_strengthening_score", 0.0)
    down_score = _row_float(row, "prob_down_strengthening_score", 0.0)
    trend_score, trend_state = compute_mid_trend_score(row)

    stock = float(base_w[0])
    target_stock = stock
    tier = 0

    tier1_signal = bool(up_score >= cfg.up_strength_bonus_threshold_1 and p20 >= 0.30 and ph < cfg.up_strength_low_vol_threshold_1)
    tier2_signal = bool(
        not cfg.disable_tier2_signal
        and up_score >= cfg.up_strength_bonus_threshold_2
        and p10 >= cfg.up_strength_confirm_10d_threshold_2
        and p20 >= cfg.up_strength_confirm_20d_threshold_2
        and ph < cfg.up_strength_low_vol_threshold_2
    )
    tier3_signal = bool(
        up_score >= cfg.up_strength_bonus_threshold_3
        and p20 >= cfg.up_strength_confirm_20d_threshold_3
        and ph < cfg.up_strength_low_vol_threshold_3
    )
    full_stock_signal = bool(
        up_score >= cfg.up_strength_full_stock_score_threshold
        and p10 >= cfg.up_strength_full_stock_10d_threshold
        and p20 >= cfg.up_strength_full_stock_20d_threshold
        and ph < cfg.up_strength_full_stock_high_vol_threshold
    )

    if tier1_signal:
        target_stock = max(target_stock, cfg.up_strength_single_20d_stock_weight)
        tier = max(tier, 1)
    if tier2_signal:
        target_stock = max(target_stock, cfg.up_strength_pair_10d_20d_stock_weight)
        tier = max(tier, 2)
    if tier3_signal:
        target_stock = max(target_stock, cfg.up_strength_all3_base_stock_weight)
        tier = max(tier, 3)
    if full_stock_signal:
        target_stock = max(target_stock, cfg.up_strength_all3_strong_stock_weight)
        tier = max(tier, 3)

    if trend_state == "BEAR" and tier in {1, 2} and not (tier3_signal or full_stock_signal):
        target_stock = min(target_stock, max(stock, 0.80))

    out_w = _redistribute_after_stock_change(target_stock, base_w)
    force_rebalance = bool((tier3_signal and cfg.force_tier3_rebalance) or (full_stock_signal and cfg.force_full_stock_rebalance))

    meta = {
        "mid_trend_score": int(trend_score),
        "mid_trend_state": trend_state,
        "up_strength_score": float(up_score),
        "down_strength_score": float(down_score),
        "up_strength_bonus": float(max(0.0, out_w[0] - base_w[0])),
        "offensive_active": bool(out_w[0] > base_w[0] + 1e-12),
        "offensive_tier": int(tier),
        "tier1_signal": bool(tier1_signal),
        "tier2_signal": bool(tier2_signal),
        "tier3_signal": bool(tier3_signal),
        "full_stock_signal": bool(full_stock_signal),
        "force_rebalance": bool(force_rebalance),
    }
    return out_w, meta


def infer_regime_from_stock_weight(stock: float) -> str:
    if stock >= 0.90:
        return "CUSTOM"
    if stock >= 0.74:
        return "NORMAL"
    if stock >= 0.55:
        return "WATCH"
    if stock >= 0.35:
        return "RISK_OFF"
    return "EXTREME_RISK"


def apply_allocation(pred_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    pred_df = pred_df.copy().reset_index(drop=True)
    prev_w: Optional[Tuple[float, float, float]] = None
    rows: List[Dict[str, object]] = []
    usage: Dict[str, int] = {}
    last_emergency_i = -10**9

    running_equity_net = cfg.initial_capital
    equity_history: List[float] = [running_equity_net]

    for i, row in pred_df.iterrows():
        ph = float(row["prob_high_vol"])
        emergency = bool(
            (
                ph >= cfg.emergency_high_vol_threshold
                or (ph >= cfg.emergency_combined_high_vol_threshold and ph >= cfg.emergency_combined_down_threshold)
            )
            and (i - last_emergency_i >= cfg.emergency_cooldown_days)
        )
        scheduled = i % cfg.rebalance_every_n_days == 0
        rebalance_due = prev_w is None or scheduled or emergency

        base_w = base_weight_from_vol_probability(ph, cfg)
        signal_w, policy_meta = apply_strength_policy(base_w, row, cfg)

        pre_structural = signal_w
        signal_w, trend_meta = apply_trend_participation_overlay(signal_w, row, cfg)
        policy_meta.update(trend_meta)

        signal_w, recovery_meta = apply_recovery_rerisk_overlay(signal_w, row, cfg)
        policy_meta.update(recovery_meta)

        dd_window = max(2, int(cfg.drawdown_guard_window))
        recent_equity = equity_history[-dd_window:]
        peak_equity = max(recent_equity) if recent_equity else running_equity_net
        portfolio_dd = running_equity_net / peak_equity - 1.0 if peak_equity > 0 else 0.0

        signal_w, guard_meta = apply_soft_drawdown_guard(
            signal_w,
            row,
            cfg,
            portfolio_dd,
            trend_bull=bool(policy_meta.get("trend_bull_regime", False)),
            recovery_risk_on=bool(policy_meta.get("recovery_risk_on", False)),
        )
        policy_meta.update(guard_meta)

        force_policy = bool(
            policy_meta.get("force_rebalance", False)
            or policy_meta.get("trend_participation_force_rebalance", False)
            or policy_meta.get("recovery_rerisk_force_rebalance", False)
            or policy_meta.get("drawdown_guard_force_rebalance", False)
        )
        if force_policy:
            rebalance_due = True

        signal_regime = infer_regime_from_stock_weight(signal_w[0])
        hold_reason = "rebalanced"

        if prev_w is None:
            w = signal_w
            hold_reason = "initial"
        elif not rebalance_due:
            stale_gap = float(prev_w[0] - signal_w[0])
            stale_up = float(row.get("prob_up_strengthening_score", 0.0))
            stale_decay = (
                cfg.enable_stale_offensive_decay
                and stale_gap >= cfg.stale_offensive_stock_gap_threshold
                and (
                    stale_up < cfg.stale_offensive_up_strength_reset_threshold
                    or ph >= cfg.stale_offensive_high_vol_threshold
                )
            )
            if stale_decay:
                w = signal_w
                hold_reason = "stale_offensive_decay"
            else:
                w = prev_w
                hold_reason = "not_rebalance_day"
        else:
            total_delta = sum(abs(signal_w[j] - prev_w[j]) for j in range(3))
            if force_policy:
                w = signal_w
                hold_reason = "policy_force"
            elif total_delta < cfg.no_trade_band:
                w = prev_w
                hold_reason = "no_trade_band"
            else:
                w = signal_w
                hold_reason = "emergency" if emergency else "scheduled"

        turnover = 0.0 if prev_w is None else sum(abs(w[j] - prev_w[j]) for j in range(3))
        gross = (
            w[0] * float(row["stock_next_return"])
            + w[1] * float(row["bond_next_return"])
            + w[2] * float(row["cash_next_return"])
        )
        cost = cfg.transaction_cost_rate * turnover
        net = gross - cost

        if emergency and rebalance_due:
            last_emergency_i = i

        running_equity_net *= (1.0 + net)
        equity_history.append(running_equity_net)

        executed_regime = infer_regime_from_stock_weight(w[0])
        out = row.to_dict()
        out.update(
            {
                "signal_regime": signal_regime,
                "allocation_regime": executed_regime,
                "executed_regime": executed_regime,
                "hold_reason": hold_reason,
                "base_signal_stock_weight": float(base_w[0]),
                "base_signal_bond_weight": float(base_w[1]),
                "base_signal_cash_weight": float(base_w[2]),
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
                "trade_executed": bool(turnover > 1e-12 or prev_w is None),
                "emergency_rebalance": bool(emergency and rebalance_due),
                "policy_force_rebalance": bool(force_policy),
                "signal_executed_stock_gap": float(w[0] - signal_w[0]),
                "abs_signal_executed_stock_gap": float(abs(w[0] - signal_w[0])),
                "structural_overlay_total": float(signal_w[0] - pre_structural[0]),
                "portfolio_drawdown_window_pre_trade": float(portfolio_dd),
                **policy_meta,
            }
        )
        rows.append(out)
        usage[executed_regime] = usage.get(executed_regime, 0) + 1
        prev_w = w

    out_df = pd.DataFrame(rows)
    out_df["strategy_equity_net"] = cfg.initial_capital * (1.0 + out_df["strategy_return_net"]).cumprod()
    out_df["strategy_equity_gross"] = cfg.initial_capital * (1.0 + out_df["strategy_return_gross"]).cumprod()
    return out_df, usage


def perf_stats(returns: pd.Series, initial_capital: float) -> Dict[str, float]:
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return {
            "final_capital": initial_capital,
            "total_return": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
        }
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


def classification_metrics(pred_df: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for h in cfg.horizons:
        if f"actual_risk_h{h}" in pred_df.columns and f"prob_high_vol_h{h}" in pred_df.columns:
            y = (pred_df[f"actual_risk_h{h}"] == "고변동").astype(int).values
            p = pred_df[f"prob_high_vol_h{h}"].astype(float).clip(0.0, 1.0).values
            metrics[f"stage1_h{h}"] = binary_cls_metrics(y, p, cfg.pred_high_vol_threshold, "고변동")
    if "actual_risk" in pred_df.columns and "prob_high_vol" in pred_df.columns:
        y = (pred_df["actual_risk"] == "고변동").astype(int).values
        p = pred_df["prob_high_vol"].astype(float).clip(0.0, 1.0).values
        metrics["stage1_ensemble_vs_primary"] = binary_cls_metrics(y, p, cfg.pred_high_vol_threshold, "고변동")

    if "actual_direction_strength_20d" in pred_df.columns:
        y_up = (pred_df["actual_direction_strength_20d"].astype(str) == "UP_STRENGTHENING").astype(int).values
        p_up = pred_df["prob_up_strengthening_score"].astype(float).clip(0.0, 1.0).values
        metrics["direction_strength_up_score_vs_20d"] = binary_cls_metrics(
            y_up, p_up, cfg.up_strength_bonus_threshold_1, "UP_STRENGTHENING_SCORE"
        )
        y_down = (pred_df["actual_direction_strength_20d"].astype(str) == "DOWN_STRENGTHENING").astype(int).values
        p_down = pred_df["prob_down_strengthening_score"].astype(float).clip(0.0, 1.0).values
        metrics["direction_strength_down_score_vs_20d"] = binary_cls_metrics(
            y_down, p_down, 0.30, "DOWN_STRENGTHENING_SCORE"
        )
    return metrics


def build_policy_context_diagnostics(pred_df: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    missing = [c for c in POLICY_CONTEXT_COLUMNS if c not in pred_df.columns]
    recomputed_scores: List[int] = []
    recomputed_states: List[str] = []
    for _, row in pred_df.iterrows():
        score, state = compute_mid_trend_score(row)
        recomputed_scores.append(score)
        recomputed_states.append(state)

    state_counts = pd.Series(recomputed_states).value_counts(normalize=True).mul(100).round(2).to_dict()
    trend_required = [
        "return_60d", "return_120d", "price_ma_60_gap", "price_ma_120_gap",
        "ma_gap_20_60", "trend_slope_60", "positive_return_ratio_60", "realized_vol_60",
    ]
    recovery_required = ["drawdown_60", "return_10d", "return_20d", "price_ma_20_gap"]
    return {
        "missing_policy_context_columns": missing,
        "policy_context_columns_expected": POLICY_CONTEXT_COLUMNS,
        "is_context_sufficient_for_trend": all(c in pred_df.columns for c in trend_required),
        "is_context_sufficient_for_recovery": all(c in pred_df.columns for c in recovery_required),
        "mid_trend_score_recomputed_min": int(np.nanmin(recomputed_scores)) if recomputed_scores else None,
        "mid_trend_score_recomputed_max": int(np.nanmax(recomputed_scores)) if recomputed_scores else None,
        "mid_trend_score_recomputed_mean": float(np.nanmean(recomputed_scores)) if recomputed_scores else None,
        "mid_trend_state_distribution_recomputed_pct": state_counts,
        "warning": (
            "policy context missing; trend/recovery overlay may be disabled"
            if missing
            else "policy context available"
        ),
    }


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
    total_n = len(pred_df)
    for regime, g in pred_df.groupby("allocation_regime", dropna=False):
        rr = g["strategy_return_net"].astype(float)
        rows.append(
            {
                "allocation_regime": str(regime),
                "count": int(len(g)),
                "pct": float(len(g) / total_n) if total_n else 0.0,
                "ann_return_est": _annualized_return(rr),
                "ann_vol_est": _annualized_vol(rr),
                "mean_daily_return": float(rr.mean()),
                "win_rate": _win_rate(rr),
                "avg_stock_weight": float(g["stock_weight"].mean()),
                "avg_turnover": float(g["turnover"].mean()),
                "annual_turnover_est": float(g["turnover"].mean() * 252.0),
                "avg_prob_high_vol": float(g["prob_high_vol"].mean()),
                "actual_high_vol_rate": float((g["actual_risk"] == "고변동").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("pct", ascending=False).reset_index(drop=True)


def build_probability_bins(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    specs = [
        ("prob_high_vol", "actual_risk", "고변동"),
        ("prob_up_strengthening_score", "actual_direction_strength_20d", "UP_STRENGTHENING"),
        ("prob_down_strengthening_score", "actual_direction_strength_20d", "DOWN_STRENGTHENING"),
    ]
    for prob_col, actual_col, positive_value in specs:
        if prob_col not in pred_df.columns or actual_col not in pred_df.columns:
            continue
        tmp = pred_df.copy()
        tmp["prob_bin"] = pd.cut(
            tmp[prob_col].astype(float).clip(0.0, 1.0),
            bins=np.linspace(0.0, 1.0, 11),
            include_lowest=True,
        )
        tmp["actual_positive"] = (tmp[actual_col].astype(str) == positive_value).astype(int)
        for b, g in tmp.groupby("prob_bin", observed=False):
            if g.empty:
                continue
            rows.append(
                {
                    "prob_col": prob_col,
                    "actual_col": actual_col,
                    "positive_value": positive_value,
                    "prob_bin": str(b),
                    "count": int(len(g)),
                    "actual_rate": float(g["actual_positive"].mean()),
                    "avg_prob": float(g[prob_col].mean()),
                    "ann_return_est": _annualized_return(g["strategy_return_net"]),
                    "avg_stock_weight": float(g["stock_weight"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_drawdown_episodes(pred_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if "strategy_equity_net" not in pred_df.columns or pred_df.empty:
        return pd.DataFrame()
    df = pred_df[["Date", "strategy_equity_net", "allocation_regime", "stock_weight", "prob_high_vol"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    equity = df["strategy_equity_net"].astype(float)
    peak = equity.cummax()
    dd = equity / peak - 1.0
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
                episodes.append(
                    {
                        "start_date": str(df.iloc[start_idx]["Date"].date()),
                        "trough_date": str(df.iloc[trough_idx]["Date"].date()),
                        "recovery_date": str(df.iloc[end_idx]["Date"].date()),
                        "depth": min_dd,
                        "duration_days": int(end_idx - start_idx),
                        "avg_stock_weight": float(seg["stock_weight"].mean()),
                        "avg_prob_high_vol": float(seg["prob_high_vol"].mean()),
                        "trough_regime": str(df.iloc[trough_idx]["allocation_regime"]),
                    }
                )
                in_dd = False

    if in_dd:
        end_idx = len(df) - 1
        seg = df.iloc[start_idx:end_idx + 1]
        episodes.append(
            {
                "start_date": str(df.iloc[start_idx]["Date"].date()),
                "trough_date": str(df.iloc[trough_idx]["Date"].date()),
                "recovery_date": "not_recovered",
                "depth": min_dd,
                "duration_days": int(end_idx - start_idx),
                "avg_stock_weight": float(seg["stock_weight"].mean()),
                "avg_prob_high_vol": float(seg["prob_high_vol"].mean()),
                "trough_regime": str(df.iloc[trough_idx]["allocation_regime"]),
            }
        )
    if not episodes:
        return pd.DataFrame()
    return pd.DataFrame(episodes).sort_values("depth").head(top_n).reset_index(drop=True)


def build_periodic_returns(pred_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    tmp = pred_df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    tmp = tmp.set_index("Date")
    rows: List[Dict[str, object]] = []
    for period, g in tmp.resample(freq):
        if g.empty:
            continue
        rows.append(
            {
                "period": str(period.date()),
                "strategy_net": float((1.0 + g["strategy_return_net"]).prod() - 1.0),
                "strategy_gross": float((1.0 + g["strategy_return_gross"]).prod() - 1.0),
                "stock_buy_hold": float((1.0 + g["stock_next_return"]).prod() - 1.0),
                "bond": float((1.0 + g["bond_next_return"]).prod() - 1.0),
                "cash": float((1.0 + g["cash_next_return"]).prod() - 1.0),
                "avg_stock_weight": float(g["stock_weight"].mean()),
                "turnover_sum": float(g["turnover"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_custom_signal_breakdown(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if pred_df.empty:
        return pd.DataFrame()
    total_n = len(pred_df)

    segments = {
        "trend_bull_true": pred_df.get("trend_bull_regime", pd.Series(False, index=pred_df.index)).astype(bool),
        "recovery_risk_on_true": pred_df.get("recovery_risk_on", pd.Series(False, index=pred_df.index)).astype(bool),
        "tier1_true": pred_df.get("tier1_signal", pd.Series(False, index=pred_df.index)).astype(bool),
        "tier2_true": pred_df.get("tier2_signal", pd.Series(False, index=pred_df.index)).astype(bool),
        "tier3_true": pred_df.get("tier3_signal", pd.Series(False, index=pred_df.index)).astype(bool),
        "full_stock_true": pred_df.get("full_stock_signal", pd.Series(False, index=pred_df.index)).astype(bool),
        "offensive_active_true": pred_df.get("offensive_active", pd.Series(False, index=pred_df.index)).astype(bool),
    }

    for name, mask in segments.items():
        g = pred_df.loc[mask].copy()
        rr = g["strategy_return_net"].astype(float) if not g.empty else pd.Series(dtype=float)
        rows.append(
            {
                "segment": name,
                "count": int(len(g)),
                "pct": float(len(g) / total_n) if total_n else 0.0,
                "win_rate": _win_rate(rr),
                "mean_daily_return": float(rr.mean()) if len(rr) else 0.0,
                "ann_return_est": _annualized_return(rr),
                "ann_vol_est": _annualized_vol(rr),
                "avg_stock_weight": float(g["stock_weight"].mean()) if not g.empty else 0.0,
                "avg_prob_high_vol": float(g["prob_high_vol"].mean()) if not g.empty else 0.0,
                "avg_up_strength_score": float(g["prob_up_strengthening_score"].mean()) if not g.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("ann_return_est", ascending=False).reset_index(drop=True)


def build_summary(pred_df: pd.DataFrame, feature_cols: List[str], gate_usage: Dict[str, int], cfg: Config) -> Dict[str, object]:
    perf = {
        "strategy_after_cost": perf_stats(pred_df["strategy_return_net"], cfg.initial_capital),
        "strategy_gross": perf_stats(pred_df["strategy_return_gross"], cfg.initial_capital),
        "stock_buy_hold": perf_stats(pred_df["stock_next_return"], cfg.initial_capital),
        "benchmark_60_40": perf_stats(
            0.6 * pred_df["stock_next_return"] + 0.4 * pred_df["bond_next_return"],
            cfg.initial_capital,
        ),
        "static_50_30_20": perf_stats(
            0.5 * pred_df["stock_next_return"]
            + 0.3 * pred_df["bond_next_return"]
            + 0.2 * pred_df["cash_next_return"],
            cfg.initial_capital,
        ),
    }
    latest = pred_df.iloc[-1]
    summary = {
        "model_type": "xgb_trend_participation_v8_7_1_policy_context_patch_full",
        "target_ticker": cfg.target_ticker,
        "bond_ticker": cfg.bond_ticker,
        "cash_ticker": cfg.cash_ticker,
        "config": asdict(cfg),
        "period": {
            "start": str(pred_df["Date"].iloc[0]),
            "end": str(pred_df["Date"].iloc[-1]),
            "rows": int(len(pred_df)),
        },
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "stage1_feature_importance_mean": pred_df.attrs.get("stage1_feature_importance_mean", {}),
        "downrisk_feature_importance_mean": pred_df.attrs.get("downrisk_feature_importance_mean", {}),
        "average_probabilities": {
            "avg_prob_normal": float(pred_df["prob_normal"].mean()),
            "avg_prob_high_vol": float(pred_df["prob_high_vol"].mean()),
            "avg_prob_up_strengthening_score": float(pred_df["prob_up_strengthening_score"].mean()),
            "avg_prob_down_strengthening_score": float(pred_df["prob_down_strengthening_score"].mean()),
        },
        "average_weights": {
            "avg_stock_weight": float(pred_df["stock_weight"].mean()),
            "avg_bond_weight": float(pred_df["bond_weight"].mean()),
            "avg_cash_weight": float(pred_df["cash_weight"].mean()),
            "min_stock_weight": float(pred_df["stock_weight"].min()),
            "max_stock_weight": float(pred_df["stock_weight"].max()),
        },
        "allocation_regime_distribution_pct": pred_df["allocation_regime"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "signal_regime_distribution_pct": pred_df["signal_regime"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "turnover": {
            "avg_daily_trade_ratio": float(pred_df["turnover"].mean()),
            "annual_turnover_estimate": float(pred_df["turnover"].mean() * 252.0),
            "total_transaction_cost_rate_sum": float(pred_df["transaction_cost"].sum()),
            "rebalance_due_ratio": float(pred_df["rebalance_due"].mean()),
            "trade_executed_ratio": float(pred_df["trade_executed"].mean()),
            "emergency_rebalance_ratio": float(pred_df["emergency_rebalance"].mean()),
        },
        "performance": perf,
        "classification": classification_metrics(pred_df, cfg),
        "gate_config_usage_top10": dict(sorted(gate_usage.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "policy_context_diagnostics": build_policy_context_diagnostics(pred_df, cfg),
        "structural_overlays": {
            "trend_participation_enabled": bool(cfg.enable_trend_participation_overlay),
            "trend_bull_rate": float(pred_df.get("trend_bull_regime", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "trend_participation_action_distribution_pct": pred_df.get("trend_participation_action", pd.Series("", index=pred_df.index)).astype(str).value_counts(normalize=True).mul(100).round(2).to_dict(),
            "avg_trend_participation_overlay": float(pred_df.get("trend_participation_overlay", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "recovery_rerisk_enabled": bool(cfg.enable_recovery_rerisk_overlay),
            "recovery_risk_on_rate": float(pred_df.get("recovery_risk_on", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
            "recovery_rerisk_action_distribution_pct": pred_df.get("recovery_rerisk_action", pd.Series("", index=pred_df.index)).astype(str).value_counts(normalize=True).mul(100).round(2).to_dict(),
            "avg_recovery_rerisk_overlay": float(pred_df.get("recovery_rerisk_overlay", pd.Series(0.0, index=pred_df.index)).astype(float).mean()),
            "soft_drawdown_guard_enabled": bool(cfg.enable_soft_drawdown_guard),
            "drawdown_guard_active_rate": float(pred_df.get("drawdown_guard_active", pd.Series(False, index=pred_df.index)).astype(bool).mean()),
        },
        "latest_prediction": {
            "date": str(latest["Date"]),
            "pred_risk": str(latest["pred_risk"]),
            "pred_overall_risk": str(latest.get("pred_overall_risk", "정상")),
            "prob_normal": round(float(latest["prob_normal"]) * 100, 2),
            "prob_high_vol": round(float(latest["prob_high_vol"]) * 100, 2),
            "prob_up_strengthening_score": round(float(latest.get("prob_up_strengthening_score", 0.0)) * 100, 2),
            "prob_up_strengthening_5d": round(float(latest.get("prob_up_strengthening_5d", 0.0)) * 100, 2),
            "prob_up_strengthening_10d": round(float(latest.get("prob_up_strengthening_10d", 0.0)) * 100, 2),
            "prob_up_strengthening_20d": round(float(latest.get("prob_up_strengthening_20d", 0.0)) * 100, 2),
            "prob_down_strengthening_score": round(float(latest.get("prob_down_strengthening_score", 0.0)) * 100, 2),
            "mid_trend_score": int(latest.get("mid_trend_score", 0)),
            "mid_trend_state": str(latest.get("mid_trend_state", "UNKNOWN")),
            "trend_bull_regime": bool(latest.get("trend_bull_regime", False)),
            "trend_participation_action": str(latest.get("trend_participation_action", "")),
            "recovery_risk_on": bool(latest.get("recovery_risk_on", False)),
            "recovery_rerisk_action": str(latest.get("recovery_rerisk_action", "")),
            "signal_regime": str(latest.get("signal_regime", "")),
            "allocation_regime": str(latest["allocation_regime"]),
            "hold_reason": str(latest.get("hold_reason", "unknown")),
            "signal_allocation": {
                "stock": round(float(latest.get("signal_stock_weight", latest["stock_weight"])) * 100, 2),
                "bond": round(float(latest.get("signal_bond_weight", latest["bond_weight"])) * 100, 2),
                "cash": round(float(latest.get("signal_cash_weight", latest["cash_weight"])) * 100, 2),
            },
            "executed_allocation": {
                "stock": round(float(latest["stock_weight"]) * 100, 2),
                "bond": round(float(latest["bond_weight"]) * 100, 2),
                "cash": round(float(latest["cash_weight"]) * 100, 2),
            },
        },
    }
    return summary


def print_summary(summary: Dict[str, object]) -> None:
    p = summary["performance"]
    w = summary["average_weights"]
    t = summary["turnover"]
    print("\n==============================")
    print("XGBoost v8.7.1 Policy Context Patch 결과 요약")
    print("==============================")
    print(f"기간: {summary['period']['start']} ~ {summary['period']['end']}")
    print(f"거래일 수: {summary['period']['rows']}")
    print(f"피처 수: {summary['feature_count']}")
    print(f"평균 주식 비중: {w['avg_stock_weight'] * 100:.2f}%")
    print(f"평균 채권 비중: {w['avg_bond_weight'] * 100:.2f}%")
    print(f"평균 현금 비중: {w['avg_cash_weight'] * 100:.2f}%")
    print(f"연간 교체율 추정: {t['annual_turnover_estimate'] * 100:.2f}%")
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
        print(f"Sortino: {st['sortino']:.4f}")
        print(f"Calmar: {st['calmar']:.4f}")

    print("\n[Policy Context Diagnostics]")
    print(json.dumps(summary["policy_context_diagnostics"], ensure_ascii=False, indent=2))

    print("\n[Structural Overlays]")
    print(json.dumps(summary["structural_overlays"], ensure_ascii=False, indent=2))

    print("\n[최신 예측]")
    print(json.dumps(summary["latest_prediction"], ensure_ascii=False, indent=2))


def apply_speed_profile(cfg: Config, profile: str) -> Config:
    if profile == "fast":
        cfg.retrain_every_n_days = 20
        cfg.stage1_n_estimators = 100
        cfg.down_n_estimators = 70
        cfg.result_dir = "results_xgb_trend_participation_v8_7_1_fast"
    elif profile == "balanced":
        cfg.retrain_every_n_days = 10
        cfg.stage1_n_estimators = 150
        cfg.down_n_estimators = 100
        cfg.result_dir = "results_xgb_trend_participation_v8_7_1_balanced"
    elif profile == "full":
        cfg.retrain_every_n_days = 10
        cfg.stage1_n_estimators = 220
        cfg.down_n_estimators = 160
        cfg.strength_n_estimators = 220
        cfg.result_dir = "results_xgb_trend_participation_v8_7_1_full"
    else:
        raise ValueError(f"알 수 없는 speed profile: {profile}")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost v8.7.1 Policy Context Patch Full Code")
    parser.add_argument("--speed-profile", choices=["fast", "balanced", "full"], default="balanced")
    parser.add_argument("--target-ticker", type=str, default=None)
    parser.add_argument("--asset-list", type=str, default=None)
    parser.add_argument("--asset-preset", choices=["etf", "mega", "mixed"], default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--backtest-start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--retrain-every", type=int, default=None)
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--h10-down-only", action="store_true", help="호환용 인자. 이 v8.7.1 재구성본에서는 high-vol ensemble만 사용합니다.")
    parser.add_argument("--no-trade-band", type=float, default=None)
    parser.add_argument("--rebalance-every", type=int, default=None)
    parser.add_argument("--execution-lag-days", type=int, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--allow-cash-download-fallback", action="store_true")

    parser.add_argument("--disable-trend-participation", action="store_true")
    parser.add_argument("--trend-sharpe60", type=float, default=None)
    parser.add_argument("--trend-sharpe120", type=float, default=None)
    parser.add_argument("--trend-positive-ratio60", type=float, default=None)
    parser.add_argument("--trend-max-high-vol", type=float, default=None)
    parser.add_argument("--trend-floor-lt25", type=float, default=None)
    parser.add_argument("--trend-floor-lt35", type=float, default=None)
    parser.add_argument("--trend-floor-lt50", type=float, default=None)
    parser.add_argument("--trend-floor-lt65", type=float, default=None)
    parser.add_argument("--trend-full-stock-low-vol", action="store_true")
    parser.add_argument("--trend-force-rebalance", action="store_true")

    parser.add_argument("--disable-recovery-rerisk", action="store_true")
    parser.add_argument("--recovery-dd60", type=float, default=None)
    parser.add_argument("--recovery-ret10", type=float, default=None)
    parser.add_argument("--recovery-ret20", type=float, default=None)
    parser.add_argument("--recovery-stock-floor", type=float, default=None)
    parser.add_argument("--recovery-strong-stock-floor", type=float, default=None)
    parser.add_argument("--recovery-no-force-rebalance", action="store_true")

    parser.add_argument("--enable-drawdown-guard", action="store_true")
    parser.add_argument("--drawdown-guard-window", type=int, default=None)
    parser.add_argument("--drawdown-guard-threshold-1", type=float, default=None)
    parser.add_argument("--drawdown-guard-threshold-2", type=float, default=None)

    parser.add_argument("--disable-tier2", action="store_true")
    parser.add_argument("--up-strength-threshold-1", type=float, default=None)
    parser.add_argument("--up-strength-threshold-2", type=float, default=None)
    parser.add_argument("--up-strength-threshold-3", type=float, default=None)
    parser.add_argument("--full-stock-score-threshold", type=float, default=None)
    parser.add_argument("--full-stock-10d-threshold", type=float, default=None)
    parser.add_argument("--full-stock-20d-threshold", type=float, default=None)
    parser.add_argument("--full-stock-high-vol-threshold", type=float, default=None)

    parser.add_argument("--no-diagnostics", action="store_true")
    return parser.parse_args()


def run_single_asset(cfg: Config, args: argparse.Namespace) -> None:
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
                f"현금 수익률을 0으로 대체하려면 --allow-cash-download-fallback을 사용하세요."
            ) from exc
        warnings.warn(f"{cfg.cash_ticker} 다운로드 실패로 cash return을 0으로 대체합니다: {exc}", RuntimeWarning)
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
    df = pd.concat([df, returns_df], axis=1).copy()

    print(f"    target: {cfg.target_ticker}")
    print(f"    피처 수: {len(feature_cols)}")
    print(f"    horizons: {cfg.horizons}")
    print(f"    execution_lag_days: {cfg.execution_lag_days}")
    print(f"    max_train_rows: {cfg.max_train_rows}")
    print(f"    trend_participation: {cfg.enable_trend_participation_overlay}")
    print(f"    recovery_rerisk: {cfg.enable_recovery_rerisk_overlay}")

    print("[3/5] Walk-forward 예측")
    pred_raw = run_walk_forward(df, feature_cols, cfg)

    print("[4/5] 배분/백테스트")
    pred_df, gate_usage = apply_allocation(pred_raw, cfg)
    pred_df.attrs.update(pred_raw.attrs)

    print("[5/5] 결과 저장")
    summary = build_summary(pred_df, feature_cols, gate_usage, cfg)

    safe_ticker = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(cfg.target_ticker)).strip("_") or "asset"
    prefix = f"{safe_ticker}_xgb_trend_participation_v8_7_1"

    pred_path = result_dir / f"{prefix}_predictions.csv"
    summary_path = result_dir / f"{prefix}_summary.json"
    latest_path = result_dir / f"{prefix}_latest.json"
    stage1_imp_path = result_dir / f"{prefix}_stage1_feature_importance.csv"
    down_imp_path = result_dir / f"{prefix}_downrisk_feature_importance.csv"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary["latest_prediction"], f, ensure_ascii=False, indent=2)

    pd.Series(summary.get("stage1_feature_importance_mean", {}), name="importance").to_csv(stage1_imp_path, encoding="utf-8-sig")
    pd.Series(summary.get("downrisk_feature_importance_mean", {}), name="importance").to_csv(down_imp_path, encoding="utf-8-sig")

    diagnostic_paths: List[Path] = []
    if not args.no_diagnostics:
        diagnostics = {
            "regime_analysis": build_regime_analysis(pred_df),
            "probability_bins": build_probability_bins(pred_df),
            "drawdown_episodes": build_drawdown_episodes(pred_df),
            "monthly_returns": build_periodic_returns(pred_df, "ME"),
            "annual_returns": build_periodic_returns(pred_df, "YE"),
            "custom_signal_breakdown": build_custom_signal_breakdown(pred_df),
        }
        for name, diag_df in diagnostics.items():
            if diag_df is not None and not diag_df.empty:
                p = result_dir / f"{prefix}_{name}.csv"
                diag_df.to_csv(p, index=False, encoding="utf-8-sig")
                diagnostic_paths.append(p)

    print_summary(summary)
    print("\n[저장 완료]")
    for p in [pred_path, summary_path, latest_path, stage1_imp_path, down_imp_path, *diagnostic_paths]:
        print(f"- {p}")


def apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.target_ticker:
        cfg.target_ticker = str(args.target_ticker).upper()
    if args.start_date:
        cfg.start_date = args.start_date
    if args.backtest_start_date:
        cfg.backtest_start_date = args.backtest_start_date
    if args.end_date:
        cfg.end_date = args.end_date
    if args.n_jobs is not None:
        cfg.n_jobs = int(args.n_jobs)
    if args.retrain_every is not None:
        cfg.retrain_every_n_days = int(args.retrain_every)
    if args.result_dir:
        cfg.result_dir = args.result_dir
    if args.no_trade_band is not None:
        cfg.no_trade_band = float(args.no_trade_band)
    if args.rebalance_every is not None:
        cfg.rebalance_every_n_days = int(args.rebalance_every)
    if args.execution_lag_days is not None:
        cfg.execution_lag_days = int(args.execution_lag_days)
    if args.max_train_rows is not None:
        cfg.max_train_rows = int(args.max_train_rows)
    if args.allow_cash_download_fallback:
        cfg.allow_cash_download_fallback = True

    if args.disable_trend_participation:
        cfg.enable_trend_participation_overlay = False
    if args.trend_sharpe60 is not None:
        cfg.trend_sharpe60_threshold = float(args.trend_sharpe60)
    if args.trend_sharpe120 is not None:
        cfg.trend_sharpe120_threshold = float(args.trend_sharpe120)
    if args.trend_positive_ratio60 is not None:
        cfg.trend_positive_ratio_60_threshold = float(args.trend_positive_ratio60)
    if args.trend_max_high_vol is not None:
        cfg.trend_max_high_vol_for_overlay = float(args.trend_max_high_vol)
    if args.trend_floor_lt25 is not None:
        cfg.trend_floor_lt25 = float(args.trend_floor_lt25)
    if args.trend_floor_lt35 is not None:
        cfg.trend_floor_lt35 = float(args.trend_floor_lt35)
    if args.trend_floor_lt50 is not None:
        cfg.trend_floor_lt50 = float(args.trend_floor_lt50)
    if args.trend_floor_lt65 is not None:
        cfg.trend_floor_lt65 = float(args.trend_floor_lt65)
    if args.trend_full_stock_low_vol:
        cfg.trend_full_stock_when_low_vol = True
    if args.trend_force_rebalance:
        cfg.trend_force_rebalance = True

    if args.disable_recovery_rerisk:
        cfg.enable_recovery_rerisk_overlay = False
    if args.recovery_dd60 is not None:
        cfg.recovery_dd60_threshold = float(args.recovery_dd60)
    if args.recovery_ret10 is not None:
        cfg.recovery_return10_threshold = float(args.recovery_ret10)
    if args.recovery_ret20 is not None:
        cfg.recovery_return20_threshold = float(args.recovery_ret20)
    if args.recovery_stock_floor is not None:
        cfg.recovery_stock_floor = float(args.recovery_stock_floor)
    if args.recovery_strong_stock_floor is not None:
        cfg.recovery_strong_stock_floor = float(args.recovery_strong_stock_floor)
    if args.recovery_no_force_rebalance:
        cfg.recovery_force_rebalance = False

    if args.enable_drawdown_guard:
        cfg.enable_soft_drawdown_guard = True
    if args.drawdown_guard_window is not None:
        cfg.drawdown_guard_window = int(args.drawdown_guard_window)
    if args.drawdown_guard_threshold_1 is not None:
        cfg.drawdown_guard_threshold_1 = float(args.drawdown_guard_threshold_1)
    if args.drawdown_guard_threshold_2 is not None:
        cfg.drawdown_guard_threshold_2 = float(args.drawdown_guard_threshold_2)

    if args.disable_tier2:
        cfg.disable_tier2_signal = True
    if args.up_strength_threshold_1 is not None:
        cfg.up_strength_bonus_threshold_1 = float(args.up_strength_threshold_1)
    if args.up_strength_threshold_2 is not None:
        cfg.up_strength_bonus_threshold_2 = float(args.up_strength_threshold_2)
    if args.up_strength_threshold_3 is not None:
        cfg.up_strength_bonus_threshold_3 = float(args.up_strength_threshold_3)
    if args.full_stock_score_threshold is not None:
        cfg.up_strength_full_stock_score_threshold = float(args.full_stock_score_threshold)
    if args.full_stock_10d_threshold is not None:
        cfg.up_strength_full_stock_10d_threshold = float(args.full_stock_10d_threshold)
    if args.full_stock_20d_threshold is not None:
        cfg.up_strength_full_stock_20d_threshold = float(args.full_stock_20d_threshold)
    if args.full_stock_high_vol_threshold is not None:
        cfg.up_strength_full_stock_high_vol_threshold = float(args.full_stock_high_vol_threshold)

    return cfg


def main() -> None:
    args = parse_args()

    if args.asset_list or args.asset_preset:
        preset_map = {
            "etf": ["QQQ", "SPY", "IWM", "DIA", "XLK", "SMH", "SOXX", "XLY", "XLF", "XLV"],
            "mega": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO"],
            "mixed": ["QQQ", "SPY", "IWM", "SMH", "SOXX", "NVDA", "MSFT", "AAPL", "TSLA"],
        }
        tickers: List[str] = []
        if args.asset_preset:
            tickers.extend(preset_map[str(args.asset_preset)])
        if args.asset_list:
            tickers.extend([x.strip().upper() for x in str(args.asset_list).split(",") if x.strip()])
        tickers = list(dict.fromkeys(tickers))

        batch_root = Path(args.result_dir or "results_xgb_trend_participation_v8_7_1_multi_asset")
        batch_root.mkdir(parents=True, exist_ok=True)

        base_args: List[str] = []
        skip_next = False
        value_args = {"--asset-list", "--asset-preset", "--target-ticker", "--result-dir"}
        for a in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if a in value_args:
                skip_next = True
                continue
            if any(a.startswith(k + "=") for k in value_args):
                continue
            base_args.append(a)

        rows: List[Dict[str, object]] = []
        for ticker in tickers:
            safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in ticker).strip("_") or "asset"
            out_dir = batch_root / safe
            cmd = [sys.executable, sys.argv[0], *base_args, "--target-ticker", ticker, "--result-dir", str(out_dir)]
            print(f"\n[BATCH] {ticker} 실행")
            proc = subprocess.run(cmd)
            row: Dict[str, object] = {"ticker": ticker, "returncode": int(proc.returncode), "result_dir": str(out_dir)}
            summary_path = out_dir / f"{safe}_xgb_trend_participation_v8_7_1_summary.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    sm = json.load(f)
                perf = sm.get("performance", {}).get("strategy_after_cost", {})
                row.update(
                    {
                        "final_capital": perf.get("final_capital"),
                        "cagr": perf.get("cagr"),
                        "mdd": perf.get("mdd"),
                        "sharpe": perf.get("sharpe"),
                        "avg_stock_weight": sm.get("average_weights", {}).get("avg_stock_weight"),
                        "turnover": sm.get("turnover", {}).get("annual_turnover_estimate"),
                        "trend_bull_rate": sm.get("structural_overlays", {}).get("trend_bull_rate"),
                        "recovery_risk_on_rate": sm.get("structural_overlays", {}).get("recovery_risk_on_rate"),
                    }
                )
            rows.append(row)

        batch_df = pd.DataFrame(rows)
        batch_path = batch_root / "multi_asset_summary.csv"
        batch_df.to_csv(batch_path, index=False, encoding="utf-8-sig")
        print("\n[MULTI-ASSET 저장 완료]")
        print(f"- {batch_path}")
        return

    cfg = apply_speed_profile(Config(), args.speed_profile)
    cfg = apply_cli_overrides(cfg, args)
    run_single_asset(cfg, args)


if __name__ == "__main__":
    main()
