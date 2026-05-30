"""
XGBoost v8 - Multi-Horizon Stage1-Gated Down-risk Volatility Timing
====================================================================

목적
- QQQ/IEF/BIL 동적 자산배분 전략
- H10/H20 정상/고변동 Stage1 모델을 앙상블
- H10/H20 하락고변동 Down-risk OVR 모델을 앙상블
- 고변동 내부 상승/하락 Stage2 방향 분류는 제거
- Stage1 고변동 확률을 1차 게이트로 사용하고, Down-risk는 방어 보조 신호로 사용
- 비용/turnover를 고려해 10거래일 단위 리밸런싱 + 긴급 리밸런싱 제한
- 고변동 라벨 quantile 정책은 고정 또는 adaptive nested validation으로 선택 가능

실행 예시
    py xgb_multi_horizon_stage1_gated_downrisk_v8.py --speed-profile balanced
    py xgb_multi_horizon_stage1_gated_downrisk_v8.py --speed-profile fast
    py xgb_multi_horizon_stage1_gated_downrisk_v8.py --speed-profile full --adaptive-label

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

    horizons: Tuple[int, int] = (10, 20)
    primary_horizon: int = 10
    min_train_rows: int = 756
    retrain_every_n_days: int = 10

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
    high_vol_weight_h10: float = 0.65
    high_vol_weight_h20: float = 0.35
    down_risk_weight_h10: float = 0.70
    down_risk_weight_h20: float = 0.30

    # Probability smoothing
    use_prob_ewma: bool = True
    prob_ewma_span: int = 7

    # Prediction thresholds for reporting
    pred_high_vol_threshold: float = 0.50
    pred_down_risk_threshold: float = 0.45

    # Allocation gate thresholds
    gate_normal_high_vol_threshold: float = 0.35
    gate_high_vol_threshold: float = 0.55
    gate_riskoff_downrisk_threshold: float = 0.45
    gate_watch_downrisk_threshold: float = 0.55

    # Base bucket allocations
    normal_stock_weight: float = 0.90
    normal_bond_weight: float = 0.07
    normal_cash_weight: float = 0.03

    watch_stock_weight: float = 0.82
    watch_bond_weight: float = 0.13
    watch_cash_weight: float = 0.05

    high_vol_stock_weight: float = 0.70
    high_vol_bond_weight: float = 0.20
    high_vol_cash_weight: float = 0.10

    risk_off_stock_weight: float = 0.50
    risk_off_bond_weight: float = 0.32
    risk_off_cash_weight: float = 0.18

    # Small continuous adjustment within bucket
    use_continuous_adjustment: bool = True
    continuous_high_vol_weight: float = 0.06
    continuous_down_risk_weight: float = 0.08
    max_continuous_stock_cut: float = 0.08

    # Trading rules
    rebalance_every_n_days: int = 10
    no_trade_band: float = 0.05
    emergency_high_vol_threshold: float = 0.75
    emergency_combined_high_vol_threshold: float = 0.60
    emergency_combined_down_threshold: float = 0.50

    # Optional small rolling allocation threshold optimization
    # 기본값 False: 속도 문제 방지. 켜도 후보 수는 작게 유지.
    use_rolling_gate_optimization: bool = False
    gate_optimize_every_n_days: int = 120
    gate_rolling_window: int = 504
    gate_min_window: int = 252
    gate_score_cagr_weight: float = 1.30
    gate_score_mdd_weight: float = 0.85
    gate_score_turnover_weight: float = 0.45

    result_dir: str = "results_xgb_multi_horizon_stage1_gated_downrisk_v8"


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
    for h in horizons:
        df[f"future_volatility_{h}d"] = ret.shift(-1).rolling(h).std().shift(-(h - 1))
        future_high = close.shift(-1).rolling(h).max().shift(-(h - 1))
        future_low = close.shift(-1).rolling(h).min().shift(-(h - 1))
        df[f"future_return_{h}d"] = close.shift(-h) / close - 1.0
        df[f"future_max_return_{h}d"] = future_high / close - 1.0
        df[f"future_min_return_{h}d"] = future_low / close - 1.0
    return df


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
    df["squeeze_on"] = (df["bb_width_20"] < df["keltner_width_20"]).astype(float)
    df["squeeze_release"] = ((df["squeeze_on"].shift(1) == 1.0) & (df["squeeze_on"] == 0.0)).astype(float)

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
        "volume_change", "volume_ratio_20", "volume_zscore_20",
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
    return df.apply(lambda row: assign_label(row, horizon, th), axis=1)


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
    down_imp_hist: List[Dict[str, float]] = []
    policy_usage: Dict[str, int] = {}

    hv_w, dn_w = ensemble_weights(cfg)

    for k, pos in enumerate(candidate_positions):
        date = all_df.index[pos]
        train_end_pos = pos - max_gap
        if train_end_pos < cfg.min_train_rows:
            continue

        need_retrain = (not models) or (last_retrain_k is None) or (k - last_retrain_k >= cfg.retrain_every_n_days)
        if need_retrain:
            train_df = all_df.iloc[:train_end_pos].copy().dropna(subset=valid_cols)
            if len(train_df) < cfg.min_train_rows:
                continue

            models = {}
            X_train = train_df[feature_cols]
            for h in cfg.horizons:
                policy, th = select_label_policy(train_df, h, feature_cols, cfg)
                labels = make_labels(train_df, h, th)
                y_high = (labels != "정상").astype(int).values
                y_down = (labels == "하락고변동").astype(int).values

                if len(np.unique(y_high)) < 2:
                    continue
                stage1_model = make_xgb_stage1(cfg, calc_scale_pos_weight(y_high))
                stage1_model.fit(X_train, y_high)
                imp1 = extract_model_importance(stage1_model, feature_cols)
                if imp1:
                    stage1_imp_hist.append(imp1)

                down_model: Optional[Pipeline] = None
                down_available = False
                if len(np.unique(y_down)) == 2 and int(y_down.sum()) >= 20:
                    down_model = make_xgb_downrisk(cfg, calc_scale_pos_weight(y_down))
                    down_model.fit(X_train, y_down)
                    down_available = True
                    impd = extract_model_importance(down_model, feature_cols)
                    if impd:
                        down_imp_hist.append(impd)

                models[h] = {
                    "stage1": stage1_model,
                    "down": down_model,
                    "down_available": down_available,
                    "thresholds": th,
                    "policy": policy,
                }
                policy_usage[f"H{h}:{policy.name}"] = policy_usage.get(f"H{h}:{policy.name}", 0) + 1

            last_retrain_k = k

        if not models:
            continue

        row_df = all_df.iloc[[pos]]
        X_now = row_df[feature_cols]
        out: Dict[str, object] = {"Date": date}

        prob_high_ens = 0.0
        prob_down_ens = 0.0
        actual_primary_label = "정상"
        actual_primary_risk = "정상"

        for h in cfg.horizons:
            if h not in models:
                continue
            m = models[h]
            stage1_model = m["stage1"]
            down_model = m["down"]
            th = m["thresholds"]
            policy = m["policy"]

            p_high = float(stage1_model.predict_proba(X_now)[0, 1])  # type: ignore[union-attr]
            if down_model is not None and bool(m["down_available"]):
                p_down = float(down_model.predict_proba(X_now)[0, 1])  # type: ignore[union-attr]
            else:
                p_down = 0.0

            actual_label_h = assign_label(all_df.iloc[pos], h, th)  # train threshold 기준 actual label
            actual_risk_h = "고변동" if actual_label_h != "정상" else "정상"

            out[f"prob_high_vol_h{h}"] = p_high
            out[f"prob_down_risk_h{h}"] = p_down
            out[f"actual_split_vol_h{h}"] = actual_label_h
            out[f"actual_risk_h{h}"] = actual_risk_h
            out[f"label_policy_h{h}"] = policy.name  # type: ignore[union-attr]

            prob_high_ens += hv_w.get(h, 0.0) * p_high
            prob_down_ens += dn_w.get(h, 0.0) * p_down

            if h == cfg.primary_horizon:
                actual_primary_label = actual_label_h
                actual_primary_risk = actual_risk_h

        prob_high_ens = float(np.clip(prob_high_ens, 0.0, 1.0))
        prob_down_ens = float(np.clip(prob_down_ens, 0.0, 1.0))
        prob_down_hv = float(np.clip(min(prob_high_ens, prob_down_ens), 0.0, 1.0))
        prob_up_proxy = float(np.clip(prob_high_ens - prob_down_hv, 0.0, 1.0))

        out.update({
            "actual_risk": actual_primary_risk,
            "actual_split_vol": actual_primary_label,
            "prob_high_vol": prob_high_ens,
            "prob_down_risk": prob_down_ens,
            "prob_normal": 1.0 - prob_high_ens,
            "prob_down_high_vol": prob_down_hv,
            "prob_up_proxy": prob_up_proxy,
            "pred_risk": "고변동" if prob_high_ens >= cfg.pred_high_vol_threshold else "정상",
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
        prob_cols = [c for c in pred_df.columns if c.startswith("prob_high_vol") or c.startswith("prob_down_risk")]
        for col in prob_cols:
            if pred_df[col].dtype.kind in "if":
                pred_df[col] = pred_df[col].ewm(span=cfg.prob_ewma_span, adjust=False).mean()
        pred_df["prob_high_vol"] = pred_df["prob_high_vol"].clip(0.0, 1.0)
        pred_df["prob_down_risk"] = pred_df["prob_down_risk"].clip(0.0, 1.0)
        pred_df["prob_normal"] = 1.0 - pred_df["prob_high_vol"]
        pred_df["prob_down_high_vol"] = np.minimum(pred_df["prob_high_vol"], pred_df["prob_down_risk"]).clip(0.0, 1.0)
        pred_df["prob_up_proxy"] = (pred_df["prob_high_vol"] - pred_df["prob_down_high_vol"]).clip(0.0, 1.0)
        pred_df["pred_risk"] = np.where(pred_df["prob_high_vol"] >= cfg.pred_high_vol_threshold, "고변동", "정상")
        pred_df["pred_split_vol"] = np.where(
            pred_df["pred_risk"] == "정상",
            "정상",
            np.where(pred_df["prob_down_risk"] >= cfg.pred_down_risk_threshold, "하락고변동", "상승고변동"),
        )

    pred_df.attrs["stage1_feature_importance_mean"] = mean_importance(stage1_imp_hist)
    pred_df.attrs["downrisk_feature_importance_mean"] = mean_importance(down_imp_hist)
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
        "name": "default_v8_gate",
    }


def classify_gate(prob_high_vol: float, prob_down_risk: float, g: Dict[str, float]) -> str:
    ph = float(np.clip(prob_high_vol, 0.0, 1.0))
    pdn = float(np.clip(prob_down_risk, 0.0, 1.0))

    if ph < g["gate_normal_high_vol_threshold"]:
        return "NORMAL"
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
    for nht in [0.30, 0.35, 0.40]:
        for hht in [0.50, 0.55, 0.60]:
            if hht <= nht:
                continue
            for rdt in [0.40, 0.45, 0.50]:
                g = dict(base)
                g["gate_normal_high_vol_threshold"] = nht
                g["gate_high_vol_threshold"] = hht
                g["gate_riskoff_downrisk_threshold"] = rdt
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


def apply_allocation(pred_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    pred_df = pred_df.copy().reset_index(drop=True)
    default_g = gate_config_from_cfg(cfg)
    grid = build_small_gate_grid(cfg)
    current_g = default_g
    usage: Dict[str, int] = {}

    prev_w: Optional[Tuple[float, float, float]] = None
    rows: List[Dict[str, object]] = []

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
        pdn = float(row["prob_down_risk"])
        emergency = (
            ph >= cfg.emergency_high_vol_threshold
            or (ph >= cfg.emergency_combined_high_vol_threshold and pdn >= cfg.emergency_combined_down_threshold)
        )
        scheduled = (i % cfg.rebalance_every_n_days == 0)
        should_rebalance = prev_w is None or scheduled or emergency

        if should_rebalance:
            w, regime = allocate_from_probs(ph, pdn, current_g, cfg, prev_w)
        else:
            w = prev_w if prev_w is not None else (cfg.normal_stock_weight, cfg.normal_bond_weight, cfg.normal_cash_weight)
            regime = classify_gate(ph, pdn, current_g)

        turnover = 0.0 if prev_w is None else sum(abs(w[j] - prev_w[j]) for j in range(3))
        gross = w[0] * row["stock_next_return"] + w[1] * row["bond_next_return"] + w[2] * row["cash_next_return"]
        cost = cfg.transaction_cost_rate * turnover
        net = gross - cost

        out = row.to_dict()
        out.update({
            "allocation_regime": regime,
            "stock_weight": float(w[0]),
            "bond_weight": float(w[1]),
            "cash_weight": float(w[2]),
            "turnover": float(turnover),
            "transaction_cost": float(cost),
            "strategy_return_gross": float(gross),
            "strategy_return_net": float(net),
            "rebalanced": bool(should_rebalance),
            "emergency_rebalance": bool(emergency),
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
        if f"actual_split_vol_h{h}" in pred_df.columns and f"prob_down_risk_h{h}" in pred_df.columns:
            y = (pred_df[f"actual_split_vol_h{h}"] == "하락고변동").astype(int).values
            p = pred_df[f"prob_down_risk_h{h}"].astype(float).clip(0.0, 1.0).values
            metrics[f"downrisk_h{h}"] = binary_cls_metrics(y, p, cfg.pred_down_risk_threshold, "하락고변동")

    y_primary = (pred_df["actual_risk"] == "고변동").astype(int).values
    p_ens = pred_df["prob_high_vol"].astype(float).clip(0.0, 1.0).values
    metrics["stage1_ensemble_vs_primary"] = binary_cls_metrics(y_primary, p_ens, cfg.pred_high_vol_threshold, "고변동")

    y_down_primary = (pred_df["actual_split_vol"] == "하락고변동").astype(int).values
    p_down_ens = pred_df["prob_down_risk"].astype(float).clip(0.0, 1.0).values
    metrics["downrisk_ensemble_vs_primary"] = binary_cls_metrics(y_down_primary, p_down_ens, cfg.pred_down_risk_threshold, "하락고변동")

    labels = ["정상", "상승고변동", "하락고변동"]
    y_true = pd.Categorical(pred_df["actual_split_vol"], categories=labels).codes
    y_pred = pd.Categorical(pred_df["pred_split_vol"], categories=labels).codes
    valid = (y_true >= 0) & (y_pred >= 0)
    metrics["final_3state_vs_primary"] = {
        "rows": int(valid.sum()),
        "accuracy": float(accuracy_score(y_true[valid], y_pred[valid])) if valid.any() else 0.0,
        "macro_f1": float(f1_score(y_true[valid], y_pred[valid], average="macro", zero_division=0)) if valid.any() else 0.0,
        "label_support": pred_df["actual_split_vol"].value_counts().to_dict(),
        "report": classification_report(y_true[valid], y_pred[valid], target_names=labels, output_dict=True, zero_division=0) if valid.any() else {},
    }
    return metrics


def build_summary(pred_df: pd.DataFrame, feature_cols: List[str], gate_usage: Dict[str, int], cfg: Config) -> Dict[str, object]:
    perf = {
        "strategy_after_cost": perf_stats(pred_df["strategy_return_net"], cfg.initial_capital),
        "strategy_gross": perf_stats(pred_df["strategy_return_gross"], cfg.initial_capital),
        "stock_buy_hold": perf_stats(pred_df["stock_next_return"], cfg.initial_capital),
        "benchmark_60_40": perf_stats(0.6 * pred_df["stock_next_return"] + 0.4 * pred_df["bond_next_return"], cfg.initial_capital),
        "static_50_30_20": perf_stats(0.5 * pred_df["stock_next_return"] + 0.3 * pred_df["bond_next_return"] + 0.2 * pred_df["cash_next_return"], cfg.initial_capital),
    }
    latest = pred_df.iloc[-1]
    return {
        "model_type": "xgb_multi_horizon_stage1_gated_downrisk_v8",
        "target_ticker": cfg.target_ticker,
        "bond_ticker": cfg.bond_ticker,
        "cash_ticker": cfg.cash_ticker,
        "config": asdict(cfg),
        "period": {"start": str(pred_df["Date"].iloc[0]), "end": str(pred_df["Date"].iloc[-1]), "rows": int(len(pred_df))},
        "feature_count": int(len(feature_cols)),
        "feature_set": "price_volume_volatility_atr_range_downside_features",
        "feature_cols": feature_cols,
        "stage1_feature_importance_mean": pred_df.attrs.get("stage1_feature_importance_mean", {}),
        "downrisk_feature_importance_mean": pred_df.attrs.get("downrisk_feature_importance_mean", {}),
        "label_policy_usage": pred_df.attrs.get("policy_usage", {}),
        "average_probabilities": {
            "avg_prob_normal": float(pred_df["prob_normal"].mean()),
            "avg_prob_high_vol": float(pred_df["prob_high_vol"].mean()),
            "avg_prob_down_risk": float(pred_df["prob_down_risk"].mean()),
            "avg_prob_down_high_vol": float(pred_df["prob_down_high_vol"].mean()),
            "avg_prob_up_proxy": float(pred_df["prob_up_proxy"].mean()),
        },
        "average_weights": {
            "avg_stock_weight": float(pred_df["stock_weight"].mean()),
            "avg_bond_weight": float(pred_df["bond_weight"].mean()),
            "avg_cash_weight": float(pred_df["cash_weight"].mean()),
            "min_stock_weight": float(pred_df["stock_weight"].min()),
            "max_stock_weight": float(pred_df["stock_weight"].max()),
        },
        "allocation_regime_distribution_pct": pred_df["allocation_regime"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "turnover": {
            "avg_daily_trade_ratio": float(pred_df["turnover"].mean()),
            "annual_turnover_estimate": float(pred_df["turnover"].mean() * 252.0),
            "total_transaction_cost_rate_sum": float(pred_df["transaction_cost"].sum()),
            "rebalance_ratio": float(pred_df["rebalanced"].mean()),
            "emergency_rebalance_ratio": float(pred_df["emergency_rebalance"].mean()),
        },
        "performance": perf,
        "classification": classification_metrics(pred_df, cfg),
        "gate_config_usage_top10": dict(sorted(gate_usage.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "latest_prediction": {
            "date": str(latest["Date"]),
            "pred_risk": str(latest["pred_risk"]),
            "pred_split_vol": str(latest["pred_split_vol"]),
            "prob_normal": round(float(latest["prob_normal"]) * 100, 2),
            "prob_high_vol": round(float(latest["prob_high_vol"]) * 100, 2),
            "prob_down_risk": round(float(latest["prob_down_risk"]) * 100, 2),
            "prob_down_high_vol": round(float(latest["prob_down_high_vol"]) * 100, 2),
            "prob_up_proxy": round(float(latest["prob_up_proxy"]) * 100, 2),
            "allocation_regime": str(latest["allocation_regime"]),
            "target_allocation": {
                "stock": round(float(latest["stock_weight"]) * 100, 2),
                "bond": round(float(latest["bond_weight"]) * 100, 2),
                "cash": round(float(latest["cash_weight"]) * 100, 2),
            },
        },
    }


def print_summary(summary: Dict[str, object]) -> None:
    p = summary["performance"]
    w = summary["average_weights"]
    t = summary["turnover"]
    cls = summary["classification"]
    print("\n==============================")
    print("XGBoost v8 Multi-Horizon Stage1-Gated Down-risk 결과 요약")
    print("H10/H20 Stage1 + Down-risk OVR Ensemble + Cost-aware Allocation")
    print("==============================")
    print(f"기간: {summary['period']['start']} ~ {summary['period']['end']}")
    print(f"거래일 수: {summary['period']['rows']}")
    print(f"피처 수: {summary['feature_count']}")
    print(f"평균 주식 비중: {w['avg_stock_weight'] * 100:.2f}%")
    print(f"평균 채권 비중: {w['avg_bond_weight'] * 100:.2f}%")
    print(f"평균 현금 비중: {w['avg_cash_weight'] * 100:.2f}%")
    print(f"연간 교체율 추정: {t['annual_turnover_estimate'] * 100:.2f}%")
    print(f"리밸런싱 발생 비율: {t['rebalance_ratio'] * 100:.2f}%")
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
    for key in ["stage1_h10", "stage1_h20", "stage1_ensemble_vs_primary", "downrisk_h10", "downrisk_h20", "downrisk_ensemble_vs_primary"]:
        if key in cls:
            m = cls[key]
            print(f"{key:30s} | ROC {m['roc_auc']} | PR {m['pr_auc']} | F1 {m['f1']:.4f} | Recall {m['recall']:.4f}")

    print("\n[최신 예측]")
    print(json.dumps(summary["latest_prediction"], ensure_ascii=False, indent=2))


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
        cfg.result_dir = "results_xgb_v8_fast"
    elif profile == "balanced":
        cfg.retrain_every_n_days = 10
        cfg.stage1_n_estimators = 150
        cfg.down_n_estimators = 100
        cfg.use_adaptive_label_policy = False
        cfg.use_rolling_gate_optimization = False
        cfg.result_dir = "results_xgb_v8_balanced"
    elif profile == "full":
        cfg.retrain_every_n_days = 10
        cfg.stage1_n_estimators = 200
        cfg.down_n_estimators = 140
        cfg.use_adaptive_label_policy = True
        cfg.use_rolling_gate_optimization = False
        cfg.result_dir = "results_xgb_v8_full_adaptive_label"
    else:
        raise ValueError(f"알 수 없는 speed profile: {profile}")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost v8 Multi-Horizon Stage1-Gated Down-risk Allocation")
    parser.add_argument("--speed-profile", choices=["fast", "balanced", "full"], default="balanced")
    parser.add_argument("--adaptive-label", action="store_true", help="라벨 quantile 정책을 nested validation으로 선택")
    parser.add_argument("--rolling-gate-opt", action="store_true", help="작은 grid로 allocation gate threshold를 rolling 최적화")
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--retrain-every", type=int, default=None)
    parser.add_argument("--result-dir", type=str, default=None)
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

    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 데이터 다운로드")
    target = download_ohlcv(cfg.target_ticker, cfg.start_date, cfg.end_date)
    bond_close = download_close(cfg.bond_ticker, cfg.start_date, cfg.end_date)
    try:
        cash_close = download_close(cfg.cash_ticker, cfg.start_date, cfg.end_date)
    except Exception:
        cash_close = pd.Series(index=target.index, data=np.nan, name=cfg.cash_ticker)

    print("[2/5] 피처 생성")
    df, feature_cols = build_features(target, cfg.horizons)
    df["stock_next_return"] = df["Close"].pct_change().shift(-1)
    df["bond_next_return"] = bond_close.pct_change().shift(-1).reindex(df.index).fillna(0.0)
    df["cash_next_return"] = cash_close.pct_change().shift(-1).reindex(df.index).fillna(0.0)
    print(f"    피처 수: {len(feature_cols)}")
    print(f"    horizons: {cfg.horizons}")
    print(f"    adaptive_label: {cfg.use_adaptive_label_policy}")
    print(f"    rolling_gate_opt: {cfg.use_rolling_gate_optimization}")

    print("[3/5] Walk-forward H10/H20 Stage1 + Down-risk OVR 예측")
    pred_raw = run_walk_forward(df, feature_cols, cfg)

    print("[4/5] 배분/백테스트")
    pred_df, gate_usage = apply_allocation(pred_raw, cfg)
    pred_df.attrs.update(pred_raw.attrs)

    print("[5/5] 결과 저장")
    summary = build_summary(pred_df, feature_cols, gate_usage, cfg)

    pred_path = result_dir / "qqq_xgb_v8_predictions.csv"
    summary_path = result_dir / "qqq_xgb_v8_summary.json"
    latest_path = result_dir / "qqq_xgb_v8_latest.json"
    importance_stage1_path = result_dir / "qqq_xgb_v8_stage1_feature_importance.csv"
    importance_down_path = result_dir / "qqq_xgb_v8_downrisk_feature_importance.csv"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary["latest_prediction"], f, ensure_ascii=False, indent=2)

    pd.Series(summary.get("stage1_feature_importance_mean", {}), name="importance").to_csv(importance_stage1_path, encoding="utf-8-sig")
    pd.Series(summary.get("downrisk_feature_importance_mean", {}), name="importance").to_csv(importance_down_path, encoding="utf-8-sig")

    print_summary(summary)
    print("\n[저장 완료]")
    print(f"- {pred_path}")
    print(f"- {summary_path}")
    print(f"- {latest_path}")
    print(f"- {importance_stage1_path}")
    print(f"- {importance_down_path}")


if __name__ == "__main__":
    main()
