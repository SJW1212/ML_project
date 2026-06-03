"""
XGBoost ATR-aware H10 v7 - Stage1-gated Down-risk Allocation
======================================================

목적
- QQQ 가격/거래량/변동성 피처 기반으로 향후 10거래일 고변동 위험을 예측
- 1단계: 정상 vs 고변동
- 보조 모델: 하락고변동 vs 비하락고변동 one-vs-rest 직접 예측
- 고변동 내부 상승/하락 방향 분류는 제거 또는 약화
- Stage1 고변동 확률을 1차 게이트로 사용하고, Down-risk OVR은 고변동/주의 구간에서만 강하게 반영
- 평가 지표는 Stage1 / Down-risk OVR / Final Cascade로 분리

실행
    py xgb_atr_h10_stage1_gated_downrisk_v7.py

필요 패키지
    pip install pandas numpy yfinance scikit-learn xgboost

주의
- 라벨 생성에는 미래 20거래일 수익률/변동성/최대상승/최대하락을 사용합니다.
- 모델 입력 피처에는 미래 컬럼을 사용하지 않습니다.
- walk-forward 학습 시 horizon_gap_days만큼 purge gap을 둡니다.
- 라벨 threshold는 매 재학습 시점의 train 구간에서만 계산합니다.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    raise ImportError(
        "xgboost가 설치되어 있지 않습니다. 먼저 `pip install xgboost`를 실행하세요."
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 0. CONFIG
# ============================================================

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

    risk_horizon_days: int = 10
    horizon_gap_days: int = 10
    min_train_rows: int = 756
    retrain_every_n_days: int = 5

    random_state: int = 42
    n_jobs: int = 1

    # XGBoost stage 1: normal vs high-vol
    stage1_n_estimators: int = 250
    stage1_learning_rate: float = 0.025
    stage1_max_depth: int = 3
    stage1_min_child_weight: float = 10.0
    stage1_subsample: float = 0.85
    stage1_colsample_bytree: float = 0.80
    stage1_reg_lambda: float = 8.0
    stage1_reg_alpha: float = 0.1

    # XGBoost auxiliary model: down-high-vol one-vs-rest
    stage2_n_estimators: int = 180
    stage2_learning_rate: float = 0.030
    stage2_max_depth: int = 2
    stage2_min_child_weight: float = 6.0
    stage2_subsample: float = 0.90
    stage2_colsample_bytree: float = 0.85
    stage2_reg_lambda: float = 10.0
    stage2_reg_alpha: float = 0.2

    # probability smoothing
    use_prob_ewma: bool = True
    prob_ewma_span: int = 7

    # allocation
    base_stock_weight: float = 0.91
    down_high_vol_penalty: float = 0.50
    up_high_vol_penalty: float = -0.05
    min_stock_weight: float = 0.30
    max_stock_weight: float = 0.92
    base_cash_ratio_in_defensive: float = 0.18
    down_high_vol_cash_sensitivity: float = 0.55
    up_high_vol_cash_sensitivity: float = -0.05
    min_cash_ratio_in_defensive: float = 0.10
    max_cash_ratio_in_defensive: float = 0.80
    weight_smoothing: float = 0.00

    # improved allocation v4
    # - prob_down_high_vol은 방어 신호로 강하게 반영
    # - prob_up_high_vol은 위험 차감 또는 기회 보정으로 약하게 반영
    # - 낮은 위험 구간은 비중 변화를 줄이고, 높은 위험 구간에서만 빠르게 축소
    risk_high_vol_weight: float = 0.70
    risk_down_high_vol_weight: float = 0.50
    risk_up_high_vol_weight: float = -0.05
    opportunity_weight: float = 0.05
    down_soft_threshold: float = 0.30
    down_hard_threshold: float = 0.40
    down_extreme_threshold: float = 0.55
    soft_stock_cap: float = 0.80
    hard_stock_cap: float = 0.72
    extreme_stock_cap: float = 0.62
    soft_risk_stock_multiplier: float = 0.80
    hard_risk_stock_multiplier: float = 0.70
    no_trade_band: float = 0.05
    max_daily_weight_change: float = 0.00
    pred_high_vol_threshold: float = 0.50
    pred_down_high_vol_threshold: float = 0.35
    pred_up_high_vol_threshold: float = 0.20
    rebalance_every_n_days: int = 10

    # --------------------------------------------------------
    # Stage1-gated allocation parameters
    # --------------------------------------------------------
    # Stage1이 정상에 가까우면 Down-risk 오탐을 약하게 보거나 무시한다.
    gate_normal_high_vol_threshold: float = 0.35
    gate_high_vol_threshold: float = 0.55
    gate_watch_downrisk_threshold: float = 0.50
    gate_riskoff_downrisk_threshold: float = 0.40

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

    # emergency rebalance thresholds for Stage1-gated version
    # Down-risk 단독이 아니라 Stage1 고변동과 같이 강할 때만 긴급 반응한다.
    emergency_down_threshold: float = 0.60
    emergency_combined_down_threshold: float = 0.50
    emergency_combined_high_vol_threshold: float = 0.55
    emergency_high_vol_threshold: float = 0.75

    # rolling allocation optimization
    use_rolling_allocation_optimization: bool = True
    allocation_rolling_window: int = 756
    allocation_min_window: int = 252
    allocation_optimize_every_n_days: int = 60
    opt_cagr_weight: float = 1.35
    opt_mdd_weight: float = 0.90
    opt_turnover_weight: float = 0.60
    opt_target_avg_stock_weight: float = 0.80
    opt_low_stock_penalty_weight: float = 0.40

    result_dir: str = "results_xgb_atr_h10_stage1_gated_downrisk_v7"


CFG = Config()


# ============================================================
# 1. DATA
# ============================================================

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance multi-index column을 단일 column으로 정리."""
    if isinstance(df.columns, pd.MultiIndex):
        # 단일 ticker 다운로드 시 보통 level 0이 OHLCV입니다.
        if len(df.columns.get_level_values(0).unique()) <= 6:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    return df


def download_ohlcv(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"{ticker} 데이터를 다운로드하지 못했습니다.")
    df = _flatten_yf_columns(df).copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker} 데이터에 필요한 컬럼이 없습니다: {missing}")
    df = df[required]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def download_close(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
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
    """최근 window 안에서 현재 값의 percentile rank. 과거 데이터만 사용."""
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
    """rolling log-price linear slope. 속도보다 안정성을 위해 numpy polyfit 사용."""
    x = np.arange(window, dtype=float)

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        ly = np.log(np.maximum(y, 1e-12))
        return float(np.polyfit(x, ly, 1)[0])

    return close.rolling(window, min_periods=window).apply(_slope, raw=True)


def add_future_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    close = df["Close"]
    ret = df["daily_return"]

    # t 기준 t+1 ~ t+h의 일간 수익률 표준편차
    df["future_volatility_20d"] = ret.shift(-1).rolling(horizon).std().shift(-(horizon - 1))

    future_high = close.shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    future_low = close.shift(-1).rolling(horizon).min().shift(-(horizon - 1))

    df["future_return_20d"] = close.shift(-horizon) / close - 1.0
    df["future_max_return_20d"] = future_high / close - 1.0
    df["future_min_return_20d"] = future_low / close - 1.0
    return df


def build_features(ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = ohlcv.copy()
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"].replace(0, np.nan)

    # Basic returns
    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))

    for w in [3, 5, 10, 20, 60, 120]:
        df[f"return_{w}d"] = close / close.shift(w) - 1.0

    # Moving average gaps
    for w in [5, 10, 20, 60, 120, 200]:
        ma = close.rolling(w).mean()
        df[f"ma_{w}"] = ma
        df[f"price_ma_{w}_gap"] = close / ma - 1.0

    df["ma_gap_5_20"] = df["ma_5"] / df["ma_20"] - 1.0
    df["ma_gap_20_60"] = df["ma_20"] / df["ma_60"] - 1.0
    df["ma_gap_60_120"] = df["ma_60"] / df["ma_120"] - 1.0
    df["ma_gap_50_200"] = close.rolling(50).mean() / close.rolling(200).mean() - 1.0

    df["trend_slope_20"] = calc_trend_slope(close, 20)
    df["trend_slope_60"] = calc_trend_slope(close, 60)
    df["ma200_slope_60"] = calc_trend_slope(df["ma_200"], 60)

    # Positive / large move ratios
    up = (df["daily_return"] > 0).astype(float)
    large_down = (df["daily_return"] <= -0.02).astype(float)
    large_up = (df["daily_return"] >= 0.02).astype(float)
    for w in [20, 60]:
        df[f"positive_return_ratio_{w}"] = up.rolling(w).mean()
    df["large_down_day_ratio_20"] = large_down.rolling(20).mean()
    df["large_up_day_ratio_20"] = large_up.rolling(20).mean()

    # Drawdown and price position
    for w in [20, 60, 120]:
        roll_high = close.rolling(w).max()
        roll_low = close.rolling(w).min()
        denom = (roll_high - roll_low).replace(0, np.nan)
        df[f"drawdown_{w}"] = close / roll_high - 1.0
        if w in [20, 60]:
            df[f"price_position_{w}"] = (close - roll_low) / denom
        if w in [20, 60]:
            df[f"close_to_{w}d_high"] = close / roll_high - 1.0

    # Volume
    df["volume_change"] = volume.pct_change()
    volume_ma20 = volume.rolling(20).mean()
    volume_std20 = volume.rolling(20).std()
    df["volume_ratio_20"] = volume / volume_ma20
    df["volume_zscore_20"] = (volume - volume_ma20) / volume_std20.replace(0, np.nan)

    # ATR / True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["true_range"] = tr
    df["true_range_pct"] = tr / close
    for w in [14, 20, 60]:
        df[f"atr_{w}"] = tr.rolling(w).mean()
        df[f"atr_pct_{w}"] = df[f"atr_{w}"] / close
    df["atr_ratio_14_60"] = df["atr_14"] / df["atr_60"]
    df["atr_ratio_20_60"] = df["atr_20"] / df["atr_60"]
    df["atr_accel_5"] = df["atr_14"] / df["atr_14"].shift(5) - 1.0
    df["atr_rank_252"] = rolling_rank_last(df["atr_pct_20"], 252)

    # Range-based volatility estimators
    log_hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(close / open_).replace([np.inf, -np.inf], np.nan)
    log_oc = np.log(open_ / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    log_cc = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
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

        # Yang-Zhang approximation
        k = 0.34 / (1.34 + (w + 1.0) / max(w - 1.0, 1.0))
        yz_var = (
            log_oc.rolling(w).var()
            + k * log_co.rolling(w).var()
            + (1.0 - k) * rs_var.rolling(w).mean()
        )
        df[f"yang_zhang_vol_{w}"] = np.sqrt(yz_var.clip(lower=0))

    df["realized_vol_ratio_20_60"] = df["realized_vol_20"] / df["realized_vol_60"]
    df["parkinson_vol_ratio_20_60"] = df["parkinson_vol_20"] / df["parkinson_vol_60"]
    df["yang_zhang_vol_ratio_20_60"] = df["yang_zhang_vol_20"] / df["yang_zhang_vol_60"]
    df["vol_of_vol_20"] = df["realized_vol_20"].rolling(20).std()

    # Downside risk
    downside_return = df["daily_return"].clip(upper=0)
    df["downside_vol_20"] = downside_return.rolling(20).std()
    df["downside_vol_60"] = downside_return.rolling(60).std()
    df["semi_vol_20"] = np.sqrt((downside_return ** 2).rolling(20).mean())
    dd20 = close / close.rolling(20).max() - 1.0
    dd60 = close / close.rolling(60).max() - 1.0
    df["ulcer_index_20"] = np.sqrt((dd20 ** 2).rolling(20).mean())
    df["ulcer_index_60"] = np.sqrt((dd60 ** 2).rolling(60).mean())
    df["ulcer_rank_252"] = rolling_rank_last(df["ulcer_index_20"], 252)

    # Volatility compression / expansion
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_width_20"] = (4.0 * std20) / ma20
    df["bb_width_rank_252"] = rolling_rank_last(df["bb_width_20"], 252)
    ema20 = close.ewm(span=20, adjust=False).mean()
    df["keltner_width_20"] = (4.0 * df["atr_20"]) / ema20
    df["squeeze_on"] = (df["bb_width_20"] < df["keltner_width_20"]).astype(float)
    df["squeeze_release"] = ((df["squeeze_on"].shift(1) == 1.0) & (df["squeeze_on"] == 0.0)).astype(float)

    # Future targets
    df = add_future_targets(df, CFG.risk_horizon_days)

    # Feature columns: 현재 시점에 알 수 있는 값만 포함
    feature_cols = [
        # 기존 V1 32개
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
        # 추세/ATR-aware 라벨 및 모델 보조 피처
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
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols


# ============================================================
# 3. LABEL DESIGN: ATR-aware split-vol
# ============================================================

def compute_thresholds(train_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "vol_q80": float(train_df["future_volatility_20d"].quantile(0.80)),
        "down_q15": float(train_df["future_min_return_20d"].quantile(0.15)),
        "down_q20": float(train_df["future_min_return_20d"].quantile(0.20)),
        "down_q25": float(train_df["future_min_return_20d"].quantile(0.25)),
        "down_q30": float(train_df["future_min_return_20d"].quantile(0.30)),
        "up_q65": float(train_df["future_max_return_20d"].quantile(0.65)),
        "up_q70": float(train_df["future_max_return_20d"].quantile(0.70)),
        "up_q75": float(train_df["future_max_return_20d"].quantile(0.75)),
        "up_q80": float(train_df["future_max_return_20d"].quantile(0.80)),
    }


def assign_atr_aware_label(row: pd.Series, th: Dict[str, float]) -> str:
    """
    normal / 상승고변동 / 하락고변동 라벨 생성.
    - 현재 ATR regime과 방향 압력을 기준으로 미래 충격 threshold를 조정
    - severe down shock는 방어 목적상 우선 분류
    """
    future_vol = row["future_volatility_20d"]
    future_max_ret = row["future_max_return_20d"]
    future_min_ret = row["future_min_return_20d"]
    future_ret = row["future_return_20d"]

    atr_rank = row.get("atr_rank_252", np.nan)
    atr_ratio = row.get("atr_ratio_20_60", np.nan)
    return_20d = row.get("return_20d", np.nan)
    drawdown_60 = row.get("drawdown_60", np.nan)
    price_position_60 = row.get("price_position_60", np.nan)
    positive_ratio_20 = row.get("positive_return_ratio_20", np.nan)
    large_down_ratio_20 = row.get("large_down_day_ratio_20", np.nan)
    ulcer_rank = row.get("ulcer_rank_252", np.nan)
    bb_rank = row.get("bb_width_rank_252", np.nan)

    # nan-safe boolean
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

    down_threshold = th["down_q20"]
    up_threshold = th["up_q80"]

    # ATR regime별 threshold 조정
    if atr_high and atr_expanding and down_pressure_now:
        down_threshold = th["down_q30"]
        up_threshold = th["up_q75"]
    elif atr_high and atr_expanding and up_pressure_now:
        down_threshold = th["down_q15"]
        up_threshold = th["up_q70"]
    elif atr_extreme:
        down_threshold = th["down_q25"]
        up_threshold = th["up_q75"]
    elif atr_compressed and squeeze_or_breakout:
        down_threshold = th["down_q25"]
        up_threshold = th["up_q70"]

    is_high_vol = (
        future_vol >= th["vol_q80"]
        or future_min_ret <= down_threshold
        or future_max_ret >= up_threshold
    )

    if not is_high_vol:
        return "정상"

    severe_down = future_min_ret <= th["down_q20"]
    if severe_down and not up_pressure_now:
        return "하락고변동"

    up_high_vol = (
        future_max_ret >= up_threshold
        and future_ret > 0
        and future_min_ret > down_threshold
    )
    if up_high_vol:
        return "상승고변동"

    if future_min_ret <= down_threshold:
        return "하락고변동"

    # 애매한 고변동은 max/min 충격 크기 비교
    if abs(future_max_ret) >= abs(future_min_ret):
        return "상승고변동"
    return "하락고변동"


def make_labels(df: pd.DataFrame, th: Dict[str, float]) -> pd.Series:
    return df.apply(lambda row: assign_atr_aware_label(row, th), axis=1)


# ============================================================
# 4. MODEL
# ============================================================

def calc_scale_pos_weight(y_binary: np.ndarray) -> float:
    pos = float(np.sum(y_binary == 1))
    neg = float(np.sum(y_binary == 0))
    if pos <= 0 or neg <= 0:
        return 1.0
    return max(0.1, min(20.0, neg / pos))


def make_xgb_stage1(cfg: Config, scale_pos_weight: float) -> Pipeline:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=cfg.stage1_n_estimators,
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
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def make_xgb_stage2(cfg: Config, scale_pos_weight: float) -> Pipeline:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=cfg.stage2_n_estimators,
        learning_rate=cfg.stage2_learning_rate,
        max_depth=cfg.stage2_max_depth,
        min_child_weight=cfg.stage2_min_child_weight,
        subsample=cfg.stage2_subsample,
        colsample_bytree=cfg.stage2_colsample_bytree,
        reg_lambda=cfg.stage2_reg_lambda,
        reg_alpha=cfg.stage2_reg_alpha,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


# ============================================================
# 5. ALLOCATION / BACKTEST
# ============================================================



def _normalize_weight_tuple(stock: float, bond: float, cash: float) -> Tuple[float, float, float]:
    """비중 합계를 1로 정규화한다."""
    vals = np.asarray([stock, bond, cash], dtype=float)
    vals = np.clip(vals, 0.0, 1.0)
    total = float(vals.sum())
    if total <= 0:
        return 1.0, 0.0, 0.0
    vals = vals / total
    return float(vals[0]), float(vals[1]), float(vals[2])


def classify_stage1_gated_bucket(
    prob_high_vol: float,
    prob_down_risk: float,
    cfg_dict: Dict[str, float],
) -> str:
    """
    Stage1 고변동 확률을 1차 게이트로 사용한다.

    핵심 의도
    - Stage1이 정상에 가까우면 높은 주식 비중 유지
    - Stage1이 주의/고변동일 때만 Down-risk를 강하게 반영
    - Down-risk 단독 오탐으로 과도하게 방어하지 않음
    """
    ph = float(np.clip(prob_high_vol, 0.0, 1.0))
    pdn = float(np.clip(prob_down_risk, 0.0, 1.0))

    if ph < cfg_dict["gate_normal_high_vol_threshold"]:
        return "NORMAL"

    if ph < cfg_dict["gate_high_vol_threshold"]:
        if pdn >= cfg_dict["gate_watch_downrisk_threshold"]:
            return "HIGH_VOL"
        return "WATCH"

    if pdn >= cfg_dict["gate_riskoff_downrisk_threshold"]:
        return "RISK_OFF"
    return "HIGH_VOL"


def build_allocation_grid(cfg: Config) -> List[Dict[str, float]]:
    """
    Stage1-gated allocation grid.

    정상 포트폴리오는 별도 최적화 가능하다는 전제에서,
    여기서는 NORMAL/WATCH/HIGH_VOL/RISK_OFF 상태별 비중과 gate threshold를
    walk-forward 중 과거 구간으로만 선택한다.
    """
    grid: List[Dict[str, float]] = []

    normal_stock_list = [0.88, 0.90, 0.92]
    watch_stock_list = [0.80, 0.82, 0.85]
    high_stock_list = [0.65, 0.70, 0.75]
    riskoff_stock_list = [0.45, 0.50, 0.55]

    normal_hv_threshold_list = [0.30, 0.35, 0.40]
    high_hv_threshold_list = [0.50, 0.55, 0.60]
    riskoff_down_threshold_list = [0.40, 0.45, 0.50]
    watch_down_threshold_list = [0.50, 0.55]
    no_trade_list = [0.04, 0.05, 0.06]

    idx = 0
    for ns in normal_stock_list:
        for ws in watch_stock_list:
            if ws >= ns:
                continue
            for hs in high_stock_list:
                if hs >= ws:
                    continue
                for ros in riskoff_stock_list:
                    if ros >= hs:
                        continue
                    for nht in normal_hv_threshold_list:
                        for hht in high_hv_threshold_list:
                            if hht <= nht:
                                continue
                            for rdt in riskoff_down_threshold_list:
                                for wdt in watch_down_threshold_list:
                                    for ntb in no_trade_list:
                                        grid.append({
                                            "name": f"stage1_gated_cfg_{idx:04d}",
                                            "gate_normal_high_vol_threshold": nht,
                                            "gate_high_vol_threshold": hht,
                                            "gate_watch_downrisk_threshold": wdt,
                                            "gate_riskoff_downrisk_threshold": rdt,
                                            "normal_stock_weight": ns,
                                            "normal_bond_weight": max(0.0, (1.0 - ns) * 0.70),
                                            "normal_cash_weight": max(0.0, (1.0 - ns) * 0.30),
                                            "watch_stock_weight": ws,
                                            "watch_bond_weight": max(0.0, (1.0 - ws) * 0.72),
                                            "watch_cash_weight": max(0.0, (1.0 - ws) * 0.28),
                                            "high_vol_stock_weight": hs,
                                            "high_vol_bond_weight": max(0.0, (1.0 - hs) * 0.68),
                                            "high_vol_cash_weight": max(0.0, (1.0 - hs) * 0.32),
                                            "risk_off_stock_weight": ros,
                                            "risk_off_bond_weight": max(0.0, (1.0 - ros) * 0.62),
                                            "risk_off_cash_weight": max(0.0, (1.0 - ros) * 0.38),
                                            "no_trade_band": ntb,
                                            "max_daily_weight_change": 0.0,
                                        })
                                        idx += 1
    return grid


def default_alloc_config(cfg: Config) -> Dict[str, float]:
    return {
        "name": "default_stage1_gated_downrisk_h10_v7",
        "gate_normal_high_vol_threshold": cfg.gate_normal_high_vol_threshold,
        "gate_high_vol_threshold": cfg.gate_high_vol_threshold,
        "gate_watch_downrisk_threshold": cfg.gate_watch_downrisk_threshold,
        "gate_riskoff_downrisk_threshold": cfg.gate_riskoff_downrisk_threshold,
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
        "max_daily_weight_change": cfg.max_daily_weight_change,
    }


def allocate_from_probs(
    prob_high_vol: float,
    prob_up_proxy: float,
    prob_down_risk: float,
    cfg_dict: Dict[str, float],
    prev_weights: Optional[Tuple[float, float, float]] = None,
    smoothing: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Stage1-gated allocation.

    기존 v5는 prob_high_vol/prob_down_risk를 연속 함수에 넣어 비중을 계속 흔들었다.
    v7은 Stage1 고변동 확률로 먼저 상태를 나누고, Down-risk는 주의/고변동 상태에서만
    방어 강도를 높이는 게이트로 사용한다.
    """
    bucket = classify_stage1_gated_bucket(prob_high_vol, prob_down_risk, cfg_dict)

    if bucket == "NORMAL":
        target = _normalize_weight_tuple(
            cfg_dict["normal_stock_weight"],
            cfg_dict["normal_bond_weight"],
            cfg_dict["normal_cash_weight"],
        )
    elif bucket == "WATCH":
        target = _normalize_weight_tuple(
            cfg_dict["watch_stock_weight"],
            cfg_dict["watch_bond_weight"],
            cfg_dict["watch_cash_weight"],
        )
    elif bucket == "HIGH_VOL":
        target = _normalize_weight_tuple(
            cfg_dict["high_vol_stock_weight"],
            cfg_dict["high_vol_bond_weight"],
            cfg_dict["high_vol_cash_weight"],
        )
    else:
        target = _normalize_weight_tuple(
            cfg_dict["risk_off_stock_weight"],
            cfg_dict["risk_off_bond_weight"],
            cfg_dict["risk_off_cash_weight"],
        )

    if prev_weights is not None:
        # 전체 비중 차이가 작으면 거래하지 않는다.
        total_delta = sum(abs(target[i] - prev_weights[i]) for i in range(3))
        if total_delta < cfg_dict["no_trade_band"]:
            return prev_weights

        max_chg = cfg_dict.get("max_daily_weight_change", 0.0)
        if max_chg is not None and max_chg > 0:
            stock = float(np.clip(target[0], prev_weights[0] - max_chg, prev_weights[0] + max_chg))
            remaining = 1.0 - stock
            defensive_target = target[1] + target[2]
            if defensive_target <= 0:
                return stock, 0.0, 0.0
            bond = remaining * target[1] / defensive_target
            cash = remaining * target[2] / defensive_target
            target = _normalize_weight_tuple(stock, bond, cash)

    if prev_weights is not None and smoothing > 0:
        target = tuple(
            smoothing * float(prev_weights[i]) + (1.0 - smoothing) * float(target[i])
            for i in range(3)
        )
        target = _normalize_weight_tuple(*target)

    return target


def perf_stats(returns: pd.Series, initial_capital: float = 100_000_000) -> Dict[str, float]:
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
        "mdd": float(mdd),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
    }


def simulate_with_config(pred_df: pd.DataFrame, cfg_dict: Dict[str, float], cfg: Config) -> Dict[str, float]:
    prev_w = None
    rets = []
    turnovers = []
    stock_weights = []

    for _, row in pred_df.iterrows():
        w = allocate_from_probs(
            row["prob_high_vol"],
            row.get("prob_up_proxy", row.get("prob_up_high_vol", 0.0)),
            row.get("prob_down_risk", row.get("prob_down_high_vol", 0.0)),
            cfg_dict,
            prev_weights=prev_w,
            smoothing=cfg.weight_smoothing,
        )
        if prev_w is None:
            turnover = 0.0
        else:
            turnover = abs(w[0] - prev_w[0]) + abs(w[1] - prev_w[1]) + abs(w[2] - prev_w[2])

        gross = w[0] * row["stock_next_return"] + w[1] * row["bond_next_return"] + w[2] * row["cash_next_return"]
        net = gross - cfg.transaction_cost_rate * turnover
        rets.append(net)
        turnovers.append(turnover)
        stock_weights.append(w[0])
        prev_w = w

    stats = perf_stats(pd.Series(rets), cfg.initial_capital)
    stats["avg_turnover"] = float(np.mean(turnovers)) if turnovers else 0.0
    stats["avg_stock_weight"] = float(np.mean(stock_weights)) if stock_weights else 0.0
    return stats


def score_config(stats: Dict[str, float], cfg: Config, avg_stock_weight: Optional[float] = None) -> float:
    cagr = stats.get("cagr", 0.0)
    mdd = abs(stats.get("mdd", 0.0))
    turnover = stats.get("avg_turnover", 0.0) * 252.0
    avg_stock = stats.get("avg_stock_weight", avg_stock_weight if avg_stock_weight is not None else cfg.opt_target_avg_stock_weight)

    # 목표 평균 주식 비중 근처를 선호.
    # 너무 낮으면 CAGR이 무너지고, 너무 높으면 MDD가 커지는 경향을 제어한다.
    target = cfg.opt_target_avg_stock_weight
    stock_penalty = 0.0
    if avg_stock < target - 0.06:
        stock_penalty += cfg.opt_low_stock_penalty_weight * (target - 0.06 - avg_stock) ** 2
    if avg_stock > target + 0.04:
        stock_penalty += 1.25 * (avg_stock - target - 0.04) ** 2

    score = (
        cfg.opt_cagr_weight * cagr
        - cfg.opt_mdd_weight * mdd
        - cfg.opt_turnover_weight * turnover
        - stock_penalty
    )
    return float(score)



def apply_rolling_allocation(pred_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    pred_df = pred_df.copy().reset_index(drop=True)
    grid = build_allocation_grid(cfg)
    default_cfg = default_alloc_config(cfg)
    current_cfg = default_cfg
    cfg_usage: Dict[str, int] = {}

    prev_w: Optional[Tuple[float, float, float]] = None
    rows = []

    for i, row in pred_df.iterrows():
        if cfg.use_rolling_allocation_optimization and i >= cfg.allocation_min_window and i % cfg.allocation_optimize_every_n_days == 0:
            start = max(0, i - cfg.allocation_rolling_window)
            hist = pred_df.iloc[start:i].copy()
            best_score = -np.inf
            best_cfg = current_cfg

            for cand in grid:
                stats = simulate_with_config(hist, cand, cfg)
                s = score_config(stats, cfg)
                if s > best_score:
                    best_score = s
                    best_cfg = cand
            current_cfg = best_cfg

        prob_down_risk_now = float(row.get("prob_down_risk", 0.0))
        prob_high_vol_now = float(row.get("prob_high_vol", 0.0))
        emergency_rebalance = (
            prob_high_vol_now >= cfg.emergency_high_vol_threshold
            or (
                prob_high_vol_now >= cfg.emergency_combined_high_vol_threshold
                and prob_down_risk_now >= cfg.emergency_combined_down_threshold
            )
            or (
                prob_high_vol_now >= cfg.pred_high_vol_threshold
                and prob_down_risk_now >= cfg.emergency_down_threshold
            )
        )
        scheduled_rebalance = (i % cfg.rebalance_every_n_days == 0)
        should_rebalance = (prev_w is None) or scheduled_rebalance or emergency_rebalance

        if should_rebalance:
            w = allocate_from_probs(
                float(row["prob_high_vol"]),
                float(row.get("prob_up_proxy", row.get("prob_up_high_vol", 0.0))),
                float(row.get("prob_down_risk", row.get("prob_down_high_vol", 0.0))),
                current_cfg,
                prev_weights=prev_w,
                smoothing=cfg.weight_smoothing,
            )
        else:
            w = prev_w if prev_w is not None else (cfg.base_stock_weight, 1.0 - cfg.base_stock_weight, 0.0)

        if prev_w is None:
            turnover = 0.0
        else:
            turnover = abs(w[0] - prev_w[0]) + abs(w[1] - prev_w[1]) + abs(w[2] - prev_w[2])

        gross = w[0] * row["stock_next_return"] + w[1] * row["bond_next_return"] + w[2] * row["cash_next_return"]
        cost = cfg.transaction_cost_rate * turnover
        net = gross - cost

        allocation_regime = classify_stage1_gated_bucket(
            float(row["prob_high_vol"]),
            float(row.get("prob_down_risk", 0.0)),
            current_cfg,
        )

        out = row.to_dict()
        out.update({
            "allocation_regime": allocation_regime,
            "stock_weight": w[0],
            "bond_weight": w[1],
            "cash_weight": w[2],
            "turnover": turnover,
            "transaction_cost": cost,
            "strategy_return_gross": gross,
            "strategy_return_net": net,
            "allocation_config": current_cfg["name"],
            "rebalanced": bool(should_rebalance),
            "emergency_rebalance": bool(emergency_rebalance),
        })
        rows.append(out)
        cfg_usage[current_cfg["name"]] = cfg_usage.get(current_cfg["name"], 0) + 1
        prev_w = w

    out_df = pd.DataFrame(rows)
    out_df["strategy_equity_net"] = cfg.initial_capital * (1.0 + out_df["strategy_return_net"]).cumprod()
    out_df["strategy_equity_gross"] = cfg.initial_capital * (1.0 + out_df["strategy_return_gross"]).cumprod()
    return out_df, cfg_usage


# ============================================================
# 6. WALK-FORWARD
# ============================================================

def extract_model_importance(pipeline: Pipeline, feature_cols: List[str]) -> Dict[str, float]:
    """Pipeline 내부 XGBClassifier의 feature_importances_를 dict로 추출."""
    try:
        model = pipeline.named_steps["model"]
        imp = np.asarray(model.feature_importances_, dtype=float)
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



def run_walk_forward(df: pd.DataFrame, feature_cols: List[str], cfg: Config) -> pd.DataFrame:
    valid_cols = feature_cols + [
        "future_volatility_20d", "future_return_20d", "future_max_return_20d", "future_min_return_20d",
        "stock_next_return", "bond_next_return", "cash_next_return",
    ]
    work = df.dropna(subset=valid_cols).copy()
    work = work[work.index >= pd.Timestamp(cfg.backtest_start_date)].copy()

    if len(work) < cfg.min_train_rows:
        raise ValueError("백테스트 가능한 데이터가 부족합니다.")

    all_df = df.copy()
    prediction_rows = []

    stage1_model: Optional[Pipeline] = None
    downrisk_model: Optional[Pipeline] = None
    thresholds: Optional[Dict[str, float]] = None
    last_train_i: Optional[int] = None
    downrisk_available = False

    stage1_importance_history: List[Dict[str, float]] = []
    downrisk_importance_history: List[Dict[str, float]] = []

    candidate_positions = [all_df.index.get_loc(idx) for idx in work.index]

    for k, pos in enumerate(candidate_positions):
        date = all_df.index[pos]
        train_end_pos = pos - cfg.horizon_gap_days
        if train_end_pos < cfg.min_train_rows:
            continue

        need_retrain = (
            stage1_model is None
            or last_train_i is None
            or (k % cfg.retrain_every_n_days == 0)
        )

        if need_retrain:
            train_df = all_df.iloc[:train_end_pos].copy()
            train_df = train_df.dropna(subset=valid_cols)
            if len(train_df) < cfg.min_train_rows:
                continue

            thresholds = compute_thresholds(train_df)
            labels = make_labels(train_df, thresholds)

            # Stage1: 정상 vs 고변동
            y_stage1 = (labels != "정상").astype(int).values
            X_train = train_df[feature_cols]
            spw1 = calc_scale_pos_weight(y_stage1)
            stage1_model = make_xgb_stage1(cfg, spw1)
            stage1_model.fit(X_train, y_stage1)
            imp1 = extract_model_importance(stage1_model, feature_cols)
            if imp1:
                stage1_importance_history.append(imp1)

            # Auxiliary: 하락고변동 one-vs-rest 직접 예측
            # 정상/상승고변동을 모두 negative로 두고, 하락고변동만 positive로 둔다.
            y_downrisk = (labels == "하락고변동").astype(int).values
            if len(np.unique(y_downrisk)) == 2 and int(y_downrisk.sum()) >= 20:
                spw_down = calc_scale_pos_weight(y_downrisk)
                downrisk_model = make_xgb_stage2(cfg, spw_down)
                downrisk_model.fit(X_train, y_downrisk)
                imp_down = extract_model_importance(downrisk_model, feature_cols)
                if imp_down:
                    downrisk_importance_history.append(imp_down)
                downrisk_available = True
            else:
                downrisk_model = None
                downrisk_available = False

            last_train_i = pos

        if stage1_model is None or thresholds is None:
            continue

        row_df = all_df.iloc[[pos]]
        X_now = row_df[feature_cols]

        prob_high = float(stage1_model.predict_proba(X_now)[0, 1])
        if downrisk_model is not None and downrisk_available:
            prob_down_risk = float(downrisk_model.predict_proba(X_now)[0, 1])
        else:
            prob_down_risk = 0.0

        # joint proxy: 하락위험은 별도 OVR 확률을 사용하되, 최종 3상태 확률 표현은 고변동 확률과 정합되게 clipping한다.
        prob_down_hv = float(np.clip(min(prob_high, prob_down_risk), 0.0, 1.0))
        prob_up_proxy = float(np.clip(max(prob_high - prob_down_hv, 0.0), 0.0, 1.0))
        prob_normal = 1.0 - prob_high

        actual_label = assign_atr_aware_label(all_df.iloc[pos], thresholds)
        actual_risk = "고변동" if actual_label != "정상" else "정상"
        pred_risk = "고변동" if prob_high >= cfg.pred_high_vol_threshold else "정상"

        if pred_risk != "고변동":
            pred_split = "정상"
        else:
            pred_split = "하락고변동" if prob_down_risk >= cfg.pred_down_high_vol_threshold else "상승고변동"

        prediction_rows.append({
            "Date": date,
            "actual_risk": actual_risk,
            "actual_split_vol": actual_label,
            "pred_risk": pred_risk,
            "pred_split_vol": pred_split,
            "prob_normal": prob_normal,
            "prob_high_vol": prob_high,
            "prob_down_risk": prob_down_risk,
            "prob_up_proxy": prob_up_proxy,
            "prob_up_high_vol": prob_up_proxy,
            "prob_down_high_vol": prob_down_hv,
            "prob_up_given_high_vol": float(prob_up_proxy / prob_high) if prob_high > 0 else 0.5,
            "prob_down_given_high_vol": float(prob_down_hv / prob_high) if prob_high > 0 else 0.5,
            "downrisk_model_used": bool(downrisk_available),
            "stage2_model_used": bool(downrisk_available),
            "stock_next_return": float(all_df.iloc[pos]["stock_next_return"]),
            "bond_next_return": float(all_df.iloc[pos]["bond_next_return"]),
            "cash_next_return": float(all_df.iloc[pos]["cash_next_return"]),
        })

    pred_df = pd.DataFrame(prediction_rows)
    if pred_df.empty:
        raise ValueError("walk-forward 예측 결과가 비어 있습니다.")

    pred_df = pred_df.sort_values("Date").reset_index(drop=True)

    if cfg.use_prob_ewma:
        for col in ["prob_high_vol", "prob_down_risk"]:
            pred_df[col] = pred_df[col].ewm(span=cfg.prob_ewma_span, adjust=False).mean()
        pred_df["prob_normal"] = 1.0 - pred_df["prob_high_vol"]
        pred_df["prob_down_high_vol"] = np.minimum(pred_df["prob_high_vol"], pred_df["prob_down_risk"]).clip(0.0, 1.0)
        pred_df["prob_up_proxy"] = (pred_df["prob_high_vol"] - pred_df["prob_down_high_vol"]).clip(0.0, 1.0)
        pred_df["prob_up_high_vol"] = pred_df["prob_up_proxy"]
        hv_denom = pred_df["prob_high_vol"].replace(0, np.nan)
        pred_df["prob_down_given_high_vol"] = (pred_df["prob_down_high_vol"] / hv_denom).fillna(0.5).clip(0.0, 1.0)
        pred_df["prob_up_given_high_vol"] = (1.0 - pred_df["prob_down_given_high_vol"]).clip(0.0, 1.0)

    pred_df["pred_risk"] = np.where(
        pred_df["prob_high_vol"] >= cfg.pred_high_vol_threshold,
        "고변동",
        "정상",
    )
    pred_df["pred_split_vol"] = np.where(
        pred_df["pred_risk"] != "고변동",
        "정상",
        np.where(pred_df["prob_down_risk"] >= cfg.pred_down_high_vol_threshold, "하락고변동", "상승고변동"),
    )

    pred_df.attrs["stage1_feature_importance_mean"] = mean_importance(stage1_importance_history)
    pred_df.attrs["downrisk_feature_importance_mean"] = mean_importance(downrisk_importance_history)
    pred_df.attrs["stage2_feature_importance_mean"] = pred_df.attrs["downrisk_feature_importance_mean"]
    pred_df.attrs["stage1_retrain_count"] = len(stage1_importance_history)
    pred_df.attrs["downrisk_retrain_count"] = len(downrisk_importance_history)
    pred_df.attrs["stage2_retrain_count"] = len(downrisk_importance_history)

    return pred_df


# ============================================================
# 7. METRICS / SAVE
# ============================================================


def classification_metrics(pred_df: pd.DataFrame) -> Dict[str, object]:
    """
    v4 평가 지표.

    1. stage1_risk_classification
       - 전체 구간 대상
       - 정상 / 고변동 이진 분류

    2. downrisk_ovr_classification
       - 전체 구간 대상
       - 하락고변동 / 비하락고변동 one-vs-rest 직접 평가
       - Stage2 상승/하락 방향 분류를 대체하는 핵심 보조 지표

    3. final_cascade_3state_classification
       - 전체 구간 대상
       - 정상 / 상승고변동 / 하락고변동 최종 출력 평가
       - 보조 지표
    """
    # Stage 1
    y_true_risk = (pred_df["actual_risk"] == "고변동").astype(int).values
    y_pred_risk = (pred_df["pred_risk"] == "고변동").astype(int).values
    p_risk = pred_df["prob_high_vol"].astype(float).clip(0.0, 1.0).values

    stage1 = {
        "rows": int(len(pred_df)),
        "accuracy": float(accuracy_score(y_true_risk, y_pred_risk)),
        "macro_f1": float(f1_score(y_true_risk, y_pred_risk, average="macro", zero_division=0)),
        "high_vol_precision": float(precision_score(y_true_risk, y_pred_risk, zero_division=0)),
        "high_vol_recall": float(recall_score(y_true_risk, y_pred_risk, zero_division=0)),
        "high_vol_f1": float(f1_score(y_true_risk, y_pred_risk, zero_division=0)),
        "normal_precision": float(precision_score(1 - y_true_risk, 1 - y_pred_risk, zero_division=0)),
        "normal_recall": float(recall_score(1 - y_true_risk, 1 - y_pred_risk, zero_division=0)),
        "normal_f1": float(f1_score(1 - y_true_risk, 1 - y_pred_risk, zero_division=0)),
        "brier": float(brier_score_loss(y_true_risk, p_risk)),
        "report": classification_report(
            y_true_risk,
            y_pred_risk,
            target_names=["정상", "고변동"],
            output_dict=True,
            zero_division=0,
        ),
    }
    if len(np.unique(y_true_risk)) == 2:
        stage1["roc_auc"] = float(roc_auc_score(y_true_risk, p_risk))
        stage1["pr_auc"] = float(average_precision_score(y_true_risk, p_risk))
    else:
        stage1["roc_auc"] = None
        stage1["pr_auc"] = None

    # Down-risk OVR
    y_true_down = (pred_df["actual_split_vol"] == "하락고변동").astype(int).values
    p_down = pred_df["prob_down_risk"].astype(float).clip(0.0, 1.0).values
    y_pred_down = (p_down >= CFG.pred_down_high_vol_threshold).astype(int)

    down_report = classification_report(
        y_true_down,
        y_pred_down,
        target_names=["비하락고변동", "하락고변동"],
        output_dict=True,
        zero_division=0,
    )
    downrisk = {
        "rows": int(len(pred_df)),
        "evaluation_scope": "all rows, down-high-vol one-vs-rest",
        "threshold": float(CFG.pred_down_high_vol_threshold),
        "accuracy": float(accuracy_score(y_true_down, y_pred_down)),
        "precision": float(precision_score(y_true_down, y_pred_down, zero_division=0)),
        "recall": float(recall_score(y_true_down, y_pred_down, zero_division=0)),
        "f1": float(f1_score(y_true_down, y_pred_down, zero_division=0)),
        "support_positive": int(y_true_down.sum()),
        "support_negative": int((1 - y_true_down).sum()),
        "pred_positive_ratio": float(y_pred_down.mean()),
        "brier": float(brier_score_loss(y_true_down, p_down)),
        "report": down_report,
    }
    if len(np.unique(y_true_down)) == 2:
        downrisk["roc_auc"] = float(roc_auc_score(y_true_down, p_down))
        downrisk["pr_auc"] = float(average_precision_score(y_true_down, p_down))
    else:
        downrisk["roc_auc"] = None
        downrisk["pr_auc"] = None

    down_diags = []
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        pred = (p_down >= t).astype(int)
        down_diags.append({
            "threshold": float(t),
            "basis": "prob_down_risk",
            "down_high_vol_precision": float(precision_score(y_true_down, pred, zero_division=0)),
            "down_high_vol_recall": float(recall_score(y_true_down, pred, zero_division=0)),
            "down_high_vol_f1": float(f1_score(y_true_down, pred, zero_division=0)),
            "pred_down_high_vol_ratio": float(pred.mean()),
        })

    # Final Cascade 3-state
    final_labels = ["정상", "상승고변동", "하락고변동"]
    y_true_final = pred_df["actual_split_vol"].values
    y_pred_final = pred_df["pred_split_vol"].values
    final_report = classification_report(
        y_true_final,
        y_pred_final,
        labels=final_labels,
        output_dict=True,
        zero_division=0,
    )

    final_cascade = {
        "rows": int(len(pred_df)),
        "evaluation_scope": "all rows, final cascade output",
        "accuracy": float(accuracy_score(y_true_final, y_pred_final)),
        "macro_f1": float(f1_score(
            y_true_final,
            y_pred_final,
            labels=final_labels,
            average="macro",
            zero_division=0,
        )),
        "report": final_report,
        "label_support": {lab: int((pred_df["actual_split_vol"] == lab).sum()) for lab in final_labels},
        "per_class": {},
    }
    for lab in final_labels:
        final_cascade["per_class"][lab] = {
            "precision": float(final_report.get(lab, {}).get("precision", 0.0)),
            "recall": float(final_report.get(lab, {}).get("recall", 0.0)),
            "f1": float(final_report.get(lab, {}).get("f1-score", 0.0)),
            "support": int(final_report.get(lab, {}).get("support", 0)),
        }

    y_down_final = (pred_df["actual_split_vol"] == "하락고변동").astype(int).values
    y_up_final = (pred_df["actual_split_vol"] == "상승고변동").astype(int).values
    if len(np.unique(y_down_final)) == 2:
        final_cascade["down_high_vol_roc_auc"] = float(roc_auc_score(y_down_final, pred_df["prob_down_risk"].values))
        final_cascade["down_high_vol_pr_auc"] = float(average_precision_score(y_down_final, pred_df["prob_down_risk"].values))
    else:
        final_cascade["down_high_vol_roc_auc"] = None
        final_cascade["down_high_vol_pr_auc"] = None

    if len(np.unique(y_up_final)) == 2:
        final_cascade["up_high_vol_roc_auc"] = float(roc_auc_score(y_up_final, pred_df["prob_up_high_vol"].values))
        final_cascade["up_high_vol_pr_auc"] = float(average_precision_score(y_up_final, pred_df["prob_up_high_vol"].values))
    else:
        final_cascade["up_high_vol_roc_auc"] = None
        final_cascade["up_high_vol_pr_auc"] = None

    return {
        "stage1_risk_classification": stage1,
        "downrisk_ovr_classification": downrisk,
        "final_cascade_3state_classification": final_cascade,

        # 호환용 alias
        "split_vol_classification": final_cascade,
        "down_high_vol_threshold_diagnostics": down_diags,
        "downrisk_threshold_diagnostics": down_diags,

        # v3와 비교용. Stage2 방향 모델은 v4 이후 제거됨.
        "stage2_direction_classification_on_true_high_vol": {
            "enabled": False,
            "reason": "v3에서 고변동 내부 상승/하락 방향 분류 ROC-AUC가 0.5 미만이어서 제거. v4는 down-risk one-vs-rest를 사용.",
        },
    }



def build_summary(pred_df: pd.DataFrame, feature_cols: List[str], cfg_usage: Dict[str, int], cfg: Config) -> Dict[str, object]:
    perf = {
        "strategy_after_cost": perf_stats(pred_df["strategy_return_net"], cfg.initial_capital),
        "strategy_gross": perf_stats(pred_df["strategy_return_gross"], cfg.initial_capital),
        "stock_buy_hold": perf_stats(pred_df["stock_next_return"], cfg.initial_capital),
        "benchmark_60_40": perf_stats(0.6 * pred_df["stock_next_return"] + 0.4 * pred_df["bond_next_return"], cfg.initial_capital),
        "static_50_30_20": perf_stats(0.5 * pred_df["stock_next_return"] + 0.3 * pred_df["bond_next_return"] + 0.2 * pred_df["cash_next_return"], cfg.initial_capital),
    }
    cls = classification_metrics(pred_df)
    latest = pred_df.iloc[-1]

    summary = {
        "target_ticker": cfg.target_ticker,
        "bond_ticker": cfg.bond_ticker,
        "cash_ticker": cfg.cash_ticker,
        "model_type": "xgb_atr_h10_stage1_gated_downrisk_v7",
        "config": asdict(cfg),
        "period": {
            "start": str(pred_df["Date"].iloc[0]),
            "end": str(pred_df["Date"].iloc[-1]),
            "rows": int(len(pred_df)),
        },
        "feature_count": len(feature_cols),
        "feature_set": "price_volume_volatility_atr_range_downside_features",
        "feature_cols": feature_cols,
        "stage1_feature_importance_mean": pred_df.attrs.get("stage1_feature_importance_mean", {}),
        "downrisk_feature_importance_mean": pred_df.attrs.get("downrisk_feature_importance_mean", {}),
        "stage2_feature_importance_mean": pred_df.attrs.get("downrisk_feature_importance_mean", {}),
        "stage1_retrain_count": int(pred_df.attrs.get("stage1_retrain_count", 0)),
        "downrisk_retrain_count": int(pred_df.attrs.get("downrisk_retrain_count", 0)),
        "label_design": {
            "target_horizon_days": cfg.risk_horizon_days,
            "purge_gap_days": cfg.horizon_gap_days,
            "stage1": "정상 / 고변동",
            "auxiliary": "하락고변동 / 비하락고변동 one-vs-rest",
            "allocation": "Stage1-gated allocation: NORMAL/WATCH/HIGH_VOL/RISK_OFF buckets; Down-risk is strongly used only after Stage1 warning/high-vol gate",
            "removed": "고변동 내부 상승고변동 / 하락고변동 Stage2 방향 분류는 v4 이후 제거",
            "high_vol_rule": "future variables are computed over cfg.risk_horizon_days trading days; high-vol if future volatility >= train q80 OR future min return <= trend/ATR-adjusted down threshold OR future max return >= trend/ATR-adjusted up threshold",
            "downrisk_rule": "actual label == 하락고변동을 positive로 직접 학습",
            "atr_aware": True,
            "down_priority": True,
        },
        "average_probabilities": {
            "avg_prob_normal": float(pred_df["prob_normal"].mean()),
            "avg_prob_high_vol": float(pred_df["prob_high_vol"].mean()),
            "avg_prob_down_risk": float(pred_df["prob_down_risk"].mean()),
            "avg_prob_up_proxy": float(pred_df["prob_up_proxy"].mean()),
            "avg_prob_up_high_vol": float(pred_df["prob_up_high_vol"].mean()),
            "avg_prob_down_high_vol": float(pred_df["prob_down_high_vol"].mean()),
            "downrisk_model_used_ratio": float(pred_df["downrisk_model_used"].mean()),
        },
        "average_weights": {
            "avg_stock_weight": float(pred_df["stock_weight"].mean()),
            "avg_bond_weight": float(pred_df["bond_weight"].mean()),
            "avg_cash_weight": float(pred_df["cash_weight"].mean()),
            "min_stock_weight": float(pred_df["stock_weight"].min()),
            "max_stock_weight": float(pred_df["stock_weight"].max()),
        },
        "allocation_regime_distribution_pct": (
            pred_df["allocation_regime"].value_counts(normalize=True).mul(100).round(2).to_dict()
            if "allocation_regime" in pred_df.columns else {}
        ),
        "turnover": {
            "avg_daily_trade_ratio": float(pred_df["turnover"].mean()),
            "annual_turnover_estimate": float(pred_df["turnover"].mean() * 252.0),
            "total_transaction_cost_rate_sum": float(pred_df["transaction_cost"].sum()),
            "rebalance_ratio": float(pred_df.get("rebalanced", pd.Series([False] * len(pred_df))).mean()),
            "emergency_rebalance_ratio": float(pred_df.get("emergency_rebalance", pd.Series([False] * len(pred_df))).mean()),
        },
        "performance": perf,
        **cls,
        "allocation_config_usage_top10": dict(sorted(cfg_usage.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "latest_prediction": {
            "date": str(latest["Date"]),
            "pred_risk": latest["pred_risk"],
            "pred_split_vol": latest["pred_split_vol"],
            "prob_normal": round(float(latest["prob_normal"]) * 100, 2),
            "prob_high_vol": round(float(latest["prob_high_vol"]) * 100, 2),
            "prob_down_risk": round(float(latest["prob_down_risk"]) * 100, 2),
            "prob_up_proxy": round(float(latest["prob_up_proxy"]) * 100, 2),
            "prob_up_high_vol": round(float(latest["prob_up_high_vol"]) * 100, 2),
            "prob_down_high_vol": round(float(latest["prob_down_high_vol"]) * 100, 2),
            "allocation_regime": str(latest.get("allocation_regime", "")),
            "target_allocation": {
                "stock": round(float(latest["stock_weight"]) * 100, 2),
                "bond": round(float(latest["bond_weight"]) * 100, 2),
                "cash": round(float(latest["cash_weight"]) * 100, 2),
            },
        },
    }
    return summary



def print_summary(summary: Dict[str, object]) -> None:
    p = summary["performance"]
    s = summary["stage1_risk_classification"]
    down = summary["downrisk_ovr_classification"]
    final = summary["final_cascade_3state_classification"]
    w = summary["average_weights"]
    t = summary["turnover"]
    latest = summary["latest_prediction"]

    print("\n==============================")
    print("XGBoost ATR-aware H10 v7 결과 요약")
    print("Stage1-gated Down-risk Allocation + Final Cascade")
    print("==============================")
    print(f"기간: {summary['period']['start']} ~ {summary['period']['end']}")
    print(f"거래일 수: {summary['period']['rows']}")
    print(f"피처 수: {summary['feature_count']}")
    print(f"평균 주식 비중: {w['avg_stock_weight'] * 100:.2f}%")
    print(f"평균 채권 비중: {w['avg_bond_weight'] * 100:.2f}%")
    print(f"평균 현금 비중: {w['avg_cash_weight'] * 100:.2f}%")
    print(f"평균 일간 교체율: {t['avg_daily_trade_ratio'] * 100:.4f}%")
    print(f"연간 교체율 추정: {t['annual_turnover_estimate'] * 100:.2f}%")
    print(f"리밸런싱 발생 비율: {t.get('rebalance_ratio', 0.0) * 100:.2f}%")
    print(f"긴급 리밸런싱 비율: {t.get('emergency_rebalance_ratio', 0.0) * 100:.2f}%")
    print(f"배분 regime 분포: {summary.get('allocation_regime_distribution_pct', {})}")

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

    print("\n[1단계 Risk 분류 성능 - 전체 구간]")
    print(f"Accuracy: {s['accuracy']:.4f}")
    print(f"Macro F1: {s['macro_f1']:.6f}")
    print(f"High-vol Precision: {s['high_vol_precision']:.6f}")
    print(f"High-vol Recall: {s['high_vol_recall']:.6f}")
    print(f"High-vol F1: {s['high_vol_f1']:.6f}")
    print(f"ROC-AUC: {s['roc_auc']}")
    print(f"PR-AUC: {s['pr_auc']}")

    print("\n[Down-risk OVR 분류 성능 - 전체 구간]")
    print(f"Threshold: {down['threshold']}")
    print(f"Accuracy: {down['accuracy']:.6f}")
    print(f"Precision: {down['precision']:.6f}")
    print(f"Recall: {down['recall']:.6f}")
    print(f"F1: {down['f1']:.6f}")
    print(f"ROC-AUC: {down['roc_auc']}")
    print(f"PR-AUC: {down['pr_auc']}")
    print(f"Support positive: {down['support_positive']}")
    print(f"Support negative: {down['support_negative']}")
    print(f"Pred positive ratio: {down['pred_positive_ratio']:.6f}")

    print("\n[최종 Cascade 3상태 출력 성능 - 전체 구간 보조지표]")
    print(f"Accuracy: {final['accuracy']:.6f}")
    print(f"Macro F1: {final['macro_f1']:.6f}")
    print(f"Down-high-vol ROC-AUC: {final['down_high_vol_roc_auc']}")
    print(f"Down-high-vol PR-AUC: {final['down_high_vol_pr_auc']}")
    print(f"Up-high-vol ROC-AUC: {final['up_high_vol_roc_auc']}")
    print(f"Up-high-vol PR-AUC: {final['up_high_vol_pr_auc']}")
    print(f"Label support: {final['label_support']}")

    print("\n[최신 예측]")
    print(json.dumps(latest, ensure_ascii=False, indent=2))


# ============================================================
# 8. MAIN
# ============================================================

def main() -> None:
    cfg = CFG
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("[1] 데이터 다운로드")
    target = download_ohlcv(cfg.target_ticker, cfg.start_date, cfg.end_date)
    bond_close = download_close(cfg.bond_ticker, cfg.start_date, cfg.end_date)
    try:
        cash_close = download_close(cfg.cash_ticker, cfg.start_date, cfg.end_date)
    except Exception:
        cash_close = pd.Series(index=target.index, data=np.nan, name=cfg.cash_ticker)

    print("[2] 피처 생성")
    df, feature_cols = build_features(target)

    # 다음 거래일 수익률: 오늘 신호로 내일 수익률에 노출된다고 가정
    df["stock_next_return"] = df["Close"].pct_change().shift(-1)

    bond_ret = bond_close.pct_change().shift(-1).reindex(df.index).fillna(0.0)
    cash_ret = cash_close.pct_change().shift(-1).reindex(df.index).fillna(0.0)
    df["bond_next_return"] = bond_ret
    df["cash_next_return"] = cash_ret

    print(f"    피처 수: {len(feature_cols)}")
    print("[3] Walk-forward XGBoost Stage1 + Down-risk OVR 예측")
    pred_raw = run_walk_forward(df, feature_cols, cfg)

    print("[4] 배분/백테스트")
    pred_df, cfg_usage = apply_rolling_allocation(pred_raw, cfg)
    pred_df.attrs.update(pred_raw.attrs)

    print("[5] 결과 저장")
    summary = build_summary(pred_df, feature_cols, cfg_usage, cfg)

    pred_path = result_dir / "qqq_xgb_atr_h10_stage1_gated_downrisk_v7_predictions.csv"
    summary_path = result_dir / "qqq_xgb_atr_h10_stage1_gated_downrisk_v7_summary.json"
    latest_path = result_dir / "qqq_xgb_atr_h10_stage1_gated_downrisk_v7_latest.json"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary["latest_prediction"], f, ensure_ascii=False, indent=2)

    print_summary(summary)
    print("\n[저장 완료]")
    print(f"- {pred_path}")
    print(f"- {summary_path}")
    print(f"- {latest_path}")


if __name__ == "__main__":
    main()
