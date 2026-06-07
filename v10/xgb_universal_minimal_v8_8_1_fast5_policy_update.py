"""
XGBoost Universal Minimal v8.8.1 Fast5 Policy Update
=================================

목적
- v8.6.39에서 쓸모가 약하거나 과도하게 복잡한 요소를 제거한 ablation 모델.
- Down-risk head, downrisk branch ensemble, direction binary, portfolio policy model,
  tier weight optimizer, condition search, adaptive label policy를 제거한다.
- HighVol + UpStrength20D + rule-based Drawdown/Trend guard만으로 성능 유지 여부를 검증한다.

핵심 설계
1) HighVol head
   - future realized volatility가 과거 rolling quantile보다 높은지 예측한다.
2) UpStrength20D head
   - 20일 미래 수익률이 현재 horizon volatility 대비 충분히 큰지 예측한다.
3) Allocation
   - prob_high_vol, drawdown_guard_score, trend_break_score로 risk_score를 만든다.
   - prob_up_strengthening_20d는 risk가 낮고 trend가 망가지지 않았을 때만 주식 비중을 올린다.
4) No Down-risk
   - down-risk 확률과 관련 출력/판단을 생성하지 않는다.

필요 패키지
    pip install pandas numpy yfinance scikit-learn xgboost

실행 예시
    python xgb_universal_minimal_v8_8_0.py --target-ticker QQQ --speed-profile fast
    python xgb_universal_minimal_v8_8_0.py --asset-list QQQ,SPY,AAPL,SOXX,NVDA --speed-profile fast

주의
- 이 파일은 "최종 성능형"이 아니라 "불필요 요소 제거 실험형"이다.
- 미래 정보 컬럼은 label 생성과 diagnostics에만 쓰고 feature input에서 제외한다.
- walk-forward 학습 시 max(horizons)만큼 purge gap을 둔다.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise ImportError("yfinance가 필요합니다. `pip install yfinance`를 실행하세요.") from exc

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
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ============================================================
# 0. CONFIG
# ============================================================

@dataclass
class Config:
    model_version: str = "v8.8.1_fast5_policy_update_no_downrisk"

    target_ticker: str = "QQQ"
    bond_ticker: str = "IEF"
    cash_ticker: str = "BIL"

    start_date: str = "1999-03-10"
    backtest_start_date: str = "2013-01-02"
    end_date: Optional[str] = None

    initial_capital: float = 100_000_000.0
    transaction_cost_rate: float = 0.001
    execution_lag_days: int = 1
    allow_cash_download_fallback: bool = False

    horizons: Tuple[int, ...] = (20,)
    high_vol_horizon: int = 20
    up_horizon: int = 20
    purge_gap: int = 20

    min_train_rows: int = 756
    retrain_every_n_days: int = 20
    max_train_rows: Optional[int] = 1260

    random_state: int = 42
    n_jobs: int = -1

    # Label rules: universal / volatility-scaled
    label_vol_window: int = 756
    high_vol_quantile: float = 0.80
    up_strength_k: float = 0.55
    min_positive: int = 20

    # XGBoost minimal heads
    n_estimators: int = 120
    learning_rate: float = 0.03
    max_depth: int = 2
    min_child_weight: float = 8.0
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 10.0
    reg_alpha: float = 0.2

    # Probability smoothing
    use_prob_ewma: bool = True
    prob_ewma_span: int = 7

    # Allocation: no down-risk, no tier optimizer, no portfolio policy model
    rebalance_every_n_days: int = 5
    no_trade_band: float = 0.12
    emergency_risk_score_threshold: float = 0.82
    emergency_cooldown_days: int = 5

    # Universal stock mapping from risk score
    stock_at_risk_0: float = 0.88
    stock_at_risk_35: float = 0.82
    stock_at_risk_50: float = 0.72
    stock_at_risk_65: float = 0.58
    stock_at_risk_80: float = 0.42
    stock_at_risk_90: float = 0.30
    stock_at_risk_100: float = 0.20

    # Drawdown / trend guard
    drawdown_guard_60_ref: float = 0.12
    drawdown_guard_120_ref: float = 0.18
    trend_break_cut: float = 0.06
    drawdown_extra_cut: float = 0.08

    # Up-strength overlay
    up_trigger: float = 0.35
    strong_up_trigger: float = 0.55
    max_up_bonus: float = 0.18
    up_high_vol_block: float = 0.58
    full_stock_up_trigger: float = 0.70
    full_stock_high_vol_max: float = 0.35

    # Defensive split
    bond_ratio_of_defensive: float = 0.65

    # Scheduled policy parameter update.
    enable_fast_policy_update: bool = True
    param_update_every: int = 5
    param_validation_window: int = 252
    param_embargo_days: int = 20
    param_min_history: int = 252
    param_min_improvement_margin: float = 0.03
    param_change_penalty_per_change: float = 0.03
    param_grid_mode: str = "asset_class"  # asset_class / broad / aggressive / common
    target_weight_ewma_alpha: float = 1.0  # 1.0 = no target smoothing

    result_dir: str = "results_v8_8_1_fast5"


# ============================================================
# 1. DATA
# ============================================================

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def download_ohlcv(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = _flatten_yf_columns(df)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: yfinance 결과에 필요한 컬럼이 없습니다: {missing}")
    df = df[required].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index)
    return df.dropna()


def download_close(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    df = download_ohlcv(ticker, start, end)
    return df["close"].rename(ticker)


# ============================================================
# 2. FEATURES / LABELS
# ============================================================

def rolling_rank_last(series: pd.Series, window: int) -> pd.Series:
    def _rank(x: np.ndarray) -> float:
        if len(x) == 0 or np.isnan(x[-1]):
            return np.nan
        valid = x[~np.isnan(x)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= x[-1]).mean())
    return series.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=True)


def calc_trend_slope(close: pd.Series, window: int) -> pd.Series:
    logp = np.log(close.replace(0, np.nan))
    return logp.diff(window) / float(window)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_features(ohlcv: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    df = ohlcv.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    ret1 = close.pct_change()
    df["ret_1d"] = ret1

    # Returns / trend
    for w in (5, 10, 20, 60, 120):
        df[f"return_{w}d"] = close.pct_change(w)
        ma = close.rolling(w, min_periods=max(5, w // 3)).mean()
        df[f"price_ma_{w}_gap"] = close / ma - 1.0
        df[f"trend_slope_{w}"] = calc_trend_slope(close, w)
        df[f"realized_vol_{w}"] = ret1.rolling(w, min_periods=max(5, w // 2)).std() * np.sqrt(252)
        df[f"drawdown_{w}"] = close / close.rolling(w, min_periods=max(5, w // 2)).max() - 1.0
        df[f"positive_return_ratio_{w}"] = (ret1 > 0).rolling(w, min_periods=max(5, w // 2)).mean()

    df["ma_gap_20_60"] = close.rolling(20).mean() / close.rolling(60).mean() - 1.0
    df["ma_gap_60_120"] = close.rolling(60).mean() / close.rolling(120).mean() - 1.0
    df["ma_gap_50_200"] = close.rolling(50).mean() / close.rolling(200).mean() - 1.0
    df["ma200_slope_60"] = close.rolling(200).mean().pct_change(60) / 60.0

    # Price position
    for w in (20, 60, 120):
        roll_min = close.rolling(w, min_periods=max(5, w // 2)).min()
        roll_max = close.rolling(w, min_periods=max(5, w // 2)).max()
        df[f"price_position_{w}"] = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)
        df[f"close_to_{w}d_high"] = close / roll_max - 1.0

    # Volume pressure
    df["volume_ratio_20"] = volume / volume.rolling(20, min_periods=10).mean() - 1.0
    vol_mean = volume.rolling(20, min_periods=10).mean()
    vol_std = volume.rolling(20, min_periods=10).std()
    df["volume_zscore_20"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    df["down_volume_ratio_20"] = ((ret1 < 0) * volume).rolling(20, min_periods=10).sum() / volume.rolling(20, min_periods=10).sum().replace(0, np.nan)
    df["high_volume_down_ratio_20"] = ((ret1 < 0) & (df["volume_zscore_20"] > 1.0)).rolling(20, min_periods=10).mean()

    # Range / ATR / volatility
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["true_range_pct"] = tr / close
    for w in (10, 14, 20, 60):
        df[f"atr_pct_{w}"] = tr.rolling(w, min_periods=max(5, w // 2)).mean() / close
    df["atr_rank_252"] = rolling_rank_last(df["atr_pct_14"], 252)

    # Downside / semi vol kept as features, not down-risk head
    neg_ret = ret1.where(ret1 < 0, 0.0)
    for w in (20, 60):
        df[f"downside_vol_{w}"] = neg_ret.rolling(w, min_periods=max(5, w // 2)).std() * np.sqrt(252)
        df[f"vol_rank_{w}_252"] = rolling_rank_last(df[f"realized_vol_{w}"], 252)

    # Vol-normalized universal features
    hvol = ret1.rolling(cfg.up_horizon, min_periods=max(5, cfg.up_horizon // 2)).std() * np.sqrt(cfg.up_horizon)
    df["current_horizon_vol"] = hvol.shift(1)
    df["return_20d_over_hvol"] = _safe_div(df["return_20d"], df["current_horizon_vol"])
    df["return_60d_over_vol60"] = _safe_div(df["return_60d"], df["realized_vol_60"] / np.sqrt(252 / 60))
    df["drawdown_60_over_vol60"] = _safe_div(df["drawdown_60"].abs(), df["realized_vol_60"] / np.sqrt(252 / 60))

    # Future targets, labels use future columns but features must not include them
    h = cfg.up_horizon
    df[f"future_return_{h}d"] = close.shift(-h) / close - 1.0
    future_ret = close.pct_change().shift(-1)
    df[f"future_realized_vol_{cfg.high_vol_horizon}d"] = (
        future_ret.rolling(cfg.high_vol_horizon, min_periods=cfg.high_vol_horizon).std().shift(-(cfg.high_vol_horizon - 1))
        * np.sqrt(252)
    )
    df[f"future_mdd_{h}d"] = (
        close.shift(-1).rolling(h, min_periods=h).min().shift(-(h - 1)) / close - 1.0
    )

    # Labels: all thresholds are asset-local and past-only
    vol_q = df[f"realized_vol_{cfg.high_vol_horizon}"].rolling(cfg.label_vol_window, min_periods=252).quantile(cfg.high_vol_quantile).shift(1)
    hv_target = df[f"future_realized_vol_{cfg.high_vol_horizon}d"]
    hv_valid = hv_target.notna() & vol_q.notna()
    df["y_high_vol"] = np.nan
    df.loc[hv_valid, "y_high_vol"] = (hv_target.loc[hv_valid] >= vol_q.loc[hv_valid]).astype(float)

    current_hvol = df["current_horizon_vol"]
    up_target = df[f"future_return_{h}d"]
    up_valid = up_target.notna() & current_hvol.notna()
    df["y_up_strength_20d"] = np.nan
    df.loc[up_valid, "y_up_strength_20d"] = (up_target.loc[up_valid] >= cfg.up_strength_k * current_hvol.loc[up_valid]).astype(float)

    # Actual diagnostic labels
    # NumPy 2.x does not promote mixed string/np.nan outputs in np.where.
    # Build label columns as pandas object/string Series instead.
    df["actual_high_vol"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df["y_high_vol"] == 1.0, "actual_high_vol"] = "고변동"
    df.loc[df["y_high_vol"] == 0.0, "actual_high_vol"] = "정상"

    df["actual_up_strength"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df["y_up_strength_20d"] == 1.0, "actual_up_strength"] = "UP_STRENGTHENING"
    df.loc[df["y_up_strength_20d"] == 0.0, "actual_up_strength"] = "NO_STRENGTH_SIGNAL"

    candidate_features = [
        # returns / trend
        "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "return_20d_over_hvol", "return_60d_over_vol60",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_20", "positive_return_ratio_60",
        # drawdown / price position
        "drawdown_20", "drawdown_60", "drawdown_120", "drawdown_60_over_vol60",
        "price_position_20", "price_position_60", "price_position_120",
        "close_to_20d_high", "close_to_60d_high", "close_to_120d_high",
        # volume
        "volume_ratio_20", "volume_zscore_20", "down_volume_ratio_20", "high_volume_down_ratio_20",
        # volatility
        "true_range_pct", "atr_pct_10", "atr_pct_14", "atr_pct_20", "atr_pct_60", "atr_rank_252",
        "realized_vol_20", "realized_vol_60", "vol_rank_20_252", "vol_rank_60_252",
        "downside_vol_20", "downside_vol_60",
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]
    return df, feature_cols


# ============================================================
# 3. MODEL HELPERS
# ============================================================

def calc_scale_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos <= 0:
        return 1.0
    return max(1.0, neg / max(pos, 1))


def make_model(cfg: Config, scale_pos_weight: float) -> Pipeline:
    if HAS_XGB:
        clf = XGBClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            scale_pos_weight=scale_pos_weight,
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=15,
            random_state=cfg.random_state,
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])


def _predict_pos_proba(model: Pipeline, x: pd.DataFrame) -> float:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        # Degenerate model in rare windows
        cls = getattr(model.named_steps["model"], "classes_", np.array([0]))
        return 1.0 if int(cls[0]) == 1 else 0.0
    return float(proba[0, 1])


def _fit_binary_model(train: pd.DataFrame, feature_cols: List[str], y_col: str, cfg: Config) -> Optional[Pipeline]:
    local = train.dropna(subset=[y_col]).copy()
    y = local[y_col].astype(int).values
    if len(local) < cfg.min_train_rows // 2:
        return None
    if y.sum() < cfg.min_positive or (len(y) - y.sum()) < cfg.min_positive:
        return None
    model = make_model(cfg, calc_scale_pos_weight(y))
    model.fit(local[feature_cols], y)
    return model


def extract_importance(model: Optional[Pipeline], feature_cols: List[str]) -> Dict[str, float]:
    if model is None or not HAS_XGB:
        return {}
    booster = model.named_steps["model"]
    if not hasattr(booster, "feature_importances_"):
        return {}
    vals = booster.feature_importances_
    return {f: float(v) for f, v in zip(feature_cols, vals)}


def mean_importance(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = sorted(set().union(*[x.keys() for x in items]))
    return {k: float(np.mean([x.get(k, 0.0) for x in items])) for k in keys}


# ============================================================
# 4. WALK-FORWARD
# ============================================================

def run_walk_forward(df: pd.DataFrame, feature_cols: List[str], cfg: Config) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    usable = df.dropna(subset=feature_cols + ["y_high_vol", "y_up_strength_20d"]).copy()
    usable = usable.loc[usable.index >= pd.to_datetime(cfg.backtest_start_date)]
    if len(usable) < cfg.min_train_rows + 50:
        raise ValueError("walk-forward에 필요한 데이터가 부족합니다.")

    all_df = df.copy()
    pred_rows: List[Dict[str, object]] = []
    hv_model: Optional[Pipeline] = None
    up_model: Optional[Pipeline] = None
    last_train_pos = -1
    hv_imps: List[Dict[str, float]] = []
    up_imps: List[Dict[str, float]] = []

    full_idx = list(all_df.index)
    start_ts = pd.to_datetime(cfg.backtest_start_date)
    start_pos = next(i for i, ts in enumerate(full_idx) if ts >= start_ts)

    for pos in range(start_pos, len(full_idx) - cfg.execution_lag_days - 1):
        ts = full_idx[pos]
        train_end_pos = pos - cfg.purge_gap
        if train_end_pos <= 0:
            continue
        train = all_df.iloc[:train_end_pos].dropna(subset=feature_cols + ["y_high_vol", "y_up_strength_20d"])
        if cfg.max_train_rows is not None and len(train) > cfg.max_train_rows:
            train = train.iloc[-cfg.max_train_rows:]
        if len(train) < cfg.min_train_rows:
            continue

        should_retrain = (hv_model is None) or ((pos - last_train_pos) >= cfg.retrain_every_n_days)
        if should_retrain:
            hv_model = _fit_binary_model(train, feature_cols, "y_high_vol", cfg)
            up_model = _fit_binary_model(train, feature_cols, "y_up_strength_20d", cfg)
            last_train_pos = pos
            if hv_model is not None:
                hv_imps.append(extract_importance(hv_model, feature_cols))
            if up_model is not None:
                up_imps.append(extract_importance(up_model, feature_cols))

        if hv_model is None or up_model is None:
            continue

        row = all_df.iloc[pos]
        x = row[feature_cols].to_frame().T
        p_hv = _predict_pos_proba(hv_model, x)
        p_up = _predict_pos_proba(up_model, x)

        out = row.to_dict()
        out["date"] = ts
        out["prob_high_vol_raw"] = p_hv
        out["prob_up_strengthening_20d_raw"] = p_up
        out["actual_high_vol"] = row.get("actual_high_vol", np.nan)
        out["actual_up_strength"] = row.get("actual_up_strength", np.nan)
        pred_rows.append(out)

    pred = pd.DataFrame(pred_rows)
    if pred.empty:
        raise RuntimeError("예측 결과가 비어 있습니다. min_train_rows 또는 기간을 확인하세요.")
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.set_index("date").sort_index()

    if cfg.use_prob_ewma:
        pred["prob_high_vol"] = pred["prob_high_vol_raw"].ewm(span=cfg.prob_ewma_span, adjust=False).mean()
        pred["prob_up_strengthening_20d"] = pred["prob_up_strengthening_20d_raw"].ewm(span=cfg.prob_ewma_span, adjust=False).mean()
    else:
        pred["prob_high_vol"] = pred["prob_high_vol_raw"]
        pred["prob_up_strengthening_20d"] = pred["prob_up_strengthening_20d_raw"]

    pred["prob_normal"] = 1.0 - pred["prob_high_vol"]
    pred["pred_risk"] = np.where(pred["prob_high_vol"] >= 0.50, "고변동", "정상")
    pred["pred_up_strength"] = np.where(pred["prob_up_strengthening_20d"] >= cfg.up_trigger, "UP_STRENGTHENING", "NO_STRENGTH_SIGNAL")

    imps = {
        "highvol_feature_importance_mean": mean_importance(hv_imps),
        "up_feature_importance_mean": mean_importance(up_imps),
    }
    return pred, imps


# ============================================================
# 5. ALLOCATION / BACKTEST
# ============================================================

def compute_mid_trend_score(row: pd.Series) -> Tuple[int, str]:
    score = 0
    conditions = [
        row.get("return_60d", 0.0) > 0,
        row.get("return_120d", 0.0) > 0,
        row.get("price_ma_60_gap", 0.0) > 0,
        row.get("price_ma_120_gap", 0.0) > 0,
        row.get("ma_gap_20_60", 0.0) > 0,
        row.get("trend_slope_60", 0.0) > 0,
    ]
    score = int(sum(bool(x) for x in conditions))
    if score >= 4:
        state = "BULL"
    elif score <= 2:
        state = "BEAR"
    else:
        state = "NEUTRAL"
    return score, state


def drawdown_guard_score(row: pd.Series, cfg: Config) -> float:
    dd60 = abs(min(float(row.get("drawdown_60", 0.0) or 0.0), 0.0))
    dd120 = abs(min(float(row.get("drawdown_120", 0.0) or 0.0), 0.0))
    score = max(dd60 / cfg.drawdown_guard_60_ref, dd120 / cfg.drawdown_guard_120_ref)
    return float(np.clip(score, 0.0, 1.0))


def trend_break_score(row: pd.Series) -> float:
    parts = [
        0.35 if float(row.get("price_ma_60_gap", 0.0) or 0.0) < 0 else 0.0,
        0.35 if float(row.get("price_ma_120_gap", 0.0) or 0.0) < 0 else 0.0,
        0.30 if float(row.get("trend_slope_60", 0.0) or 0.0) < 0 else 0.0,
    ]
    return float(np.clip(sum(parts), 0.0, 1.0))


def compute_risk_score(row: pd.Series, cfg: Config) -> float:
    hv = float(row.get("prob_high_vol", 0.0) or 0.0)
    dd = drawdown_guard_score(row, cfg)
    tb = trend_break_score(row)
    risk = 0.70 * hv + 0.20 * dd + 0.10 * tb
    return float(np.clip(risk, 0.0, 1.0))


def stock_from_risk_score(risk: float, cfg: Config) -> float:
    x = float(np.clip(risk, 0.0, 1.0))
    points = [
        (0.00, cfg.stock_at_risk_0),
        (0.35, cfg.stock_at_risk_35),
        (0.50, cfg.stock_at_risk_50),
        (0.65, cfg.stock_at_risk_65),
        (0.80, cfg.stock_at_risk_80),
        (0.90, cfg.stock_at_risk_90),
        (1.00, cfg.stock_at_risk_100),
    ]
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if x0 <= x <= x1:
            t = (x - x0) / max(x1 - x0, 1e-9)
            return float(y0 + t * (y1 - y0))
    return float(points[-1][1])


def split_defensive(stock: float, cfg: Config) -> Tuple[float, float, float]:
    stock = float(np.clip(stock, 0.0, 1.0))
    defensive = 1.0 - stock
    bond = defensive * cfg.bond_ratio_of_defensive
    cash = defensive - bond
    return stock, bond, cash


def target_allocation(row: pd.Series, cfg: Config) -> Dict[str, object]:
    hv = float(row.get("prob_high_vol", 0.0) or 0.0)
    up = float(row.get("prob_up_strengthening_20d", 0.0) or 0.0)
    risk = compute_risk_score(row, cfg)
    mid_score, mid_state = compute_mid_trend_score(row)

    base_stock = stock_from_risk_score(risk, cfg)
    up_bonus = 0.0
    up_allowed = (hv <= cfg.up_high_vol_block) and (mid_state != "BEAR")
    if up_allowed and up > cfg.up_trigger:
        up_bonus = cfg.max_up_bonus * min(1.0, (up - cfg.up_trigger) / max(cfg.strong_up_trigger - cfg.up_trigger, 1e-9))

    stock = base_stock + up_bonus
    if up_allowed and up >= cfg.full_stock_up_trigger and hv <= cfg.full_stock_high_vol_max:
        stock = 1.0

    # Extra guards are rule-based, not ML down-risk.
    dd_score = drawdown_guard_score(row, cfg)
    tb_score = trend_break_score(row)
    stock -= cfg.drawdown_extra_cut * max(0.0, dd_score - 0.50) / 0.50
    stock -= cfg.trend_break_cut * tb_score
    stock = float(np.clip(stock, 0.0, 1.0))
    s, b, c = split_defensive(stock, cfg)

    if risk >= 0.90:
        regime = "EXTREME_RISK"
    elif risk >= 0.65:
        regime = "HIGH_VOL"
    elif risk >= 0.50:
        regime = "WATCH"
    elif up_allowed and up > cfg.up_trigger:
        regime = "CUSTOM_UP"
    else:
        regime = "NORMAL"

    return {
        "stock": s,
        "bond": b,
        "cash": c,
        "risk_score": risk,
        "drawdown_guard_score": dd_score,
        "trend_break_score": tb_score,
        "mid_trend_score": mid_score,
        "mid_trend_state": mid_state,
        "allocation_regime": regime,
        "up_overlay_bonus": up_bonus,
    }


def _normalize_weights(stock: float, bond: float, cash: float) -> Tuple[float, float, float]:
    arr = np.array([stock, bond, cash], dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    total = float(arr.sum())
    if total <= 0:
        return 0.0, 0.65, 0.35
    arr = arr / total
    return float(arr[0]), float(arr[1]), float(arr[2])



POLICY_PARAM_KEYS = [
    "up_trigger",
    "strong_up_trigger",
    "max_up_bonus",
    "full_stock_up_trigger",
    "full_stock_high_vol_max",
    "no_trade_band",
]


def current_policy_params(cfg: Config) -> Dict[str, float]:
    return {k: float(getattr(cfg, k)) for k in POLICY_PARAM_KEYS}


def cfg_with_policy_params(cfg: Config, params: Dict[str, float]) -> Config:
    data = asdict(cfg)
    for k in POLICY_PARAM_KEYS:
        if k in params:
            data[k] = float(params[k])
    return Config(**data)


def asset_policy_group(ticker: str) -> str:
    t = ticker.upper().strip()
    broad = {"SPY", "QQQ", "DIA", "IWM", "VOO", "IVV", "VTI"}
    sector = {"SOXX", "SMH", "XLK", "XLY", "XLF", "XLV", "XLE", "XLI", "XLC", "XLP", "XLU", "XLB"}
    high_vol_growth = {"NVDA", "TSLA", "AMD", "AVGO", "MU", "PLTR", "COIN", "MSTR"}
    if t in broad:
        return "broad_index"
    if t in high_vol_growth:
        return "aggressive_growth"
    if t in sector:
        return "sector_etf"
    return "mega_or_single_stock"


def _grid_product(grid: Dict[str, List[float]]) -> List[Dict[str, float]]:
    from itertools import product
    keys = list(grid.keys())
    out: List[Dict[str, float]] = []
    for values in product(*[grid[k] for k in keys]):
        d = {k: float(v) for k, v in zip(keys, values)}
        # Keep logically ordered thresholds.
        if d["strong_up_trigger"] < d["up_trigger"]:
            continue
        if d["full_stock_up_trigger"] < d["strong_up_trigger"]:
            continue
        out.append(d)
    return out


def policy_param_candidates(cfg: Config) -> List[Dict[str, float]]:
    mode = str(cfg.param_grid_mode or "asset_class").lower()
    group = asset_policy_group(cfg.target_ticker)

    common_grid = {
        "up_trigger": [0.45, 0.50],
        "strong_up_trigger": [0.65, 0.70],
        "max_up_bonus": [0.04, 0.08],
        "full_stock_up_trigger": [0.80, 0.85],
        "full_stock_high_vol_max": [0.20, 0.25],
        "no_trade_band": [0.12, 0.16],
    }
    broad_grid = {
        "up_trigger": [0.50, 0.55],
        "strong_up_trigger": [0.70, 0.75],
        "max_up_bonus": [0.03, 0.06],
        "full_stock_up_trigger": [0.85, 0.90],
        "full_stock_high_vol_max": [0.15, 0.20],
        "no_trade_band": [0.12, 0.16],
    }
    aggressive_grid = {
        "up_trigger": [0.40, 0.45],
        "strong_up_trigger": [0.60, 0.65],
        "max_up_bonus": [0.08, 0.12],
        "full_stock_up_trigger": [0.75, 0.80],
        "full_stock_high_vol_max": [0.25, 0.35],
        "no_trade_band": [0.12, 0.16],
    }

    if mode == "broad":
        grid = broad_grid
    elif mode == "aggressive":
        grid = aggressive_grid
    elif mode == "common":
        grid = common_grid
    elif group == "broad_index":
        grid = broad_grid
    elif group in {"sector_etf", "aggressive_growth"}:
        grid = aggressive_grid
    else:
        grid = common_grid
    return _grid_product(grid)


def target_mdd_limit(cfg: Config) -> float:
    group = asset_policy_group(cfg.target_ticker)
    if group == "broad_index":
        return 0.24 if cfg.target_ticker.upper() == "SPY" else 0.30
    if group == "sector_etf":
        return 0.38
    if group == "aggressive_growth":
        return 0.55
    return 0.32


def custom_up_limit(cfg: Config) -> float:
    group = asset_policy_group(cfg.target_ticker)
    if group == "broad_index":
        return 0.45
    if group in {"sector_etf", "aggressive_growth"}:
        return 0.55
    return 0.50


def _allocation_core(
    out: pd.DataFrame,
    cfg: Config,
    initial_weights: Tuple[float, float, float] = (0.72, 0.18, 0.10),
    start_i: int = 0,
) -> pd.DataFrame:
    """Fixed-policy allocation simulator used both for validation scoring and final OOS execution."""
    current = tuple(float(x) for x in initial_weights)
    last_emergency_i = -10_000
    rows = []
    for j, (dt, row) in enumerate(out.iterrows()):
        i = start_i + j
        ta = target_allocation(row, cfg)
        target = (ta["stock"], ta["bond"], ta["cash"])
        target = _normalize_weights(*target)

        due = (i % cfg.rebalance_every_n_days) == 0
        gap = float(np.sum(np.abs(np.array(target) - np.array(current))))
        emergency = bool(ta["risk_score"] >= cfg.emergency_risk_score_threshold and (i - last_emergency_i) >= cfg.emergency_cooldown_days)
        up_force = bool(ta["allocation_regime"] == "CUSTOM_UP" and gap > cfg.no_trade_band)

        if due or emergency or up_force:
            if gap >= cfg.no_trade_band or emergency or up_force:
                executed = target
                hold_reason = "executed"
                if emergency:
                    hold_reason = "emergency_risk_rebalance"
                    last_emergency_i = i
                elif up_force:
                    hold_reason = "up_force_rebalance"
            else:
                executed = current
                hold_reason = "held_by_no_trade_band"
        else:
            executed = current
            hold_reason = "held_by_schedule"

        turnover = float(np.sum(np.abs(np.array(executed) - np.array(current))))
        cost = turnover * cfg.transaction_cost_rate
        gross_ret = (
            executed[0] * float(row["target_ret_fwd"])
            + executed[1] * float(row["bond_ret_fwd"])
            + executed[2] * float(row["cash_ret_fwd"])
        )
        net_ret = gross_ret - cost
        current = executed

        rec = row.to_dict()
        rec.update({
            "signal_stock_weight": target[0],
            "signal_bond_weight": target[1],
            "signal_cash_weight": target[2],
            "stock_weight": executed[0],
            "bond_weight": executed[1],
            "cash_weight": executed[2],
            "turnover": turnover,
            "transaction_cost": cost,
            "strategy_return_gross": gross_ret,
            "strategy_return_net": net_ret,
            "hold_reason": hold_reason,
            **ta,
        })
        rows.append((dt, rec))
    if not rows:
        return pd.DataFrame(index=out.index)
    res = pd.DataFrame([r for _, r in rows], index=[dt for dt, _ in rows])
    res.index.name = "date"
    return res


def score_validation_segment(sim: pd.DataFrame, cfg: Config, previous_params: Optional[Dict[str, float]], params: Dict[str, float]) -> Dict[str, float]:
    if sim.empty or len(sim) < 40:
        return {
            "score": -1e9,
            "sharpe": np.nan,
            "calmar": np.nan,
            "mdd": np.nan,
            "annual_turnover": np.nan,
            "custom_up_ratio": np.nan,
            "changed_params": 0,
        }
    p = perf_stats(sim["strategy_return_net"], 1.0)
    sharpe = float(p.get("sharpe", np.nan)) if pd.notna(p.get("sharpe", np.nan)) else -5.0
    calmar = float(p.get("calmar", np.nan)) if pd.notna(p.get("calmar", np.nan)) else -5.0
    mdd = float(p.get("mdd", 0.0)) if pd.notna(p.get("mdd", np.nan)) else -1.0
    annual_turnover = float(sim["turnover"].mean() * 252) if "turnover" in sim else 0.0
    custom_up_ratio = float((sim["allocation_regime"] == "CUSTOM_UP").mean()) if "allocation_regime" in sim else 0.0

    mdd_penalty = 1.50 * max(0.0, abs(mdd) - target_mdd_limit(cfg))
    turnover_penalty = 0.08 * annual_turnover
    custom_up_penalty = 0.60 * max(0.0, custom_up_ratio - custom_up_limit(cfg))
    changed = 0
    if previous_params:
        changed = int(sum(abs(float(params.get(k, 0.0)) - float(previous_params.get(k, 0.0))) > 1e-12 for k in POLICY_PARAM_KEYS))
    change_penalty = cfg.param_change_penalty_per_change * changed
    score = float(sharpe + 0.80 * calmar - mdd_penalty - turnover_penalty - custom_up_penalty - change_penalty)
    return {
        "score": score,
        "sharpe": sharpe,
        "calmar": calmar,
        "mdd": mdd,
        "annual_turnover": annual_turnover,
        "custom_up_ratio": custom_up_ratio,
        "changed_params": changed,
        "mdd_penalty": mdd_penalty,
        "turnover_penalty": turnover_penalty,
        "custom_up_penalty": custom_up_penalty,
        "change_penalty": change_penalty,
    }


def select_policy_params(
    validation: pd.DataFrame,
    cfg: Config,
    previous_params: Dict[str, float],
    previous_score: float,
) -> Tuple[Dict[str, float], Dict[str, object], pd.DataFrame]:
    candidates = policy_param_candidates(cfg)
    rows = []
    best_params = previous_params.copy()
    best_metrics: Dict[str, object] = {"score": -1e9}
    for cand_id, params in enumerate(candidates):
        cand_cfg = cfg_with_policy_params(cfg, params)
        sim = _allocation_core(validation, cand_cfg)
        metrics = score_validation_segment(sim, cfg, previous_params, params)
        rec = {"candidate_id": cand_id, **params, **metrics}
        rows.append(rec)
        if float(metrics["score"]) > float(best_metrics.get("score", -1e9)):
            best_metrics = metrics
            best_params = params.copy()
    diag = pd.DataFrame(rows)
    kept_previous = False
    if previous_score > -1e8 and float(best_metrics.get("score", -1e9)) <= previous_score + cfg.param_min_improvement_margin:
        kept_previous = True
        best_params = previous_params.copy()
        best_metrics = {**best_metrics, "kept_previous_by_margin": True, "effective_score": previous_score}
    else:
        best_metrics = {**best_metrics, "kept_previous_by_margin": False, "effective_score": float(best_metrics.get("score", -1e9))}
    best_metrics["kept_previous"] = kept_previous
    return best_params, best_metrics, diag


def apply_allocation_fast_policy_update(pred: pd.DataFrame, asset_returns: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = pred.copy().join(asset_returns, how="left")
    out = out.dropna(subset=["target_ret_fwd", "bond_ret_fwd", "cash_ret_fwd"])
    if out.empty:
        return out

    current = (0.72, 0.18, 0.10)
    previous_target = current
    last_emergency_i = -10_000
    previous_params = current_policy_params(cfg)
    previous_score = -1e9
    active_cfg = cfg_with_policy_params(cfg, previous_params)

    history_rows = []
    candidate_diag_rows = []
    oos_segment_rows = []
    current_segment_start_dt = None
    current_segment_returns: List[float] = []
    current_segment_turnover: List[float] = []
    current_segment_params = previous_params.copy()

    rows = []
    n = len(out)
    for i, (dt, row) in enumerate(out.iterrows()):
        if i == 0 or (i % cfg.param_update_every) == 0:
            # Flush previous OOS segment diagnostic.
            if current_segment_start_dt is not None and current_segment_returns:
                seg_r = pd.Series(current_segment_returns)
                oos_segment_rows.append({
                    "segment_start": str(current_segment_start_dt),
                    "segment_end": str(prev_dt),
                    "rows": len(current_segment_returns),
                    "segment_return": float((1.0 + seg_r).prod() - 1.0),
                    "avg_turnover": float(np.mean(current_segment_turnover)) if current_segment_turnover else 0.0,
                    **{f"param_{k}": v for k, v in current_segment_params.items()},
                })
            current_segment_start_dt = dt
            current_segment_returns = []
            current_segment_turnover = []

            val_end_pos = max(0, i - cfg.param_embargo_days)
            val_start_pos = max(0, val_end_pos - cfg.param_validation_window)
            has_validation = (val_end_pos - val_start_pos) >= cfg.param_min_history
            selected_params = previous_params.copy()
            metrics: Dict[str, object] = {
                "score": previous_score,
                "kept_previous": True,
                "reason": "insufficient_validation_history",
            }
            diag = pd.DataFrame()
            if has_validation:
                validation = out.iloc[val_start_pos:val_end_pos].copy()
                selected_params, metrics, diag = select_policy_params(validation, cfg, previous_params, previous_score)
                if not diag.empty:
                    diag.insert(0, "selection_date", str(dt))
                    diag.insert(1, "validation_start", str(validation.index.min()))
                    diag.insert(2, "validation_end", str(validation.index.max()))
                    candidate_diag_rows.append(diag)
            active_cfg = cfg_with_policy_params(cfg, selected_params)
            changed_count = int(sum(abs(float(selected_params.get(k, 0.0)) - float(previous_params.get(k, 0.0))) > 1e-12 for k in POLICY_PARAM_KEYS))
            history_rows.append({
                "selection_date": str(dt),
                "row_index": i,
                "validation_start": str(out.index[val_start_pos]) if has_validation and val_start_pos < n else None,
                "validation_end": str(out.index[val_end_pos - 1]) if has_validation and val_end_pos > val_start_pos else None,
                "has_validation": bool(has_validation),
                "changed_count": changed_count,
                **{f"param_{k}": float(selected_params[k]) for k in POLICY_PARAM_KEYS},
                **{f"metric_{k}": v for k, v in metrics.items() if isinstance(v, (int, float, bool, str, np.floating, np.integer))},
            })
            previous_params = selected_params.copy()
            previous_score = float(metrics.get("effective_score", metrics.get("score", previous_score))) if has_validation else previous_score
            current_segment_params = selected_params.copy()

        ta = target_allocation(row, active_cfg)
        target = (ta["stock"], ta["bond"], ta["cash"])
        target = _normalize_weights(*target)
        if cfg.target_weight_ewma_alpha < 1.0:
            alpha = float(np.clip(cfg.target_weight_ewma_alpha, 0.0, 1.0))
            target = tuple(float(alpha * target[k] + (1.0 - alpha) * previous_target[k]) for k in range(3))
            target = _normalize_weights(*target)
        previous_target = target

        due = (i % cfg.rebalance_every_n_days) == 0
        gap = float(np.sum(np.abs(np.array(target) - np.array(current))))
        emergency = bool(ta["risk_score"] >= active_cfg.emergency_risk_score_threshold and (i - last_emergency_i) >= active_cfg.emergency_cooldown_days)
        up_force = bool(ta["allocation_regime"] == "CUSTOM_UP" and gap > active_cfg.no_trade_band)

        if due or emergency or up_force:
            if gap >= active_cfg.no_trade_band or emergency or up_force:
                executed = target
                hold_reason = "executed"
                if emergency:
                    hold_reason = "emergency_risk_rebalance"
                    last_emergency_i = i
                elif up_force:
                    hold_reason = "up_force_rebalance"
            else:
                executed = current
                hold_reason = "held_by_no_trade_band"
        else:
            executed = current
            hold_reason = "held_by_schedule"

        turnover = float(np.sum(np.abs(np.array(executed) - np.array(current))))
        cost = turnover * active_cfg.transaction_cost_rate
        gross_ret = (
            executed[0] * float(row["target_ret_fwd"])
            + executed[1] * float(row["bond_ret_fwd"])
            + executed[2] * float(row["cash_ret_fwd"])
        )
        net_ret = gross_ret - cost
        current = executed
        current_segment_returns.append(net_ret)
        current_segment_turnover.append(turnover)

        rec = row.to_dict()
        rec.update({
            "signal_stock_weight": target[0],
            "signal_bond_weight": target[1],
            "signal_cash_weight": target[2],
            "stock_weight": executed[0],
            "bond_weight": executed[1],
            "cash_weight": executed[2],
            "turnover": turnover,
            "transaction_cost": cost,
            "strategy_return_gross": gross_ret,
            "strategy_return_net": net_ret,
            "hold_reason": hold_reason,
            **ta,
            **{f"policy_{k}": float(getattr(active_cfg, k)) for k in POLICY_PARAM_KEYS},
        })
        rows.append((dt, rec))
        prev_dt = dt

    if current_segment_start_dt is not None and current_segment_returns:
        seg_r = pd.Series(current_segment_returns)
        oos_segment_rows.append({
            "segment_start": str(current_segment_start_dt),
            "segment_end": str(prev_dt),
            "rows": len(current_segment_returns),
            "segment_return": float((1.0 + seg_r).prod() - 1.0),
            "avg_turnover": float(np.mean(current_segment_turnover)) if current_segment_turnover else 0.0,
            **{f"param_{k}": v for k, v in current_segment_params.items()},
        })

    res = pd.DataFrame([r for _, r in rows], index=[dt for dt, _ in rows])
    res.index.name = "date"
    res["strategy_equity_net"] = cfg.initial_capital * (1.0 + res["strategy_return_net"].fillna(0.0)).cumprod()
    res["strategy_equity_gross"] = cfg.initial_capital * (1.0 + res["strategy_return_gross"].fillna(0.0)).cumprod()
    res["stock_buyhold_equity"] = cfg.initial_capital * (1.0 + res["target_ret_fwd"].fillna(0.0)).cumprod()
    res["benchmark_60_40_return"] = 0.60 * res["target_ret_fwd"] + 0.40 * res["bond_ret_fwd"]
    res["benchmark_60_40_equity"] = cfg.initial_capital * (1.0 + res["benchmark_60_40_return"].fillna(0.0)).cumprod()
    res["static_50_30_20_return"] = 0.50 * res["target_ret_fwd"] + 0.30 * res["bond_ret_fwd"] + 0.20 * res["cash_ret_fwd"]
    res["static_50_30_20_equity"] = cfg.initial_capital * (1.0 + res["static_50_30_20_return"].fillna(0.0)).cumprod()
    res.attrs["param_history"] = pd.DataFrame(history_rows)
    res.attrs["param_selection_diagnostics"] = pd.concat(candidate_diag_rows, ignore_index=True) if candidate_diag_rows else pd.DataFrame()
    res.attrs["oos_5d_segment_performance"] = pd.DataFrame(oos_segment_rows)
    return res


def apply_allocation(pred: pd.DataFrame, asset_returns: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if bool(getattr(cfg, "enable_fast_policy_update", False)):
        return apply_allocation_fast_policy_update(pred, asset_returns, cfg)

    out = pred.copy().join(asset_returns, how="left")
    out = out.dropna(subset=["target_ret_fwd", "bond_ret_fwd", "cash_ret_fwd"])

    current = (0.72, 0.18, 0.10)
    last_emergency_i = -10_000
    rows = []
    for i, (dt, row) in enumerate(out.iterrows()):
        ta = target_allocation(row, cfg)
        target = (ta["stock"], ta["bond"], ta["cash"])
        target = _normalize_weights(*target)

        due = (i % cfg.rebalance_every_n_days) == 0
        gap = float(np.sum(np.abs(np.array(target) - np.array(current))))
        emergency = bool(ta["risk_score"] >= cfg.emergency_risk_score_threshold and (i - last_emergency_i) >= cfg.emergency_cooldown_days)
        up_force = bool(ta["allocation_regime"] == "CUSTOM_UP" and gap > cfg.no_trade_band)

        if due or emergency or up_force:
            if gap >= cfg.no_trade_band or emergency or up_force:
                executed = target
                hold_reason = "executed"
                if emergency:
                    hold_reason = "emergency_risk_rebalance"
                    last_emergency_i = i
                elif up_force:
                    hold_reason = "up_force_rebalance"
            else:
                executed = current
                hold_reason = "held_by_no_trade_band"
        else:
            executed = current
            hold_reason = "held_by_schedule"

        turnover = float(np.sum(np.abs(np.array(executed) - np.array(current))))
        cost = turnover * cfg.transaction_cost_rate
        gross_ret = (
            executed[0] * float(row["target_ret_fwd"])
            + executed[1] * float(row["bond_ret_fwd"])
            + executed[2] * float(row["cash_ret_fwd"])
        )
        net_ret = gross_ret - cost
        current = executed

        rec = row.to_dict()
        rec.update({
            "signal_stock_weight": target[0],
            "signal_bond_weight": target[1],
            "signal_cash_weight": target[2],
            "stock_weight": executed[0],
            "bond_weight": executed[1],
            "cash_weight": executed[2],
            "turnover": turnover,
            "transaction_cost": cost,
            "strategy_return_gross": gross_ret,
            "strategy_return_net": net_ret,
            "hold_reason": hold_reason,
            **ta,
        })
        rows.append((dt, rec))

    res = pd.DataFrame([r for _, r in rows], index=[dt for dt, _ in rows])
    res.index.name = "date"
    res["strategy_equity_net"] = cfg.initial_capital * (1.0 + res["strategy_return_net"].fillna(0.0)).cumprod()
    res["strategy_equity_gross"] = cfg.initial_capital * (1.0 + res["strategy_return_gross"].fillna(0.0)).cumprod()
    res["stock_buyhold_equity"] = cfg.initial_capital * (1.0 + res["target_ret_fwd"].fillna(0.0)).cumprod()
    res["benchmark_60_40_return"] = 0.60 * res["target_ret_fwd"] + 0.40 * res["bond_ret_fwd"]
    res["benchmark_60_40_equity"] = cfg.initial_capital * (1.0 + res["benchmark_60_40_return"].fillna(0.0)).cumprod()
    res["static_50_30_20_return"] = 0.50 * res["target_ret_fwd"] + 0.30 * res["bond_ret_fwd"] + 0.20 * res["cash_ret_fwd"]
    res["static_50_30_20_equity"] = cfg.initial_capital * (1.0 + res["static_50_30_20_return"].fillna(0.0)).cumprod()
    return res


# ============================================================
# 6. METRICS / DIAGNOSTICS
# ============================================================

def perf_stats(returns: pd.Series, initial_capital: float) -> Dict[str, float]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {"final_capital": initial_capital, "total_return": 0.0, "cagr": np.nan, "mdd": np.nan, "sharpe": np.nan, "sortino": np.nan, "calmar": np.nan}
    equity = initial_capital * (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0)
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = float(dd.min())
    vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    downside = r[r < 0]
    sortino = float(r.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else np.nan
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan
    return {
        "final_capital": float(equity.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "ann_vol": vol,
    }


def safe_auc(y_true: np.ndarray, p: np.ndarray, kind: str) -> Optional[float]:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, p) if kind == "roc" else average_precision_score(y_true, p))
    except Exception:
        return None


def binary_metrics(y_true: pd.Series, prob: pd.Series, threshold: float, pos_name: str) -> Dict[str, object]:
    m = pd.DataFrame({"y": y_true, "p": prob}).dropna()
    if m.empty:
        return {}
    y = m["y"].astype(int).values
    p = m["p"].astype(float).values
    pred = (p >= threshold).astype(int)
    out = {
        "rows": int(len(m)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "support_positive": int(y.sum()),
        "support_negative": int(len(y) - y.sum()),
        "pred_positive_ratio": float(pred.mean()),
        "positive_class": pos_name,
        "roc_auc": safe_auc(y, p, "roc"),
        "pr_auc": safe_auc(y, p, "pr"),
    }
    return out


def annual_returns(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df.index.year)
    rows = []
    for year, x in g:
        rows.append({
            "year": int(year),
            "strategy_return": float((1 + x["strategy_return_net"]).prod() - 1),
            "stock_buyhold_return": float((1 + x["target_ret_fwd"]).prod() - 1),
            "benchmark_60_40_return": float((1 + x["benchmark_60_40_return"]).prod() - 1),
            "static_50_30_20_return": float((1 + x["static_50_30_20_return"]).prod() - 1),
            "strategy_excess_vs_bh": float(((1 + x["strategy_return_net"]).prod() - 1) - ((1 + x["target_ret_fwd"]).prod() - 1)),
        })
    return pd.DataFrame(rows)


def periodic_returns(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # pandas 3.x compatibility: legacy aliases like "M"/"Y" are no longer accepted.
    # Keep backward-compatible caller inputs and normalize them here.
    freq_alias = {
        "M": "ME",      # month-end
        "Y": "YE",      # year-end
        "A": "YE",      # year-end legacy alias
        "Q": "QE",      # quarter-end legacy alias
    }.get(str(freq).upper(), freq)
    g = df.groupby(pd.Grouper(freq=freq_alias))
    rows = []
    for dt, x in g:
        if x.empty:
            continue
        rows.append({
            "period": str(dt.date()),
            "strategy_return": float((1 + x["strategy_return_net"]).prod() - 1),
            "stock_buyhold_return": float((1 + x["target_ret_fwd"]).prod() - 1),
            "benchmark_60_40_return": float((1 + x["benchmark_60_40_return"]).prod() - 1),
        })
    return pd.DataFrame(rows)


def regime_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, x in df.groupby("allocation_regime"):
        if x.empty:
            continue
        rows.append({
            "regime": regime,
            "count": int(len(x)),
            "pct": float(len(x) / len(df)),
            "ann_return_est": float((1 + x["strategy_return_net"].mean()) ** 252 - 1) if len(x) > 0 else np.nan,
            "ann_vol_est": float(x["strategy_return_net"].std() * np.sqrt(252)),
            "win_rate": float((x["strategy_return_net"] > 0).mean()),
            "avg_stock_weight": float(x["stock_weight"].mean()),
            "avg_prob_high_vol": float(x["prob_high_vol"].mean()),
            "avg_prob_up_strength": float(x["prob_up_strengthening_20d"].mean()),
            "avg_risk_score": float(x["risk_score"].mean()),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def probability_bins(df: pd.DataFrame, prob_col: str, y_col: str, bins: int = 10) -> pd.DataFrame:
    if prob_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()
    x = df[[prob_col, y_col, "strategy_return_net", "stock_weight"]].dropna().copy()
    if x.empty:
        return pd.DataFrame()
    x["bin"] = pd.cut(x[prob_col], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    rows = []
    for b, g in x.groupby("bin", observed=False):
        if g.empty:
            continue
        rows.append({
            "prob_col": prob_col,
            "bin": str(b),
            "count": int(len(g)),
            "avg_prob": float(g[prob_col].mean()),
            "actual_rate": float(g[y_col].astype(float).mean()),
            "ann_return_est": float((1 + g["strategy_return_net"].mean()) ** 252 - 1),
            "avg_stock_weight": float(g["stock_weight"].mean()),
        })
    return pd.DataFrame(rows)


def drawdown_episodes(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    equity = df["strategy_equity_net"].dropna()
    if equity.empty:
        return pd.DataFrame()
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    episodes = []
    in_dd = False
    start = trough = None
    trough_depth = 0.0
    for dt, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            start = dt
            trough = dt
            trough_depth = float(val)
        elif val < 0 and in_dd:
            if float(val) < trough_depth:
                trough_depth = float(val)
                trough = dt
        elif val >= 0 and in_dd:
            seg = df.loc[start:dt]
            episodes.append({
                "start_date": str(start.date()),
                "trough_date": str(trough.date()),
                "recovery_date": str(dt.date()),
                "depth": trough_depth,
                "duration_days": int((dt - start).days),
                "days_to_trough": int((trough - start).days),
                "avg_stock_weight": float(seg["stock_weight"].mean()),
                "avg_prob_high_vol": float(seg["prob_high_vol"].mean()),
                "avg_risk_score": float(seg["risk_score"].mean()),
                "trough_regime": str(df.loc[trough, "allocation_regime"]),
            })
            in_dd = False
    if in_dd and start is not None and trough is not None:
        dt = equity.index[-1]
        seg = df.loc[start:dt]
        episodes.append({
            "start_date": str(start.date()),
            "trough_date": str(trough.date()),
            "recovery_date": "NOT_RECOVERED",
            "depth": trough_depth,
            "duration_days": int((dt - start).days),
            "days_to_trough": int((trough - start).days),
            "avg_stock_weight": float(seg["stock_weight"].mean()),
            "avg_prob_high_vol": float(seg["prob_high_vol"].mean()),
            "avg_risk_score": float(seg["risk_score"].mean()),
            "trough_regime": str(df.loc[trough, "allocation_regime"]),
        })
    return pd.DataFrame(episodes).sort_values("depth").head(top_n)


def build_summary(df: pd.DataFrame, feature_cols: List[str], imps: Dict[str, Dict[str, float]], cfg: Config) -> Dict[str, object]:
    classification = {
        "high_vol": binary_metrics(df["y_high_vol"], df["prob_high_vol"], 0.50, "고변동"),
        "up_strength_20d": binary_metrics(df["y_up_strength_20d"], df["prob_up_strengthening_20d"], cfg.up_trigger, "UP_STRENGTHENING_20D"),
    }
    perf = {
        "strategy_after_cost": perf_stats(df["strategy_return_net"], cfg.initial_capital),
        "strategy_gross": perf_stats(df["strategy_return_gross"], cfg.initial_capital),
        "stock_buy_hold": perf_stats(df["target_ret_fwd"], cfg.initial_capital),
        "benchmark_60_40": perf_stats(df["benchmark_60_40_return"], cfg.initial_capital),
        "static_50_30_20": perf_stats(df["static_50_30_20_return"], cfg.initial_capital),
    }
    latest = latest_prediction(df)
    summary = {
        "model_type": cfg.model_version,
        "removed_components": [
            "downrisk_head",
            "downrisk_multi_branch_ensemble",
            "direction_binary_prob_up_down",
            "portfolio_policy_model",
            "tier_weight_optimizer",
            "condition_search",
            "adaptive_label_policy",
            "weak_probability_outputs",
        ],
        "kept_components": [
            "highvol_head",
            "up_strength_20d_head",
            "drawdown_guard_rule",
            "trend_break_rule",
            "volatility_scaled_up_label",
            "asset_local_highvol_quantile_label",
            "walk_forward_training",
            "transaction_cost_backtest",
            "scheduled_policy_parameter_update_5d",
        ],
        "target_ticker": cfg.target_ticker,
        "bond_ticker": cfg.bond_ticker,
        "cash_ticker": cfg.cash_ticker,
        "config": asdict(cfg),
        "period": {
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "rows": int(len(df)),
        },
        "feature_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "average_probabilities": {
            "avg_prob_normal": float(df["prob_normal"].mean()),
            "avg_prob_high_vol": float(df["prob_high_vol"].mean()),
            "avg_prob_up_strengthening_20d": float(df["prob_up_strengthening_20d"].mean()),
            "avg_risk_score": float(df["risk_score"].mean()),
        },
        "average_weights": {
            "avg_stock_weight": float(df["stock_weight"].mean()),
            "avg_bond_weight": float(df["bond_weight"].mean()),
            "avg_cash_weight": float(df["cash_weight"].mean()),
            "min_stock_weight": float(df["stock_weight"].min()),
            "max_stock_weight": float(df["stock_weight"].max()),
        },
        "allocation_regime_distribution_pct": {k: float(v) for k, v in (df["allocation_regime"].value_counts(normalize=True) * 100).round(2).to_dict().items()},
        "turnover": {
            "avg_daily_trade_ratio": float(df["turnover"].mean()),
            "annual_turnover_estimate": float(df["turnover"].mean() * 252),
            "total_transaction_cost_rate_sum": float(df["transaction_cost"].sum()),
            "trade_executed_ratio": float((df["turnover"] > 0).mean()),
            "emergency_rebalance_ratio": float((df["hold_reason"] == "emergency_risk_rebalance").mean()),
        },
        "scheduled_policy_update": {
            "enabled": bool(getattr(cfg, "enable_fast_policy_update", False)),
            "update_every": int(getattr(cfg, "param_update_every", 0)),
            "validation_window": int(getattr(cfg, "param_validation_window", 0)),
            "embargo_days": int(getattr(cfg, "param_embargo_days", 0)),
            "asset_policy_group": asset_policy_group(cfg.target_ticker),
            "avg_selected_up_trigger": float(df["policy_up_trigger"].mean()) if "policy_up_trigger" in df else None,
            "avg_selected_max_up_bonus": float(df["policy_max_up_bonus"].mean()) if "policy_max_up_bonus" in df else None,
            "unique_policy_count": int(df[[f"policy_{k}" for k in POLICY_PARAM_KEYS if f"policy_{k}" in df]].drop_duplicates().shape[0]) if any(f"policy_{k}" in df for k in POLICY_PARAM_KEYS) else 0,
        },
        "performance": perf,
        "classification": classification,
        "latest_prediction": latest,
        **imps,
    }
    return summary


def latest_prediction(df: pd.DataFrame) -> Dict[str, object]:
    row = df.iloc[-1]
    return {
        "date": str(df.index[-1]),
        "pred_risk": str(row.get("pred_risk", "")),
        "pred_up_strength": str(row.get("pred_up_strength", "")),
        "prob_normal": round(float(row.get("prob_normal", 0.0)) * 100, 2),
        "prob_high_vol": round(float(row.get("prob_high_vol", 0.0)) * 100, 2),
        "prob_up_strengthening_20d": round(float(row.get("prob_up_strengthening_20d", 0.0)) * 100, 2),
        "risk_score": round(float(row.get("risk_score", 0.0)) * 100, 2),
        "drawdown_guard_score": round(float(row.get("drawdown_guard_score", 0.0)) * 100, 2),
        "trend_break_score": round(float(row.get("trend_break_score", 0.0)) * 100, 2),
        "mid_trend_score": int(row.get("mid_trend_score", 0)),
        "mid_trend_state": str(row.get("mid_trend_state", "")),
        "allocation_regime": str(row.get("allocation_regime", "")),
        "hold_reason": str(row.get("hold_reason", "")),
        "executed_allocation": {
            "stock": round(float(row.get("stock_weight", 0.0)) * 100, 2),
            "bond": round(float(row.get("bond_weight", 0.0)) * 100, 2),
            "cash": round(float(row.get("cash_weight", 0.0)) * 100, 2),
        },
        "active_policy_params": {
            k: round(float(row.get(f"policy_{k}", np.nan)), 4) for k in POLICY_PARAM_KEYS if f"policy_{k}" in row.index
        },
        "removed_outputs": [
            "prob_down_strengthening_score",
            "prob_down",
            "prob_up",
            "prob_overall_risk_by_downrisk",
        ],
    }


# ============================================================
# 7. ORCHESTRATION
# ============================================================

def build_asset_returns(target: pd.Series, bond: pd.Series, cash: pd.Series, cfg: Config) -> pd.DataFrame:
    px = pd.concat([target.rename("target"), bond.rename("bond"), cash.rename("cash")], axis=1).dropna()
    lag = int(max(cfg.execution_lag_days, 0))
    returns = pd.DataFrame(index=px.index)
    # Signal at t. With lag=1, return from t+1 to t+2.
    returns["target_ret_fwd"] = px["target"].pct_change().shift(-(lag + 1))
    returns["bond_ret_fwd"] = px["bond"].pct_change().shift(-(lag + 1))
    returns["cash_ret_fwd"] = px["cash"].pct_change().shift(-(lag + 1))
    return returns


def save_outputs(df: pd.DataFrame, summary: Dict[str, object], diagnostics: Dict[str, pd.DataFrame], cfg: Config) -> None:
    outdir = Path(cfg.result_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{cfg.target_ticker.lower()}_xgb_universal_minimal_v8_8_1_fast5"

    pred_path = outdir / f"{prefix}_predictions.csv"
    summary_path = outdir / f"{prefix}_summary.json"
    latest_path = outdir / f"{prefix}_latest.json"

    df.to_csv(pred_path, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    latest_path.write_text(json.dumps(summary["latest_prediction"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    for name, table in diagnostics.items():
        if table is not None and not table.empty:
            table.to_csv(outdir / f"{prefix}_{name}.csv", index=False, encoding="utf-8-sig")

    # Feature importance tables
    for key in ["highvol_feature_importance_mean", "up_feature_importance_mean"]:
        imp = summary.get(key, {})
        if isinstance(imp, dict) and imp:
            pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())}).sort_values(
                "importance", ascending=False
            ).to_csv(outdir / f"{prefix}_{key}.csv", index=False, encoding="utf-8-sig")


def run_single(cfg: Config) -> Dict[str, object]:
    print(f"\n[RUN] {cfg.model_version} target={cfg.target_ticker}")
    ohlcv = download_ohlcv(cfg.target_ticker, cfg.start_date, cfg.end_date)
    bond_close = download_close(cfg.bond_ticker, cfg.start_date, cfg.end_date)
    try:
        cash_close = download_close(cfg.cash_ticker, cfg.start_date, cfg.end_date)
    except Exception as exc:
        if not cfg.allow_cash_download_fallback:
            raise
        print(f"[WARN] {cfg.cash_ticker} 다운로드 실패. 현금 수익률 0으로 대체합니다: {exc}")
        cash_close = pd.Series(index=ohlcv.index, data=1.0, name=cfg.cash_ticker)

    feat_df, feature_cols = build_features(ohlcv, cfg)
    pred, imps = run_walk_forward(feat_df, feature_cols, cfg)
    rets = build_asset_returns(ohlcv["close"], bond_close, cash_close, cfg)
    result = apply_allocation(pred, rets, cfg)

    diagnostics = {
        "annual_returns": annual_returns(result),
        "monthly_returns": periodic_returns(result, "ME"),
        "regime_analysis": regime_analysis(result),
        "drawdown_episodes": drawdown_episodes(result),
        "probability_bins_highvol": probability_bins(result, "prob_high_vol", "y_high_vol"),
        "probability_bins_up20": probability_bins(result, "prob_up_strengthening_20d", "y_up_strength_20d"),
        "param_history": result.attrs.get("param_history", pd.DataFrame()),
        "param_selection_diagnostics": result.attrs.get("param_selection_diagnostics", pd.DataFrame()),
        "oos_5d_segment_performance": result.attrs.get("oos_5d_segment_performance", pd.DataFrame()),
    }
    summary = build_summary(result, feature_cols, imps, cfg)
    save_outputs(result, summary, diagnostics, cfg)

    p = summary["performance"]["strategy_after_cost"]
    bh = summary["performance"]["stock_buy_hold"]
    print(f"[DONE] {cfg.target_ticker}: CAGR={p['cagr']:.2%}, MDD={p['mdd']:.2%}, Sharpe={p['sharpe']:.3f}, B&H CAGR={bh['cagr']:.2%}")
    return summary


def apply_speed_profile(cfg: Config, profile: str) -> Config:
    profile = profile.lower()
    if profile == "fast":
        cfg.n_estimators = 80
        cfg.retrain_every_n_days = 30
        cfg.max_train_rows = 1000
    elif profile == "balanced":
        cfg.n_estimators = 120
        cfg.retrain_every_n_days = 20
        cfg.max_train_rows = 1260
    elif profile == "full":
        cfg.n_estimators = 180
        cfg.retrain_every_n_days = 10
        cfg.max_train_rows = None
    else:
        raise ValueError("speed-profile은 fast/balanced/full 중 하나여야 합니다.")
    return cfg


def preset_assets(name: str) -> List[str]:
    presets = {
        "etf": ["QQQ", "SPY", "IWM", "DIA", "XLK", "SMH", "SOXX", "XLY", "XLF", "XLV"],
        "mega": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO"],
        "mixed": ["QQQ", "SPY", "IWM", "XLK", "SMH", "SOXX", "AAPL", "MSFT", "NVDA", "TSLA"],
        "semis": ["QQQ", "SMH", "SOXX", "NVDA", "AVGO", "AMD", "TSM", "ASML", "QCOM", "MU"],
    }
    if name not in presets:
        raise ValueError(f"지원하지 않는 asset preset: {name}")
    return presets[name]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XGBoost Universal Minimal v8.8.1 - fast 5-day policy parameter update, no Down-risk")
    p.add_argument("--target-ticker", default="QQQ")
    p.add_argument("--bond-ticker", default="IEF")
    p.add_argument("--cash-ticker", default="BIL")
    p.add_argument("--asset-list", default="", help="콤마 구분 다종목 예: QQQ,SPY,AAPL,SOXX,NVDA")
    p.add_argument("--asset-preset", default="", choices=["", "etf", "mega", "mixed", "semis"])
    p.add_argument("--start-date", default="1999-03-10")
    p.add_argument("--backtest-start-date", default="2013-01-02")
    p.add_argument("--end-date", default=None)
    p.add_argument("--initial-capital", type=float, default=100_000_000)
    p.add_argument("--transaction-cost-rate", type=float, default=0.001)
    p.add_argument("--execution-lag-days", type=int, default=1)
    p.add_argument("--speed-profile", default="balanced", choices=["fast", "balanced", "full"])
    p.add_argument("--result-dir", default="results_v8_8_1_fast5")
    p.add_argument("--allow-cash-download-fallback", action="store_true")

    # Key ablation knobs
    p.add_argument("--high-vol-quantile", type=float, default=0.80)
    p.add_argument("--up-strength-k", type=float, default=0.55)
    p.add_argument("--up-trigger", type=float, default=0.35)
    p.add_argument("--strong-up-trigger", type=float, default=0.55)
    p.add_argument("--up-high-vol-block", type=float, default=0.58)
    p.add_argument("--no-trade-band", type=float, default=0.12)
    p.add_argument("--rebalance-every", type=int, default=5)
    p.add_argument("--disable-prob-ewma", action="store_true")

    # v8.8.1 scheduled policy update knobs
    p.add_argument("--disable-fast-policy-update", action="store_true")
    p.add_argument("--param-update-every", type=int, default=5)
    p.add_argument("--param-validation-window", type=int, default=252)
    p.add_argument("--param-embargo-days", type=int, default=20)
    p.add_argument("--param-min-history", type=int, default=252)
    p.add_argument("--param-min-improvement-margin", type=float, default=0.03)
    p.add_argument("--param-change-penalty", type=float, default=0.03)
    p.add_argument("--param-grid-mode", default="asset_class", choices=["asset_class", "broad", "aggressive", "common"])
    p.add_argument("--target-weight-ewma-alpha", type=float, default=1.0)
    return p.parse_args()


def summary_row(summary: Dict[str, object]) -> Dict[str, object]:
    p = summary["performance"]["strategy_after_cost"]
    bh = summary["performance"]["stock_buy_hold"]
    b6040 = summary["performance"]["benchmark_60_40"]
    w = summary["average_weights"]
    c = summary["classification"]
    t = summary["turnover"]
    return {
        "ticker": summary["target_ticker"],
        "strategy_cagr": p.get("cagr"),
        "strategy_mdd": p.get("mdd"),
        "strategy_sharpe": p.get("sharpe"),
        "strategy_calmar": p.get("calmar"),
        "buyhold_cagr": bh.get("cagr"),
        "buyhold_mdd": bh.get("mdd"),
        "benchmark_60_40_cagr": b6040.get("cagr"),
        "excess_cagr_vs_bh": p.get("cagr") - bh.get("cagr"),
        "mdd_improvement_vs_bh": p.get("mdd") - bh.get("mdd"),
        "avg_stock_weight": w.get("avg_stock_weight"),
        "annual_turnover_estimate": t.get("annual_turnover_estimate"),
        "highvol_roc_auc": c.get("high_vol", {}).get("roc_auc"),
        "highvol_pr_auc": c.get("high_vol", {}).get("pr_auc"),
        "up20_roc_auc": c.get("up_strength_20d", {}).get("roc_auc"),
        "up20_pr_auc": c.get("up_strength_20d", {}).get("pr_auc"),
    }


def main() -> None:
    args = parse_args()
    tickers: List[str]
    if args.asset_list.strip():
        tickers = [x.strip().upper() for x in args.asset_list.split(",") if x.strip()]
    elif args.asset_preset.strip():
        tickers = preset_assets(args.asset_preset.strip())
    else:
        tickers = [args.target_ticker.upper()]

    summaries = []
    for ticker in tickers:
        cfg = Config(
            target_ticker=ticker,
            bond_ticker=args.bond_ticker.upper(),
            cash_ticker=args.cash_ticker.upper(),
            start_date=args.start_date,
            backtest_start_date=args.backtest_start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital,
            transaction_cost_rate=args.transaction_cost_rate,
            execution_lag_days=args.execution_lag_days,
            result_dir=str(Path(args.result_dir) / ticker.lower()) if len(tickers) > 1 else args.result_dir,
            high_vol_quantile=args.high_vol_quantile,
            up_strength_k=args.up_strength_k,
            up_trigger=args.up_trigger,
            strong_up_trigger=args.strong_up_trigger,
            up_high_vol_block=args.up_high_vol_block,
            no_trade_band=args.no_trade_band,
            rebalance_every_n_days=args.rebalance_every,
            use_prob_ewma=not args.disable_prob_ewma,
            allow_cash_download_fallback=args.allow_cash_download_fallback,
            enable_fast_policy_update=not args.disable_fast_policy_update,
            param_update_every=args.param_update_every,
            param_validation_window=args.param_validation_window,
            param_embargo_days=args.param_embargo_days,
            param_min_history=args.param_min_history,
            param_min_improvement_margin=args.param_min_improvement_margin,
            param_change_penalty_per_change=args.param_change_penalty,
            param_grid_mode=args.param_grid_mode,
            target_weight_ewma_alpha=args.target_weight_ewma_alpha,
        )
        cfg = apply_speed_profile(cfg, args.speed_profile)
        summaries.append(run_single(cfg))

    if len(summaries) > 1:
        root = Path(args.result_dir)
        root.mkdir(parents=True, exist_ok=True)
        comp = pd.DataFrame([summary_row(s) for s in summaries])
        comp = comp.sort_values(["strategy_sharpe", "strategy_cagr"], ascending=[False, False])
        comp.to_csv(root / "v8_8_1_fast5_multi_asset_summary.csv", index=False, encoding="utf-8-sig")
        try:
            (root / "v8_8_1_fast5_multi_asset_summary.md").write_text(comp.to_markdown(index=False), encoding="utf-8")
        except Exception:
            pass
        print(f"\n[MULTI] saved: {root / 'v8_8_1_fast5_multi_asset_summary.csv'}")


if __name__ == "__main__":
    main()
