# -*- coding: utf-8 -*-
"""
downside_dual_head_experiment_v7_1.py

7.1 Downside Dual-Head Experiment
=================================

목적
----
기존 단일 Defensive/Down head의 문제(Brier Skill 악화, decile inversion, risk-off trigger 부적합)를 개선하기 위해
하방 탐지를 2중 구조로 분리합니다.

구조
----
1) Down Candidate Head
   - 라벨: y_down_touch
   - 의미: 향후 H거래일 안에 intraday low가 변동성 기반 하단 barrier를 터치했는가?
   - 역할: high-recall 후보 탐지

2) Down Confirm Head
   - 라벨: y_close_mdd_confirm
   - 의미: 향후 H거래일 안에 종가 기준 최대하락(close drawdown)이 변동성 기준을 넘었는가?
   - 역할: precision-oriented 방어 확인 필터

3) Dual Downside Policy
   - down_candidate_score_percentile >= threshold_candidate
   - down_confirm_score_percentile >= threshold_confirm
   - 둘 다 만족할 때만 CONFIRMED_DOWNSIDE_RISK

주의
----
- score_percentile은 실제 확률이 아니라 calibration-window 내 순위 점수입니다.
- 이 스크립트는 하방 head 진단/개선용입니다.
- 최종 포트폴리오 비중/자동매매 신호가 아닙니다.

필수 입력
---------
개별 OHLCV CSV:
  --ohlcv-inputs "QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv"
  --asset-names "QQQ,SPY,SOXX,XLK"

또는 통합 CSV:
  --ohlcv-all "ohlcv_all_tickers.csv"

OHLCV 필수 컬럼:
  date, open, high, low, close, volume
선택 컬럼:
  asset_name 또는 ticker, adj_close

선택 입력
---------
기존 v6.4 예측 파일을 넣으면 기존 defensive_down_score와 비교합니다.
  --reference-predictions "v6_4_signal_predictions.csv"

출력
----
output_dir/
├─ dual_downside_summary.json
├─ dual_downside_config.json
├─ oos_dual_downside_predictions.csv
├─ fold_head_metrics.csv
├─ head_metric_summary.csv
├─ decile_candidate_score.csv
├─ decile_confirm_score.csv
├─ dual_policy_summary.csv
├─ threshold_sweep.csv
├─ asset_policy_summary.csv
├─ annual_policy_summary.csv
├─ fold_policy_summary.csv
├─ label_distribution.csv
├─ feature_importance_candidate.csv
├─ feature_importance_confirm.csv
├─ feature_cols.csv
└─ reference_single_head_comparison.csv

실행 예시 CMD
-------------
python downside_dual_head_experiment_v7_1.py ^
  --ohlcv-inputs "QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv" ^
  --asset-names "QQQ,SPY,SOXX,XLK" ^
  --reference-predictions "v6_4_signal_predictions.csv" ^
  --output-dir "downside_dual_head_v7_1_output"
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

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore", category=FutureWarning)
try:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
except Exception:
    pass


# ============================================================
# Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    horizon: int = 10
    vol_window: int = 60

    # Candidate = intraday low touch.
    k_down_touch: float = 1.00

    # Confirm = close-based drawdown. Default lower than 1.0 to avoid too few positives.
    k_close_mdd: float = 0.75

    # Optional close-to-close loss label for diagnostics.
    k_close_loss: float = 0.75

    min_train_days: int = 756
    test_days: int = 126
    step_days: int = 126
    embargo_days: int = 10
    calibration_days: int = 252

    candidate_model: str = "extratrees"
    confirm_model: str = "extratrees"

    n_estimators: int = 500
    max_features: str = "sqrt"
    min_samples_leaf: int = 20
    random_state: int = 42

    candidate_thresholds: Tuple[float, ...] = (0.70, 0.80, 0.85, 0.90, 0.95)
    confirm_thresholds: Tuple[float, ...] = (0.70, 0.80, 0.85, 0.90, 0.95)

    primary_candidate_threshold: float = 0.80
    primary_confirm_threshold: float = 0.80


# ============================================================
# IO utilities
# ============================================================

def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def save_csv(path: str | Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_json(path: str | Path, data: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    return path


def parse_csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    rename_map = {
        "datetime": "date",
        "timestamp": "date",
        "time": "date",
        "ticker": "asset_name",
        "symbol": "asset_name",
        "adj close": "adj_close",
        "adj_close": "adj_close",
        "adjusted_close": "adj_close",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV missing columns: {missing}. columns={list(out.columns)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]

    return out


def load_ohlcv(args) -> pd.DataFrame:
    if args.ohlcv_all:
        df = pd.read_csv(args.ohlcv_all)
        df = standardize_ohlcv_columns(df)
        if "asset_name" not in df.columns:
            raise ValueError("--ohlcv-all requires asset_name/ticker/symbol column")
        df["asset_name"] = df["asset_name"].astype(str)
        return df.sort_values(["asset_name", "date"]).reset_index(drop=True)

    inputs = parse_csv_list(args.ohlcv_inputs)
    names = parse_csv_list(args.asset_names)

    if not inputs:
        raise ValueError("Provide --ohlcv-inputs or --ohlcv-all")
    if names and len(names) != len(inputs):
        raise ValueError("--asset-names length must match --ohlcv-inputs length")

    parts = []
    for i, path in enumerate(inputs):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"OHLCV file not found: {p}")
        df = pd.read_csv(p)
        df = standardize_ohlcv_columns(df)

        if names:
            asset = names[i]
        elif "asset_name" in df.columns:
            asset = str(df["asset_name"].iloc[0])
        else:
            asset = p.stem.replace("_ohlcv", "").upper()

        df["asset_name"] = asset
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["asset_name", "date"]).reset_index(drop=True)


def load_reference_predictions(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reference predictions file not found: {p}")
    ref = pd.read_csv(p)
    ref.columns = [str(c).strip() for c in ref.columns]
    if "date" not in ref.columns or "asset_name" not in ref.columns:
        raise ValueError("reference predictions must contain date and asset_name")
    ref["date"] = pd.to_datetime(ref["date"], errors="coerce")
    keep = ["asset_name", "date"]
    for c in [
        "defensive_down_score_percentile",
        "balanced_down_score_percentile",
        "return_score_percentile",
        "v6_4_signal",
        "v7_signal",
    ]:
        if c in ref.columns:
            keep.append(c)
    return ref[keep].copy()


# ============================================================
# Feature engineering and labels
# ============================================================

def forward_window_min(s: pd.Series, horizon: int) -> pd.Series:
    return pd.concat([s.shift(-i) for i in range(1, horizon + 1)], axis=1).min(axis=1)


def forward_window_max(s: pd.Series, horizon: int) -> pd.Series:
    return pd.concat([s.shift(-i) for i in range(1, horizon + 1)], axis=1).max(axis=1)


def build_features_one_asset(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()

    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    open_ = g["open"].astype(float)
    volume = g["volume"].astype(float).replace(0, np.nan)

    ret = close.pct_change()
    log_ret = np.log(close).diff()

    g["ret_1d"] = ret
    g["log_ret_1d"] = log_ret
    g["hl_range"] = high / low - 1.0
    g["oc_ret"] = close / open_ - 1.0
    g["gap_ret"] = open_ / close.shift(1) - 1.0
    g["upper_shadow"] = (high - np.maximum(open_, close)) / close
    g["lower_shadow"] = (np.minimum(open_, close) - low) / close
    g["body_size"] = (close - open_).abs() / close

    windows = [2, 3, 5, 10, 20, 40, 60, 120, 252]
    for w in windows:
        g[f"ret_{w}d"] = close / close.shift(w) - 1.0
        g[f"log_ret_sum_{w}d"] = log_ret.rolling(w).sum()
        g[f"vol_{w}d"] = ret.rolling(w).std()
        g[f"down_vol_{w}d"] = ret.clip(upper=0).rolling(w).std()
        g[f"up_vol_{w}d"] = ret.clip(lower=0).rolling(w).std()
        g[f"ret_z_{w}d"] = ret.rolling(w).mean() / (ret.rolling(w).std() + 1e-12)

    ma_windows = [5, 10, 20, 40, 60, 120, 200]
    for w in ma_windows:
        ma = close.rolling(w).mean()
        g[f"ma_gap_{w}d"] = close / ma - 1.0
        g[f"ma_slope_{w}d_5d"] = ma / ma.shift(5) - 1.0
        g[f"ma_slope_{w}d_20d"] = ma / ma.shift(20) - 1.0

    range_windows = [10, 20, 40, 60, 120, 252]
    for w in range_windows:
        roll_high = high.rolling(w).max()
        roll_low = low.rolling(w).min()
        denom = (roll_high - roll_low).replace(0, np.nan)
        g[f"dist_to_high_{w}d"] = close / roll_high - 1.0
        g[f"dist_to_low_{w}d"] = close / roll_low - 1.0
        g[f"range_position_{w}d"] = (close - roll_low) / denom
        g[f"breakout_pressure_{w}d"] = close / roll_high.shift(1) - 1.0
        g[f"breakdown_pressure_{w}d"] = close / roll_low.shift(1) - 1.0

    for short, long in [(5, 20), (10, 40), (20, 60), (20, 120), (60, 252)]:
        g[f"vol_ratio_{short}_{long}"] = g[f"vol_{short}d"] / (g[f"vol_{long}d"] + 1e-12)
        g[f"down_vol_ratio_{short}_{long}"] = g[f"down_vol_{short}d"] / (g[f"down_vol_{long}d"] + 1e-12)

    tr1 = high / low - 1.0
    tr2 = (high / close.shift(1) - 1.0).abs()
    tr3 = (low / close.shift(1) - 1.0).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for w in [5, 10, 20, 40, 60]:
        g[f"atr_pct_{w}d"] = tr.rolling(w).mean()
        g[f"range_vol_{w}d"] = g["hl_range"].rolling(w).std()

    g["volume_log"] = np.log1p(volume)
    for w in [5, 10, 20, 40, 60, 120]:
        vol_ma = volume.rolling(w).mean()
        vol_std = volume.rolling(w).std()
        g[f"volume_ratio_{w}d"] = volume / (vol_ma + 1e-12)
        g[f"volume_chg_{w}d"] = volume / volume.shift(w) - 1.0
        g[f"volume_z_{w}d"] = (volume - vol_ma) / (vol_std + 1e-12)
        g[f"price_volume_corr_{w}d"] = ret.rolling(w).corr(volume.pct_change())

    g["trend_score_20_60"] = (g["ma_gap_20d"] > 0).astype(float) + (g["ma_gap_60d"] > 0).astype(float)
    g["vol_adjusted_momentum_20"] = g["ret_20d"] / (g["vol_20d"] * math.sqrt(20) + 1e-12)

    # Defragment after many feature insertions.
    return g.copy()


def add_labels_one_asset(g: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    g = g.sort_values("date").copy()

    close = g["close"].astype(float)
    low = g["low"].astype(float)
    ret = close.pct_change()

    current_horizon_vol = ret.rolling(cfg.vol_window).std().shift(1) * math.sqrt(cfg.horizon)
    future_min_low = forward_window_min(low, cfg.horizon)
    future_min_close = forward_window_min(close, cfg.horizon)
    future_close = close.shift(-cfg.horizon)

    future_close_return = future_close / close - 1.0
    future_intraday_mdd = future_min_low / close - 1.0
    future_close_mdd = future_min_close / close - 1.0

    lower_barrier = close * (1.0 - cfg.k_down_touch * current_horizon_vol)

    g["current_horizon_vol"] = current_horizon_vol
    g["future_min_low_h"] = future_min_low
    g["future_min_close_h"] = future_min_close
    g["future_close_h"] = future_close
    g["future_close_return_h"] = future_close_return
    g["future_intraday_mdd_h"] = future_intraday_mdd
    g["future_close_mdd_h"] = future_close_mdd

    # Stage 1: broad candidate based on intraday low touch.
    g["y_down_touch"] = (future_min_low <= lower_barrier).astype(float)

    # Stage 2: confirm based on close-level drawdown, not wick-only intraday touch.
    g["y_close_mdd_confirm"] = (future_close_mdd <= -cfg.k_close_mdd * current_horizon_vol).astype(float)

    # Diagnostic label: close-to-close loss at horizon.
    g["y_close_loss_h"] = (future_close_return <= -cfg.k_close_loss * current_horizon_vol).astype(float)

    # Data validity.
    invalid = (
        current_horizon_vol.isna()
        | future_min_low.isna()
        | future_min_close.isna()
        | future_close.isna()
    )
    for col in ["y_down_touch", "y_close_mdd_confirm", "y_close_loss_h"]:
        g.loc[invalid, col] = np.nan

    return g


def build_dataset(raw: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    parts = []
    for asset, g in raw.groupby("asset_name", sort=False):
        fg = build_features_one_asset(g)
        lg = add_labels_one_asset(fg, cfg)
        parts.append(lg)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["date", "asset_name"]).reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude_prefixes = ("y_", "future_")
    exclude = {
        "date", "asset_name",
        "open", "high", "low", "close", "adj_close", "volume",
        "current_horizon_vol",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith(exclude_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# ============================================================
# Splits, models, metrics
# ============================================================

def make_walk_forward_splits(df: pd.DataFrame, cfg: ExperimentConfig) -> List[Dict]:
    dates = np.array(sorted(df["date"].dropna().unique()))
    splits = []

    start = cfg.min_train_days
    fold_id = 0
    while start + cfg.test_days <= len(dates):
        test_start_idx = start
        test_end_idx = start + cfg.test_days

        train_end_idx = max(0, test_start_idx - cfg.embargo_days)
        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        if len(train_dates) >= cfg.min_train_days and len(test_dates) > 0:
            splits.append({
                "fold_id": fold_id,
                "train_start": pd.Timestamp(train_dates[0]),
                "train_end": pd.Timestamp(train_dates[-1]),
                "test_start": pd.Timestamp(test_dates[0]),
                "test_end": pd.Timestamp(test_dates[-1]),
                "train_dates": set(pd.to_datetime(train_dates)),
                "test_dates": set(pd.to_datetime(test_dates)),
            })
            fold_id += 1

        start += cfg.step_days

    return splits


def make_model(kind: str, cfg: ExperimentConfig):
    kind = kind.lower()
    if kind == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=cfg.n_estimators,
            max_features=cfg.max_features,
            min_samples_leaf=cfg.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=cfg.random_state,
        )
    elif kind == "randomforest":
        clf = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_features=cfg.max_features,
            min_samples_leaf=cfg.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=cfg.random_state,
        )
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=cfg.random_state,
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])


def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        cls = getattr(model.named_steps["model"], "classes_", np.array([0]))
        if len(cls) == 1 and cls[0] == 1:
            return np.ones(len(X))
        return np.zeros(len(X))
    return proba[:, 1]


def percentile_from_reference(ref_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref_scores, dtype=float)
    ref = ref[~np.isnan(ref)]
    if len(ref) == 0:
        return np.full(len(scores), np.nan)
    ref = np.sort(ref)
    return np.searchsorted(ref, scores, side="right") / len(ref)


def ece_score(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return np.nan

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    mask = ~np.isnan(y) & ~np.isnan(p)
    y = y[mask]
    p = p[mask]
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    mask = ~np.isnan(y) & ~np.isnan(p)
    y = y[mask]
    p = p[mask]
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def safe_brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    mask = ~np.isnan(y) & ~np.isnan(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return np.nan
    return float(brier_score_loss(y, np.clip(p, 0, 1)))


def head_metrics(y: np.ndarray, p: np.ndarray, name: str) -> Dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(p)
    yv = y[mask]
    pv = p[mask]
    base = float(np.mean(yv)) if len(yv) else np.nan

    brier = safe_brier(yv, pv)
    ref_brier = base * (1 - base) if not np.isnan(base) else np.nan
    brier_skill = 1 - brier / ref_brier if ref_brier and ref_brier > 0 else np.nan

    return {
        "head": name,
        "n": int(len(yv)),
        "positive_rate": base,
        "roc_auc": safe_auc(yv, pv),
        "pr_auc": safe_ap(yv, pv),
        "pr_ratio": safe_ap(yv, pv) / base if base and base > 0 else np.nan,
        "brier": brier,
        "brier_skill": brier_skill,
        "ece": ece_score(yv, pv),
        "mean_score": float(np.mean(pv)) if len(pv) else np.nan,
    }


# ============================================================
# Training and prediction
# ============================================================

def fit_predict_fold(
    df: pd.DataFrame,
    feature_cols: List[str],
    split: Dict,
    cfg: ExperimentConfig,
) -> Tuple[pd.DataFrame, List[Dict], List[pd.DataFrame]]:
    train_mask = df["date"].isin(split["train_dates"])
    test_mask = df["date"].isin(split["test_dates"])

    fold_train = df.loc[train_mask].copy()
    fold_test = df.loc[test_mask].copy()

    # Drop rows with unavailable labels in train/test.
    fold_train = fold_train.dropna(subset=["y_down_touch", "y_close_mdd_confirm"])
    fold_test = fold_test.dropna(subset=["y_down_touch", "y_close_mdd_confirm"])

    if fold_train.empty or fold_test.empty:
        return pd.DataFrame(), [], []

    # Calibration window = last N unique dates in train.
    train_dates = np.array(sorted(fold_train["date"].unique()))
    if len(train_dates) > cfg.calibration_days + 100:
        cal_dates = set(pd.to_datetime(train_dates[-cfg.calibration_days:]))
        fit_dates = set(pd.to_datetime(train_dates[:-cfg.calibration_days]))
        fit_data = fold_train[fold_train["date"].isin(fit_dates)].copy()
        cal_data = fold_train[fold_train["date"].isin(cal_dates)].copy()
    else:
        fit_data = fold_train.copy()
        cal_data = fold_train.copy()

    candidate_model = make_model(cfg.candidate_model, cfg)
    confirm_model = make_model(cfg.confirm_model, cfg)

    X_fit = fit_data[feature_cols]
    X_cal = cal_data[feature_cols]
    X_test = fold_test[feature_cols]

    y_candidate_fit = fit_data["y_down_touch"].astype(int)
    y_confirm_fit = fit_data["y_close_mdd_confirm"].astype(int)

    # If label is one-class in a fold, skip that fold.
    if y_candidate_fit.nunique() < 2 or y_confirm_fit.nunique() < 2:
        return pd.DataFrame(), [], []

    candidate_model.fit(X_fit, y_candidate_fit)
    confirm_model.fit(X_fit, y_confirm_fit)

    cand_cal_raw = predict_proba_safe(candidate_model, X_cal)
    conf_cal_raw = predict_proba_safe(confirm_model, X_cal)
    cand_test_raw = predict_proba_safe(candidate_model, X_test)
    conf_test_raw = predict_proba_safe(confirm_model, X_test)

    cand_test_pct = percentile_from_reference(cand_cal_raw, cand_test_raw)
    conf_test_pct = percentile_from_reference(conf_cal_raw, conf_test_raw)

    pred = fold_test[[
        "asset_name", "date",
        "y_down_touch", "y_close_mdd_confirm", "y_close_loss_h",
        "future_close_return_h", "future_intraday_mdd_h", "future_close_mdd_h",
        "current_horizon_vol",
    ]].copy()

    pred["fold_id"] = split["fold_id"]
    pred["candidate_raw_proba"] = cand_test_raw
    pred["confirm_raw_proba"] = conf_test_raw
    pred["candidate_score_percentile"] = cand_test_pct
    pred["confirm_score_percentile"] = conf_test_pct

    fold_metrics = []
    # Each tuple is: (label_column, score_column, metric_name).
    # The previous version mixed 4-field and 3-field tuples, causing:
    # ValueError: not enough values to unpack (expected 4, got 3)
    for label_col, score_col, head in [
        ("y_down_touch", "candidate_raw_proba", "candidate_down_touch_raw"),
        ("y_down_touch", "candidate_score_percentile", "candidate_down_touch_percentile"),
        ("y_close_mdd_confirm", "confirm_raw_proba", "confirm_mdd_raw"),
        ("y_close_mdd_confirm", "confirm_score_percentile", "confirm_mdd_percentile"),
    ]:
        m = head_metrics(pred[label_col].to_numpy(), pred[score_col].to_numpy(), head)
        m["fold_id"] = split["fold_id"]
        m["test_start"] = split["test_start"]
        m["test_end"] = split["test_end"]
        fold_metrics.append(m)

    # Feature importance.
    imps = []
    for model_name, model in [
        ("candidate", candidate_model),
        ("confirm", confirm_model),
    ]:
        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            imp = pd.DataFrame({
                "feature": feature_cols,
                "importance": estimator.feature_importances_,
                "fold_id": split["fold_id"],
                "model": model_name,
            })
            imps.append(imp)

    return pred, fold_metrics, imps


def run_walk_forward(df: pd.DataFrame, feature_cols: List[str], cfg: ExperimentConfig):
    splits = make_walk_forward_splits(df, cfg)
    pred_parts = []
    metric_rows = []
    imp_parts = []

    for split in splits:
        pred, metrics, imps = fit_predict_fold(df, feature_cols, split, cfg)
        if not pred.empty:
            pred_parts.append(pred)
        metric_rows.extend(metrics)
        imp_parts.extend(imps)

    preds = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    imps = pd.concat(imp_parts, ignore_index=True) if imp_parts else pd.DataFrame()

    return preds, metrics, imps, splits


# ============================================================
# Analysis
# ============================================================

def add_dual_policy(preds: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    out = preds.copy()
    c = out["candidate_score_percentile"]
    m = out["confirm_score_percentile"]

    out["dual_downside_signal"] = np.select(
        [
            (c >= cfg.primary_candidate_threshold) & (m >= cfg.primary_confirm_threshold),
            (c >= 0.90) & (m < cfg.primary_confirm_threshold),
            (c >= cfg.primary_candidate_threshold),
        ],
        [
            "CONFIRMED_DOWNSIDE_RISK",
            "DOWNSIDE_CANDIDATE_HIGH_BUT_UNCONFIRMED",
            "DOWNSIDE_WATCH",
        ],
        default="NO_DOWNSIDE_EDGE",
    )
    return out


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for head, g in metrics.groupby("head"):
        rows.append({
            "head": head,
            "fold_count": int(g["fold_id"].nunique()),
            "mean_positive_rate": float(g["positive_rate"].mean()),
            "mean_roc_auc": float(g["roc_auc"].mean()),
            "median_roc_auc": float(g["roc_auc"].median()),
            "mean_pr_auc": float(g["pr_auc"].mean()),
            "median_pr_auc": float(g["pr_auc"].median()),
            "mean_pr_ratio": float(g["pr_ratio"].mean()),
            "median_pr_ratio": float(g["pr_ratio"].median()),
            "mean_brier": float(g["brier"].mean()),
            "mean_brier_skill": float(g["brier_skill"].mean()),
            "median_brier_skill": float(g["brier_skill"].median()),
            "mean_ece": float(g["ece"].mean()),
            "positive_pr_ratio_fold_rate": float((g["pr_ratio"] > 1.0).mean()),
            "positive_brier_skill_fold_rate": float((g["brier_skill"] > 0).mean()),
        })
    return pd.DataFrame(rows)


def decile_summary(preds: pd.DataFrame, score_col: str, label_col: str, name: str) -> pd.DataFrame:
    d = preds[[score_col, label_col, "asset_name", "date"]].dropna().copy()
    if d.empty:
        return pd.DataFrame()

    # duplicates='drop' handles tied percentiles.
    d["decile"] = pd.qcut(d[score_col].rank(method="first"), 10, labels=False, duplicates="drop") + 1

    rows = []
    global_rate = d[label_col].mean()
    for dec, g in d.groupby("decile"):
        rows.append({
            "score": name,
            "decile": int(dec),
            "count": int(len(g)),
            "label_rate": float(g[label_col].mean()),
            "global_rate": float(global_rate),
            "lift": float(g[label_col].mean() / global_rate) if global_rate > 0 else np.nan,
            "score_min": float(g[score_col].min()),
            "score_max": float(g[score_col].max()),
            "score_mean": float(g[score_col].mean()),
        })
    return pd.DataFrame(rows)


def policy_summary(preds: pd.DataFrame, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = []

    rows = []
    group_iter = preds.groupby(group_cols, dropna=False) if group_cols else [((), preds)]

    for key, base in group_iter:
        if not isinstance(key, tuple):
            key = (key,)

        global_confirm_rate = base["y_close_mdd_confirm"].mean()
        global_candidate_rate = base["y_down_touch"].mean()

        for sig, g in base.groupby("dual_downside_signal"):
            row = dict(zip(group_cols, key))
            row.update({
                "dual_downside_signal": sig,
                "rows": int(len(g)),
                "signal_rate": float(len(g) / len(base)) if len(base) else np.nan,
                "candidate_down_touch_rate": float(g["y_down_touch"].mean()),
                "confirm_mdd_rate": float(g["y_close_mdd_confirm"].mean()),
                "close_loss_rate": float(g["y_close_loss_h"].mean()),
                "candidate_lift": float(g["y_down_touch"].mean() / global_candidate_rate) if global_candidate_rate > 0 else np.nan,
                "confirm_lift": float(g["y_close_mdd_confirm"].mean() / global_confirm_rate) if global_confirm_rate > 0 else np.nan,
                "future_close_return_mean": float(g["future_close_return_h"].mean()),
                "future_close_return_median": float(g["future_close_return_h"].median()),
                "future_close_mdd_mean": float(g["future_close_mdd_h"].mean()),
                "future_close_mdd_median": float(g["future_close_mdd_h"].median()),
                "future_intraday_mdd_mean": float(g["future_intraday_mdd_h"].mean()),
                "future_intraday_mdd_median": float(g["future_intraday_mdd_h"].median()),
            })
            rows.append(row)

    return pd.DataFrame(rows)


def threshold_sweep(preds: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    base_confirm = preds["y_close_mdd_confirm"].mean()
    base_candidate = preds["y_down_touch"].mean()

    for ct in cfg.candidate_thresholds:
        for mt in cfg.confirm_thresholds:
            sig = (preds["candidate_score_percentile"] >= ct) & (preds["confirm_score_percentile"] >= mt)
            g = preds.loc[sig]
            rows.append({
                "candidate_threshold": ct,
                "confirm_threshold": mt,
                "rows": int(len(g)),
                "signal_rate": float(sig.mean()),
                "candidate_down_touch_rate": float(g["y_down_touch"].mean()) if len(g) else np.nan,
                "confirm_mdd_rate": float(g["y_close_mdd_confirm"].mean()) if len(g) else np.nan,
                "close_loss_rate": float(g["y_close_loss_h"].mean()) if len(g) else np.nan,
                "candidate_lift": float(g["y_down_touch"].mean() / base_candidate) if len(g) and base_candidate > 0 else np.nan,
                "confirm_lift": float(g["y_close_mdd_confirm"].mean() / base_confirm) if len(g) and base_confirm > 0 else np.nan,
                "future_close_return_mean": float(g["future_close_return_h"].mean()) if len(g) else np.nan,
                "future_close_mdd_mean": float(g["future_close_mdd_h"].mean()) if len(g) else np.nan,
                "future_intraday_mdd_mean": float(g["future_intraday_mdd_h"].mean()) if len(g) else np.nan,
                "coverage_of_confirm_events": float((sig & (preds["y_close_mdd_confirm"] == 1)).sum() / (preds["y_close_mdd_confirm"] == 1).sum())
                    if (preds["y_close_mdd_confirm"] == 1).sum() > 0 else np.nan,
                "false_alarm_rate": float((g["y_close_mdd_confirm"] == 0).mean()) if len(g) else np.nan,
            })
    return pd.DataFrame(rows)


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, g in df.groupby("asset_name"):
        row = {
            "asset_name": asset,
            "rows": int(len(g)),
            "date_start": str(g["date"].min().date()),
            "date_end": str(g["date"].max().date()),
            "y_down_touch_rate": float(g["y_down_touch"].mean()),
            "y_close_mdd_confirm_rate": float(g["y_close_mdd_confirm"].mean()),
            "y_close_loss_h_rate": float(g["y_close_loss_h"].mean()),
        }
        rows.append(row)

    all_row = {
        "asset_name": "ALL",
        "rows": int(len(df)),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "y_down_touch_rate": float(df["y_down_touch"].mean()),
        "y_close_mdd_confirm_rate": float(df["y_close_mdd_confirm"].mean()),
        "y_close_loss_h_rate": float(df["y_close_loss_h"].mean()),
    }
    rows.append(all_row)
    return pd.DataFrame(rows)


def feature_importance_summary(imps: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if imps.empty:
        return pd.DataFrame()
    g = imps[imps["model"] == model_name].copy()
    if g.empty:
        return pd.DataFrame()
    out = g.groupby("feature").agg(
        mean_importance=("importance", "mean"),
        median_importance=("importance", "median"),
        std_importance=("importance", "std"),
        fold_count=("fold_id", "nunique"),
    ).reset_index()
    return out.sort_values("mean_importance", ascending=False).reset_index(drop=True)


def reference_comparison(preds: pd.DataFrame, ref: Optional[pd.DataFrame]) -> pd.DataFrame:
    if ref is None or ref.empty:
        return pd.DataFrame()

    d = preds.merge(ref, on=["asset_name", "date"], how="left", suffixes=("", "_ref"))
    if "defensive_down_score_percentile" not in d.columns:
        return pd.DataFrame()

    rows = []
    base_confirm = d["y_close_mdd_confirm"].mean()
    for threshold in [0.70, 0.80, 0.90, 0.95]:
        old_sig = d["defensive_down_score_percentile"] >= threshold
        new_sig = (
            (d["candidate_score_percentile"] >= threshold)
            & (d["confirm_score_percentile"] >= threshold)
        )

        for name, sig in [
            ("old_defensive_single_head", old_sig),
            ("new_dual_downside", new_sig),
        ]:
            g = d.loc[sig]
            rows.append({
                "system": name,
                "threshold": threshold,
                "rows": int(len(g)),
                "signal_rate": float(sig.mean()),
                "confirm_mdd_rate": float(g["y_close_mdd_confirm"].mean()) if len(g) else np.nan,
                "confirm_lift": float(g["y_close_mdd_confirm"].mean() / base_confirm) if len(g) and base_confirm > 0 else np.nan,
                "future_close_return_mean": float(g["future_close_return_h"].mean()) if len(g) else np.nan,
                "future_close_mdd_mean": float(g["future_close_mdd_h"].mean()) if len(g) else np.nan,
                "coverage_of_confirm_events": float((sig & (d["y_close_mdd_confirm"] == 1)).sum() / (d["y_close_mdd_confirm"] == 1).sum())
                    if (d["y_close_mdd_confirm"] == 1).sum() > 0 else np.nan,
                "false_alarm_rate": float((g["y_close_mdd_confirm"] == 0).mean()) if len(g) else np.nan,
            })
    return pd.DataFrame(rows)


# ============================================================
# Main runner
# ============================================================

def run(args) -> Dict[str, Path]:
    cfg = ExperimentConfig(
        horizon=args.horizon,
        vol_window=args.vol_window,
        k_down_touch=args.k_down_touch,
        k_close_mdd=args.k_close_mdd,
        k_close_loss=args.k_close_loss,
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        embargo_days=args.embargo_days,
        calibration_days=args.calibration_days,
        candidate_model=args.candidate_model,
        confirm_model=args.confirm_model,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        primary_candidate_threshold=args.primary_candidate_threshold,
        primary_confirm_threshold=args.primary_confirm_threshold,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_ohlcv(args)
    df = build_dataset(raw, cfg)
    feature_cols = get_feature_cols(df)

    # Remove rows with missing target labels for label distribution only after saving enough info.
    valid_df = df.dropna(subset=["y_down_touch", "y_close_mdd_confirm"]).copy()

    preds, fold_metrics, imps, splits = run_walk_forward(valid_df, feature_cols, cfg)
    if preds.empty:
        raise RuntimeError("No OOS predictions generated. Check split settings and data length.")

    preds = add_dual_policy(preds, cfg)

    ref = load_reference_predictions(args.reference_predictions)
    ref_comp = reference_comparison(preds, ref)

    metric_summary = summarize_metrics(fold_metrics)
    dec_cand = decile_summary(preds, "candidate_score_percentile", "y_down_touch", "candidate_down_touch")
    dec_conf = decile_summary(preds, "confirm_score_percentile", "y_close_mdd_confirm", "confirm_close_mdd")
    pol = policy_summary(preds)
    pol_asset = policy_summary(preds, ["asset_name"])
    preds["year"] = preds["date"].dt.year
    pol_annual = policy_summary(preds, ["asset_name", "year"])
    pol_fold = policy_summary(preds, ["asset_name", "fold_id"])
    sweep = threshold_sweep(preds, cfg)

    lab_dist = label_distribution(valid_df)
    cand_imp = feature_importance_summary(imps, "candidate")
    conf_imp = feature_importance_summary(imps, "confirm")
    feature_df = pd.DataFrame({"feature": feature_cols})

    # Primary decision.
    primary_sig = (
        (preds["candidate_score_percentile"] >= cfg.primary_candidate_threshold)
        & (preds["confirm_score_percentile"] >= cfg.primary_confirm_threshold)
    )
    primary_rows = preds.loc[primary_sig]
    base_confirm = preds["y_close_mdd_confirm"].mean()

    decision = {
        "primary_candidate_threshold": cfg.primary_candidate_threshold,
        "primary_confirm_threshold": cfg.primary_confirm_threshold,
        "rows": int(len(primary_rows)),
        "signal_rate": float(primary_sig.mean()),
        "confirm_mdd_rate": float(primary_rows["y_close_mdd_confirm"].mean()) if len(primary_rows) else np.nan,
        "confirm_lift": float(primary_rows["y_close_mdd_confirm"].mean() / base_confirm) if len(primary_rows) and base_confirm > 0 else np.nan,
        "coverage_of_confirm_events": float((primary_sig & (preds["y_close_mdd_confirm"] == 1)).sum() / (preds["y_close_mdd_confirm"] == 1).sum())
            if (preds["y_close_mdd_confirm"] == 1).sum() > 0 else np.nan,
        "future_close_return_mean": float(primary_rows["future_close_return_h"].mean()) if len(primary_rows) else np.nan,
        "future_close_mdd_mean": float(primary_rows["future_close_mdd_h"].mean()) if len(primary_rows) else np.nan,
        "future_intraday_mdd_mean": float(primary_rows["future_intraday_mdd_h"].mean()) if len(primary_rows) else np.nan,
    }

    # Conservative acceptance rule.
    risk_flags = []
    if decision["rows"] < 50:
        risk_flags.append("small_signal_count")
    if not np.isnan(decision["confirm_lift"]) and decision["confirm_lift"] < 1.25:
        risk_flags.append("weak_confirm_lift")
    if not np.isnan(decision["coverage_of_confirm_events"]) and decision["coverage_of_confirm_events"] < 0.15:
        risk_flags.append("low_confirm_event_coverage")
    if not np.isnan(decision["future_close_return_mean"]) and decision["future_close_return_mean"] >= 0:
        risk_flags.append("not_negative_future_close_return_mean")

    if not risk_flags:
        decision["decision"] = "PASS_AS_CONFIRMED_DOWNSIDE_CANDIDATE"
    elif "small_signal_count" in risk_flags and len(risk_flags) == 1:
        decision["decision"] = "PROMISING_BUT_SMALL_SAMPLE"
    else:
        decision["decision"] = "WATCH_OR_REDESIGN_NEEDED"
    decision["risk_flags"] = risk_flags

    config_json = {
        "experiment": "downside_dual_head_experiment_v7_1",
        "config": asdict(cfg),
        "feature_count": len(feature_cols),
        "features_exclude_future_and_labels": True,
        "label_definitions": {
            "y_down_touch": "future_min_low[t+1:t+H] <= close_t * (1 - k_down_touch * current_horizon_vol_t)",
            "y_close_mdd_confirm": "min(future_close[t+1:t+H]) / close_t - 1 <= -k_close_mdd * current_horizon_vol_t",
            "y_close_loss_h": "future_close[t+H] / close_t - 1 <= -k_close_loss * current_horizon_vol_t",
        },
        "score_interpretation": "score_percentile is calibration-window rank, not literal probability",
    }

    summary_json = {
        "experiment": "downside_dual_head_experiment_v7_1",
        "asset_count": int(valid_df["asset_name"].nunique()),
        "rows_valid": int(len(valid_df)),
        "oos_rows": int(len(preds)),
        "date_start": str(valid_df["date"].min().date()),
        "date_end": str(valid_df["date"].max().date()),
        "fold_count": int(preds["fold_id"].nunique()),
        "feature_count": int(len(feature_cols)),
        "label_distribution_all": lab_dist[lab_dist["asset_name"] == "ALL"].to_dict(orient="records")[0],
        "head_metric_summary": metric_summary.to_dict(orient="records"),
        "primary_decision": decision,
        "note": "This is a dual downside diagnostic experiment, not a final allocation model.",
    }

    outputs = {
        "summary": save_json(out_dir / "dual_downside_summary.json", summary_json),
        "config": save_json(out_dir / "dual_downside_config.json", config_json),
        "oos_predictions": save_csv(out_dir / "oos_dual_downside_predictions.csv", preds),
        "fold_head_metrics": save_csv(out_dir / "fold_head_metrics.csv", fold_metrics),
        "head_metric_summary": save_csv(out_dir / "head_metric_summary.csv", metric_summary),
        "decile_candidate_score": save_csv(out_dir / "decile_candidate_score.csv", dec_cand),
        "decile_confirm_score": save_csv(out_dir / "decile_confirm_score.csv", dec_conf),
        "dual_policy_summary": save_csv(out_dir / "dual_policy_summary.csv", pol),
        "asset_policy_summary": save_csv(out_dir / "asset_policy_summary.csv", pol_asset),
        "annual_policy_summary": save_csv(out_dir / "annual_policy_summary.csv", pol_annual),
        "fold_policy_summary": save_csv(out_dir / "fold_policy_summary.csv", pol_fold),
        "threshold_sweep": save_csv(out_dir / "threshold_sweep.csv", sweep),
        "label_distribution": save_csv(out_dir / "label_distribution.csv", lab_dist),
        "feature_importance_candidate": save_csv(out_dir / "feature_importance_candidate.csv", cand_imp),
        "feature_importance_confirm": save_csv(out_dir / "feature_importance_confirm.csv", conf_imp),
        "feature_cols": save_csv(out_dir / "feature_cols.csv", feature_df),
        "reference_single_head_comparison": save_csv(out_dir / "reference_single_head_comparison.csv", ref_comp),
    }

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()

    # Data input
    parser.add_argument("--ohlcv-inputs", default=None, help="Comma-separated OHLCV files")
    parser.add_argument("--asset-names", default=None, help="Comma-separated asset names matching --ohlcv-inputs")
    parser.add_argument("--ohlcv-all", default=None, help="Unified OHLCV file with asset_name/ticker column")
    parser.add_argument("--reference-predictions", default=None, help="Optional v6_4_signal_predictions.csv")

    # Output
    parser.add_argument("--output-dir", default="downside_dual_head_v7_1_output")

    # Label settings
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--k-down-touch", type=float, default=1.00)
    parser.add_argument("--k-close-mdd", type=float, default=0.75)
    parser.add_argument("--k-close-loss", type=float, default=0.75)

    # Split settings
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=126)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--calibration-days", type=int, default=252)

    # Model settings
    parser.add_argument("--candidate-model", default="extratrees", choices=["extratrees", "randomforest", "hgb"])
    parser.add_argument("--confirm-model", default="extratrees", choices=["extratrees", "randomforest", "hgb"])
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)

    # Policy thresholds
    parser.add_argument("--primary-candidate-threshold", type=float, default=0.80)
    parser.add_argument("--primary-confirm-threshold", type=float, default=0.80)

    args = parser.parse_args()

    outputs = run(args)
    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))

    print("[OK] Downside Dual-Head Experiment v7.1 completed.")
    print(json.dumps({
        "asset_count": summary["asset_count"],
        "rows_valid": summary["rows_valid"],
        "oos_rows": summary["oos_rows"],
        "date_start": summary["date_start"],
        "date_end": summary["date_end"],
        "fold_count": summary["fold_count"],
        "feature_count": summary["feature_count"],
        "label_distribution_all": summary["label_distribution_all"],
        "primary_decision": summary["primary_decision"],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
