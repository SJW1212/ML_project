# -*- coding: utf-8 -*-
"""
downside_confirm_label_repair_v7_2.py

7.2 Downside Confirm Label Repair
=================================

목적
----
7.1 Dual Downside 실험에서 2중 하방 탐지는 기존 single defensive head보다 개선됐지만,
primary policy가 평균 미래수익률을 음수로 분리하지 못했습니다.

이 스크립트는 confirm 라벨을 여러 후보로 재설계하여 비교합니다.

고정 구조
---------
Candidate Head:
  y_down_touch_h10_k1.0

Confirm Head 후보:
  1. close_mdd_075
  2. close_mdd_100
  3. close_mdd_125
  4. close_mdd_075_and_close_neg
  5. close_mdd_100_and_close_neg
  6. close_loss_075
  7. close_loss_100

최종 비교
---------
candidate_score_percentile >= threshold_candidate
confirm_score_percentile >= threshold_confirm

출력
----
output_dir/
├─ confirm_label_repair_summary.json
├─ confirm_label_repair_config.json
├─ confirm_variant_head_metrics.csv
├─ confirm_variant_policy_summary.csv
├─ confirm_variant_threshold_sweep.csv
├─ confirm_variant_decile_summary.csv
├─ confirm_variant_asset_summary.csv
├─ confirm_variant_annual_summary.csv
├─ confirm_variant_fold_summary.csv
├─ confirm_variant_decision_table.csv
├─ confirm_variant_oos_predictions.csv
├─ candidate_head_metrics.csv
├─ label_distribution.csv
└─ feature_cols.csv

실행 예시
---------
python downside_confirm_label_repair_v7_2.py ^
  --ohlcv-inputs "QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv" ^
  --asset-names "QQQ,SPY,SOXX,XLK" ^
  --output-dir "downside_confirm_label_repair_v7_2_output"
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
except Exception:
    pass


@dataclass
class Config:
    horizon: int = 10
    vol_window: int = 60
    k_down_touch: float = 1.00

    min_train_days: int = 756
    test_days: int = 126
    step_days: int = 126
    embargo_days: int = 10
    calibration_days: int = 252

    model_kind: str = "extratrees"
    n_estimators: int = 250
    min_samples_leaf: int = 20
    random_state: int = 42

    candidate_thresholds: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)
    confirm_thresholds: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)

    primary_candidate_threshold: float = 0.90
    primary_confirm_threshold: float = 0.90


CONFIRM_VARIANTS = {
    "close_mdd_075": {
        "kind": "close_mdd",
        "k": 0.75,
        "require_close_negative": False,
    },
    "close_mdd_100": {
        "kind": "close_mdd",
        "k": 1.00,
        "require_close_negative": False,
    },
    "close_mdd_125": {
        "kind": "close_mdd",
        "k": 1.25,
        "require_close_negative": False,
    },
    "close_mdd_075_and_close_neg": {
        "kind": "close_mdd",
        "k": 0.75,
        "require_close_negative": True,
    },
    "close_mdd_100_and_close_neg": {
        "kind": "close_mdd",
        "k": 1.00,
        "require_close_negative": True,
    },
    "close_loss_075": {
        "kind": "close_loss",
        "k": 0.75,
        "require_close_negative": True,
    },
    "close_loss_100": {
        "kind": "close_loss",
        "k": 1.00,
        "require_close_negative": True,
    },
}


# ============================================================
# IO
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


def save_json(path: str | Path, data: Dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    return p


def save_csv(path: str | Path, df: pd.DataFrame) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def parse_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    rename = {
        "datetime": "date",
        "timestamp": "date",
        "ticker": "asset_name",
        "symbol": "asset_name",
        "adj close": "adj_close",
        "adjusted_close": "adj_close",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}; columns={list(out.columns)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()

    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]
    return out


def load_ohlcv(args) -> pd.DataFrame:
    if args.ohlcv_all:
        df = standardize_ohlcv(pd.read_csv(args.ohlcv_all))
        if "asset_name" not in df.columns:
            raise ValueError("--ohlcv-all requires asset_name/ticker/symbol column")
        df["asset_name"] = df["asset_name"].astype(str)
        return df.sort_values(["asset_name", "date"]).reset_index(drop=True)

    inputs = parse_list(args.ohlcv_inputs)
    names = parse_list(args.asset_names)
    if not inputs:
        raise ValueError("Provide --ohlcv-inputs or --ohlcv-all")
    if names and len(names) != len(inputs):
        raise ValueError("--asset-names length must match --ohlcv-inputs")

    parts = []
    for i, path in enumerate(inputs):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"OHLCV file not found: {p}")
        df = standardize_ohlcv(pd.read_csv(p))
        asset = names[i] if names else (df["asset_name"].iloc[0] if "asset_name" in df.columns else p.stem.replace("_ohlcv", "").upper())
        df["asset_name"] = str(asset)
        parts.append(df)

    return pd.concat(parts, ignore_index=True).sort_values(["asset_name", "date"]).reset_index(drop=True)


# ============================================================
# Feature and label engineering
# ============================================================

def forward_min(s: pd.Series, h: int) -> pd.Series:
    return pd.concat([s.shift(-i) for i in range(1, h + 1)], axis=1).min(axis=1)


def build_features_one_asset(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()

    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    open_ = g["open"].astype(float)
    volume = g["volume"].astype(float).replace(0, np.nan)

    ret = close.pct_change()
    log_ret = np.log(close).diff()

    feats = {}
    feats["ret_1d"] = ret
    feats["log_ret_1d"] = log_ret
    feats["hl_range"] = high / low - 1.0
    feats["oc_ret"] = close / open_ - 1.0
    feats["gap_ret"] = open_ / close.shift(1) - 1.0
    feats["upper_shadow"] = (high - np.maximum(open_, close)) / close
    feats["lower_shadow"] = (np.minimum(open_, close) - low) / close
    feats["body_size"] = (close - open_).abs() / close

    windows = [2, 3, 5, 10, 20, 40, 60, 120, 252]
    for w in windows:
        vol = ret.rolling(w).std()
        feats[f"ret_{w}d"] = close / close.shift(w) - 1.0
        feats[f"log_ret_sum_{w}d"] = log_ret.rolling(w).sum()
        feats[f"vol_{w}d"] = vol
        feats[f"down_vol_{w}d"] = ret.clip(upper=0).rolling(w).std()
        feats[f"up_vol_{w}d"] = ret.clip(lower=0).rolling(w).std()
        feats[f"ret_z_{w}d"] = ret.rolling(w).mean() / (vol + 1e-12)

    ma_windows = [5, 10, 20, 40, 60, 120, 200]
    for w in ma_windows:
        ma = close.rolling(w).mean()
        feats[f"ma_gap_{w}d"] = close / ma - 1.0
        feats[f"ma_slope_{w}d_5d"] = ma / ma.shift(5) - 1.0
        feats[f"ma_slope_{w}d_20d"] = ma / ma.shift(20) - 1.0

    range_windows = [10, 20, 40, 60, 120, 252]
    for w in range_windows:
        rh = high.rolling(w).max()
        rl = low.rolling(w).min()
        denom = (rh - rl).replace(0, np.nan)
        feats[f"dist_to_high_{w}d"] = close / rh - 1.0
        feats[f"dist_to_low_{w}d"] = close / rl - 1.0
        feats[f"range_position_{w}d"] = (close - rl) / denom
        feats[f"breakout_pressure_{w}d"] = close / rh.shift(1) - 1.0
        feats[f"breakdown_pressure_{w}d"] = close / rl.shift(1) - 1.0

    for short, long in [(5, 20), (10, 40), (20, 60), (20, 120), (60, 252)]:
        feats[f"vol_ratio_{short}_{long}"] = feats[f"vol_{short}d"] / (feats[f"vol_{long}d"] + 1e-12)
        feats[f"down_vol_ratio_{short}_{long}"] = feats[f"down_vol_{short}d"] / (feats[f"down_vol_{long}d"] + 1e-12)

    tr1 = high / low - 1.0
    tr2 = (high / close.shift(1) - 1.0).abs()
    tr3 = (low / close.shift(1) - 1.0).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for w in [5, 10, 20, 40, 60]:
        feats[f"atr_pct_{w}d"] = tr.rolling(w).mean()
        feats[f"range_vol_{w}d"] = feats["hl_range"].rolling(w).std()

    feats["volume_log"] = np.log1p(volume)
    volume_pct = volume.pct_change()
    for w in [5, 10, 20, 40, 60, 120]:
        vma = volume.rolling(w).mean()
        vstd = volume.rolling(w).std()
        feats[f"volume_ratio_{w}d"] = volume / (vma + 1e-12)
        feats[f"volume_chg_{w}d"] = volume / volume.shift(w) - 1.0
        feats[f"volume_z_{w}d"] = (volume - vma) / (vstd + 1e-12)
        feats[f"price_volume_corr_{w}d"] = ret.rolling(w).corr(volume_pct)

    feats["trend_score_20_60"] = ((close / close.rolling(20).mean() - 1.0) > 0).astype(float) + ((close / close.rolling(60).mean() - 1.0) > 0).astype(float)
    feats["vol_adjusted_momentum_20"] = feats["ret_20d"] / (feats["vol_20d"] * math.sqrt(20) + 1e-12)

    # Downside-specific features for confirm repair.
    for w in [10, 20, 40, 60, 120]:
        rolling_peak = close.rolling(w).max()
        dd_from_peak = close / rolling_peak - 1.0
        feats[f"drawdown_from_peak_{w}d"] = dd_from_peak
        feats[f"drawdown_min_{w}d"] = dd_from_peak.rolling(w).min()
        feats[f"negative_return_count_{w}d"] = (ret < 0).rolling(w).sum()
        feats[f"large_down_day_count_{w}d"] = (ret < -ret.rolling(w).std()).rolling(w).sum()
        feats[f"close_below_ma_count_{w}d"] = (close < close.rolling(w).mean()).rolling(w).sum()

    feat_df = pd.DataFrame(feats, index=g.index)
    return pd.concat([g, feat_df], axis=1).copy()


def add_labels_one_asset(g: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    g = g.sort_values("date").copy()

    close = g["close"].astype(float)
    low = g["low"].astype(float)
    ret = close.pct_change()

    current_horizon_vol = ret.rolling(cfg.vol_window).std().shift(1) * math.sqrt(cfg.horizon)
    future_min_low = forward_min(low, cfg.horizon)
    future_min_close = forward_min(close, cfg.horizon)
    future_close = close.shift(-cfg.horizon)

    future_close_return = future_close / close - 1.0
    future_intraday_mdd = future_min_low / close - 1.0
    future_close_mdd = future_min_close / close - 1.0

    labels = pd.DataFrame(index=g.index)
    labels["current_horizon_vol"] = current_horizon_vol
    labels["future_min_low_h"] = future_min_low
    labels["future_min_close_h"] = future_min_close
    labels["future_close_h"] = future_close
    labels["future_close_return_h"] = future_close_return
    labels["future_intraday_mdd_h"] = future_intraday_mdd
    labels["future_close_mdd_h"] = future_close_mdd

    lower_barrier = close * (1.0 - cfg.k_down_touch * current_horizon_vol)
    labels["y_down_touch"] = (future_min_low <= lower_barrier).astype(float)

    for name, spec in CONFIRM_VARIANTS.items():
        if spec["kind"] == "close_mdd":
            y = future_close_mdd <= -spec["k"] * current_horizon_vol
        elif spec["kind"] == "close_loss":
            y = future_close_return <= -spec["k"] * current_horizon_vol
        else:
            raise ValueError(f"unknown variant kind: {spec['kind']}")

        if spec.get("require_close_negative", False):
            y = y & (future_close_return < 0)

        labels[f"y_confirm_{name}"] = y.astype(float)

    invalid = (
        current_horizon_vol.isna()
        | future_min_low.isna()
        | future_min_close.isna()
        | future_close.isna()
    )
    for c in labels.columns:
        if c.startswith("y_"):
            labels.loc[invalid, c] = np.nan

    return pd.concat([g, labels], axis=1).copy()


def build_dataset(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    parts = []
    for _, g in raw.groupby("asset_name", sort=False):
        fg = build_features_one_asset(g)
        lg = add_labels_one_asset(fg, cfg)
        parts.append(lg)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "asset_name"]).reset_index(drop=True)


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"date", "asset_name", "open", "high", "low", "close", "adj_close", "volume", "current_horizon_vol"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("y_") or c.startswith("future_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# ============================================================
# Modeling
# ============================================================

def make_splits(df: pd.DataFrame, cfg: Config) -> List[Dict]:
    dates = np.array(sorted(df["date"].dropna().unique()))
    splits = []
    start = cfg.min_train_days
    fold = 0
    while start + cfg.test_days <= len(dates):
        test_dates = dates[start:start + cfg.test_days]
        train_end = max(0, start - cfg.embargo_days)
        train_dates = dates[:train_end]
        if len(train_dates) >= cfg.min_train_days:
            splits.append({
                "fold_id": fold,
                "train_dates": set(pd.to_datetime(train_dates)),
                "test_dates": set(pd.to_datetime(test_dates)),
                "test_start": pd.Timestamp(test_dates[0]),
                "test_end": pd.Timestamp(test_dates[-1]),
            })
            fold += 1
        start += cfg.step_days
    return splits


def make_model(cfg: Config):
    kind = cfg.model_kind.lower()
    if kind == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=cfg.n_estimators,
            max_features="sqrt",
            min_samples_leaf=cfg.min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=cfg.random_state,
        )
    elif kind == "randomforest":
        clf = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_features="sqrt",
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
        raise ValueError(f"unknown model_kind: {cfg.model_kind}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])


def predict_proba_safe(model, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    if p.shape[1] == 1:
        cls = getattr(model.named_steps["model"], "classes_", np.array([0]))
        return np.ones(len(X)) if len(cls) == 1 and cls[0] == 1 else np.zeros(len(X))
    return p[:, 1]


def percentile_from_ref(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref, dtype=float)
    ref = ref[~np.isnan(ref)]
    if len(ref) == 0:
        return np.full(len(x), np.nan)
    ref = np.sort(ref)
    return np.searchsorted(ref, x, side="right") / len(ref)


def safe_roc(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def safe_ap(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def safe_brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    return float(brier_score_loss(y, np.clip(p, 0, 1))) if len(y) else np.nan


def ece(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    val = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mm = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        if mm.any():
            val += mm.mean() * abs(p[mm].mean() - y[mm].mean())
    return float(val)


def metric_row(y, p, name, fold_id=None, variant=None):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~np.isnan(y) & ~np.isnan(p)
    yv, pv = y[m], p[m]
    base = float(yv.mean()) if len(yv) else np.nan
    ap = safe_ap(yv, pv)
    br = safe_brier(yv, pv)
    ref_br = base * (1 - base) if not np.isnan(base) else np.nan
    return {
        "variant": variant,
        "head": name,
        "fold_id": fold_id,
        "n": int(len(yv)),
        "positive_rate": base,
        "roc_auc": safe_roc(yv, pv),
        "pr_auc": ap,
        "pr_ratio": ap / base if base and base > 0 else np.nan,
        "brier": br,
        "brier_skill": 1 - br / ref_br if ref_br and ref_br > 0 else np.nan,
        "ece": ece(yv, pv),
        "mean_score": float(pv.mean()) if len(pv) else np.nan,
    }


def train_fold(df: pd.DataFrame, feature_cols: List[str], split: Dict, cfg: Config):
    train = df[df["date"].isin(split["train_dates"])].copy()
    test = df[df["date"].isin(split["test_dates"])].copy()

    train = train.dropna(subset=["y_down_touch"])
    test = test.dropna(subset=["y_down_touch"])
    if train.empty or test.empty:
        return pd.DataFrame(), [], []

    dates = np.array(sorted(train["date"].unique()))
    if len(dates) > cfg.calibration_days + 100:
        cal_dates = set(pd.to_datetime(dates[-cfg.calibration_days:]))
        fit_dates = set(pd.to_datetime(dates[:-cfg.calibration_days]))
        fit = train[train["date"].isin(fit_dates)].copy()
        cal = train[train["date"].isin(cal_dates)].copy()
    else:
        fit = train.copy()
        cal = train.copy()

    X_fit = fit[feature_cols]
    X_cal = cal[feature_cols]
    X_test = test[feature_cols]

    out_base = test[[
        "asset_name", "date",
        "y_down_touch",
        "future_close_return_h", "future_close_mdd_h", "future_intraday_mdd_h",
        "current_horizon_vol",
    ]].copy()
    out_base["fold_id"] = split["fold_id"]

    metric_rows = []
    pred_parts = []

    # Candidate model once per fold.
    if fit["y_down_touch"].nunique() < 2:
        return pd.DataFrame(), [], []

    cand_model = make_model(cfg)
    cand_model.fit(X_fit, fit["y_down_touch"].astype(int))
    cand_cal_raw = predict_proba_safe(cand_model, X_cal)
    cand_test_raw = predict_proba_safe(cand_model, X_test)
    cand_test_pct = percentile_from_ref(cand_cal_raw, cand_test_raw)

    metric_rows.append(metric_row(
        test["y_down_touch"].to_numpy(),
        cand_test_raw,
        "candidate_down_touch_raw",
        fold_id=split["fold_id"],
        variant="candidate"
    ))
    metric_rows.append(metric_row(
        test["y_down_touch"].to_numpy(),
        cand_test_pct,
        "candidate_down_touch_percentile",
        fold_id=split["fold_id"],
        variant="candidate"
    ))

    for variant in CONFIRM_VARIANTS.keys():
        ycol = f"y_confirm_{variant}"
        tr = fit.dropna(subset=[ycol])
        ca = cal.dropna(subset=[ycol])
        te = test.dropna(subset=[ycol])

        if tr.empty or ca.empty or te.empty or tr[ycol].nunique() < 2:
            continue

        model = make_model(cfg)
        model.fit(tr[feature_cols], tr[ycol].astype(int))

        cal_raw = predict_proba_safe(model, ca[feature_cols])
        test_raw = predict_proba_safe(model, te[feature_cols])
        test_pct = percentile_from_ref(cal_raw, test_raw)

        p = te[[
            "asset_name", "date",
            "y_down_touch",
            ycol,
            "future_close_return_h", "future_close_mdd_h", "future_intraday_mdd_h",
            "current_horizon_vol",
        ]].copy()
        p = p.rename(columns={ycol: "y_confirm"})
        p["fold_id"] = split["fold_id"]
        p["variant"] = variant
        p["candidate_raw_proba"] = cand_test_raw[test.index.get_indexer(te.index)]
        p["candidate_score_percentile"] = cand_test_pct[test.index.get_indexer(te.index)]
        p["confirm_raw_proba"] = test_raw
        p["confirm_score_percentile"] = test_pct
        pred_parts.append(p)

        metric_rows.append(metric_row(te[ycol].to_numpy(), test_raw, "confirm_raw", fold_id=split["fold_id"], variant=variant))
        metric_rows.append(metric_row(te[ycol].to_numpy(), test_pct, "confirm_percentile", fold_id=split["fold_id"], variant=variant))

    pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return pred, metric_rows, []


def run_walk_forward(df: pd.DataFrame, feature_cols: List[str], cfg: Config):
    splits = make_splits(df, cfg)
    preds, metrics = [], []
    for sp in splits:
        p, m, _ = train_fold(df, feature_cols, sp, cfg)
        if not p.empty:
            preds.append(p)
        metrics.extend(m)
    return (
        pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        pd.DataFrame(metrics),
        splits,
    )


# ============================================================
# Analysis
# ============================================================

def summarize_head_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for (variant, head), g in metrics.groupby(["variant", "head"], dropna=False):
        rows.append({
            "variant": variant,
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


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, g in list(df.groupby("asset_name")) + [("ALL", df)]:
        row = {
            "asset_name": asset,
            "rows": int(len(g)),
            "date_start": str(g["date"].min().date()),
            "date_end": str(g["date"].max().date()),
            "y_down_touch_rate": float(g["y_down_touch"].mean()),
        }
        for v in CONFIRM_VARIANTS:
            row[f"y_confirm_{v}_rate"] = float(g[f"y_confirm_{v}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def decile_summary(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, d in preds.groupby("variant"):
        for score_col, label_col, score_name in [
            ("candidate_score_percentile", "y_down_touch", "candidate"),
            ("confirm_score_percentile", "y_confirm", "confirm"),
        ]:
            x = d[[score_col, label_col]].dropna().copy()
            if x.empty:
                continue
            x["decile"] = pd.qcut(x[score_col].rank(method="first"), 10, labels=False, duplicates="drop") + 1
            base = x[label_col].mean()
            for dec, g in x.groupby("decile"):
                rows.append({
                    "variant": variant,
                    "score": score_name,
                    "decile": int(dec),
                    "rows": int(len(g)),
                    "label_rate": float(g[label_col].mean()),
                    "base_rate": float(base),
                    "lift": float(g[label_col].mean() / base) if base > 0 else np.nan,
                    "score_mean": float(g[score_col].mean()),
                })
    return pd.DataFrame(rows)


def apply_policy(preds: pd.DataFrame, cthr: float, mthr: float) -> pd.Series:
    return (preds["candidate_score_percentile"] >= cthr) & (preds["confirm_score_percentile"] >= mthr)


def threshold_sweep(preds: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for variant, d in preds.groupby("variant"):
        base_confirm = d["y_confirm"].mean()
        for ct in cfg.candidate_thresholds:
            for mt in cfg.confirm_thresholds:
                sig = apply_policy(d, ct, mt)
                g = d[sig]
                rows.append({
                    "variant": variant,
                    "candidate_threshold": ct,
                    "confirm_threshold": mt,
                    "rows": int(len(g)),
                    "signal_rate": float(sig.mean()),
                    "confirm_rate": float(g["y_confirm"].mean()) if len(g) else np.nan,
                    "confirm_lift": float(g["y_confirm"].mean() / base_confirm) if len(g) and base_confirm > 0 else np.nan,
                    "coverage_of_confirm_events": float((sig & (d["y_confirm"] == 1)).sum() / (d["y_confirm"] == 1).sum()) if (d["y_confirm"] == 1).sum() > 0 else np.nan,
                    "false_alarm_rate": float((g["y_confirm"] == 0).mean()) if len(g) else np.nan,
                    "future_close_return_mean": float(g["future_close_return_h"].mean()) if len(g) else np.nan,
                    "future_close_return_median": float(g["future_close_return_h"].median()) if len(g) else np.nan,
                    "future_close_mdd_mean": float(g["future_close_mdd_h"].mean()) if len(g) else np.nan,
                    "future_intraday_mdd_mean": float(g["future_intraday_mdd_h"].mean()) if len(g) else np.nan,
                })
    return pd.DataFrame(rows)


def policy_summary(preds: pd.DataFrame, cfg: Config, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = []
    rows = []
    group_iter = preds.groupby(["variant"] + group_cols, dropna=False)
    for key, d in group_iter:
        if not isinstance(key, tuple):
            key = (key,)
        variant = key[0]
        group_vals = key[1:]
        sig = apply_policy(d, cfg.primary_candidate_threshold, cfg.primary_confirm_threshold)
        for name, mask in [
            ("CONFIRMED_DOWNSIDE_RISK", sig),
            ("NO_CONFIRMED_DOWNSIDE", ~sig),
        ]:
            g = d[mask]
            row = {"variant": variant, "policy_signal": name}
            row.update(dict(zip(group_cols, group_vals)))
            row.update({
                "rows": int(len(g)),
                "signal_rate": float(len(g) / len(d)) if len(d) else np.nan,
                "confirm_rate": float(g["y_confirm"].mean()) if len(g) else np.nan,
                "future_close_return_mean": float(g["future_close_return_h"].mean()) if len(g) else np.nan,
                "future_close_return_median": float(g["future_close_return_h"].median()) if len(g) else np.nan,
                "future_close_mdd_mean": float(g["future_close_mdd_h"].mean()) if len(g) else np.nan,
                "future_intraday_mdd_mean": float(g["future_intraday_mdd_h"].mean()) if len(g) else np.nan,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def decision_table(sweep: pd.DataFrame, preds: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    primary = sweep[
        (sweep["candidate_threshold"] == cfg.primary_candidate_threshold)
        & (sweep["confirm_threshold"] == cfg.primary_confirm_threshold)
    ].copy()

    # Also take best few candidates by a conservative score.
    sweep2 = sweep.copy()
    # Prefer negative future return, worse MDD, reasonable coverage, high lift.
    sweep2["score"] = (
        sweep2["confirm_lift"].fillna(0)
        + 2.0 * (-sweep2["future_close_return_mean"].fillna(0))
        + 5.0 * (-sweep2["future_close_mdd_mean"].fillna(0))
        + 0.5 * sweep2["coverage_of_confirm_events"].fillna(0)
        - 0.2 * sweep2["false_alarm_rate"].fillna(1)
    )
    best = sweep2.sort_values("score", ascending=False).groupby("variant").head(3)
    cand = pd.concat([primary, best], ignore_index=True).drop_duplicates(["variant", "candidate_threshold", "confirm_threshold"])

    for _, r in cand.iterrows():
        flags = []
        decision = "REJECT_OR_WATCH_ONLY"

        if r["rows"] < 50:
            flags.append("small_signal_count")
        if r["confirm_lift"] < 1.25:
            flags.append("weak_confirm_lift")
        if r["coverage_of_confirm_events"] < 0.08:
            flags.append("low_coverage")
        if r["future_close_return_mean"] >= 0:
            flags.append("future_return_not_negative")
        if r["future_close_mdd_mean"] > -0.025:
            flags.append("mdd_not_severe_enough")
        if r["false_alarm_rate"] > 0.65:
            flags.append("high_false_alarm")

        if not flags:
            decision = "PASS_AS_DOWNSIDE_CONFIRM_CANDIDATE"
        elif (
            r["future_close_return_mean"] < 0
            and r["future_close_mdd_mean"] <= -0.025
            and r["confirm_lift"] >= 1.20
        ):
            decision = "PROMISING_REQUIRE_BACKTEST"
        elif r["confirm_lift"] >= 1.25:
            decision = "WATCH_LABEL_OR_THRESHOLD"

        rows.append({
            **r.to_dict(),
            "decision": decision,
            "risk_flags": ";".join(flags) if flags else "none",
        })
    return pd.DataFrame(rows).sort_values(["decision", "score"], ascending=[True, False])


def run(args):
    cfg = Config(
        horizon=args.horizon,
        vol_window=args.vol_window,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        model_kind=args.model_kind,
        primary_candidate_threshold=args.primary_candidate_threshold,
        primary_confirm_threshold=args.primary_confirm_threshold,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_ohlcv(args)
    df = build_dataset(raw, cfg)
    feature_cols = get_feature_cols(df)
    valid = df.dropna(subset=["y_down_touch"]).copy()

    preds, metrics, splits = run_walk_forward(valid, feature_cols, cfg)
    if preds.empty:
        raise RuntimeError("No OOS predictions. Check data and split settings.")

    metrics_summary = summarize_head_metrics(metrics)
    lab = label_distribution(valid)
    dec = decile_summary(preds)
    sw = threshold_sweep(preds, cfg)
    pol = policy_summary(preds, cfg)
    pol_asset = policy_summary(preds, cfg, ["asset_name"])
    preds["year"] = preds["date"].dt.year
    pol_annual = policy_summary(preds, cfg, ["asset_name", "year"])
    pol_fold = policy_summary(preds, cfg, ["asset_name", "fold_id"])
    decisions = decision_table(sw, preds, cfg)

    config_json = {
        "experiment": "downside_confirm_label_repair_v7_2",
        "config": asdict(cfg),
        "confirm_variants": CONFIRM_VARIANTS,
        "feature_count": len(feature_cols),
        "feature_note": "future_* and y_* columns are excluded from feature_cols",
        "score_note": "score_percentile is a calibration-window rank, not literal probability",
    }

    summary_json = {
        "experiment": "downside_confirm_label_repair_v7_2",
        "asset_count": int(valid["asset_name"].nunique()),
        "rows_valid": int(len(valid)),
        "oos_rows": int(len(preds)),
        "date_start": str(valid["date"].min().date()),
        "date_end": str(valid["date"].max().date()),
        "fold_count": int(preds["fold_id"].nunique()),
        "feature_count": int(len(feature_cols)),
        "label_distribution_all": lab[lab["asset_name"] == "ALL"].to_dict(orient="records")[0],
        "top_decisions": decisions.head(20).to_dict(orient="records"),
        "head_metric_summary": metrics_summary.to_dict(orient="records"),
        "note": "This is label repair diagnostics, not final allocation backtest.",
    }

    outputs = {
        "summary": save_json(out_dir / "confirm_label_repair_summary.json", summary_json),
        "config": save_json(out_dir / "confirm_label_repair_config.json", config_json),
        "oos_predictions": save_csv(out_dir / "confirm_variant_oos_predictions.csv", preds),
        "head_metrics": save_csv(out_dir / "confirm_variant_head_metrics.csv", metrics),
        "head_metric_summary": save_csv(out_dir / "confirm_variant_head_metric_summary.csv", metrics_summary),
        "label_distribution": save_csv(out_dir / "label_distribution.csv", lab),
        "decile_summary": save_csv(out_dir / "confirm_variant_decile_summary.csv", dec),
        "threshold_sweep": save_csv(out_dir / "confirm_variant_threshold_sweep.csv", sw),
        "policy_summary": save_csv(out_dir / "confirm_variant_policy_summary.csv", pol),
        "asset_summary": save_csv(out_dir / "confirm_variant_asset_summary.csv", pol_asset),
        "annual_summary": save_csv(out_dir / "confirm_variant_annual_summary.csv", pol_annual),
        "fold_summary": save_csv(out_dir / "confirm_variant_fold_summary.csv", pol_fold),
        "decision_table": save_csv(out_dir / "confirm_variant_decision_table.csv", decisions),
        "feature_cols": save_csv(out_dir / "feature_cols.csv", pd.DataFrame({"feature": feature_cols})),
    }
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv-inputs", default=None)
    parser.add_argument("--asset-names", default=None)
    parser.add_argument("--ohlcv-all", default=None)
    parser.add_argument("--output-dir", default="downside_confirm_label_repair_v7_2_output")

    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--vol-window", type=int, default=60)

    parser.add_argument("--model-kind", default="extratrees", choices=["extratrees", "randomforest", "hgb"])
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--min-samples-leaf", type=int, default=20)

    parser.add_argument("--primary-candidate-threshold", type=float, default=0.90)
    parser.add_argument("--primary-confirm-threshold", type=float, default=0.90)

    args = parser.parse_args()
    outputs = run(args)
    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    print("[OK] Downside Confirm Label Repair v7.2 completed.")
    print(json.dumps({
        "asset_count": summary["asset_count"],
        "rows_valid": summary["rows_valid"],
        "oos_rows": summary["oos_rows"],
        "date_start": summary["date_start"],
        "date_end": summary["date_end"],
        "fold_count": summary["fold_count"],
        "feature_count": summary["feature_count"],
        "label_distribution_all": summary["label_distribution_all"],
        "top_decisions": summary["top_decisions"][:10],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
