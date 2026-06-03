# -*- coding: utf-8 -*-
"""
return_seeking_percentile_threshold_validator.py

5단계: Return-Seeking Up-touch percentile threshold 검증 스크립트.

목적
----
4단계에서 확정된 수익추구형 모델:
- Label: y_up_touch_fixed_h10_k1.0
- Model: ExtraTreesClassifier
- Output: score_percentile

이 모델의 score_percentile threshold가 실제로 안정적인지 검증합니다.

검증 threshold
--------------
기본:
- score_percentile >= 0.90
- score_percentile >= 0.80
- score_percentile >= 0.70
- score_percentile >= 0.60

검증 지표
---------
1. signal_count
2. signal_rate
3. actual_up_touch_rate among signals
4. base_up_touch_rate
5. lift = signal_actual_rate / base_rate
6. recall = captured_positive / total_positive
7. no_signal_actual_rate
8. spread = signal_actual_rate - no_signal_actual_rate
9. fold_positive_lift_rate
10. asset/year stability

중요
----
이 스크립트도 allocation / portfolio 성과는 평가하지 않습니다.
오직 percentile threshold가 유효한 ranking filter인지 검증합니다.

실행 예시
---------
python return_seeking_percentile_threshold_validator.py ^
  --inputs "QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv" ^
  --asset-names "QQQ,SPY,SOXX,XLK" ^
  --output-dir "return_seeking_threshold_validation_all"

출력
----
output_dir/
├─ threshold_validation_summary.json
├─ best_threshold_config.json
├─ threshold_summary.csv
├─ asset_threshold_summary.csv
├─ fold_threshold_metrics.csv
├─ annual_threshold_summary.csv
├─ score_bin_analysis.csv
├─ oos_predictions.csv
└─ score_distribution.csv

의존성
------
python>=3.10
pandas
numpy
scikit-learn
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)


# ============================================================
# 0. Utils
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


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def safe_divide(a: float, b: float, default: float = np.nan) -> float:
    try:
        if b == 0 or pd.isna(b):
            return default
        return float(a / b)
    except Exception:
        return default


def rolling_min_periods(window: int, floor: int = 5, frac: float = 1 / 3) -> int:
    return max(1, min(int(window), max(int(floor), int(window * frac))))


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    rename_map = {
        "datetime": "date",
        "timestamp": "date",
        "adjclose": "adj_close",
        "adj_close": "adj_close",
        "adjusted_close": "adj_close",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}. columns={list(out.columns)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]

    if "volume" not in out.columns:
        out["volume"] = np.nan

    return out


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    return normalize_ohlcv_columns(pd.read_csv(path))


# ============================================================
# 1. Feature Builder
# ============================================================

def rolling_mdd(close: pd.Series, window: int) -> pd.Series:
    roll_max = close.rolling(window, min_periods=rolling_min_periods(window, floor=10)).max()
    return close / roll_max.replace(0, np.nan) - 1.0


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=rolling_min_periods(window, floor=5)).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=rolling_min_periods(window, floor=5)).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy().sort_values("date").reset_index(drop=True)

    close = d["adj_close"].astype(float)
    raw_close = d["close"].astype(float)
    open_ = d["open"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    volume = d["volume"].astype(float)

    new_cols: Dict[str, pd.Series] = {}

    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    new_cols["ret_1d"] = ret
    new_cols["log_ret_1d"] = log_ret

    for w in [2, 3, 5, 10, 20, 40, 60, 120, 252]:
        new_cols[f"ret_{w}d"] = close.pct_change(w)
        new_cols[f"log_ret_sum_{w}d"] = log_ret.rolling(w, min_periods=rolling_min_periods(w)).sum()
        new_cols[f"vol_{w}d"] = ret.rolling(w, min_periods=rolling_min_periods(w)).std()
        new_cols[f"down_vol_{w}d"] = ret.where(ret < 0, 0.0).rolling(w, min_periods=rolling_min_periods(w)).std()
        new_cols[f"up_vol_{w}d"] = ret.where(ret > 0, 0.0).rolling(w, min_periods=rolling_min_periods(w)).std()
        new_cols[f"mdd_{w}d"] = rolling_mdd(close, w)
        new_cols[f"ret_z_{w}d"] = (
            ret.rolling(w, min_periods=rolling_min_periods(w)).mean()
            / ret.rolling(w, min_periods=rolling_min_periods(w)).std().replace(0, np.nan)
        )
        new_cols[f"skew_{w}d"] = ret.rolling(w, min_periods=rolling_min_periods(w)).skew()
        new_cols[f"kurt_{w}d"] = ret.rolling(w, min_periods=rolling_min_periods(w)).kurt()

    for a, b in [(5, 20), (10, 40), (20, 60), (20, 120), (60, 252)]:
        new_cols[f"vol_ratio_{a}_{b}"] = new_cols[f"vol_{a}d"] / new_cols[f"vol_{b}d"].replace(0, np.nan)
        new_cols[f"down_vol_ratio_{a}_{b}"] = new_cols[f"down_vol_{a}d"] / new_cols[f"down_vol_{b}d"].replace(0, np.nan)

    for w in [5, 10, 20, 40, 60, 120, 200]:
        ma = close.rolling(w, min_periods=rolling_min_periods(w)).mean()
        new_cols[f"ma_gap_{w}d"] = close / ma.replace(0, np.nan) - 1.0
        new_cols[f"ma_slope_5d_{w}d"] = ma.pct_change(5)
        new_cols[f"ma_slope_20d_{w}d"] = ma.pct_change(20)

    for w in [7, 14, 21]:
        new_cols[f"rsi_{w}d"] = rsi(close, w)

    for w in [10, 20, 40, 60, 120, 252]:
        rolling_high = high.rolling(w, min_periods=rolling_min_periods(w)).max()
        rolling_low = low.rolling(w, min_periods=rolling_min_periods(w)).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)

        new_cols[f"dist_to_high_{w}d"] = close / rolling_high.replace(0, np.nan) - 1.0
        new_cols[f"dist_to_low_{w}d"] = close / rolling_low.replace(0, np.nan) - 1.0
        new_cols[f"range_position_{w}d"] = (close - rolling_low) / price_range
        new_cols[f"breakout_pressure_{w}d"] = high / rolling_high.shift(1).replace(0, np.nan) - 1.0
        new_cols[f"breakdown_pressure_{w}d"] = low / rolling_low.shift(1).replace(0, np.nan) - 1.0

    hl_range = (high - low) / raw_close.replace(0, np.nan)
    oc_ret = raw_close / open_.replace(0, np.nan) - 1.0
    gap_ret = open_ / raw_close.shift(1).replace(0, np.nan) - 1.0
    upper_shadow = (high - np.maximum(open_, raw_close)) / raw_close.replace(0, np.nan)
    lower_shadow = (np.minimum(open_, raw_close) - low) / raw_close.replace(0, np.nan)
    body_size = (raw_close - open_).abs() / raw_close.replace(0, np.nan)

    new_cols["hl_range"] = hl_range
    new_cols["oc_ret"] = oc_ret
    new_cols["gap_ret"] = gap_ret
    new_cols["upper_shadow"] = upper_shadow
    new_cols["lower_shadow"] = lower_shadow
    new_cols["body_size"] = body_size

    prev_close = raw_close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    for w in [5, 10, 20, 40, 60]:
        new_cols[f"atr_pct_{w}d"] = true_range.rolling(w, min_periods=rolling_min_periods(w)).mean() / raw_close.replace(0, np.nan)
        new_cols[f"range_vol_{w}d"] = hl_range.rolling(w, min_periods=rolling_min_periods(w)).mean()

    if not volume.isna().all():
        vol_log = np.log1p(volume)
        new_cols["volume_log"] = vol_log
        for w in [5, 10, 20, 60, 120]:
            vol_ma = volume.rolling(w, min_periods=rolling_min_periods(w, floor=3)).mean()
            vol_std = vol_log.rolling(w, min_periods=rolling_min_periods(w, floor=3)).std()
            new_cols[f"volume_ratio_{w}d"] = volume / vol_ma.replace(0, np.nan)
            new_cols[f"volume_chg_{w}d"] = volume.pct_change(w)
            new_cols[f"volume_z_{w}d"] = (
                (vol_log - vol_log.rolling(w, min_periods=rolling_min_periods(w, floor=3)).mean())
                / vol_std.replace(0, np.nan)
            )
            new_cols[f"price_volume_corr_{w}d"] = ret.rolling(w, min_periods=rolling_min_periods(w, floor=5)).corr(volume.pct_change())

    new_cols["trend_score_20_60"] = (
        0.5 * (close / close.rolling(20, min_periods=10).mean().replace(0, np.nan) - 1.0)
        + 0.5 * (close / close.rolling(60, min_periods=20).mean().replace(0, np.nan) - 1.0)
    )
    new_cols["vol_adjusted_momentum_20"] = close.pct_change(20) / ret.rolling(20, min_periods=10).std().replace(0, np.nan)
    new_cols["drawdown_recovery_20_120"] = rolling_mdd(close, 20) - rolling_mdd(close, 120)

    feat_df = pd.DataFrame(new_cols)
    d = pd.concat([d, feat_df], axis=1).copy()

    excluded_cols = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    feature_cols = [
        c for c in d.columns
        if c not in excluded_cols
        and not c.startswith(("y_", "future_", "meta_", "label_", "target_"))
        and pd.api.types.is_numeric_dtype(d[c])
    ]

    bad = [c for c in feature_cols if c.startswith(("y_", "future_", "meta_", "label_", "target_"))]
    if bad:
        raise RuntimeError(f"leakage columns in features: {bad}")

    return d, feature_cols


# ============================================================
# 2. Label Builder
# ============================================================

def explicit_future_high_low(high: pd.Series, low: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series]:
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    n = len(h)

    future_high = np.full(n, np.nan)
    future_low = np.full(n, np.nan)

    for i in range(0, n - horizon):
        future_high[i] = np.nanmax(h[i + 1 : i + 1 + horizon])
        future_low[i] = np.nanmin(l[i + 1 : i + 1 + horizon])

    return pd.Series(future_high, index=high.index), pd.Series(future_low, index=low.index)


def current_horizon_volatility(close: pd.Series, horizon: int, vol_window: int) -> pd.Series:
    returns = close.astype(float).pct_change()
    min_periods = max(20, min(vol_window, vol_window // 2))
    return returns.rolling(vol_window, min_periods=min_periods).std().shift(1) * math.sqrt(horizon)


def make_up_touch_label(df: pd.DataFrame, horizon: int, vol_window: int, k: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    out["current_horizon_vol"] = current_horizon_volatility(close, horizon, vol_window)
    out["upper_barrier"] = close * (1.0 + k * out["current_horizon_vol"])
    fh, _ = explicit_future_high_low(high, low, horizon)
    out["future_high_h"] = fh

    # Extra diagnostics, not used as model target.
    out["future_max_high_return"] = out["future_high_h"] / close.replace(0, np.nan) - 1.0

    out["y_up_touch"] = (out["future_high_h"] >= out["upper_barrier"]).astype(float)

    invalid = (
        out["current_horizon_vol"].isna()
        | out["upper_barrier"].isna()
        | out["future_high_h"].isna()
    )
    out.loc[invalid, "y_up_touch"] = np.nan
    return out


# ============================================================
# 3. Fold / Model / Score
# ============================================================

@dataclass
class Fold:
    fold_id: int
    train_start: int
    train_end: int
    cal_start: int
    cal_end: int
    test_start: int
    test_end: int


def make_rolling_folds(n: int, train_window: int, calibration_window: int, test_window: int, embargo: int, max_folds: int = 0) -> List[Fold]:
    folds: List[Fold] = []
    start = 0
    fold_id = 0

    while True:
        train_start = start
        train_end = train_start + train_window
        cal_start = train_end
        cal_end = cal_start + calibration_window
        test_start = cal_end + embargo
        test_end = test_start + test_window

        if test_end > n:
            break

        folds.append(Fold(fold_id, train_start, train_end, cal_start, cal_end, test_start, test_end))
        fold_id += 1
        if max_folds and fold_id >= max_folds:
            break
        start += test_window

    return folds


def fold_indices(fold: Fold, horizon: int, n: int) -> Dict[str, np.ndarray]:
    train_end_safe = max(fold.train_start, fold.train_end - horizon)
    cal_end_safe = max(fold.cal_start, fold.cal_end - horizon)

    return {
        "train": np.arange(fold.train_start, train_end_safe),
        "cal": np.arange(fold.cal_start, cal_end_safe),
        "test": np.arange(fold.test_start, min(fold.test_end, n)),
    }


def make_model(args, random_state: int) -> Pipeline:
    model_name = args.model.lower()

    if model_name == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", clf),
        ])

    if model_name == "randomforest":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", clf),
        ])

    if model_name == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=args.hgb_iter,
            learning_rate=args.hgb_learning_rate,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            l2_regularization=args.hgb_l2,
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", clf),
        ])

    if model_name == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs",
                C=args.logistic_c,
                max_iter=1500,
                class_weight="balanced",
                random_state=random_state,
            )),
        ])

    raise ValueError(f"unknown model: {args.model}")


def predict_positive_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def percentile_from_calibration(raw_cal: np.ndarray, raw_test: np.ndarray) -> np.ndarray:
    cal = pd.Series(raw_cal).dropna().astype(float)
    if len(cal) == 0:
        return np.full(len(raw_test), np.nan)
    sorted_cal = np.sort(cal.to_numpy())
    return np.searchsorted(sorted_cal, raw_test, side="right") / len(sorted_cal)


def expected_calibration_error(y_true: pd.Series, prob: pd.Series, bins: int = 10) -> Tuple[float, pd.DataFrame]:
    y = pd.Series(y_true).astype(float)
    p = pd.Series(prob).astype(float)
    mask = y.notna() & p.notna() & np.isfinite(p)
    y = y[mask].astype(int)
    p = p[mask].astype(float)

    if len(y) == 0:
        return np.nan, pd.DataFrame()

    edges = np.linspace(0, 1, bins + 1)
    ids = np.digitize(p, edges[1:-1], right=True)

    rows = []
    ece = 0.0
    for b in range(bins):
        m = ids == b
        if not np.any(m):
            rows.append({"bin": b, "count": 0, "avg_prob": np.nan, "actual_rate": np.nan, "abs_gap": np.nan})
            continue
        avg_prob = float(p[m].mean())
        actual_rate = float(y[m].mean())
        gap = abs(avg_prob - actual_rate)
        ece += float(np.mean(m)) * gap
        rows.append({"bin": b, "count": int(np.sum(m)), "avg_prob": avg_prob, "actual_rate": actual_rate, "abs_gap": gap})
    return float(ece), pd.DataFrame(rows)


def binary_ranking_metrics(y_true: pd.Series, prob: pd.Series) -> Dict:
    y = pd.Series(y_true).astype(float)
    p = pd.Series(prob).astype(float)
    mask = y.notna() & p.notna() & np.isfinite(p)
    y = y[mask].astype(int)
    p = p[mask].astype(float)

    out = {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "pr_auc": np.nan,
        "pr_ratio": np.nan,
        "pr_lift": np.nan,
        "roc_auc": np.nan,
        "brier": np.nan,
        "brier_skill": np.nan,
        "ece": np.nan,
    }

    if len(y) == 0:
        return out

    base = out["positive_rate"]

    if y.nunique() == 2:
        pr_auc = float(average_precision_score(y, p))
        out["pr_auc"] = pr_auc
        out["pr_ratio"] = safe_divide(pr_auc, base)
        out["pr_lift"] = pr_auc - base
        out["roc_auc"] = float(roc_auc_score(y, p))
        brier = float(brier_score_loss(y, p))
        base_brier = float(brier_score_loss(y, np.full(len(y), base)))
        out["brier"] = brier
        out["brier_skill"] = 1.0 - safe_divide(brier, base_brier)
        ece, _ = expected_calibration_error(y, p, 10)
        out["ece"] = ece
    else:
        out["pr_auc"] = base
        out["pr_ratio"] = 1.0
        out["pr_lift"] = 0.0

    return out


# ============================================================
# 4. Threshold Metrics
# ============================================================

def threshold_metrics_for_group(g: pd.DataFrame, threshold: float) -> Dict:
    valid = g.dropna(subset=["y_true", "score_percentile"]).copy()
    if valid.empty:
        return {
            "rows": 0,
            "threshold": threshold,
        }

    y = valid["y_true"].astype(int)
    s = valid["score_percentile"].astype(float)
    signal = s >= threshold

    total_rows = len(valid)
    base_rate = float(y.mean())
    signal_count = int(signal.sum())
    signal_rate = signal_count / total_rows if total_rows else np.nan
    positive_count = int(y.sum())

    if signal_count > 0:
        signal_actual_rate = float(y[signal].mean())
        signal_positive_count = int(y[signal].sum())
        future_max_high_return_mean = float(valid.loc[signal, "future_max_high_return"].mean()) if "future_max_high_return" in valid.columns else np.nan
        future_max_high_return_median = float(valid.loc[signal, "future_max_high_return"].median()) if "future_max_high_return" in valid.columns else np.nan
    else:
        signal_actual_rate = np.nan
        signal_positive_count = 0
        future_max_high_return_mean = np.nan
        future_max_high_return_median = np.nan

    no_signal = ~signal
    if no_signal.sum() > 0:
        no_signal_actual_rate = float(y[no_signal].mean())
    else:
        no_signal_actual_rate = np.nan

    lift = safe_divide(signal_actual_rate, base_rate)
    recall = safe_divide(signal_positive_count, positive_count)
    precision_minus_base = signal_actual_rate - base_rate if not pd.isna(signal_actual_rate) and not pd.isna(base_rate) else np.nan
    spread_vs_no_signal = signal_actual_rate - no_signal_actual_rate if not pd.isna(signal_actual_rate) and not pd.isna(no_signal_actual_rate) else np.nan

    return {
        "rows": int(total_rows),
        "threshold": threshold,
        "base_rate": base_rate,
        "signal_count": signal_count,
        "signal_rate": signal_rate,
        "signal_actual_rate": signal_actual_rate,
        "signal_positive_count": signal_positive_count,
        "positive_count": positive_count,
        "recall": recall,
        "lift": lift,
        "precision_minus_base": precision_minus_base,
        "no_signal_actual_rate": no_signal_actual_rate,
        "spread_vs_no_signal": spread_vs_no_signal,
        "future_max_high_return_mean": future_max_high_return_mean,
        "future_max_high_return_median": future_max_high_return_median,
    }


def score_bin_analysis(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()

    rows = []
    for asset_name, g in preds.groupby("asset_name"):
        valid = g.dropna(subset=["y_true", "score_percentile"]).copy()
        if valid.empty:
            continue

        try:
            valid["score_bin"] = pd.qcut(
                valid["score_percentile"].rank(method="first"),
                q=bins,
                labels=False,
                duplicates="drop",
            ) + 1
        except Exception:
            valid["score_bin"] = np.nan

        base = float(valid["y_true"].mean())
        for b, bg in valid.groupby("score_bin"):
            actual = float(bg["y_true"].mean())
            rows.append({
                "asset_name": asset_name,
                "score_bin": int(b),
                "count": int(len(bg)),
                "base_rate": base,
                "actual_rate": actual,
                "lift": safe_divide(actual, base),
                "mean_score_percentile": float(bg["score_percentile"].mean()),
                "mean_prob": float(bg["prob"].mean()),
                "future_max_high_return_mean": float(bg["future_max_high_return"].mean()) if "future_max_high_return" in bg.columns else np.nan,
                "future_max_high_return_median": float(bg["future_max_high_return"].median()) if "future_max_high_return" in bg.columns else np.nan,
            })

    return pd.DataFrame(rows)


def summarize_thresholds(preds: pd.DataFrame, thresholds: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    asset_rows = []
    annual_rows = []

    preds = preds.copy()
    preds["year"] = pd.to_datetime(preds["date"]).dt.year

    for threshold in thresholds:
        # Fold-level
        for (asset_name, fold_id), g in preds.groupby(["asset_name", "fold_id"]):
            row = threshold_metrics_for_group(g, threshold)
            row.update({
                "asset_name": asset_name,
                "fold_id": fold_id,
            })
            fold_rows.append(row)

        # Asset-level
        for asset_name, g in preds.groupby("asset_name"):
            row = threshold_metrics_for_group(g, threshold)
            row.update({
                "asset_name": asset_name,
            })
            asset_rows.append(row)

        # Annual asset-level
        for (asset_name, year), g in preds.groupby(["asset_name", "year"]):
            row = threshold_metrics_for_group(g, threshold)
            row.update({
                "asset_name": asset_name,
                "year": int(year),
            })
            annual_rows.append(row)

    return pd.DataFrame(fold_rows), pd.DataFrame(asset_rows), pd.DataFrame(annual_rows)


def aggregate_threshold_summary(fold_df: pd.DataFrame, asset_df: pd.DataFrame, annual_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()

    rows = []
    for threshold, g in fold_df.groupby("threshold"):
        ag = asset_df[asset_df["threshold"] == threshold].copy()
        yg = annual_df[annual_df["threshold"] == threshold].copy()

        row = {
            "threshold": threshold,
            "asset_count": int(ag["asset_name"].nunique()) if not ag.empty else 0,
            "fold_count": int(g["fold_id"].nunique()),
            "total_signal_count": int(g["signal_count"].sum()),
            "mean_signal_rate": float(g["signal_rate"].mean()),
            "median_signal_rate": float(g["signal_rate"].median()),
            "mean_signal_actual_rate": float(g["signal_actual_rate"].mean()),
            "median_signal_actual_rate": float(g["signal_actual_rate"].median()),
            "mean_base_rate": float(g["base_rate"].mean()),
            "median_base_rate": float(g["base_rate"].median()),
            "mean_lift": float(g["lift"].mean()),
            "median_lift": float(g["lift"].median()),
            "min_asset_lift": float(ag["lift"].min()) if not ag.empty else np.nan,
            "mean_asset_lift": float(ag["lift"].mean()) if not ag.empty else np.nan,
            "fold_positive_lift_rate": float((g["lift"] > 1.0).mean()),
            "asset_positive_lift_rate": float((ag["lift"] > 1.0).mean()) if not ag.empty else np.nan,
            "annual_positive_lift_rate": float((yg["lift"] > 1.0).mean()) if not yg.empty else np.nan,
            "mean_recall": float(g["recall"].mean()),
            "median_recall": float(g["recall"].median()),
            "mean_spread_vs_no_signal": float(g["spread_vs_no_signal"].mean()),
            "median_spread_vs_no_signal": float(g["spread_vs_no_signal"].median()),
            "mean_future_max_high_return": float(g["future_max_high_return_mean"].mean()),
            "median_future_max_high_return": float(g["future_max_high_return_median"].median()),
        }

        # Scoring balances lift and enough coverage.
        signal_rate_penalty = 0.0
        if row["mean_signal_rate"] < 0.03:
            signal_rate_penalty = 0.30
        elif row["mean_signal_rate"] < 0.05:
            signal_rate_penalty = 0.15

        row["threshold_score"] = (
            0.35 * row["median_lift"]
            + 0.25 * row["fold_positive_lift_rate"]
            + 0.20 * row["min_asset_lift"]
            + 0.10 * row["annual_positive_lift_rate"]
            + 0.10 * row["median_recall"]
            - signal_rate_penalty
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("threshold_score", ascending=False)


def score_distribution(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()

    rows = []
    for asset_name, g in preds.groupby("asset_name"):
        for col in ["prob", "raw_prob", "score_percentile"]:
            s = g[col].dropna().astype(float)
            if s.empty:
                continue
            rows.append({
                "asset_name": asset_name,
                "score_col": col,
                "count": int(len(s)),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "p05": float(s.quantile(0.05)),
                "p10": float(s.quantile(0.10)),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "p90": float(s.quantile(0.90)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            })
    return pd.DataFrame(rows)


# ============================================================
# 5. Runner
# ============================================================

def fit_predict_asset(asset_name: str, input_path: str, args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_ohlcv(input_path)
    if args.start_date:
        raw = raw[raw["date"] >= pd.to_datetime(args.start_date)].copy()
    if args.end_date:
        raw = raw[raw["date"] <= pd.to_datetime(args.end_date)].copy()
    raw = raw.reset_index(drop=True)

    data, feature_cols = build_features(raw)
    labels = make_up_touch_label(data, horizon=args.horizon, vol_window=args.vol_window, k=args.k)
    y = labels["y_up_touch"]

    folds = make_rolling_folds(
        n=len(data),
        train_window=args.train_window,
        calibration_window=args.calibration_window,
        test_window=args.test_window,
        embargo=max(args.embargo, args.horizon),
        max_folds=args.max_folds,
    )

    metric_rows = []
    pred_parts = []
    cal_bin_parts = []

    for fold in folds:
        idx = fold_indices(fold, args.horizon, len(data))
        train_idx, cal_idx, test_idx = idx["train"], idx["cal"], idx["test"]

        y_train = y.iloc[train_idx]
        y_cal = y.iloc[cal_idx]
        y_test = y.iloc[test_idx]

        train_mask = y_train.notna()
        cal_mask = y_cal.notna()

        base_metric = {
            "asset_name": asset_name,
            "fold_id": fold.fold_id,
            "train_rows": int(train_mask.sum()),
            "cal_rows": int(cal_mask.sum()),
            "test_rows": int(y_test.notna().sum()),
            "train_positive_rate": float(y_train[train_mask].mean()) if train_mask.sum() else np.nan,
            "cal_positive_rate": float(y_cal[cal_mask].mean()) if cal_mask.sum() else np.nan,
            "usable": False,
            "train_start_date": data["date"].iloc[fold.train_start],
            "train_end_date": data["date"].iloc[fold.train_end - 1],
            "cal_start_date": data["date"].iloc[fold.cal_start],
            "cal_end_date": data["date"].iloc[fold.cal_end - 1],
            "test_start_date": data["date"].iloc[fold.test_start],
            "test_end_date": data["date"].iloc[fold.test_end - 1],
        }

        if train_mask.sum() < args.min_train_rows or y_train[train_mask].nunique() < 2:
            metric_rows.append(base_metric)
            continue

        model = make_model(args, random_state=args.random_state + fold.fold_id)
        model.fit(data.iloc[train_idx].loc[train_mask, feature_cols], y_train[train_mask].astype(int))

        raw_cal = predict_positive_proba(model, data.iloc[cal_idx][feature_cols])
        raw_test = predict_positive_proba(model, data.iloc[test_idx][feature_cols])
        score_percentile = percentile_from_calibration(raw_cal, raw_test)

        # Use raw model probability for probability column.
        # Threshold validation uses score_percentile, not literal probability.
        prob_test = raw_test

        ranking_metric = binary_ranking_metrics(y_test.reset_index(drop=True), pd.Series(prob_test))
        base_metric.update(ranking_metric)
        base_metric["usable"] = True
        metric_rows.append(base_metric)

        ece, cal_bins = expected_calibration_error(y_test.reset_index(drop=True), pd.Series(prob_test), 10)
        if not cal_bins.empty:
            cal_bins["asset_name"] = asset_name
            cal_bins["fold_id"] = fold.fold_id
            cal_bin_parts.append(cal_bins)

        pred = pd.DataFrame({
            "asset_name": asset_name,
            "date": data["date"].iloc[test_idx].to_numpy(),
            "fold_id": fold.fold_id,
            "y_true": y.iloc[test_idx].to_numpy(),
            "prob": prob_test,
            "raw_prob": raw_test,
            "score_percentile": score_percentile,
            "future_max_high_return": labels["future_max_high_return"].iloc[test_idx].to_numpy(),
        })
        pred_parts.append(pred)

    fold_model_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    calibration_bins = pd.concat(cal_bin_parts, ignore_index=True) if cal_bin_parts else pd.DataFrame()

    return fold_model_metrics, predictions, calibration_bins


def run(args) -> Dict[str, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = parse_list(args.inputs)
    asset_names = parse_list(args.asset_names)
    thresholds = parse_float_list(args.thresholds)

    if len(inputs) != len(asset_names):
        raise ValueError(f"inputs count != asset_names count: {len(inputs)} vs {len(asset_names)}")

    all_metrics = []
    all_preds = []
    all_cal = []
    asset_periods = []

    for asset_name, input_path in zip(asset_names, inputs):
        raw = load_ohlcv(input_path)
        if args.start_date:
            raw = raw[raw["date"] >= pd.to_datetime(args.start_date)].copy()
        if args.end_date:
            raw = raw[raw["date"] <= pd.to_datetime(args.end_date)].copy()
        asset_periods.append({
            "asset_name": asset_name,
            "input": str(input_path),
            "start": str(raw["date"].min().date()),
            "end": str(raw["date"].max().date()),
            "rows": int(len(raw)),
        })

        fm, preds, cal = fit_predict_asset(asset_name, input_path, args)
        all_metrics.append(fm)
        all_preds.append(preds)
        all_cal.append(cal)

    fold_model_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    calibration_bins = pd.concat([x for x in all_cal if not x.empty], ignore_index=True) if all_cal else pd.DataFrame()

    fold_threshold, asset_threshold, annual_threshold = summarize_thresholds(predictions, thresholds)
    threshold_summary = aggregate_threshold_summary(fold_threshold, asset_threshold, annual_threshold)
    bin_analysis = score_bin_analysis(predictions, bins=10)
    dist = score_distribution(predictions)

    best_row = threshold_summary.iloc[0].to_dict() if not threshold_summary.empty else {}

    best_config = {
        "experiment": "return_seeking_percentile_threshold_validator",
        "asset_periods": asset_periods,
        "model": {
            "objective": "return_seeking",
            "model": args.model,
            "label": f"y_up_touch_fixed_h{args.horizon}_k{args.k}",
            "score": "score_percentile",
            "score_interpretation": "calibration-window percentile rank of raw model score",
        },
        "thresholds_tested": thresholds,
        "best_threshold": best_row,
        "usage_note": {
            "literal_probability": False,
            "recommended_use": "Use score_percentile threshold as ranking filter only.",
            "do_not_use_as": "Direct buy/sell or allocation rule without separate portfolio validation.",
        },
    }

    summary = {
        "experiment": "return_seeking_percentile_threshold_validator",
        "asset_count": len(asset_names),
        "asset_periods": asset_periods,
        "model": args.model,
        "horizon": args.horizon,
        "k": args.k,
        "thresholds": thresholds,
        "fold_model_metric_rows": int(len(fold_model_metrics)),
        "prediction_rows": int(len(predictions)),
        "best_threshold": best_row,
        "decision_note": (
            "This validates percentile thresholds for the return-seeking Up-touch ranker. "
            "No allocation or portfolio returns are evaluated."
        ),
    }

    outputs = {
        "summary": save_json(out_dir / "threshold_validation_summary.json", summary),
        "best_config": save_json(out_dir / "best_threshold_config.json", best_config),
        "fold_model_metrics": save_csv(out_dir / "fold_model_metrics.csv", fold_model_metrics),
        "threshold_summary": save_csv(out_dir / "threshold_summary.csv", threshold_summary),
        "asset_threshold_summary": save_csv(out_dir / "asset_threshold_summary.csv", asset_threshold),
        "fold_threshold_metrics": save_csv(out_dir / "fold_threshold_metrics.csv", fold_threshold),
        "annual_threshold_summary": save_csv(out_dir / "annual_threshold_summary.csv", annual_threshold),
        "score_bin_analysis": save_csv(out_dir / "score_bin_analysis.csv", bin_analysis),
        "score_distribution": save_csv(out_dir / "score_distribution.csv", dist),
        "calibration_bins": save_csv(out_dir / "calibration_bins.csv", calibration_bins),
        "oos_predictions": save_csv(out_dir / "oos_predictions.csv", predictions),
    }

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputs", required=True, help="Comma-separated OHLCV CSV paths")
    parser.add_argument("--asset-names", required=True, help="Comma-separated asset names")
    parser.add_argument("--output-dir", default="return_seeking_threshold_validation_output")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")

    parser.add_argument("--model", default="extratrees", choices=["extratrees", "randomforest", "hgb", "logistic"])
    parser.add_argument("--thresholds", default="0.90,0.80,0.70,0.60")

    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--vol-window", type=int, default=60)

    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=0)

    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--max-features", default="sqrt")

    parser.add_argument("--hgb-iter", type=int, default=180)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=0.05)

    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    outputs = run(args)
    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))

    print("[OK] Return-seeking percentile threshold validation completed.")
    print(json.dumps({
        "asset_count": summary["asset_count"],
        "model": summary["model"],
        "prediction_rows": summary["prediction_rows"],
        "thresholds": summary["thresholds"],
        "best_threshold": summary["best_threshold"],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
