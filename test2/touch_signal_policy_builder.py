# -*- coding: utf-8 -*-
"""
touch_signal_policy_builder.py

6단계: Signal Policy Builder

목적
----
수익추구형 / 안정형 / 방어형 모델 출력을 하나의 해석 가능한 신호 체계로 통합합니다.
이 단계도 allocation, portfolio 수익률, 매수/매도 성과는 평가하지 않습니다.

통합하는 head
-------------
1. Return-Seeking Head
   - label: y_up_touch_fixed_h10_k1.0
   - model: ExtraTrees
   - score: return_score_percentile
   - signal:
     - >= 0.90: STRONG_UP
     - >= 0.80: STANDARD_UP
     - >= 0.70: WEAK_UP_WATCH

2. Balanced Head
   - label: y_up_touch_target_h10_rate30 + y_down_touch_target_h10_rate30
   - model: HGB
   - score: balanced_up_score_percentile / balanced_down_score_percentile
   - regime:
     - BALANCED_UP_EDGE
     - BALANCED_DOWN_RISK
     - BALANCED_CONFLICT_BOTH_HIGH
     - BALANCED_NEUTRAL

3. Defensive Head
   - label: y_down_touch_fixed_h10_k1.0
   - model: HGB
   - score: defensive_down_score_percentile
   - signal:
     - >= 0.90: DEFENSIVE_RISK_HIGH
     - >= 0.80: DEFENSIVE_RISK_WATCH

최종 정책 신호
--------------
- STRONG_RETURN_SEEKING
- STANDARD_RETURN_SEEKING
- WEAK_RETURN_WATCH
- DEFENSIVE_RISK
- CONFLICT_UP_AND_DOWN
- NEUTRAL_NO_EDGE

출력 파일
---------
output_dir/
├─ signal_policy_summary.json
├─ signal_policy_config.json
├─ oos_signal_predictions.csv
├─ policy_summary.csv
├─ asset_policy_summary.csv
├─ fold_policy_summary.csv
├─ annual_policy_summary.csv
├─ regime_crosstab.csv
├─ score_distribution.csv
└─ head_metric_summary.csv

실행 예시
---------
python touch_signal_policy_builder.py ^
  --inputs "QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv" ^
  --asset-names "QQQ,SPY,SOXX,XLK" ^
  --output-dir "touch_signal_policy_all"

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

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

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


def make_touch_labels_with_k(df: pd.DataFrame, horizon: int, vol_window: int, k_up: float, k_down: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    out["current_horizon_vol"] = current_horizon_volatility(close, horizon, vol_window)
    out["upper_barrier"] = close * (1.0 + k_up * out["current_horizon_vol"])
    out["lower_barrier"] = close * (1.0 - k_down * out["current_horizon_vol"])

    fh, fl = explicit_future_high_low(high, low, horizon)
    out["future_high_h"] = fh
    out["future_low_h"] = fl

    out["future_max_high_return"] = out["future_high_h"] / close.replace(0, np.nan) - 1.0
    out["future_min_low_return"] = out["future_low_h"] / close.replace(0, np.nan) - 1.0

    out["y_up_touch"] = (out["future_high_h"] >= out["upper_barrier"]).astype(float)
    out["y_down_touch"] = (out["future_low_h"] <= out["lower_barrier"]).astype(float)

    invalid = (
        out["current_horizon_vol"].isna()
        | out["upper_barrier"].isna()
        | out["lower_barrier"].isna()
        | out["future_high_h"].isna()
        | out["future_low_h"].isna()
    )
    out.loc[invalid, ["y_up_touch", "y_down_touch"]] = np.nan
    return out


def touch_rate_for_k_on_indices(df: pd.DataFrame, indices: np.ndarray, horizon: int, vol_window: int, side: str, k: float) -> float:
    if side == "up":
        labels = make_touch_labels_with_k(df, horizon, vol_window, k_up=k, k_down=999.0)
        y = labels["y_up_touch"].iloc[indices].dropna()
    elif side == "down":
        labels = make_touch_labels_with_k(df, horizon, vol_window, k_up=999.0, k_down=k)
        y = labels["y_down_touch"].iloc[indices].dropna()
    else:
        raise ValueError(side)

    if len(y) == 0:
        return np.nan
    return float(y.mean())


def find_k_for_target_rate(df: pd.DataFrame, indices: np.ndarray, horizon: int, vol_window: int, side: str, target_rate: float, k_min: float, k_max: float, grid_size: int) -> Dict:
    """
    Vectorized target-rate k search.
    Avoids regenerating the whole label frame for every k.
    """
    ks = np.linspace(k_min, k_max, grid_size)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    current_vol = current_horizon_volatility(close, horizon, vol_window)
    future_high, future_low = explicit_future_high_low(high, low, horizon)

    idx = np.asarray(indices, dtype=int)
    valid_base = (
        idx[(idx >= 0) & (idx < len(df))]
    )

    if len(valid_base) == 0:
        return {"best_k": np.nan, "achieved_positive_rate": np.nan, "abs_error": np.nan}

    c = close.iloc[valid_base].to_numpy(dtype=float)
    v = current_vol.iloc[valid_base].to_numpy(dtype=float)

    if side == "up":
        f = future_high.iloc[valid_base].to_numpy(dtype=float)
    elif side == "down":
        f = future_low.iloc[valid_base].to_numpy(dtype=float)
    else:
        raise ValueError(side)

    valid = np.isfinite(c) & np.isfinite(v) & np.isfinite(f)
    c = c[valid]
    v = v[valid]
    f = f[valid]

    if len(c) == 0:
        return {"best_k": np.nan, "achieved_positive_rate": np.nan, "abs_error": np.nan}

    best_k = np.nan
    best_rate = np.nan
    best_err = np.inf

    for k in ks:
        if side == "up":
            barrier = c * (1.0 + float(k) * v)
            y = f >= barrier
        else:
            barrier = c * (1.0 - float(k) * v)
            y = f <= barrier

        rate = float(np.mean(y))
        err = abs(rate - target_rate)
        if err < best_err:
            best_k = float(k)
            best_rate = rate
            best_err = err

    return {
        "best_k": best_k,
        "achieved_positive_rate": best_rate,
        "abs_error": float(best_err) if np.isfinite(best_err) else np.nan,
    }


# ============================================================
# 3. Split / Model
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
        "k_source": np.arange(fold.train_start, train_end_safe),
    }


def make_extratrees(args, random_state: int) -> Pipeline:
    clf = ExtraTreesClassifier(
        n_estimators=args.return_n_estimators,
        max_depth=args.return_max_depth,
        min_samples_leaf=args.return_min_samples_leaf,
        max_features=args.max_features,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", clf),
    ])


def make_hgb(args, random_state: int) -> Pipeline:
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


def predict_positive_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def percentile_from_calibration(raw_cal: np.ndarray, raw_test: np.ndarray) -> np.ndarray:
    cal = pd.Series(raw_cal).dropna().astype(float)
    if len(cal) == 0:
        return np.full(len(raw_test), np.nan)
    sorted_cal = np.sort(cal.to_numpy())
    return np.searchsorted(sorted_cal, raw_test, side="right") / len(sorted_cal)


def fit_predict_head(
    model: Pipeline,
    data: pd.DataFrame,
    feature_cols: List[str],
    y: pd.Series,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    min_train_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    y_train = y.iloc[train_idx]
    y_cal = y.iloc[cal_idx]
    y_test = y.iloc[test_idx]

    train_mask = y_train.notna()

    info = {
        "train_rows": int(train_mask.sum()),
        "cal_rows": int(y_cal.notna().sum()),
        "test_rows": int(y_test.notna().sum()),
        "train_positive_rate": float(y_train[train_mask].mean()) if train_mask.sum() else np.nan,
        "usable": False,
    }

    if train_mask.sum() < min_train_rows or y_train[train_mask].nunique() < 2:
        n = len(test_idx)
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), info

    model.fit(data.iloc[train_idx].loc[train_mask, feature_cols], y_train[train_mask].astype(int))

    raw_cal = predict_positive_proba(model, data.iloc[cal_idx][feature_cols])
    raw_test = predict_positive_proba(model, data.iloc[test_idx][feature_cols])
    score_percentile = percentile_from_calibration(raw_cal, raw_test)

    info["usable"] = True
    return raw_test, raw_cal, score_percentile, info


# ============================================================
# 4. Policy Rules
# ============================================================

def return_signal(score: float, strong: float, standard: float, weak: float) -> str:
    if pd.isna(score):
        return "RETURN_SIGNAL_MISSING"
    if score >= strong:
        return "STRONG_UP"
    if score >= standard:
        return "STANDARD_UP"
    if score >= weak:
        return "WEAK_UP_WATCH"
    return "NO_RETURN_EDGE"


def defensive_signal(score: float, high: float, watch: float) -> str:
    if pd.isna(score):
        return "DEFENSIVE_SIGNAL_MISSING"
    if score >= high:
        return "DEFENSIVE_RISK_HIGH"
    if score >= watch:
        return "DEFENSIVE_RISK_WATCH"
    return "NO_DEFENSIVE_RISK"


def balanced_regime(up_score: float, down_score: float, edge: float, watch: float) -> str:
    if pd.isna(up_score) or pd.isna(down_score):
        return "BALANCED_SIGNAL_MISSING"

    up_high = up_score >= edge
    down_high = down_score >= edge
    up_watch = up_score >= watch
    down_watch = down_score >= watch

    if up_high and down_high:
        return "BALANCED_CONFLICT_BOTH_HIGH"
    if up_high and not down_watch:
        return "BALANCED_UP_EDGE"
    if down_high and not up_watch:
        return "BALANCED_DOWN_RISK"
    if up_watch and down_watch:
        return "BALANCED_MIXED_WATCH"
    if up_watch:
        return "BALANCED_MILD_UP"
    if down_watch:
        return "BALANCED_MILD_DOWN"
    return "BALANCED_NEUTRAL"


def final_policy_signal(row: pd.Series) -> str:
    rs = row["return_signal"]
    br = row["balanced_regime"]
    ds = row["defensive_signal"]

    # Conflict/risk overrides.
    if ds == "DEFENSIVE_RISK_HIGH" and rs in {"STRONG_UP", "STANDARD_UP"}:
        return "CONFLICT_UP_AND_DEFENSIVE_RISK"
    if br == "BALANCED_CONFLICT_BOTH_HIGH":
        return "CONFLICT_BALANCED_BOTH_HIGH"
    if br == "BALANCED_DOWN_RISK" or ds == "DEFENSIVE_RISK_HIGH":
        return "DEFENSIVE_RISK"

    # Return-seeking signals.
    if rs == "STRONG_UP" and br in {"BALANCED_UP_EDGE", "BALANCED_MILD_UP", "BALANCED_NEUTRAL"}:
        return "STRONG_RETURN_SEEKING"
    if rs == "STANDARD_UP" and br in {"BALANCED_UP_EDGE", "BALANCED_MILD_UP", "BALANCED_NEUTRAL", "BALANCED_MIXED_WATCH"}:
        return "STANDARD_RETURN_SEEKING"
    if rs == "WEAK_UP_WATCH":
        return "WEAK_RETURN_WATCH"

    # Mild risk/watch states.
    if br in {"BALANCED_MILD_DOWN"} or ds == "DEFENSIVE_RISK_WATCH":
        return "RISK_WATCH"

    return "NEUTRAL_NO_EDGE"


# ============================================================
# 5. Metrics
# ============================================================

def binary_metrics(y_true: pd.Series, prob: pd.Series) -> Dict:
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
    else:
        out["pr_auc"] = base
        out["pr_ratio"] = 1.0
        out["pr_lift"] = 0.0

    return out


def summarize_by_group(preds: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    if preds.empty:
        return pd.DataFrame()

    global_up = float(preds["y_up_fixed"].dropna().mean())
    global_down = float(preds["y_down_fixed"].dropna().mean())

    for key, g in preds.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(zip(group_cols, key))
        up = g["y_up_fixed"].dropna()
        down = g["y_down_fixed"].dropna()

        row.update({
            "rows": int(len(g)),
            "row_rate": np.nan,  # filled by caller if needed
            "up_touch_rate": float(up.mean()) if len(up) else np.nan,
            "down_touch_rate": float(down.mean()) if len(down) else np.nan,
            "up_lift_vs_global": safe_divide(float(up.mean()), global_up) if len(up) else np.nan,
            "down_lift_vs_global": safe_divide(float(down.mean()), global_down) if len(down) else np.nan,
            "future_max_high_return_mean": float(g["future_max_high_return"].mean()),
            "future_max_high_return_median": float(g["future_max_high_return"].median()),
            "future_min_low_return_mean": float(g["future_min_low_return"].mean()),
            "future_min_low_return_median": float(g["future_min_low_return"].median()),
            "return_score_mean": float(g["return_score_percentile"].mean()),
            "balanced_up_score_mean": float(g["balanced_up_score_percentile"].mean()),
            "balanced_down_score_mean": float(g["balanced_down_score_percentile"].mean()),
            "defensive_down_score_mean": float(g["defensive_down_score_percentile"].mean()),
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        total = len(preds)
        out["row_rate"] = out["rows"] / total
    return out


def score_distribution(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset_name, g in preds.groupby("asset_name"):
        for col in [
            "return_score_percentile",
            "balanced_up_score_percentile",
            "balanced_down_score_percentile",
            "defensive_down_score_percentile",
        ]:
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
                "p10": float(s.quantile(0.10)),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "p90": float(s.quantile(0.90)),
                "max": float(s.max()),
            })
    return pd.DataFrame(rows)


# ============================================================
# 6. Asset Runner
# ============================================================

def run_asset(asset_name: str, input_path: str, args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_ohlcv(input_path)
    if args.start_date:
        raw = raw[raw["date"] >= pd.to_datetime(args.start_date)].copy()
    if args.end_date:
        raw = raw[raw["date"] <= pd.to_datetime(args.end_date)].copy()
    raw = raw.reset_index(drop=True)

    data, feature_cols = build_features(raw)

    fixed_labels = make_touch_labels_with_k(data, args.horizon, args.vol_window, k_up=args.fixed_k, k_down=args.fixed_k)
    y_up_fixed = fixed_labels["y_up_touch"]
    y_down_fixed = fixed_labels["y_down_touch"]

    folds = make_rolling_folds(
        n=len(data),
        train_window=args.train_window,
        calibration_window=args.calibration_window,
        test_window=args.test_window,
        embargo=max(args.embargo, args.horizon),
        max_folds=args.max_folds,
    )

    pred_parts = []
    head_metric_rows = []
    k_rows = []

    for fold in folds:
        idx = fold_indices(fold, args.horizon, len(data))
        train_idx, cal_idx, test_idx, k_source_idx = idx["train"], idx["cal"], idx["test"], idx["k_source"]

        # Target-rate k for balanced labels, fold-internal only.
        k_up_info = find_k_for_target_rate(
            data, k_source_idx, args.horizon, args.vol_window, "up",
            args.target_rate, args.k_min, args.k_max, args.k_grid_size,
        )
        k_down_info = find_k_for_target_rate(
            data, k_source_idx, args.horizon, args.vol_window, "down",
            args.target_rate, args.k_min, args.k_max, args.k_grid_size,
        )
        k_up_target = k_up_info["best_k"]
        k_down_target = k_down_info["best_k"]

        target_labels = make_touch_labels_with_k(data, args.horizon, args.vol_window, k_up=k_up_target, k_down=k_down_target)

        k_rows.append({
            "asset_name": asset_name,
            "fold_id": fold.fold_id,
            "target_rate": args.target_rate,
            "k_up_target": k_up_target,
            "k_down_target": k_down_target,
            "k_up_source_rate": k_up_info["achieved_positive_rate"],
            "k_down_source_rate": k_down_info["achieved_positive_rate"],
            "test_start_date": data["date"].iloc[fold.test_start],
            "test_end_date": data["date"].iloc[fold.test_end - 1],
        })

        # Head 1: Return-seeking up fixed, ExtraTrees.
        ret_model = make_extratrees(args, random_state=args.random_state + fold.fold_id)
        ret_raw, _, ret_score, ret_info = fit_predict_head(
            ret_model, data, feature_cols, y_up_fixed,
            train_idx, cal_idx, test_idx, args.min_train_rows,
        )

        # Head 2: Balanced up target, HGB.
        bal_up_model = make_hgb(args, random_state=args.random_state + 1000 + fold.fold_id)
        bal_up_raw, _, bal_up_score, bal_up_info = fit_predict_head(
            bal_up_model, data, feature_cols, target_labels["y_up_touch"],
            train_idx, cal_idx, test_idx, args.min_train_rows,
        )

        # Head 3: Balanced down target, HGB.
        bal_dn_model = make_hgb(args, random_state=args.random_state + 2000 + fold.fold_id)
        bal_dn_raw, _, bal_dn_score, bal_dn_info = fit_predict_head(
            bal_dn_model, data, feature_cols, target_labels["y_down_touch"],
            train_idx, cal_idx, test_idx, args.min_train_rows,
        )

        # Head 4: Defensive down fixed, HGB.
        def_model = make_hgb(args, random_state=args.random_state + 3000 + fold.fold_id)
        def_raw, _, def_score, def_info = fit_predict_head(
            def_model, data, feature_cols, y_down_fixed,
            train_idx, cal_idx, test_idx, args.min_train_rows,
        )

        # Head metrics on fixed labels for consistent policy diagnostics.
        head_metric_rows.append({
            "asset_name": asset_name,
            "fold_id": fold.fold_id,
            "head": "return_up_fixed_extratrees",
            **binary_metrics(y_up_fixed.iloc[test_idx].reset_index(drop=True), pd.Series(ret_raw)),
            **{f"fit_{k}": v for k, v in ret_info.items()},
        })
        head_metric_rows.append({
            "asset_name": asset_name,
            "fold_id": fold.fold_id,
            "head": "balanced_up_target_hgb",
            **binary_metrics(target_labels["y_up_touch"].iloc[test_idx].reset_index(drop=True), pd.Series(bal_up_raw)),
            **{f"fit_{k}": v for k, v in bal_up_info.items()},
        })
        head_metric_rows.append({
            "asset_name": asset_name,
            "fold_id": fold.fold_id,
            "head": "balanced_down_target_hgb",
            **binary_metrics(target_labels["y_down_touch"].iloc[test_idx].reset_index(drop=True), pd.Series(bal_dn_raw)),
            **{f"fit_{k}": v for k, v in bal_dn_info.items()},
        })
        head_metric_rows.append({
            "asset_name": asset_name,
            "fold_id": fold.fold_id,
            "head": "defensive_down_fixed_hgb",
            **binary_metrics(y_down_fixed.iloc[test_idx].reset_index(drop=True), pd.Series(def_raw)),
            **{f"fit_{k}": v for k, v in def_info.items()},
        })

        pred = pd.DataFrame({
            "asset_name": asset_name,
            "date": data["date"].iloc[test_idx].to_numpy(),
            "fold_id": fold.fold_id,
            "y_up_fixed": y_up_fixed.iloc[test_idx].to_numpy(),
            "y_down_fixed": y_down_fixed.iloc[test_idx].to_numpy(),
            "y_up_target": target_labels["y_up_touch"].iloc[test_idx].to_numpy(),
            "y_down_target": target_labels["y_down_touch"].iloc[test_idx].to_numpy(),
            "future_max_high_return": fixed_labels["future_max_high_return"].iloc[test_idx].to_numpy(),
            "future_min_low_return": fixed_labels["future_min_low_return"].iloc[test_idx].to_numpy(),
            "return_raw_prob": ret_raw,
            "return_score_percentile": ret_score,
            "balanced_up_raw_prob": bal_up_raw,
            "balanced_up_score_percentile": bal_up_score,
            "balanced_down_raw_prob": bal_dn_raw,
            "balanced_down_score_percentile": bal_dn_score,
            "defensive_down_raw_prob": def_raw,
            "defensive_down_score_percentile": def_score,
            "k_up_target": k_up_target,
            "k_down_target": k_down_target,
        })

        pred_parts.append(pred)

    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    head_metrics = pd.DataFrame(head_metric_rows)
    k_df = pd.DataFrame(k_rows)

    return predictions, head_metrics, k_df


# ============================================================
# 7. Main
# ============================================================

def apply_policy(preds: pd.DataFrame, args) -> pd.DataFrame:
    out = preds.copy()

    out["return_signal"] = out["return_score_percentile"].apply(
        lambda x: return_signal(x, args.return_strong_threshold, args.return_standard_threshold, args.return_weak_threshold)
    )
    out["defensive_signal"] = out["defensive_down_score_percentile"].apply(
        lambda x: defensive_signal(x, args.defensive_high_threshold, args.defensive_watch_threshold)
    )
    out["balanced_regime"] = [
        balanced_regime(u, d, args.balanced_edge_threshold, args.balanced_watch_threshold)
        for u, d in zip(out["balanced_up_score_percentile"], out["balanced_down_score_percentile"])
    ]
    out["final_policy_signal"] = out.apply(final_policy_signal, axis=1)

    return out


def aggregate_head_metrics(head_metrics: pd.DataFrame) -> pd.DataFrame:
    if head_metrics.empty:
        return pd.DataFrame()

    rows = []
    for head, g in head_metrics.groupby("head"):
        u = g[g["fit_usable"] == True].copy() if "fit_usable" in g.columns else g.copy()
        rows.append({
            "head": head,
            "fold_count": int(g["fold_id"].nunique()),
            "usable_fold_count": int(u["fold_id"].nunique()) if not u.empty else 0,
            "mean_positive_rate": float(u["positive_rate"].mean()) if not u.empty else np.nan,
            "mean_pr_auc": float(u["pr_auc"].mean()) if not u.empty else np.nan,
            "median_pr_auc": float(u["pr_auc"].median()) if not u.empty else np.nan,
            "mean_pr_ratio": float(u["pr_ratio"].mean()) if not u.empty else np.nan,
            "median_pr_ratio": float(u["pr_ratio"].median()) if not u.empty else np.nan,
            "positive_pr_lift_rate": float((u["pr_lift"] > 0).mean()) if not u.empty else np.nan,
            "mean_roc_auc": float(u["roc_auc"].mean()) if not u.empty else np.nan,
            "mean_brier_skill": float(u["brier_skill"].mean()) if not u.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values("median_pr_ratio", ascending=False)


def crosstab_regimes(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()

    ct = pd.crosstab(
        [preds["return_signal"], preds["balanced_regime"], preds["defensive_signal"]],
        preds["final_policy_signal"],
        dropna=False,
    ).reset_index()
    return ct


def run(args) -> Dict[str, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = parse_list(args.inputs)
    asset_names = parse_list(args.asset_names)

    if len(inputs) != len(asset_names):
        raise ValueError(f"inputs count != asset_names count: {len(inputs)} vs {len(asset_names)}")

    all_preds = []
    all_head_metrics = []
    all_k = []
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

        preds, head_metrics, k_df = run_asset(asset_name, input_path, args)
        all_preds.append(preds)
        all_head_metrics.append(head_metrics)
        all_k.append(k_df)

    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    head_metrics = pd.concat(all_head_metrics, ignore_index=True) if all_head_metrics else pd.DataFrame()
    target_k_by_fold = pd.concat(all_k, ignore_index=True) if all_k else pd.DataFrame()

    predictions = apply_policy(predictions, args)
    predictions["year"] = pd.to_datetime(predictions["date"]).dt.year

    policy_summary = summarize_by_group(predictions, ["final_policy_signal"])
    asset_policy_summary = summarize_by_group(predictions, ["asset_name", "final_policy_signal"])
    fold_policy_summary = summarize_by_group(predictions, ["asset_name", "fold_id", "final_policy_signal"])
    annual_policy_summary = summarize_by_group(predictions, ["asset_name", "year", "final_policy_signal"])
    regime_ct = crosstab_regimes(predictions)
    score_dist = score_distribution(predictions)
    head_metric_summary = aggregate_head_metrics(head_metrics)

    # Basic decision diagnostics for the two key return-seeking signals.
    key_signals = ["STRONG_RETURN_SEEKING", "STANDARD_RETURN_SEEKING", "CONFLICT_UP_AND_DEFENSIVE_RISK", "DEFENSIVE_RISK"]
    key_policy = policy_summary[policy_summary["final_policy_signal"].isin(key_signals)].copy() if not policy_summary.empty else pd.DataFrame()

    config = {
        "experiment": "touch_signal_policy_builder",
        "asset_periods": asset_periods,
        "policy_thresholds": {
            "return_strong_threshold": args.return_strong_threshold,
            "return_standard_threshold": args.return_standard_threshold,
            "return_weak_threshold": args.return_weak_threshold,
            "balanced_edge_threshold": args.balanced_edge_threshold,
            "balanced_watch_threshold": args.balanced_watch_threshold,
            "defensive_high_threshold": args.defensive_high_threshold,
            "defensive_watch_threshold": args.defensive_watch_threshold,
        },
        "heads": {
            "return_seeking": "fixed_h10_k1.0 + ExtraTrees + Up-touch",
            "balanced": "target_h10_rate30 + HGB + Up/Down-touch",
            "defensive": "fixed_h10_k1.0 + HGB + Down-touch",
        },
        "usage_note": {
            "portfolio_allocation": False,
            "literal_probability": False,
            "recommended_interpretation": "Use final_policy_signal as signal taxonomy only. Portfolio validation must be separate.",
        },
    }

    summary = {
        "experiment": "touch_signal_policy_builder",
        "asset_count": len(asset_names),
        "asset_periods": asset_periods,
        "prediction_rows": int(len(predictions)),
        "head_metric_rows": int(len(head_metrics)),
        "policy_signal_counts": predictions["final_policy_signal"].value_counts(dropna=False).to_dict() if not predictions.empty else {},
        "key_policy_summary": key_policy.to_dict("records") if not key_policy.empty else [],
        "decision_note": (
            "This step integrates head outputs into a signal taxonomy. "
            "It does not evaluate portfolio return, MDD, turnover, or transaction costs."
        ),
    }

    outputs = {
        "summary": save_json(out_dir / "signal_policy_summary.json", summary),
        "config": save_json(out_dir / "signal_policy_config.json", config),
        "oos_signal_predictions": save_csv(out_dir / "oos_signal_predictions.csv", predictions),
        "policy_summary": save_csv(out_dir / "policy_summary.csv", policy_summary),
        "asset_policy_summary": save_csv(out_dir / "asset_policy_summary.csv", asset_policy_summary),
        "fold_policy_summary": save_csv(out_dir / "fold_policy_summary.csv", fold_policy_summary),
        "annual_policy_summary": save_csv(out_dir / "annual_policy_summary.csv", annual_policy_summary),
        "regime_crosstab": save_csv(out_dir / "regime_crosstab.csv", regime_ct),
        "score_distribution": save_csv(out_dir / "score_distribution.csv", score_dist),
        "head_metrics": save_csv(out_dir / "head_metrics.csv", head_metrics),
        "head_metric_summary": save_csv(out_dir / "head_metric_summary.csv", head_metric_summary),
        "target_k_by_fold": save_csv(out_dir / "target_k_by_fold.csv", target_k_by_fold),
    }

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputs", required=True, help="Comma-separated OHLCV CSV paths")
    parser.add_argument("--asset-names", required=True, help="Comma-separated asset names")
    parser.add_argument("--output-dir", default="touch_signal_policy_output")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")

    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--fixed-k", type=float, default=1.0)
    parser.add_argument("--target-rate", type=float, default=0.30)
    parser.add_argument("--k-min", type=float, default=0.25)
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--k-grid-size", type=int, default=80)

    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=0)

    parser.add_argument("--return-n-estimators", type=int, default=180)
    parser.add_argument("--return-max-depth", type=int, default=5)
    parser.add_argument("--return-min-samples-leaf", type=int, default=20)

    parser.add_argument("--hgb-iter", type=int, default=180)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2", type=float, default=0.05)
    parser.add_argument("--max-features", default="sqrt")

    parser.add_argument("--return-strong-threshold", type=float, default=0.90)
    parser.add_argument("--return-standard-threshold", type=float, default=0.80)
    parser.add_argument("--return-weak-threshold", type=float, default=0.70)

    parser.add_argument("--balanced-edge-threshold", type=float, default=0.80)
    parser.add_argument("--balanced-watch-threshold", type=float, default=0.60)

    parser.add_argument("--defensive-high-threshold", type=float, default=0.90)
    parser.add_argument("--defensive-watch-threshold", type=float, default=0.80)

    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    outputs = run(args)
    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))

    print("[OK] Touch signal policy builder completed.")
    print(json.dumps({
        "asset_count": summary["asset_count"],
        "prediction_rows": summary["prediction_rows"],
        "policy_signal_counts": summary["policy_signal_counts"],
        "key_policy_summary": summary["key_policy_summary"],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
