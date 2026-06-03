# -*- coding: utf-8 -*-
"""
multi_objective_touch_model_optimizer.py

3단계: 목적별 Touch Head 모델 성능 최적화 스크립트.

목적
----
Allocation / Portfolio 성과는 완전히 제외하고,
수익추구형 / 안정형 / 방어형 모델의 head-level 예측 성능만 평가합니다.

목적별 모델
-----------
1. return_seeking
   - Up-touch 예측 성능 최대화
   - y_up_touch_target_h10_rate30 중심

2. balanced
   - Up-touch + Down-touch를 둘 다 안정적으로 예측
   - y_up_touch / y_down_touch pair 중심

3. defensive
   - Down-touch 예측 성능 최대화
   - y_down_touch_target_h10_rate30 중심

라벨
----
메인:
- target_h10_rate30

비교 baseline:
- fixed_h10_k1.0

중요 누수 방지
--------------
target-rate k는 전체 데이터에서 계산하지 않습니다.
각 walk-forward fold의 train 구간에서만 k를 산출한 뒤,
동일 k를 train / calibration / test에 적용합니다.

평가 지표
---------
- PR-AUC
- PR lift = PR-AUC - positive_rate
- PR ratio = PR-AUC / positive_rate
- ROC-AUC
- Brier
- Brier Skill
- ECE
- Top-decile precision
- Fold stability

실행 예시 - 단일 자산
--------------------
python multi_objective_touch_model_optimizer.py ^
  --inputs "QQQ_ohlcv.csv" ^
  --asset-names "QQQ" ^
  --output-dir "touch_model_optimizer_QQQ"

실행 예시 - 4개 자산
--------------------
python multi_objective_touch_model_optimizer.py ^
  --inputs "QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv" ^
  --asset-names "QQQ,SPY,SOXX,XLK" ^
  --output-dir "touch_model_optimizer_all"

출력
----
output_dir/
├─ model_optimizer_summary.json
├─ best_model_config.json
├─ fold_metrics.csv
├─ head_model_summary.csv
├─ objective_summary.csv
├─ objective_top20.csv
├─ label_k_by_fold.csv
├─ label_distribution_by_fold.csv
├─ calibration_bins.csv
└─ oos_predictions.csv

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
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


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def safe_divide(a: float, b: float, default: float = np.nan) -> float:
    try:
        if b == 0 or pd.isna(b):
            return default
        return float(a / b)
    except Exception:
        return default


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


def rolling_min_periods(window: int, floor: int = 5, frac: float = 1 / 3) -> int:
    return max(1, min(int(window), max(int(floor), int(window * frac))))


# ============================================================
# 1. Feature Builder
# ============================================================

def rolling_mdd(close: pd.Series, window: int) -> pd.Series:
    roll_max = close.rolling(window, min_periods=rolling_min_periods(window, floor=10)).max()
    return close / roll_max.replace(0, np.nan) - 1.0


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy().sort_values("date").reset_index(drop=True)

    close = d["adj_close"].astype(float)
    raw_close = d["close"].astype(float)
    open_ = d["open"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    volume = d["volume"].astype(float)

    ret = close.pct_change()
    d["ret_1d"] = ret
    d["log_ret_1d"] = np.log(close / close.shift(1))

    # Return / trend / volatility
    for w in [2, 3, 5, 10, 20, 40, 60, 120, 252]:
        d[f"ret_{w}d"] = close.pct_change(w)
        d[f"log_ret_sum_{w}d"] = d["log_ret_1d"].rolling(w, min_periods=rolling_min_periods(w)).sum()
        d[f"vol_{w}d"] = ret.rolling(w, min_periods=rolling_min_periods(w)).std()
        d[f"down_vol_{w}d"] = ret.where(ret < 0, 0.0).rolling(w, min_periods=rolling_min_periods(w)).std()
        d[f"up_vol_{w}d"] = ret.where(ret > 0, 0.0).rolling(w, min_periods=rolling_min_periods(w)).std()
        d[f"mdd_{w}d"] = rolling_mdd(close, w)
        d[f"ret_z_{w}d"] = (
            ret.rolling(w, min_periods=rolling_min_periods(w)).mean()
            / ret.rolling(w, min_periods=rolling_min_periods(w)).std().replace(0, np.nan)
        )

    # Volatility term structure / ratios
    for a, b in [(5, 20), (10, 40), (20, 60), (20, 120), (60, 252)]:
        if f"vol_{a}d" in d.columns and f"vol_{b}d" in d.columns:
            d[f"vol_ratio_{a}_{b}"] = d[f"vol_{a}d"] / d[f"vol_{b}d"].replace(0, np.nan)
        if f"down_vol_{a}d" in d.columns and f"down_vol_{b}d" in d.columns:
            d[f"down_vol_ratio_{a}_{b}"] = d[f"down_vol_{a}d"] / d[f"down_vol_{b}d"].replace(0, np.nan)

    # Moving average gap and slopes
    for w in [5, 10, 20, 40, 60, 120, 200]:
        ma = close.rolling(w, min_periods=rolling_min_periods(w)).mean()
        d[f"ma_gap_{w}d"] = close / ma.replace(0, np.nan) - 1.0
        d[f"ma_slope_5d_{w}d"] = ma.pct_change(5)
        d[f"ma_slope_20d_{w}d"] = ma.pct_change(20)

    # Price location / breakout pressure
    for w in [10, 20, 40, 60, 120, 252]:
        rolling_high = high.rolling(w, min_periods=rolling_min_periods(w)).max()
        rolling_low = low.rolling(w, min_periods=rolling_min_periods(w)).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)

        d[f"dist_to_high_{w}d"] = close / rolling_high.replace(0, np.nan) - 1.0
        d[f"dist_to_low_{w}d"] = close / rolling_low.replace(0, np.nan) - 1.0
        d[f"range_position_{w}d"] = (close - rolling_low) / price_range
        d[f"breakout_pressure_{w}d"] = high / rolling_high.shift(1).replace(0, np.nan) - 1.0
        d[f"breakdown_pressure_{w}d"] = low / rolling_low.shift(1).replace(0, np.nan) - 1.0

    # Candle/range/ATR-like
    d["hl_range"] = (high - low) / raw_close.replace(0, np.nan)
    d["oc_ret"] = raw_close / open_.replace(0, np.nan) - 1.0
    d["gap_ret"] = open_ / raw_close.shift(1).replace(0, np.nan) - 1.0
    d["upper_shadow"] = (high - np.maximum(open_, raw_close)) / raw_close.replace(0, np.nan)
    d["lower_shadow"] = (np.minimum(open_, raw_close) - low) / raw_close.replace(0, np.nan)
    d["body_size"] = (raw_close - open_).abs() / raw_close.replace(0, np.nan)

    prev_close = raw_close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    for w in [5, 10, 20, 40]:
        d[f"atr_pct_{w}d"] = true_range.rolling(w, min_periods=rolling_min_periods(w)).mean() / raw_close.replace(0, np.nan)
        d[f"range_vol_{w}d"] = d["hl_range"].rolling(w, min_periods=rolling_min_periods(w)).mean()

    # Volume features
    if not volume.isna().all():
        vol_log = np.log1p(volume)
        d["volume_log"] = vol_log
        for w in [5, 10, 20, 60]:
            vol_ma = volume.rolling(w, min_periods=rolling_min_periods(w, floor=3)).mean()
            d[f"volume_ratio_{w}d"] = volume / vol_ma.replace(0, np.nan)
            d[f"volume_chg_{w}d"] = volume.pct_change(w)
            d[f"volume_z_{w}d"] = (
                (vol_log - vol_log.rolling(w, min_periods=rolling_min_periods(w, floor=3)).mean())
                / vol_log.rolling(w, min_periods=rolling_min_periods(w, floor=3)).std().replace(0, np.nan)
            )

    # Leakage guard
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
# 2. Touch Labels
# ============================================================

def explicit_future_high_low(
    high: pd.Series,
    low: pd.Series,
    horizon: int,
) -> Tuple[pd.Series, pd.Series]:
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


def make_touch_labels_with_k(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int,
    k_up: float,
    k_down: float,
) -> pd.DataFrame:
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

    out["y_up_touch"] = (out["future_high_h"] >= out["upper_barrier"]).astype(float)
    out["y_down_touch"] = (out["future_low_h"] <= out["lower_barrier"]).astype(float)

    invalid = (
        out["current_horizon_vol"].isna()
        | out["future_high_h"].isna()
        | out["future_low_h"].isna()
        | out["upper_barrier"].isna()
        | out["lower_barrier"].isna()
    )
    out.loc[invalid, ["y_up_touch", "y_down_touch"]] = np.nan
    return out


def touch_rate_for_k_on_indices(
    df: pd.DataFrame,
    indices: np.ndarray,
    horizon: int,
    vol_window: int,
    side: str,
    k: float,
) -> float:
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


def find_k_for_target_rate(
    df: pd.DataFrame,
    indices: np.ndarray,
    horizon: int,
    vol_window: int,
    side: str,
    target_rate: float,
    k_min: float,
    k_max: float,
    grid_size: int,
) -> Dict:
    ks = np.linspace(k_min, k_max, grid_size)
    rows = []
    for k in ks:
        rate = touch_rate_for_k_on_indices(df, indices, horizon, vol_window, side, float(k))
        rows.append({
            "k": float(k),
            "positive_rate": rate,
            "abs_error": abs(rate - target_rate) if not pd.isna(rate) else np.inf,
        })
    best = pd.DataFrame(rows).sort_values(["abs_error", "k"]).iloc[0].to_dict()
    return {
        "side": side,
        "target_rate": target_rate,
        "best_k": float(best["k"]),
        "achieved_positive_rate": float(best["positive_rate"]) if not pd.isna(best["positive_rate"]) else np.nan,
        "abs_error": float(best["abs_error"]) if np.isfinite(best["abs_error"]) else np.nan,
    }


@dataclass
class LabelSpec:
    label_scheme: str
    method: str
    horizon: int
    target_rate: Optional[float] = None
    fixed_k: Optional[float] = None


def parse_label_schemes(s: str) -> List[LabelSpec]:
    specs: List[LabelSpec] = []
    for item in parse_list(s):
        raw = item.strip().lower()
        if raw.startswith("target_h") and "_rate" in raw:
            # target_h10_rate30 or target_h10_rate0p30
            left, rate_part = raw.split("_rate", 1)
            horizon = int(left.replace("target_h", ""))
            rate_s = rate_part.replace("p", ".")
            rate = float(rate_s)
            if rate > 1:
                rate = rate / 100.0
            specs.append(LabelSpec(label_scheme=f"target_h{horizon}_rate{rate:.2f}".replace(".", "p"), method="target", horizon=horizon, target_rate=rate))
        elif raw.startswith("fixed_h") and "_k" in raw:
            # fixed_h10_k1p0
            left, k_part = raw.split("_k", 1)
            horizon = int(left.replace("fixed_h", ""))
            k = float(k_part.replace("p", "."))
            specs.append(LabelSpec(label_scheme=f"fixed_h{horizon}_k{k:.2f}".replace(".", "p"), method="fixed", horizon=horizon, fixed_k=k))
        else:
            raise ValueError(f"unknown label scheme: {item}")
    return specs


# ============================================================
# 3. Rolling Split
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


def fold_indices_for_label(fold: Fold, horizon: int, n: int, target_k_source: str) -> Dict[str, np.ndarray]:
    # Drop samples whose future label horizon crosses train/cal boundary for train and calibration.
    train_end_label_safe = max(fold.train_start, fold.train_end - horizon)
    cal_end_label_safe = max(fold.cal_start, fold.cal_end - horizon)

    train_idx = np.arange(fold.train_start, train_end_label_safe)
    cal_idx = np.arange(fold.cal_start, cal_end_label_safe)
    test_idx = np.arange(fold.test_start, min(fold.test_end, n))

    if target_k_source == "train":
        k_source_idx = train_idx
    elif target_k_source == "train_cal":
        k_source_idx = np.concatenate([train_idx, cal_idx])
    else:
        raise ValueError(f"unknown target_k_source: {target_k_source}")

    return {
        "train": train_idx,
        "cal": cal_idx,
        "test": test_idx,
        "k_source": k_source_idx,
    }


# ============================================================
# 4. Models / Calibration / Metrics
# ============================================================

def make_model(name: str, random_state: int, n_estimators: int, max_depth: int, min_samples_leaf: int) -> Pipeline:
    name = name.lower()

    if name == "logistic":
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs",
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ])
        return model

    if name == "extratrees":
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", model),
        ])

    if name == "randomforest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", model),
        ])

    if name == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=max(50, n_estimators),
            max_leaf_nodes=31,
            learning_rate=0.05,
            l2_regularization=0.05,
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", model),
        ])

    raise ValueError(f"unknown model: {name}")


def predict_positive_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Pipeline exposes predict_proba if final estimator does.
    return model.predict_proba(X)[:, 1]


def platt_calibrate(raw_cal: np.ndarray, y_cal: np.ndarray, raw_test: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Calibrate test probabilities using 1D logistic regression over raw probabilities.
    This avoids CalibratedClassifierCV prefit deprecation issues.
    """
    info = {"calibrated": False, "calibration_method": "none"}

    raw_cal = np.asarray(raw_cal, dtype=float)
    raw_test = np.asarray(raw_test, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)

    mask = np.isfinite(raw_cal) & np.isfinite(y_cal)
    if mask.sum() < 30 or len(np.unique(y_cal[mask])) < 2:
        return raw_test, info

    try:
        eps = 1e-6
        x_cal = np.log(np.clip(raw_cal[mask], eps, 1 - eps) / np.clip(1 - raw_cal[mask], eps, 1 - eps)).reshape(-1, 1)
        x_test = np.log(np.clip(raw_test, eps, 1 - eps) / np.clip(1 - raw_test, eps, 1 - eps)).reshape(-1, 1)

        calibrator = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
        calibrator.fit(x_cal, y_cal[mask])
        out = calibrator.predict_proba(x_test)[:, 1]
        info = {"calibrated": True, "calibration_method": "platt_logistic_on_raw_logit"}
        return out, info
    except Exception as e:
        info["calibration_error"] = str(e)
        return raw_test, info


def expected_calibration_error(y_true: pd.Series, prob: pd.Series, bins: int = 10) -> Tuple[float, pd.DataFrame]:
    y = pd.Series(y_true).astype(float)
    p = pd.Series(prob).astype(float)
    mask = y.notna() & p.notna() & np.isfinite(p)
    y = y[mask].astype(int)
    p = p[mask].astype(float)

    if len(y) == 0:
        return np.nan, pd.DataFrame()

    bin_edges = np.linspace(0, 1, bins + 1)
    bin_ids = np.digitize(p, bin_edges[1:-1], right=True)

    rows = []
    ece = 0.0
    for b in range(bins):
        m = bin_ids == b
        if not np.any(m):
            rows.append({
                "bin": b,
                "count": 0,
                "avg_prob": np.nan,
                "actual_rate": np.nan,
                "abs_gap": np.nan,
            })
            continue
        avg_prob = float(p[m].mean())
        actual = float(y[m].mean())
        gap = abs(avg_prob - actual)
        weight = float(np.mean(m))
        ece += weight * gap
        rows.append({
            "bin": b,
            "count": int(np.sum(m)),
            "avg_prob": avg_prob,
            "actual_rate": actual,
            "abs_gap": gap,
        })

    return float(ece), pd.DataFrame(rows)


def binary_metrics(y_true: pd.Series, prob: pd.Series) -> Dict[str, float]:
    y = pd.Series(y_true).astype(float)
    p = pd.Series(prob).astype(float)
    mask = y.notna() & p.notna() & np.isfinite(p)
    y = y[mask].astype(int)
    p = p[mask].astype(float)

    out = {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "pr_lift": np.nan,
        "pr_ratio": np.nan,
        "brier": np.nan,
        "brier_skill": np.nan,
        "ece": np.nan,
        "best_f1": np.nan,
        "top_decile_precision": np.nan,
        "top_decile_lift": np.nan,
    }

    if len(y) == 0:
        return out

    pos_rate = out["positive_rate"]

    if y.nunique() == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        out["pr_lift"] = out["pr_auc"] - pos_rate
        out["pr_ratio"] = safe_divide(out["pr_auc"], pos_rate)

        brier = float(brier_score_loss(y, p))
        baseline_brier = float(brier_score_loss(y, np.full(len(y), pos_rate)))
        out["brier"] = brier
        out["brier_skill"] = 1.0 - safe_divide(brier, baseline_brier)

        ece, _ = expected_calibration_error(y, p, bins=10)
        out["ece"] = ece

        try:
            precision, recall, _ = precision_recall_curve(y, p)
            f1_vals = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
            out["best_f1"] = float(np.nanmax(f1_vals))
        except Exception:
            pass
    else:
        out["pr_auc"] = pos_rate
        out["pr_lift"] = 0.0
        out["pr_ratio"] = 1.0

    # Top-decile precision.
    n_top = max(1, int(math.ceil(0.10 * len(y))))
    order = np.argsort(-p.to_numpy())[:n_top]
    top_prec = float(y.iloc[order].mean())
    out["top_decile_precision"] = top_prec
    out["top_decile_lift"] = safe_divide(top_prec, pos_rate)

    return out


def fit_predict_one_head(
    data: pd.DataFrame,
    feature_cols: List[str],
    y: pd.Series,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    random_state: int,
    args,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    train = data.iloc[train_idx].copy()
    cal = data.iloc[cal_idx].copy()
    test = data.iloc[test_idx].copy()

    y_train = y.iloc[train_idx]
    y_cal = y.iloc[cal_idx]
    y_test = y.iloc[test_idx]

    train_mask = y_train.notna()
    cal_mask = y_cal.notna()

    info = {
        "train_rows": int(train_mask.sum()),
        "cal_rows": int(cal_mask.sum()),
        "test_rows": int(y_test.notna().sum()),
        "train_positive_rate": float(y_train[train_mask].mean()) if train_mask.sum() else np.nan,
        "cal_positive_rate": float(y_cal[cal_mask].mean()) if cal_mask.sum() else np.nan,
        "usable": False,
    }

    if train_mask.sum() < args.min_train_rows or y_train[train_mask].nunique() < 2:
        return np.full(len(test_idx), np.nan), np.full(len(cal_idx), np.nan), info

    model = make_model(
        model_name,
        random_state=random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )

    model.fit(train.loc[train_mask, feature_cols], y_train[train_mask].astype(int))

    raw_cal = predict_positive_proba(model, cal[feature_cols]) if len(cal_idx) else np.array([])
    raw_test = predict_positive_proba(model, test[feature_cols]) if len(test_idx) else np.array([])

    if args.probability_mode == "platt":
        valid_cal = y_cal.notna()
        test_prob, cal_info = platt_calibrate(raw_cal[valid_cal.to_numpy()], y_cal[valid_cal].astype(int).to_numpy(), raw_test)
    else:
        test_prob = raw_test
        cal_info = {"calibrated": False, "calibration_method": "raw"}

    info.update(cal_info)
    info["usable"] = True

    return test_prob, raw_cal, info


# ============================================================
# 5. Experiment Runner
# ============================================================

def objectives_for_side(side: str) -> List[str]:
    if side == "up":
        return ["return_seeking", "balanced"]
    if side == "down":
        return ["defensive", "balanced"]
    raise ValueError(side)


def run_asset(asset_name: str, input_path: str, args, label_specs: List[LabelSpec]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_ohlcv(input_path)
    if args.start_date:
        raw = raw[raw["date"] >= pd.to_datetime(args.start_date)].copy()
    if args.end_date:
        raw = raw[raw["date"] <= pd.to_datetime(args.end_date)].copy()
    raw = raw.reset_index(drop=True)

    data, feature_cols = build_features(raw)
    n = len(data)

    max_horizon = max(spec.horizon for spec in label_specs)
    embargo = max(args.embargo, max_horizon)

    folds = make_rolling_folds(
        n=n,
        train_window=args.train_window,
        calibration_window=args.calibration_window,
        test_window=args.test_window,
        embargo=embargo,
        max_folds=args.max_folds,
    )
    if not folds:
        raise RuntimeError(f"No folds generated for {asset_name}. rows={n}")

    models = parse_list(args.models)

    fold_metric_rows = []
    pred_rows = []
    k_rows = []
    label_dist_rows = []
    cal_bin_rows = []

    for spec in label_specs:
        for fold in folds:
            idx = fold_indices_for_label(fold, spec.horizon, n, args.target_k_source)
            train_idx = idx["train"]
            cal_idx = idx["cal"]
            test_idx = idx["test"]
            k_source_idx = idx["k_source"]

            if spec.method == "fixed":
                k_up = float(spec.fixed_k)
                k_down = float(spec.fixed_k)
                k_info_up = {"achieved_positive_rate": np.nan, "abs_error": np.nan}
                k_info_down = {"achieved_positive_rate": np.nan, "abs_error": np.nan}
            elif spec.method == "target":
                k_info_up = find_k_for_target_rate(
                    data, k_source_idx, spec.horizon, args.vol_window, "up", spec.target_rate,
                    args.k_min, args.k_max, args.k_grid_size,
                )
                k_info_down = find_k_for_target_rate(
                    data, k_source_idx, spec.horizon, args.vol_window, "down", spec.target_rate,
                    args.k_min, args.k_max, args.k_grid_size,
                )
                k_up = k_info_up["best_k"]
                k_down = k_info_down["best_k"]
            else:
                raise ValueError(spec.method)

            k_rows.append({
                "asset_name": asset_name,
                "fold_id": fold.fold_id,
                "label_scheme": spec.label_scheme,
                "method": spec.method,
                "horizon": spec.horizon,
                "target_rate": spec.target_rate,
                "k_up": k_up,
                "k_down": k_down,
                "k_source": args.target_k_source if spec.method == "target" else "fixed",
                "up_k_source_rate": k_info_up.get("achieved_positive_rate"),
                "down_k_source_rate": k_info_down.get("achieved_positive_rate"),
                "up_k_abs_error": k_info_up.get("abs_error"),
                "down_k_abs_error": k_info_down.get("abs_error"),
                "train_start_date": data["date"].iloc[fold.train_start],
                "train_end_date": data["date"].iloc[fold.train_end - 1],
                "cal_start_date": data["date"].iloc[fold.cal_start],
                "cal_end_date": data["date"].iloc[fold.cal_end - 1],
                "test_start_date": data["date"].iloc[fold.test_start],
                "test_end_date": data["date"].iloc[fold.test_end - 1],
            })

            labels = make_touch_labels_with_k(data, spec.horizon, args.vol_window, k_up, k_down)
            y_map = {
                "up": labels["y_up_touch"],
                "down": labels["y_down_touch"],
            }

            # Label distribution for train/cal/test.
            for split_name, split_idx in [("train", train_idx), ("cal", cal_idx), ("test", test_idx)]:
                up_y = y_map["up"].iloc[split_idx].dropna()
                down_y = y_map["down"].iloc[split_idx].dropna()
                both = ((y_map["up"].iloc[split_idx] == 1) & (y_map["down"].iloc[split_idx] == 1)).dropna()
                none = ((y_map["up"].iloc[split_idx] == 0) & (y_map["down"].iloc[split_idx] == 0)).dropna()
                label_dist_rows.append({
                    "asset_name": asset_name,
                    "fold_id": fold.fold_id,
                    "label_scheme": spec.label_scheme,
                    "split": split_name,
                    "rows": int(len(split_idx)),
                    "up_valid_rows": int(len(up_y)),
                    "down_valid_rows": int(len(down_y)),
                    "up_positive_rate": float(up_y.mean()) if len(up_y) else np.nan,
                    "down_positive_rate": float(down_y.mean()) if len(down_y) else np.nan,
                    "both_touch_rate": float(both.mean()) if len(both) else np.nan,
                    "no_touch_rate": float(none.mean()) if len(none) else np.nan,
                })

            # Model fits.
            for side in ["up", "down"]:
                y = y_map[side]
                for model_name in models:
                    test_prob, raw_cal, fit_info = fit_predict_one_head(
                        data=data,
                        feature_cols=feature_cols,
                        y=y,
                        train_idx=train_idx,
                        cal_idx=cal_idx,
                        test_idx=test_idx,
                        model_name=model_name,
                        random_state=args.random_state + fold.fold_id,
                        args=args,
                    )

                    y_test = y.iloc[test_idx].reset_index(drop=True)
                    metric = binary_metrics(y_test, pd.Series(test_prob))

                    row = {
                        "asset_name": asset_name,
                        "fold_id": fold.fold_id,
                        "label_scheme": spec.label_scheme,
                        "method": spec.method,
                        "horizon": spec.horizon,
                        "side": side,
                        "model": model_name,
                        "k_up": k_up,
                        "k_down": k_down,
                        **fit_info,
                        **metric,
                    }
                    fold_metric_rows.append(row)

                    # Calibration bins
                    ece, bins_df = expected_calibration_error(y_test, pd.Series(test_prob), bins=10)
                    if not bins_df.empty:
                        bins_df["asset_name"] = asset_name
                        bins_df["fold_id"] = fold.fold_id
                        bins_df["label_scheme"] = spec.label_scheme
                        bins_df["side"] = side
                        bins_df["model"] = model_name
                        cal_bin_rows.append(bins_df)

                    # OOS prediction rows - save only if requested/all or top models can be filtered later.
                    if args.save_predictions:
                        pred = pd.DataFrame({
                            "asset_name": asset_name,
                            "date": data["date"].iloc[test_idx].to_numpy(),
                            "fold_id": fold.fold_id,
                            "label_scheme": spec.label_scheme,
                            "side": side,
                            "model": model_name,
                            "y_true": y.iloc[test_idx].to_numpy(),
                            "prob": test_prob,
                        })
                        pred_rows.append(pred)

    fold_metrics = pd.DataFrame(fold_metric_rows)
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    k_df = pd.DataFrame(k_rows)
    label_dist = pd.DataFrame(label_dist_rows)
    cal_bins = pd.concat(cal_bin_rows, ignore_index=True) if cal_bin_rows else pd.DataFrame()

    return fold_metrics, preds, k_df, label_dist, cal_bins


def summarize_head_models(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()

    group_cols = ["asset_name", "label_scheme", "method", "horizon", "side", "model"]
    rows = []

    for key, g in fold_metrics.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, key))
        usable = g[g["usable"] == True].copy()

        if usable.empty:
            d.update({"fold_count": int(g["fold_id"].nunique()), "usable_fold_count": 0, "head_model_score": -999.0})
            rows.append(d)
            continue

        d.update({
            "fold_count": int(g["fold_id"].nunique()),
            "usable_fold_count": int(usable["fold_id"].nunique()),
            "mean_positive_rate": float(usable["positive_rate"].mean()),
            "median_positive_rate": float(usable["positive_rate"].median()),
            "mean_pr_auc": float(usable["pr_auc"].mean()),
            "median_pr_auc": float(usable["pr_auc"].median()),
            "mean_pr_lift": float(usable["pr_lift"].mean()),
            "median_pr_lift": float(usable["pr_lift"].median()),
            "positive_pr_lift_rate": float((usable["pr_lift"] > 0).mean()),
            "mean_pr_ratio": float(usable["pr_ratio"].mean()),
            "median_pr_ratio": float(usable["pr_ratio"].median()),
            "mean_roc_auc": float(usable["roc_auc"].mean()),
            "median_roc_auc": float(usable["roc_auc"].median()),
            "mean_brier_skill": float(usable["brier_skill"].mean()),
            "median_brier_skill": float(usable["brier_skill"].median()),
            "positive_brier_skill_rate": float((usable["brier_skill"] > 0).mean()),
            "mean_ece": float(usable["ece"].mean()),
            "median_ece": float(usable["ece"].median()),
            "mean_top_decile_precision": float(usable["top_decile_precision"].mean()),
            "median_top_decile_precision": float(usable["top_decile_precision"].median()),
            "mean_top_decile_lift": float(usable["top_decile_lift"].mean()),
            "median_top_decile_lift": float(usable["top_decile_lift"].median()),
        })

        # Model score: ranking first, calibration second.
        d["head_model_score"] = (
            0.35 * d["median_pr_ratio"]
            + 0.25 * d["positive_pr_lift_rate"]
            + 0.20 * d["median_top_decile_lift"]
            + 0.10 * max(d["median_brier_skill"], -1.0)
            - 0.10 * d["median_ece"]
        )
        rows.append(d)

    return pd.DataFrame(rows).sort_values("head_model_score", ascending=False)


def summarize_objectives(head_summary: pd.DataFrame) -> pd.DataFrame:
    if head_summary.empty:
        return pd.DataFrame()

    rows = []

    # Return-seeking: up side only.
    up = head_summary[head_summary["side"] == "up"].copy()
    for _, r in up.iterrows():
        rows.append({
            "objective": "return_seeking",
            "asset_name": r["asset_name"],
            "label_scheme": r["label_scheme"],
            "method": r["method"],
            "horizon": r["horizon"],
            "model": r["model"],
            "side_or_pair": "up",
            "primary_score": r["head_model_score"],
            "mean_pr_auc": r["mean_pr_auc"],
            "median_pr_auc": r["median_pr_auc"],
            "mean_pr_ratio": r["mean_pr_ratio"],
            "median_pr_ratio": r["median_pr_ratio"],
            "positive_pr_lift_rate": r["positive_pr_lift_rate"],
            "median_top_decile_lift": r["median_top_decile_lift"],
            "median_brier_skill": r["median_brier_skill"],
            "median_ece": r["median_ece"],
            "objective_score": (
                0.45 * r["head_model_score"]
                + 0.30 * r["median_pr_ratio"]
                + 0.25 * r["median_top_decile_lift"]
            ),
        })

    # Defensive: down side only.
    down = head_summary[head_summary["side"] == "down"].copy()
    for _, r in down.iterrows():
        rows.append({
            "objective": "defensive",
            "asset_name": r["asset_name"],
            "label_scheme": r["label_scheme"],
            "method": r["method"],
            "horizon": r["horizon"],
            "model": r["model"],
            "side_or_pair": "down",
            "primary_score": r["head_model_score"],
            "mean_pr_auc": r["mean_pr_auc"],
            "median_pr_auc": r["median_pr_auc"],
            "mean_pr_ratio": r["mean_pr_ratio"],
            "median_pr_ratio": r["median_pr_ratio"],
            "positive_pr_lift_rate": r["positive_pr_lift_rate"],
            "median_top_decile_lift": r["median_top_decile_lift"],
            "median_brier_skill": r["median_brier_skill"],
            "median_ece": r["median_ece"],
            "objective_score": (
                0.45 * r["head_model_score"]
                + 0.35 * r["median_pr_ratio"]
                + 0.20 * r["positive_pr_lift_rate"]
            ),
        })

    # Balanced: pair up/down for same asset/label/model.
    pair_keys = ["asset_name", "label_scheme", "method", "horizon", "model"]
    for key, g in head_summary.groupby(pair_keys, dropna=False):
        sides = set(g["side"])
        if not {"up", "down"}.issubset(sides):
            continue

        up_r = g[g["side"] == "up"].iloc[0]
        dn_r = g[g["side"] == "down"].iloc[0]

        mean_pr_ratio = float(np.nanmean([up_r["median_pr_ratio"], dn_r["median_pr_ratio"]]))
        worst_pr_ratio = float(np.nanmin([up_r["median_pr_ratio"], dn_r["median_pr_ratio"]]))
        mean_score = float(np.nanmean([up_r["head_model_score"], dn_r["head_model_score"]]))
        worst_score = float(np.nanmin([up_r["head_model_score"], dn_r["head_model_score"]]))
        mean_lift_rate = float(np.nanmean([up_r["positive_pr_lift_rate"], dn_r["positive_pr_lift_rate"]]))

        rows.append({
            "objective": "balanced",
            "asset_name": key[0],
            "label_scheme": key[1],
            "method": key[2],
            "horizon": key[3],
            "model": key[4],
            "side_or_pair": "up_down_pair",
            "primary_score": mean_score,
            "mean_pr_auc": float(np.nanmean([up_r["mean_pr_auc"], dn_r["mean_pr_auc"]])),
            "median_pr_auc": float(np.nanmean([up_r["median_pr_auc"], dn_r["median_pr_auc"]])),
            "mean_pr_ratio": float(np.nanmean([up_r["mean_pr_ratio"], dn_r["mean_pr_ratio"]])),
            "median_pr_ratio": mean_pr_ratio,
            "worst_side_pr_ratio": worst_pr_ratio,
            "positive_pr_lift_rate": mean_lift_rate,
            "worst_side_score": worst_score,
            "median_top_decile_lift": float(np.nanmean([up_r["median_top_decile_lift"], dn_r["median_top_decile_lift"]])),
            "median_brier_skill": float(np.nanmean([up_r["median_brier_skill"], dn_r["median_brier_skill"]])),
            "median_ece": float(np.nanmean([up_r["median_ece"], dn_r["median_ece"]])),
            "objective_score": (
                0.40 * mean_score
                + 0.30 * worst_score
                + 0.20 * worst_pr_ratio
                + 0.10 * mean_lift_rate
            ),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["objective", "objective_score"], ascending=[True, False])
    return out


def aggregate_objectives_across_assets(objective_summary: pd.DataFrame) -> pd.DataFrame:
    if objective_summary.empty:
        return pd.DataFrame()

    group_cols = ["objective", "label_scheme", "method", "horizon", "model"]
    rows = []
    for key, g in objective_summary.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, key))
        d.update({
            "asset_count": int(g["asset_name"].nunique()),
            "mean_objective_score": float(g["objective_score"].mean()),
            "median_objective_score": float(g["objective_score"].median()),
            "min_objective_score": float(g["objective_score"].min()),
            "mean_median_pr_ratio": float(g["median_pr_ratio"].mean()),
            "min_median_pr_ratio": float(g["median_pr_ratio"].min()),
            "mean_positive_pr_lift_rate": float(g["positive_pr_lift_rate"].mean()),
            "mean_median_brier_skill": float(g["median_brier_skill"].mean()),
            "mean_median_ece": float(g["median_ece"].mean()),
        })
        # Cross-asset score: penalize asset-specific one-hit results.
        d["cross_asset_objective_score"] = (
            0.50 * d["median_objective_score"]
            + 0.25 * d["min_objective_score"]
            + 0.15 * d["mean_median_pr_ratio"]
            + 0.10 * d["mean_positive_pr_lift_rate"]
        )
        rows.append(d)

    return pd.DataFrame(rows).sort_values(["objective", "cross_asset_objective_score"], ascending=[True, False])


def run(args) -> Dict[str, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = parse_list(args.inputs)
    asset_names = parse_list(args.asset_names)

    if len(inputs) != len(asset_names):
        raise ValueError(f"inputs count != asset_names count: {len(inputs)} vs {len(asset_names)}")

    label_specs = parse_label_schemes(args.label_schemes)

    all_fold_metrics = []
    all_preds = []
    all_k = []
    all_label_dist = []
    all_cal_bins = []
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

        fold_metrics, preds, k_df, label_dist, cal_bins = run_asset(asset_name, input_path, args, label_specs)
        all_fold_metrics.append(fold_metrics)
        all_preds.append(preds)
        all_k.append(k_df)
        all_label_dist.append(label_dist)
        all_cal_bins.append(cal_bins)

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True) if all_fold_metrics else pd.DataFrame()
    non_empty_preds = [x for x in all_preds if not x.empty]
    oos_predictions = pd.concat(non_empty_preds, ignore_index=True) if non_empty_preds else pd.DataFrame()
    k_by_fold = pd.concat(all_k, ignore_index=True) if all_k else pd.DataFrame()
    label_dist = pd.concat(all_label_dist, ignore_index=True) if all_label_dist else pd.DataFrame()
    non_empty_cal_bins = [x for x in all_cal_bins if not x.empty]
    cal_bins = pd.concat(non_empty_cal_bins, ignore_index=True) if non_empty_cal_bins else pd.DataFrame()

    head_summary = summarize_head_models(fold_metrics)
    objective_summary = summarize_objectives(head_summary)
    objective_agg = aggregate_objectives_across_assets(objective_summary)

    objective_top20 = objective_agg.groupby("objective", group_keys=False).head(20) if not objective_agg.empty else pd.DataFrame()

    # Best per objective.
    best_rows = []
    for obj, g in objective_agg.groupby("objective") if not objective_agg.empty else []:
        best_rows.append(g.sort_values("cross_asset_objective_score", ascending=False).iloc[0].to_dict())

    best_config = {
        "experiment": "multi_objective_touch_model_optimizer",
        "asset_periods": asset_periods,
        "config": {
            "label_schemes": [spec.__dict__ for spec in label_specs],
            "models": parse_list(args.models),
            "train_window": args.train_window,
            "calibration_window": args.calibration_window,
            "test_window": args.test_window,
            "embargo": max(args.embargo, max(spec.horizon for spec in label_specs)),
            "target_k_source": args.target_k_source,
            "probability_mode": args.probability_mode,
            "metrics_note": "Allocation is not evaluated. This is head-level prediction performance only.",
        },
        "best_by_objective": best_rows,
        "decision_note": (
            "Use these results to choose prediction heads. Portfolio allocation must be evaluated separately later."
        ),
    }

    summary = {
        "experiment": "multi_objective_touch_model_optimizer",
        "asset_count": len(asset_names),
        "asset_periods": asset_periods,
        "fold_metric_rows": int(len(fold_metrics)),
        "head_summary_rows": int(len(head_summary)),
        "objective_summary_rows": int(len(objective_summary)),
        "objective_aggregate_rows": int(len(objective_agg)),
        "best_by_objective": best_rows,
    }

    outputs = {
        "summary": save_json(out_dir / "model_optimizer_summary.json", summary),
        "best_model_config": save_json(out_dir / "best_model_config.json", best_config),
        "fold_metrics": save_csv(out_dir / "fold_metrics.csv", fold_metrics),
        "head_model_summary": save_csv(out_dir / "head_model_summary.csv", head_summary),
        "objective_summary": save_csv(out_dir / "objective_summary.csv", objective_summary),
        "objective_aggregate": save_csv(out_dir / "objective_aggregate.csv", objective_agg),
        "objective_top20": save_csv(out_dir / "objective_top20.csv", objective_top20),
        "label_k_by_fold": save_csv(out_dir / "label_k_by_fold.csv", k_by_fold),
        "label_distribution_by_fold": save_csv(out_dir / "label_distribution_by_fold.csv", label_dist),
        "calibration_bins": save_csv(out_dir / "calibration_bins.csv", cal_bins),
    }

    if args.save_predictions:
        outputs["oos_predictions"] = save_csv(out_dir / "oos_predictions.csv", oos_predictions)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputs", required=True, help="Comma-separated OHLCV csv paths")
    parser.add_argument("--asset-names", required=True, help="Comma-separated asset names")
    parser.add_argument("--output-dir", default="touch_model_optimizer_output")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")

    parser.add_argument("--label-schemes", default="target_h10_rate30,fixed_h10_k1p0")
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--target-k-source", choices=["train", "train_cal"], default="train")
    parser.add_argument("--k-min", type=float, default=0.25)
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--k-grid-size", type=int, default=80)

    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=0)

    parser.add_argument("--models", default="extratrees,hgb,logistic")
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--probability-mode", choices=["raw", "platt"], default="platt")
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--save-predictions", action="store_true")

    args = parser.parse_args()

    outputs = run(args)
    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))

    print("[OK] Multi-objective touch model optimization completed.")
    print(json.dumps({
        "asset_count": summary["asset_count"],
        "fold_metric_rows": summary["fold_metric_rows"],
        "head_summary_rows": summary["head_summary_rows"],
        "objective_summary_rows": summary["objective_summary_rows"],
        "best_by_objective": summary["best_by_objective"],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
