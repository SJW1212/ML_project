# -*- coding: utf-8 -*-
"""
return_seeking_touch_deep_dive.py

4단계: 수익추구형 Up-touch 모델 고도화 전용 스크립트.

핵심 대상
---------
현재 3단계 최고 후보:
- Objective: return_seeking
- Label: fixed_h10_k1.0
- Target: y_up_touch
- Model: ExtraTrees

목적
----
1. 수익추구형 Up-touch 모델의 성능을 더 깊게 진단
2. ExtraTrees 중심으로 feature importance 안정성 분석
3. top-decile / top-quintile precision 분석
4. raw probability를 직접 확률로 쓰지 않고 percentile score로 변환
5. ExtraTrees / HGB / Logistic / RandomForest / 선택적 XGBoost / LightGBM / CatBoost 비교
6. 모델별 fold stability와 자산별 성능 확인
7. 5단계로 넘길 best return-seeking model config 생성

주의
----
이 스크립트는 portfolio allocation을 전혀 평가하지 않습니다.
수익추구형 Up-touch head의 예측 성능만 평가합니다.

실행 예시
---------
python return_seeking_touch_deep_dive.py ^
  --inputs "QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv" ^
  --asset-names "QQQ,SPY,SOXX,XLK" ^
  --output-dir "return_seeking_deep_dive_all"

선택 모델 추가:
python return_seeking_touch_deep_dive.py ^
  --inputs "QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv" ^
  --asset-names "QQQ,SPY,SOXX,XLK" ^
  --models "extratrees,hgb,logistic,randomforest,xgboost,lightgbm,catboost" ^
  --output-dir "return_seeking_deep_dive_all_boosting"

출력 파일
---------
output_dir/
├─ return_seeking_deep_dive_summary.json
├─ best_return_seeking_config.json
├─ fold_metrics.csv
├─ model_comparison.csv
├─ asset_model_summary.csv
├─ quantile_precision.csv
├─ quantile_precision_aggregate.csv
├─ feature_importance.csv
├─ feature_importance_stability.csv
├─ calibration_bins.csv
├─ score_distribution.csv
└─ oos_predictions.csv   [--save-predictions 사용 시]

의존성
------
python>=3.10
pandas
numpy
scikit-learn

선택 의존성:
xgboost
lightgbm
catboost
"""

from __future__ import annotations

import argparse
import importlib
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

    # Return / trend / volatility
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

    # Volatility ratios
    for a, b in [(5, 20), (10, 40), (20, 60), (20, 120), (60, 252)]:
        if f"vol_{a}d" in new_cols and f"vol_{b}d" in new_cols:
            new_cols[f"vol_ratio_{a}_{b}"] = new_cols[f"vol_{a}d"] / new_cols[f"vol_{b}d"].replace(0, np.nan)
        if f"down_vol_{a}d" in new_cols and f"down_vol_{b}d" in new_cols:
            new_cols[f"down_vol_ratio_{a}_{b}"] = new_cols[f"down_vol_{a}d"] / new_cols[f"down_vol_{b}d"].replace(0, np.nan)

    # Moving average gap and slopes
    for w in [5, 10, 20, 40, 60, 120, 200]:
        ma = close.rolling(w, min_periods=rolling_min_periods(w)).mean()
        new_cols[f"ma_gap_{w}d"] = close / ma.replace(0, np.nan) - 1.0
        new_cols[f"ma_slope_5d_{w}d"] = ma.pct_change(5)
        new_cols[f"ma_slope_20d_{w}d"] = ma.pct_change(20)

    # Momentum oscillators
    for w in [7, 14, 21]:
        new_cols[f"rsi_{w}d"] = rsi(close, w)

    # Price location / breakout pressure
    for w in [10, 20, 40, 60, 120, 252]:
        rolling_high = high.rolling(w, min_periods=rolling_min_periods(w)).max()
        rolling_low = low.rolling(w, min_periods=rolling_min_periods(w)).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)

        new_cols[f"dist_to_high_{w}d"] = close / rolling_high.replace(0, np.nan) - 1.0
        new_cols[f"dist_to_low_{w}d"] = close / rolling_low.replace(0, np.nan) - 1.0
        new_cols[f"range_position_{w}d"] = (close - rolling_low) / price_range
        new_cols[f"breakout_pressure_{w}d"] = high / rolling_high.shift(1).replace(0, np.nan) - 1.0
        new_cols[f"breakdown_pressure_{w}d"] = low / rolling_low.shift(1).replace(0, np.nan) - 1.0

    # Candle / ATR-like features
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

    # Volume features
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

    # Composite risk/trend features
    new_cols["trend_score_20_60"] = (
        0.5 * (close / close.rolling(20, min_periods=10).mean().replace(0, np.nan) - 1.0)
        + 0.5 * (close / close.rolling(60, min_periods=20).mean().replace(0, np.nan) - 1.0)
    )
    new_cols["vol_adjusted_momentum_20"] = close.pct_change(20) / ret.rolling(20, min_periods=10).std().replace(0, np.nan)
    new_cols["drawdown_recovery_20_120"] = rolling_mdd(close, 20) - rolling_mdd(close, 120)

    feat_df = pd.DataFrame(new_cols)
    d = pd.concat([d, feat_df], axis=1)
    d = d.copy()  # defragment

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


def make_up_touch_label(
    df: pd.DataFrame,
    horizon: int = 10,
    vol_window: int = 60,
    k: float = 1.0,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    out["current_horizon_vol"] = current_horizon_volatility(close, horizon, vol_window)
    out["upper_barrier"] = close * (1.0 + k * out["current_horizon_vol"])
    fh, _ = explicit_future_high_low(high, low, horizon)
    out["future_high_h"] = fh
    out["y_up_touch"] = (out["future_high_h"] >= out["upper_barrier"]).astype(float)

    invalid = (
        out["current_horizon_vol"].isna()
        | out["upper_barrier"].isna()
        | out["future_high_h"].isna()
    )
    out.loc[invalid, "y_up_touch"] = np.nan
    return out


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


def fold_indices(fold: Fold, horizon: int, n: int) -> Dict[str, np.ndarray]:
    # Avoid label overlap across train/cal boundary.
    train_end_safe = max(fold.train_start, fold.train_end - horizon)
    cal_end_safe = max(fold.cal_start, fold.cal_end - horizon)

    return {
        "train": np.arange(fold.train_start, train_end_safe),
        "cal": np.arange(fold.cal_start, cal_end_safe),
        "test": np.arange(fold.test_start, min(fold.test_end, n)),
    }


# ============================================================
# 4. Models
# ============================================================

def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def available_model_names(requested: List[str]) -> Tuple[List[str], List[str]]:
    ok = []
    skipped = []
    for m in requested:
        ml = m.lower()
        if ml == "xgboost" and not module_available("xgboost"):
            skipped.append(m)
        elif ml == "lightgbm" and not module_available("lightgbm"):
            skipped.append(m)
        elif ml == "catboost" and not module_available("catboost"):
            skipped.append(m)
        else:
            ok.append(ml)
    return ok, skipped


def make_model(name: str, args, random_state: int) -> Pipeline:
    name = name.lower()

    if name == "logistic":
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

    if name == "extratrees":
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

    if name == "randomforest":
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

    if name == "hgb":
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

    if name == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=max(2, min(args.max_depth, 8)),
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", clf),
        ])

    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(
            n_estimators=args.n_estimators,
            max_depth=max(2, min(args.max_depth, 8)),
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            num_leaves=31,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", clf),
        ])

    if name == "catboost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(
            iterations=args.n_estimators,
            depth=max(2, min(args.max_depth, 8)),
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            verbose=False,
            random_seed=random_state,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", clf),
        ])

    raise ValueError(f"unknown model: {name}")


def predict_positive_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def extract_feature_importance(model: Pipeline, feature_cols: List[str], model_name: str) -> pd.Series:
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=feature_cols, dtype=float)

    if hasattr(clf, "coef_"):
        coefs = np.ravel(clf.coef_)
        return pd.Series(np.abs(coefs), index=feature_cols, dtype=float)

    return pd.Series(np.nan, index=feature_cols, dtype=float)


# ============================================================
# 5. Metrics / Calibration / Quantiles
# ============================================================

def platt_calibrate(raw_cal: np.ndarray, y_cal: np.ndarray, raw_test: np.ndarray) -> Tuple[np.ndarray, Dict]:
    info = {"calibrated": False, "calibration_method": "raw"}

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
        cal = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
        cal.fit(x_cal, y_cal[mask])
        return cal.predict_proba(x_test)[:, 1], {"calibrated": True, "calibration_method": "platt"}
    except Exception as e:
        return raw_test, {"calibrated": False, "calibration_method": "raw", "calibration_error": str(e)}


def percentile_from_calibration(raw_cal: np.ndarray, raw_test: np.ndarray) -> np.ndarray:
    cal = pd.Series(raw_cal).dropna().astype(float)
    if len(cal) == 0:
        return np.full(len(raw_test), np.nan)

    sorted_cal = np.sort(cal.to_numpy())
    # Percentile rank: share of calibration probabilities <= test probability.
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


def binary_metrics(y_true: pd.Series, prob: pd.Series) -> Dict:
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
        "top_quintile_precision": np.nan,
        "top_quintile_lift": np.nan,
    }

    if len(y) == 0:
        return out

    base = out["positive_rate"]

    if y.nunique() == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        out["pr_lift"] = out["pr_auc"] - base
        out["pr_ratio"] = safe_divide(out["pr_auc"], base)

        brier = float(brier_score_loss(y, p))
        baseline_brier = float(brier_score_loss(y, np.full(len(y), base)))
        out["brier"] = brier
        out["brier_skill"] = 1.0 - safe_divide(brier, baseline_brier)

        ece, _ = expected_calibration_error(y, p, bins=10)
        out["ece"] = ece

        try:
            prec, rec, _ = precision_recall_curve(y, p)
            f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
            out["best_f1"] = float(np.nanmax(f1))
        except Exception:
            pass
    else:
        out["pr_auc"] = base
        out["pr_lift"] = 0.0
        out["pr_ratio"] = 1.0

    for name, frac in [("top_decile", 0.10), ("top_quintile", 0.20)]:
        n_top = max(1, int(math.ceil(frac * len(y))))
        idx = np.argsort(-p.to_numpy())[:n_top]
        top_prec = float(y.iloc[idx].mean())
        out[f"{name}_precision"] = top_prec
        out[f"{name}_lift"] = safe_divide(top_prec, base)

    return out


def quantile_precision_table(y_true: pd.Series, score: pd.Series, bins: int = 10) -> pd.DataFrame:
    y = pd.Series(y_true).astype(float)
    s = pd.Series(score).astype(float)
    mask = y.notna() & s.notna() & np.isfinite(s)
    y = y[mask].astype(int).reset_index(drop=True)
    s = s[mask].astype(float).reset_index(drop=True)

    if len(y) == 0:
        return pd.DataFrame()

    # score percentile high = better. Decile 10 is highest.
    try:
        q = pd.qcut(s.rank(method="first"), q=bins, labels=False, duplicates="drop") + 1
    except Exception:
        q = pd.Series(np.nan, index=s.index)

    base = float(y.mean())
    rows = []
    for b in sorted(pd.Series(q).dropna().unique()):
        m = q == b
        rows.append({
            "quantile_bin": int(b),
            "count": int(m.sum()),
            "actual_rate": float(y[m].mean()),
            "base_rate": base,
            "lift": safe_divide(float(y[m].mean()), base),
            "avg_score": float(s[m].mean()),
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. Main Experiment
# ============================================================

def fit_predict_fold(
    data: pd.DataFrame,
    feature_cols: List[str],
    y: pd.Series,
    fold: Fold,
    horizon: int,
    model_name: str,
    args,
    random_state: int,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = fold_indices(fold, horizon, len(data))
    train_idx, cal_idx, test_idx = idx["train"], idx["cal"], idx["test"]

    y_train = y.iloc[train_idx]
    y_cal = y.iloc[cal_idx]
    y_test = y.iloc[test_idx]

    train_mask = y_train.notna()
    cal_mask = y_cal.notna()

    base_row = {
        "fold_id": fold.fold_id,
        "model": model_name,
        "train_rows": int(train_mask.sum()),
        "cal_rows": int(cal_mask.sum()),
        "test_rows": int(y_test.notna().sum()),
        "train_positive_rate": float(y_train[train_mask].mean()) if train_mask.sum() else np.nan,
        "cal_positive_rate": float(y_cal[cal_mask].mean()) if cal_mask.sum() else np.nan,
        "usable": False,
    }

    if train_mask.sum() < args.min_train_rows or y_train[train_mask].nunique() < 2:
        return base_row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    model = make_model(model_name, args, random_state)
    model.fit(data.iloc[train_idx].loc[train_mask, feature_cols], y_train[train_mask].astype(int))

    raw_cal = predict_positive_proba(model, data.iloc[cal_idx][feature_cols])
    raw_test = predict_positive_proba(model, data.iloc[test_idx][feature_cols])

    if args.probability_mode == "platt":
        valid_cal = y_cal.notna()
        prob_test, cal_info = platt_calibrate(raw_cal[valid_cal.to_numpy()], y_cal[valid_cal].astype(int).to_numpy(), raw_test)
    else:
        prob_test = raw_test
        cal_info = {"calibrated": False, "calibration_method": "raw"}

    score_percentile = percentile_from_calibration(raw_cal, raw_test)

    metrics = binary_metrics(y_test.reset_index(drop=True), pd.Series(prob_test))
    base_row.update(metrics)
    base_row.update(cal_info)
    base_row["usable"] = True

    # Feature importance
    imp = extract_feature_importance(model, feature_cols, model_name)
    imp_df = (
        imp.reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
        .sort_values("importance", ascending=False)
    )
    imp_df["fold_id"] = fold.fold_id
    imp_df["model"] = model_name

    # Quantile precision by probability and percentile score.
    q_prob = quantile_precision_table(y_test.reset_index(drop=True), pd.Series(prob_test), bins=10)
    if not q_prob.empty:
        q_prob["score_type"] = "calibrated_probability"
        q_prob["fold_id"] = fold.fold_id
        q_prob["model"] = model_name

    q_pct = quantile_precision_table(y_test.reset_index(drop=True), pd.Series(score_percentile), bins=10)
    if not q_pct.empty:
        q_pct["score_type"] = "calibration_percentile"
        q_pct["fold_id"] = fold.fold_id
        q_pct["model"] = model_name

    q_df = pd.concat([x for x in [q_prob, q_pct] if not x.empty], ignore_index=True) if (not q_prob.empty or not q_pct.empty) else pd.DataFrame()

    # Calibration bins
    ece, cal_bins = expected_calibration_error(y_test.reset_index(drop=True), pd.Series(prob_test), bins=10)
    if not cal_bins.empty:
        cal_bins["fold_id"] = fold.fold_id
        cal_bins["model"] = model_name

    # Predictions
    pred_df = pd.DataFrame({
        "date": data["date"].iloc[test_idx].to_numpy(),
        "fold_id": fold.fold_id,
        "model": model_name,
        "y_true": y.iloc[test_idx].to_numpy(),
        "prob": prob_test,
        "raw_prob": raw_test,
        "score_percentile": score_percentile,
    })

    return base_row, imp_df, q_df, cal_bins, pred_df


def summarize_model_comparison(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()

    rows = []
    for model, g in fold_metrics.groupby("model"):
        u = g[g["usable"] == True].copy()
        if u.empty:
            rows.append({"model": model, "usable_fold_count": 0, "model_score": -999.0})
            continue

        row = {
            "model": model,
            "fold_count": int(g["fold_id"].nunique()),
            "usable_fold_count": int(u["fold_id"].nunique()),
            "mean_positive_rate": float(u["positive_rate"].mean()),
            "median_positive_rate": float(u["positive_rate"].median()),
            "mean_pr_auc": float(u["pr_auc"].mean()),
            "median_pr_auc": float(u["pr_auc"].median()),
            "mean_pr_ratio": float(u["pr_ratio"].mean()),
            "median_pr_ratio": float(u["pr_ratio"].median()),
            "min_pr_ratio": float(u["pr_ratio"].min()),
            "positive_pr_lift_rate": float((u["pr_lift"] > 0).mean()),
            "mean_roc_auc": float(u["roc_auc"].mean()),
            "median_roc_auc": float(u["roc_auc"].median()),
            "mean_brier_skill": float(u["brier_skill"].mean()),
            "median_brier_skill": float(u["brier_skill"].median()),
            "positive_brier_skill_rate": float((u["brier_skill"] > 0).mean()),
            "mean_ece": float(u["ece"].mean()),
            "median_ece": float(u["ece"].median()),
            "mean_top_decile_lift": float(u["top_decile_lift"].mean()),
            "median_top_decile_lift": float(u["top_decile_lift"].median()),
            "mean_top_quintile_lift": float(u["top_quintile_lift"].mean()),
            "median_top_quintile_lift": float(u["top_quintile_lift"].median()),
        }
        row["model_score"] = (
            0.35 * row["median_pr_ratio"]
            + 0.25 * row["positive_pr_lift_rate"]
            + 0.25 * row["median_top_decile_lift"]
            + 0.10 * max(row["median_brier_skill"], -1.0)
            - 0.05 * row["median_ece"]
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("model_score", ascending=False)


def summarize_feature_importance(feature_importance: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if feature_importance.empty:
        return pd.DataFrame()

    valid = feature_importance.dropna(subset=["importance"]).copy()
    if valid.empty:
        return pd.DataFrame()

    rows = []
    for feature, g in valid.groupby("feature"):
        mean_imp = float(g["importance"].mean())
        median_imp = float(g["importance"].median())
        std_imp = float(g["importance"].std(ddof=1)) if len(g) > 1 else 0.0
        nonzero_rate = float((g["importance"] > 0).mean())
        top10_rate = float((g["rank_in_fold"] <= 10).mean()) if "rank_in_fold" in g.columns else np.nan
        top20_rate = float((g["rank_in_fold"] <= 20).mean()) if "rank_in_fold" in g.columns else np.nan
        rows.append({
            "feature": feature,
            "mean_importance": mean_imp,
            "median_importance": median_imp,
            "std_importance": std_imp,
            "nonzero_rate": nonzero_rate,
            "top10_rate": top10_rate,
            "top20_rate": top20_rate,
            "importance_stability_score": median_imp * (0.5 + 0.5 * top20_rate) if not pd.isna(top20_rate) else median_imp,
        })

    return pd.DataFrame(rows).sort_values("importance_stability_score", ascending=False).head(top_n)


def aggregate_quantile_precision(q: pd.DataFrame) -> pd.DataFrame:
    if q.empty:
        return pd.DataFrame()
    group_cols = ["asset_name", "model", "score_type", "quantile_bin"]
    return (
        q.groupby(group_cols, dropna=False)
        .agg(
            fold_count=("fold_id", "nunique"),
            mean_actual_rate=("actual_rate", "mean"),
            median_actual_rate=("actual_rate", "median"),
            mean_base_rate=("base_rate", "mean"),
            mean_lift=("lift", "mean"),
            median_lift=("lift", "median"),
            mean_count=("count", "mean"),
        )
        .reset_index()
        .sort_values(["asset_name", "model", "score_type", "quantile_bin"])
    )


def score_distribution(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in preds.groupby(["asset_name", "model"], dropna=False):
        asset_name, model = keys
        for col in ["prob", "raw_prob", "score_percentile"]:
            s = g[col].dropna().astype(float)
            if s.empty:
                continue
            rows.append({
                "asset_name": asset_name,
                "model": model,
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


def run_asset(asset_name: str, input_path: str, args, models: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    imp_parts = []
    q_parts = []
    cal_parts = []
    pred_parts = []

    for fold in folds:
        for model_name in models:
            metric, imp_df, q_df, cal_df, pred_df = fit_predict_fold(
                data=data,
                feature_cols=feature_cols,
                y=y,
                fold=fold,
                horizon=args.horizon,
                model_name=model_name,
                args=args,
                random_state=args.random_state + fold.fold_id,
            )

            metric.update({
                "asset_name": asset_name,
                "horizon": args.horizon,
                "k": args.k,
                "train_start_date": data["date"].iloc[fold.train_start],
                "train_end_date": data["date"].iloc[fold.train_end - 1],
                "cal_start_date": data["date"].iloc[fold.cal_start],
                "cal_end_date": data["date"].iloc[fold.cal_end - 1],
                "test_start_date": data["date"].iloc[fold.test_start],
                "test_end_date": data["date"].iloc[fold.test_end - 1],
            })
            metric_rows.append(metric)

            if not imp_df.empty:
                imp_df["asset_name"] = asset_name
                imp_df["rank_in_fold"] = imp_df.groupby(["asset_name", "model", "fold_id"])["importance"].rank(ascending=False, method="first")
                imp_parts.append(imp_df)

            if not q_df.empty:
                q_df["asset_name"] = asset_name
                q_parts.append(q_df)

            if not cal_df.empty:
                cal_df["asset_name"] = asset_name
                cal_parts.append(cal_df)

            if not pred_df.empty:
                pred_df["asset_name"] = asset_name
                pred_parts.append(pred_df)

    fold_metrics = pd.DataFrame(metric_rows)
    feature_importance = pd.concat(imp_parts, ignore_index=True) if imp_parts else pd.DataFrame()
    quantile_precision = pd.concat(q_parts, ignore_index=True) if q_parts else pd.DataFrame()
    calibration_bins = pd.concat(cal_parts, ignore_index=True) if cal_parts else pd.DataFrame()
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()

    asset_summary = summarize_model_comparison(fold_metrics)
    if not asset_summary.empty:
        asset_summary["asset_name"] = asset_name

    return fold_metrics, asset_summary, feature_importance, quantile_precision, calibration_bins, predictions


def run(args) -> Dict[str, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = parse_list(args.inputs)
    asset_names = parse_list(args.asset_names)
    if len(inputs) != len(asset_names):
        raise ValueError(f"inputs count != asset_names count: {len(inputs)} vs {len(asset_names)}")

    requested_models = parse_list(args.models)
    models, skipped_models = available_model_names(requested_models)
    if not models:
        raise RuntimeError("No usable models. Check --models and optional dependencies.")

    all_metrics = []
    all_asset_summary = []
    all_imp = []
    all_q = []
    all_cal = []
    all_preds = []
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

        metrics, asset_summary, imp, q, cal, preds = run_asset(asset_name, input_path, args, models)
        all_metrics.append(metrics)
        all_asset_summary.append(asset_summary)
        all_imp.append(imp)
        all_q.append(q)
        all_cal.append(cal)
        all_preds.append(preds)

    fold_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    asset_model_summary = pd.concat(all_asset_summary, ignore_index=True) if all_asset_summary else pd.DataFrame()
    feature_importance = pd.concat([x for x in all_imp if not x.empty], ignore_index=True) if all_imp else pd.DataFrame()
    quantile_precision = pd.concat([x for x in all_q if not x.empty], ignore_index=True) if all_q else pd.DataFrame()
    calibration_bins = pd.concat([x for x in all_cal if not x.empty], ignore_index=True) if all_cal else pd.DataFrame()
    predictions = pd.concat([x for x in all_preds if not x.empty], ignore_index=True) if all_preds else pd.DataFrame()

    # Cross-asset model comparison
    if not asset_model_summary.empty:
        rows = []
        for model, g in asset_model_summary.groupby("model"):
            row = {
                "model": model,
                "asset_count": int(g["asset_name"].nunique()),
                "mean_model_score": float(g["model_score"].mean()),
                "median_model_score": float(g["model_score"].median()),
                "min_model_score": float(g["model_score"].min()),
                "mean_median_pr_ratio": float(g["median_pr_ratio"].mean()),
                "min_median_pr_ratio": float(g["median_pr_ratio"].min()),
                "mean_positive_pr_lift_rate": float(g["positive_pr_lift_rate"].mean()),
                "mean_median_top_decile_lift": float(g["median_top_decile_lift"].mean()),
                "mean_median_brier_skill": float(g["median_brier_skill"].mean()),
                "mean_median_ece": float(g["median_ece"].mean()),
            }
            row["cross_asset_score"] = (
                0.45 * row["median_model_score"]
                + 0.25 * row["min_model_score"]
                + 0.20 * row["mean_median_pr_ratio"]
                + 0.10 * row["mean_positive_pr_lift_rate"]
            )
            rows.append(row)
        model_comparison = pd.DataFrame(rows).sort_values("cross_asset_score", ascending=False)
    else:
        model_comparison = pd.DataFrame()

    feature_importance_stability = summarize_feature_importance(feature_importance, top_n=args.top_features)
    quantile_precision_agg = aggregate_quantile_precision(quantile_precision)
    score_dist = score_distribution(predictions)

    best_row = model_comparison.iloc[0].to_dict() if not model_comparison.empty else {}
    best_model = best_row.get("model")

    best_config = {
        "experiment": "return_seeking_touch_deep_dive",
        "asset_periods": asset_periods,
        "target": {
            "objective": "return_seeking",
            "label": "y_up_touch_fixed_h10_k1.0",
            "horizon": args.horizon,
            "k": args.k,
            "vol_window": args.vol_window,
        },
        "models_requested": requested_models,
        "models_used": models,
        "models_skipped_missing_dependency": skipped_models,
        "best_model": best_row,
        "interpretation": {
            "use_probability_as_literal_probability": False,
            "recommended_score": "score_percentile or ranking percentile",
            "reason": "Calibration may remain weak even when ranking metrics are strong.",
        },
        "next_step": "Validate best return-seeking model with percentile-score thresholds and then test portfolio/allocation separately.",
    }

    summary = {
        "experiment": "return_seeking_touch_deep_dive",
        "asset_count": len(asset_names),
        "asset_periods": asset_periods,
        "models_used": models,
        "models_skipped_missing_dependency": skipped_models,
        "fold_metric_rows": int(len(fold_metrics)),
        "asset_summary_rows": int(len(asset_model_summary)),
        "feature_importance_rows": int(len(feature_importance)),
        "best_model": best_row,
        "decision_note": (
            "This is a return-seeking head-level deep dive. "
            "No allocation or portfolio performance is evaluated."
        ),
    }

    outputs = {
        "summary": save_json(out_dir / "return_seeking_deep_dive_summary.json", summary),
        "best_config": save_json(out_dir / "best_return_seeking_config.json", best_config),
        "fold_metrics": save_csv(out_dir / "fold_metrics.csv", fold_metrics),
        "model_comparison": save_csv(out_dir / "model_comparison.csv", model_comparison),
        "asset_model_summary": save_csv(out_dir / "asset_model_summary.csv", asset_model_summary),
        "feature_importance": save_csv(out_dir / "feature_importance.csv", feature_importance),
        "feature_importance_stability": save_csv(out_dir / "feature_importance_stability.csv", feature_importance_stability),
        "quantile_precision": save_csv(out_dir / "quantile_precision.csv", quantile_precision),
        "quantile_precision_aggregate": save_csv(out_dir / "quantile_precision_aggregate.csv", quantile_precision_agg),
        "calibration_bins": save_csv(out_dir / "calibration_bins.csv", calibration_bins),
        "score_distribution": save_csv(out_dir / "score_distribution.csv", score_dist),
    }

    if args.save_predictions:
        outputs["oos_predictions"] = save_csv(out_dir / "oos_predictions.csv", predictions)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputs", required=True, help="Comma-separated OHLCV CSV paths")
    parser.add_argument("--asset-names", required=True, help="Comma-separated asset names")
    parser.add_argument("--output-dir", default="return_seeking_deep_dive_output")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")

    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--vol-window", type=int, default=60)

    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=0)

    parser.add_argument("--models", default="extratrees,hgb,logistic,randomforest")
    parser.add_argument("--probability-mode", choices=["raw", "platt"], default="platt")

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

    parser.add_argument("--top-features", type=int, default=80)
    parser.add_argument("--save-predictions", action="store_true")

    args = parser.parse_args()

    outputs = run(args)
    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))

    print("[OK] Return-seeking touch deep dive completed.")
    print(json.dumps({
        "asset_count": summary["asset_count"],
        "models_used": summary["models_used"],
        "models_skipped_missing_dependency": summary["models_skipped_missing_dependency"],
        "fold_metric_rows": summary["fold_metric_rows"],
        "best_model": summary["best_model"],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
