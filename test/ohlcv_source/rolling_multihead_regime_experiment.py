# -*- coding: utf-8 -*-
"""
rolling_multihead_regime_experiment.py

Portfolio Regime Advisor - Rolling Multi-head Regime Experiment.

목적
----
기존 HighVol 단일 실험을 넘어, 원래 설계했던 Multi-head 구조를 rolling OOS로 구현합니다.

출력 Head
---------
1. Direction Head
   - p_down
   - p_sideways
   - p_up
   - direction_pred

2. HighVol Head
   - p_high_vol
   - highvol_threshold
   - highvol_signal_raw
   - highvol_signal_persistent
   - highvol_signal_executed

3. RiskOff Head
   - p_risk_off
   - riskoff_threshold
   - riskoff_signal_raw
   - riskoff_signal_executed
   - 기본값에서는 allocation hard trigger로 쓰지 않고 warning only

4. Uncertainty
   - direction_entropy
   - highvol_entropy
   - riskoff_entropy
   - uncertainty_score
   - confidence_level

5. Conflict / Regime / Allocation
   - conflict_type
   - resolved_regime
   - equity_weight
   - bond_weight
   - cash_weight

기본 설계
---------
- 동일 rolling fold에서 Direction / HighVol / RiskOff를 동시에 예측
- fold 구조: [core train][calibration][embargo][test]
- threshold는 calibration window에서만 계산
- signal은 1거래일 지연 적용
- feature leakage guard: y_, future_, meta_ prefix 제외

기본값
------
- Direction: H20, volatility-scaled 3-class
- HighVol: h20_current, H20
- RiskOff: H40, k_mdd=2.0
- Model: ExtraTreesClassifier
- Binary calibration: sigmoid
- HighVol threshold: q=0.75
- HighVol persistence: 2of3
- HighVol allocation: equity60_cash40
- RiskOff: warning_only

실행 예시
--------
python rolling_multihead_regime_experiment.py ^
  --equity-input QQQ_ohlcv.csv ^
  --bond-input IEF_ohlcv.csv ^
  --ticker QQQ ^
  --bond-ticker IEF ^
  --output-dir rolling_multihead_results_qqq_ief ^
  --train-window 1260 ^
  --calibration-window 252 ^
  --test-window 63 ^
  --transaction-cost-bps 10

smoke:
python rolling_multihead_regime_experiment.py --smoke-test

의존성
------
pip install pandas numpy scikit-learn
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

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 0. constants / utilities
# ============================================================

LABEL_PREFIX = "y_"
FUTURE_PREFIX = "future_"
META_PREFIX = "meta_"


def parse_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def safe_divide(a: float, b: float, default: float = np.nan) -> float:
    if b == 0 or pd.isna(b):
        return default
    return float(a / b)


def to_jsonable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return str(obj)


def save_csv(path: str | Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_json(path: str | Path, data: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=to_jsonable), encoding="utf-8")
    return path


def normalized_entropy_binary(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return h / np.log(2.0)


def normalized_entropy_multiclass(proba: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), 1e-8, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    h = -(p * np.log(p)).sum(axis=1)
    return h / np.log(p.shape[1])


# ============================================================
# 1. data
# ============================================================

def load_ohlcv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        cwd = Path.cwd()
        raise FileNotFoundError(
            "\n[입력 파일을 찾을 수 없습니다]\n"
            f"- 입력 경로: {path}\n"
            f"- 현재 실행 위치: {cwd}\n"
            f"- 절대 경로 기준: {(cwd / path).resolve() if not path.is_absolute() else path}\n"
        )

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("CSV must include 'date' column")
    if "close" not in df.columns:
        raise ValueError("CSV must include 'close' column")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    else:
        out["volume"] = np.nan

    out = (
        out.sort_values("date")
        .dropna(subset=["close"])
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    return out


def make_synthetic_ohlcv(n: int = 1500, seed: int = 42, ticker: str = "QQQ") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n)

    vol = np.full(n, 0.011)
    drift = np.full(n, 0.00035)

    for start, end, local_vol, local_drift in [
        (280, 360, 0.026, -0.0010),
        (700, 790, 0.030, -0.0012),
        (1060, 1135, 0.028, -0.0009),
    ]:
        vol[start:end] = local_vol
        drift[start:end] = local_drift

    ret = rng.normal(drift, vol)
    close = 100 * np.cumprod(1.0 + ret)
    volume = rng.integers(1_000_000, 10_000_000, n)

    return pd.DataFrame({"date": dates, "ticker": ticker, "close": close, "volume": volume})


# ============================================================
# 2. feature builder
# ============================================================

def rolling_mdd(close: pd.Series, window: int) -> pd.Series:
    rolling_max = close.rolling(window, min_periods=max(5, window // 4)).max()
    dd = close / rolling_max - 1.0
    return dd.rolling(window, min_periods=max(5, window // 4)).min()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    ret = close.pct_change()

    out["return_1d"] = ret

    for w in [5, 10, 20, 40, 60, 120, 252]:
        out[f"return_{w}d"] = close.pct_change(w)
        out[f"volatility_{w}d"] = ret.rolling(w, min_periods=max(3, w // 4)).std()
        out[f"downside_volatility_{w}d"] = ret.clip(upper=0).rolling(w, min_periods=max(3, w // 4)).std()
        out[f"ma_{w}d"] = close.rolling(w, min_periods=max(3, w // 4)).mean()
        out[f"ma_gap_{w}d"] = close / out[f"ma_{w}d"] - 1.0
        out[f"mdd_{w}d"] = rolling_mdd(close, w)

    for w in [20, 60, 120]:
        out[f"momentum_{w}d"] = close.pct_change(w)
        out[f"price_slope_{w}d"] = close.pct_change(w) / w

    if "volume" in out.columns:
        volume = out["volume"].astype(float)
        vmean = volume.rolling(20, min_periods=5).mean()
        vstd = volume.rolling(20, min_periods=5).std()
        out["volume_change_20d"] = volume.pct_change(20)
        out["volume_zscore_20d"] = (volume - vmean) / vstd
        out["volume_ma_gap_20d"] = volume / vmean - 1.0
    else:
        out["volume_change_20d"] = np.nan
        out["volume_zscore_20d"] = np.nan
        out["volume_ma_gap_20d"] = np.nan

    return out


FEATURE_SETS: Dict[str, List[str]] = {
    "down_core": [
        "return_5d", "return_20d", "return_60d",
        "downside_volatility_20d", "downside_volatility_60d", "downside_volatility_120d",
        "mdd_20d", "mdd_60d", "mdd_120d", "mdd_252d",
        "volatility_20d", "volatility_60d",
        "ma_gap_20d", "ma_gap_60d",
    ],
    "compact_mixed": [
        "return_5d", "return_20d", "return_60d", "return_120d",
        "volatility_20d", "volatility_60d",
        "downside_volatility_20d", "downside_volatility_60d",
        "mdd_20d", "mdd_60d", "mdd_120d",
        "ma_gap_20d", "ma_gap_60d", "ma_gap_120d",
        "momentum_20d", "momentum_60d", "momentum_120d",
        "volume_change_20d", "volume_zscore_20d",
    ],
}


def select_features(df: pd.DataFrame, feature_set: str) -> List[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set: {feature_set}")

    cols = []
    for c in FEATURE_SETS[feature_set]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            if not c.startswith((LABEL_PREFIX, FUTURE_PREFIX, META_PREFIX)):
                cols.append(c)

    leaked = [c for c in cols if c.startswith((LABEL_PREFIX, FUTURE_PREFIX, META_PREFIX))]
    if leaked:
        raise AssertionError(f"Leakage columns detected: {leaked}")

    return cols


# ============================================================
# 3. labels
# ============================================================

def compute_forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def compute_forward_realized_vol(returns: pd.Series, horizon: int) -> pd.Series:
    values = returns.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)

    for i in range(n):
        future = values[i + 1:i + horizon + 1]
        future = future[np.isfinite(future)]
        if len(future) >= max(2, horizon // 2):
            out[i] = np.std(future, ddof=1) * math.sqrt(horizon)

    return pd.Series(out, index=returns.index)


def compute_forward_mdd(close: pd.Series, horizon: int) -> pd.Series:
    values = close.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)

    for i in range(n):
        start = values[i]
        if not np.isfinite(start) or start <= 0:
            continue
        future = values[i + 1:i + horizon + 1]
        future = future[np.isfinite(future)]
        if len(future) == 0:
            continue
        out[i] = np.min(future / start - 1.0)

    return pd.Series(out, index=close.index)


def add_direction_label(
    df: pd.DataFrame,
    horizon: int = 20,
    vol_window: int = 60,
    direction_k: float = 0.25,
) -> pd.DataFrame:
    """
    y_direction:
    - 0: down
    - 1: sideways
    - 2: up
    """
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    future_return = compute_forward_return(close, horizon)
    current_horizon_vol = returns.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().shift(1) * math.sqrt(horizon)
    up_th = direction_k * current_horizon_vol
    down_th = -direction_k * current_horizon_vol

    y = pd.Series(np.nan, index=out.index)
    y[future_return >= up_th] = 2
    y[(future_return > down_th) & (future_return < up_th)] = 1
    y[future_return <= down_th] = 0

    invalid = future_return.isna() | current_horizon_vol.isna()
    out["future_return_direction_h"] = future_return
    out["meta_direction_current_horizon_vol"] = current_horizon_vol
    out["y_direction"] = y.mask(invalid, np.nan)

    return out


def add_highvol_label(
    df: pd.DataFrame,
    label_mode: str = "h20_current",
    horizon: int = 20,
    vol_window: int = 60,
    high_vol_quantile: float = 0.75,
    high_vol_lookback: int = 252,
    expansion_mult: float = 1.25,
) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    daily_vol_t = returns.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().shift(1)
    current_horizon_vol = daily_vol_t * math.sqrt(horizon)
    future_realized_vol = compute_forward_realized_vol(returns, horizon)

    if label_mode == "h20_current":
        threshold = current_horizon_vol.rolling(
            high_vol_lookback,
            min_periods=max(30, high_vol_lookback // 4),
        ).quantile(high_vol_quantile)
        y = (future_realized_vol >= threshold).astype(float)
        invalid = current_horizon_vol.isna() | future_realized_vol.isna() | threshold.isna()
        out["meta_highvol_threshold"] = threshold

    elif label_mode == "vol_expansion_ratio":
        ratio = future_realized_vol / current_horizon_vol
        y = (ratio >= expansion_mult).astype(float)
        invalid = current_horizon_vol.isna() | future_realized_vol.isna() | ratio.isna()
        out["future_highvol_expansion_ratio"] = ratio

    else:
        raise ValueError(f"unsupported highvol label_mode: {label_mode}")

    out["future_realized_vol_highvol_h"] = future_realized_vol
    out["meta_highvol_current_horizon_vol"] = current_horizon_vol
    out["y_high_vol"] = y.mask(invalid, np.nan)

    return out


def add_riskoff_label(
    df: pd.DataFrame,
    horizon: int = 40,
    vol_window: int = 60,
    k_mdd: float = 2.0,
) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    current_horizon_vol = returns.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().shift(1) * math.sqrt(horizon)
    future_mdd = compute_forward_mdd(close, horizon)
    riskoff_threshold = -k_mdd * current_horizon_vol

    y = (future_mdd <= riskoff_threshold).astype(float)
    invalid = current_horizon_vol.isna() | future_mdd.isna() | riskoff_threshold.isna()

    out["future_mdd_riskoff_h"] = future_mdd
    out["meta_riskoff_current_horizon_vol"] = current_horizon_vol
    out["meta_riskoff_threshold"] = riskoff_threshold
    out["y_risk_off"] = y.mask(invalid, np.nan)

    return out


def build_labeled_dataset(
    df: pd.DataFrame,
    highvol_label_mode: str,
    direction_horizon: int,
    highvol_horizon: int,
    riskoff_horizon: int,
    vol_window: int,
    direction_k: float,
    high_vol_quantile: float,
    high_vol_lookback: int,
    highvol_expansion_mult: float,
    riskoff_k_mdd: float,
) -> pd.DataFrame:
    out = build_features(df)
    out = add_direction_label(out, horizon=direction_horizon, vol_window=vol_window, direction_k=direction_k)
    out = add_highvol_label(
        out,
        label_mode=highvol_label_mode,
        horizon=highvol_horizon,
        vol_window=vol_window,
        high_vol_quantile=high_vol_quantile,
        high_vol_lookback=high_vol_lookback,
        expansion_mult=highvol_expansion_mult,
    )
    out = add_riskoff_label(out, horizon=riskoff_horizon, vol_window=vol_window, k_mdd=riskoff_k_mdd)
    return out


# ============================================================
# 4. models / calibration
# ============================================================

def make_extratrees_binary(random_state: int = 42, n_estimators: int = 150) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )),
    ])


def make_extratrees_multiclass(random_state: int = 42, n_estimators: int = 150) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )),
    ])


class ProbabilityCalibrator:
    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.model = None

    def fit(self, raw_prob: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        raw_prob = np.asarray(raw_prob, dtype=float)
        y_true = np.asarray(y_true, dtype=int)

        mask = np.isfinite(raw_prob) & np.isfinite(y_true)
        raw_prob = np.clip(raw_prob[mask], 1e-8, 1 - 1e-8)
        y_true = y_true[mask]

        if self.method == "none" or len(raw_prob) < 30 or len(np.unique(y_true)) < 2:
            self.model = None
            return self

        if self.method == "sigmoid":
            x = np.log(raw_prob / (1.0 - raw_prob)).reshape(-1, 1)
            lr = LogisticRegression(max_iter=1000, solver="lbfgs")
            lr.fit(x, y_true)
            self.model = ("sigmoid", lr)
            return self

        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_prob, y_true)
            self.model = ("isotonic", iso)
            return self

        raise ValueError(f"unsupported calibration method: {self.method}")

    def transform(self, raw_prob: np.ndarray) -> np.ndarray:
        raw_prob = np.asarray(raw_prob, dtype=float)
        raw_prob = np.clip(raw_prob, 1e-8, 1 - 1e-8)

        if self.model is None:
            return raw_prob

        name, model = self.model
        if name == "sigmoid":
            x = np.log(raw_prob / (1.0 - raw_prob)).reshape(-1, 1)
            return model.predict_proba(x)[:, 1]

        if name == "isotonic":
            return np.clip(model.transform(raw_prob), 0.0, 1.0)

        return raw_prob


# ============================================================
# 5. rolling folds
# ============================================================

@dataclass
class RollingFold:
    fold_id: int
    core_start: int
    core_end: int
    cal_start: int
    cal_end: int
    test_start: int
    test_end: int
    core_start_date: str
    core_end_date: str
    cal_start_date: str
    cal_end_date: str
    test_start_date: str
    test_end_date: str


def build_rolling_folds(
    data: pd.DataFrame,
    train_window: int,
    calibration_window: int,
    test_window: int,
    embargo: int,
    step: Optional[int] = None,
) -> List[RollingFold]:
    if step is None:
        step = test_window

    n = len(data)
    folds: List[RollingFold] = []
    test_start = train_window + calibration_window + embargo
    fold_id = 0

    while test_start + test_window <= n:
        core_start = test_start - embargo - calibration_window - train_window
        core_end = core_start + train_window
        cal_start = core_end
        cal_end = cal_start + calibration_window
        test_end = test_start + test_window

        if core_start >= 0:
            folds.append(
                RollingFold(
                    fold_id=fold_id,
                    core_start=core_start,
                    core_end=core_end,
                    cal_start=cal_start,
                    cal_end=cal_end,
                    test_start=test_start,
                    test_end=test_end,
                    core_start_date=str(data.iloc[core_start]["date"].date()),
                    core_end_date=str(data.iloc[core_end - 1]["date"].date()),
                    cal_start_date=str(data.iloc[cal_start]["date"].date()),
                    cal_end_date=str(data.iloc[cal_end - 1]["date"].date()),
                    test_start_date=str(data.iloc[test_start]["date"].date()),
                    test_end_date=str(data.iloc[test_end - 1]["date"].date()),
                )
            )
            fold_id += 1

        test_start += step

    if not folds:
        raise ValueError("No rolling folds generated. Reduce window sizes or provide longer data.")

    return folds


# ============================================================
# 6. metrics
# ============================================================

def calibration_ece(y_true: Sequence[float], prob: Sequence[float], n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)

    y = y[mask]
    p = np.clip(p[mask], 0.0, 1.0)

    if len(y) == 0:
        return np.nan

    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (p >= left) & (p <= right)
        else:
            m = (p >= left) & (p < right)

        if not m.any():
            continue

        ece += (m.sum() / len(y)) * abs(float(p[m].mean()) - float(y[m].mean()))

    return float(ece)


def binary_metrics(y_true: Sequence[float], prob: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)

    y = y[mask]
    p = np.clip(p[mask], 1e-8, 1 - 1e-8)

    if len(y) == 0:
        return {}

    positive_rate = float(y.mean())
    brier = float(brier_score_loss(y, p))
    brier_baseline = float(positive_rate * (1 - positive_rate))
    brier_skill = 1 - brier / brier_baseline if brier_baseline > 0 else np.nan

    try:
        auc = float(roc_auc_score(y, p))
        inv_auc = float(roc_auc_score(y, 1 - p))
    except ValueError:
        auc = np.nan
        inv_auc = np.nan

    try:
        pr = float(average_precision_score(y, p))
    except ValueError:
        pr = np.nan

    pred = (p >= 0.5).astype(int)

    return {
        "eval_rows": int(len(y)),
        "positive_rate": positive_rate,
        "roc_auc": auc,
        "inverse_roc_auc": inv_auc,
        "probability_polarity": "normal_better" if np.isfinite(auc) and np.isfinite(inv_auc) and auc >= inv_auc else "inverse_better",
        "pr_auc": pr,
        "pr_gain": pr - positive_rate if np.isfinite(pr) else np.nan,
        "pr_ratio": safe_divide(pr, positive_rate),
        "brier": brier,
        "brier_baseline": brier_baseline,
        "brier_skill": brier_skill,
        "ece": calibration_ece(y, p),
        "precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, pred, zero_division=0)),
        "prob_mean": float(p.mean()),
        "prob_std": float(p.std()),
    }


def direction_metrics(y_true: Sequence[float], proba_df: pd.DataFrame) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    cols = ["p_down", "p_sideways", "p_up"]
    p = proba_df[cols].to_numpy(dtype=float)

    mask = np.isfinite(y) & np.isfinite(p).all(axis=1)
    y = y[mask].astype(int)
    p = p[mask]

    if len(y) == 0:
        return {}

    pred = np.argmax(p, axis=1)

    out = {
        "eval_rows": int(len(y)),
        "class_0_rate_down": float((y == 0).mean()),
        "class_1_rate_sideways": float((y == 1).mean()),
        "class_2_rate_up": float((y == 2).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "pred_down_rate": float((pred == 0).mean()),
        "pred_sideways_rate": float((pred == 1).mean()),
        "pred_up_rate": float((pred == 2).mean()),
    }

    # multiclass brier: mean sum((onehot - p)^2)
    onehot = np.zeros_like(p)
    for i, cls in enumerate(y):
        if 0 <= cls <= 2:
            onehot[i, cls] = 1.0
    out["multiclass_brier"] = float(np.mean(np.sum((onehot - p) ** 2, axis=1)))
    out["direction_entropy_mean"] = float(normalized_entropy_multiclass(p).mean())

    return out


def performance_metrics(equity_curve: np.ndarray, returns: np.ndarray, periods_per_year: int = 252) -> Dict[str, float]:
    curve = pd.Series(equity_curve, dtype=float)
    ret = pd.Series(returns, dtype=float)

    if len(curve) < 2:
        return {}

    total_return = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    years = len(ret) / periods_per_year
    cagr = float((curve.iloc[-1] / curve.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(ret.std() * math.sqrt(periods_per_year))
    sharpe = float(ret.mean() / ret.std() * math.sqrt(periods_per_year)) if ret.std() > 0 else np.nan

    dd = curve / curve.cummax() - 1.0
    mdd = float(dd.min())
    calmar = safe_divide(cagr, abs(mdd))

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
    }


# ============================================================
# 7. prediction and allocation
# ============================================================

def get_class_proba(model: Pipeline, x: pd.DataFrame, classes: Sequence[int]) -> pd.DataFrame:
    raw = model.predict_proba(x)
    model_classes = list(model.named_steps["model"].classes_)
    out = np.zeros((len(x), len(classes)), dtype=float)

    for j, cls in enumerate(classes):
        if cls in model_classes:
            out[:, j] = raw[:, model_classes.index(cls)]

    # missing class fallback
    row_sum = out.sum(axis=1)
    missing = row_sum == 0
    if missing.any():
        out[missing, :] = 1.0 / len(classes)
        row_sum = out.sum(axis=1)

    out = out / row_sum.reshape(-1, 1)
    return pd.DataFrame(out, columns=["p_down", "p_sideways", "p_up"], index=x.index)


def apply_persistence(signal: pd.Series, mode: str) -> pd.Series:
    s = pd.Series(signal).fillna(0).astype(int)

    if mode == "none":
        return s

    if mode == "2of3":
        return (s.rolling(3, min_periods=1).sum() >= 2).astype(int)

    if mode == "3of5":
        return (s.rolling(5, min_periods=1).sum() >= 3).astype(int)

    raise ValueError(f"unsupported persistence mode: {mode}")


def allocation_weights(
    highvol_signal: np.ndarray,
    riskoff_signal: np.ndarray,
    direction_pred: np.ndarray,
    allocation_mode: str,
    riskoff_mode: str = "warning_only",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(highvol_signal)
    equity_w = np.ones(n)
    bond_w = np.zeros(n)
    cash_w = np.zeros(n)

    hv = highvol_signal.astype(bool)
    ro = riskoff_signal.astype(bool)

    if allocation_mode == "equity60_cash40":
        equity_w[hv] = 0.60
        cash_w[hv] = 0.40

    elif allocation_mode == "equity70_cash30":
        equity_w[hv] = 0.70
        cash_w[hv] = 0.30

    elif allocation_mode == "equity80_cash20":
        equity_w[hv] = 0.80
        cash_w[hv] = 0.20

    elif allocation_mode == "equity80_bond20":
        equity_w[hv] = 0.80
        bond_w[hv] = 0.20

    elif allocation_mode == "equity70_bond20_cash10":
        equity_w[hv] = 0.70
        bond_w[hv] = 0.20
        cash_w[hv] = 0.10

    else:
        raise ValueError(f"unsupported allocation_mode: {allocation_mode}")

    # RiskOff는 기본 warning_only. hard trigger는 명시적으로 켠 경우에만 적용.
    if riskoff_mode == "hard_cash":
        equity_w[ro] = 0.30
        bond_w[ro] = 0.0
        cash_w[ro] = 0.70
    elif riskoff_mode == "hard_bond_cash":
        equity_w[ro] = 0.30
        bond_w[ro] = 0.30
        cash_w[ro] = 0.40
    elif riskoff_mode == "warning_only":
        pass
    else:
        raise ValueError(f"unsupported riskoff_mode: {riskoff_mode}")

    total = equity_w + bond_w + cash_w
    return equity_w / total, bond_w / total, cash_w / total


def resolve_regime(row: pd.Series) -> Tuple[str, str]:
    direction = row.get("direction_pred_label", "unknown")
    hv = bool(row.get("highvol_signal_executed", 0))
    ro = bool(row.get("riskoff_signal_executed", 0))

    if ro and direction == "up":
        return "UP_RISKOFF_CONFLICT", "risk_warning"
    if ro and hv:
        return "HIGHVOL_RISKOFF", "risk_warning"
    if ro:
        return "RISKOFF_WARNING", "risk_warning"
    if hv and direction == "down":
        return "DOWN_HIGHVOL", "defensive_highvol"
    if hv:
        return "HIGHVOL", "defensive_highvol"
    if direction == "up":
        return "UP_NORMAL", "normal"
    if direction == "down":
        return "DOWN_NORMAL", "watch"
    return "SIDEWAYS_OR_UNKNOWN", "normal"


def align_returns(equity_df: pd.DataFrame, bond_df: Optional[pd.DataFrame], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    eq = equity_df[(equity_df["date"] >= start_date) & (equity_df["date"] <= end_date)][["date", "close"]].copy()
    eq = eq.rename(columns={"close": "equity_close"})
    eq["equity_ret"] = eq["equity_close"].pct_change().fillna(0.0)

    if bond_df is not None:
        bd = bond_df[(bond_df["date"] >= start_date) & (bond_df["date"] <= end_date)][["date", "close"]].copy()
        bd = bd.rename(columns={"close": "bond_close"})
        bd["bond_ret"] = bd["bond_close"].pct_change().fillna(0.0)
        out = eq.merge(bd[["date", "bond_ret"]], on="date", how="left")
        out["bond_ret"] = out["bond_ret"].fillna(0.0)
    else:
        out = eq.copy()
        out["bond_ret"] = 0.0

    out["cash_ret"] = 0.0
    return out


# ============================================================
# 8. rolling multi-head experiment
# ============================================================

def run_rolling_multihead_predictions(
    equity_df: pd.DataFrame,
    feature_set: str,
    highvol_label_mode: str,
    direction_horizon: int,
    highvol_horizon: int,
    riskoff_horizon: int,
    vol_window: int,
    direction_k: float,
    high_vol_quantile: float,
    high_vol_lookback: int,
    highvol_expansion_mult: float,
    riskoff_k_mdd: float,
    train_window: int,
    calibration_window: int,
    test_window: int,
    highvol_threshold_quantile: float,
    riskoff_threshold_quantile: float,
    highvol_persistence: str,
    riskoff_persistence: str,
    calibration_method: str,
    random_state: int,
    n_estimators: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = build_labeled_dataset(
        equity_df,
        highvol_label_mode=highvol_label_mode,
        direction_horizon=direction_horizon,
        highvol_horizon=highvol_horizon,
        riskoff_horizon=riskoff_horizon,
        vol_window=vol_window,
        direction_k=direction_k,
        high_vol_quantile=high_vol_quantile,
        high_vol_lookback=high_vol_lookback,
        highvol_expansion_mult=highvol_expansion_mult,
        riskoff_k_mdd=riskoff_k_mdd,
    )

    feature_cols = select_features(labeled, feature_set)
    required = ["date", "close", "y_direction", "y_high_vol", "y_risk_off"] + feature_cols
    data = labeled[required].dropna(subset=["y_direction", "y_high_vol", "y_risk_off"] + feature_cols).reset_index(drop=True)

    folds = build_rolling_folds(
        data,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        embargo=max(direction_horizon, highvol_horizon, riskoff_horizon),
        step=test_window,
    )

    prediction_parts: List[pd.DataFrame] = []
    fold_metric_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []

    for fold in folds:
        core = data.iloc[fold.core_start:fold.core_end].copy()
        cal = data.iloc[fold.cal_start:fold.cal_end].copy()
        test = data.iloc[fold.test_start:fold.test_end].copy()

        row_base = {
            "fold_id": fold.fold_id,
            **asdict(fold),
            "core_rows": int(len(core)),
            "calibration_rows": int(len(cal)),
            "test_rows": int(len(test)),
        }

        try:
            # Direction Head
            direction_status = "ok"
            if core["y_direction"].nunique() < 2:
                direction_status = "skipped_core_one_class"
                direction_cal_proba = pd.DataFrame(
                    np.full((len(cal), 3), 1 / 3),
                    columns=["p_down", "p_sideways", "p_up"],
                    index=cal.index,
                )
                direction_test_proba = pd.DataFrame(
                    np.full((len(test), 3), 1 / 3),
                    columns=["p_down", "p_sideways", "p_up"],
                    index=test.index,
                )
            else:
                direction_model = make_extratrees_multiclass(
                    random_state=random_state + fold.fold_id * 10 + 1,
                    n_estimators=n_estimators,
                )
                direction_model.fit(core[feature_cols], core["y_direction"].astype(int))
                direction_cal_proba = get_class_proba(direction_model, cal[feature_cols], classes=[0, 1, 2])
                direction_test_proba = get_class_proba(direction_model, test[feature_cols], classes=[0, 1, 2])

            # HighVol Head
            highvol_model = make_extratrees_binary(
                random_state=random_state + fold.fold_id * 10 + 2,
                n_estimators=n_estimators,
            )
            if core["y_high_vol"].nunique() < 2:
                raise ValueError("highvol core train has one class")
            highvol_model.fit(core[feature_cols], core["y_high_vol"].astype(int))
            raw_hv_cal = highvol_model.predict_proba(cal[feature_cols])[:, 1]
            raw_hv_test = highvol_model.predict_proba(test[feature_cols])[:, 1]

            hv_calibrator = ProbabilityCalibrator(method=calibration_method)
            if cal["y_high_vol"].nunique() >= 2:
                hv_calibrator.fit(raw_hv_cal, cal["y_high_vol"].astype(int).to_numpy())
                p_hv_cal = hv_calibrator.transform(raw_hv_cal)
                p_hv_test = hv_calibrator.transform(raw_hv_test)
                hv_calibration_status = "calibrated"
            else:
                p_hv_cal = raw_hv_cal
                p_hv_test = raw_hv_test
                hv_calibration_status = "skipped_single_class_calibration_set"

            hv_threshold = float(np.quantile(p_hv_cal, highvol_threshold_quantile))

            # RiskOff Head
            riskoff_model = make_extratrees_binary(
                random_state=random_state + fold.fold_id * 10 + 3,
                n_estimators=n_estimators,
            )
            if core["y_risk_off"].nunique() < 2:
                # riskoff is sparse. fallback to low constant probability.
                raw_ro_cal = np.full(len(cal), float(core["y_risk_off"].mean()))
                raw_ro_test = np.full(len(test), float(core["y_risk_off"].mean()))
                p_ro_cal = raw_ro_cal
                p_ro_test = raw_ro_test
                ro_calibration_status = "skipped_core_one_class_constant_probability"
            else:
                riskoff_model.fit(core[feature_cols], core["y_risk_off"].astype(int))
                raw_ro_cal = riskoff_model.predict_proba(cal[feature_cols])[:, 1]
                raw_ro_test = riskoff_model.predict_proba(test[feature_cols])[:, 1]

                ro_calibrator = ProbabilityCalibrator(method=calibration_method)
                if cal["y_risk_off"].nunique() >= 2:
                    ro_calibrator.fit(raw_ro_cal, cal["y_risk_off"].astype(int).to_numpy())
                    p_ro_cal = ro_calibrator.transform(raw_ro_cal)
                    p_ro_test = ro_calibrator.transform(raw_ro_test)
                    ro_calibration_status = "calibrated"
                else:
                    p_ro_cal = raw_ro_cal
                    p_ro_test = raw_ro_test
                    ro_calibration_status = "skipped_single_class_calibration_set"

            ro_threshold = float(np.quantile(p_ro_cal, riskoff_threshold_quantile))

            # Fold prediction table
            pred = test[["date", "close", "y_direction", "y_high_vol", "y_risk_off"]].copy()
            pred["fold_id"] = fold.fold_id
            pred["test_start_date"] = fold.test_start_date
            pred["test_end_date"] = fold.test_end_date

            pred["p_down"] = direction_test_proba["p_down"].to_numpy()
            pred["p_sideways"] = direction_test_proba["p_sideways"].to_numpy()
            pred["p_up"] = direction_test_proba["p_up"].to_numpy()
            pred["direction_pred"] = np.argmax(pred[["p_down", "p_sideways", "p_up"]].to_numpy(), axis=1)
            pred["direction_pred_label"] = pred["direction_pred"].map({0: "down", 1: "sideways", 2: "up"})

            pred["p_high_vol_raw"] = raw_hv_test
            pred["p_high_vol"] = p_hv_test
            pred["highvol_threshold"] = hv_threshold
            pred["highvol_signal_raw"] = (pred["p_high_vol"] >= hv_threshold).astype(int)

            pred["p_risk_off_raw"] = raw_ro_test
            pred["p_risk_off"] = p_ro_test
            pred["riskoff_threshold"] = ro_threshold
            pred["riskoff_signal_raw"] = (pred["p_risk_off"] >= ro_threshold).astype(int)

            # Persistence is applied within fold; execution shift is done after concat globally.
            pred["highvol_signal_persistent"] = apply_persistence(pred["highvol_signal_raw"], highvol_persistence).astype(int)
            pred["riskoff_signal_persistent"] = apply_persistence(pred["riskoff_signal_raw"], riskoff_persistence).astype(int)

            pred["direction_entropy"] = normalized_entropy_multiclass(pred[["p_down", "p_sideways", "p_up"]].to_numpy())
            pred["highvol_entropy"] = normalized_entropy_binary(pred["p_high_vol"].to_numpy())
            pred["riskoff_entropy"] = normalized_entropy_binary(pred["p_risk_off"].to_numpy())
            pred["uncertainty_score"] = (
                0.4 * pred["direction_entropy"]
                + 0.4 * pred["highvol_entropy"]
                + 0.2 * pred["riskoff_entropy"]
            )
            pred["confidence_level"] = pd.cut(
                pred["uncertainty_score"],
                bins=[-np.inf, 0.35, 0.65, np.inf],
                labels=["high", "medium", "low"],
            ).astype(str)

            prediction_parts.append(pred)

            # Fold metrics
            d_metrics = direction_metrics(pred["y_direction"], pred[["p_down", "p_sideways", "p_up"]])
            hv_metrics = binary_metrics(pred["y_high_vol"], pred["p_high_vol"])
            ro_metrics = binary_metrics(pred["y_risk_off"], pred["p_risk_off"])

            fold_metric_rows.append({
                **row_base,
                "head": "direction",
                "status": direction_status,
                "calibration_status": "not_calibrated_multiclass_raw_probability",
                **d_metrics,
            })
            fold_metric_rows.append({
                **row_base,
                "head": "highvol",
                "status": "ok",
                "calibration_status": hv_calibration_status,
                **hv_metrics,
            })
            fold_metric_rows.append({
                **row_base,
                "head": "riskoff",
                "status": "ok",
                "calibration_status": ro_calibration_status,
                **ro_metrics,
            })

            threshold_rows.append({
                **row_base,
                "head": "highvol",
                "threshold_quantile": highvol_threshold_quantile,
                "threshold": hv_threshold,
                "cal_signal_rate": float((p_hv_cal >= hv_threshold).mean()),
                "test_signal_rate": float((p_hv_test >= hv_threshold).mean()),
                "cal_positive_rate": float(cal["y_high_vol"].mean()),
                "test_positive_rate": float(test["y_high_vol"].mean()),
            })
            threshold_rows.append({
                **row_base,
                "head": "riskoff",
                "threshold_quantile": riskoff_threshold_quantile,
                "threshold": ro_threshold,
                "cal_signal_rate": float((p_ro_cal >= ro_threshold).mean()),
                "test_signal_rate": float((p_ro_test >= ro_threshold).mean()),
                "cal_positive_rate": float(cal["y_risk_off"].mean()),
                "test_positive_rate": float(test["y_risk_off"].mean()),
            })

        except Exception as e:
            fold_metric_rows.append({
                **row_base,
                "head": "multihead",
                "status": "error",
                "error": str(e),
            })

    if not prediction_parts:
        raise RuntimeError("No valid multi-head predictions generated.")

    pred_all = pd.concat(prediction_parts, axis=0, ignore_index=True).sort_values("date").reset_index(drop=True)

    # Global execution shift to prevent same-day look-ahead execution.
    pred_all["highvol_signal_executed"] = pred_all["highvol_signal_persistent"].shift(1).fillna(0).astype(int)
    pred_all["riskoff_signal_executed"] = pred_all["riskoff_signal_persistent"].shift(1).fillna(0).astype(int)

    # Conflict / regime
    conflicts = pred_all.apply(resolve_regime, axis=1, result_type="expand")
    pred_all["conflict_type"] = conflicts[0]
    pred_all["resolved_regime"] = conflicts[1]

    return pred_all, pd.DataFrame(fold_metric_rows), pd.DataFrame(threshold_rows)


def run_strategy_backtest(
    equity_df: pd.DataFrame,
    bond_df: Optional[pd.DataFrame],
    predictions: pd.DataFrame,
    allocation_mode: str,
    riskoff_mode: str,
    transaction_cost_bps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_date = pd.to_datetime(predictions["date"]).min()
    end_date = pd.to_datetime(predictions["date"]).max()
    ret_df = align_returns(equity_df, bond_df, start_date, end_date)

    pred_cols = [
        "date", "p_down", "p_sideways", "p_up", "direction_pred_label",
        "p_high_vol", "p_risk_off",
        "highvol_signal_executed", "riskoff_signal_executed",
        "uncertainty_score", "confidence_level", "conflict_type", "resolved_regime",
    ]
    df = ret_df.merge(predictions[pred_cols], on="date", how="left")
    df["highvol_signal_executed"] = df["highvol_signal_executed"].fillna(0).astype(int)
    df["riskoff_signal_executed"] = df["riskoff_signal_executed"].fillna(0).astype(int)
    df["direction_pred_label"] = df["direction_pred_label"].fillna("unknown")

    # Strategy weights
    eq_w, bd_w, cash_w = allocation_weights(
        highvol_signal=df["highvol_signal_executed"].to_numpy(),
        riskoff_signal=df["riskoff_signal_executed"].to_numpy(),
        direction_pred=df["direction_pred_label"].to_numpy(),
        allocation_mode=allocation_mode,
        riskoff_mode=riskoff_mode,
    )

    n = len(df)
    turnover = np.zeros(n)
    turnover[1:] = np.abs(np.diff(eq_w)) + np.abs(np.diff(bd_w)) + np.abs(np.diff(cash_w))
    cost = turnover * (transaction_cost_bps / 10000.0)

    strat_ret = eq_w * df["equity_ret"].to_numpy() + bd_w * df["bond_ret"].to_numpy() + cash_w * df["cash_ret"].to_numpy() - cost
    curve = np.cumprod(1.0 + strat_ret)

    df["strategy"] = "multihead_allocation"
    df["equity_weight"] = eq_w
    df["bond_weight"] = bd_w
    df["cash_weight"] = cash_w
    df["turnover"] = turnover
    df["cost"] = cost
    df["strategy_ret"] = strat_ret
    df["equity_curve"] = curve

    # Benchmarks
    daily_parts = [df.copy()]
    summary_rows = [{
        "strategy": "multihead_allocation",
        "allocation_mode": allocation_mode,
        "riskoff_mode": riskoff_mode,
        "rows": int(n),
        "avg_equity_weight": float(np.mean(eq_w)),
        "avg_bond_weight": float(np.mean(bd_w)),
        "avg_cash_weight": float(np.mean(cash_w)),
        "turnover_total": float(np.sum(turnover)),
        "transaction_cost_total": float(np.sum(cost)),
        **performance_metrics(curve, strat_ret),
    }]

    for bench in ["buy_hold", "constant_normal"] + (["sixty_forty"] if bond_df is not None else []):
        b = ret_df.copy()
        if bench == "buy_hold":
            bw_eq, bw_bd, bw_cash = np.ones(n), np.zeros(n), np.zeros(n)
        elif bench == "constant_normal":
            bw_eq, bw_bd, bw_cash = np.full(n, 0.80), np.full(n, 0.10), np.full(n, 0.10)
        elif bench == "sixty_forty":
            bw_eq, bw_bd, bw_cash = np.full(n, 0.60), np.full(n, 0.40), np.zeros(n)
        else:
            continue

        b_turnover = np.zeros(n)
        b_cost = b_turnover * (transaction_cost_bps / 10000.0)
        b_ret = bw_eq * b["equity_ret"].to_numpy() + bw_bd * b["bond_ret"].to_numpy() + bw_cash * b["cash_ret"].to_numpy() - b_cost
        b_curve = np.cumprod(1.0 + b_ret)

        b["strategy"] = bench
        b["equity_weight"] = bw_eq
        b["bond_weight"] = bw_bd
        b["cash_weight"] = bw_cash
        b["turnover"] = b_turnover
        b["cost"] = b_cost
        b["strategy_ret"] = b_ret
        b["equity_curve"] = b_curve

        daily_parts.append(b)

        summary_rows.append({
            "strategy": bench,
            "allocation_mode": bench,
            "riskoff_mode": "none",
            "rows": int(n),
            "avg_equity_weight": float(np.mean(bw_eq)),
            "avg_bond_weight": float(np.mean(bw_bd)),
            "avg_cash_weight": float(np.mean(bw_cash)),
            "turnover_total": float(np.sum(b_turnover)),
            "transaction_cost_total": float(np.sum(b_cost)),
            **performance_metrics(b_curve, b_ret),
        })

    summary = pd.DataFrame(summary_rows)

    bh = summary[summary["strategy"] == "buy_hold"]
    if not bh.empty:
        bh_row = bh.iloc[0]
        for m in ["cagr", "mdd", "calmar", "sharpe", "volatility", "total_return"]:
            summary[f"{m}_diff_vs_buy_hold"] = summary[m] - bh_row[m]

    daily_all = pd.concat(daily_parts, axis=0, ignore_index=True)
    summary = summary.sort_values("calmar", ascending=False).reset_index(drop=True)

    return summary, daily_all


def summarize_head_metrics(pred: pd.DataFrame, fold_metrics: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}

    out["direction_overall"] = direction_metrics(pred["y_direction"], pred[["p_down", "p_sideways", "p_up"]])
    out["highvol_overall"] = binary_metrics(pred["y_high_vol"], pred["p_high_vol"])
    out["riskoff_overall"] = binary_metrics(pred["y_risk_off"], pred["p_risk_off"])

    head_summaries = []
    for head, g in fold_metrics.groupby("head"):
        row = {"head": head, "fold_rows": int(len(g))}
        for col in ["pr_auc", "brier_skill", "ece", "macro_f1", "balanced_accuracy"]:
            if col in g.columns:
                row[f"mean_{col}"] = float(pd.to_numeric(g[col], errors="coerce").mean())
                row[f"median_{col}"] = float(pd.to_numeric(g[col], errors="coerce").median())
        if "probability_polarity" in g.columns:
            row["normal_polarity_rate"] = float((g["probability_polarity"] == "normal_better").mean())
        if "brier_skill" in g.columns:
            bs = pd.to_numeric(g["brier_skill"], errors="coerce")
            row["positive_brier_skill_rate"] = float((bs > 0).mean())
        head_summaries.append(row)

    out["fold_head_summary"] = head_summaries
    return out


# ============================================================
# 9. runner
# ============================================================

def run_experiment(
    equity_df: pd.DataFrame,
    bond_df: Optional[pd.DataFrame],
    output_dir: str | Path,
    ticker: str,
    bond_ticker: Optional[str],
    feature_set: str,
    highvol_label_mode: str,
    direction_horizon: int,
    highvol_horizon: int,
    riskoff_horizon: int,
    vol_window: int,
    direction_k: float,
    high_vol_quantile: float,
    high_vol_lookback: int,
    highvol_expansion_mult: float,
    riskoff_k_mdd: float,
    train_window: int,
    calibration_window: int,
    test_window: int,
    highvol_threshold_quantile: float,
    riskoff_threshold_quantile: float,
    highvol_persistence: str,
    riskoff_persistence: str,
    allocation_mode: str,
    riskoff_mode: str,
    calibration_method: str,
    transaction_cost_bps: float,
    random_state: int,
    n_estimators: int,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, fold_metrics, thresholds = run_rolling_multihead_predictions(
        equity_df=equity_df,
        feature_set=feature_set,
        highvol_label_mode=highvol_label_mode,
        direction_horizon=direction_horizon,
        highvol_horizon=highvol_horizon,
        riskoff_horizon=riskoff_horizon,
        vol_window=vol_window,
        direction_k=direction_k,
        high_vol_quantile=high_vol_quantile,
        high_vol_lookback=high_vol_lookback,
        highvol_expansion_mult=highvol_expansion_mult,
        riskoff_k_mdd=riskoff_k_mdd,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        highvol_threshold_quantile=highvol_threshold_quantile,
        riskoff_threshold_quantile=riskoff_threshold_quantile,
        highvol_persistence=highvol_persistence,
        riskoff_persistence=riskoff_persistence,
        calibration_method=calibration_method,
        random_state=random_state,
        n_estimators=n_estimators,
    )

    strategy_summary, strategy_daily = run_strategy_backtest(
        equity_df=equity_df,
        bond_df=bond_df,
        predictions=predictions,
        allocation_mode=allocation_mode,
        riskoff_mode=riskoff_mode,
        transaction_cost_bps=transaction_cost_bps,
    )

    head_summary = summarize_head_metrics(predictions, fold_metrics)

    outputs = {
        "predictions": save_csv(output_dir / "multihead_oos_predictions.csv", predictions),
        "fold_metrics": save_csv(output_dir / "multihead_fold_metrics.csv", fold_metrics),
        "thresholds": save_csv(output_dir / "multihead_thresholds.csv", thresholds),
        "strategy_summary": save_csv(output_dir / "multihead_strategy_summary.csv", strategy_summary),
        "strategy_daily": save_csv(output_dir / "multihead_strategy_daily_returns.csv", strategy_daily),
    }

    best_strategy = strategy_summary.head(1).to_dict("records")[0] if not strategy_summary.empty else None

    summary = {
        "experiment": "rolling_multihead_regime_experiment",
        "ticker": ticker,
        "bond_ticker": bond_ticker,
        "oos_start": str(pd.to_datetime(predictions["date"]).min().date()),
        "oos_end": str(pd.to_datetime(predictions["date"]).max().date()),
        "rows": int(len(predictions)),
        "feature_set": feature_set,
        "highvol_label_mode": highvol_label_mode,
        "direction_horizon": direction_horizon,
        "highvol_horizon": highvol_horizon,
        "riskoff_horizon": riskoff_horizon,
        "train_window": train_window,
        "calibration_window": calibration_window,
        "test_window": test_window,
        "embargo": max(direction_horizon, highvol_horizon, riskoff_horizon),
        "highvol_threshold_quantile": highvol_threshold_quantile,
        "riskoff_threshold_quantile": riskoff_threshold_quantile,
        "highvol_persistence": highvol_persistence,
        "riskoff_persistence": riskoff_persistence,
        "allocation_mode": allocation_mode,
        "riskoff_mode": riskoff_mode,
        "signal_execution": "persistent signals are shifted by 1 trading day before returns",
        "transaction_cost_bps": transaction_cost_bps,
        "head_summary": head_summary,
        "best_strategy_by_calmar": best_strategy,
        "strategy_rows": strategy_summary.to_dict("records"),
        "decision_note": (
            "This script restores the intended multi-head output structure. "
            "HighVol remains the primary allocation trigger by default; "
            "Direction and RiskOff are auxiliary/warning heads unless explicitly enabled."
        ),
    }
    outputs["summary"] = save_json(output_dir / "multihead_regime_summary.json", summary)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity-input", default="")
    parser.add_argument("--bond-input", default="")
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--bond-ticker", default="IEF")
    parser.add_argument("--output-dir", default="rolling_multihead_results")

    parser.add_argument("--feature-set", default="down_core")
    parser.add_argument("--highvol-label-mode", default="h20_current", choices=["h20_current", "vol_expansion_ratio"])
    parser.add_argument("--direction-horizon", type=int, default=20)
    parser.add_argument("--highvol-horizon", type=int, default=20)
    parser.add_argument("--riskoff-horizon", type=int, default=40)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--direction-k", type=float, default=0.25)
    parser.add_argument("--high-vol-quantile", type=float, default=0.75)
    parser.add_argument("--high-vol-lookback", type=int, default=252)
    parser.add_argument("--highvol-expansion-mult", type=float, default=1.25)
    parser.add_argument("--riskoff-k-mdd", type=float, default=2.0)

    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--highvol-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--riskoff-threshold-quantile", type=float, default=0.80)
    parser.add_argument("--highvol-persistence", default="2of3", choices=["none", "2of3", "3of5"])
    parser.add_argument("--riskoff-persistence", default="none", choices=["none", "2of3", "3of5"])
    parser.add_argument("--allocation-mode", default="equity60_cash40", choices=[
        "equity60_cash40",
        "equity70_cash30",
        "equity80_cash20",
        "equity80_bond20",
        "equity70_bond20_cash10",
    ])
    parser.add_argument("--riskoff-mode", default="warning_only", choices=["warning_only", "hard_cash", "hard_bond_cash"])

    parser.add_argument("--calibration-method", default="sigmoid", choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        equity_df = make_synthetic_ohlcv(n=1500, seed=42, ticker=args.ticker)
        bond_df = make_synthetic_ohlcv(n=1500, seed=7, ticker=args.bond_ticker)
        bond_df["close"] = 100 * np.cumprod(1 + np.random.default_rng(7).normal(0.00005, 0.003, len(bond_df)))

        train_window = 520
        calibration_window = 126
        test_window = 42
        n_estimators = 40
    else:
        if not args.equity_input:
            raise ValueError("--equity-input is required unless --smoke-test is used")
        equity_df = load_ohlcv(args.equity_input)
        bond_df = load_ohlcv(args.bond_input) if args.bond_input else None

        train_window = args.train_window
        calibration_window = args.calibration_window
        test_window = args.test_window
        n_estimators = args.n_estimators

    outputs = run_experiment(
        equity_df=equity_df,
        bond_df=bond_df,
        output_dir=args.output_dir,
        ticker=args.ticker,
        bond_ticker=args.bond_ticker if bond_df is not None else None,
        feature_set=args.feature_set,
        highvol_label_mode=args.highvol_label_mode,
        direction_horizon=args.direction_horizon,
        highvol_horizon=args.highvol_horizon,
        riskoff_horizon=args.riskoff_horizon,
        vol_window=args.vol_window,
        direction_k=args.direction_k,
        high_vol_quantile=args.high_vol_quantile,
        high_vol_lookback=args.high_vol_lookback,
        highvol_expansion_mult=args.highvol_expansion_mult,
        riskoff_k_mdd=args.riskoff_k_mdd,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        highvol_threshold_quantile=args.highvol_threshold_quantile,
        riskoff_threshold_quantile=args.riskoff_threshold_quantile,
        highvol_persistence=args.highvol_persistence,
        riskoff_persistence=args.riskoff_persistence,
        allocation_mode=args.allocation_mode,
        riskoff_mode=args.riskoff_mode,
        calibration_method=args.calibration_method,
        transaction_cost_bps=args.transaction_cost_bps,
        random_state=args.random_state,
        n_estimators=n_estimators,
    )

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    best = summary.get("best_strategy_by_calmar") or {}
    highvol = summary.get("head_summary", {}).get("highvol_overall", {})
    riskoff = summary.get("head_summary", {}).get("riskoff_overall", {})
    direction = summary.get("head_summary", {}).get("direction_overall", {})

    print("[OK] Rolling multi-head regime experiment completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print(json.dumps(
        {
            "ticker": summary["ticker"],
            "oos_start": summary["oos_start"],
            "oos_end": summary["oos_end"],
            "rows": summary["rows"],
            "highvol_label_mode": summary["highvol_label_mode"],
            "allocation_mode": summary["allocation_mode"],
            "riskoff_mode": summary["riskoff_mode"],
            "direction_macro_f1": direction.get("macro_f1"),
            "highvol_pr_auc": highvol.get("pr_auc"),
            "highvol_brier_skill": highvol.get("brier_skill"),
            "riskoff_pr_auc": riskoff.get("pr_auc"),
            "riskoff_brier_skill": riskoff.get("brier_skill"),
            "best_strategy": best.get("strategy"),
            "best_cagr": best.get("cagr"),
            "best_mdd": best.get("mdd"),
            "best_calmar": best.get("calmar"),
            "best_calmar_diff_vs_buy_hold": best.get("calmar_diff_vs_buy_hold"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
