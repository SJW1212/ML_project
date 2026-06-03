# -*- coding: utf-8 -*-
"""
dual_highvol_hybrid_sweep.py

Dual-HighVol Hybrid Sweep for Portfolio Regime Advisor.

목적
----
multihead_ablation 결과에서 확인된 문제:

1. h20_current label
   - 전략 성과는 가장 좋음
   - 그러나 HighVol head fold 안정성은 낮음

2. vol_expansion_ratio label
   - head 안정성은 상대적으로 좋음
   - 그러나 allocation 성과는 약함

따라서 이 코드는 두 HighVol head를 같은 rolling fold에서 동시에 학습하고,
다음 hybrid signal을 비교합니다.

Hybrid HighVol Signal
---------------------
1. h20_only
2. expansion_only
3. h20_or_expansion
4. h20_and_expansion
5. h20_with_expansion_confirm
   - h20 signal이 켜지고 expansion probability가 calibration lower quantile 이상이면 방어
   - 기본 confirm quantile = 0.50

Allocation Sweep
----------------
- highvol 발생 시 equity weight를 0.60~0.90 범위에서 sweep
- 나머지는 cash / bond / bond_cash_mix 중 선택

Multi-head Output
-----------------
- Direction Head: p_down, p_sideways, p_up
- HighVol H20 Head: p_highvol_h20
- HighVol Expansion Head: p_highvol_expansion
- RiskOff Head: p_risk_off
- Hybrid highvol signal
- Uncertainty
- Allocation weights

Leakage 방지
------------
- rolling fold: [core train][calibration][embargo][test]
- threshold는 calibration window에서만 계산
- test window에는 고정 threshold 적용
- signal은 1거래일 shift 후 수익률에 적용
- feature에는 y_, future_, meta_ prefix 제외

실행 예시
--------
python dual_highvol_hybrid_sweep.py ^
  --equity-input QQQ_ohlcv.csv ^
  --bond-input IEF_ohlcv.csv ^
  --ticker QQQ ^
  --bond-ticker IEF ^
  --output-dir dual_highvol_hybrid_results ^
  --train-window 1260 ^
  --calibration-window 252 ^
  --test-window 63 ^
  --transaction-cost-bps 10

빠른 축소 실행
-------------
python dual_highvol_hybrid_sweep.py ^
  --equity-input QQQ_ohlcv.csv ^
  --bond-input IEF_ohlcv.csv ^
  --output-dir dual_highvol_hybrid_fast ^
  --hybrid-modes h20_only,h20_with_expansion_confirm,h20_and_expansion ^
  --defensive-equity-weights 0.60,0.70,0.80 ^
  --defense-assets cash ^
  --transaction-cost-bps 10

smoke:
python dual_highvol_hybrid_sweep.py --smoke-test

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

LABEL_PREFIX = "y_"
FUTURE_PREFIX = "future_"
META_PREFIX = "meta_"


# ============================================================
# 0. Utils
# ============================================================

def parse_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


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
# 1. Data
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
# 2. Features
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
    else:
        out["volume_change_20d"] = np.nan
        out["volume_zscore_20d"] = np.nan

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
# 3. Labels
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


def add_direction_label(df: pd.DataFrame, horizon: int = 20, vol_window: int = 60, direction_k: float = 0.25) -> pd.DataFrame:
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


def add_h20_highvol_label(
    df: pd.DataFrame,
    horizon: int = 20,
    vol_window: int = 60,
    high_vol_quantile: float = 0.75,
    high_vol_lookback: int = 252,
) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    daily_vol_t = returns.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().shift(1)
    current_horizon_vol = daily_vol_t * math.sqrt(horizon)
    future_realized_vol = compute_forward_realized_vol(returns, horizon)
    threshold = current_horizon_vol.rolling(high_vol_lookback, min_periods=max(30, high_vol_lookback // 4)).quantile(high_vol_quantile)

    y = (future_realized_vol >= threshold).astype(float)
    invalid = current_horizon_vol.isna() | future_realized_vol.isna() | threshold.isna()

    out["future_realized_vol_h20"] = future_realized_vol
    out["meta_highvol_h20_current_horizon_vol"] = current_horizon_vol
    out["meta_highvol_h20_threshold"] = threshold
    out["y_highvol_h20"] = y.mask(invalid, np.nan)
    return out


def add_expansion_highvol_label(
    df: pd.DataFrame,
    horizon: int = 20,
    vol_window: int = 60,
    expansion_mult: float = 1.25,
) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    daily_vol_t = returns.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().shift(1)
    current_horizon_vol = daily_vol_t * math.sqrt(horizon)
    future_realized_vol = compute_forward_realized_vol(returns, horizon)
    ratio = future_realized_vol / current_horizon_vol

    y = (ratio >= expansion_mult).astype(float)
    invalid = current_horizon_vol.isna() | future_realized_vol.isna() | ratio.isna()

    out["future_realized_vol_expansion"] = future_realized_vol
    out["meta_highvol_exp_current_horizon_vol"] = current_horizon_vol
    out["future_highvol_expansion_ratio"] = ratio
    out["y_highvol_expansion"] = y.mask(invalid, np.nan)
    return out


def add_riskoff_label(df: pd.DataFrame, horizon: int = 40, vol_window: int = 60, k_mdd: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    current_horizon_vol = returns.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().shift(1) * math.sqrt(horizon)
    future_mdd = compute_forward_mdd(close, horizon)
    threshold = -k_mdd * current_horizon_vol

    y = (future_mdd <= threshold).astype(float)
    invalid = current_horizon_vol.isna() | future_mdd.isna() | threshold.isna()

    out["future_mdd_riskoff_h"] = future_mdd
    out["meta_riskoff_current_horizon_vol"] = current_horizon_vol
    out["meta_riskoff_threshold"] = threshold
    out["y_risk_off"] = y.mask(invalid, np.nan)
    return out


def build_labeled_dataset(
    df: pd.DataFrame,
    direction_horizon: int,
    highvol_horizon: int,
    riskoff_horizon: int,
    vol_window: int,
    direction_k: float,
    high_vol_quantile: float,
    high_vol_lookback: int,
    expansion_mult: float,
    riskoff_k_mdd: float,
) -> pd.DataFrame:
    out = build_features(df)
    out = add_direction_label(out, horizon=direction_horizon, vol_window=vol_window, direction_k=direction_k)
    out = add_h20_highvol_label(
        out,
        horizon=highvol_horizon,
        vol_window=vol_window,
        high_vol_quantile=high_vol_quantile,
        high_vol_lookback=high_vol_lookback,
    )
    out = add_expansion_highvol_label(out, horizon=highvol_horizon, vol_window=vol_window, expansion_mult=expansion_mult)
    out = add_riskoff_label(out, horizon=riskoff_horizon, vol_window=vol_window, k_mdd=riskoff_k_mdd)
    return out


# ============================================================
# 4. Model / Calibration
# ============================================================

def make_extratrees(random_state: int = 42, n_estimators: int = 150) -> Pipeline:
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


def get_multiclass_proba(model: Pipeline, x: pd.DataFrame, classes: Sequence[int]) -> pd.DataFrame:
    raw = model.predict_proba(x)
    model_classes = list(model.named_steps["model"].classes_)
    out = np.zeros((len(x), len(classes)), dtype=float)

    for j, cls in enumerate(classes):
        if cls in model_classes:
            out[:, j] = raw[:, model_classes.index(cls)]

    row_sum = out.sum(axis=1)
    missing = row_sum == 0
    if missing.any():
        out[missing, :] = 1.0 / len(classes)
        row_sum = out.sum(axis=1)

    out = out / row_sum.reshape(-1, 1)
    return pd.DataFrame(out, columns=["p_down", "p_sideways", "p_up"], index=x.index)


# ============================================================
# 5. Folds / Metrics
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
    p = proba_df[["p_down", "p_sideways", "p_up"]].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(p).all(axis=1)
    y = y[mask].astype(int)
    p = p[mask]

    if len(y) == 0:
        return {}

    pred = np.argmax(p, axis=1)
    return {
        "eval_rows": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "pred_down_rate": float((pred == 0).mean()),
        "pred_sideways_rate": float((pred == 1).mean()),
        "pred_up_rate": float((pred == 2).mean()),
    }


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
# 6. Prediction
# ============================================================

def train_binary_head(
    core: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    calibration_method: str,
    random_state: int,
    n_estimators: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if core[target_col].nunique() < 2:
        base = float(core[target_col].mean())
        raw_cal = np.full(len(cal), base)
        raw_test = np.full(len(test), base)
        return raw_cal, raw_cal, raw_test, raw_test, "skipped_core_one_class_constant_probability"

    model = make_extratrees(random_state=random_state, n_estimators=n_estimators)
    model.fit(core[feature_cols], core[target_col].astype(int))

    raw_cal = model.predict_proba(cal[feature_cols])[:, 1]
    raw_test = model.predict_proba(test[feature_cols])[:, 1]

    calibrator = ProbabilityCalibrator(method=calibration_method)
    if cal[target_col].nunique() >= 2:
        calibrator.fit(raw_cal, cal[target_col].astype(int).to_numpy())
        p_cal = calibrator.transform(raw_cal)
        p_test = calibrator.transform(raw_test)
        status = "calibrated"
    else:
        p_cal = raw_cal
        p_test = raw_test
        status = "skipped_single_class_calibration_set"

    return raw_cal, p_cal, raw_test, p_test, status


def run_dual_head_predictions(
    equity_df: pd.DataFrame,
    feature_set: str,
    direction_horizon: int,
    highvol_horizon: int,
    riskoff_horizon: int,
    vol_window: int,
    direction_k: float,
    high_vol_quantile: float,
    high_vol_lookback: int,
    expansion_mult: float,
    riskoff_k_mdd: float,
    train_window: int,
    calibration_window: int,
    test_window: int,
    h20_threshold_quantile: float,
    expansion_threshold_quantile: float,
    expansion_confirm_quantile: float,
    riskoff_threshold_quantile: float,
    calibration_method: str,
    random_state: int,
    n_estimators: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = build_labeled_dataset(
        equity_df,
        direction_horizon=direction_horizon,
        highvol_horizon=highvol_horizon,
        riskoff_horizon=riskoff_horizon,
        vol_window=vol_window,
        direction_k=direction_k,
        high_vol_quantile=high_vol_quantile,
        high_vol_lookback=high_vol_lookback,
        expansion_mult=expansion_mult,
        riskoff_k_mdd=riskoff_k_mdd,
    )

    feature_cols = select_features(labeled, feature_set)
    required = ["date", "close", "y_direction", "y_highvol_h20", "y_highvol_expansion", "y_risk_off"] + feature_cols
    data = labeled[required].dropna(subset=["y_direction", "y_highvol_h20", "y_highvol_expansion", "y_risk_off"] + feature_cols).reset_index(drop=True)

    folds = build_rolling_folds(
        data,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        embargo=max(direction_horizon, highvol_horizon, riskoff_horizon),
        step=test_window,
    )

    pred_parts: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, object]] = []
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

        # Direction
        if core["y_direction"].nunique() < 2:
            p_dir = pd.DataFrame(
                np.full((len(test), 3), 1 / 3),
                columns=["p_down", "p_sideways", "p_up"],
                index=test.index,
            )
            direction_status = "skipped_core_one_class"
        else:
            direction_model = make_extratrees(random_state=random_state + fold.fold_id * 10 + 1, n_estimators=n_estimators)
            direction_model.fit(core[feature_cols], core["y_direction"].astype(int))
            p_dir = get_multiclass_proba(direction_model, test[feature_cols], classes=[0, 1, 2])
            direction_status = "ok"

        # H20 head
        raw_h20_cal, p_h20_cal, raw_h20_test, p_h20_test, h20_cal_status = train_binary_head(
            core, cal, test, feature_cols, "y_highvol_h20", calibration_method,
            random_state + fold.fold_id * 10 + 2, n_estimators,
        )
        h20_threshold = float(np.quantile(p_h20_cal, h20_threshold_quantile))

        # Expansion head
        raw_exp_cal, p_exp_cal, raw_exp_test, p_exp_test, exp_cal_status = train_binary_head(
            core, cal, test, feature_cols, "y_highvol_expansion", calibration_method,
            random_state + fold.fold_id * 10 + 3, n_estimators,
        )
        exp_threshold = float(np.quantile(p_exp_cal, expansion_threshold_quantile))
        exp_confirm_threshold = float(np.quantile(p_exp_cal, expansion_confirm_quantile))

        # RiskOff head
        raw_ro_cal, p_ro_cal, raw_ro_test, p_ro_test, ro_cal_status = train_binary_head(
            core, cal, test, feature_cols, "y_risk_off", calibration_method,
            random_state + fold.fold_id * 10 + 4, n_estimators,
        )
        ro_threshold = float(np.quantile(p_ro_cal, riskoff_threshold_quantile))

        pred = test[["date", "close", "y_direction", "y_highvol_h20", "y_highvol_expansion", "y_risk_off"]].copy()
        pred["fold_id"] = fold.fold_id
        pred["test_start_date"] = fold.test_start_date
        pred["test_end_date"] = fold.test_end_date

        pred["p_down"] = p_dir["p_down"].to_numpy()
        pred["p_sideways"] = p_dir["p_sideways"].to_numpy()
        pred["p_up"] = p_dir["p_up"].to_numpy()
        pred["direction_pred"] = np.argmax(pred[["p_down", "p_sideways", "p_up"]].to_numpy(), axis=1)
        pred["direction_pred_label"] = pred["direction_pred"].map({0: "down", 1: "sideways", 2: "up"})

        pred["p_h20_raw"] = raw_h20_test
        pred["p_h20"] = p_h20_test
        pred["h20_threshold"] = h20_threshold
        pred["h20_signal_raw"] = (pred["p_h20"] >= h20_threshold).astype(int)

        pred["p_expansion_raw"] = raw_exp_test
        pred["p_expansion"] = p_exp_test
        pred["expansion_threshold"] = exp_threshold
        pred["expansion_confirm_threshold"] = exp_confirm_threshold
        pred["expansion_signal_raw"] = (pred["p_expansion"] >= exp_threshold).astype(int)
        pred["expansion_confirm_raw"] = (pred["p_expansion"] >= exp_confirm_threshold).astype(int)

        pred["p_risk_off_raw"] = raw_ro_test
        pred["p_risk_off"] = p_ro_test
        pred["riskoff_threshold"] = ro_threshold
        pred["riskoff_signal_raw"] = (pred["p_risk_off"] >= ro_threshold).astype(int)

        pred_parts.append(pred)

        # Metrics
        metric_rows.append({**row_base, "head": "direction", "status": direction_status, **direction_metrics(pred["y_direction"], pred[["p_down", "p_sideways", "p_up"]])})
        metric_rows.append({**row_base, "head": "highvol_h20", "status": "ok", "calibration_status": h20_cal_status, **binary_metrics(pred["y_highvol_h20"], pred["p_h20"])})
        metric_rows.append({**row_base, "head": "highvol_expansion", "status": "ok", "calibration_status": exp_cal_status, **binary_metrics(pred["y_highvol_expansion"], pred["p_expansion"])})
        metric_rows.append({**row_base, "head": "riskoff", "status": "ok", "calibration_status": ro_cal_status, **binary_metrics(pred["y_risk_off"], pred["p_risk_off"])})

        threshold_rows.extend([
            {**row_base, "head": "highvol_h20", "threshold_quantile": h20_threshold_quantile, "threshold": h20_threshold,
             "cal_signal_rate": float((p_h20_cal >= h20_threshold).mean()), "test_signal_rate": float((p_h20_test >= h20_threshold).mean()),
             "cal_positive_rate": float(cal["y_highvol_h20"].mean()), "test_positive_rate": float(test["y_highvol_h20"].mean())},
            {**row_base, "head": "highvol_expansion", "threshold_quantile": expansion_threshold_quantile, "threshold": exp_threshold,
             "confirm_quantile": expansion_confirm_quantile, "confirm_threshold": exp_confirm_threshold,
             "cal_signal_rate": float((p_exp_cal >= exp_threshold).mean()), "test_signal_rate": float((p_exp_test >= exp_threshold).mean()),
             "cal_positive_rate": float(cal["y_highvol_expansion"].mean()), "test_positive_rate": float(test["y_highvol_expansion"].mean())},
            {**row_base, "head": "riskoff", "threshold_quantile": riskoff_threshold_quantile, "threshold": ro_threshold,
             "cal_signal_rate": float((p_ro_cal >= ro_threshold).mean()), "test_signal_rate": float((p_ro_test >= ro_threshold).mean()),
             "cal_positive_rate": float(cal["y_risk_off"].mean()), "test_positive_rate": float(test["y_risk_off"].mean())},
        ])

    predictions = pd.concat(pred_parts, axis=0, ignore_index=True).sort_values("date").reset_index(drop=True)
    return predictions, pd.DataFrame(metric_rows), pd.DataFrame(threshold_rows)


# ============================================================
# 7. Hybrid signals / Backtest
# ============================================================

def apply_persistence(signal: pd.Series, mode: str) -> pd.Series:
    s = pd.Series(signal).fillna(0).astype(int)

    if mode == "none":
        return s
    if mode == "2of3":
        return (s.rolling(3, min_periods=1).sum() >= 2).astype(int)
    if mode == "3of5":
        return (s.rolling(5, min_periods=1).sum() >= 3).astype(int)

    raise ValueError(f"unsupported persistence mode: {mode}")


def make_hybrid_signal(pred: pd.DataFrame, hybrid_mode: str) -> pd.Series:
    h20 = pred["h20_signal_raw"].fillna(0).astype(int)
    exp = pred["expansion_signal_raw"].fillna(0).astype(int)
    exp_confirm = pred["expansion_confirm_raw"].fillna(0).astype(int)

    if hybrid_mode == "h20_only":
        return h20
    if hybrid_mode == "expansion_only":
        return exp
    if hybrid_mode == "h20_or_expansion":
        return ((h20 == 1) | (exp == 1)).astype(int)
    if hybrid_mode == "h20_and_expansion":
        return ((h20 == 1) & (exp == 1)).astype(int)
    if hybrid_mode == "h20_with_expansion_confirm":
        return ((h20 == 1) & (exp_confirm == 1)).astype(int)

    raise ValueError(f"unsupported hybrid_mode: {hybrid_mode}")


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


def weights_for_defense(
    highvol: np.ndarray,
    defensive_equity_weight: float,
    defense_asset: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(highvol)
    eq_w = np.ones(n)
    bd_w = np.zeros(n)
    cash_w = np.zeros(n)

    hv = highvol.astype(bool)
    eq_w[hv] = defensive_equity_weight
    remain = 1.0 - defensive_equity_weight

    if defense_asset == "cash":
        cash_w[hv] = remain
    elif defense_asset == "bond":
        bd_w[hv] = remain
    elif defense_asset == "bond_cash_mix":
        bd_w[hv] = remain * 0.5
        cash_w[hv] = remain * 0.5
    else:
        raise ValueError(f"unsupported defense_asset: {defense_asset}")

    total = eq_w + bd_w + cash_w
    return eq_w / total, bd_w / total, cash_w / total


def simulate_strategy(
    ret_df: pd.DataFrame,
    pred: pd.DataFrame,
    hybrid_mode: str,
    persistence_mode: str,
    defensive_equity_weight: float,
    defense_asset: str,
    riskoff_mode: str,
    transaction_cost_bps: float,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    df = ret_df.copy()
    sig_cols = [
        "date", "p_h20", "p_expansion", "p_risk_off",
        "h20_signal_raw", "expansion_signal_raw", "expansion_confirm_raw", "riskoff_signal_raw",
        "direction_pred_label",
    ]
    df = df.merge(pred[sig_cols], on="date", how="left")
    for c in ["h20_signal_raw", "expansion_signal_raw", "expansion_confirm_raw", "riskoff_signal_raw"]:
        df[c] = df[c].fillna(0).astype(int)

    raw_hybrid = make_hybrid_signal(df, hybrid_mode)
    persistent = apply_persistence(raw_hybrid, persistence_mode)
    executed = persistent.shift(1).fillna(0).astype(int)

    if riskoff_mode == "hard_cash":
        riskoff_executed = df["riskoff_signal_raw"].shift(1).fillna(0).astype(int)
        executed = ((executed == 1) | (riskoff_executed == 1)).astype(int)

    eq_w, bd_w, cash_w = weights_for_defense(
        highvol=executed.to_numpy(),
        defensive_equity_weight=defensive_equity_weight,
        defense_asset=defense_asset,
    )

    n = len(df)
    turnover = np.zeros(n)
    turnover[1:] = np.abs(np.diff(eq_w)) + np.abs(np.diff(bd_w)) + np.abs(np.diff(cash_w))
    cost = turnover * (transaction_cost_bps / 10000.0)

    strat_ret = eq_w * df["equity_ret"].to_numpy() + bd_w * df["bond_ret"].to_numpy() + cash_w * df["cash_ret"].to_numpy() - cost
    curve = np.cumprod(1.0 + strat_ret)

    daily = df.copy()
    daily["strategy"] = "dual_highvol_hybrid"
    daily["hybrid_mode"] = hybrid_mode
    daily["persistence_mode"] = persistence_mode
    daily["defensive_equity_weight"] = defensive_equity_weight
    daily["defense_asset"] = defense_asset
    daily["riskoff_mode"] = riskoff_mode
    daily["raw_hybrid_signal"] = raw_hybrid.astype(int)
    daily["persistent_signal"] = persistent.astype(int)
    daily["executed_signal"] = executed.astype(int)
    daily["equity_weight"] = eq_w
    daily["bond_weight"] = bd_w
    daily["cash_weight"] = cash_w
    daily["turnover"] = turnover
    daily["cost"] = cost
    daily["strategy_ret"] = strat_ret
    daily["equity_curve"] = curve

    summary = {
        "strategy": "dual_highvol_hybrid",
        "hybrid_mode": hybrid_mode,
        "persistence_mode": persistence_mode,
        "defensive_equity_weight": defensive_equity_weight,
        "defense_asset": defense_asset,
        "riskoff_mode": riskoff_mode,
        "rows": int(n),
        "raw_signal_rate": float(raw_hybrid.mean()),
        "persistent_signal_rate": float(persistent.mean()),
        "executed_signal_rate": float(executed.mean()),
        "avg_equity_weight": float(eq_w.mean()),
        "avg_bond_weight": float(bd_w.mean()),
        "avg_cash_weight": float(cash_w.mean()),
        "turnover_total": float(turnover.sum()),
        "transaction_cost_total": float(cost.sum()),
        **performance_metrics(curve, strat_ret),
    }
    return summary, daily


def simulate_benchmark(ret_df: pd.DataFrame, strategy: str, transaction_cost_bps: float) -> Tuple[Dict[str, object], pd.DataFrame]:
    df = ret_df.copy()
    n = len(df)

    if strategy == "buy_hold":
        eq_w, bd_w, cash_w = np.ones(n), np.zeros(n), np.zeros(n)
    elif strategy == "constant_normal":
        eq_w, bd_w, cash_w = np.full(n, 0.80), np.full(n, 0.10), np.full(n, 0.10)
    elif strategy == "sixty_forty":
        eq_w, bd_w, cash_w = np.full(n, 0.60), np.full(n, 0.40), np.zeros(n)
    else:
        raise ValueError(f"unsupported benchmark: {strategy}")

    turnover = np.zeros(n)
    cost = turnover * (transaction_cost_bps / 10000.0)
    ret = eq_w * df["equity_ret"].to_numpy() + bd_w * df["bond_ret"].to_numpy() + cash_w * df["cash_ret"].to_numpy() - cost
    curve = np.cumprod(1.0 + ret)

    daily = df.copy()
    daily["strategy"] = strategy
    daily["hybrid_mode"] = "benchmark"
    daily["persistence_mode"] = "none"
    daily["defensive_equity_weight"] = np.nan
    daily["defense_asset"] = strategy
    daily["riskoff_mode"] = "none"
    daily["executed_signal"] = 0
    daily["equity_weight"] = eq_w
    daily["bond_weight"] = bd_w
    daily["cash_weight"] = cash_w
    daily["turnover"] = turnover
    daily["cost"] = cost
    daily["strategy_ret"] = ret
    daily["equity_curve"] = curve

    summary = {
        "strategy": strategy,
        "hybrid_mode": "benchmark",
        "persistence_mode": "none",
        "defensive_equity_weight": np.nan,
        "defense_asset": strategy,
        "riskoff_mode": "none",
        "rows": int(n),
        "raw_signal_rate": 0.0,
        "persistent_signal_rate": 0.0,
        "executed_signal_rate": 0.0,
        "avg_equity_weight": float(eq_w.mean()),
        "avg_bond_weight": float(bd_w.mean()),
        "avg_cash_weight": float(cash_w.mean()),
        "turnover_total": float(turnover.sum()),
        "transaction_cost_total": float(cost.sum()),
        **performance_metrics(curve, ret),
    }
    return summary, daily


# ============================================================
# 8. Runner
# ============================================================

def summarize_head_fold_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for head, g in metrics.groupby("head"):
        row = {"head": head, "fold_count": int(len(g))}
        for col in ["pr_auc", "brier_skill", "ece", "macro_f1", "balanced_accuracy"]:
            if col in g.columns:
                row[f"mean_{col}"] = float(pd.to_numeric(g[col], errors="coerce").mean())
                row[f"median_{col}"] = float(pd.to_numeric(g[col], errors="coerce").median())
        if "probability_polarity" in g.columns:
            row["normal_polarity_rate"] = float((g["probability_polarity"] == "normal_better").mean())
        if "brier_skill" in g.columns:
            bs = pd.to_numeric(g["brier_skill"], errors="coerce")
            row["positive_brier_skill_rate"] = float((bs > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(
    equity_df: pd.DataFrame,
    bond_df: Optional[pd.DataFrame],
    output_dir: str | Path,
    ticker: str,
    bond_ticker: Optional[str],
    feature_set: str,
    direction_horizon: int,
    highvol_horizon: int,
    riskoff_horizon: int,
    vol_window: int,
    direction_k: float,
    high_vol_quantile: float,
    high_vol_lookback: int,
    expansion_mult: float,
    riskoff_k_mdd: float,
    train_window: int,
    calibration_window: int,
    test_window: int,
    h20_threshold_quantile: float,
    expansion_threshold_quantile: float,
    expansion_confirm_quantile: float,
    riskoff_threshold_quantile: float,
    hybrid_modes: Sequence[str],
    persistence_modes: Sequence[str],
    defensive_equity_weights: Sequence[float],
    defense_assets: Sequence[str],
    riskoff_modes: Sequence[str],
    calibration_method: str,
    transaction_cost_bps: float,
    random_state: int,
    n_estimators: int,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, fold_metrics, thresholds = run_dual_head_predictions(
        equity_df=equity_df,
        feature_set=feature_set,
        direction_horizon=direction_horizon,
        highvol_horizon=highvol_horizon,
        riskoff_horizon=riskoff_horizon,
        vol_window=vol_window,
        direction_k=direction_k,
        high_vol_quantile=high_vol_quantile,
        high_vol_lookback=high_vol_lookback,
        expansion_mult=expansion_mult,
        riskoff_k_mdd=riskoff_k_mdd,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        h20_threshold_quantile=h20_threshold_quantile,
        expansion_threshold_quantile=expansion_threshold_quantile,
        expansion_confirm_quantile=expansion_confirm_quantile,
        riskoff_threshold_quantile=riskoff_threshold_quantile,
        calibration_method=calibration_method,
        random_state=random_state,
        n_estimators=n_estimators,
    )

    start_date = pd.to_datetime(predictions["date"]).min()
    end_date = pd.to_datetime(predictions["date"]).max()
    ret_df = align_returns(equity_df, bond_df, start_date, end_date)

    summary_rows = []
    daily_parts = []

    # benchmarks
    for bench in ["buy_hold", "constant_normal"] + (["sixty_forty"] if bond_df is not None else []):
        row, daily = simulate_benchmark(ret_df, bench, transaction_cost_bps)
        summary_rows.append(row)
        daily_parts.append(daily)

    # hybrid strategies
    for hybrid in hybrid_modes:
        for persistence in persistence_modes:
            for eq_weight in defensive_equity_weights:
                for defense_asset in defense_assets:
                    if defense_asset in {"bond", "bond_cash_mix"} and bond_df is None:
                        continue
                    for ro_mode in riskoff_modes:
                        row, daily = simulate_strategy(
                            ret_df=ret_df,
                            pred=predictions,
                            hybrid_mode=hybrid,
                            persistence_mode=persistence,
                            defensive_equity_weight=eq_weight,
                            defense_asset=defense_asset,
                            riskoff_mode=ro_mode,
                            transaction_cost_bps=transaction_cost_bps,
                        )
                        summary_rows.append(row)
                        daily_parts.append(daily)

    strategy_summary = pd.DataFrame(summary_rows)

    # benchmark diffs
    bh = strategy_summary[strategy_summary["strategy"] == "buy_hold"]
    if not bh.empty:
        bh_row = bh.iloc[0]
        for m in ["cagr", "mdd", "calmar", "sharpe", "volatility", "total_return"]:
            strategy_summary[f"{m}_diff_vs_buy_hold"] = strategy_summary[m] - bh_row[m]

    # conservative candidate score
    strategy_summary["candidate_score"] = (
        strategy_summary.get("calmar_diff_vs_buy_hold", 0).fillna(0) * 2.0
        + strategy_summary.get("mdd_diff_vs_buy_hold", 0).fillna(0) * 1.5
        + strategy_summary.get("cagr_diff_vs_buy_hold", 0).fillna(0) * 1.0
        - strategy_summary.get("transaction_cost_total", 0).fillna(0) * 0.5
    )

    strategy_summary["stable_economic_gate"] = (
        (strategy_summary["calmar_diff_vs_buy_hold"] > 0.03)
        & (strategy_summary["mdd_diff_vs_buy_hold"] > 0.03)
        & (strategy_summary["cagr_diff_vs_buy_hold"] > -0.02)
    )

    strategy_summary = strategy_summary.sort_values(
        ["stable_economic_gate", "candidate_score", "calmar"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    daily_all = pd.concat(daily_parts, axis=0, ignore_index=True)
    head_summary = summarize_head_fold_metrics(fold_metrics)

    outputs = {
        "summary": save_csv(output_dir / "dual_highvol_strategy_summary.csv", strategy_summary),
        "top20": save_csv(output_dir / "dual_highvol_top20.csv", strategy_summary.head(20)),
        "predictions": save_csv(output_dir / "dual_highvol_oos_predictions.csv", predictions),
        "fold_metrics": save_csv(output_dir / "dual_highvol_fold_metrics.csv", fold_metrics),
        "head_summary": save_csv(output_dir / "dual_highvol_head_summary.csv", head_summary),
        "thresholds": save_csv(output_dir / "dual_highvol_thresholds.csv", thresholds),
        "daily_returns": save_csv(output_dir / "dual_highvol_strategy_daily_returns.csv", daily_all),
    }

    best = strategy_summary.head(1).to_dict("records")[0] if not strategy_summary.empty else None

    json_summary = {
        "experiment": "dual_highvol_hybrid_sweep",
        "ticker": ticker,
        "bond_ticker": bond_ticker,
        "oos_start": str(start_date.date()),
        "oos_end": str(end_date.date()),
        "rows": int(len(predictions)),
        "feature_set": feature_set,
        "expansion_mult": expansion_mult,
        "h20_threshold_quantile": h20_threshold_quantile,
        "expansion_threshold_quantile": expansion_threshold_quantile,
        "expansion_confirm_quantile": expansion_confirm_quantile,
        "riskoff_threshold_quantile": riskoff_threshold_quantile,
        "train_window": train_window,
        "calibration_window": calibration_window,
        "test_window": test_window,
        "embargo": max(direction_horizon, highvol_horizon, riskoff_horizon),
        "hybrid_modes": list(hybrid_modes),
        "persistence_modes": list(persistence_modes),
        "defensive_equity_weights": list(map(float, defensive_equity_weights)),
        "defense_assets": list(defense_assets),
        "riskoff_modes": list(riskoff_modes),
        "transaction_cost_bps": transaction_cost_bps,
        "head_summary": head_summary.to_dict("records"),
        "best_candidate": best,
        "top20": strategy_summary.head(20).to_dict("records"),
        "decision_note": (
            "This test combines h20_current strategy strength with vol_expansion_ratio confirmation. "
            "Use stable_economic_gate as economic pass gate; still validate head stability before Stable promotion."
        ),
        "output_files": {k: str(v) for k, v in outputs.items()},
    }
    outputs["json"] = save_json(output_dir / "dual_highvol_summary.json", json_summary)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity-input", default="")
    parser.add_argument("--bond-input", default="")
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--bond-ticker", default="IEF")
    parser.add_argument("--output-dir", default="dual_highvol_hybrid_results")

    parser.add_argument("--feature-set", default="down_core")
    parser.add_argument("--direction-horizon", type=int, default=20)
    parser.add_argument("--highvol-horizon", type=int, default=20)
    parser.add_argument("--riskoff-horizon", type=int, default=40)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--direction-k", type=float, default=0.25)
    parser.add_argument("--high-vol-quantile", type=float, default=0.75)
    parser.add_argument("--high-vol-lookback", type=int, default=252)
    parser.add_argument("--expansion-mult", type=float, default=1.25)
    parser.add_argument("--riskoff-k-mdd", type=float, default=2.0)

    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--h20-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--expansion-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--expansion-confirm-quantile", type=float, default=0.50)
    parser.add_argument("--riskoff-threshold-quantile", type=float, default=0.80)

    parser.add_argument("--hybrid-modes", default="h20_only,expansion_only,h20_or_expansion,h20_and_expansion,h20_with_expansion_confirm")
    parser.add_argument("--persistence-modes", default="none,2of3,3of5")
    parser.add_argument("--defensive-equity-weights", default="0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--defense-assets", default="cash,bond,bond_cash_mix")
    parser.add_argument("--riskoff-modes", default="warning_only")

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
        hybrid_modes = ["h20_only", "h20_with_expansion_confirm", "h20_and_expansion"]
        persistence_modes = ["2of3", "3of5"]
        defensive_equity_weights = [0.60, 0.70, 0.80]
        defense_assets = ["cash", "bond"]
        riskoff_modes = ["warning_only"]
    else:
        if not args.equity_input:
            raise ValueError("--equity-input is required unless --smoke-test is used")

        equity_df = load_ohlcv(args.equity_input)
        bond_df = load_ohlcv(args.bond_input) if args.bond_input else None

        train_window = args.train_window
        calibration_window = args.calibration_window
        test_window = args.test_window
        n_estimators = args.n_estimators
        hybrid_modes = parse_list(args.hybrid_modes)
        persistence_modes = parse_list(args.persistence_modes)
        defensive_equity_weights = parse_float_list(args.defensive_equity_weights)
        defense_assets = parse_list(args.defense_assets)
        riskoff_modes = parse_list(args.riskoff_modes)

    outputs = run_experiment(
        equity_df=equity_df,
        bond_df=bond_df,
        output_dir=args.output_dir,
        ticker=args.ticker,
        bond_ticker=args.bond_ticker if bond_df is not None else None,
        feature_set=args.feature_set,
        direction_horizon=args.direction_horizon,
        highvol_horizon=args.highvol_horizon,
        riskoff_horizon=args.riskoff_horizon,
        vol_window=args.vol_window,
        direction_k=args.direction_k,
        high_vol_quantile=args.high_vol_quantile,
        high_vol_lookback=args.high_vol_lookback,
        expansion_mult=args.expansion_mult,
        riskoff_k_mdd=args.riskoff_k_mdd,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        h20_threshold_quantile=args.h20_threshold_quantile,
        expansion_threshold_quantile=args.expansion_threshold_quantile,
        expansion_confirm_quantile=args.expansion_confirm_quantile,
        riskoff_threshold_quantile=args.riskoff_threshold_quantile,
        hybrid_modes=hybrid_modes,
        persistence_modes=persistence_modes,
        defensive_equity_weights=defensive_equity_weights,
        defense_assets=defense_assets,
        riskoff_modes=riskoff_modes,
        calibration_method=args.calibration_method,
        transaction_cost_bps=args.transaction_cost_bps,
        random_state=args.random_state,
        n_estimators=n_estimators,
    )

    summary = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    best = summary.get("best_candidate") or {}

    print("[OK] Dual-HighVol hybrid sweep completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print(json.dumps(
        {
            "ticker": summary["ticker"],
            "oos_start": summary["oos_start"],
            "oos_end": summary["oos_end"],
            "rows": summary["rows"],
            "best_hybrid_mode": best.get("hybrid_mode"),
            "best_persistence_mode": best.get("persistence_mode"),
            "best_defensive_equity_weight": best.get("defensive_equity_weight"),
            "best_defense_asset": best.get("defense_asset"),
            "best_riskoff_mode": best.get("riskoff_mode"),
            "best_cagr": best.get("cagr"),
            "best_mdd": best.get("mdd"),
            "best_calmar": best.get("calmar"),
            "best_cagr_diff_vs_buy_hold": best.get("cagr_diff_vs_buy_hold"),
            "best_mdd_diff_vs_buy_hold": best.get("mdd_diff_vs_buy_hold"),
            "best_calmar_diff_vs_buy_hold": best.get("calmar_diff_vs_buy_hold"),
            "stable_economic_gate": best.get("stable_economic_gate"),
            "candidate_score": best.get("candidate_score"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
