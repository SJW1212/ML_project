# -*- coding: utf-8 -*-
"""
riskoff_highvol_walkforward_experiment.py

RiskOff / HighVol 전수 탐색 + OOF 검증 + calibration 진단 스크립트.

목적
----
Direction 전수 탐색 결과가 "weak auxiliary signal"로 판정되었으므로,
다음 우선순위인 RiskOff / HighVol head를 검증하기 위한 단일 실행 파일입니다.

핵심 기능
---------
1. OHLCV CSV 입력
2. Volatility-scaled label 생성
   - y_high_vol
   - y_risk_off
   - y_direction 참고용
3. LeakageGuard
   - y_, future_, meta_ 컬럼 feature 유입 차단
4. Shared Walk-forward OOF split
   - train/test 정보 집합 정합성 유지
5. RiskOff / HighVol binary model 학습
   - logistic
   - hist_gradient_boosting
   - extratrees
   - randomforest
6. raw probability / calibrated probability 분리
   - none
   - sigmoid
   - isotonic
7. PR-AUC, positive rate, PR gain, ROC-AUC, flipped AUC, Brier, Brier skill, ECE
8. RiskOff MDD-event recall
9. trial 결과, top20, best prediction, summary 저장
10. smoke test 지원

입력 CSV 요구 컬럼
----------------
필수:
- date
- close

권장:
- open
- high
- low
- volume

실행 예시
--------
python riskoff_highvol_walkforward_experiment.py ^
  --input qqq_ohlcv.csv ^
  --ticker QQQ ^
  --output-dir riskoff_highvol_results ^
  --horizons 10,20,40,60,120 ^
  --tasks risk_off,high_vol ^
  --models logistic,hgb,extratrees ^
  --calibration-methods none,sigmoid ^
  --test-window 20 ^
  --step 20 ^
  --min-train-rows 756

smoke test:
python riskoff_highvol_walkforward_experiment.py --smoke-test

의존성
------
- Python 3.10+
- numpy
- pandas
- scikit-learn

주의
----
- 이 코드는 검증/실험용입니다.
- 결과가 좋더라도 바로 Stable 모델로 채택하면 안 됩니다.
- holdout, after-cost benchmark, ablation을 추가로 통과해야 합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
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


warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 0. 공통 상수 / 유틸
# ============================================================

LABEL_PREFIX = "y_"
FUTURE_PREFIX = "future_"
META_PREFIX = "meta_"


def stable_hash_index(index_like: Sequence[int] | np.ndarray | pd.Index) -> str:
    arr = np.asarray(index_like, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def parse_csv_list(value: str, cast=str) -> List:
    if value is None or str(value).strip() == "":
        return []
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


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
    if hasattr(obj, "__dict__"):
        try:
            return asdict(obj)
        except Exception:
            return str(obj)
    return str(obj)


def save_json(path: str | Path, data: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=to_jsonable),
        encoding="utf-8",
    )
    return path


def save_csv(path: str | Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ============================================================
# 1. 데이터 로딩
# ============================================================

def load_ohlcv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("input CSV must include 'date' column")
    if "close" not in df.columns:
        raise ValueError("input CSV must include 'close' column")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def make_synthetic_ohlcv(n: int = 1800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n, freq="B")

    # regime-like synthetic returns
    vol = np.full(n, 0.012)
    vol[500:650] = 0.025
    vol[1000:1120] = 0.030
    drift = np.full(n, 0.00035)
    drift[500:650] = -0.0008
    drift[1000:1120] = -0.0012

    ret = rng.normal(drift, vol)
    close = 100 * np.cumprod(1.0 + ret)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.000, 0.008, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.000, 0.008, n))
    volume = rng.integers(1_000_000, 8_000_000, n)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


# ============================================================
# 2. FeatureBuilder
# ============================================================

def rolling_mdd(close: pd.Series, window: int) -> pd.Series:
    rolling_max = close.rolling(window, min_periods=max(5, window // 4)).max()
    dd = close / rolling_max - 1.0
    return dd.rolling(window, min_periods=max(5, window // 4)).min()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    기본 feature 생성.

    주의:
    - 미래 정보 사용 금지.
    - 모든 rolling feature는 현재 및 과거 데이터만 사용.
    """
    out = df.copy()
    close = out["close"].astype(float)
    ret = close.pct_change()
    log_ret = np.log(close).diff()

    out["return_1d"] = ret
    out["log_return_1d"] = log_ret

    for w in [5, 10, 20, 40, 60, 120, 252]:
        out[f"return_{w}d"] = close.pct_change(w)
        out[f"rolling_return_mean_{w}d"] = ret.rolling(w, min_periods=max(3, w // 4)).mean()
        out[f"volatility_{w}d"] = ret.rolling(w, min_periods=max(3, w // 4)).std()
        out[f"downside_volatility_{w}d"] = ret.clip(upper=0).rolling(w, min_periods=max(3, w // 4)).std()
        out[f"ma_{w}d"] = close.rolling(w, min_periods=max(3, w // 4)).mean()
        out[f"ma_gap_{w}d"] = close / out[f"ma_{w}d"] - 1.0
        out[f"mdd_{w}d"] = rolling_mdd(close, w)

    # slope: 단순 pct change를 기간으로 나눈 근사
    for w in [20, 60, 120]:
        out[f"price_slope_{w}d"] = close.pct_change(w) / w
        out[f"momentum_{w}d"] = close.pct_change(w)

    if "volume" in out.columns:
        volu = out["volume"].astype(float)
        out["volume_change_20d"] = volu.pct_change(20)
        vol_mean = volu.rolling(20, min_periods=5).mean()
        vol_std = volu.rolling(20, min_periods=5).std()
        out["volume_zscore_20d"] = (volu - vol_mean) / vol_std
        out["volume_ma_gap_20d"] = volu / vol_mean - 1.0
    else:
        out["volume_change_20d"] = np.nan
        out["volume_zscore_20d"] = np.nan
        out["volume_ma_gap_20d"] = np.nan

    # 간단한 trend strength
    out["trend_strength_score"] = (
        (out["ma_gap_20d"] > 0).astype(float)
        + (out["ma_gap_60d"] > 0).astype(float)
        + (out["ma_gap_120d"] > 0).astype(float)
    ) / 3.0

    return out


FEATURE_SETS: Dict[str, List[str]] = {
    "trend_volume": [
        "return_5d", "return_10d", "return_20d", "return_40d", "return_60d", "return_120d",
        "ma_gap_20d", "ma_gap_40d", "ma_gap_60d", "ma_gap_120d", "ma_gap_252d",
        "price_slope_20d", "price_slope_60d", "price_slope_120d",
        "momentum_20d", "momentum_60d", "momentum_120d",
        "trend_strength_score",
        "volume_change_20d", "volume_zscore_20d", "volume_ma_gap_20d",
    ],
    "vol_risk_core": [
        "volatility_5d", "volatility_10d", "volatility_20d", "volatility_40d", "volatility_60d", "volatility_120d",
        "downside_volatility_20d", "downside_volatility_40d", "downside_volatility_60d", "downside_volatility_120d",
        "mdd_20d", "mdd_40d", "mdd_60d", "mdd_120d", "mdd_252d",
        "return_20d", "return_60d", "return_120d",
    ],
    "compact_mixed": [
        "return_5d", "return_20d", "return_60d", "return_120d",
        "volatility_20d", "volatility_60d", "downside_volatility_20d", "downside_volatility_60d",
        "mdd_20d", "mdd_60d", "mdd_120d",
        "ma_gap_20d", "ma_gap_60d", "ma_gap_120d",
        "price_slope_20d", "price_slope_60d",
        "trend_strength_score",
        "volume_change_20d", "volume_zscore_20d",
    ],
    "down_core": [
        "return_5d", "return_20d", "return_60d",
        "downside_volatility_20d", "downside_volatility_60d", "downside_volatility_120d",
        "mdd_20d", "mdd_60d", "mdd_120d", "mdd_252d",
        "volatility_20d", "volatility_60d",
        "ma_gap_20d", "ma_gap_60d",
    ],
}


class LeakageGuard:
    def __init__(self, excluded_prefixes: Tuple[str, ...] = (LABEL_PREFIX, FUTURE_PREFIX, META_PREFIX)):
        self.excluded_prefixes = excluded_prefixes

    def select(self, df: pd.DataFrame, requested_cols: Sequence[str]) -> List[str]:
        cols: List[str] = []
        for c in requested_cols:
            if c not in df.columns:
                continue
            if c.startswith(self.excluded_prefixes):
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            cols.append(c)
        self.assert_no_leakage(cols)
        return cols

    def assert_no_leakage(self, feature_cols: Sequence[str]) -> None:
        leaked = [c for c in feature_cols if c.startswith(self.excluded_prefixes)]
        if leaked:
            raise AssertionError(f"Leakage columns detected: {leaked}")


# ============================================================
# 3. LabelBuilder
# ============================================================

@dataclass
class LabelConfig:
    horizon: int
    vol_window: int = 60
    k_direction: float = 0.8
    k_mdd: float = 1.5
    high_vol_quantile: float = 0.75
    high_vol_lookback: int = 252


def compute_forward_realized_vol(returns: pd.Series, horizon: int) -> pd.Series:
    values = returns.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)

    for i in range(n):
        future = values[i + 1 : i + horizon + 1]
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
        future = values[i + 1 : i + horizon + 1]
        future = future[np.isfinite(future)]
        if len(future) == 0:
            continue
        out[i] = np.min(future / start - 1.0)

    return pd.Series(out, index=close.index)


def build_labels(df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    daily_vol_t = returns.rolling(cfg.vol_window, min_periods=max(10, cfg.vol_window // 3)).std().shift(1)
    current_horizon_vol = daily_vol_t * math.sqrt(cfg.horizon)

    future_return_h = close.shift(-cfg.horizon) / close - 1.0
    future_realized_vol_h = compute_forward_realized_vol(returns, cfg.horizon)
    future_mdd_h = compute_forward_mdd(close, cfg.horizon)

    up_threshold = cfg.k_direction * current_horizon_vol
    down_threshold = -cfg.k_direction * current_horizon_vol
    risk_off_threshold = -cfg.k_mdd * current_horizon_vol

    # high vol threshold는 t 이전 정보 기반의 current_horizon_vol 분포 사용
    high_vol_threshold = current_horizon_vol.rolling(
        cfg.high_vol_lookback,
        min_periods=max(30, cfg.high_vol_lookback // 4),
    ).quantile(cfg.high_vol_quantile)

    y_direction = pd.Series("sideways", index=out.index, dtype="object")
    y_direction = y_direction.mask(future_return_h > up_threshold, "up")
    y_direction = y_direction.mask(future_return_h < down_threshold, "down")

    y_high_vol = (future_realized_vol_h >= high_vol_threshold).astype(float)
    y_risk_off = (future_mdd_h <= risk_off_threshold).astype(float)

    invalid = (
        current_horizon_vol.isna()
        | future_return_h.isna()
        | future_realized_vol_h.isna()
        | future_mdd_h.isna()
        | high_vol_threshold.isna()
    )

    out["future_return_h"] = future_return_h
    out["future_realized_vol_h"] = future_realized_vol_h
    out["future_mdd_h"] = future_mdd_h
    out["meta_current_horizon_vol"] = current_horizon_vol
    out["meta_high_vol_threshold"] = high_vol_threshold

    out["y_direction"] = y_direction.mask(invalid, np.nan)
    out["y_high_vol"] = y_high_vol.mask(invalid, np.nan)
    out["y_risk_off"] = y_risk_off.mask(invalid, np.nan)

    return out


# ============================================================
# 4. Shared Walk-forward Split
# ============================================================

@dataclass
class FoldInfo:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_idx_hash: str
    test_idx_hash: str
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo: int


@dataclass
class SplitConfig:
    test_window: int = 20
    step: int = 20
    min_train_rows: int = 756
    max_train_rows: Optional[int] = None
    embargo: int = 20


class SharedWalkForwardSplitter:
    def __init__(self, config: SplitConfig):
        self.config = config

    def split(self, df: pd.DataFrame) -> List[FoldInfo]:
        n = len(df)
        cfg = self.config
        folds: List[FoldInfo] = []

        test_start = cfg.min_train_rows + cfg.embargo
        fold_id = 0

        while test_start + cfg.test_window <= n:
            train_end = test_start - cfg.embargo
            if cfg.max_train_rows is None:
                train_start = 0
            else:
                train_start = max(0, train_end - cfg.max_train_rows)

            train_idx = np.arange(train_start, train_end, dtype=np.int64)
            test_idx = np.arange(test_start, test_start + cfg.test_window, dtype=np.int64)

            if len(train_idx) >= cfg.min_train_rows:
                folds.append(
                    FoldInfo(
                        fold_id=fold_id,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        train_idx_hash=stable_hash_index(train_idx),
                        test_idx_hash=stable_hash_index(test_idx),
                        train_start=int(train_start),
                        train_end=int(train_end),
                        test_start=int(test_start),
                        test_end=int(test_start + cfg.test_window),
                        embargo=int(cfg.embargo),
                    )
                )
                fold_id += 1

            test_start += cfg.step

        if not folds:
            raise ValueError("No valid folds generated. Check min_train_rows/test_window/embargo.")

        return folds


# ============================================================
# 5. 모델 / Calibration
# ============================================================

def make_model(model_type: str, random_state: int = 42):
    model_type = model_type.lower()

    if model_type == "logistic":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_type == "hgb":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_type == "extratrees":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_type == "randomforest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"unsupported model_type: {model_type}")


class ProbabilityCalibrator:
    """
    Train fold 내부에서 core_train / calibration_train을 시간순 분리해 calibration 수행.

    calibration_method:
    - none
    - sigmoid
    - isotonic

    주의:
    - calibration set에 class가 하나뿐이면 raw probability를 그대로 반환.
    """

    def __init__(self, method: str = "none"):
        self.method = method
        self.model = None

    def fit(self, raw_prob: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        method = self.method.lower()
        raw_prob = np.asarray(raw_prob, dtype=float)
        y_true = np.asarray(y_true, dtype=int)

        mask = np.isfinite(raw_prob) & np.isfinite(y_true)
        raw_prob = np.clip(raw_prob[mask], 1e-8, 1 - 1e-8)
        y_true = y_true[mask]

        if method == "none" or len(np.unique(y_true)) < 2 or len(y_true) < 30:
            self.model = None
            return self

        if method == "sigmoid":
            # Platt-like one-dimensional logistic calibration
            x = np.log(raw_prob / (1.0 - raw_prob)).reshape(-1, 1)
            lr = LogisticRegression(max_iter=1000, solver="lbfgs")
            lr.fit(x, y_true)
            self.model = ("sigmoid", lr)
            return self

        if method == "isotonic":
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

        method, model = self.model
        if method == "sigmoid":
            x = np.log(raw_prob / (1.0 - raw_prob)).reshape(-1, 1)
            return model.predict_proba(x)[:, 1]

        if method == "isotonic":
            return np.clip(model.transform(raw_prob), 0.0, 1.0)

        return raw_prob


# ============================================================
# 6. Metrics
# ============================================================

@dataclass
class CalibrationBin:
    bin_id: int
    count: int
    prob_mean: float
    actual_rate: float
    abs_gap: float


def calibration_ece(y_true: Sequence[float], prob: Sequence[float], n_bins: int = 10) -> Tuple[float, List[CalibrationBin]]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = np.clip(p[mask], 0.0, 1.0)

    if len(y) == 0:
        return float("nan"), []

    edges = np.linspace(0, 1, n_bins + 1)
    bins: List[CalibrationBin] = []
    ece = 0.0

    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (p >= left) & (p <= right)
        else:
            m = (p >= left) & (p < right)

        count = int(m.sum())
        if count == 0:
            bins.append(CalibrationBin(i, 0, np.nan, np.nan, np.nan))
            continue

        prob_mean = float(p[m].mean())
        actual_rate = float(y[m].mean())
        gap = abs(prob_mean - actual_rate)
        ece += count / len(y) * gap
        bins.append(CalibrationBin(i, count, prob_mean, actual_rate, float(gap)))

    return float(ece), bins


def binary_metrics(y_true: Sequence[float], prob: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
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
        roc = float(roc_auc_score(y, p))
        inv_roc = float(roc_auc_score(y, 1 - p))
    except ValueError:
        roc = np.nan
        inv_roc = np.nan

    try:
        pr = float(average_precision_score(y, p))
        inv_pr = float(average_precision_score(y, 1 - p))
    except ValueError:
        pr = np.nan
        inv_pr = np.nan

    ece, _ = calibration_ece(y, p, n_bins=10)
    pred = (p >= threshold).astype(int)

    return {
        "eval_rows": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "positive_rate": positive_rate,
        "roc_auc": roc,
        "inverse_roc_auc": inv_roc,
        "best_roc_after_inversion": max(roc, inv_roc) if np.isfinite(roc) and np.isfinite(inv_roc) else np.nan,
        "probability_polarity": (
            "normal_better" if roc >= inv_roc else "inverse_better"
        ) if np.isfinite(roc) and np.isfinite(inv_roc) else "ambiguous",
        "pr_auc": pr,
        "inverse_pr_auc": inv_pr,
        "pr_gain": pr - positive_rate if np.isfinite(pr) else np.nan,
        "pr_ratio": safe_divide(pr, positive_rate),
        "brier": brier,
        "brier_baseline": brier_baseline,
        "brier_skill": brier_skill,
        "ece": ece,
        "f1_at_0_5": float(f1_score(y, pred, zero_division=0)),
        "prob_mean": float(p.mean()),
        "prob_std": float(p.std()),
    }


def mdd_event_recall(df: pd.DataFrame, prob_col: str, event_col: str = "y_risk_off", q: float = 0.75) -> Dict[str, float]:
    if prob_col not in df.columns or event_col not in df.columns:
        return {}

    clean = df[[prob_col, event_col]].dropna()
    if clean.empty:
        return {}

    threshold = float(clean[prob_col].quantile(q))
    signal = clean[prob_col] >= threshold
    event = clean[event_col] == 1

    event_count = int(event.sum())
    signal_count = int(signal.sum())

    return {
        "mdd_event_signal_quantile": float(q),
        "mdd_event_signal_threshold": threshold,
        "mdd_event_count": event_count,
        "mdd_event_signal_count": signal_count,
        "mdd_event_recall_at_q": safe_divide(int((signal & event).sum()), event_count),
        "mdd_event_precision_at_q": safe_divide(int((signal & event).sum()), signal_count),
    }


# ============================================================
# 7. Experiment Runner
# ============================================================

@dataclass
class TrialConfig:
    ticker: str
    horizon: int
    task: Literal["risk_off", "high_vol"]
    feature_set: str
    model_type: str
    calibration_method: str
    vol_window: int
    k_direction: float
    k_mdd: float
    high_vol_quantile: float
    test_window: int
    step: int
    min_train_rows: int
    max_train_rows: Optional[int]


def get_target_col(task: str) -> str:
    if task == "risk_off":
        return "y_risk_off"
    if task == "high_vol":
        return "y_high_vol"
    raise ValueError(f"unsupported task: {task}")


def split_train_calibration(train_idx: np.ndarray, calibration_frac: float = 0.2, min_cal_rows: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    n = len(train_idx)
    cal_size = max(min_cal_rows, int(n * calibration_frac))
    if n - cal_size < 200:
        # train이 너무 작으면 calibration 분리 생략
        return train_idx, np.array([], dtype=np.int64)
    return train_idx[:-cal_size], train_idx[-cal_size:]


def run_single_trial(
    df_base: pd.DataFrame,
    cfg: TrialConfig,
    random_state: int = 42,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    label_cfg = LabelConfig(
        horizon=cfg.horizon,
        vol_window=cfg.vol_window,
        k_direction=cfg.k_direction,
        k_mdd=cfg.k_mdd,
        high_vol_quantile=cfg.high_vol_quantile,
    )

    labeled = build_labels(df_base, label_cfg)
    target_col = get_target_col(cfg.task)

    guard = LeakageGuard()
    requested_features = FEATURE_SETS[cfg.feature_set]
    feature_cols = guard.select(labeled, requested_features)

    # 라벨과 피처가 모두 있는 행만 사용
    # target_col이 y_risk_off 또는 y_high_vol과 중복될 수 있으므로 컬럼 중복을 제거한다.
    required_cols = list(dict.fromkeys(
        feature_cols + [target_col, "date", "close", "y_risk_off", "y_high_vol", "y_direction"]
    ))
    clean = labeled[required_cols].dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    split_cfg = SplitConfig(
        test_window=cfg.test_window,
        step=cfg.step,
        min_train_rows=cfg.min_train_rows,
        max_train_rows=cfg.max_train_rows,
        embargo=cfg.horizon,
    )
    folds = SharedWalkForwardSplitter(split_cfg).split(clean)

    oof_parts: List[pd.DataFrame] = []
    skipped_folds = 0
    train_rows_list: List[int] = []

    for fold in folds:
        train_idx = fold.train_idx
        test_idx = fold.test_idx

        train_df = clean.iloc[train_idx]
        test_df = clean.iloc[test_idx]

        # train set에 class가 하나뿐이면 skip
        if train_df[target_col].nunique() < 2:
            skipped_folds += 1
            continue

        core_train_idx, cal_idx = split_train_calibration(train_idx)
        core_train = clean.iloc[core_train_idx]
        cal_df = clean.iloc[cal_idx] if len(cal_idx) else pd.DataFrame()

        if core_train[target_col].nunique() < 2:
            skipped_folds += 1
            continue

        model = make_model(cfg.model_type, random_state=random_state)
        model.fit(core_train[feature_cols], core_train[target_col].astype(int))

        raw_test = model.predict_proba(test_df[feature_cols])[:, 1]

        calibrator = ProbabilityCalibrator(cfg.calibration_method)
        if len(cal_df) and cal_df[target_col].nunique() >= 2:
            raw_cal = model.predict_proba(cal_df[feature_cols])[:, 1]
            calibrator.fit(raw_cal, cal_df[target_col].astype(int).to_numpy())
            cal_test = calibrator.transform(raw_test)
        else:
            cal_test = raw_test

        # target_col이 y_risk_off 또는 y_high_vol과 중복되므로 명시적으로 y_true를 따로 만든다.
        part_cols = list(dict.fromkeys(["date", "close", "y_risk_off", "y_high_vol", "y_direction"]))
        part = test_df[part_cols].copy()
        part["y_true"] = test_df[target_col].astype(int).to_numpy()
        part["meta_fold_id"] = fold.fold_id
        part["meta_train_idx_hash"] = fold.train_idx_hash
        part["meta_test_idx_hash"] = fold.test_idx_hash
        part["prob_raw"] = raw_test
        part["prob_cal"] = cal_test
        part["task"] = cfg.task
        part["horizon"] = cfg.horizon
        part["feature_set"] = cfg.feature_set
        part["model_type"] = cfg.model_type
        part["calibration_method"] = cfg.calibration_method

        oof_parts.append(part)
        train_rows_list.append(len(core_train))

    if not oof_parts:
        raise RuntimeError(f"No OOF predictions generated for cfg={cfg}")

    oof = pd.concat(oof_parts, axis=0, ignore_index=True).sort_values("date").reset_index(drop=True)

    raw_metrics = binary_metrics(oof["y_true"], oof["prob_raw"])
    cal_metrics = binary_metrics(oof["y_true"], oof["prob_cal"])

    # task별 주요 probability는 calibrated 기준
    event_metrics = {}
    if cfg.task == "risk_off":
        event_metrics = mdd_event_recall(oof.assign(y_risk_off=oof["y_true"]), prob_col="prob_cal", event_col="y_risk_off", q=0.75)

    # 보수적 score: PR gain, Brier skill, ECE, polarity 반영
    score = (
        1.5 * cal_metrics.get("pr_gain", 0.0)
        + 1.0 * cal_metrics.get("brier_skill", 0.0)
        + 0.5 * max(cal_metrics.get("roc_auc", 0.5) - 0.5, 0.0)
        - 0.5 * max(cal_metrics.get("ece", 0.0), 0.0)
    )

    result = {
        "ticker": cfg.ticker,
        "horizon": cfg.horizon,
        "task": cfg.task,
        "feature_set": cfg.feature_set,
        "model_type": cfg.model_type,
        "calibration_method": cfg.calibration_method,
        "vol_window": cfg.vol_window,
        "k_direction": cfg.k_direction,
        "k_mdd": cfg.k_mdd,
        "high_vol_quantile": cfg.high_vol_quantile,
        "fold_count": int(len(folds)),
        "skipped_folds": int(skipped_folds),
        "avg_train_rows": float(np.mean(train_rows_list)) if train_rows_list else np.nan,
        "feature_count": int(len(feature_cols)),
        "feature_cols": "|".join(feature_cols),
        "eval_rows": int(cal_metrics.get("eval_rows", 0)),
        "positive_count": int(cal_metrics.get("positive_count", 0)),
        "negative_count": int(cal_metrics.get("negative_count", 0)),
        "positive_rate": cal_metrics.get("positive_rate"),
        "raw_roc_auc": raw_metrics.get("roc_auc"),
        "raw_pr_auc": raw_metrics.get("pr_auc"),
        "raw_brier": raw_metrics.get("brier"),
        "raw_brier_skill": raw_metrics.get("brier_skill"),
        "raw_ece": raw_metrics.get("ece"),
        "roc_auc": cal_metrics.get("roc_auc"),
        "inverse_roc_auc": cal_metrics.get("inverse_roc_auc"),
        "best_roc_after_inversion": cal_metrics.get("best_roc_after_inversion"),
        "probability_polarity": cal_metrics.get("probability_polarity"),
        "pr_auc": cal_metrics.get("pr_auc"),
        "pr_gain": cal_metrics.get("pr_gain"),
        "pr_ratio": cal_metrics.get("pr_ratio"),
        "brier": cal_metrics.get("brier"),
        "brier_baseline": cal_metrics.get("brier_baseline"),
        "brier_skill": cal_metrics.get("brier_skill"),
        "ece": cal_metrics.get("ece"),
        "f1_at_0_5": cal_metrics.get("f1_at_0_5"),
        "score": float(score),
        **event_metrics,
    }

    return result, oof


def classify_candidate(row: pd.Series) -> str:
    """
    RiskOff/HighVol 후보 판정.
    """
    pr_gain = float(row.get("pr_gain", np.nan))
    brier_skill = float(row.get("brier_skill", np.nan))
    ece = float(row.get("ece", np.nan))
    polarity = row.get("probability_polarity", "ambiguous")
    roc = float(row.get("roc_auc", np.nan))

    if (
        pr_gain > 0
        and brier_skill > 0
        and ece < 0.10
        and polarity == "normal_better"
        and roc >= 0.55
    ):
        return "holdout_candidate"

    if (
        pr_gain > 0
        and polarity == "normal_better"
        and (brier_skill > -0.05 or ece < 0.20)
    ):
        return "weak_candidate"

    if polarity == "inverse_better":
        return "diagnostic_inverse_signal"

    return "reject_or_diagnostic_only"


def run_experiment(
    df: pd.DataFrame,
    ticker: str,
    horizons: Sequence[int],
    tasks: Sequence[str],
    feature_sets: Sequence[str],
    models: Sequence[str],
    calibration_methods: Sequence[str],
    output_dir: str | Path,
    vol_window: int = 60,
    k_direction: float = 0.8,
    k_mdd: float = 1.5,
    high_vol_quantile: float = 0.75,
    test_window: int = 20,
    step: int = 20,
    min_train_rows: int = 756,
    max_train_rows: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_features = build_features(df)
    trials: List[Dict[str, object]] = []
    best_oof: Optional[pd.DataFrame] = None
    best_score = -np.inf
    best_trial_id = None

    trial_id_num = 0

    for horizon in horizons:
        for task in tasks:
            for feature_set in feature_sets:
                if feature_set not in FEATURE_SETS:
                    raise ValueError(f"unknown feature_set: {feature_set}")
                for model_type in models:
                    for cal_method in calibration_methods:
                        trial_id = f"rh_{trial_id_num:04d}"
                        trial_id_num += 1

                        cfg = TrialConfig(
                            ticker=ticker,
                            horizon=int(horizon),
                            task=task,
                            feature_set=feature_set,
                            model_type=model_type,
                            calibration_method=cal_method,
                            vol_window=vol_window,
                            k_direction=k_direction,
                            k_mdd=k_mdd,
                            high_vol_quantile=high_vol_quantile,
                            test_window=test_window,
                            step=step,
                            min_train_rows=min_train_rows,
                            max_train_rows=max_train_rows,
                        )

                        try:
                            result, oof = run_single_trial(df_features, cfg, random_state=random_state)
                            result["trial_id"] = trial_id
                            result["status"] = "ok"
                        except Exception as e:
                            result = {
                                "trial_id": trial_id,
                                "ticker": ticker,
                                "horizon": horizon,
                                "task": task,
                                "feature_set": feature_set,
                                "model_type": model_type,
                                "calibration_method": cal_method,
                                "status": "error",
                                "error": str(e),
                                "score": -np.inf,
                            }
                            oof = None

                        trials.append(result)

                        score = float(result.get("score", -np.inf))
                        if result.get("status") == "ok" and score > best_score and oof is not None:
                            best_score = score
                            best_oof = oof.copy()
                            best_trial_id = trial_id

    trials_df = pd.DataFrame(trials)
    ok = trials_df[trials_df["status"] == "ok"].copy()

    if not ok.empty:
        ok["candidate_decision"] = ok.apply(classify_candidate, axis=1)
        trials_df = trials_df.merge(
            ok[["trial_id", "candidate_decision"]],
            on="trial_id",
            how="left",
        )
        trials_df["candidate_decision"] = trials_df["candidate_decision"].fillna("error")
        trials_df = trials_df.sort_values("score", ascending=False).reset_index(drop=True)

    top20 = trials_df[trials_df["status"] == "ok"].head(20).copy()

    outputs: Dict[str, Path] = {}
    outputs["trials"] = save_csv(output_dir / "riskoff_highvol_trials.csv", trials_df)
    outputs["top20"] = save_csv(output_dir / "riskoff_highvol_trials_top20.csv", top20)

    if best_oof is not None:
        outputs["best_predictions"] = save_csv(output_dir / "riskoff_highvol_best_predictions.csv", best_oof)

        ece, bins = calibration_ece(best_oof["y_true"], best_oof["prob_cal"])
        outputs["best_calibration_bins"] = save_csv(
            output_dir / "riskoff_highvol_best_calibration_bins.csv",
            pd.DataFrame([asdict(b) for b in bins]),
        )

        yearly = []
        tmp = best_oof.copy()
        tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
        for year, part in tmp.groupby("year"):
            row = binary_metrics(part["y_true"], part["prob_cal"])
            row["year"] = int(year)
            yearly.append(row)
        outputs["best_yearly"] = save_csv(
            output_dir / "riskoff_highvol_best_yearly.csv",
            pd.DataFrame(yearly).sort_values("year") if yearly else pd.DataFrame(),
        )

    # group summaries
    group_cols = ["task", "horizon", "feature_set", "model_type", "calibration_method"]
    group_summary = []
    if not ok.empty:
        for col in group_cols:
            for key, part in ok.groupby(col):
                group_summary.append(
                    {
                        "group_col": col,
                        "group_value": key,
                        "trial_count": int(len(part)),
                        "mean_pr_gain": float(part["pr_gain"].mean()),
                        "max_pr_gain": float(part["pr_gain"].max()),
                        "mean_pr_ratio": float(part["pr_ratio"].mean()),
                        "mean_brier_skill": float(part["brier_skill"].mean()),
                        "positive_brier_skill_rate": float((part["brier_skill"] > 0).mean()),
                        "mean_ece": float(part["ece"].mean()),
                        "mean_roc_auc": float(part["roc_auc"].mean()),
                        "max_roc_auc": float(part["roc_auc"].max()),
                        "holdout_candidate_count": int((part["candidate_decision"] == "holdout_candidate").sum()),
                        "weak_candidate_count": int((part["candidate_decision"] == "weak_candidate").sum()),
                    }
                )
    outputs["group_summary"] = save_csv(output_dir / "riskoff_highvol_group_summary.csv", pd.DataFrame(group_summary))

    best_row = {}
    if best_trial_id is not None and not trials_df.empty:
        match = trials_df[trials_df["trial_id"] == best_trial_id]
        if not match.empty:
            best_row = match.iloc[0].to_dict()

    summary = {
        "ticker": ticker,
        "input_rows": int(len(df)),
        "trial_count": int(len(trials_df)),
        "ok_trial_count": int((trials_df["status"] == "ok").sum()),
        "error_trial_count": int((trials_df["status"] == "error").sum()),
        "best_trial_id": best_trial_id,
        "best_trial": best_row,
        "candidate_counts": trials_df["candidate_decision"].value_counts(dropna=False).to_dict() if "candidate_decision" in trials_df.columns else {},
        "config": {
            "horizons": list(map(int, horizons)),
            "tasks": list(tasks),
            "feature_sets": list(feature_sets),
            "models": list(models),
            "calibration_methods": list(calibration_methods),
            "vol_window": vol_window,
            "k_direction": k_direction,
            "k_mdd": k_mdd,
            "high_vol_quantile": high_vol_quantile,
            "test_window": test_window,
            "step": step,
            "min_train_rows": min_train_rows,
            "max_train_rows": max_train_rows,
        },
        "interpretation": {
            "holdout_candidate": "후속 holdout 검증 대상으로 올릴 수 있는 후보",
            "weak_candidate": "보조 후보. Stable 채택 불가",
            "diagnostic_inverse_signal": "신호 방향 반전/라벨 매핑/폴드 안정성 점검 대상",
            "reject_or_diagnostic_only": "구조 채택 후보로 부적합",
        },
        "do_not_do": [
            "Do not promote any candidate to stable without holdout and after-cost benchmark.",
            "Do not use raw probability directly for AllocationService.",
            "Do not interpret PR-AUC without positive_rate.",
        ],
    }
    outputs["summary"] = save_json(output_dir / "riskoff_highvol_summary.json", summary)

    return outputs


# ============================================================
# 8. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="OHLCV CSV path with date, close columns")
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--output-dir", default="riskoff_highvol_results")

    parser.add_argument("--horizons", default="10,20,40,60,120")
    parser.add_argument("--tasks", default="risk_off,high_vol")
    parser.add_argument("--feature-sets", default="compact_mixed,vol_risk_core,trend_volume,down_core")
    parser.add_argument("--models", default="logistic,hgb,extratrees")
    parser.add_argument("--calibration-methods", default="none,sigmoid")

    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--k-direction", type=float, default=0.8)
    parser.add_argument("--k-mdd", type=float, default=1.5)
    parser.add_argument("--high-vol-quantile", type=float, default=0.75)

    parser.add_argument("--test-window", type=int, default=20)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--min-train-rows", type=int, default=756)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        df = make_synthetic_ohlcv(n=1200)
        output_dir = Path(args.output_dir)
        # smoke는 빠른 정상 동작 확인용이므로 조합을 최소화한다.
        horizons = [20]
        tasks = ["risk_off", "high_vol"]
        feature_sets = ["compact_mixed"]
        models = ["logistic"]
        calibration_methods = ["none", "sigmoid"]
        min_train_rows = 500
        args.test_window = 60
        args.step = 60
    else:
        if not args.input:
            raise ValueError("--input is required unless --smoke-test is used")
        df = load_ohlcv(args.input)
        output_dir = Path(args.output_dir)
        horizons = parse_csv_list(args.horizons, int)
        tasks = parse_csv_list(args.tasks, str)
        feature_sets = parse_csv_list(args.feature_sets, str)
        models = parse_csv_list(args.models, str)
        calibration_methods = parse_csv_list(args.calibration_methods, str)
        min_train_rows = args.min_train_rows

    outputs = run_experiment(
        df=df,
        ticker=args.ticker,
        horizons=horizons,
        tasks=tasks,
        feature_sets=feature_sets,
        models=models,
        calibration_methods=calibration_methods,
        output_dir=output_dir,
        vol_window=args.vol_window,
        k_direction=args.k_direction,
        k_mdd=args.k_mdd,
        high_vol_quantile=args.high_vol_quantile,
        test_window=args.test_window,
        step=args.step,
        min_train_rows=min_train_rows,
        max_train_rows=args.max_train_rows,
        random_state=args.random_state,
    )

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))

    print("[OK] RiskOff/HighVol experiment completed.")
    print(f"[OK] Output dir: {Path(output_dir).resolve()}")
    print(json.dumps(
        {
            "trial_count": summary["trial_count"],
            "ok_trial_count": summary["ok_trial_count"],
            "error_trial_count": summary["error_trial_count"],
            "best_trial_id": summary["best_trial_id"],
            "best_trial_task": summary["best_trial"].get("task"),
            "best_trial_horizon": summary["best_trial"].get("horizon"),
            "best_trial_model": summary["best_trial"].get("model_type"),
            "best_trial_calibration": summary["best_trial"].get("calibration_method"),
            "best_trial_pr_auc": summary["best_trial"].get("pr_auc"),
            "best_trial_positive_rate": summary["best_trial"].get("positive_rate"),
            "best_trial_pr_gain": summary["best_trial"].get("pr_gain"),
            "best_trial_brier_skill": summary["best_trial"].get("brier_skill"),
            "best_trial_ece": summary["best_trial"].get("ece"),
            "candidate_counts": summary["candidate_counts"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
