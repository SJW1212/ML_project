# -*- coding: utf-8 -*-
"""
rolling_highvol_improvement_experiment.py

HighVol Label + Persistence + Softer Allocation 개선 실험 코드.

목적
----
기존 rolling HighVol 검증에서 다음 문제가 확인되었습니다.

1. H20 단일 HighVol label의 fold 안정성 부족
2. inverse polarity fold 비율 높음
3. median PR-AUC 낮음
4. 확률값 자체의 calibration 품질 부족
5. HighVol only는 후보지만 Stable 채택 불가

따라서 본 코드는 다음 개선 축을 rolling OOS로 비교합니다.

실험 축
-------
Label mode:
1. h20_current
   - 기존 H20 future realized volatility label

2. h10_h20_ensemble
   - H10 또는 H20 중 하나라도 high_vol이면 1

3. vol_expansion_ratio
   - future_realized_vol_h / current_horizon_vol >= expansion_mult

Persistence:
1. none
2. 2of3
3. 3of5

Allocation:
1. equity60_cash40
2. equity70_cash30
3. equity80_cash20
4. equity80_bond20
5. equity70_bond20_cash10

Threshold:
1. q=0.75
2. q=0.80

Benchmark:
1. buy_hold
2. constant_normal
3. sixty_forty, bond CSV 제공 시

Leakage 방지
------------
1. rolling fold: [core train][calibration][embargo][test]
2. threshold는 calibration window에서만 산출
3. test window에는 고정 threshold 적용
4. 신호는 1거래일 shift 후 수익률에 적용
5. label/future/meta prefix feature 자동 제외

실행 예시
--------
python rolling_highvol_improvement_experiment.py ^
  --equity-input QQQ_ohlcv.csv ^
  --bond-input IEF_ohlcv.csv ^
  --ticker QQQ ^
  --bond-ticker IEF ^
  --output-dir rolling_highvol_improvement_results ^
  --train-window 1260 ^
  --calibration-window 252 ^
  --test-window 63 ^
  --threshold-quantiles 0.75,0.80 ^
  --persistence-modes none,2of3,3of5 ^
  --allocation-modes equity60_cash40,equity70_cash30,equity80_cash20,equity80_bond20,equity70_bond20_cash10 ^
  --transaction-cost-bps 10

빠른 테스트:
python rolling_highvol_improvement_experiment.py --smoke-test

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
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 0. 상수 / 유틸
# ============================================================

LABEL_PREFIX = "y_"
FUTURE_PREFIX = "future_"
META_PREFIX = "meta_"


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


# ============================================================
# 1. 데이터
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


def make_synthetic_ohlcv(n: int = 1400, seed: int = 42, ticker: str = "QQQ") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n)

    vol = np.full(n, 0.011)
    drift = np.full(n, 0.00035)

    for start, end, local_vol, local_drift in [
        (300, 390, 0.028, -0.0010),
        (720, 790, 0.030, -0.0011),
        (1050, 1130, 0.026, -0.0007),
    ]:
        vol[start:end] = local_vol
        drift[start:end] = local_drift

    ret = rng.normal(drift, vol)
    close = 100 * np.cumprod(1 + ret)
    volume = rng.integers(1_000_000, 10_000_000, n)
    return pd.DataFrame({"date": dates, "ticker": ticker, "close": close, "volume": volume})


# ============================================================
# 2. Feature Builder
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
    "vol_risk_core": [
        "volatility_5d", "volatility_10d", "volatility_20d", "volatility_40d", "volatility_60d", "volatility_120d",
        "downside_volatility_20d", "downside_volatility_40d", "downside_volatility_60d", "downside_volatility_120d",
        "mdd_20d", "mdd_40d", "mdd_60d", "mdd_120d", "mdd_252d",
        "return_20d", "return_60d", "return_120d",
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
# 3. Label Builder
# ============================================================

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


def build_single_highvol_label(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int,
    high_vol_quantile: float,
    high_vol_lookback: int,
) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    daily_vol_t = returns.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().shift(1)
    current_horizon_vol = daily_vol_t * math.sqrt(horizon)
    future_realized_vol_h = compute_forward_realized_vol(returns, horizon)

    high_vol_threshold = current_horizon_vol.rolling(
        high_vol_lookback,
        min_periods=max(30, high_vol_lookback // 4),
    ).quantile(high_vol_quantile)

    y_high_vol = (future_realized_vol_h >= high_vol_threshold).astype(float)
    invalid = current_horizon_vol.isna() | future_realized_vol_h.isna() | high_vol_threshold.isna()

    out[f"future_realized_vol_h{horizon}"] = future_realized_vol_h
    out[f"meta_current_horizon_vol_h{horizon}"] = current_horizon_vol
    out[f"meta_high_vol_threshold_h{horizon}"] = high_vol_threshold
    out[f"y_high_vol_h{horizon}"] = y_high_vol.mask(invalid, np.nan)

    return out


def build_vol_expansion_label(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int,
    expansion_mult: float,
) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    daily_vol_t = returns.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().shift(1)
    current_horizon_vol = daily_vol_t * math.sqrt(horizon)
    future_realized_vol_h = compute_forward_realized_vol(returns, horizon)

    ratio = future_realized_vol_h / current_horizon_vol
    y = (ratio >= expansion_mult).astype(float)
    invalid = current_horizon_vol.isna() | future_realized_vol_h.isna() | ratio.isna()

    out[f"future_realized_vol_h{horizon}"] = future_realized_vol_h
    out[f"meta_current_horizon_vol_h{horizon}"] = current_horizon_vol
    out[f"future_vol_expansion_ratio_h{horizon}"] = ratio
    out[f"y_vol_expansion_h{horizon}"] = y.mask(invalid, np.nan)
    return out


def build_target_label(
    df: pd.DataFrame,
    label_mode: str,
    horizon: int,
    vol_window: int,
    high_vol_quantile: float,
    high_vol_lookback: int,
    expansion_mult: float,
) -> pd.DataFrame:
    """
    최종 target 컬럼 이름은 항상 y_target으로 통일.
    """
    out = df.copy()

    if label_mode == "h20_current":
        out = build_single_highvol_label(
            out,
            horizon=20,
            vol_window=vol_window,
            high_vol_quantile=high_vol_quantile,
            high_vol_lookback=high_vol_lookback,
        )
        out["y_target"] = out["y_high_vol_h20"]

    elif label_mode == "h10_h20_ensemble":
        out = build_single_highvol_label(
            out,
            horizon=10,
            vol_window=vol_window,
            high_vol_quantile=high_vol_quantile,
            high_vol_lookback=high_vol_lookback,
        )
        out = build_single_highvol_label(
            out,
            horizon=20,
            vol_window=vol_window,
            high_vol_quantile=high_vol_quantile,
            high_vol_lookback=high_vol_lookback,
        )
        y10 = out["y_high_vol_h10"]
        y20 = out["y_high_vol_h20"]
        valid = y10.notna() & y20.notna()
        out["y_target"] = np.where(valid, ((y10 == 1) | (y20 == 1)).astype(float), np.nan)

    elif label_mode == "vol_expansion_ratio":
        out = build_vol_expansion_label(
            out,
            horizon=horizon,
            vol_window=vol_window,
            expansion_mult=expansion_mult,
        )
        out["y_target"] = out[f"y_vol_expansion_h{horizon}"]

    else:
        raise ValueError(f"unsupported label_mode: {label_mode}")

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


# ============================================================
# 5. Rolling folds
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
        raise ValueError(
            "No rolling folds generated. Reduce train_window/calibration_window/test_window or provide longer data."
        )

    return folds


# ============================================================
# 6. Metrics
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
        auc = float(roc_auc_score(y, p))
        inv_auc = float(roc_auc_score(y, 1 - p))
    except ValueError:
        auc = np.nan
        inv_auc = np.nan

    try:
        pr = float(average_precision_score(y, p))
    except ValueError:
        pr = np.nan

    pred = (p >= threshold).astype(int)

    return {
        "eval_rows": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "positive_rate": positive_rate,
        "roc_auc": auc,
        "inverse_roc_auc": inv_auc,
        "probability_polarity": (
            "normal_better"
            if np.isfinite(auc) and np.isfinite(inv_auc) and auc >= inv_auc
            else "inverse_better"
        ),
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

    running_max = curve.cummax()
    dd = curve / running_max - 1.0
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
# 7. Experiment configs
# ============================================================

@dataclass
class ExperimentConfig:
    experiment_id: str
    label_mode: str
    feature_set: str
    horizon: int
    expansion_mult: float
    threshold_quantile: float
    persistence_mode: str
    allocation_mode: str


def build_experiment_grid(
    label_modes: Sequence[str],
    feature_sets: Sequence[str],
    threshold_quantiles: Sequence[float],
    persistence_modes: Sequence[str],
    allocation_modes: Sequence[str],
    expansion_mults: Sequence[float],
) -> List[ExperimentConfig]:
    configs: List[ExperimentConfig] = []

    for label_mode in label_modes:
        for feature_set in feature_sets:
            for q in threshold_quantiles:
                for persistence_mode in persistence_modes:
                    for allocation_mode in allocation_modes:
                        # vol_expansion_ratio만 expansion_mult sweep 적용
                        if label_mode == "vol_expansion_ratio":
                            for em in expansion_mults:
                                configs.append(
                                    ExperimentConfig(
                                        experiment_id=(
                                            f"{label_mode}_{feature_set}_em{em}_q{q}_"
                                            f"{persistence_mode}_{allocation_mode}"
                                        ).replace(".", "p"),
                                        label_mode=label_mode,
                                        feature_set=feature_set,
                                        horizon=20,
                                        expansion_mult=em,
                                        threshold_quantile=float(q),
                                        persistence_mode=persistence_mode,
                                        allocation_mode=allocation_mode,
                                    )
                                )
                        else:
                            configs.append(
                                ExperimentConfig(
                                    experiment_id=(
                                        f"{label_mode}_{feature_set}_q{q}_"
                                        f"{persistence_mode}_{allocation_mode}"
                                    ).replace(".", "p"),
                                    label_mode=label_mode,
                                    feature_set=feature_set,
                                    horizon=20,
                                    expansion_mult=np.nan,
                                    threshold_quantile=float(q),
                                    persistence_mode=persistence_mode,
                                    allocation_mode=allocation_mode,
                                )
                            )

    return configs


# ============================================================
# 8. Rolling predictions by label config
# ============================================================

def run_rolling_predictions_for_label(
    df: pd.DataFrame,
    label_mode: str,
    feature_set: str,
    horizon: int,
    vol_window: int,
    high_vol_quantile: float,
    high_vol_lookback: int,
    expansion_mult: float,
    train_window: int,
    calibration_window: int,
    test_window: int,
    threshold_quantiles: Sequence[float],
    calibration_method: str,
    random_state: int,
    n_estimators: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    featured = build_features(df)
    labeled = build_target_label(
        featured,
        label_mode=label_mode,
        horizon=horizon,
        vol_window=vol_window,
        high_vol_quantile=high_vol_quantile,
        high_vol_lookback=high_vol_lookback,
        expansion_mult=expansion_mult,
    )

    feature_cols = select_features(labeled, feature_set)
    required = ["date", "close", "y_target"] + feature_cols
    data = labeled[required].dropna(subset=["y_target"] + feature_cols).reset_index(drop=True)

    folds = build_rolling_folds(
        data,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        embargo=horizon,
        step=test_window,
    )

    prediction_parts = []
    fold_rows = []
    threshold_rows = []

    for fold in folds:
        core = data.iloc[fold.core_start:fold.core_end].copy()
        cal = data.iloc[fold.cal_start:fold.cal_end].copy()
        test = data.iloc[fold.test_start:fold.test_end].copy()

        try:
            if core["y_target"].nunique() < 2:
                raise ValueError("core train has one class")

            model = make_extratrees(random_state=random_state + fold.fold_id, n_estimators=n_estimators)
            model.fit(core[feature_cols], core["y_target"].astype(int))

            raw_cal = model.predict_proba(cal[feature_cols])[:, 1]
            raw_test = model.predict_proba(test[feature_cols])[:, 1]

            calibrator = ProbabilityCalibrator(method=calibration_method)
            if cal["y_target"].nunique() >= 2:
                calibrator.fit(raw_cal, cal["y_target"].astype(int).to_numpy())
                cal_prob = calibrator.transform(raw_cal)
                test_prob = calibrator.transform(raw_test)
                calibration_status = "calibrated"
            else:
                cal_prob = raw_cal
                test_prob = raw_test
                calibration_status = "skipped_single_class_calibration_set"

            pred = test[["date", "close", "y_target"]].copy().rename(columns={"y_target": "y_true"})
            pred["fold_id"] = fold.fold_id
            pred["label_mode"] = label_mode
            pred["feature_set"] = feature_set
            pred["horizon"] = horizon
            pred["expansion_mult"] = expansion_mult
            pred["prob_raw"] = raw_test
            pred["prob_cal"] = test_prob
            pred["calibration_status"] = calibration_status
            pred["test_start_date"] = fold.test_start_date
            pred["test_end_date"] = fold.test_end_date

            for q in threshold_quantiles:
                th = float(np.quantile(cal_prob, q))
                pred[f"threshold_q{q:.2f}"] = th
                pred[f"signal_q{q:.2f}"] = (pred["prob_cal"] >= th).astype(int)

                threshold_rows.append({
                    "label_mode": label_mode,
                    "feature_set": feature_set,
                    "horizon": horizon,
                    "expansion_mult": expansion_mult,
                    "fold_id": fold.fold_id,
                    "threshold_quantile": float(q),
                    "threshold_source": "rolling_calibration_window_only",
                    "threshold": th,
                    "cal_signal_rate": float(np.mean(cal_prob >= th)),
                    "test_signal_rate": float(np.mean(test_prob >= th)),
                    "core_positive_rate": float(core["y_target"].mean()),
                    "cal_positive_rate": float(cal["y_target"].mean()),
                    "test_positive_rate": float(test["y_target"].mean()),
                    **asdict(fold),
                })

            metrics = binary_metrics(pred["y_true"], pred["prob_cal"])
            fold_rows.append({
                "label_mode": label_mode,
                "feature_set": feature_set,
                "horizon": horizon,
                "expansion_mult": expansion_mult,
                "fold_id": fold.fold_id,
                "status": "ok",
                "calibration_status": calibration_status,
                "core_rows": int(len(core)),
                "calibration_rows": int(len(cal)),
                "test_rows": int(len(test)),
                "core_positive_rate": float(core["y_target"].mean()),
                "cal_positive_rate": float(cal["y_target"].mean()),
                "test_positive_rate": float(test["y_target"].mean()),
                **metrics,
                **asdict(fold),
            })

            prediction_parts.append(pred)

        except Exception as e:
            fold_rows.append({
                "label_mode": label_mode,
                "feature_set": feature_set,
                "horizon": horizon,
                "expansion_mult": expansion_mult,
                "fold_id": fold.fold_id,
                "status": "error",
                "error": str(e),
                **asdict(fold),
            })

    predictions = pd.concat(prediction_parts, axis=0, ignore_index=True) if prediction_parts else pd.DataFrame()
    fold_metrics = pd.DataFrame(fold_rows)
    thresholds = pd.DataFrame(threshold_rows)

    if predictions.empty:
        raise RuntimeError(f"No valid predictions for label_mode={label_mode}, feature_set={feature_set}")

    return predictions, fold_metrics, thresholds


# ============================================================
# 9. Persistence / Allocation
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


def allocation_weights_from_mode(mode: str, highvol: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    equity_w = np.ones(n)
    bond_w = np.zeros(n)
    cash_w = np.zeros(n)

    if mode == "equity60_cash40":
        equity_w[highvol] = 0.60
        cash_w[highvol] = 0.40

    elif mode == "equity70_cash30":
        equity_w[highvol] = 0.70
        cash_w[highvol] = 0.30

    elif mode == "equity80_cash20":
        equity_w[highvol] = 0.80
        cash_w[highvol] = 0.20

    elif mode == "equity80_bond20":
        equity_w[highvol] = 0.80
        bond_w[highvol] = 0.20

    elif mode == "equity70_bond20_cash10":
        equity_w[highvol] = 0.70
        bond_w[highvol] = 0.20
        cash_w[highvol] = 0.10

    else:
        raise ValueError(f"unsupported allocation mode: {mode}")

    total = equity_w + bond_w + cash_w
    return equity_w / total, bond_w / total, cash_w / total


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


def simulate_benchmark(
    ret_df: pd.DataFrame,
    strategy: str,
    transaction_cost_bps: float,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    df = ret_df.copy()
    n = len(df)

    if strategy == "buy_hold":
        equity_w = np.ones(n)
        bond_w = np.zeros(n)
        cash_w = np.zeros(n)
    elif strategy == "constant_normal":
        equity_w = np.full(n, 0.80)
        bond_w = np.full(n, 0.10)
        cash_w = np.full(n, 0.10)
    elif strategy == "sixty_forty":
        equity_w = np.full(n, 0.60)
        bond_w = np.full(n, 0.40)
        cash_w = np.zeros(n)
    else:
        raise ValueError(f"unknown benchmark: {strategy}")

    turnover = np.zeros(n)
    cost = turnover * (transaction_cost_bps / 10000.0)

    net_ret = (
        equity_w * df["equity_ret"].to_numpy()
        + bond_w * df["bond_ret"].to_numpy()
        + cash_w * df["cash_ret"].to_numpy()
        - cost
    )
    curve = np.cumprod(1 + net_ret)

    daily = df.copy()
    daily["strategy"] = strategy
    daily["experiment_id"] = strategy
    daily["threshold_quantile"] = np.nan
    daily["persistence_mode"] = "none"
    daily["allocation_mode"] = strategy
    daily["equity_weight"] = equity_w
    daily["bond_weight"] = bond_w
    daily["cash_weight"] = cash_w
    daily["turnover"] = turnover
    daily["cost"] = cost
    daily["strategy_ret"] = net_ret
    daily["equity_curve"] = curve

    summary = {
        "experiment_id": strategy,
        "strategy": strategy,
        "label_mode": "benchmark",
        "feature_set": "benchmark",
        "threshold_quantile": np.nan,
        "persistence_mode": "none",
        "allocation_mode": strategy,
        "rows": int(n),
        "avg_equity_weight": float(np.mean(equity_w)),
        "avg_bond_weight": float(np.mean(bond_w)),
        "avg_cash_weight": float(np.mean(cash_w)),
        "turnover_total": float(np.sum(turnover)),
        "transaction_cost_total": float(np.sum(cost)),
        **performance_metrics(curve, net_ret),
    }

    return summary, daily


def simulate_experiment_strategy(
    ret_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cfg: ExperimentConfig,
    transaction_cost_bps: float,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    df = ret_df.copy()

    signal_col = f"signal_q{cfg.threshold_quantile:.2f}"
    threshold_col = f"threshold_q{cfg.threshold_quantile:.2f}"

    sig = pred_df[["date", "prob_cal", "fold_id", signal_col, threshold_col]].copy()
    df = df.merge(sig, on="date", how="left")

    # rolling test window 사이 비는 값 보완
    df["prob_cal"] = df["prob_cal"].ffill()
    df[signal_col] = df[signal_col].fillna(0).astype(int)
    df[threshold_col] = df[threshold_col].ffill()

    # persistence 적용 후 1거래일 지연
    persisted = apply_persistence(df[signal_col], cfg.persistence_mode)
    executed_signal = persisted.shift(1).fillna(0).astype(int)

    highvol = executed_signal.to_numpy() == 1
    n = len(df)
    equity_w, bond_w, cash_w = allocation_weights_from_mode(cfg.allocation_mode, highvol, n)

    turnover = np.zeros(n)
    turnover[1:] = np.abs(np.diff(equity_w)) + np.abs(np.diff(bond_w)) + np.abs(np.diff(cash_w))
    cost = turnover * (transaction_cost_bps / 10000.0)

    net_ret = (
        equity_w * df["equity_ret"].to_numpy()
        + bond_w * df["bond_ret"].to_numpy()
        + cash_w * df["cash_ret"].to_numpy()
        - cost
    )
    curve = np.cumprod(1 + net_ret)

    daily = df.copy()
    daily["strategy"] = "highvol_improved"
    daily["experiment_id"] = cfg.experiment_id
    daily["label_mode"] = cfg.label_mode
    daily["feature_set"] = cfg.feature_set
    daily["threshold_quantile"] = cfg.threshold_quantile
    daily["persistence_mode"] = cfg.persistence_mode
    daily["allocation_mode"] = cfg.allocation_mode
    daily["raw_signal"] = df[signal_col].astype(int)
    daily["persisted_signal"] = persisted.astype(int)
    daily["executed_signal"] = executed_signal.astype(int)
    daily["equity_weight"] = equity_w
    daily["bond_weight"] = bond_w
    daily["cash_weight"] = cash_w
    daily["turnover"] = turnover
    daily["cost"] = cost
    daily["strategy_ret"] = net_ret
    daily["equity_curve"] = curve

    summary = {
        "experiment_id": cfg.experiment_id,
        "strategy": "highvol_improved",
        "label_mode": cfg.label_mode,
        "feature_set": cfg.feature_set,
        "horizon": cfg.horizon,
        "expansion_mult": cfg.expansion_mult,
        "threshold_quantile": cfg.threshold_quantile,
        "persistence_mode": cfg.persistence_mode,
        "allocation_mode": cfg.allocation_mode,
        "rows": int(n),
        "raw_signal_rate": float(daily["raw_signal"].mean()),
        "persisted_signal_rate": float(daily["persisted_signal"].mean()),
        "executed_signal_rate": float(daily["executed_signal"].mean()),
        "avg_equity_weight": float(np.mean(equity_w)),
        "avg_bond_weight": float(np.mean(bond_w)),
        "avg_cash_weight": float(np.mean(cash_w)),
        "turnover_total": float(np.sum(turnover)),
        "transaction_cost_total": float(np.sum(cost)),
        **performance_metrics(curve, net_ret),
    }

    return summary, daily


# ============================================================
# 10. Experiment runner
# ============================================================

def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> Dict[str, float]:
    ok = fold_metrics[fold_metrics["status"] == "ok"].copy()
    if ok.empty:
        return {}

    return {
        "fold_count": int(len(fold_metrics)),
        "ok_fold_count": int(len(ok)),
        "mean_pr_auc": float(ok["pr_auc"].mean()) if "pr_auc" in ok.columns else np.nan,
        "median_pr_auc": float(ok["pr_auc"].median()) if "pr_auc" in ok.columns else np.nan,
        "mean_brier_skill": float(ok["brier_skill"].mean()) if "brier_skill" in ok.columns else np.nan,
        "median_brier_skill": float(ok["brier_skill"].median()) if "brier_skill" in ok.columns else np.nan,
        "positive_brier_skill_rate": float((ok["brier_skill"] > 0).mean()) if "brier_skill" in ok.columns else np.nan,
        "normal_polarity_rate": float((ok["probability_polarity"] == "normal_better").mean()) if "probability_polarity" in ok.columns else np.nan,
        "extreme_imbalance_rate": float(((ok["test_positive_rate"] <= 0.05) | (ok["test_positive_rate"] >= 0.95)).mean())
            if "test_positive_rate" in ok.columns else np.nan,
    }


def run_experiment(
    equity_df: pd.DataFrame,
    bond_df: Optional[pd.DataFrame],
    output_dir: str | Path,
    ticker: str,
    bond_ticker: Optional[str],
    label_modes: Sequence[str],
    feature_sets: Sequence[str],
    threshold_quantiles: Sequence[float],
    persistence_modes: Sequence[str],
    allocation_modes: Sequence[str],
    expansion_mults: Sequence[float],
    vol_window: int,
    high_vol_quantile: float,
    high_vol_lookback: int,
    train_window: int,
    calibration_window: int,
    test_window: int,
    calibration_method: str,
    transaction_cost_bps: float,
    random_state: int,
    n_estimators: int,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = build_experiment_grid(
        label_modes=label_modes,
        feature_sets=feature_sets,
        threshold_quantiles=threshold_quantiles,
        persistence_modes=persistence_modes,
        allocation_modes=allocation_modes,
        expansion_mults=expansion_mults,
    )

    # label-feature-expansion별 prediction은 1번만 생성하고 strategy 조합에 재사용
    pred_cache: Dict[Tuple[str, str, float], pd.DataFrame] = {}
    fold_cache: Dict[Tuple[str, str, float], pd.DataFrame] = {}
    threshold_cache: Dict[Tuple[str, str, float], pd.DataFrame] = {}

    all_fold_metrics = []
    all_thresholds = []
    all_predictions_sample = []

    summary_rows = []
    daily_parts = []

    # 각 label-feature-expansion 조합별 rolling prediction 생성
    unique_keys = sorted(set((cfg.label_mode, cfg.feature_set, cfg.expansion_mult) for cfg in grid), key=str)

    for idx, (label_mode, feature_set, expansion_mult) in enumerate(unique_keys):
        preds, folds, ths = run_rolling_predictions_for_label(
            df=equity_df,
            label_mode=label_mode,
            feature_set=feature_set,
            horizon=20,
            vol_window=vol_window,
            high_vol_quantile=high_vol_quantile,
            high_vol_lookback=high_vol_lookback,
            expansion_mult=expansion_mult,
            train_window=train_window,
            calibration_window=calibration_window,
            test_window=test_window,
            threshold_quantiles=threshold_quantiles,
            calibration_method=calibration_method,
            random_state=random_state + idx * 100,
            n_estimators=n_estimators,
        )

        key = (label_mode, feature_set, expansion_mult)
        pred_cache[key] = preds
        fold_cache[key] = folds
        threshold_cache[key] = ths

        all_fold_metrics.append(folds)
        all_thresholds.append(ths)

        # 너무 큰 파일 방지: 예측은 전체 저장하되 label별로 concat
        all_predictions_sample.append(preds)

    # OOS date range는 첫 prediction 기준으로 통일
    all_pred_concat = pd.concat(all_predictions_sample, axis=0, ignore_index=True)
    start_date = pd.to_datetime(all_pred_concat["date"]).min()
    end_date = pd.to_datetime(all_pred_concat["date"]).max()
    ret_df = align_returns(equity_df, bond_df, start_date, end_date)

    # benchmark
    benchmark_strategies = ["buy_hold", "constant_normal"]
    if bond_df is not None:
        benchmark_strategies.append("sixty_forty")

    for bench in benchmark_strategies:
        row, daily = simulate_benchmark(ret_df, bench, transaction_cost_bps)
        summary_rows.append(row)
        daily_parts.append(daily)

    # strategy grid
    for cfg in grid:
        key = (cfg.label_mode, cfg.feature_set, cfg.expansion_mult)
        preds = pred_cache[key]
        row, daily = simulate_experiment_strategy(
            ret_df=ret_df,
            pred_df=preds,
            cfg=cfg,
            transaction_cost_bps=transaction_cost_bps,
        )

        # classification quality summary 연결
        fold_summary = summarize_fold_metrics(fold_cache[key])
        for k, v in fold_summary.items():
            row[f"classifier_{k}"] = v

        # Buy & Hold와 비교는 나중에 채움
        summary_rows.append(row)
        daily_parts.append(daily)

    strategy_summary = pd.DataFrame(summary_rows)

    # benchmark relative columns
    bh = strategy_summary[strategy_summary["strategy"] == "buy_hold"]
    if not bh.empty:
        bh_row = bh.iloc[0]
        for m in ["cagr", "mdd", "calmar", "sharpe", "volatility", "total_return"]:
            if m in strategy_summary.columns:
                strategy_summary[f"{m}_diff_vs_buy_hold"] = strategy_summary[m] - bh_row[m]

    cn = strategy_summary[strategy_summary["strategy"] == "constant_normal"]
    if not cn.empty:
        cn_row = cn.iloc[0]
        for m in ["cagr", "mdd", "calmar", "sharpe", "volatility", "total_return"]:
            if m in strategy_summary.columns:
                strategy_summary[f"{m}_diff_vs_constant_normal"] = strategy_summary[m] - cn_row[m]

    # 채택 점수: Calmar 개선 + MDD 개선 + CAGR 훼손 제한 + fold 안정성
    strategy_summary["candidate_score"] = (
        strategy_summary.get("calmar_diff_vs_buy_hold", 0).fillna(0) * 2.0
        + strategy_summary.get("mdd_diff_vs_buy_hold", 0).fillna(0) * 1.5
        + strategy_summary.get("cagr_diff_vs_buy_hold", 0).fillna(0) * 1.0
        + strategy_summary.get("classifier_positive_brier_skill_rate", 0).fillna(0) * 0.5
        + strategy_summary.get("classifier_normal_polarity_rate", 0).fillna(0) * 0.5
    )

    strategy_summary = strategy_summary.sort_values(
        ["candidate_score", "calmar"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    all_folds = pd.concat(all_fold_metrics, axis=0, ignore_index=True)
    all_thresholds_df = pd.concat(all_thresholds, axis=0, ignore_index=True)
    all_daily = pd.concat(daily_parts, axis=0, ignore_index=True)

    outputs = {
        "strategy_summary": save_csv(output_dir / "improvement_strategy_summary.csv", strategy_summary),
        "fold_metrics": save_csv(output_dir / "improvement_fold_metrics.csv", all_folds),
        "thresholds": save_csv(output_dir / "improvement_thresholds.csv", all_thresholds_df),
        "predictions": save_csv(output_dir / "improvement_oos_predictions.csv", all_pred_concat),
        "daily_returns": save_csv(output_dir / "improvement_strategy_daily_returns.csv", all_daily),
    }

    # label별 fold summary
    fold_group_summary = (
        all_folds[all_folds["status"] == "ok"]
        .groupby(["label_mode", "feature_set", "expansion_mult"], dropna=False)
        .agg(
            fold_count=("fold_id", "count"),
            mean_pr_auc=("pr_auc", "mean"),
            median_pr_auc=("pr_auc", "median"),
            mean_brier_skill=("brier_skill", "mean"),
            median_brier_skill=("brier_skill", "median"),
            positive_brier_skill_rate=("brier_skill", lambda x: float((x > 0).mean())),
            normal_polarity_rate=("probability_polarity", lambda x: float((x == "normal_better").mean())),
            mean_test_positive_rate=("test_positive_rate", "mean"),
            median_test_positive_rate=("test_positive_rate", "median"),
        )
        .reset_index()
    )
    outputs["fold_group_summary"] = save_csv(output_dir / "improvement_fold_group_summary.csv", fold_group_summary)

    top = strategy_summary.head(20).to_dict("records")
    best = strategy_summary.head(1).to_dict("records")[0] if not strategy_summary.empty else None

    summary = {
        "experiment": "rolling_highvol_label_persistence_allocation_improvement",
        "ticker": ticker,
        "bond_ticker": bond_ticker,
        "oos_start": str(start_date.date()),
        "oos_end": str(end_date.date()),
        "label_modes": list(label_modes),
        "feature_sets": list(feature_sets),
        "threshold_quantiles": list(map(float, threshold_quantiles)),
        "persistence_modes": list(persistence_modes),
        "allocation_modes": list(allocation_modes),
        "expansion_mults": list(map(float, expansion_mults)),
        "train_window": train_window,
        "calibration_window": calibration_window,
        "test_window": test_window,
        "embargo": 20,
        "threshold_source": "rolling_calibration_window_only",
        "signal_execution": "persistent signal is shifted by 1 trading day before returns",
        "transaction_cost_bps": transaction_cost_bps,
        "grid_count": int(len(grid)),
        "unique_model_fit_groups": int(len(unique_keys)),
        "best_candidate": best,
        "top20_candidates": top,
        "fold_group_summary": fold_group_summary.to_dict("records"),
        "decision_rules": {
            "stable_candidate": [
                "Calmar > buy_hold",
                "MDD > buy_hold, meaning less negative drawdown",
                "CAGR degradation not excessive",
                "positive_brier_skill_rate improved",
                "normal_polarity_rate improved",
                "result stable across threshold/persistence/allocation variants",
            ],
            "reject_or_hold": [
                "candidate only wins by tiny margin",
                "classifier fold stability remains poor",
                "high turnover or high cost",
                "performance depends on one narrow configuration",
            ],
        },
    }
    outputs["summary"] = save_json(output_dir / "improvement_experiment_summary.json", summary)

    return outputs


# ============================================================
# 11. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity-input", default="")
    parser.add_argument("--bond-input", default="")
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--bond-ticker", default="IEF")
    parser.add_argument("--output-dir", default="rolling_highvol_improvement_results")

    parser.add_argument("--label-modes", default="h20_current,h10_h20_ensemble,vol_expansion_ratio")
    parser.add_argument("--feature-sets", default="down_core")
    parser.add_argument("--threshold-quantiles", default="0.75,0.80")
    parser.add_argument("--persistence-modes", default="none,2of3,3of5")
    parser.add_argument(
        "--allocation-modes",
        default="equity60_cash40,equity70_cash30,equity80_cash20,equity80_bond20,equity70_bond20_cash10",
    )
    parser.add_argument("--expansion-mults", default="1.25,1.50")

    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--high-vol-quantile", type=float, default=0.75)
    parser.add_argument("--high-vol-lookback", type=int, default=252)

    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--calibration-method", default="sigmoid")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        equity_df = make_synthetic_ohlcv(n=1300, seed=42, ticker=args.ticker)
        bond_df = make_synthetic_ohlcv(n=1300, seed=7, ticker=args.bond_ticker)
        bond_df["close"] = 100 * np.cumprod(1 + np.random.default_rng(7).normal(0.00005, 0.003, len(bond_df)))

        # smoke는 빠르게
        train_window = 420
        calibration_window = 126
        test_window = 42
        n_estimators = 40
        label_modes = ["h20_current", "h10_h20_ensemble", "vol_expansion_ratio"]
        feature_sets = ["down_core"]
        threshold_quantiles = [0.75, 0.80]
        persistence_modes = ["none", "2of3"]
        allocation_modes = ["equity60_cash40", "equity80_cash20", "equity80_bond20"]
        expansion_mults = [1.25]
    else:
        if not args.equity_input:
            raise ValueError("--equity-input is required unless --smoke-test is used")

        equity_df = load_ohlcv(args.equity_input)
        bond_df = load_ohlcv(args.bond_input) if args.bond_input else None

        train_window = args.train_window
        calibration_window = args.calibration_window
        test_window = args.test_window
        n_estimators = args.n_estimators
        label_modes = parse_list(args.label_modes)
        feature_sets = parse_list(args.feature_sets)
        threshold_quantiles = parse_float_list(args.threshold_quantiles)
        persistence_modes = parse_list(args.persistence_modes)
        allocation_modes = parse_list(args.allocation_modes)
        expansion_mults = parse_float_list(args.expansion_mults)

    outputs = run_experiment(
        equity_df=equity_df,
        bond_df=bond_df,
        output_dir=args.output_dir,
        ticker=args.ticker,
        bond_ticker=args.bond_ticker if bond_df is not None else None,
        label_modes=label_modes,
        feature_sets=feature_sets,
        threshold_quantiles=threshold_quantiles,
        persistence_modes=persistence_modes,
        allocation_modes=allocation_modes,
        expansion_mults=expansion_mults,
        vol_window=args.vol_window,
        high_vol_quantile=args.high_vol_quantile,
        high_vol_lookback=args.high_vol_lookback,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        calibration_method=args.calibration_method,
        transaction_cost_bps=args.transaction_cost_bps,
        random_state=args.random_state,
        n_estimators=n_estimators,
    )

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    best = summary.get("best_candidate") or {}

    print("[OK] Rolling HighVol improvement experiment completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print(json.dumps(
        {
            "ticker": summary["ticker"],
            "oos_start": summary["oos_start"],
            "oos_end": summary["oos_end"],
            "grid_count": summary["grid_count"],
            "unique_model_fit_groups": summary["unique_model_fit_groups"],
            "best_experiment_id": best.get("experiment_id"),
            "best_label_mode": best.get("label_mode"),
            "best_threshold_quantile": best.get("threshold_quantile"),
            "best_persistence_mode": best.get("persistence_mode"),
            "best_allocation_mode": best.get("allocation_mode"),
            "best_cagr": best.get("cagr"),
            "best_mdd": best.get("mdd"),
            "best_calmar": best.get("calmar"),
            "best_cagr_diff_vs_buy_hold": best.get("cagr_diff_vs_buy_hold"),
            "best_mdd_diff_vs_buy_hold": best.get("mdd_diff_vs_buy_hold"),
            "best_calmar_diff_vs_buy_hold": best.get("calmar_diff_vs_buy_hold"),
            "classifier_positive_brier_skill_rate": best.get("classifier_positive_brier_skill_rate"),
            "classifier_normal_polarity_rate": best.get("classifier_normal_polarity_rate"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
