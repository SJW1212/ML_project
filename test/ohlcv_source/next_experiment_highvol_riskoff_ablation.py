# -*- coding: utf-8 -*-
"""
next_experiment_highvol_riskoff_ablation.py

Portfolio Regime Advisor 다음 실험 진행 코드.

목적
----
이전 실험 결론을 반영해 다음 3단계 실험을 한 번에 수행합니다.

1. HighVol holdout 검증
   - 이전 결과에서 유망했던 H10/H20 HighVol 후보를 별도 holdout에서 검증

2. RiskOff 라벨 재설계 sweep
   - k_mdd ∈ {1.0, 1.25, 1.5, 1.75, 2.0}
   - horizon ∈ {10, 20, 40, 60}
   - PR-AUC, MDD-event recall, false alarm, Brier/ECE 확인

3. Portfolio-level ablation
   - Buy & Hold
   - Constant NORMAL
   - HighVol only
   - RiskOff only
   - HighVol + RiskOff
   - optional 60/40, bond CSV가 있으면 사용

입력
----
필수:
- --equity-input: date, close가 포함된 OHLCV CSV

선택:
- --bond-input: date, close가 포함된 채권/방어자산 CSV
  제공하면 60/40 benchmark 계산

출력
----
output_dir/
├─ highvol_holdout_results.csv
├─ riskoff_sweep_results.csv
├─ allocation_ablation_results.csv
├─ highvol_holdout_predictions.csv
├─ riskoff_best_predictions.csv
└─ next_experiment_summary.json

실행 예시
--------
python next_experiment_highvol_riskoff_ablation.py ^
  --equity-input ohlcv_source/QQQ_ohlcv.csv ^
  --ticker QQQ ^
  --output-dir next_experiment_results ^
  --holdout-start 2023-01-01

bond 포함:
python next_experiment_highvol_riskoff_ablation.py ^
  --equity-input ohlcv_source/QQQ_ohlcv.csv ^
  --bond-input ohlcv_source/IEF_ohlcv.csv ^
  --ticker QQQ ^
  --bond-ticker IEF ^
  --output-dir next_experiment_results ^
  --holdout-start 2023-01-01

smoke test:
python next_experiment_highvol_riskoff_ablation.py --smoke-test

의존성
------
- Python 3.10+
- numpy
- pandas
- scikit-learn

주의
----
- 이 코드는 실험 코드입니다.
- 결과가 좋아도 바로 Stable 모델 채택 금지.
- 최종 채택은 별도 holdout, after-cost benchmark, fold stability, PBO/multiple testing 검토 후 가능합니다.
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

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 0. 기본 유틸
# ============================================================

LABEL_PREFIX = "y_"
FUTURE_PREFIX = "future_"
META_PREFIX = "meta_"


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


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
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=to_jsonable),
        encoding="utf-8",
    )
    return path


# ============================================================
# 1. 데이터 로딩 / synthetic
# ============================================================

def load_ohlcv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("CSV must include date column")
    if "close" not in df.columns:
        raise ValueError("CSV must include close column")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    else:
        out["volume"] = np.nan

    out = out.sort_values("date").dropna(subset=["close"]).drop_duplicates("date").reset_index(drop=True)
    return out


def make_synthetic_ohlcv(n: int = 1800, seed: int = 42, ticker: str = "QQQ") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n)

    vol = np.full(n, 0.012)
    drift = np.full(n, 0.00035)

    # 위기/고변동 구간
    for start, end, local_vol, local_drift in [
        (420, 520, 0.030, -0.0012),
        (900, 980, 0.026, -0.0010),
        (1250, 1320, 0.028, -0.0008),
    ]:
        vol[start:end] = local_vol
        drift[start:end] = local_drift

    ret = rng.normal(drift, vol)
    close = 100 * np.cumprod(1 + ret)
    volume = rng.integers(1_000_000, 9_000_000, n)

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
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
        out[f"price_slope_{w}d"] = close.pct_change(w) / w
        out[f"momentum_{w}d"] = close.pct_change(w)

    if "volume" in out.columns:
        volume = out["volume"].astype(float)
        vol_mean = volume.rolling(20, min_periods=5).mean()
        vol_std = volume.rolling(20, min_periods=5).std()
        out["volume_change_20d"] = volume.pct_change(20)
        out["volume_zscore_20d"] = (volume - vol_mean) / vol_std
        out["volume_ma_gap_20d"] = volume / vol_mean - 1.0
    else:
        out["volume_change_20d"] = np.nan
        out["volume_zscore_20d"] = np.nan
        out["volume_ma_gap_20d"] = np.nan

    out["trend_strength_score"] = (
        (out["ma_gap_20d"] > 0).astype(float)
        + (out["ma_gap_60d"] > 0).astype(float)
        + (out["ma_gap_120d"] > 0).astype(float)
    ) / 3.0

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
        "price_slope_20d", "price_slope_60d",
        "trend_strength_score",
        "volume_change_20d", "volume_zscore_20d",
    ],
    "trend_volume": [
        "return_5d", "return_10d", "return_20d", "return_40d", "return_60d", "return_120d",
        "ma_gap_20d", "ma_gap_40d", "ma_gap_60d", "ma_gap_120d", "ma_gap_252d",
        "price_slope_20d", "price_slope_60d", "price_slope_120d",
        "momentum_20d", "momentum_60d", "momentum_120d",
        "trend_strength_score",
        "volume_change_20d", "volume_zscore_20d", "volume_ma_gap_20d",
    ],
}


def select_features(df: pd.DataFrame, feature_set: str) -> List[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set: {feature_set}")

    cols = []
    for c in FEATURE_SETS[feature_set]:
        if c not in df.columns:
            continue
        if c.startswith((LABEL_PREFIX, FUTURE_PREFIX, META_PREFIX)):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    leaked = [c for c in cols if c.startswith((LABEL_PREFIX, FUTURE_PREFIX, META_PREFIX))]
    if leaked:
        raise AssertionError(f"Leakage columns detected: {leaked}")
    return cols


# ============================================================
# 3. LabelBuilder
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


def build_labels(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int = 60,
    k_direction: float = 0.8,
    k_mdd: float = 1.5,
    high_vol_quantile: float = 0.75,
    high_vol_lookback: int = 252,
) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    daily_vol_t = returns.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().shift(1)
    current_horizon_vol = daily_vol_t * math.sqrt(horizon)

    future_return_h = close.shift(-horizon) / close - 1.0
    future_realized_vol_h = compute_forward_realized_vol(returns, horizon)
    future_mdd_h = compute_forward_mdd(close, horizon)

    up_threshold = k_direction * current_horizon_vol
    down_threshold = -k_direction * current_horizon_vol
    risk_off_threshold = -k_mdd * current_horizon_vol

    high_vol_threshold = current_horizon_vol.rolling(
        high_vol_lookback,
        min_periods=max(30, high_vol_lookback // 4),
    ).quantile(high_vol_quantile)

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
# 4. 모델 / Calibration
# ============================================================

def make_model(model_type: str, random_state: int = 42):
    model_type = model_type.lower()

    if model_type == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=random_state,
                )),
            ]
        )

    if model_type == "extratrees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(
                    n_estimators=80,
                    max_depth=5,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=random_state,
                )),
            ]
        )

    if model_type == "hgb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(
                    max_iter=80,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=0.1,
                    random_state=random_state,
                )),
            ]
        )

    if model_type == "randomforest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=80,
                    max_depth=5,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=random_state,
                )),
            ]
        )

    raise ValueError(f"unsupported model_type: {model_type}")


class ProbabilityCalibrator:
    def __init__(self, method: str = "none"):
        self.method = method.lower()
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
            return np.clip(model.transform(raw_prob), 0, 1)

        return raw_prob


def split_train_calibration(train_df: pd.DataFrame, calibration_frac: float = 0.2, min_cal_rows: int = 120) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(train_df)
    cal_size = max(min_cal_rows, int(n * calibration_frac))
    if n - cal_size < 300:
        return train_df, pd.DataFrame()
    return train_df.iloc[:-cal_size].copy(), train_df.iloc[-cal_size:].copy()


def fit_predict_holdout(
    df_labeled: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    train_mask: pd.Series,
    holdout_mask: pd.Series,
    model_type: str,
    calibration_method: str,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    required = list(feature_cols) + [target_col, "date", "close"]
    data = df_labeled[required].dropna(subset=list(feature_cols) + [target_col]).copy()

    train_df = data[train_mask.loc[data.index]].copy()
    holdout_df = data[holdout_mask.loc[data.index]].copy()

    if len(train_df) < 300:
        raise ValueError("train rows too small")
    if len(holdout_df) < 30:
        raise ValueError("holdout rows too small")
    if train_df[target_col].nunique() < 2:
        raise ValueError("train target has only one class")
    if holdout_df[target_col].nunique() < 2:
        # metrics 일부가 NaN일 수 있지만 prediction은 저장
        pass

    core_train, cal_df = split_train_calibration(train_df)
    if core_train[target_col].nunique() < 2:
        raise ValueError("core train target has only one class")

    model = make_model(model_type, random_state=random_state)
    model.fit(core_train[feature_cols], core_train[target_col].astype(int))

    raw_holdout = model.predict_proba(holdout_df[feature_cols])[:, 1]

    calibrator = ProbabilityCalibrator(calibration_method)
    if not cal_df.empty and cal_df[target_col].nunique() >= 2:
        raw_cal = model.predict_proba(cal_df[feature_cols])[:, 1]
        calibrator.fit(raw_cal, cal_df[target_col].astype(int).to_numpy())
        cal_holdout = calibrator.transform(raw_holdout)
    else:
        cal_holdout = raw_holdout

    pred = holdout_df[["date", "close", target_col]].copy()
    pred = pred.rename(columns={target_col: "y_true"})
    pred["prob_raw"] = raw_holdout
    pred["prob_cal"] = cal_holdout

    info = {
        "train_rows": int(len(train_df)),
        "core_train_rows": int(len(core_train)),
        "calibration_rows": int(len(cal_df)),
        "holdout_rows": int(len(holdout_df)),
        "train_positive_rate": float(train_df[target_col].mean()),
        "holdout_positive_rate": float(holdout_df[target_col].mean()),
    }
    return pred, info


# ============================================================
# 5. Metrics
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
        "f1_at_0_5": float(f1_score(y, pred, zero_division=0)),
        "prob_mean": float(p.mean()),
        "prob_std": float(p.std()),
    }


def event_recall_at_quantile(y_true: Sequence[float], prob: Sequence[float], q: float = 0.75) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]

    if len(y) == 0:
        return {}

    threshold = float(np.quantile(p, q))
    signal = p >= threshold
    event = y == 1
    tp = int((signal & event).sum())
    signal_count = int(signal.sum())
    event_count = int(event.sum())

    return {
        "signal_quantile": float(q),
        "signal_threshold": threshold,
        "signal_count": signal_count,
        "event_count": event_count,
        "event_recall_at_q": safe_divide(tp, event_count),
        "event_precision_at_q": safe_divide(tp, signal_count),
        "false_alarm_rate_at_q": safe_divide(int((signal & ~event).sum()), signal_count),
    }


# ============================================================
# 6. Holdout / RiskOff sweep
# ============================================================

@dataclass
class CandidateConfig:
    name: str
    task: str
    horizon: int
    feature_set: str
    model_type: str
    calibration_method: str
    k_mdd: float = 1.5
    high_vol_quantile: float = 0.75


DEFAULT_HIGHVOL_CANDIDATES = [
    CandidateConfig("hv_h10_down_extra_sig", "high_vol", 10, "down_core", "extratrees", "sigmoid"),
    CandidateConfig("hv_h10_volrisk_extra_sig", "high_vol", 10, "vol_risk_core", "extratrees", "sigmoid"),
    CandidateConfig("hv_h10_down_logistic_sig", "high_vol", 10, "down_core", "logistic", "sigmoid"),
    CandidateConfig("hv_h20_down_extra_sig", "high_vol", 20, "down_core", "extratrees", "sigmoid"),
    CandidateConfig("hv_h20_volrisk_extra_sig", "high_vol", 20, "vol_risk_core", "extratrees", "sigmoid"),
]


def target_col_for_task(task: str) -> str:
    if task == "high_vol":
        return "y_high_vol"
    if task == "risk_off":
        return "y_risk_off"
    raise ValueError(f"unsupported task: {task}")


def make_time_masks(df: pd.DataFrame, holdout_start: Optional[str], holdout_frac: float = 0.25) -> Tuple[pd.Series, pd.Series, str]:
    if holdout_start:
        start = pd.to_datetime(holdout_start)
        holdout_mask = df["date"] >= start
        train_mask = df["date"] < start
        mode = f"holdout_start={holdout_start}"
    else:
        split_idx = int(len(df) * (1 - holdout_frac))
        split_date = df.iloc[split_idx]["date"]
        train_mask = pd.Series(df.index < split_idx, index=df.index)
        holdout_mask = pd.Series(df.index >= split_idx, index=df.index)
        mode = f"holdout_frac={holdout_frac}, split_date={split_date}"
    return train_mask, holdout_mask, mode


def evaluate_candidate(
    df_features: pd.DataFrame,
    candidate: CandidateConfig,
    train_mask_base: pd.Series,
    holdout_mask_base: pd.Series,
    vol_window: int,
    random_state: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    labeled = build_labels(
        df_features,
        horizon=candidate.horizon,
        vol_window=vol_window,
        k_mdd=candidate.k_mdd,
        high_vol_quantile=candidate.high_vol_quantile,
    )
    feature_cols = select_features(labeled, candidate.feature_set)
    target_col = target_col_for_task(candidate.task)

    pred, fit_info = fit_predict_holdout(
        labeled,
        target_col=target_col,
        feature_cols=feature_cols,
        train_mask=train_mask_base,
        holdout_mask=holdout_mask_base,
        model_type=candidate.model_type,
        calibration_method=candidate.calibration_method,
        random_state=random_state,
    )

    raw_metrics = binary_metrics(pred["y_true"], pred["prob_raw"])
    cal_metrics = binary_metrics(pred["y_true"], pred["prob_cal"])
    q_metrics = event_recall_at_quantile(pred["y_true"], pred["prob_cal"], q=0.75)

    row = {
        "candidate_name": candidate.name,
        "task": candidate.task,
        "horizon": candidate.horizon,
        "feature_set": candidate.feature_set,
        "model_type": candidate.model_type,
        "calibration_method": candidate.calibration_method,
        "k_mdd": candidate.k_mdd,
        "high_vol_quantile": candidate.high_vol_quantile,
        "feature_count": len(feature_cols),
        "feature_cols": "|".join(feature_cols),
        **{f"fit_{k}": v for k, v in fit_info.items()},
        **{f"raw_{k}": v for k, v in raw_metrics.items()},
        **cal_metrics,
        **q_metrics,
    }

    score = (
        1.5 * row.get("pr_gain", 0.0)
        + 1.0 * row.get("brier_skill", 0.0)
        + 0.5 * max(row.get("roc_auc", 0.5) - 0.5, 0.0)
        - 0.5 * max(row.get("ece", 0.0), 0.0)
    )
    row["score"] = float(score)

    pred["candidate_name"] = candidate.name
    pred["task"] = candidate.task
    pred["horizon"] = candidate.horizon
    pred["model_type"] = candidate.model_type
    pred["feature_set"] = candidate.feature_set
    pred["calibration_method"] = candidate.calibration_method

    return row, pred


def run_highvol_holdout(
    df_features: pd.DataFrame,
    train_mask: pd.Series,
    holdout_mask: pd.Series,
    vol_window: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_parts = []

    for candidate in DEFAULT_HIGHVOL_CANDIDATES:
        try:
            row, pred = evaluate_candidate(
                df_features,
                candidate,
                train_mask,
                holdout_mask,
                vol_window=vol_window,
                random_state=random_state,
            )
            row["status"] = "ok"
            rows.append(row)
            pred_parts.append(pred)
        except Exception as e:
            rows.append({
                "candidate_name": candidate.name,
                "task": candidate.task,
                "horizon": candidate.horizon,
                "feature_set": candidate.feature_set,
                "model_type": candidate.model_type,
                "calibration_method": candidate.calibration_method,
                "status": "error",
                "error": str(e),
                "score": -np.inf,
            })

    result = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    preds = pd.concat(pred_parts, axis=0, ignore_index=True) if pred_parts else pd.DataFrame()
    return result, preds


def run_riskoff_sweep(
    df_features: pd.DataFrame,
    train_mask: pd.Series,
    holdout_mask: pd.Series,
    horizons: Sequence[int],
    k_mdd_values: Sequence[float],
    vol_window: int,
    random_state: int,
    feature_sets: Sequence[str] = ("down_core", "vol_risk_core", "compact_mixed"),
    model_types: Sequence[str] = ("logistic", "extratrees", "hgb"),
    calibration_methods: Sequence[str] = ("none", "sigmoid"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_parts = []

    feature_sets = list(feature_sets)
    model_types = list(model_types)
    calibration_methods = list(calibration_methods)

    for h in horizons:
        for k_mdd in k_mdd_values:
            for fs in feature_sets:
                for mt in model_types:
                    for cal in calibration_methods:
                        name = f"ro_h{h}_k{k_mdd}_{fs}_{mt}_{cal}".replace(".", "p")
                        candidate = CandidateConfig(
                            name=name,
                            task="risk_off",
                            horizon=int(h),
                            feature_set=fs,
                            model_type=mt,
                            calibration_method=cal,
                            k_mdd=float(k_mdd),
                        )
                        try:
                            row, pred = evaluate_candidate(
                                df_features,
                                candidate,
                                train_mask,
                                holdout_mask,
                                vol_window=vol_window,
                                random_state=random_state,
                            )
                            row["status"] = "ok"
                            rows.append(row)

                            if row["score"] > 0:
                                pred_parts.append(pred)
                        except Exception as e:
                            rows.append({
                                "candidate_name": name,
                                "task": "risk_off",
                                "horizon": h,
                                "k_mdd": k_mdd,
                                "feature_set": fs,
                                "model_type": mt,
                                "calibration_method": cal,
                                "status": "error",
                                "error": str(e),
                                "score": -np.inf,
                            })

    result = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    preds = pd.concat(pred_parts, axis=0, ignore_index=True) if pred_parts else pd.DataFrame()
    return result, preds


# ============================================================
# 7. Portfolio ablation
# ============================================================

def align_returns(equity_df: pd.DataFrame, bond_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    eq = equity_df[["date", "close"]].copy().rename(columns={"close": "equity_close"})
    eq["equity_ret"] = eq["equity_close"].pct_change().fillna(0.0)

    if bond_df is not None:
        bd = bond_df[["date", "close"]].copy().rename(columns={"close": "bond_close"})
        bd["bond_ret"] = bd["bond_close"].pct_change().fillna(0.0)
        out = eq.merge(bd[["date", "bond_ret"]], on="date", how="left")
        out["bond_ret"] = out["bond_ret"].fillna(0.0)
    else:
        out = eq.copy()
        out["bond_ret"] = 0.0

    out["cash_ret"] = 0.0
    return out


def make_signal_panel(
    highvol_predictions: pd.DataFrame,
    riskoff_predictions: pd.DataFrame,
    best_highvol_name: Optional[str],
    best_riskoff_name: Optional[str],
) -> pd.DataFrame:
    parts = []

    if best_highvol_name and not highvol_predictions.empty:
        hv = highvol_predictions[highvol_predictions["candidate_name"] == best_highvol_name].copy()
        hv = hv[["date", "prob_cal"]].rename(columns={"prob_cal": "p_high_vol"})
        parts.append(hv)

    if best_riskoff_name and not riskoff_predictions.empty:
        ro = riskoff_predictions[riskoff_predictions["candidate_name"] == best_riskoff_name].copy()
        ro = ro[["date", "prob_cal"]].rename(columns={"prob_cal": "p_risk_off"})
        parts.append(ro)

    if not parts:
        return pd.DataFrame(columns=["date", "p_high_vol", "p_risk_off"])

    panel = parts[0]
    for p in parts[1:]:
        panel = panel.merge(p, on="date", how="outer")

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    return panel


def simulate_strategy(
    ret_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    strategy: str,
    transaction_cost_bps: float = 10.0,
    highvol_q: float = 0.75,
    riskoff_q: float = 0.75,
) -> Dict[str, object]:
    df = ret_df.merge(signal_df, on="date", how="left")
    df["p_high_vol"] = df.get("p_high_vol", pd.Series(np.nan, index=df.index))
    df["p_risk_off"] = df.get("p_risk_off", pd.Series(np.nan, index=df.index))

    # threshold는 holdout signal 내부 분위수. 실제 운영에서는 train-window threshold로 바꿔야 함.
    hv_th = float(df["p_high_vol"].quantile(highvol_q)) if df["p_high_vol"].notna().any() else np.nan
    ro_th = float(df["p_risk_off"].quantile(riskoff_q)) if df["p_risk_off"].notna().any() else np.nan

    equity_w = np.ones(len(df))
    bond_w = np.zeros(len(df))
    cash_w = np.zeros(len(df))

    if strategy == "buy_hold":
        equity_w[:] = 1.0
        bond_w[:] = 0.0
        cash_w[:] = 0.0

    elif strategy == "constant_normal":
        equity_w[:] = 0.80
        bond_w[:] = 0.10
        cash_w[:] = 0.10

    elif strategy == "sixty_forty":
        equity_w[:] = 0.60
        bond_w[:] = 0.40
        cash_w[:] = 0.0

    elif strategy == "highvol_only":
        equity_w[:] = 1.0
        mask = df["p_high_vol"] >= hv_th
        equity_w[mask.to_numpy()] = 0.60
        cash_w[mask.to_numpy()] = 0.40

    elif strategy == "riskoff_only":
        equity_w[:] = 1.0
        mask = df["p_risk_off"] >= ro_th
        equity_w[mask.to_numpy()] = 0.30
        cash_w[mask.to_numpy()] = 0.70

    elif strategy == "highvol_riskoff":
        equity_w[:] = 1.0
        hv = (df["p_high_vol"] >= hv_th).fillna(False)
        ro = (df["p_risk_off"] >= ro_th).fillna(False)

        # high vol only: 변동성 확대, 완전 방어는 아님
        equity_w[hv.to_numpy()] = 0.60
        cash_w[hv.to_numpy()] = 0.40

        # risk off 또는 highvol+riskoff: 방어
        defensive = ro
        equity_w[defensive.to_numpy()] = 0.30
        cash_w[defensive.to_numpy()] = 0.70

    else:
        raise ValueError(f"unknown strategy: {strategy}")

    # bond가 제공되면 cash 일부 대신 bond 사용 가능
    if "bond_ret" in df.columns and strategy in {"constant_normal", "sixty_forty"}:
        pass

    # weight normalization
    total_w = equity_w + bond_w + cash_w
    equity_w = equity_w / total_w
    bond_w = bond_w / total_w
    cash_w = cash_w / total_w

    turnover = np.zeros(len(df))
    turnover[1:] = np.abs(np.diff(equity_w)) + np.abs(np.diff(bond_w)) + np.abs(np.diff(cash_w))
    cost = turnover * (transaction_cost_bps / 10000.0)

    gross_ret = equity_w * df["equity_ret"].to_numpy() + bond_w * df["bond_ret"].to_numpy() + cash_w * df["cash_ret"].to_numpy()
    net_ret = gross_ret - cost

    equity_curve = np.cumprod(1.0 + net_ret)
    metrics = performance_metrics(equity_curve, net_ret)

    out = {
        "strategy": strategy,
        "rows": int(len(df)),
        "hv_threshold": hv_th,
        "ro_threshold": ro_th,
        "avg_equity_weight": float(np.mean(equity_w)),
        "avg_bond_weight": float(np.mean(bond_w)),
        "avg_cash_weight": float(np.mean(cash_w)),
        "turnover_total": float(np.sum(turnover)),
        "transaction_cost_total": float(np.sum(cost)),
        **metrics,
    }
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


def run_allocation_ablation(
    equity_df: pd.DataFrame,
    bond_df: Optional[pd.DataFrame],
    highvol_results: pd.DataFrame,
    highvol_predictions: pd.DataFrame,
    riskoff_results: pd.DataFrame,
    riskoff_predictions: pd.DataFrame,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    holdout_start = None
    if not highvol_predictions.empty:
        holdout_start = pd.to_datetime(highvol_predictions["date"]).min()
    elif not riskoff_predictions.empty:
        holdout_start = pd.to_datetime(riskoff_predictions["date"]).min()

    eq_holdout = equity_df.copy()
    if holdout_start is not None:
        eq_holdout = eq_holdout[eq_holdout["date"] >= holdout_start].copy()

    bd_holdout = None
    if bond_df is not None:
        bd_holdout = bond_df.copy()
        if holdout_start is not None:
            bd_holdout = bd_holdout[bd_holdout["date"] >= holdout_start].copy()

    ret_df = align_returns(eq_holdout, bd_holdout)

    best_hv = None
    if not highvol_results.empty and "status" in highvol_results.columns:
        ok = highvol_results[highvol_results["status"] == "ok"]
        if not ok.empty:
            best_hv = ok.sort_values("score", ascending=False).iloc[0]["candidate_name"]

    best_ro = None
    if not riskoff_results.empty and "status" in riskoff_results.columns:
        ok = riskoff_results[riskoff_results["status"] == "ok"]
        if not ok.empty:
            best_ro = ok.sort_values("score", ascending=False).iloc[0]["candidate_name"]

    signal_panel = make_signal_panel(highvol_predictions, riskoff_predictions, best_hv, best_ro)

    strategies = ["buy_hold", "constant_normal", "highvol_only", "riskoff_only", "highvol_riskoff"]
    if bond_df is not None:
        strategies.append("sixty_forty")

    rows = []
    for s in strategies:
        try:
            rows.append(simulate_strategy(ret_df, signal_panel, s, transaction_cost_bps=transaction_cost_bps))
        except Exception as e:
            rows.append({"strategy": s, "error": str(e)})

    return pd.DataFrame(rows)


# ============================================================
# 8. Runner
# ============================================================

def run_next_experiment(
    equity_df: pd.DataFrame,
    ticker: str,
    output_dir: str | Path,
    bond_df: Optional[pd.DataFrame] = None,
    bond_ticker: str = "BOND",
    holdout_start: Optional[str] = None,
    holdout_frac: float = 0.25,
    vol_window: int = 60,
    k_mdd_values: Sequence[float] = (1.0, 1.25, 1.5, 1.75, 2.0),
    riskoff_horizons: Sequence[int] = (10, 20, 40, 60),
    riskoff_feature_sets: Sequence[str] = ("down_core", "vol_risk_core", "compact_mixed"),
    riskoff_models: Sequence[str] = ("logistic", "extratrees", "hgb"),
    riskoff_calibration_methods: Sequence[str] = ("none", "sigmoid"),
    transaction_cost_bps: float = 10.0,
    random_state: int = 42,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_features = build_features(equity_df)
    train_mask, holdout_mask, split_mode = make_time_masks(df_features, holdout_start, holdout_frac)

    hv_results, hv_preds = run_highvol_holdout(
        df_features,
        train_mask,
        holdout_mask,
        vol_window=vol_window,
        random_state=random_state,
    )

    ro_results, ro_preds = run_riskoff_sweep(
        df_features,
        train_mask,
        holdout_mask,
        horizons=riskoff_horizons,
        k_mdd_values=k_mdd_values,
        vol_window=vol_window,
        random_state=random_state,
        feature_sets=riskoff_feature_sets,
        model_types=riskoff_models,
        calibration_methods=riskoff_calibration_methods,
    )

    alloc_results = run_allocation_ablation(
        equity_df=equity_df,
        bond_df=bond_df,
        highvol_results=hv_results,
        highvol_predictions=hv_preds,
        riskoff_results=ro_results,
        riskoff_predictions=ro_preds,
        transaction_cost_bps=transaction_cost_bps,
    )

    outputs = {
        "highvol_holdout_results": save_csv(output_dir / "highvol_holdout_results.csv", hv_results),
        "highvol_holdout_predictions": save_csv(output_dir / "highvol_holdout_predictions.csv", hv_preds),
        "riskoff_sweep_results": save_csv(output_dir / "riskoff_sweep_results.csv", ro_results),
        "riskoff_best_predictions": save_csv(output_dir / "riskoff_best_predictions.csv", ro_preds),
        "allocation_ablation_results": save_csv(output_dir / "allocation_ablation_results.csv", alloc_results),
    }

    best_hv = hv_results[hv_results["status"] == "ok"].sort_values("score", ascending=False).head(1).to_dict("records")
    best_ro = ro_results[ro_results["status"] == "ok"].sort_values("score", ascending=False).head(1).to_dict("records")
    best_alloc = alloc_results.sort_values("calmar", ascending=False, na_position="last").head(1).to_dict("records") if "calmar" in alloc_results.columns else []

    summary = {
        "ticker": ticker,
        "bond_ticker": bond_ticker if bond_df is not None else None,
        "input_rows": int(len(equity_df)),
        "split_mode": split_mode,
        "holdout_start_effective": str(pd.to_datetime(equity_df.loc[holdout_mask, "date"]).min()) if holdout_mask.any() else None,
        "train_rows_mask": int(train_mask.sum()),
        "holdout_rows_mask": int(holdout_mask.sum()),
        "highvol_candidate_count": int(len(hv_results)),
        "riskoff_sweep_count": int(len(ro_results)),
        "best_highvol": best_hv[0] if best_hv else None,
        "best_riskoff": best_ro[0] if best_ro else None,
        "best_allocation_by_calmar": best_alloc[0] if best_alloc else None,
        "interpretation_rules": {
            "highvol": "holdout PR-AUC, Brier skill, ECE가 양호하면 allocation volatility control 후보",
            "riskoff": "PR-AUC보다 MDD-event recall, false alarm, Brier skill을 함께 판단",
            "allocation": "after-cost benchmark를 이기지 못하면 모델 복잡화 금지",
        },
        "do_not_do": [
            "Do not promote highvol candidate to stable before separate holdout/fold stability.",
            "Do not use riskoff as hard trigger if brier_skill is negative.",
            "Do not claim portfolio improvement before after-cost benchmark comparison.",
        ],
    }

    outputs["summary"] = save_json(output_dir / "next_experiment_summary.json", summary)
    return outputs


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity-input", default="")
    parser.add_argument("--bond-input", default="")
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--bond-ticker", default="IEF")
    parser.add_argument("--output-dir", default="next_experiment_results")
    parser.add_argument("--holdout-start", default=None)
    parser.add_argument("--holdout-frac", type=float, default=0.25)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--k-mdd-values", default="1.0,1.25,1.5,1.75,2.0")
    parser.add_argument("--riskoff-horizons", default="10,20,40,60")
    parser.add_argument("--riskoff-feature-sets", default="down_core,vol_risk_core,compact_mixed")
    parser.add_argument("--riskoff-models", default="logistic,extratrees,hgb")
    parser.add_argument("--riskoff-calibration-methods", default="none,sigmoid")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        equity_df = make_synthetic_ohlcv(n=900, seed=42, ticker=args.ticker)
        bond_df = make_synthetic_ohlcv(n=900, seed=7, ticker=args.bond_ticker)
        # synthetic bond를 더 안정적으로 변환
        bond_df["close"] = 100 * np.cumprod(1 + np.random.default_rng(7).normal(0.00005, 0.003, len(bond_df)))
        output_dir = args.output_dir
        riskoff_horizons = [10]
        k_mdd_values = [1.0]
        riskoff_feature_sets = ["down_core"]
        riskoff_models = ["logistic"]
        riskoff_calibration_methods = ["sigmoid"]
        global DEFAULT_HIGHVOL_CANDIDATES
        DEFAULT_HIGHVOL_CANDIDATES = [
            CandidateConfig("hv_smoke_h10_down_logistic_sig", "high_vol", 10, "down_core", "logistic", "sigmoid")
        ]
    else:
        if not args.equity_input:
            raise ValueError("--equity-input is required unless --smoke-test is used")
        equity_df = load_ohlcv(args.equity_input)
        bond_df = load_ohlcv(args.bond_input) if args.bond_input else None
        output_dir = args.output_dir
        riskoff_horizons = parse_int_list(args.riskoff_horizons)
        k_mdd_values = parse_float_list(args.k_mdd_values)
        riskoff_feature_sets = [x.strip() for x in args.riskoff_feature_sets.split(",") if x.strip()]
        riskoff_models = [x.strip() for x in args.riskoff_models.split(",") if x.strip()]
        riskoff_calibration_methods = [x.strip() for x in args.riskoff_calibration_methods.split(",") if x.strip()]

    outputs = run_next_experiment(
        equity_df=equity_df,
        ticker=args.ticker,
        output_dir=output_dir,
        bond_df=bond_df,
        bond_ticker=args.bond_ticker,
        holdout_start=args.holdout_start,
        holdout_frac=args.holdout_frac,
        vol_window=args.vol_window,
        k_mdd_values=k_mdd_values,
        riskoff_horizons=riskoff_horizons,
        riskoff_feature_sets=riskoff_feature_sets,
        riskoff_models=riskoff_models,
        riskoff_calibration_methods=riskoff_calibration_methods,
        transaction_cost_bps=args.transaction_cost_bps,
        random_state=args.random_state,
    )

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))

    print("[OK] Next experiment completed.")
    print(f"[OK] Output dir: {Path(output_dir).resolve()}")
    print(json.dumps(
        {
            "ticker": summary["ticker"],
            "input_rows": summary["input_rows"],
            "split_mode": summary["split_mode"],
            "train_rows_mask": summary["train_rows_mask"],
            "holdout_rows_mask": summary["holdout_rows_mask"],
            "best_highvol_name": summary["best_highvol"].get("candidate_name") if summary["best_highvol"] else None,
            "best_highvol_pr_auc": summary["best_highvol"].get("pr_auc") if summary["best_highvol"] else None,
            "best_highvol_brier_skill": summary["best_highvol"].get("brier_skill") if summary["best_highvol"] else None,
            "best_highvol_ece": summary["best_highvol"].get("ece") if summary["best_highvol"] else None,
            "best_riskoff_name": summary["best_riskoff"].get("candidate_name") if summary["best_riskoff"] else None,
            "best_riskoff_pr_auc": summary["best_riskoff"].get("pr_auc") if summary["best_riskoff"] else None,
            "best_riskoff_brier_skill": summary["best_riskoff"].get("brier_skill") if summary["best_riskoff"] else None,
            "best_allocation_by_calmar": summary["best_allocation_by_calmar"].get("strategy") if summary["best_allocation_by_calmar"] else None,
            "best_allocation_calmar": summary["best_allocation_by_calmar"].get("calmar") if summary["best_allocation_by_calmar"] else None,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
