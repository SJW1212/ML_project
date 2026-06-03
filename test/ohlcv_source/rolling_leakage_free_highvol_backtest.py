# -*- coding: utf-8 -*-
"""
rolling_leakage_free_highvol_backtest.py

Rolling Leakage-Free HighVol Allocation Backtest.

목적
----
단일 holdout에서 좋았던 HighVol only 전략을 rolling walk-forward 방식으로 재검증합니다.

핵심 원칙
---------
1. threshold는 test window가 아니라 calibration window에서만 산출
2. test window에는 고정 threshold만 적용
3. horizon만큼 embargo 적용
4. 같은 날 close 기반 feature로 만든 신호를 같은 날 수익률에 쓰지 않음
   - 전략 수익률 계산 시 signal을 1거래일 shift
5. 전체 OOS 구간에서 benchmark와 after-cost 비교

기본 모델
---------
- target: y_high_vol
- horizon: H20
- feature_set: down_core
- model: ExtraTrees
- calibration: sigmoid
- thresholds: q=0.75, 0.80 기본

출력 파일
---------
output_dir/
├─ rolling_oos_predictions.csv
├─ rolling_fold_metrics.csv
├─ rolling_thresholds.csv
├─ rolling_strategy_daily_returns.csv
├─ rolling_strategy_summary.csv
└─ rolling_highvol_summary.json

실행 예시
--------
python rolling_leakage_free_highvol_backtest.py ^
  --equity-input QQQ_ohlcv.csv ^
  --bond-input IEF_ohlcv.csv ^
  --ticker QQQ ^
  --bond-ticker IEF ^
  --output-dir rolling_highvol_results_qqq_ief ^
  --train-window 1260 ^
  --calibration-window 252 ^
  --test-window 63 ^
  --horizon 20 ^
  --threshold-quantiles 0.75,0.80 ^
  --transaction-cost-bps 10

smoke test:
python rolling_leakage_free_highvol_backtest.py --smoke-test

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
# 0. 유틸
# ============================================================

LABEL_PREFIX = "y_"
FUTURE_PREFIX = "future_"
META_PREFIX = "meta_"


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
# 1. 데이터 로딩
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
# 2. Feature / Label
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
    "vol_risk_core": [
        "volatility_5d", "volatility_10d", "volatility_20d", "volatility_40d", "volatility_60d", "volatility_120d",
        "downside_volatility_20d", "downside_volatility_40d", "downside_volatility_60d", "downside_volatility_120d",
        "mdd_20d", "mdd_40d", "mdd_60d", "mdd_120d", "mdd_252d",
        "return_20d", "return_60d", "return_120d",
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


def build_labels(
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
    future_realized_vol_h = compute_forward_realized_vol(returns, horizon)

    high_vol_threshold = current_horizon_vol.rolling(
        high_vol_lookback,
        min_periods=max(30, high_vol_lookback // 4),
    ).quantile(high_vol_quantile)

    y_high_vol = (future_realized_vol_h >= high_vol_threshold).astype(float)
    invalid = current_horizon_vol.isna() | future_realized_vol_h.isna() | high_vol_threshold.isna()

    out["future_realized_vol_h"] = future_realized_vol_h
    out["meta_current_horizon_vol"] = current_horizon_vol
    out["meta_high_vol_threshold"] = high_vol_threshold
    out["y_high_vol"] = y_high_vol.mask(invalid, np.nan)

    return out


# ============================================================
# 3. Model / Calibration
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
# 4. Rolling split
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
    """
    index 기준 rolling folds 생성.

    구조:
    [core train][calibration][embargo][test]
    """
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
# 6. Rolling model experiment
# ============================================================

def run_rolling_predictions(
    df: pd.DataFrame,
    feature_set: str,
    horizon: int,
    vol_window: int,
    high_vol_quantile: float,
    train_window: int,
    calibration_window: int,
    test_window: int,
    threshold_quantiles: Sequence[float],
    calibration_method: str,
    random_state: int,
    n_estimators: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    featured = build_features(df)
    labeled = build_labels(
        featured,
        horizon=horizon,
        vol_window=vol_window,
        high_vol_quantile=high_vol_quantile,
    )

    feature_cols = select_features(labeled, feature_set)
    required = ["date", "close", "y_high_vol"] + feature_cols

    data = labeled[required].dropna(subset=["y_high_vol"] + feature_cols).reset_index(drop=True)

    folds = build_rolling_folds(
        data,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        embargo=horizon,
        step=test_window,
    )

    prediction_parts: List[pd.DataFrame] = []
    threshold_rows: List[Dict[str, object]] = []
    fold_metric_rows: List[Dict[str, object]] = []

    for fold in folds:
        core = data.iloc[fold.core_start:fold.core_end].copy()
        cal = data.iloc[fold.cal_start:fold.cal_end].copy()
        test = data.iloc[fold.test_start:fold.test_end].copy()

        fold_status = "ok"
        fold_error = ""

        try:
            if core["y_high_vol"].nunique() < 2:
                raise ValueError("core train has one class")

            model = make_extratrees(random_state=random_state + fold.fold_id, n_estimators=n_estimators)
            model.fit(core[feature_cols], core["y_high_vol"].astype(int))

            raw_cal = model.predict_proba(cal[feature_cols])[:, 1]
            raw_test = model.predict_proba(test[feature_cols])[:, 1]

            calibrator = ProbabilityCalibrator(method=calibration_method)
            if cal["y_high_vol"].nunique() >= 2:
                calibrator.fit(raw_cal, cal["y_high_vol"].astype(int).to_numpy())
                cal_prob = calibrator.transform(raw_cal)
                test_prob = calibrator.transform(raw_test)
                calibration_status = "calibrated"
            else:
                cal_prob = raw_cal
                test_prob = raw_test
                calibration_status = "skipped_single_class_calibration_set"

            pred = test[["date", "close", "y_high_vol"]].copy()
            pred = pred.rename(columns={"y_high_vol": "y_true"})
            pred["fold_id"] = fold.fold_id
            pred["prob_raw"] = raw_test
            pred["prob_cal"] = test_prob
            pred["calibration_status"] = calibration_status
            pred["test_start_date"] = fold.test_start_date
            pred["test_end_date"] = fold.test_end_date

            for q in threshold_quantiles:
                threshold = float(np.quantile(cal_prob, q))
                pred[f"threshold_q{q:.2f}"] = threshold
                pred[f"signal_q{q:.2f}"] = (pred["prob_cal"] >= threshold).astype(int)

                threshold_rows.append({
                    "fold_id": fold.fold_id,
                    "threshold_quantile": float(q),
                    "threshold_source": "rolling_calibration_window_only",
                    "threshold": threshold,
                    "cal_signal_rate": float(np.mean(cal_prob >= threshold)),
                    "test_signal_rate": float(np.mean(test_prob >= threshold)),
                    "core_positive_rate": float(core["y_high_vol"].mean()),
                    "cal_positive_rate": float(cal["y_high_vol"].mean()),
                    "test_positive_rate": float(test["y_high_vol"].mean()),
                    **asdict(fold),
                })

            metrics = binary_metrics(pred["y_true"], pred["prob_cal"])
            fold_metric_rows.append({
                "fold_id": fold.fold_id,
                "status": "ok",
                "calibration_status": calibration_status,
                "core_rows": int(len(core)),
                "calibration_rows": int(len(cal)),
                "test_rows": int(len(test)),
                "core_positive_rate": float(core["y_high_vol"].mean()),
                "cal_positive_rate": float(cal["y_high_vol"].mean()),
                "test_positive_rate": float(test["y_high_vol"].mean()),
                **metrics,
                **asdict(fold),
            })

            prediction_parts.append(pred)

        except Exception as e:
            fold_status = "error"
            fold_error = str(e)
            fold_metric_rows.append({
                "fold_id": fold.fold_id,
                "status": fold_status,
                "error": fold_error,
                **asdict(fold),
            })

    predictions = pd.concat(prediction_parts, axis=0, ignore_index=True) if prediction_parts else pd.DataFrame()
    thresholds = pd.DataFrame(threshold_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)

    if predictions.empty:
        raise RuntimeError("No valid rolling predictions generated.")

    return predictions, thresholds, fold_metrics


# ============================================================
# 7. Allocation backtest
# ============================================================

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


def simulate_strategy(
    ret_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    strategy: str,
    threshold_quantile: Optional[float],
    transaction_cost_bps: float,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    df = ret_df.copy()

    if not pred_df.empty:
        signal_cols = ["date", "prob_cal"]
        if threshold_quantile is not None:
            signal_cols.append(f"signal_q{threshold_quantile:.2f}")
            signal_cols.append(f"threshold_q{threshold_quantile:.2f}")

        sig = pred_df[signal_cols].copy()
        df = df.merge(sig, on="date", how="left")
    else:
        df["prob_cal"] = np.nan

    # 같은 날 close 기반 신호를 같은 날 수익률에 쓰지 않도록 1거래일 지연 적용
    if threshold_quantile is not None and f"signal_q{threshold_quantile:.2f}" in df.columns:
        signal_col = f"signal_q{threshold_quantile:.2f}"
        threshold_col = f"threshold_q{threshold_quantile:.2f}"
        df[signal_col] = df[signal_col].ffill().shift(1).fillna(0)
        df[threshold_col] = df[threshold_col].ffill()
    else:
        signal_col = None

    n = len(df)
    equity_w = np.ones(n)
    bond_w = np.zeros(n)
    cash_w = np.zeros(n)

    if strategy == "buy_hold":
        equity_w[:] = 1.0

    elif strategy == "constant_normal":
        equity_w[:] = 0.80
        bond_w[:] = 0.10
        cash_w[:] = 0.10

    elif strategy == "sixty_forty":
        equity_w[:] = 0.60
        bond_w[:] = 0.40

    elif strategy == "highvol_only":
        equity_w[:] = 1.0
        if signal_col is not None:
            highvol = df[signal_col].astype(int).to_numpy() == 1
            equity_w[highvol] = 0.60
            cash_w[highvol] = 0.40

    else:
        raise ValueError(f"unknown strategy: {strategy}")

    total_w = equity_w + bond_w + cash_w
    equity_w = equity_w / total_w
    bond_w = bond_w / total_w
    cash_w = cash_w / total_w

    turnover = np.zeros(n)
    turnover[1:] = np.abs(np.diff(equity_w)) + np.abs(np.diff(bond_w)) + np.abs(np.diff(cash_w))
    cost = turnover * (transaction_cost_bps / 10000.0)

    gross_ret = (
        equity_w * df["equity_ret"].to_numpy()
        + bond_w * df["bond_ret"].to_numpy()
        + cash_w * df["cash_ret"].to_numpy()
    )
    net_ret = gross_ret - cost
    curve = np.cumprod(1.0 + net_ret)

    df["strategy"] = strategy
    df["threshold_quantile"] = threshold_quantile
    df["equity_weight"] = equity_w
    df["bond_weight"] = bond_w
    df["cash_weight"] = cash_w
    df["turnover"] = turnover
    df["cost"] = cost
    df["strategy_ret"] = net_ret
    df["equity_curve"] = curve

    metrics = {
        "strategy": strategy,
        "threshold_quantile": threshold_quantile,
        "rows": int(n),
        "avg_equity_weight": float(np.mean(equity_w)),
        "avg_bond_weight": float(np.mean(bond_w)),
        "avg_cash_weight": float(np.mean(cash_w)),
        "turnover_total": float(np.sum(turnover)),
        "transaction_cost_total": float(np.sum(cost)),
        **performance_metrics(curve, net_ret),
    }

    return metrics, df


def run_allocation_backtest(
    equity_df: pd.DataFrame,
    bond_df: Optional[pd.DataFrame],
    predictions: pd.DataFrame,
    threshold_quantiles: Sequence[float],
    transaction_cost_bps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_date = pd.to_datetime(predictions["date"]).min()
    end_date = pd.to_datetime(predictions["date"]).max()
    ret_df = align_returns(equity_df, bond_df, start_date, end_date)

    summary_rows: List[Dict[str, object]] = []
    daily_parts: List[pd.DataFrame] = []

    benchmark_strategies = ["buy_hold", "constant_normal"]
    if bond_df is not None:
        benchmark_strategies.append("sixty_forty")

    for strategy in benchmark_strategies:
        row, daily = simulate_strategy(ret_df, pd.DataFrame(), strategy, None, transaction_cost_bps)
        summary_rows.append(row)
        daily_parts.append(daily)

    for q in threshold_quantiles:
        row, daily = simulate_strategy(ret_df, predictions, "highvol_only", q, transaction_cost_bps)
        summary_rows.append(row)
        daily_parts.append(daily)

    summary = pd.DataFrame(summary_rows).sort_values("calmar", ascending=False, na_position="last").reset_index(drop=True)
    daily_all = pd.concat(daily_parts, axis=0, ignore_index=True)

    return summary, daily_all


# ============================================================
# 8. Main runner
# ============================================================

def run_experiment(
    equity_df: pd.DataFrame,
    bond_df: Optional[pd.DataFrame],
    output_dir: str | Path,
    ticker: str,
    bond_ticker: Optional[str],
    feature_set: str,
    horizon: int,
    vol_window: int,
    high_vol_quantile: float,
    train_window: int,
    calibration_window: int,
    test_window: int,
    threshold_quantiles: Sequence[float],
    calibration_method: str,
    transaction_cost_bps: float,
    random_state: int,
    n_estimators: int,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions, thresholds, fold_metrics = run_rolling_predictions(
        df=equity_df,
        feature_set=feature_set,
        horizon=horizon,
        vol_window=vol_window,
        high_vol_quantile=high_vol_quantile,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        threshold_quantiles=threshold_quantiles,
        calibration_method=calibration_method,
        random_state=random_state,
        n_estimators=n_estimators,
    )

    strategy_summary, strategy_daily = run_allocation_backtest(
        equity_df=equity_df,
        bond_df=bond_df,
        predictions=predictions,
        threshold_quantiles=threshold_quantiles,
        transaction_cost_bps=transaction_cost_bps,
    )

    overall_metrics = binary_metrics(predictions["y_true"], predictions["prob_cal"])

    outputs = {
        "predictions": save_csv(output_dir / "rolling_oos_predictions.csv", predictions),
        "thresholds": save_csv(output_dir / "rolling_thresholds.csv", thresholds),
        "fold_metrics": save_csv(output_dir / "rolling_fold_metrics.csv", fold_metrics),
        "strategy_daily": save_csv(output_dir / "rolling_strategy_daily_returns.csv", strategy_daily),
        "strategy_summary": save_csv(output_dir / "rolling_strategy_summary.csv", strategy_summary),
    }

    ok_folds = fold_metrics[fold_metrics["status"] == "ok"].copy() if "status" in fold_metrics.columns else fold_metrics
    best_strategy = strategy_summary.head(1).to_dict("records")[0] if not strategy_summary.empty else None

    summary = {
        "experiment": "rolling_leakage_free_highvol_backtest",
        "ticker": ticker,
        "bond_ticker": bond_ticker,
        "feature_set": feature_set,
        "horizon": horizon,
        "vol_window": vol_window,
        "high_vol_quantile": high_vol_quantile,
        "train_window": train_window,
        "calibration_window": calibration_window,
        "test_window": test_window,
        "embargo": horizon,
        "threshold_quantiles": list(map(float, threshold_quantiles)),
        "threshold_source": "rolling_calibration_window_only",
        "signal_execution": "probability signal is shifted by 1 trading day before applying to returns",
        "transaction_cost_bps": transaction_cost_bps,
        "fold_count": int(len(fold_metrics)),
        "ok_fold_count": int(len(ok_folds)),
        "oos_prediction_rows": int(len(predictions)),
        "oos_start": str(pd.to_datetime(predictions["date"]).min().date()),
        "oos_end": str(pd.to_datetime(predictions["date"]).max().date()),
        "overall_highvol_oos_metrics": overall_metrics,
        "fold_metric_summary": {
            "mean_pr_auc": float(ok_folds["pr_auc"].mean()) if "pr_auc" in ok_folds.columns else np.nan,
            "median_pr_auc": float(ok_folds["pr_auc"].median()) if "pr_auc" in ok_folds.columns else np.nan,
            "mean_brier_skill": float(ok_folds["brier_skill"].mean()) if "brier_skill" in ok_folds.columns else np.nan,
            "positive_brier_skill_rate": float((ok_folds["brier_skill"] > 0).mean()) if "brier_skill" in ok_folds.columns else np.nan,
            "normal_polarity_rate": float((ok_folds["probability_polarity"] == "normal_better").mean()) if "probability_polarity" in ok_folds.columns else np.nan,
        },
        "best_strategy_by_calmar": best_strategy,
        "top_strategy_rows": strategy_summary.head(10).to_dict("records"),
        "critical_notes": [
            "Thresholds are computed only from each rolling calibration window.",
            "Signals are shifted by one trading day before strategy return calculation.",
            "This is still not final Stable adoption; test additional regimes and multiple testing risk.",
        ],
    }

    outputs["summary"] = save_json(output_dir / "rolling_highvol_summary.json", summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity-input", default="")
    parser.add_argument("--bond-input", default="")
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--bond-ticker", default="IEF")
    parser.add_argument("--output-dir", default="rolling_highvol_results")

    parser.add_argument("--feature-set", default="down_core")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--high-vol-quantile", type=float, default=0.75)

    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--threshold-quantiles", default="0.75,0.80")
    parser.add_argument("--calibration-method", default="sigmoid")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        equity_df = make_synthetic_ohlcv(n=1400, seed=42, ticker=args.ticker)
        bond_df = make_synthetic_ohlcv(n=1400, seed=7, ticker=args.bond_ticker)
        bond_df["close"] = 100 * np.cumprod(1 + np.random.default_rng(7).normal(0.00005, 0.003, len(bond_df)))
        train_window = 500
        calibration_window = 126
        test_window = 42
        n_estimators = 50
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
        horizon=args.horizon,
        vol_window=args.vol_window,
        high_vol_quantile=args.high_vol_quantile,
        train_window=train_window,
        calibration_window=calibration_window,
        test_window=test_window,
        threshold_quantiles=parse_float_list(args.threshold_quantiles),
        calibration_method=args.calibration_method,
        transaction_cost_bps=args.transaction_cost_bps,
        random_state=args.random_state,
        n_estimators=n_estimators,
    )

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    best = summary.get("best_strategy_by_calmar", {})

    print("[OK] Rolling leakage-free HighVol backtest completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print(json.dumps(
        {
            "ticker": summary["ticker"],
            "oos_start": summary["oos_start"],
            "oos_end": summary["oos_end"],
            "fold_count": summary["fold_count"],
            "ok_fold_count": summary["ok_fold_count"],
            "threshold_source": summary["threshold_source"],
            "signal_execution": summary["signal_execution"],
            "overall_pr_auc": summary["overall_highvol_oos_metrics"].get("pr_auc"),
            "overall_brier_skill": summary["overall_highvol_oos_metrics"].get("brier_skill"),
            "mean_fold_pr_auc": summary["fold_metric_summary"].get("mean_pr_auc"),
            "positive_brier_skill_rate": summary["fold_metric_summary"].get("positive_brier_skill_rate"),
            "best_strategy": best.get("strategy"),
            "best_threshold_quantile": best.get("threshold_quantile"),
            "best_cagr": best.get("cagr"),
            "best_mdd": best.get("mdd"),
            "best_calmar": best.get("calmar"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
