# -*- coding: utf-8 -*-
"""
riskoff_free_touch_head_validator.py

RiskOff Head를 제외하고, 기존 Dual-HighVol 구조에
Down-touch / Up-touch / BearTrend 후보 head를 추가했을 때
실제로 전략 성과와 drawdown timing이 개선되는지 검증합니다.

핵심 목적
---------
1. RiskOff Head 제거
2. 기존 Dual-HighVol baseline 유지
3. Down-touch Head 추가 검증
4. Up-touch veto 추가 검증
5. BearTrend Head 추가 검증
6. 후보 조합별 after-cost 성과 / MDD / Calmar / signal timing 비교

기본 비교 전략
--------------
- buy_hold
- dual_highvol_baseline
- down_touch_only
- beartrend_only
- dual_and_down_confirm
- dual_or_down_early
- dual_with_up_veto
- dual_or_beartrend
- dual_or_down_or_beartrend

라벨
----
HighVol H20:
    미래 H일 realized volatility가 과거 기준 threshold 이상인가

Expansion:
    미래 H일 변동성이 현재 horizon volatility 대비 expansion_mult 이상인가

Down-touch:
    향후 H일 안에 아래쪽 barrier를 터치하는가
    future_low_1_to_H <= close_t * (1 - k * current_horizon_vol)

Up-touch:
    향후 H일 안에 위쪽 barrier를 터치하는가
    future_high_1_to_H >= close_t * (1 + k * current_horizon_vol)

BearTrend:
    향후 H일 수익률이 음수 방향으로 충분히 크고,
    미래 drawdown도 동반되는가

누수 방지
---------
- feature는 현재 및 과거 데이터만 사용
- label은 미래 구간 사용
- threshold는 calibration window에서만 산출
- test window threshold 재계산 금지
- signal은 1거래일 shift 후 수익률에 적용

실행 예시 - CMD
---------------
python riskoff_free_touch_head_validator.py ^
  --equity-input QQQ_ohlcv.csv ^
  --bond-input IEF_ohlcv.csv ^
  --asset-name QQQ ^
  --output-dir riskoff_free_touch_QQQ

Cross-asset QQQ 결과 폴더와 같은 위치에서 실행하려면:
python riskoff_free_touch_head_validator.py ^
  --equity-input QQQ_ohlcv.csv ^
  --bond-input IEF_ohlcv.csv ^
  --asset-name QQQ ^
  --output-dir riskoff_free_touch_QQQ ^
  --train-window 1260 ^
  --calibration-window 252 ^
  --test-window 63 ^
  --embargo 40

출력
----
output_dir/
├─ riskoff_free_touch_summary.json
├─ strategy_summary.csv
├─ strategy_daily_returns.csv
├─ head_metrics.csv
├─ fold_metrics.csv
├─ oos_predictions.csv
├─ signal_timing_summary.csv
└─ drawdown_event_summary.csv

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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")


# ============================================================
# 0. IO / Utils
# ============================================================

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


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    return str(obj)


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_divide(a: float, b: float, default=np.nan) -> float:
    if b == 0 or pd.isna(b):
        return default
    return float(a / b)



def rolling_min_periods(window: int, min_floor: int = 5, frac: float = 1 / 3) -> int:
    """
    Safe min_periods for rolling operations.
    Ensures 1 <= min_periods <= window.
    """
    return max(1, min(int(window), max(min_floor, int(window * frac))))


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    rename_map = {
        "datetime": "date",
        "timestamp": "date",
        "adj_close": "adj_close",
        "adjclose": "adj_close",
        "adjusted_close": "adj_close",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV missing required columns: {missing}. columns={list(out.columns)}")

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
        raise FileNotFoundError(f"OHLCV file not found: {path}")
    return normalize_ohlcv_columns(pd.read_csv(path))


# ============================================================
# 1. Feature / Label Builder
# ============================================================

def rolling_mdd_from_close(close: pd.Series, window: int) -> pd.Series:
    roll_max = close.rolling(window, min_periods=rolling_min_periods(window, min_floor=10)).max()
    return close / roll_max - 1.0


def future_window_max(s: pd.Series, horizon: int) -> pd.Series:
    return s.shift(-1).rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))


def future_window_min(s: pd.Series, horizon: int) -> pd.Series:
    return s.shift(-1).rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))


def future_window_std(ret: pd.Series, horizon: int) -> pd.Series:
    return ret.shift(-1).rolling(horizon, min_periods=horizon).std().shift(-(horizon - 1))


def build_features_and_labels(
    df: pd.DataFrame,
    highvol_horizon: int = 20,
    touch_horizon: int = 20,
    bear_horizon: int = 60,
    vol_window: int = 60,
    highvol_lookback: int = 252,
    highvol_quantile: float = 0.75,
    expansion_mult: float = 1.25,
    touch_k: float = 0.75,
    bear_return_k: float = 0.75,
    bear_mdd_k: float = 1.25,
) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy().sort_values("date").reset_index(drop=True)

    close = d["adj_close"].astype(float)
    raw_close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    volume = d["volume"].astype(float)

    ret = close.pct_change()
    d["ret_1d"] = ret

    # Core return features
    for w in [3, 5, 10, 20, 40, 60, 120, 252]:
        d[f"return_{w}d"] = close.pct_change(w)
        d[f"volatility_{w}d"] = ret.rolling(w, min_periods=rolling_min_periods(w, min_floor=5)).std()
        d[f"downside_volatility_{w}d"] = ret.where(ret < 0, 0.0).rolling(w, min_periods=rolling_min_periods(w, min_floor=5)).std()
        d[f"mdd_{w}d"] = rolling_mdd_from_close(close, w)

    # Moving average gap / slope-like features
    for w in [10, 20, 60, 120, 200]:
        ma = close.rolling(w, min_periods=rolling_min_periods(w, min_floor=5)).mean()
        d[f"ma_gap_{w}d"] = close / ma - 1.0
        d[f"ma_slope_{w}d"] = ma.pct_change(5)

    # Range and volume features
    d["hl_range"] = (high - low) / raw_close.replace(0, np.nan)
    d["oc_return"] = raw_close / d["open"].replace(0, np.nan) - 1.0
    d["gap_return"] = d["open"] / raw_close.shift(1).replace(0, np.nan) - 1.0

    if not volume.isna().all():
        for w in [5, 20, 60]:
            vol_ma = volume.rolling(w, min_periods=rolling_min_periods(w, min_floor=3)).mean()
            d[f"volume_ratio_{w}d"] = volume / vol_ma.replace(0, np.nan)
            d[f"volume_change_{w}d"] = volume.pct_change(w)

    # Current horizon volatility: no current/future leakage
    d[f"current_horizon_vol_h{highvol_horizon}"] = (
        ret.rolling(vol_window, min_periods=rolling_min_periods(vol_window, min_floor=20, frac=0.5)).std().shift(1)
        * math.sqrt(highvol_horizon)
    )
    d[f"current_horizon_vol_touch_h{touch_horizon}"] = (
        ret.rolling(vol_window, min_periods=rolling_min_periods(vol_window, min_floor=20, frac=0.5)).std().shift(1)
        * math.sqrt(touch_horizon)
    )
    d[f"current_horizon_vol_bear_h{bear_horizon}"] = (
        ret.rolling(vol_window, min_periods=rolling_min_periods(vol_window, min_floor=20, frac=0.5)).std().shift(1)
        * math.sqrt(bear_horizon)
    )

    # HighVol label
    future_vol_h = future_window_std(ret, highvol_horizon) * math.sqrt(highvol_horizon)
    current_realized_vol = ret.rolling(highvol_horizon, min_periods=rolling_min_periods(highvol_horizon, min_floor=5, frac=0.5)).std().shift(1) * math.sqrt(highvol_horizon)
    highvol_threshold = current_realized_vol.rolling(highvol_lookback, min_periods=rolling_min_periods(highvol_lookback, min_floor=60, frac=0.5)).quantile(highvol_quantile).shift(1)
    d["future_vol_h20"] = future_vol_h
    d["meta_highvol_threshold"] = highvol_threshold
    d["y_highvol_h20"] = (future_vol_h >= highvol_threshold).astype(float)

    # Expansion label
    cur_hv = d[f"current_horizon_vol_h{highvol_horizon}"]
    d["y_highvol_expansion"] = ((future_vol_h / cur_hv.replace(0, np.nan)) >= expansion_mult).astype(float)

    # Touch labels
    future_max_high = future_window_max(high, touch_horizon)
    future_min_low = future_window_min(low, touch_horizon)
    cur_touch_vol = d[f"current_horizon_vol_touch_h{touch_horizon}"]
    upper = raw_close * (1.0 + touch_k * cur_touch_vol)
    lower = raw_close * (1.0 - touch_k * cur_touch_vol)

    d[f"meta_upper_barrier_h{touch_horizon}_k{touch_k}"] = upper
    d[f"meta_lower_barrier_h{touch_horizon}_k{touch_k}"] = lower
    d[f"future_high_h{touch_horizon}"] = future_max_high
    d[f"future_low_h{touch_horizon}"] = future_min_low

    d["y_up_touch"] = (future_max_high >= upper).astype(float)
    d["y_down_touch"] = (future_min_low <= lower).astype(float)

    # BearTrend label: long horizon negative return + meaningful future drawdown
    future_close_bear = close.shift(-bear_horizon)
    future_return_bear = future_close_bear / close - 1.0
    future_min_low_bear = future_window_min(low, bear_horizon)
    future_mdd_bear = future_min_low_bear / raw_close - 1.0
    cur_bear_vol = d[f"current_horizon_vol_bear_h{bear_horizon}"]

    d[f"future_return_h{bear_horizon}"] = future_return_bear
    d[f"future_mdd_h{bear_horizon}"] = future_mdd_bear

    d["y_beartrend"] = (
        (future_return_bear <= -bear_return_k * cur_bear_vol)
        & (future_mdd_bear <= -bear_mdd_k * cur_bear_vol)
    ).astype(float)

    # Remove rows where labels are impossible due to insufficient future/past data
    for y in ["y_highvol_h20", "y_highvol_expansion", "y_up_touch", "y_down_touch", "y_beartrend"]:
        invalid = d[y].isna()
        d.loc[invalid, y] = np.nan

    # Feature columns: exclude future/meta/labels/raw OHLCV identity columns
    excluded_prefix = ("y_", "future_", "meta_")
    excluded_cols = {
        "date", "open", "high", "low", "close", "adj_close", "volume",
    }
    feature_cols = [
        c for c in d.columns
        if c not in excluded_cols
        and not c.startswith(excluded_prefix)
        and pd.api.types.is_numeric_dtype(d[c])
    ]

    # explicit leakage guard
    leakage_cols = [c for c in feature_cols if c.startswith(("y_", "future_", "meta_"))]
    if leakage_cols:
        raise RuntimeError(f"Leakage columns found in features: {leakage_cols}")

    return d, feature_cols


# ============================================================
# 2. Splits / Models / Metrics
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


def make_rolling_folds(n: int, train_window: int, calibration_window: int, test_window: int, embargo: int) -> List[Fold]:
    folds = []
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
        start += test_window
        fold_id += 1
    return folds


def make_model(random_state: int, n_estimators: int, max_depth: int, min_samples_leaf: int) -> Pipeline:
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
        ("model", model),
    ])


def fit_predict_binary(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    calibration_method: str = "sigmoid",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    train = train_df.dropna(subset=[label_col])
    cal = cal_df.dropna(subset=[label_col])
    test = test_df.copy()

    # default neutral probabilities if not enough class diversity
    default_test_prob = np.full(len(test), np.nan)
    default_cal_prob = np.full(len(cal_df), np.nan)

    y_train = train[label_col].astype(int)
    y_cal = cal[label_col].astype(int) if not cal.empty else pd.Series(dtype=int)

    diagnostics = {
        "train_rows": int(len(train)),
        "cal_rows": int(len(cal)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(y_train.mean()) if len(y_train) else np.nan,
        "cal_positive_rate": float(y_cal.mean()) if len(y_cal) else np.nan,
        "usable": False,
    }

    if len(train) < 100 or y_train.nunique() < 2:
        return default_cal_prob, default_test_prob, diagnostics

    base = make_model(random_state, n_estimators, max_depth, min_samples_leaf)
    base.fit(train[feature_cols], y_train)

    # raw probabilities
    raw_cal = base.predict_proba(cal_df[feature_cols])[:, 1] if len(cal_df) else default_cal_prob
    raw_test = base.predict_proba(test[feature_cols])[:, 1] if len(test) else default_test_prob

    # calibrate only when calibration set has both classes
    if calibration_method in {"sigmoid", "isotonic"} and len(cal) >= 50 and y_cal.nunique() == 2:
        try:
            calibrated = CalibratedClassifierCV(base, method=calibration_method, cv="prefit")
            calibrated.fit(cal[feature_cols], y_cal)
            cal_prob = calibrated.predict_proba(cal_df[feature_cols])[:, 1]
            test_prob = calibrated.predict_proba(test[feature_cols])[:, 1]
            diagnostics["calibrated"] = True
        except Exception:
            cal_prob = raw_cal
            test_prob = raw_test
            diagnostics["calibrated"] = False
    else:
        cal_prob = raw_cal
        test_prob = raw_test
        diagnostics["calibrated"] = False

    diagnostics["usable"] = True
    return cal_prob, test_prob, diagnostics


def binary_metrics(y_true: pd.Series, prob: pd.Series) -> Dict[str, float]:
    y = pd.Series(y_true).astype(float)
    p = pd.Series(prob).astype(float)
    mask = y.notna() & p.notna()
    y = y[mask].astype(int)
    p = p[mask].astype(float)

    if len(y) == 0:
        return {
            "rows": 0,
            "positive_rate": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "pr_lift": np.nan,
            "pr_ratio": np.nan,
            "brier": np.nan,
            "brier_skill": np.nan,
            "best_f1": np.nan,
        }

    pos_rate = float(y.mean())
    out = {"rows": int(len(y)), "positive_rate": pos_rate}

    if y.nunique() == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        out["pr_lift"] = out["pr_auc"] - pos_rate
        out["pr_ratio"] = safe_divide(out["pr_auc"], pos_rate)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = pos_rate
        out["pr_lift"] = 0.0
        out["pr_ratio"] = 1.0

    try:
        brier = float(brier_score_loss(y, p))
        baseline = float(brier_score_loss(y, np.full(len(y), pos_rate)))
        out["brier"] = brier
        out["brier_skill"] = 1.0 - safe_divide(brier, baseline)
    except Exception:
        out["brier"] = np.nan
        out["brier_skill"] = np.nan

    try:
        precision, recall, thresholds = precision_recall_curve(y, p)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        out["best_f1"] = float(np.nanmax(f1))
    except Exception:
        out["best_f1"] = np.nan

    return out


def threshold_from_calibration(cal_prob: np.ndarray, q: float, default: float = 0.5) -> float:
    s = pd.Series(cal_prob).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return default
    return float(s.quantile(q))


def persistence_filter(signal: pd.Series, mode: str) -> pd.Series:
    s = pd.Series(signal).fillna(0).astype(int)
    if mode == "none":
        return s
    if mode == "2of3":
        return (s.rolling(3, min_periods=1).sum() >= 2).astype(int)
    if mode == "3of5":
        return (s.rolling(5, min_periods=1).sum() >= 3).astype(int)
    raise ValueError(f"unknown persistence mode: {mode}")


# ============================================================
# 3. Strategy Simulation
# ============================================================

def performance_metrics(returns: pd.Series, dates: Optional[pd.Series] = None, periods_per_year: int = 252) -> Dict[str, float]:
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(r) < 2:
        return {
            "rows": int(len(r)),
            "total_return": np.nan,
            "cagr": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "mdd": np.nan,
            "calmar": np.nan,
        }

    curve = (1.0 + r).cumprod()
    total_return = float(curve.iloc[-1] - 1.0)

    if dates is not None and len(dates) == len(r):
        d = pd.to_datetime(dates)
        years = max((d.max() - d.min()).days / 365.25, len(r) / periods_per_year)
    else:
        years = len(r) / periods_per_year

    cagr = float(curve.iloc[-1] ** (1.0 / years) - 1.0) if curve.iloc[-1] > 0 and years > 0 else np.nan
    vol = float(r.std() * math.sqrt(periods_per_year))
    sharpe = float(r.mean() / r.std() * math.sqrt(periods_per_year)) if r.std() > 0 else np.nan
    dd = curve / curve.cummax() - 1.0
    mdd = float(dd.min())
    calmar = safe_divide(cagr, abs(mdd))

    return {
        "rows": int(len(r)),
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
    }


def build_strategy_signals(pred: pd.DataFrame, args) -> Dict[str, pd.Series]:
    h20 = pred["signal_h20"].fillna(0).astype(int)
    exp = pred["signal_expansion"].fillna(0).astype(int)
    down = pred["signal_down_touch"].fillna(0).astype(int)
    up = pred["signal_up_touch"].fillna(0).astype(int)
    bear = pred["signal_beartrend"].fillna(0).astype(int)

    dual_raw = ((h20 == 1) & (exp == 1)).astype(int)

    signals = {
        "dual_highvol_baseline": dual_raw,
        "down_touch_only": down,
        "beartrend_only": bear,
        "dual_and_down_confirm": ((dual_raw == 1) & (down == 1)).astype(int),
        "dual_or_down_early": ((dual_raw == 1) | (down == 1)).astype(int),
        "dual_with_up_veto": ((dual_raw == 1) & ~(up == 1) | ((dual_raw == 1) & (down == 1))).astype(int),
        "dual_or_beartrend": ((dual_raw == 1) | (bear == 1)).astype(int),
        "dual_or_down_or_beartrend": ((dual_raw == 1) | (down == 1) | (bear == 1)).astype(int),
        "down_or_beartrend": ((down == 1) | (bear == 1)).astype(int),
        "dual_and_down_or_beartrend": (((dual_raw == 1) & (down == 1)) | (bear == 1)).astype(int),
    }

    # persistence and execution lag
    out = {}
    for name, sig in signals.items():
        if name in {"dual_highvol_baseline", "dual_with_up_veto"}:
            persist_mode = args.dual_persistence
        elif "beartrend" in name:
            persist_mode = args.bear_persistence
        else:
            persist_mode = args.touch_persistence

        p = persistence_filter(sig, persist_mode)
        out[name] = p.shift(args.execution_lag).fillna(0).astype(int)

    return out


def simulate_strategies(pred: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = pred.copy().sort_values("date").reset_index(drop=True)

    d["equity_ret"] = pd.to_numeric(d["equity_ret"], errors="coerce").fillna(0.0)
    d["bond_ret"] = pd.to_numeric(d.get("bond_ret", 0.0), errors="coerce").fillna(0.0)
    d["cash_ret"] = 0.0

    strategy_signals = build_strategy_signals(d, args)

    daily_rows = []

    # buy hold
    bh = d[["date", "equity_ret", "bond_ret", "cash_ret"]].copy()
    bh["strategy"] = "buy_hold"
    bh["signal"] = 0
    bh["equity_weight"] = 1.0
    bh["bond_weight"] = 0.0
    bh["cash_weight"] = 0.0
    bh["turnover"] = 0.0
    bh["strategy_ret"] = bh["equity_ret"]
    daily_rows.append(bh)

    # 60/40 if bond exists, else equity/cash baseline
    base = d[["date", "equity_ret", "bond_ret", "cash_ret"]].copy()
    base["strategy"] = "static_60_40_bond" if args.use_bond_baseline else "static_60_40_cash"
    base["signal"] = 0
    base["equity_weight"] = 0.6
    base["bond_weight"] = 0.4 if args.use_bond_baseline else 0.0
    base["cash_weight"] = 0.0 if args.use_bond_baseline else 0.4
    base["turnover"] = 0.0
    base["strategy_ret"] = (
        base["equity_weight"] * base["equity_ret"]
        + base["bond_weight"] * base["bond_ret"]
        + base["cash_weight"] * base["cash_ret"]
    )
    daily_rows.append(base)

    for name, signal in strategy_signals.items():
        s = d[["date", "equity_ret", "bond_ret", "cash_ret"]].copy()
        s["strategy"] = name
        s["signal"] = signal.astype(int)

        # signal ON => defensive weight
        s["equity_weight"] = np.where(s["signal"] == 1, args.defensive_equity_weight, 1.0)

        if args.defense_asset == "bond":
            s["bond_weight"] = np.where(s["signal"] == 1, 1.0 - args.defensive_equity_weight, 0.0)
            s["cash_weight"] = 0.0
        elif args.defense_asset == "bond_cash_mix":
            defense = 1.0 - args.defensive_equity_weight
            s["bond_weight"] = np.where(s["signal"] == 1, defense * 0.5, 0.0)
            s["cash_weight"] = np.where(s["signal"] == 1, defense * 0.5, 0.0)
        else:
            s["bond_weight"] = 0.0
            s["cash_weight"] = np.where(s["signal"] == 1, 1.0 - args.defensive_equity_weight, 0.0)

        # turnover: abs change in weights
        w = s[["equity_weight", "bond_weight", "cash_weight"]].astype(float)
        s["turnover"] = w.diff().abs().sum(axis=1).fillna(0.0)
        cost = s["turnover"] * (args.transaction_cost_bps / 10000.0)

        s["strategy_ret"] = (
            s["equity_weight"] * s["equity_ret"]
            + s["bond_weight"] * s["bond_ret"]
            + s["cash_weight"] * s["cash_ret"]
            - cost
        )
        daily_rows.append(s)

    daily = pd.concat(daily_rows, ignore_index=True)

    summary_rows = []
    bh_metrics = performance_metrics(
        daily.loc[daily["strategy"] == "buy_hold", "strategy_ret"],
        daily.loc[daily["strategy"] == "buy_hold", "date"],
    )

    for name, g in daily.groupby("strategy"):
        m = performance_metrics(g["strategy_ret"], g["date"])
        row = {"strategy": name, **m}
        row["avg_equity_weight"] = float(g["equity_weight"].mean())
        row["avg_bond_weight"] = float(g["bond_weight"].mean())
        row["avg_cash_weight"] = float(g["cash_weight"].mean())
        row["signal_rate"] = float(g["signal"].mean())
        row["turnover_total"] = float(g["turnover"].sum())
        row["transaction_cost_total"] = float(g["turnover"].sum() * args.transaction_cost_bps / 10000.0)

        for k in ["total_return", "cagr", "mdd", "calmar", "sharpe", "volatility"]:
            row[f"{k}_diff_vs_buy_hold"] = row.get(k, np.nan) - bh_metrics.get(k, np.nan)

        # Candidate score: prioritizes Calmar/MDD, penalizes huge CAGR drag and excessive turnover.
        row["candidate_score"] = (
            1.50 * max(row.get("calmar_diff_vs_buy_hold", 0.0), -1.0)
            + 1.00 * max(row.get("mdd_diff_vs_buy_hold", 0.0), -1.0)
            + 0.50 * max(row.get("cagr_diff_vs_buy_hold", 0.0), -1.0)
            - 0.005 * row["turnover_total"]
        )

        row["economic_gate"] = bool(
            row.get("calmar_diff_vs_buy_hold", -999) > args.min_calmar_diff
            and row.get("mdd_diff_vs_buy_hold", -999) > args.min_mdd_diff
            and row.get("cagr_diff_vs_buy_hold", -999) > -args.max_cagr_drag
        )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["economic_gate", "candidate_score"], ascending=[False, False])
    return daily, summary


# ============================================================
# 4. Diagnostics
# ============================================================

def signal_timing_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g0 in daily.groupby("strategy"):
        if strategy in {"buy_hold", "static_60_40_bond", "static_60_40_cash"}:
            continue
        g = g0.copy()
        for sig, sg in g.groupby("signal"):
            rows.append({
                "strategy": strategy,
                "signal": int(sig),
                "days": int(len(sg)),
                "day_rate": float(len(sg) / len(g)),
                "avg_strategy_ret": float(sg["strategy_ret"].mean()),
                "median_strategy_ret": float(sg["strategy_ret"].median()),
                "avg_equity_ret": float(sg["equity_ret"].mean()),
                "median_equity_ret": float(sg["equity_ret"].median()),
                "worst_strategy_day": float(sg["strategy_ret"].min()),
                "worst_equity_day": float(sg["equity_ret"].min()),
                "best_strategy_day": float(sg["strategy_ret"].max()),
                "best_equity_day": float(sg["equity_ret"].max()),
                "avg_equity_weight": float(sg["equity_weight"].mean()),
                "avg_cash_weight": float(sg["cash_weight"].mean()),
                "avg_bond_weight": float(sg["bond_weight"].mean()),
            })
    return pd.DataFrame(rows)


def drawdown_series(ret: pd.Series) -> pd.Series:
    curve = (1.0 + pd.Series(ret).fillna(0.0)).cumprod()
    return curve / curve.cummax() - 1.0


def drawdown_event_summary(daily: pd.DataFrame, threshold_values: List[float] = [-0.05, -0.10, -0.15, -0.20]) -> pd.DataFrame:
    # compare each strategy signal coverage in buy_hold drawdown periods
    bh = daily[daily["strategy"] == "buy_hold"][["date", "strategy_ret"]].copy()
    bh = bh.sort_values("date").reset_index(drop=True)
    bh["buy_hold_dd"] = drawdown_series(bh["strategy_ret"])

    rows = []
    for strategy, g0 in daily.groupby("strategy"):
        if strategy in {"buy_hold", "static_60_40_bond", "static_60_40_cash"}:
            continue
        g = g0.sort_values("date").reset_index(drop=True)
        merged = bh.merge(g[["date", "signal", "strategy_ret"]], on="date", how="inner", suffixes=("_bh", "_strategy"))
        for th in threshold_values:
            event_mask = merged["buy_hold_dd"] <= th
            signal_mask = merged["signal"] == 1
            rows.append({
                "strategy": strategy,
                "threshold": th,
                "event_days": int(event_mask.sum()),
                "event_day_rate": float(event_mask.mean()),
                "signal_days_total": int(signal_mask.sum()),
                "signal_day_rate": float(signal_mask.mean()),
                "signal_days_in_event": int((event_mask & signal_mask).sum()),
                "signal_coverage_in_event": safe_divide(int((event_mask & signal_mask).sum()), int(event_mask.sum())),
                "false_alarm_like_rate": safe_divide(int((~event_mask & signal_mask).sum()), int((~event_mask).sum())),
                "avg_strategy_ret_event": float(merged.loc[event_mask, "strategy_ret_strategy"].mean()) if event_mask.any() else np.nan,
                "avg_buy_hold_ret_event": float(merged.loc[event_mask, "strategy_ret_bh"].mean()) if event_mask.any() else np.nan,
            })
    return pd.DataFrame(rows)


# ============================================================
# 5. Main rolling experiment
# ============================================================

def run_experiment(args) -> Dict[str, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    equity = load_ohlcv(args.equity_input)
    equity, feature_cols = build_features_and_labels(
        equity,
        highvol_horizon=args.highvol_horizon,
        touch_horizon=args.touch_horizon,
        bear_horizon=args.bear_horizon,
        vol_window=args.vol_window,
        highvol_lookback=args.highvol_lookback,
        highvol_quantile=args.highvol_quantile,
        expansion_mult=args.expansion_mult,
        touch_k=args.touch_k,
        bear_return_k=args.bear_return_k,
        bear_mdd_k=args.bear_mdd_k,
    )

    # returns for simulation
    equity["equity_ret"] = equity["adj_close"].pct_change().fillna(0.0)

    if args.bond_input:
        try:
            bond = load_ohlcv(args.bond_input)
            bond["bond_ret"] = bond["adj_close"].pct_change().fillna(0.0)
            equity = equity.merge(bond[["date", "bond_ret"]], on="date", how="left")
            equity["bond_ret"] = equity["bond_ret"].fillna(0.0)
            args.use_bond_baseline = True
        except Exception:
            equity["bond_ret"] = 0.0
            args.use_bond_baseline = False
    else:
        equity["bond_ret"] = 0.0
        args.use_bond_baseline = False

    # Drop rows without enough features/labels.
    label_cols = ["y_highvol_h20", "y_highvol_expansion", "y_down_touch", "y_up_touch", "y_beartrend"]
    all_cols = ["date", "equity_ret", "bond_ret"] + feature_cols + label_cols
    data = equity[all_cols].replace([np.inf, -np.inf], np.nan).copy()

    if args.start_date:
        data = data[data["date"] >= pd.to_datetime(args.start_date)].copy()
    if args.end_date:
        data = data[data["date"] <= pd.to_datetime(args.end_date)].copy()

    data = data.dropna(subset=["date", "equity_ret"]).reset_index(drop=True)

    folds = make_rolling_folds(
        n=len(data),
        train_window=args.train_window,
        calibration_window=args.calibration_window,
        test_window=args.test_window,
        embargo=args.embargo,
    )

    if not folds:
        raise RuntimeError(
            f"No folds generated. rows={len(data)}, train={args.train_window}, "
            f"cal={args.calibration_window}, test={args.test_window}, embargo={args.embargo}"
        )

    head_specs = {
        "h20": "y_highvol_h20",
        "expansion": "y_highvol_expansion",
        "down_touch": "y_down_touch",
        "up_touch": "y_up_touch",
        "beartrend": "y_beartrend",
    }

    pred_parts = []
    fold_metric_rows = []
    threshold_rows = []

    for fold in folds:
        train_df = data.iloc[fold.train_start:fold.train_end].copy()
        cal_df = data.iloc[fold.cal_start:fold.cal_end].copy()
        test_df = data.iloc[fold.test_start:fold.test_end].copy()

        pred = test_df[["date", "equity_ret", "bond_ret"]].copy()
        pred["fold_id"] = fold.fold_id

        for head, label_col in head_specs.items():
            cal_prob, test_prob, diag = fit_predict_binary(
                train_df=train_df,
                cal_df=cal_df,
                test_df=test_df,
                feature_cols=feature_cols,
                label_col=label_col,
                random_state=args.random_state + fold.fold_id,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                calibration_method=args.calibration_method,
            )

            if head == "h20":
                q = args.h20_threshold_quantile
            elif head == "expansion":
                q = args.expansion_threshold_quantile
            elif head == "down_touch":
                q = args.down_touch_threshold_quantile
            elif head == "up_touch":
                q = args.up_touch_threshold_quantile
            else:
                q = args.beartrend_threshold_quantile

            threshold = threshold_from_calibration(cal_prob, q=q, default=args.default_threshold)

            pred[f"prob_{head}"] = test_prob
            pred[f"y_{head}"] = test_df[label_col].values
            pred[f"signal_{head}"] = (pd.Series(test_prob) >= threshold).astype(int).values

            # metrics
            metrics = binary_metrics(test_df[label_col], pd.Series(test_prob, index=test_df.index))
            fold_metric_rows.append({
                "fold_id": fold.fold_id,
                "head": head,
                "label_col": label_col,
                "threshold": threshold,
                **diag,
                **metrics,
            })
            threshold_rows.append({
                "fold_id": fold.fold_id,
                "head": head,
                "threshold_quantile": q,
                "threshold": threshold,
            })

        pred_parts.append(pred)

    oos_pred = pd.concat(pred_parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    thresholds = pd.DataFrame(threshold_rows)

    # head summary
    head_summary_rows = []
    for head, g in fold_metrics.groupby("head"):
        head_summary_rows.append({
            "head": head,
            "fold_count": int(g["fold_id"].nunique()),
            "mean_positive_rate": float(g["positive_rate"].mean()),
            "median_positive_rate": float(g["positive_rate"].median()),
            "mean_pr_auc": float(g["pr_auc"].mean()),
            "median_pr_auc": float(g["pr_auc"].median()),
            "mean_pr_lift": float(g["pr_lift"].mean()),
            "median_pr_lift": float(g["pr_lift"].median()),
            "positive_pr_lift_rate": float((g["pr_lift"] > 0).mean()),
            "mean_brier_skill": float(g["brier_skill"].mean()),
            "median_brier_skill": float(g["brier_skill"].median()),
            "positive_brier_skill_rate": float((g["brier_skill"] > 0).mean()),
            "mean_threshold": float(g["threshold"].mean()),
        })
    head_summary = pd.DataFrame(head_summary_rows).sort_values("head").reset_index(drop=True)

    daily, strategy_summary = simulate_strategies(oos_pred, args)
    signal_summary = signal_timing_summary(daily)
    dd_summary = drawdown_event_summary(daily)

    # best candidate excluding benchmark
    candidate_summary = strategy_summary[
        ~strategy_summary["strategy"].isin(["buy_hold", "static_60_40_bond", "static_60_40_cash"])
    ].copy()
    best = candidate_summary.sort_values(["economic_gate", "candidate_score"], ascending=[False, False]).head(1)
    best_dict = best.to_dict("records")[0] if not best.empty else {}

    baseline = candidate_summary[candidate_summary["strategy"].eq("dual_highvol_baseline")].head(1)
    baseline_dict = baseline.to_dict("records")[0] if not baseline.empty else {}

    decision = {
        "experiment": "riskoff_free_touch_head_validator",
        "asset_name": args.asset_name,
        "equity_input": str(args.equity_input),
        "bond_input": str(args.bond_input) if args.bond_input else None,
        "period": {
            "oos_start": str(oos_pred["date"].min().date()),
            "oos_end": str(oos_pred["date"].max().date()),
            "rows": int(len(oos_pred)),
            "fold_count": int(len(folds)),
        },
        "config": {
            "riskoff_removed": True,
            "highvol_horizon": args.highvol_horizon,
            "touch_horizon": args.touch_horizon,
            "bear_horizon": args.bear_horizon,
            "touch_k": args.touch_k,
            "expansion_mult": args.expansion_mult,
            "defensive_equity_weight": args.defensive_equity_weight,
            "defense_asset": args.defense_asset,
            "transaction_cost_bps": args.transaction_cost_bps,
            "execution_lag": args.execution_lag,
        },
        "baseline_dual_highvol": baseline_dict,
        "best_candidate": best_dict,
        "head_summary": head_summary.to_dict("records"),
        "decision_note": (
            "If a touch/bear candidate improves MDD/Calmar without unacceptable CAGR drag, "
            "it can replace or augment the current Dual-HighVol overlay. "
            "RiskOff is not used as a head or trigger in this experiment."
        ),
    }

    outputs = {
        "summary": save_json(out_dir / "riskoff_free_touch_summary.json", decision),
        "strategy_summary": save_csv(out_dir / "strategy_summary.csv", strategy_summary),
        "strategy_daily_returns": save_csv(out_dir / "strategy_daily_returns.csv", daily),
        "head_summary": save_csv(out_dir / "head_summary.csv", head_summary),
        "fold_metrics": save_csv(out_dir / "fold_metrics.csv", fold_metrics),
        "oos_predictions": save_csv(out_dir / "oos_predictions.csv", oos_pred),
        "thresholds": save_csv(out_dir / "thresholds.csv", thresholds),
        "signal_timing_summary": save_csv(out_dir / "signal_timing_summary.csv", signal_summary),
        "drawdown_event_summary": save_csv(out_dir / "drawdown_event_summary.csv", dd_summary),
    }
    return outputs


# ============================================================
# 6. CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--equity-input", required=True)
    parser.add_argument("--bond-input", default="")
    parser.add_argument("--asset-name", default="QQQ")
    parser.add_argument("--output-dir", default="riskoff_free_touch_output")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")

    # windows
    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=40)

    # labels
    parser.add_argument("--highvol-horizon", type=int, default=20)
    parser.add_argument("--touch-horizon", type=int, default=20)
    parser.add_argument("--bear-horizon", type=int, default=60)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--highvol-lookback", type=int, default=252)
    parser.add_argument("--highvol-quantile", type=float, default=0.75)
    parser.add_argument("--expansion-mult", type=float, default=1.25)
    parser.add_argument("--touch-k", type=float, default=0.75)
    parser.add_argument("--bear-return-k", type=float, default=0.75)
    parser.add_argument("--bear-mdd-k", type=float, default=1.25)

    # thresholds
    parser.add_argument("--h20-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--expansion-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--down-touch-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--up-touch-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--beartrend-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--default-threshold", type=float, default=0.5)

    # model
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--calibration-method", choices=["sigmoid", "isotonic", "none"], default="sigmoid")
    parser.add_argument("--random-state", type=int, default=42)

    # strategy
    parser.add_argument("--defensive-equity-weight", type=float, default=0.60)
    parser.add_argument("--defense-asset", choices=["cash", "bond", "bond_cash_mix"], default="cash")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--execution-lag", type=int, default=1)
    parser.add_argument("--dual-persistence", choices=["none", "2of3", "3of5"], default="3of5")
    parser.add_argument("--touch-persistence", choices=["none", "2of3", "3of5"], default="2of3")
    parser.add_argument("--bear-persistence", choices=["none", "2of3", "3of5"], default="3of5")

    # economic gate
    parser.add_argument("--min-calmar-diff", type=float, default=0.03)
    parser.add_argument("--min-mdd-diff", type=float, default=0.03)
    parser.add_argument("--max-cagr-drag", type=float, default=0.02)

    args = parser.parse_args()
    args.use_bond_baseline = False

    outputs = run_experiment(args)

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    print("[OK] RiskOff-free touch head validation completed.")
    print(json.dumps({
        "asset_name": summary["asset_name"],
        "period": summary["period"],
        "baseline_dual_highvol": {
            "cagr": summary.get("baseline_dual_highvol", {}).get("cagr"),
            "mdd": summary.get("baseline_dual_highvol", {}).get("mdd"),
            "calmar": summary.get("baseline_dual_highvol", {}).get("calmar"),
            "cagr_diff_vs_buy_hold": summary.get("baseline_dual_highvol", {}).get("cagr_diff_vs_buy_hold"),
            "mdd_diff_vs_buy_hold": summary.get("baseline_dual_highvol", {}).get("mdd_diff_vs_buy_hold"),
            "calmar_diff_vs_buy_hold": summary.get("baseline_dual_highvol", {}).get("calmar_diff_vs_buy_hold"),
        },
        "best_candidate": {
            "strategy": summary.get("best_candidate", {}).get("strategy"),
            "cagr": summary.get("best_candidate", {}).get("cagr"),
            "mdd": summary.get("best_candidate", {}).get("mdd"),
            "calmar": summary.get("best_candidate", {}).get("calmar"),
            "cagr_diff_vs_buy_hold": summary.get("best_candidate", {}).get("cagr_diff_vs_buy_hold"),
            "mdd_diff_vs_buy_hold": summary.get("best_candidate", {}).get("mdd_diff_vs_buy_hold"),
            "calmar_diff_vs_buy_hold": summary.get("best_candidate", {}).get("calmar_diff_vs_buy_hold"),
            "economic_gate": summary.get("best_candidate", {}).get("economic_gate"),
        },
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
