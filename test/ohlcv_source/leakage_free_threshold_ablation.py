# -*- coding: utf-8 -*-
"""
leakage_free_threshold_ablation.py

HighVol/RiskOff threshold leakage 제거 재검증 스크립트.

목적
----
이전 allocation ablation의 핵심 취약점은 threshold를 holdout 구간의 예측확률 분위수로 계산했다는 점입니다.
이 파일은 threshold를 train 내부 calibration 구간에서만 계산한 뒤,
그 고정 threshold를 holdout 구간에 적용합니다.

수행 실험
---------
1. HighVol 후보 재학습
   - 기본: H20 + down_core + ExtraTrees + sigmoid
2. RiskOff 후보 재학습
   - 기본: H40 + k_mdd=2.0 + down_core + ExtraTrees + sigmoid
3. threshold 산출
   - train 내부 calibration 구간의 calibrated probability 분위수 사용
   - q ∈ {0.65, 0.70, 0.75, 0.80, 0.85}
4. holdout 적용
   - HighVol only
   - RiskOff only
   - HighVol + RiskOff
   - Buy & Hold
   - Constant NORMAL
   - 60/40, bond CSV가 있을 때
5. 성과 비교
   - CAGR
   - MDD
   - Calmar
   - Sharpe
   - turnover
   - transaction cost

입력
----
필수:
- --equity-input QQQ_ohlcv.csv

선택:
- --bond-input IEF_ohlcv.csv

실행 예시
--------
python leakage_free_threshold_ablation.py ^
  --equity-input QQQ_ohlcv.csv ^
  --bond-input IEF_ohlcv.csv ^
  --ticker QQQ ^
  --bond-ticker IEF ^
  --output-dir leakage_free_threshold_results ^
  --holdout-start 2023-01-01 ^
  --threshold-quantiles 0.65,0.70,0.75,0.80,0.85 ^
  --transaction-cost-bps 10

smoke test:
python leakage_free_threshold_ablation.py --smoke-test

주의
----
- 이 코드는 "threshold leakage 제거"를 위한 실험 코드입니다.
- 이 결과가 좋아도 Stable 채택 전 walk-forward threshold, 다른 holdout 기간, PBO/multiple-testing 검토가 필요합니다.
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
from sklearn.preprocessing import RobustScaler


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


def make_synthetic_ohlcv(n: int = 1000, seed: int = 42, ticker: str = "QQQ") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-01", periods=n)

    vol = np.full(n, 0.011)
    drift = np.full(n, 0.00035)

    for start, end, local_vol, local_drift in [
        (250, 310, 0.028, -0.0010),
        (580, 650, 0.026, -0.0008),
        (780, 830, 0.030, -0.0012),
    ]:
        vol[start:end] = local_vol
        drift[start:end] = local_drift

    ret = rng.normal(drift, vol)
    close = 100 * np.cumprod(1.0 + ret)
    volume = rng.integers(1_000_000, 9_000_000, n)

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
        volu = out["volume"].astype(float)
        vol_mean = volu.rolling(20, min_periods=5).mean()
        vol_std = volu.rolling(20, min_periods=5).std()
        out["volume_change_20d"] = volu.pct_change(20)
        out["volume_zscore_20d"] = (volu - vol_mean) / vol_std
    else:
        out["volume_change_20d"] = np.nan
        out["volume_zscore_20d"] = np.nan

    return out


FEATURE_SETS = {
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
    k_mdd: float = 1.5,
    high_vol_quantile: float = 0.75,
    high_vol_lookback: int = 252,
) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].astype(float)
    returns = close.pct_change()

    daily_vol_t = returns.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().shift(1)
    current_horizon_vol = daily_vol_t * math.sqrt(horizon)

    future_realized_vol_h = compute_forward_realized_vol(returns, horizon)
    future_mdd_h = compute_forward_mdd(close, horizon)

    risk_off_threshold = -k_mdd * current_horizon_vol
    high_vol_threshold = current_horizon_vol.rolling(
        high_vol_lookback,
        min_periods=max(30, high_vol_lookback // 4),
    ).quantile(high_vol_quantile)

    y_high_vol = (future_realized_vol_h >= high_vol_threshold).astype(float)
    y_risk_off = (future_mdd_h <= risk_off_threshold).astype(float)

    invalid = (
        current_horizon_vol.isna()
        | future_realized_vol_h.isna()
        | future_mdd_h.isna()
        | high_vol_threshold.isna()
    )

    out["future_realized_vol_h"] = future_realized_vol_h
    out["future_mdd_h"] = future_mdd_h
    out["meta_current_horizon_vol"] = current_horizon_vol
    out["meta_high_vol_threshold"] = high_vol_threshold
    out["y_high_vol"] = y_high_vol.mask(invalid, np.nan)
    out["y_risk_off"] = y_risk_off.mask(invalid, np.nan)
    return out


# ============================================================
# 3. Model / Calibration
# ============================================================

def make_extratrees(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=150,
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


@dataclass
class HeadConfig:
    name: str
    target_col: str
    horizon: int
    feature_set: str
    k_mdd: float
    high_vol_quantile: float
    calibration_method: str = "sigmoid"


def split_train_cal_holdout(
    df: pd.DataFrame,
    holdout_start: str,
    calibration_frac: float = 0.20,
    min_cal_rows: int = 120,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    holdout_date = pd.to_datetime(holdout_start)
    train_all_mask = df["date"] < holdout_date
    holdout_mask = df["date"] >= holdout_date

    train_idx = df.index[train_all_mask].to_numpy()
    cal_size = max(min_cal_rows, int(len(train_idx) * calibration_frac))

    if len(train_idx) - cal_size < 300:
        raise ValueError("train rows too small after calibration split")

    core_idx = train_idx[:-cal_size]
    cal_idx = train_idx[-cal_size:]

    core_mask = pd.Series(False, index=df.index)
    cal_mask = pd.Series(False, index=df.index)
    holdout_mask_s = pd.Series(holdout_mask.to_numpy(), index=df.index)

    core_mask.loc[core_idx] = True
    cal_mask.loc[cal_idx] = True
    return core_mask, cal_mask, holdout_mask_s


def fit_head_predictions(
    df_features: pd.DataFrame,
    cfg: HeadConfig,
    holdout_start: str,
    vol_window: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    labeled = build_labels(
        df_features,
        horizon=cfg.horizon,
        vol_window=vol_window,
        k_mdd=cfg.k_mdd,
        high_vol_quantile=cfg.high_vol_quantile,
    )
    feature_cols = select_features(labeled, cfg.feature_set)
    required = ["date", "close", cfg.target_col] + feature_cols
    data = labeled[required].dropna(subset=[cfg.target_col] + feature_cols).copy()

    core_mask, cal_mask, holdout_mask = split_train_cal_holdout(data, holdout_start)

    core = data[core_mask].copy()
    cal = data[cal_mask].copy()
    holdout = data[holdout_mask].copy()

    if core[cfg.target_col].nunique() < 2:
        raise ValueError(f"{cfg.name}: core train has one class")

    model = make_extratrees(random_state=random_state)
    model.fit(core[feature_cols], core[cfg.target_col].astype(int))

    raw_cal = model.predict_proba(cal[feature_cols])[:, 1]
    raw_holdout = model.predict_proba(holdout[feature_cols])[:, 1]

    # RiskOff처럼 희소 이벤트는 calibration 구간에 한 클래스만 존재할 수 있음.
    # 이 경우 sigmoid/isotonic 보정은 불가능하므로 raw probability를 그대로 사용한다.
    calibrator = ProbabilityCalibrator(cfg.calibration_method)
    if cal[cfg.target_col].nunique() >= 2:
        calibrator.fit(raw_cal, cal[cfg.target_col].astype(int).to_numpy())
        p_cal_cal = calibrator.transform(raw_cal)
        p_cal_holdout = calibrator.transform(raw_holdout)
        calibration_status = "calibrated"
    else:
        p_cal_cal = raw_cal
        p_cal_holdout = raw_holdout
        calibration_status = "skipped_single_class_calibration_set"

    cal_pred = cal[["date", "close", cfg.target_col]].copy().rename(columns={cfg.target_col: "y_true"})
    cal_pred["prob_raw"] = raw_cal
    cal_pred["prob_cal"] = p_cal_cal
    cal_pred["head_name"] = cfg.name

    holdout_pred = holdout[["date", "close", cfg.target_col]].copy().rename(columns={cfg.target_col: "y_true"})
    holdout_pred["prob_raw"] = raw_holdout
    holdout_pred["prob_cal"] = p_cal_holdout
    holdout_pred["head_name"] = cfg.name

    meta = {
        "head_name": cfg.name,
        "target_col": cfg.target_col,
        "horizon": cfg.horizon,
        "feature_set": cfg.feature_set,
        "k_mdd": cfg.k_mdd,
        "high_vol_quantile": cfg.high_vol_quantile,
        "feature_count": len(feature_cols),
        "core_rows": int(len(core)),
        "calibration_rows": int(len(cal)),
        "holdout_rows": int(len(holdout)),
        "core_positive_rate": float(core[cfg.target_col].mean()),
        "calibration_positive_rate": float(cal[cfg.target_col].mean()),
        "holdout_positive_rate": float(holdout[cfg.target_col].mean()),
        "calibration_status": calibration_status,
    }
    return cal_pred, holdout_pred, meta


# ============================================================
# 4. Metrics / Allocation
# ============================================================

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
        "prob_mean": float(p.mean()),
        "prob_std": float(p.std()),
    }


def align_returns(equity_df: pd.DataFrame, bond_df: Optional[pd.DataFrame], start_date: pd.Timestamp) -> pd.DataFrame:
    eq = equity_df[equity_df["date"] >= start_date][["date", "close"]].copy().rename(columns={"close": "equity_close"})
    eq["equity_ret"] = eq["equity_close"].pct_change().fillna(0.0)

    if bond_df is not None:
        bd = bond_df[bond_df["date"] >= start_date][["date", "close"]].copy().rename(columns={"close": "bond_close"})
        bd["bond_ret"] = bd["bond_close"].pct_change().fillna(0.0)
        out = eq.merge(bd[["date", "bond_ret"]], on="date", how="left")
        out["bond_ret"] = out["bond_ret"].fillna(0.0)
    else:
        out = eq.copy()
        out["bond_ret"] = 0.0

    out["cash_ret"] = 0.0
    return out


def performance_metrics(equity_curve: np.ndarray, returns: np.ndarray, periods_per_year: int = 252) -> Dict[str, float]:
    curve = pd.Series(equity_curve, dtype=float)
    ret = pd.Series(returns, dtype=float)

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


def build_signal_panel(hv_holdout: pd.DataFrame, ro_holdout: pd.DataFrame) -> pd.DataFrame:
    hv = hv_holdout[["date", "prob_cal"]].copy().rename(columns={"prob_cal": "p_high_vol"})
    ro = ro_holdout[["date", "prob_cal"]].copy().rename(columns={"prob_cal": "p_risk_off"})
    sig = hv.merge(ro, on="date", how="outer").sort_values("date").reset_index(drop=True)
    return sig


def simulate_strategy(
    ret_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    strategy: str,
    hv_threshold: Optional[float],
    ro_threshold: Optional[float],
    transaction_cost_bps: float,
) -> Dict[str, object]:
    df = ret_df.merge(signal_df, on="date", how="left")
    df["p_high_vol"] = df["p_high_vol"].ffill()
    df["p_risk_off"] = df["p_risk_off"].ffill()

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
        if hv_threshold is not None:
            hv = (df["p_high_vol"] >= hv_threshold).fillna(False).to_numpy()
            equity_w[hv] = 0.60
            cash_w[hv] = 0.40

    elif strategy == "riskoff_only":
        equity_w[:] = 1.0
        if ro_threshold is not None:
            ro = (df["p_risk_off"] >= ro_threshold).fillna(False).to_numpy()
            equity_w[ro] = 0.30
            cash_w[ro] = 0.70

    elif strategy == "highvol_riskoff":
        equity_w[:] = 1.0
        hv = (df["p_high_vol"] >= hv_threshold).fillna(False).to_numpy() if hv_threshold is not None else np.zeros(n, dtype=bool)
        ro = (df["p_risk_off"] >= ro_threshold).fillna(False).to_numpy() if ro_threshold is not None else np.zeros(n, dtype=bool)
        equity_w[hv] = 0.60
        cash_w[hv] = 0.40
        equity_w[ro] = 0.30
        cash_w[ro] = 0.70

    else:
        raise ValueError(f"unknown strategy: {strategy}")

    total_w = equity_w + bond_w + cash_w
    equity_w = equity_w / total_w
    bond_w = bond_w / total_w
    cash_w = cash_w / total_w

    turnover = np.zeros(n)
    turnover[1:] = np.abs(np.diff(equity_w)) + np.abs(np.diff(bond_w)) + np.abs(np.diff(cash_w))
    cost = turnover * (transaction_cost_bps / 10000.0)

    gross_ret = equity_w * df["equity_ret"].to_numpy() + bond_w * df["bond_ret"].to_numpy() + cash_w * df["cash_ret"].to_numpy()
    net_ret = gross_ret - cost
    curve = np.cumprod(1.0 + net_ret)

    return {
        "strategy": strategy,
        "rows": int(n),
        "avg_equity_weight": float(np.mean(equity_w)),
        "avg_bond_weight": float(np.mean(bond_w)),
        "avg_cash_weight": float(np.mean(cash_w)),
        "turnover_total": float(np.sum(turnover)),
        "transaction_cost_total": float(np.sum(cost)),
        **performance_metrics(curve, net_ret),
    }


# ============================================================
# 5. Experiment
# ============================================================

def run_experiment(
    equity_df: pd.DataFrame,
    bond_df: Optional[pd.DataFrame],
    output_dir: str | Path,
    holdout_start: str,
    threshold_quantiles: Sequence[float],
    transaction_cost_bps: float,
    vol_window: int,
    random_state: int,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_features = build_features(equity_df)

    hv_cfg = HeadConfig(
        name="highvol_h20_down_extra_sigmoid",
        target_col="y_high_vol",
        horizon=20,
        feature_set="down_core",
        k_mdd=1.5,
        high_vol_quantile=0.75,
        calibration_method="sigmoid",
    )
    ro_cfg = HeadConfig(
        name="riskoff_h40_k2_down_extra_sigmoid",
        target_col="y_risk_off",
        horizon=40,
        feature_set="down_core",
        k_mdd=2.0,
        high_vol_quantile=0.75,
        calibration_method="sigmoid",
    )

    hv_cal, hv_holdout, hv_meta = fit_head_predictions(df_features, hv_cfg, holdout_start, vol_window, random_state)
    ro_cal, ro_holdout, ro_meta = fit_head_predictions(df_features, ro_cfg, holdout_start, vol_window, random_state)

    hv_cal_metrics = binary_metrics(hv_cal["y_true"], hv_cal["prob_cal"])
    hv_holdout_metrics = binary_metrics(hv_holdout["y_true"], hv_holdout["prob_cal"])
    ro_cal_metrics = binary_metrics(ro_cal["y_true"], ro_cal["prob_cal"])
    ro_holdout_metrics = binary_metrics(ro_holdout["y_true"], ro_holdout["prob_cal"])

    threshold_rows = []
    allocation_rows = []

    signal_panel = build_signal_panel(hv_holdout, ro_holdout)
    start_date = pd.to_datetime(holdout_start)
    ret_df = align_returns(equity_df, bond_df, start_date=start_date)

    # benchmark는 q와 무관하므로 q=NaN으로 1회 저장
    for strategy in ["buy_hold", "constant_normal"]:
        allocation_rows.append({
            "threshold_quantile": np.nan,
            "threshold_source": "none",
            "hv_threshold": np.nan,
            "ro_threshold": np.nan,
            **simulate_strategy(ret_df, signal_panel, strategy, None, None, transaction_cost_bps),
        })

    if bond_df is not None:
        allocation_rows.append({
            "threshold_quantile": np.nan,
            "threshold_source": "none",
            "hv_threshold": np.nan,
            "ro_threshold": np.nan,
            **simulate_strategy(ret_df, signal_panel, "sixty_forty", None, None, transaction_cost_bps),
        })

    for q in threshold_quantiles:
        hv_th = float(hv_cal["prob_cal"].quantile(q))
        ro_th = float(ro_cal["prob_cal"].quantile(q))

        threshold_rows.append({
            "threshold_quantile": float(q),
            "threshold_source": "calibration_set_only",
            "hv_threshold": hv_th,
            "ro_threshold": ro_th,
            "hv_cal_signal_rate": float((hv_cal["prob_cal"] >= hv_th).mean()),
            "hv_holdout_signal_rate": float((hv_holdout["prob_cal"] >= hv_th).mean()),
            "ro_cal_signal_rate": float((ro_cal["prob_cal"] >= ro_th).mean()),
            "ro_holdout_signal_rate": float((ro_holdout["prob_cal"] >= ro_th).mean()),
        })

        for strategy in ["highvol_only", "riskoff_only", "highvol_riskoff"]:
            allocation_rows.append({
                "threshold_quantile": float(q),
                "threshold_source": "calibration_set_only",
                "hv_threshold": hv_th,
                "ro_threshold": ro_th,
                **simulate_strategy(ret_df, signal_panel, strategy, hv_th, ro_th, transaction_cost_bps),
            })

    threshold_df = pd.DataFrame(threshold_rows)
    allocation_df = pd.DataFrame(allocation_rows).sort_values("calmar", ascending=False, na_position="last").reset_index(drop=True)

    head_metrics_df = pd.DataFrame([
        {"head": "highvol", "split": "calibration", **hv_meta, **hv_cal_metrics},
        {"head": "highvol", "split": "holdout", **hv_meta, **hv_holdout_metrics},
        {"head": "riskoff", "split": "calibration", **ro_meta, **ro_cal_metrics},
        {"head": "riskoff", "split": "holdout", **ro_meta, **ro_holdout_metrics},
    ])

    outputs = {
        "head_metrics": save_csv(output_dir / "head_metrics_calibration_vs_holdout.csv", head_metrics_df),
        "thresholds": save_csv(output_dir / "fixed_thresholds_from_calibration.csv", threshold_df),
        "allocation": save_csv(output_dir / "leakage_free_allocation_ablation.csv", allocation_df),
        "highvol_calibration_predictions": save_csv(output_dir / "highvol_calibration_predictions.csv", hv_cal),
        "highvol_holdout_predictions": save_csv(output_dir / "highvol_holdout_predictions.csv", hv_holdout),
        "riskoff_calibration_predictions": save_csv(output_dir / "riskoff_calibration_predictions.csv", ro_cal),
        "riskoff_holdout_predictions": save_csv(output_dir / "riskoff_holdout_predictions.csv", ro_holdout),
    }

    summary = {
        "experiment": "leakage_free_threshold_ablation",
        "threshold_source": "calibration_set_only",
        "holdout_start": holdout_start,
        "transaction_cost_bps": transaction_cost_bps,
        "threshold_quantiles": list(map(float, threshold_quantiles)),
        "highvol_head": hv_meta,
        "riskoff_head": ro_meta,
        "highvol_holdout_metrics": hv_holdout_metrics,
        "riskoff_holdout_metrics": ro_holdout_metrics,
        "best_allocation_by_calmar": allocation_df.head(1).to_dict("records")[0] if not allocation_df.empty else None,
        "top_allocation_rows": allocation_df.head(10).to_dict("records"),
        "critical_note": (
            "thresholds are computed only from the calibration split before holdout, "
            "not from the holdout probability distribution."
        ),
    }
    outputs["summary"] = save_json(output_dir / "leakage_free_threshold_summary.json", summary)
    return outputs


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equity-input", default="")
    parser.add_argument("--bond-input", default="")
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--bond-ticker", default="IEF")
    parser.add_argument("--output-dir", default="leakage_free_threshold_results")
    parser.add_argument("--holdout-start", default="2023-01-01")
    parser.add_argument("--threshold-quantiles", default="0.65,0.70,0.75,0.80,0.85")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        equity_df = make_synthetic_ohlcv(n=1000, seed=42, ticker=args.ticker)
        bond_df = make_synthetic_ohlcv(n=1000, seed=7, ticker=args.bond_ticker)
        bond_df["close"] = 100 * np.cumprod(1 + np.random.default_rng(7).normal(0.00005, 0.003, len(bond_df)))
        holdout_start = str(equity_df.iloc[int(len(equity_df) * 0.75)]["date"].date())
    else:
        if not args.equity_input:
            raise ValueError("--equity-input is required unless --smoke-test is used")
        equity_df = load_ohlcv(args.equity_input)
        bond_df = load_ohlcv(args.bond_input) if args.bond_input else None
        holdout_start = args.holdout_start

    outputs = run_experiment(
        equity_df=equity_df,
        bond_df=bond_df,
        output_dir=args.output_dir,
        holdout_start=holdout_start,
        threshold_quantiles=parse_float_list(args.threshold_quantiles),
        transaction_cost_bps=args.transaction_cost_bps,
        vol_window=args.vol_window,
        random_state=args.random_state,
    )

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    best = summary.get("best_allocation_by_calmar", {})

    print("[OK] Leakage-free threshold ablation completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print(json.dumps(
        {
            "threshold_source": summary["threshold_source"],
            "holdout_start": summary["holdout_start"],
            "highvol_holdout_pr_auc": summary["highvol_holdout_metrics"].get("pr_auc"),
            "highvol_holdout_brier_skill": summary["highvol_holdout_metrics"].get("brier_skill"),
            "riskoff_holdout_pr_auc": summary["riskoff_holdout_metrics"].get("pr_auc"),
            "riskoff_holdout_brier_skill": summary["riskoff_holdout_metrics"].get("brier_skill"),
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
