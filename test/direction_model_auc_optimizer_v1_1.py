"""
Direction Model AUC Optimizer v1
================================

목적
- 포트폴리오/자산배분 성과를 잠시 제외하고, 방향성 분류 모델 자체의 ROC-AUC / PR-AUC를 최대한 끌어올릴 수 있는
  라벨·horizon·feature set·모델 조합을 walk-forward 방식으로 탐색한다.
- 기존 v8.6.x 백테스트 코드의 feature builder를 재사용하되, allocation 로직은 사용하지 않는다.

핵심 실험
1) 기존 방식: up_vs_rest / down_vs_rest
2) 중립 제거 방식: up_vs_down, 즉 강한 상승과 강한 하락만 비교
3) simple return threshold sweep
4) volatility-adjusted return threshold sweep
5) approximate triple-barrier label sweep
6) low-vol train/eval filter 옵션
7) feature set별 비교: pruned_all, trend_core, trend_volume, down_core, vol_risk_core
8) model별 비교: xgb, hgb, extratrees, logistic

필요 패키지
    pip install pandas numpy scikit-learn yfinance xgboost

실행 예시
    py direction_model_auc_optimizer_v1.py --profile quick
    py direction_model_auc_optimizer_v1.py --profile balanced
    py direction_model_auc_optimizer_v1.py --profile full

    # 기존 v8.6.2 feature builder 사용
    py direction_model_auc_optimizer_v1.py --base-script xgb_multi_branch_directional_risk_v8_6_2.py --profile balanced

    # 이미 저장한 OHLCV CSV 사용. 컬럼: Date, Open, High, Low, Close, Volume
    py direction_model_auc_optimizer_v1.py --ohlcv-csv qqq_ohlcv.csv --profile balanced

출력
    results_direction_auc_lab/direction_auc_trials.csv
    results_direction_auc_lab/direction_auc_trials_top20.csv
    results_direction_auc_lab/direction_auc_best_predictions.csv
    results_direction_auc_lab/direction_auc_summary.json

중요
- feature[t]로 future_return[t+h]를 예측한다.
- walk-forward split마다 train 구간만으로 모델을 학습한다.
- 이 스크립트는 allocation 성과를 보지 않는다.
- ROC-AUC 0.5 이상을 목표로 하되, PR-AUC는 반드시 base_rate 대비 lift로 판단한다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    XGBClassifier = None
    HAS_XGB = False


@dataclass(frozen=True)
class TrialConfig:
    trial_id: str
    horizon: int
    task: str  # up_vs_rest, down_vs_rest, up_vs_down, barrier_up_vs_down
    label_mode: str  # simple, vol_adj, barrier
    threshold: float
    barrier_mult: float
    feature_set: str
    model_type: str
    train_filter: str  # all, low_vol_only, non_extreme_vol
    eval_filter: str  # all, low_vol_only, non_extreme_vol
    min_train_rows: int
    retrain_every_n_days: int
    max_train_rows: Optional[int]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_base_script(base_script: str) -> Path:
    """Resolve base model script robustly on Windows/macOS/Linux.

    The first release used a /mnt/data/... default path, which only exists in the
    ChatGPT sandbox. This resolver searches the current working directory, the
    optimizer script directory, and a few parent directories so the script works
    when copied into a local project folder.
    """
    raw = str(base_script).strip().strip('"').strip("'")
    p = Path(raw)

    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(Path(__file__).resolve().parent / p)
        # Useful when optimizer is in ML_project/test and base script is in ML_project.
        for parent in Path(__file__).resolve().parents[:5]:
            candidates.append(parent / p.name)

    # If user kept the old sandbox default, fall back to local filenames.
    if raw.replace("\\", "/").endswith("xgb_multi_branch_pruned_features_v8_6_5.py"):
        for name in [
            "xgb_multi_branch_pruned_features_v8_6_5.py",
            "xgb_multi_branch_directional_risk_v8_6_2.py",
        ]:
            candidates.append(Path.cwd() / name)
            candidates.append(Path(__file__).resolve().parent / name)
            for parent in Path(__file__).resolve().parents[:5]:
                candidates.append(parent / name)

    seen = set()
    unique_candidates: List[Path] = []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen:
            unique_candidates.append(c)
            seen.add(key)

    for c in unique_candidates:
        if c.exists() and c.is_file():
            return c.resolve()

    searched = "\n  - ".join(str(c) for c in unique_candidates[:30])
    raise FileNotFoundError(
        "base script not found. Put xgb_multi_branch_pruned_features_v8_6_5.py "
        "or xgb_multi_branch_directional_risk_v8_6_2.py in the same folder, "
        "or pass --base-script with the exact Windows path.\n"
        f"requested: {base_script}\nsearched:\n  - {searched}"
    )


def load_base_module(base_script: str):
    path = resolve_base_script(base_script)
    spec = importlib.util.spec_from_file_location("base_model_features", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import base script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["base_model_features"] = module
    spec.loader.exec_module(module)
    return module


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return MultiIndex. Prefer first level if it contains OHLCV names.
        if set(["Open", "High", "Low", "Close", "Volume"]).issubset(set(df.columns.get_level_values(0))):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    return df


def download_ohlcv_local(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance가 필요합니다. pip install yfinance 후 실행하거나 --ohlcv-csv를 사용하세요.") from e
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = _flatten_yf_columns(df)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"downloaded OHLCV missing columns: {missing}")
    return df[required].dropna().copy()


def load_ohlcv(args: argparse.Namespace, base_module: Any) -> pd.DataFrame:
    if args.ohlcv_csv:
        df = pd.read_csv(args.ohlcv_csv)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"--ohlcv-csv missing columns: {missing}")
        return df[required].sort_index().dropna().copy()

    if hasattr(base_module, "download_ohlcv"):
        return base_module.download_ohlcv(args.ticker, args.start_date, args.end_date)
    return download_ohlcv_local(args.ticker, args.start_date, args.end_date)


def safe_auc(y_true: np.ndarray, prob: np.ndarray, kind: str) -> Optional[float]:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    mask = np.isfinite(prob) & np.isfinite(y_true)
    y_true = y_true[mask].astype(int)
    prob = prob[mask].astype(float)
    if len(y_true) < 20 or len(np.unique(y_true)) < 2:
        return None
    try:
        if kind == "roc":
            return float(roc_auc_score(y_true, prob))
        if kind == "pr":
            return float(average_precision_score(y_true, prob))
    except Exception:
        return None
    raise ValueError(kind)


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    mask = np.isfinite(prob) & np.isfinite(y_true)
    y = np.asarray(y_true)[mask].astype(int)
    p = np.asarray(prob)[mask].astype(float)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return None, None
    precision, recall, thresholds = precision_recall_curve(y, p)
    if thresholds.size == 0:
        return None, None
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def rolling_rank_last(series: pd.Series, window: int) -> pd.Series:
    def _rank(x: np.ndarray) -> float:
        if len(x) == 0 or not np.isfinite(x[-1]):
            return np.nan
        return float(np.mean(x <= x[-1]))
    return series.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=True)


def add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "realized_vol_20" in out.columns:
        vol = out["realized_vol_20"]
    elif "daily_return" in out.columns:
        vol = out["daily_return"].rolling(20).std()
        out["realized_vol_20"] = vol
    else:
        vol = out["Close"].pct_change().rolling(20).std()
        out["realized_vol_20"] = vol
    out["realized_vol_20_rank_252"] = rolling_rank_last(vol, 252)
    return out


def build_feature_sets(feature_cols: Sequence[str]) -> Dict[str, List[str]]:
    available = set(feature_cols)
    def keep(cols: Sequence[str]) -> List[str]:
        return [c for c in cols if c in available]

    trend_core = keep([
        "return_10d", "return_20d", "return_60d", "return_120d",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_20", "positive_return_ratio_60",
        "trend_consistency_20", "trend_consistency_60",
        "price_position_20", "price_position_60", "close_to_20d_high", "close_to_60d_high",
        "large_up_day_ratio_20", "large_down_day_ratio_20",
    ])

    trend_volume = keep(trend_core + [
        "volume_change", "volume_ratio_20", "volume_zscore_20",
        "down_volume_ratio_20", "high_volume_down_ratio_20", "volume_shock_20", "volume_shock_rank_252",
    ])

    down_core = keep([
        "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "drawdown_20", "drawdown_60", "drawdown_120",
        "price_position_20", "price_position_60",
        "close_to_20d_high", "close_to_60d_high",
        "large_down_day_ratio_20", "lower_high_20", "bearish_ma_stack",
        "positive_return_ratio_20", "positive_return_ratio_60",
        "trend_consistency_20", "trend_consistency_60",
        "volume_ratio_20", "volume_zscore_20", "down_volume_ratio_20", "high_volume_down_ratio_20", "volume_shock_20",
    ])

    vol_risk_core = keep([
        "true_range_pct", "atr_pct_14", "atr_pct_20", "atr_pct_60", "atr_rank_252",
        "atr_ratio_14_60", "atr_ratio_20_60", "atr_accel_5",
        "realized_vol_20", "realized_vol_60", "ewma_vol_20", "ewma_vol_60",
        "parkinson_vol_20", "parkinson_vol_60", "garman_klass_vol_20", "rogers_satchell_vol_20",
        "yang_zhang_vol_20", "yang_zhang_vol_60",
        "downside_vol_20", "downside_vol_60", "semi_vol_20",
        "ulcer_index_20", "ulcer_index_60", "ulcer_rank_252",
        "bb_width_20", "bb_width_rank_252", "keltner_width_20", "vol_of_vol_20",
        "drawdown_20", "drawdown_60", "drawdown_120", "large_down_day_ratio_20",
    ])

    compact_mixed = keep([
        "return_20d", "return_60d", "return_120d",
        "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_60", "ma200_slope_60", "positive_return_ratio_60",
        "drawdown_20", "drawdown_60", "drawdown_120", "price_position_60", "close_to_60d_high",
        "volume_ratio_20", "volume_zscore_20", "down_volume_ratio_20", "volume_shock_rank_252",
        "atr_pct_14", "atr_pct_20", "atr_rank_252", "ewma_vol_20", "semi_vol_20", "ulcer_index_20",
    ])

    return {
        "pruned_all": list(feature_cols),
        "trend_core": trend_core,
        "trend_volume": trend_volume,
        "down_core": down_core,
        "vol_risk_core": vol_risk_core,
        "compact_mixed": compact_mixed,
    }


def expected_horizon_vol(df: pd.DataFrame, horizon: int) -> pd.Series:
    vol = df.get("realized_vol_20")
    if vol is None:
        vol = df["Close"].pct_change().rolling(20).std()
    return vol * math.sqrt(max(horizon, 1))


def make_binary_label(df: pd.DataFrame, cfg: TrialConfig) -> Tuple[pd.Series, pd.Series]:
    """Return (label, valid_mask). label is 0/1 where valid_mask is True."""
    h = cfg.horizon
    ret_col = f"future_return_{h}d"
    max_col = f"future_max_return_{h}d"
    min_col = f"future_min_return_{h}d"
    if ret_col not in df.columns:
        raise KeyError(f"missing {ret_col}. Check horizons passed to build_features.")

    ret = df[ret_col]
    y = pd.Series(np.nan, index=df.index, dtype=float)
    valid = pd.Series(False, index=df.index)

    if cfg.label_mode == "simple":
        thr = cfg.threshold
        up = ret > thr
        down = ret < -thr
    elif cfg.label_mode == "vol_adj":
        scale = expected_horizon_vol(df, h).replace(0, np.nan)
        z = ret / scale
        thr = cfg.threshold
        up = z > thr
        down = z < -thr
    elif cfg.label_mode == "barrier":
        # Approximate triple-barrier: if both barriers are touched in the horizon, use the stronger excursion.
        # This is not exact first-touch ordering but is useful as a fast screening label.
        scale = expected_horizon_vol(df, h).replace(0, np.nan)
        up_bar = cfg.barrier_mult * scale
        dn_bar = -cfg.barrier_mult * scale
        fmax = df[max_col]
        fmin = df[min_col]
        up_touch = fmax >= up_bar
        down_touch = fmin <= dn_bar
        # tie-breaker by stronger standardized excursion
        up_strength = fmax / scale
        down_strength = (-fmin) / scale
        up = up_touch & (~down_touch | (up_strength >= down_strength))
        down = down_touch & (~up_touch | (down_strength > up_strength))
    else:
        raise ValueError(cfg.label_mode)

    if cfg.task == "up_vs_rest":
        y = up.astype(float)
        valid = ret.notna()
    elif cfg.task == "down_vs_rest":
        y = down.astype(float)
        valid = ret.notna()
    elif cfg.task in ("up_vs_down", "barrier_up_vs_down"):
        # Exclude ambiguous neutral/no-touch observations.
        y.loc[up] = 1.0
        y.loc[down] = 0.0
        valid = (up | down) & ret.notna()
    else:
        raise ValueError(cfg.task)

    valid = valid & y.notna()
    return y.astype(float), valid.astype(bool)


def mask_by_filter(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "all":
        return pd.Series(True, index=df.index)
    rank = df.get("realized_vol_20_rank_252")
    if rank is None:
        rank = rolling_rank_last(df["realized_vol_20"], 252)
    if mode == "low_vol_only":
        return rank <= 0.65
    if mode == "non_extreme_vol":
        return rank <= 0.80
    raise ValueError(mode)


def scale_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos <= 0:
        return 1.0
    return float(np.clip(neg / pos, 0.25, 8.0))


def make_model(model_type: str, y_train: np.ndarray, random_state: int, n_jobs: int) -> Pipeline:
    spw = scale_pos_weight(y_train)
    if model_type == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed. Use --models hgb,extratrees,logistic or install xgboost.")
        clf = XGBClassifier(
            n_estimators=120,
            learning_rate=0.025,
            max_depth=2,
            min_child_weight=8.0,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_lambda=10.0,
            reg_alpha=0.2,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_type == "xgb_mid":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed.")
        clf = XGBClassifier(
            n_estimators=160,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=10.0,
            subsample=0.80,
            colsample_bytree=0.75,
            reg_lambda=12.0,
            reg_alpha=0.3,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_type == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.15,
            min_samples_leaf=35,
            random_state=random_state,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_type == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=350,
            max_depth=5,
            min_samples_leaf=25,
            max_features="sqrt",
            class_weight="balanced_subsample",
            bootstrap=False,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=350,
            max_depth=5,
            min_samples_leaf=25,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_type == "logistic":
        clf = LogisticRegression(
            C=0.25,
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", clf),
        ])

    raise ValueError(model_type)


def predict_proba_1(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fallback for decision_function models
    s = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-s))


def walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: TrialConfig,
    random_state: int,
    n_jobs: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    y, label_valid = make_binary_label(df, cfg)
    train_filter_mask = mask_by_filter(df, cfg.train_filter)
    eval_filter_mask = mask_by_filter(df, cfg.eval_filter)
    usable_features = [c for c in feature_cols if c in df.columns]
    if len(usable_features) == 0:
        raise ValueError(f"empty feature set: {cfg.feature_set}")

    n = len(df)
    prob = pd.Series(np.nan, index=df.index, dtype=float)
    fold_count = 0
    trained_rows_total = 0
    skipped_folds = 0

    # Predict in blocks. Retrain only at block boundaries.
    start_i = cfg.min_train_rows
    while start_i < n:
        end_i = min(start_i + cfg.retrain_every_n_days, n)
        train_end = start_i - cfg.horizon  # purge by horizon
        if train_end <= 0:
            start_i = end_i
            continue
        train_start = 0 if cfg.max_train_rows is None else max(0, train_end - cfg.max_train_rows)
        train_idx = df.index[train_start:train_end]
        test_idx = df.index[start_i:end_i]

        train_mask = label_valid.loc[train_idx] & train_filter_mask.loc[train_idx]
        train_idx2 = train_idx[train_mask.values]
        if len(train_idx2) < 300 or len(np.unique(y.loc[train_idx2].astype(int))) < 2:
            skipped_folds += 1
            start_i = end_i
            continue

        X_train = df.loc[train_idx2, usable_features]
        y_train = y.loc[train_idx2].astype(int).values
        X_test = df.loc[test_idx, usable_features]
        try:
            model = make_model(cfg.model_type, y_train, random_state=random_state + fold_count, n_jobs=n_jobs)
            model.fit(X_train, y_train)
            prob.loc[test_idx] = predict_proba_1(model, X_test)
            fold_count += 1
            trained_rows_total += len(train_idx2)
        except Exception as e:
            skipped_folds += 1
        start_i = end_i

    out = pd.DataFrame(
        {
            "date": df.index,
            "y_true": y.values,
            "prob": prob.values,
            "label_valid": label_valid.values,
            "eval_filter": eval_filter_mask.values,
        },
        index=df.index,
    )
    meta = {
        "fold_count": fold_count,
        "skipped_folds": skipped_folds,
        "avg_train_rows": (trained_rows_total / fold_count) if fold_count else 0,
        "feature_count": len(usable_features),
    }
    return out, meta


def evaluate_predictions(pred: pd.DataFrame, cfg: TrialConfig, meta: Dict[str, Any]) -> Dict[str, Any]:
    mask = pred["label_valid"].astype(bool) & pred["eval_filter"].astype(bool) & pred["prob"].notna()
    y = pred.loc[mask, "y_true"].astype(int).values
    p = pred.loc[mask, "prob"].astype(float).values
    if len(y) < 50 or len(np.unique(y)) < 2:
        return {
            **asdict(cfg),
            **meta,
            "eval_rows": int(len(y)),
            "base_rate": None,
            "roc_auc": None,
            "pr_auc": None,
            "pr_lift": None,
            "inverse_roc_auc": None,
            "inverse_pr_auc": None,
            "best_roc_after_inversion": None,
            "probability_polarity": "insufficient",
            "best_f1": None,
            "best_f1_threshold": None,
            "brier": None,
        }

    roc = safe_auc(y, p, "roc")
    pr = safe_auc(y, p, "pr")
    inv_roc = safe_auc(y, 1.0 - p, "roc")
    inv_pr = safe_auc(y, 1.0 - p, "pr")
    base = float(np.mean(y))
    thr, bf1 = best_f1_threshold(y, p)
    try:
        brier = float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    except Exception:
        brier = None

    best_roc = None
    polarity = "normal"
    if roc is not None and inv_roc is not None:
        if inv_roc > roc + 0.015:
            polarity = "inverse_better"
        elif roc > inv_roc + 0.015:
            polarity = "normal_better"
        else:
            polarity = "ambiguous"
        best_roc = max(roc, inv_roc)

    # Score: prioritize ROC > 0.5 and PR lift. Penalize inverse-better because it is a sign of unstable polarity.
    score = 0.0
    if roc is not None:
        score += 2.0 * (roc - 0.5)
    if pr is not None:
        score += 1.5 * (pr - base)
    if bf1 is not None:
        score += 0.2 * bf1
    if polarity == "inverse_better":
        score -= 0.05

    return {
        **asdict(cfg),
        **meta,
        "eval_rows": int(len(y)),
        "positive_count": int(np.sum(y == 1)),
        "negative_count": int(np.sum(y == 0)),
        "base_rate": base,
        "roc_auc": roc,
        "pr_auc": pr,
        "pr_lift": (pr - base) if pr is not None else None,
        "inverse_roc_auc": inv_roc,
        "inverse_pr_auc": inv_pr,
        "best_roc_after_inversion": best_roc,
        "probability_polarity": polarity,
        "best_f1": bf1,
        "best_f1_threshold": thr,
        "brier": brier,
        "score": float(score),
    }


def parse_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def make_trial_grid(args: argparse.Namespace, feature_sets: Dict[str, List[str]]) -> List[TrialConfig]:
    horizons = [int(x) for x in parse_csv_arg(args.horizons)]
    models = parse_csv_arg(args.models)

    if args.profile == "quick":
        tasks = ["up_vs_down", "up_vs_rest", "down_vs_rest"]
        label_specs = [("simple", 0.01, 1.0), ("simple", 0.015, 1.0), ("vol_adj", 0.50, 1.0)]
        feature_names = [n for n in ["trend_volume", "down_core", "compact_mixed"] if n in feature_sets]
        filters = [("all", "all"), ("non_extreme_vol", "all")]
        models = [m for m in models if m in ["xgb", "hgb", "extratrees", "logistic"]]
    elif args.profile == "balanced":
        tasks = ["up_vs_down", "up_vs_rest", "down_vs_rest"]
        label_specs = [
            ("simple", 0.005, 1.0), ("simple", 0.01, 1.0), ("simple", 0.015, 1.0),
            ("vol_adj", 0.35, 1.0), ("vol_adj", 0.50, 1.0), ("vol_adj", 0.75, 1.0),
            ("barrier", 0.0, 0.75), ("barrier", 0.0, 1.00),
        ]
        feature_names = [n for n in ["trend_core", "trend_volume", "down_core", "vol_risk_core", "compact_mixed", "pruned_all"] if n in feature_sets]
        filters = [("all", "all"), ("non_extreme_vol", "all"), ("low_vol_only", "low_vol_only")]
    else:
        tasks = ["up_vs_down", "up_vs_rest", "down_vs_rest", "barrier_up_vs_down"]
        label_specs = [
            ("simple", 0.0, 1.0), ("simple", 0.005, 1.0), ("simple", 0.01, 1.0), ("simple", 0.015, 1.0), ("simple", 0.02, 1.0),
            ("vol_adj", 0.25, 1.0), ("vol_adj", 0.35, 1.0), ("vol_adj", 0.50, 1.0), ("vol_adj", 0.75, 1.0), ("vol_adj", 1.00, 1.0),
            ("barrier", 0.0, 0.75), ("barrier", 0.0, 1.00), ("barrier", 0.0, 1.25), ("barrier", 0.0, 1.50),
        ]
        feature_names = list(feature_sets.keys())
        filters = [("all", "all"), ("non_extreme_vol", "all"), ("non_extreme_vol", "non_extreme_vol"), ("low_vol_only", "low_vol_only")]

    trials: List[TrialConfig] = []
    k = 0
    for h in horizons:
        for task in tasks:
            for label_mode, threshold, barrier_mult in label_specs:
                if task == "barrier_up_vs_down" and label_mode != "barrier":
                    continue
                if label_mode == "barrier" and task in ["up_vs_rest", "down_vs_rest"]:
                    # allowed, but often less useful. Keep only in full.
                    if args.profile != "full":
                        continue
                for fs in feature_names:
                    for model in models:
                        if model.startswith("xgb") and not HAS_XGB:
                            continue
                        for train_filter, eval_filter in filters:
                            k += 1
                            trials.append(
                                TrialConfig(
                                    trial_id=f"t{k:04d}",
                                    horizon=h,
                                    task=task,
                                    label_mode=label_mode,
                                    threshold=threshold,
                                    barrier_mult=barrier_mult,
                                    feature_set=fs,
                                    model_type=model,
                                    train_filter=train_filter,
                                    eval_filter=eval_filter,
                                    min_train_rows=args.min_train_rows,
                                    retrain_every_n_days=args.retrain_every_n_days,
                                    max_train_rows=args.max_train_rows,
                                )
                            )
    if args.max_trials and len(trials) > args.max_trials:
        trials = trials[: args.max_trials]
    return trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-script", default="xgb_multi_branch_pruned_features_v8_6_5.py")
    parser.add_argument("--ohlcv-csv", default=None)
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--start-date", default="1999-03-10")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--backtest-start-date", default="2013-01-02")
    parser.add_argument("--horizons", default="10,20,40,60,120")
    parser.add_argument("--profile", choices=["quick", "balanced", "full"], default="quick")
    parser.add_argument("--models", default="xgb,hgb,extratrees,logistic")
    parser.add_argument("--min-train-rows", type=int, default=756)
    parser.add_argument("--retrain-every-n-days", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--result-dir", default="results_direction_auc_lab")
    args = parser.parse_args()

    out_dir = Path(args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"base script 로드 요청: {args.base_script}")
    base_script_path = resolve_base_script(args.base_script)
    log(f"base script 실제 경로: {base_script_path}")
    base_module = load_base_module(str(base_script_path))

    horizons = sorted(set(int(x) for x in parse_csv_arg(args.horizons)))
    log(f"OHLCV 로드: {args.ticker}")
    ohlcv = load_ohlcv(args, base_module)

    if not hasattr(base_module, "build_features"):
        raise RuntimeError("base script must expose build_features(ohlcv, horizons)")

    log(f"피처 생성: horizons={horizons}")
    df, feature_cols = base_module.build_features(ohlcv, horizons)
    df = add_helper_columns(df)
    df = df[df.index >= pd.to_datetime(args.backtest_start_date)].copy()
    # Keep rows with at least basic features; future labels are allowed to be NaN near the end and will be masked.
    log(f"rows={len(df)}, base_features={len(feature_cols)}")

    feature_sets = build_feature_sets(feature_cols)
    for k, v in feature_sets.items():
        log(f"feature_set {k}: {len(v)} features")

    trials = make_trial_grid(args, feature_sets)
    log(f"trial_count={len(trials)}, profile={args.profile}, HAS_XGB={HAS_XGB}")

    rows: List[Dict[str, Any]] = []
    best_pred: Optional[pd.DataFrame] = None
    best_row: Optional[Dict[str, Any]] = None

    for i, cfg in enumerate(trials, start=1):
        if i == 1 or i % 10 == 0 or i == len(trials):
            log(f"trial {i}/{len(trials)}: {cfg.trial_id} h{cfg.horizon} {cfg.task} {cfg.label_mode} thr={cfg.threshold} bar={cfg.barrier_mult} fs={cfg.feature_set} model={cfg.model_type} train={cfg.train_filter} eval={cfg.eval_filter}")
        fs_cols = feature_sets[cfg.feature_set]
        try:
            pred, meta = walk_forward_predict(df, fs_cols, cfg, args.random_state, args.n_jobs)
            row = evaluate_predictions(pred, cfg, meta)
        except Exception as e:
            row = {**asdict(cfg), "error": str(e), "score": -999.0}
        rows.append(row)

        if row.get("score") is not None and np.isfinite(row.get("score", np.nan)):
            if best_row is None or row["score"] > best_row.get("score", -999):
                best_row = row
                if "pred" in locals():
                    best_pred = pred.copy()
                    best_pred["trial_id"] = cfg.trial_id

    res = pd.DataFrame(rows)
    # Sort robustly: score first, then ROC, then PR lift.
    sort_cols = [c for c in ["score", "roc_auc", "pr_lift", "pr_auc", "best_f1"] if c in res.columns]
    res = res.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    res.to_csv(out_dir / "direction_auc_trials.csv", index=False, encoding="utf-8-sig")
    res.head(20).to_csv(out_dir / "direction_auc_trials_top20.csv", index=False, encoding="utf-8-sig")

    if best_pred is not None:
        best_pred.to_csv(out_dir / "direction_auc_best_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": args.ticker,
        "period": {"start": str(df.index.min()), "end": str(df.index.max()), "rows": int(len(df))},
        "profile": args.profile,
        "trial_count": int(len(trials)),
        "base_script": args.base_script,
        "base_feature_count": int(len(feature_cols)),
        "feature_set_counts": {k: len(v) for k, v in feature_sets.items()},
        "best_trial": best_row,
        "top10": res.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "notes": [
            "PR-AUC는 base_rate보다 얼마나 높은지 pr_lift를 함께 봐야 합니다.",
            "probability_polarity가 inverse_better이면 신호 방향 반전 가능성을 의심하되, fold별 일관성을 추가 확인해야 합니다.",
            "up_vs_down은 중립 샘플을 제외하므로 전체 날짜 예측용이 아니라 방향성이 뚜렷한 구간 탐지용입니다.",
        ],
    }
    with open(out_dir / "direction_auc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("완료")
    log(f"저장: {out_dir / 'direction_auc_trials.csv'}")
    if best_row:
        log("best_trial:")
        print(json.dumps(best_row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
