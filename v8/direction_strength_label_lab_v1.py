r"""
Direction Strength Label Lab v1
===============================

목적
- 기존 UP/DOWN 이진 라벨 대신, 미래 수익률 방향과 추세 강화/약화 여부를 결합한 5-class 라벨을 실험한다.
- 포트폴리오 성과는 보지 않고, 분류 성능만 walk-forward 방식으로 평가한다.

라벨
    UP_STRENGTHENING
    UP_WEAKENING
    DOWN_STRENGTHENING
    DOWN_WEAKENING
    SIDEWAYS

핵심 아이디어
- r_h의 단순 부호가 아니라 volatility-adjusted return threshold를 사용한다.
- trend_delta는 라벨 생성에만 사용하고 feature로는 사용하지 않는다.
- multiclass 모델을 학습한 뒤, 클래스별 one-vs-rest ROC-AUC / PR-AUC / PR lift를 출력한다.

필요 패키지
    pip install pandas numpy scikit-learn yfinance xgboost

실행 예시
    py direction_strength_label_lab_v1.py --profile quick
    py direction_strength_label_lab_v1.py --profile balanced
    py direction_strength_label_lab_v1.py --base-script ..\xgb_multi_branch_pruned_features_v8_6_5.py --profile balanced
    py direction_strength_label_lab_v1.py --ohlcv-csv qqq_ohlcv.csv --profile quick

출력
    results_direction_strength_lab/direction_strength_trials.csv
    results_direction_strength_lab/direction_strength_trials_top20.csv
    results_direction_strength_lab/direction_strength_best_predictions.csv
    results_direction_strength_lab/direction_strength_summary.json
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, label_binarize

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    XGBClassifier = None
    HAS_XGB = False


LABELS = [
    "UP_STRENGTHENING",
    "UP_WEAKENING",
    "DOWN_STRENGTHENING",
    "DOWN_WEAKENING",
    "SIDEWAYS",
]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class StrengthTrialConfig:
    trial_id: str
    horizon: int
    ret_eps_k: float
    strength_eps: float
    strength_method: str       # score_delta, slope_delta, hybrid_delta
    feature_set: str
    model_type: str            # xgb, hgb, extratrees, rf, logistic
    train_filter: str          # all, low_vol_only, non_extreme_vol
    eval_filter: str           # all, low_vol_only, non_extreme_vol
    min_train_rows: int
    retrain_every_n_days: int
    max_train_rows: Optional[int]
    drop_sideways_train: bool


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def resolve_base_script(base_script: str) -> Path:
    raw = str(base_script).strip().strip('"').strip("'")
    p = Path(raw)
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(Path(__file__).resolve().parent / p)
        for parent in Path(__file__).resolve().parents[:6]:
            candidates.append(parent / p.name)

    # local fallback names
    for name in [
        "xgb_multi_branch_pruned_features_v8_6_5.py",
        "xgb_multi_branch_directional_risk_v8_6_2.py",
        "xgb_policy_lab_v8_6_6.py",
    ]:
        candidates.append(Path.cwd() / name)
        candidates.append(Path(__file__).resolve().parent / name)
        for parent in Path(__file__).resolve().parents[:6]:
            candidates.append(parent / name)

    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    for c in uniq:
        if c.exists() and c.is_file():
            return c.resolve()

    searched = "\n  - ".join(str(c) for c in uniq[:40])
    raise FileNotFoundError(
        "base script not found. Put xgb_multi_branch_pruned_features_v8_6_5.py "
        "in the same folder, or pass --base-script with the exact path.\n"
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


def rolling_rank_last(series: pd.Series, window: int) -> pd.Series:
    def _rank(x: np.ndarray) -> float:
        if len(x) == 0 or not np.isfinite(x[-1]):
            return np.nan
        return float(np.mean(x <= x[-1]))
    return series.rolling(window, min_periods=max(20, window // 4)).apply(_rank, raw=True)


def add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "daily_return" not in out.columns:
        out["daily_return"] = out["Close"].pct_change()
    if "realized_vol_20" not in out.columns:
        out["realized_vol_20"] = out["daily_return"].rolling(20).std()
    out["realized_vol_20_rank_252"] = rolling_rank_last(out["realized_vol_20"], 252)

    # Add missing robust trend helper columns if the base script did not create them.
    close = out["Close"]
    for w in [20, 60, 120, 200]:
        ma_col = f"ma_{w}"
        if ma_col not in out.columns:
            out[ma_col] = close.rolling(w, min_periods=max(5, w // 4)).mean()
        gap_col = f"price_ma_{w}_gap"
        if gap_col not in out.columns:
            out[gap_col] = close / out[ma_col] - 1.0
    if "return_60d" not in out.columns:
        out["return_60d"] = close.pct_change(60)
    if "return_120d" not in out.columns:
        out["return_120d"] = close.pct_change(120)
    if "ma_gap_20_60" not in out.columns:
        out["ma_gap_20_60"] = out["ma_20"] / out["ma_60"] - 1.0
    if "ma_gap_60_120" not in out.columns:
        out["ma_gap_60_120"] = out["ma_60"] / out["ma_120"] - 1.0
    if "trend_slope_60" not in out.columns:
        out["trend_slope_60"] = close.pct_change(60) / 60.0
    if "ma200_slope_60" not in out.columns:
        out["ma200_slope_60"] = out["ma_200"].pct_change(60) / 60.0
    return out


def expected_horizon_vol(df: pd.DataFrame, horizon: int) -> pd.Series:
    vol = df.get("realized_vol_20")
    if vol is None:
        vol = df["Close"].pct_change().rolling(20).std()
    return vol * math.sqrt(max(horizon, 1))


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

    strength_core = keep([
        "return_20d", "return_60d", "return_120d",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "positive_return_ratio_60", "trend_consistency_60",
        "price_position_60", "close_to_60d_high",
        "volume_ratio_20", "volume_zscore_20", "volume_shock_rank_252",
        "atr_rank_252", "realized_vol_20", "ewma_vol_20", "ulcer_index_20",
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
        "strength_core": strength_core,
        "compact_mixed": compact_mixed,
    }


def trend_score_components(df: pd.DataFrame) -> pd.DataFrame:
    """Current-time trend score. This can be shifted forward for label generation.

    This function only uses columns available at time t. The future value is created
    via shift(-h) and is used only as a target label component, never as a feature.
    """
    comps = pd.DataFrame(index=df.index)
    comps["ret60_pos"] = (df.get("return_60d", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ret120_pos"] = (df.get("return_120d", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ma60_gap_pos"] = (df.get("price_ma_60_gap", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ma120_gap_pos"] = (df.get("price_ma_120_gap", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["ma20_60_pos"] = (df.get("ma_gap_20_60", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    comps["slope60_pos"] = (df.get("trend_slope_60", pd.Series(np.nan, index=df.index)) > 0).astype(float)
    return comps


def current_trend_score(df: pd.DataFrame) -> pd.Series:
    comps = trend_score_components(df)
    return comps.sum(axis=1)


def current_trend_continuous(df: pd.DataFrame) -> pd.Series:
    """Continuous trend proxy for delta calculation."""
    idx = df.index
    parts = []
    for col in ["return_60d", "return_120d", "price_ma_60_gap", "price_ma_120_gap", "ma_gap_20_60", "trend_slope_60"]:
        if col in df.columns:
            s = df[col].replace([np.inf, -np.inf], np.nan)
            scale = s.rolling(252, min_periods=60).std().replace(0, np.nan)
            parts.append((s / scale).clip(-3, 3))
    if not parts:
        return pd.Series(np.nan, index=idx)
    return pd.concat(parts, axis=1).mean(axis=1)


def build_strength_label(df: pd.DataFrame, cfg: StrengthTrialConfig) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    h = cfg.horizon
    ret_col = f"future_return_{h}d"
    if ret_col not in df.columns:
        raise KeyError(f"missing {ret_col}. Check horizons passed to build_features.")

    r_h = df[ret_col]
    vol_h = expected_horizon_vol(df, h).replace(0, np.nan)
    ret_eps = cfg.ret_eps_k * vol_h

    trend_score_t = current_trend_score(df)
    trend_score_f = trend_score_t.shift(-h)
    score_delta = trend_score_f - trend_score_t

    trend_cont_t = current_trend_continuous(df)
    trend_cont_f = trend_cont_t.shift(-h)
    slope_delta = trend_cont_f - trend_cont_t

    if cfg.strength_method == "score_delta":
        trend_delta = score_delta
    elif cfg.strength_method == "slope_delta":
        trend_delta = slope_delta
    elif cfg.strength_method == "hybrid_delta":
        # Score delta captures directional regime change; continuous delta captures magnitude.
        trend_delta = 0.65 * score_delta + 0.35 * slope_delta
    else:
        raise ValueError(cfg.strength_method)

    y_label = pd.Series("SIDEWAYS", index=df.index, dtype=object)
    valid = r_h.notna() & vol_h.notna() & trend_delta.notna()

    up = r_h >= ret_eps
    down = r_h <= -ret_eps
    strengthening_up = trend_delta > cfg.strength_eps
    weakening_up = trend_delta <= cfg.strength_eps
    strengthening_down = trend_delta < -cfg.strength_eps
    weakening_down = trend_delta >= -cfg.strength_eps

    y_label.loc[valid & up & strengthening_up] = "UP_STRENGTHENING"
    y_label.loc[valid & up & weakening_up] = "UP_WEAKENING"
    y_label.loc[valid & down & strengthening_down] = "DOWN_STRENGTHENING"
    y_label.loc[valid & down & weakening_down] = "DOWN_WEAKENING"
    y_label.loc[valid & ~(up | down)] = "SIDEWAYS"

    aux = pd.DataFrame({
        "future_return": r_h,
        "ret_eps": ret_eps,
        "trend_score_t": trend_score_t,
        "trend_score_future": trend_score_f,
        "score_delta": score_delta,
        "slope_delta": slope_delta,
        "trend_delta": trend_delta,
        "label_name": y_label,
    }, index=df.index)

    y_id = y_label.map(LABEL_TO_ID).astype(float)
    return y_id, valid.astype(bool), aux


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


def class_weight_dict(y: np.ndarray) -> Dict[int, float]:
    values, counts = np.unique(y.astype(int), return_counts=True)
    n = len(y)
    k = len(values)
    weights = {}
    for v, c in zip(values, counts):
        if c <= 0:
            weights[int(v)] = 1.0
        else:
            weights[int(v)] = float(np.clip(n / (k * c), 0.25, 8.0))
    return weights


def make_model(model_type: str, y_train: np.ndarray, random_state: int, n_jobs: int) -> Pipeline:
    n_classes = len(np.unique(y_train.astype(int)))
    if n_classes < 2:
        raise ValueError("need at least 2 classes")

    if model_type == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed. Use --models hgb,extratrees,logistic or install xgboost.")
        clf = XGBClassifier(
            n_estimators=160,
            learning_rate=0.025,
            max_depth=2,
            min_child_weight=8.0,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_lambda=10.0,
            reg_alpha=0.2,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_type == "xgb_mid":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed.")
        clf = XGBClassifier(
            n_estimators=220,
            learning_rate=0.018,
            max_depth=3,
            min_child_weight=10.0,
            subsample=0.80,
            colsample_bytree=0.75,
            reg_lambda=12.0,
            reg_alpha=0.3,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_type == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.20,
            min_samples_leaf=35,
            random_state=random_state,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_type == "extratrees":
        clf = ExtraTreesClassifier(
            n_estimators=450,
            max_depth=6,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            bootstrap=False,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=450,
            max_depth=6,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])

    if model_type == "logistic":
        clf = LogisticRegression(
            C=0.35,
            l1_ratio=0.0,
            solver="lbfgs",
            max_iter=2500,
            class_weight="balanced",
            multi_class="auto",
            random_state=random_state,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", clf),
        ])

    raise ValueError(model_type)


def align_proba_columns(model: Pipeline, proba: np.ndarray) -> np.ndarray:
    """Return probability matrix with 5 fixed class columns."""
    clf = model.named_steps.get("clf")
    # XGBoost is fitted on fold-local contiguous labels; original_label_ids_ maps
    # each probability column back to the global label id. Other sklearn models
    # keep original class ids in classes_.
    classes = getattr(clf, "original_label_ids_", getattr(clf, "classes_", np.arange(proba.shape[1])))
    out = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
    for j, cls in enumerate(classes):
        if int(cls) in ID_TO_LABEL and j < proba.shape[1]:
            out[:, int(cls)] = proba[:, j]
    row_sum = out.sum(axis=1)
    missing = row_sum <= 0
    if missing.any():
        out[missing, :] = 1.0 / len(LABELS)
    else:
        out = out / np.maximum(row_sum[:, None], 1e-12)
    return out


def predict_proba_5(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("model does not support predict_proba")
    return align_proba_columns(model, model.predict_proba(X))


def walk_forward_predict_multiclass(
    df: pd.DataFrame,
    feature_cols: List[str],
    cfg: StrengthTrialConfig,
    random_state: int,
    n_jobs: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    y, label_valid, aux = build_strength_label(df, cfg)
    train_filter_mask = mask_by_filter(df, cfg.train_filter)
    eval_filter_mask = mask_by_filter(df, cfg.eval_filter)
    usable_features = [c for c in feature_cols if c in df.columns]
    if len(usable_features) == 0:
        raise ValueError(f"empty feature set: {cfg.feature_set}")

    n = len(df)
    prob_mat = pd.DataFrame(np.nan, index=df.index, columns=[f"prob_{name}" for name in LABELS], dtype=float)
    fold_count = 0
    skipped_folds = 0
    trained_rows_total = 0

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
        if cfg.drop_sideways_train:
            train_mask = train_mask & (y.loc[train_idx] != LABEL_TO_ID["SIDEWAYS"])
        train_idx2 = train_idx[train_mask.values]

        if len(train_idx2) < 300:
            skipped_folds += 1
            start_i = end_i
            continue
        y_train = y.loc[train_idx2].astype(int).values
        if len(np.unique(y_train)) < 2:
            skipped_folds += 1
            start_i = end_i
            continue

        X_train = df.loc[train_idx2, usable_features]
        X_test = df.loc[test_idx, usable_features]

        try:
            model = make_model(cfg.model_type, y_train, random_state=random_state + fold_count, n_jobs=n_jobs)
            sample_weight = None
            if cfg.model_type.startswith("xgb"):
                # XGBoost's sklearn wrapper expects fold-local labels to be contiguous
                # 0..K-1. Map global class ids to local ids per fold, then store the
                # reverse mapping for probability alignment.
                orig_classes = np.array(sorted(np.unique(y_train).astype(int)))
                local_map = {int(c): i for i, c in enumerate(orig_classes)}
                y_fit = np.asarray([local_map[int(v)] for v in y_train], dtype=int)
                w = class_weight_dict(y_fit)
                sample_weight = np.asarray([w.get(int(v), 1.0) for v in y_fit], dtype=float)
                model.fit(X_train, y_fit, clf__sample_weight=sample_weight)
                model.named_steps["clf"].original_label_ids_ = orig_classes
            else:
                model.fit(X_train, y_train)
            p5 = predict_proba_5(model, X_test)
            prob_mat.loc[test_idx, :] = p5
            fold_count += 1
            trained_rows_total += len(train_idx2)
        except Exception:
            skipped_folds += 1

        start_i = end_i

    out = pd.DataFrame(index=df.index)
    out["date"] = df.index
    out["y_true"] = y.values
    out["label_name"] = aux["label_name"].values
    out["label_valid"] = label_valid.values
    out["eval_filter"] = eval_filter_mask.values
    out = pd.concat([out, prob_mat], axis=1)
    out["pred_class"] = prob_mat.values.argmax(axis=1)
    out["pred_label"] = out["pred_class"].map(ID_TO_LABEL)
    out.loc[prob_mat.isna().all(axis=1), ["pred_class", "pred_label"]] = np.nan

    meta = {
        "fold_count": fold_count,
        "skipped_folds": skipped_folds,
        "avg_train_rows": (trained_rows_total / fold_count) if fold_count else 0,
        "feature_count": len(usable_features),
    }
    return out, meta


def _safe_ovr_metrics(y_binary: np.ndarray, p: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float], float, int]:
    mask = np.isfinite(p) & np.isfinite(y_binary)
    y = y_binary[mask].astype(int)
    pp = p[mask].astype(float)
    if len(y) < 50:
        return None, None, None, float(np.mean(y)) if len(y) else np.nan, int(np.sum(y == 1))
    base = float(np.mean(y))
    pos_count = int(np.sum(y == 1))
    roc = None
    pr = None
    lift = None
    if len(np.unique(y)) >= 2:
        try:
            roc = float(roc_auc_score(y, pp))
        except Exception:
            roc = None
        try:
            pr = float(average_precision_score(y, pp))
            lift = pr - base
        except Exception:
            pr = None
            lift = None
    return roc, pr, lift, base, pos_count


def evaluate_multiclass(pred: pd.DataFrame, cfg: StrengthTrialConfig, meta: Dict[str, Any]) -> Dict[str, Any]:
    prob_cols = [f"prob_{name}" for name in LABELS]
    mask = pred["label_valid"].astype(bool) & pred["eval_filter"].astype(bool) & pred[prob_cols].notna().all(axis=1)
    y = pred.loc[mask, "y_true"].astype(int).values
    P = pred.loc[mask, prob_cols].astype(float).values
    y_pred = P.argmax(axis=1)

    row: Dict[str, Any] = {**asdict(cfg), **meta, "eval_rows": int(len(y))}
    if len(y) < 50 or len(np.unique(y)) < 2:
        row.update({
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "log_loss": None,
            "score": -999.0,
            "error": "insufficient_eval_rows_or_classes",
        })
        return row

    row["accuracy"] = float(accuracy_score(y, y_pred))
    row["balanced_accuracy"] = float(balanced_accuracy_score(y, y_pred))
    row["macro_f1"] = float(f1_score(y, y_pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))
    row["weighted_f1"] = float(f1_score(y, y_pred, labels=list(range(len(LABELS))), average="weighted", zero_division=0))
    try:
        row["log_loss"] = float(log_loss(y, np.clip(P, 1e-8, 1 - 1e-8), labels=list(range(len(LABELS)))))
    except Exception:
        row["log_loss"] = None

    # Class distribution and per-class one-vs-rest metrics.
    class_counts = {name: int(np.sum(y == idx)) for idx, name in ID_TO_LABEL.items()}
    row["class_counts_json"] = json.dumps(class_counts, ensure_ascii=False)

    macro_ovr_roc_values = []
    macro_ovr_pr_lift_values = []
    key_score = 0.0
    for idx, name in ID_TO_LABEL.items():
        y_bin = (y == idx).astype(int)
        roc, pr, lift, base, pos_count = _safe_ovr_metrics(y_bin, P[:, idx])
        row[f"{name}_base_rate"] = base
        row[f"{name}_count"] = pos_count
        row[f"{name}_roc_auc"] = roc
        row[f"{name}_pr_auc"] = pr
        row[f"{name}_pr_lift"] = lift
        if roc is not None:
            macro_ovr_roc_values.append(roc)
        if lift is not None:
            macro_ovr_pr_lift_values.append(lift)

    row["macro_ovr_roc_auc"] = float(np.mean(macro_ovr_roc_values)) if macro_ovr_roc_values else None
    row["macro_ovr_pr_lift"] = float(np.mean(macro_ovr_pr_lift_values)) if macro_ovr_pr_lift_values else None

    # Class precision/recall/F1.
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=list(range(len(LABELS))), zero_division=0
    )
    for idx, name in ID_TO_LABEL.items():
        row[f"{name}_precision"] = float(precision[idx])
        row[f"{name}_recall"] = float(recall[idx])
        row[f"{name}_f1"] = float(f1[idx])
        row[f"{name}_support"] = int(support[idx])

    # Score prioritizes actionable classes while still considering multiclass quality.
    up_s_roc = row.get("UP_STRENGTHENING_roc_auc") or 0.5
    dn_s_roc = row.get("DOWN_STRENGTHENING_roc_auc") or 0.5
    up_s_lift = row.get("UP_STRENGTHENING_pr_lift") or 0.0
    dn_s_lift = row.get("DOWN_STRENGTHENING_pr_lift") or 0.0
    side_f1 = row.get("SIDEWAYS_f1") or 0.0
    row["score"] = float(
        1.20 * (row["macro_f1"] or 0.0)
        + 0.80 * (row["balanced_accuracy"] or 0.0)
        + 0.90 * (up_s_roc - 0.5)
        + 1.10 * (dn_s_roc - 0.5)
        + 1.20 * up_s_lift
        + 1.40 * dn_s_lift
        - 0.05 * max(0.0, 0.70 - side_f1)  # avoid models that predict only action classes or only sideways
    )
    return row


def make_trial_grid(args: argparse.Namespace, feature_sets: Dict[str, List[str]]) -> List[StrengthTrialConfig]:
    horizons = [int(x) for x in parse_csv_arg(args.horizons)]
    models = parse_csv_arg(args.models)
    if HAS_XGB is False:
        models = [m for m in models if not m.startswith("xgb")]

    if args.profile == "quick":
        ret_eps_ks = [0.10, 0.20]
        strength_eps_values = [0.0, 0.5]
        strength_methods = ["score_delta", "hybrid_delta"]
        feature_names = [n for n in ["strength_core", "trend_volume", "compact_mixed"] if n in feature_sets]
        filters = [("all", "all"), ("non_extreme_vol", "all")]
        drop_sideways_options = [False]
        models = [m for m in models if m in ["xgb", "hgb", "extratrees", "logistic"]]
    elif args.profile == "balanced":
        ret_eps_ks = [0.10, 0.20, 0.30]
        strength_eps_values = [0.0, 0.5, 1.0]
        strength_methods = ["score_delta", "hybrid_delta", "slope_delta"]
        feature_names = [n for n in ["strength_core", "trend_core", "trend_volume", "down_core", "compact_mixed", "pruned_all"] if n in feature_sets]
        filters = [("all", "all"), ("non_extreme_vol", "all"), ("low_vol_only", "low_vol_only")]
        drop_sideways_options = [False, True]
    else:
        ret_eps_ks = [0.00, 0.10, 0.20, 0.30, 0.40]
        strength_eps_values = [0.0, 0.25, 0.5, 1.0]
        strength_methods = ["score_delta", "hybrid_delta", "slope_delta"]
        feature_names = list(feature_sets.keys())
        filters = [("all", "all"), ("non_extreme_vol", "all"), ("non_extreme_vol", "non_extreme_vol"), ("low_vol_only", "low_vol_only")]
        drop_sideways_options = [False, True]

    trials: List[StrengthTrialConfig] = []
    k = 0
    for h in horizons:
        for ret_eps_k in ret_eps_ks:
            for strength_eps in strength_eps_values:
                for strength_method in strength_methods:
                    for fs in feature_names:
                        if len(feature_sets.get(fs, [])) == 0:
                            continue
                        for model_type in models:
                            for train_filter, eval_filter in filters:
                                for drop_sideways in drop_sideways_options:
                                    # If SIDEWAYS is dropped from train, keep eval all to check if model can handle real-world full classes.
                                    if drop_sideways and eval_filter != "all":
                                        continue
                                    k += 1
                                    trials.append(
                                        StrengthTrialConfig(
                                            trial_id=f"ds{k:04d}",
                                            horizon=h,
                                            ret_eps_k=float(ret_eps_k),
                                            strength_eps=float(strength_eps),
                                            strength_method=strength_method,
                                            feature_set=fs,
                                            model_type=model_type,
                                            train_filter=train_filter,
                                            eval_filter=eval_filter,
                                            min_train_rows=args.min_train_rows,
                                            retrain_every_n_days=args.retrain_every_n_days,
                                            max_train_rows=args.max_train_rows,
                                            drop_sideways_train=bool(drop_sideways),
                                        )
                                    )
    if args.max_trials and len(trials) > args.max_trials:
        trials = trials[: args.max_trials]
    return trials


def save_best_diagnostics(pred: pd.DataFrame, out_dir: Path) -> None:
    prob_cols = [f"prob_{name}" for name in LABELS]
    mask = pred["label_valid"].astype(bool) & pred["eval_filter"].astype(bool) & pred[prob_cols].notna().all(axis=1)
    y = pred.loc[mask, "y_true"].astype(int).values
    y_pred = pred.loc[mask, "pred_class"].astype(int).values
    cm = confusion_matrix(y, y_pred, labels=list(range(len(LABELS))))
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in LABELS], columns=[f"pred_{x}" for x in LABELS])
    cm_df.to_csv(out_dir / "direction_strength_best_confusion_matrix.csv", encoding="utf-8-sig")

    report = classification_report(y, y_pred, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0, output_dict=True)
    with open(out_dir / "direction_strength_best_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-script", default="xgb_multi_branch_pruned_features_v8_6_5.py")
    parser.add_argument("--ohlcv-csv", default=None)
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--start-date", default="1999-03-10")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--backtest-start-date", default="2013-01-02")
    parser.add_argument("--horizons", default="20,40,60")
    parser.add_argument("--profile", choices=["quick", "balanced", "full"], default="quick")
    parser.add_argument("--models", default="xgb,hgb,extratrees,logistic")
    parser.add_argument("--min-train-rows", type=int, default=756)
    parser.add_argument("--retrain-every-n-days", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--result-dir", default="results_direction_strength_lab")
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
            log(
                f"trial {i}/{len(trials)}: {cfg.trial_id} h{cfg.horizon} "
                f"ret_eps_k={cfg.ret_eps_k} strength_eps={cfg.strength_eps} {cfg.strength_method} "
                f"fs={cfg.feature_set} model={cfg.model_type} train={cfg.train_filter} eval={cfg.eval_filter} "
                f"drop_sideways={cfg.drop_sideways_train}"
            )
        fs_cols = feature_sets[cfg.feature_set]
        try:
            pred, meta = walk_forward_predict_multiclass(df, fs_cols, cfg, args.random_state, args.n_jobs)
            row = evaluate_multiclass(pred, cfg, meta)
        except Exception as e:
            row = {**asdict(cfg), "error": str(e), "score": -999.0}
            pred = None
        rows.append(row)

        score = row.get("score")
        if score is not None and np.isfinite(score):
            if best_row is None or float(score) > float(best_row.get("score", -999)):
                best_row = row
                if pred is not None:
                    best_pred = pred.copy()
                    best_pred["trial_id"] = cfg.trial_id

    res = pd.DataFrame(rows)
    sort_cols = [c for c in [
        "score", "macro_f1", "balanced_accuracy", "DOWN_STRENGTHENING_roc_auc",
        "DOWN_STRENGTHENING_pr_lift", "UP_STRENGTHENING_roc_auc", "UP_STRENGTHENING_pr_lift"
    ] if c in res.columns]
    res = res.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    res.to_csv(out_dir / "direction_strength_trials.csv", index=False, encoding="utf-8-sig")
    res.head(20).to_csv(out_dir / "direction_strength_trials_top20.csv", index=False, encoding="utf-8-sig")

    if best_pred is not None:
        best_pred.to_csv(out_dir / "direction_strength_best_predictions.csv", index=False, encoding="utf-8-sig")
        save_best_diagnostics(best_pred, out_dir)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": args.ticker,
        "period": {"start": str(df.index.min()), "end": str(df.index.max()), "rows": int(len(df))},
        "profile": args.profile,
        "trial_count": int(len(trials)),
        "base_script": str(base_script_path),
        "base_feature_count": int(len(feature_cols)),
        "feature_set_counts": {k: len(v) for k, v in feature_sets.items()},
        "labels": LABELS,
        "best_trial": best_row,
        "top10": res.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "notes": [
            "trend_delta는 라벨 생성에만 사용됩니다. feature_cols에 추가하지 않습니다.",
            "UP_STRENGTHENING과 DOWN_STRENGTHENING의 one-vs-rest ROC-AUC / PR lift를 핵심으로 보세요.",
            "SIDEWAYS support가 지나치게 크면 ret_eps_k를 낮추고, 너무 작으면 ret_eps_k를 높이세요.",
            "drop_sideways_train=True는 action class 분리에 유리할 수 있지만 전체 실전 예측에서는 SIDEWAYS 판단력이 약해질 수 있습니다.",
        ],
    }
    with open(out_dir / "direction_strength_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log("완료")
    log(f"저장: {out_dir / 'direction_strength_trials.csv'}")
    if best_row:
        log("best_trial:")
        print(json.dumps(best_row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
