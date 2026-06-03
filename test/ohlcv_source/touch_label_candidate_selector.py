# -*- coding: utf-8 -*-
"""
touch_label_candidate_selector.py

2단계: Up-touch / Down-touch 라벨 후보군 확정 전용 스크립트.

목적
----
1. 모델 학습 없이 라벨 후보만 비교
2. fixed-k 라벨과 target positive rate 라벨을 비교
3. 수익추구형 / 안정형 / 방어형 모델에 사용할 라벨 후보를 추천
4. 자산별·연도별 positive rate 안정성 확인
5. Up/Down joint distribution 확인
6. 3단계 모델 성능 최적화에 넘길 best label config 생성

권장 기본 비교
--------------
A. fixed_h10_k100
B. fixed_h20_k100
C. fixed_h40_k100
D. target_h10_rate30
E. target_h20_rate30
F. target_h40_rate30

주의
----
이 스크립트는 라벨 분포/안정성 기반 후보 선정 도구입니다.
아직 모델 성능을 평가하지 않습니다.
target-rate k는 전체 데이터 진단용으로 계산합니다.
3단계 실제 walk-forward 학습에서는 train/calibration window 안에서만 k를 재산출해야 합니다.

실행 예시 - 단일 자산
--------------------
python touch_label_candidate_selector.py ^
  --inputs "QQQ_ohlcv.csv" ^
  --asset-names "QQQ" ^
  --output-dir "touch_label_candidates_QQQ"

실행 예시 - 다중 자산
--------------------
python touch_label_candidate_selector.py ^
  --inputs "QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv" ^
  --asset-names "QQQ,SPY,SOXX,XLK" ^
  --output-dir "touch_label_candidates_all"

출력 파일
---------
output_dir/
├─ label_candidate_selection_summary.json
├─ candidate_pair_summary.csv
├─ side_label_summary.csv
├─ annual_rate_stability.csv
├─ joint_touch_distribution.csv
├─ target_k_search.csv
├─ objective_recommendations.csv
└─ best_label_config.json

의존성
------
python>=3.10
pandas
numpy
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 0. Utils
# ============================================================

def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


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


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def safe_divide(a: float, b: float, default: float = np.nan) -> float:
    try:
        if b == 0 or pd.isna(b):
            return default
        return float(a / b)
    except Exception:
        return default


def coefficient_of_variation(s: pd.Series) -> float:
    x = pd.Series(s).dropna().astype(float)
    if len(x) < 2 or x.mean() == 0:
        return np.nan
    return float(x.std(ddof=1) / x.mean())


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    rename_map = {
        "datetime": "date",
        "timestamp": "date",
        "adjclose": "adj_close",
        "adj_close": "adj_close",
        "adjusted_close": "adj_close",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}. columns={list(out.columns)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]

    return out


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    return normalize_ohlcv_columns(pd.read_csv(path))


# ============================================================
# 1. Label Generation
# ============================================================

def explicit_future_high_low(
    high: pd.Series,
    low: pd.Series,
    horizon: int,
) -> Tuple[pd.Series, pd.Series]:
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    n = len(h)

    future_high = np.full(n, np.nan)
    future_low = np.full(n, np.nan)

    for i in range(0, n - horizon):
        future_high[i] = np.nanmax(h[i + 1 : i + 1 + horizon])
        future_low[i] = np.nanmin(l[i + 1 : i + 1 + horizon])

    return pd.Series(future_high, index=high.index), pd.Series(future_low, index=low.index)


def current_horizon_volatility(close: pd.Series, horizon: int, vol_window: int) -> pd.Series:
    returns = close.astype(float).pct_change()
    min_periods = max(20, min(vol_window, vol_window // 2))
    return returns.rolling(vol_window, min_periods=min_periods).std().shift(1) * math.sqrt(horizon)


def make_touch_labels(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int,
    k_up: float,
    k_down: float,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    out["current_horizon_vol"] = current_horizon_volatility(close, horizon=horizon, vol_window=vol_window)
    out["upper_barrier"] = close * (1.0 + k_up * out["current_horizon_vol"])
    out["lower_barrier"] = close * (1.0 - k_down * out["current_horizon_vol"])

    future_high, future_low = explicit_future_high_low(high, low, horizon=horizon)
    out["future_high_h"] = future_high
    out["future_low_h"] = future_low

    out["y_up_touch"] = (out["future_high_h"] >= out["upper_barrier"]).astype(float)
    out["y_down_touch"] = (out["future_low_h"] <= out["lower_barrier"]).astype(float)

    invalid = (
        out["current_horizon_vol"].isna()
        | out["future_high_h"].isna()
        | out["future_low_h"].isna()
        | out["upper_barrier"].isna()
        | out["lower_barrier"].isna()
    )
    out.loc[invalid, ["y_up_touch", "y_down_touch"]] = np.nan

    cond_up_only = (out["y_up_touch"] == 1) & (out["y_down_touch"] == 0)
    cond_down_only = (out["y_up_touch"] == 0) & (out["y_down_touch"] == 1)
    cond_both = (out["y_up_touch"] == 1) & (out["y_down_touch"] == 1)
    cond_none = (out["y_up_touch"] == 0) & (out["y_down_touch"] == 0)

    out["touch_regime"] = "invalid"
    out.loc[cond_up_only, "touch_regime"] = "up_only"
    out.loc[cond_down_only, "touch_regime"] = "down_only"
    out.loc[cond_both, "touch_regime"] = "both_touch"
    out.loc[cond_none, "touch_regime"] = "no_touch"

    return out


def touch_rate_for_k(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int,
    side: str,
    k: float,
) -> float:
    if side == "up":
        labels = make_touch_labels(df, horizon=horizon, vol_window=vol_window, k_up=k, k_down=999.0)
        return float(labels["y_up_touch"].dropna().mean())
    if side == "down":
        labels = make_touch_labels(df, horizon=horizon, vol_window=vol_window, k_up=999.0, k_down=k)
        return float(labels["y_down_touch"].dropna().mean())
    raise ValueError(f"unknown side: {side}")


def find_k_for_target_rate_grid(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int,
    side: str,
    target_rate: float,
    k_min: float,
    k_max: float,
    grid_size: int,
) -> Dict:
    ks = np.linspace(k_min, k_max, grid_size)
    rows = []

    for k in ks:
        rate = touch_rate_for_k(df, horizon=horizon, vol_window=vol_window, side=side, k=float(k))
        rows.append({
            "side": side,
            "horizon": horizon,
            "target_rate": target_rate,
            "k": float(k),
            "positive_rate": float(rate),
            "abs_error": abs(float(rate) - target_rate),
        })

    best = pd.DataFrame(rows).sort_values(["abs_error", "k"]).iloc[0].to_dict()
    return {
        "side": side,
        "horizon": horizon,
        "target_rate": target_rate,
        "best_k": float(best["k"]),
        "achieved_positive_rate": float(best["positive_rate"]),
        "abs_error": float(best["abs_error"]),
        "k_min": k_min,
        "k_max": k_max,
        "grid_size": grid_size,
    }


# ============================================================
# 2. Candidate Diagnostics
# ============================================================

def candidate_pair_id(method: str, horizon: int, k_up: float, k_down: float, target_rate: float | None) -> str:
    if method == "fixed":
        return f"fixed_h{horizon}_kup{k_up:.3f}_kdn{k_down:.3f}".replace(".", "p")
    if method == "target":
        return f"target_h{horizon}_rate{target_rate:.2f}".replace(".", "p")
    raise ValueError(method)


def side_label_id(method: str, side: str, horizon: int, k: float, target_rate: float | None) -> str:
    if method == "fixed":
        return f"{side}_touch_fixed_h{horizon}_k{k:.3f}".replace(".", "p")
    if method == "target":
        return f"{side}_touch_target_h{horizon}_rate{target_rate:.2f}_k{k:.3f}".replace(".", "p")
    raise ValueError(method)


def annual_positive_rates(labels: pd.DataFrame, asset_name: str, candidate_id: str, horizon: int, method: str, k_up: float, k_down: float, target_rate: float | None) -> pd.DataFrame:
    valid = labels.dropna(subset=["y_up_touch", "y_down_touch"]).copy()
    if valid.empty:
        return pd.DataFrame()

    valid["year"] = pd.to_datetime(valid["date"]).dt.year
    rows = []
    for year, g in valid.groupby("year"):
        rows.append({
            "asset_name": asset_name,
            "candidate_id": candidate_id,
            "method": method,
            "horizon": horizon,
            "target_rate": target_rate,
            "k_up": k_up,
            "k_down": k_down,
            "year": int(year),
            "rows": int(len(g)),
            "up_touch_rate": float(g["y_up_touch"].mean()),
            "down_touch_rate": float(g["y_down_touch"].mean()),
            "both_touch_rate": float(((g["y_up_touch"] == 1) & (g["y_down_touch"] == 1)).mean()),
            "no_touch_rate": float(((g["y_up_touch"] == 0) & (g["y_down_touch"] == 0)).mean()),
            "up_only_rate": float(((g["y_up_touch"] == 1) & (g["y_down_touch"] == 0)).mean()),
            "down_only_rate": float(((g["y_up_touch"] == 0) & (g["y_down_touch"] == 1)).mean()),
        })
    return pd.DataFrame(rows)


def joint_touch_distribution(labels: pd.DataFrame, asset_name: str, candidate_id: str, horizon: int, method: str, k_up: float, k_down: float, target_rate: float | None) -> pd.DataFrame:
    valid = labels.dropna(subset=["y_up_touch", "y_down_touch"]).copy()
    if valid.empty:
        return pd.DataFrame()

    g = (
        valid.groupby(["y_up_touch", "y_down_touch", "touch_regime"])
        .size()
        .reset_index(name="count")
    )
    g["asset_name"] = asset_name
    g["candidate_id"] = candidate_id
    g["method"] = method
    g["horizon"] = horizon
    g["target_rate"] = target_rate
    g["k_up"] = k_up
    g["k_down"] = k_down
    g["rate"] = g["count"] / g["count"].sum()
    return g[[
        "asset_name", "candidate_id", "method", "horizon", "target_rate", "k_up", "k_down",
        "y_up_touch", "y_down_touch", "touch_regime", "count", "rate",
    ]]


def summarize_pair(labels: pd.DataFrame, asset_name: str, candidate_id: str, method: str, horizon: int, k_up: float, k_down: float, target_rate: float | None) -> Dict:
    valid = labels.dropna(subset=["y_up_touch", "y_down_touch"]).copy()

    if valid.empty:
        return {
            "asset_name": asset_name,
            "candidate_id": candidate_id,
            "method": method,
            "horizon": horizon,
            "target_rate": target_rate,
            "k_up": k_up,
            "k_down": k_down,
            "valid_rows": 0,
        }

    up = valid["y_up_touch"]
    down = valid["y_down_touch"]

    up_rate = float(up.mean())
    down_rate = float(down.mean())
    both_rate = float(((up == 1) & (down == 1)).mean())
    no_rate = float(((up == 0) & (down == 0)).mean())
    up_only_rate = float(((up == 1) & (down == 0)).mean())
    down_only_rate = float(((up == 0) & (down == 1)).mean())

    # Label-quality heuristic scores.
    # These do not measure model performance. They select labels with usable class balance and stability.
    desired_center = target_rate if method == "target" and target_rate is not None else 0.30
    up_balance_penalty = abs(up_rate - desired_center)
    down_balance_penalty = abs(down_rate - desired_center)
    pair_balance_penalty = abs(up_rate - down_rate)

    # Too many both-touch/no-touch can weaken regime separability.
    both_penalty = max(0.0, both_rate - 0.20)
    no_touch_penalty = max(0.0, no_rate - 0.55)

    pair_quality_score = (
        1.0
        - 0.80 * up_balance_penalty
        - 0.80 * down_balance_penalty
        - 0.60 * pair_balance_penalty
        - 0.40 * both_penalty
        - 0.20 * no_touch_penalty
    )

    return {
        "asset_name": asset_name,
        "candidate_id": candidate_id,
        "method": method,
        "horizon": horizon,
        "target_rate": target_rate,
        "k_up": k_up,
        "k_down": k_down,
        "valid_rows": int(len(valid)),
        "up_touch_rate": up_rate,
        "down_touch_rate": down_rate,
        "rate_gap_abs": abs(up_rate - down_rate),
        "both_touch_rate": both_rate,
        "no_touch_rate": no_rate,
        "up_only_rate": up_only_rate,
        "down_only_rate": down_only_rate,
        "avg_current_horizon_vol": float(valid["current_horizon_vol"].mean()),
        "median_current_horizon_vol": float(valid["current_horizon_vol"].median()),
        "pair_quality_score": float(pair_quality_score),
    }


def summarize_side_labels(pair_summary: Dict, annual_df: pd.DataFrame) -> List[Dict]:
    rows = []

    asset_name = pair_summary["asset_name"]
    candidate_id = pair_summary["candidate_id"]
    method = pair_summary["method"]
    horizon = pair_summary["horizon"]
    target_rate = pair_summary["target_rate"]
    k_up = pair_summary["k_up"]
    k_down = pair_summary["k_down"]

    # annual stability
    if not annual_df.empty:
        g = annual_df[
            (annual_df["asset_name"] == asset_name)
            & (annual_df["candidate_id"] == candidate_id)
        ].copy()
    else:
        g = pd.DataFrame()

    up_cv = coefficient_of_variation(g["up_touch_rate"]) if not g.empty else np.nan
    down_cv = coefficient_of_variation(g["down_touch_rate"]) if not g.empty else np.nan

    # Side quality: class balance + annual stability.
    for side, rate_col, k, rate, cv in [
        ("up", "up_touch_rate", k_up, pair_summary.get("up_touch_rate", np.nan), up_cv),
        ("down", "down_touch_rate", k_down, pair_summary.get("down_touch_rate", np.nan), down_cv),
    ]:
        label_id = side_label_id(method, side, horizon, k, target_rate)

        # desired: rate around 0.25~0.35 and lower annual CV.
        desired_center = target_rate if method == "target" and target_rate is not None else 0.30
        balance_penalty = abs(rate - desired_center) if not pd.isna(rate) else 1.0
        cv_penalty = min(cv, 1.0) if not pd.isna(cv) else 1.0

        side_quality_score = 1.0 - 1.0 * balance_penalty - 0.5 * cv_penalty

        rows.append({
            "asset_name": asset_name,
            "candidate_id": candidate_id,
            "label_id": label_id,
            "side": side,
            "method": method,
            "horizon": horizon,
            "target_rate": target_rate,
            "k": k,
            "positive_rate": rate,
            "annual_cv": cv,
            "annual_cv_gt_0p30": bool(cv > 0.30) if not pd.isna(cv) else None,
            "side_quality_score": float(side_quality_score),
        })

    return rows


def add_annual_stability_to_pair_summary(pair_df: pd.DataFrame, annual_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty or annual_df.empty:
        return pair_df

    rows = []
    for _, row in pair_df.iterrows():
        g = annual_df[
            (annual_df["asset_name"] == row["asset_name"])
            & (annual_df["candidate_id"] == row["candidate_id"])
        ].copy()

        out = row.to_dict()
        out["up_touch_annual_cv"] = coefficient_of_variation(g["up_touch_rate"])
        out["down_touch_annual_cv"] = coefficient_of_variation(g["down_touch_rate"])
        out["both_touch_annual_cv"] = coefficient_of_variation(g["both_touch_rate"])
        out["no_touch_annual_cv"] = coefficient_of_variation(g["no_touch_rate"])
        out["up_rate_cv_gt_0p30"] = bool(out["up_touch_annual_cv"] > 0.30) if not pd.isna(out["up_touch_annual_cv"]) else None
        out["down_rate_cv_gt_0p30"] = bool(out["down_touch_annual_cv"] > 0.30) if not pd.isna(out["down_touch_annual_cv"]) else None

        # Penalize annual instability after initial score.
        instability_penalty = 0.20 * min(out["up_touch_annual_cv"], 1.0) + 0.20 * min(out["down_touch_annual_cv"], 1.0)
        out["pair_quality_score_with_stability"] = float(out["pair_quality_score"] - instability_penalty)

        rows.append(out)

    return pd.DataFrame(rows)


# ============================================================
# 3. Objective Recommendations
# ============================================================

def aggregate_across_assets(pair_df: pd.DataFrame, side_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_keys = ["candidate_id", "method", "horizon", "target_rate"]
    pair_agg = (
        pair_df
        .groupby(pair_keys, dropna=False)
        .agg(
            asset_count=("asset_name", "nunique"),
            mean_pair_score=("pair_quality_score_with_stability", "mean"),
            median_pair_score=("pair_quality_score_with_stability", "median"),
            mean_up_rate=("up_touch_rate", "mean"),
            mean_down_rate=("down_touch_rate", "mean"),
            mean_rate_gap=("rate_gap_abs", "mean"),
            mean_both_rate=("both_touch_rate", "mean"),
            mean_no_touch_rate=("no_touch_rate", "mean"),
            mean_up_annual_cv=("up_touch_annual_cv", "mean"),
            mean_down_annual_cv=("down_touch_annual_cv", "mean"),
            unstable_up_asset_rate=("up_rate_cv_gt_0p30", "mean"),
            unstable_down_asset_rate=("down_rate_cv_gt_0p30", "mean"),
            mean_k_up=("k_up", "mean"),
            mean_k_down=("k_down", "mean"),
        )
        .reset_index()
    )

    side_keys = ["label_id", "side", "method", "horizon", "target_rate"]
    side_agg = (
        side_df
        .groupby(side_keys, dropna=False)
        .agg(
            asset_count=("asset_name", "nunique"),
            mean_side_score=("side_quality_score", "mean"),
            median_side_score=("side_quality_score", "median"),
            mean_positive_rate=("positive_rate", "mean"),
            median_positive_rate=("positive_rate", "median"),
            mean_annual_cv=("annual_cv", "mean"),
            median_annual_cv=("annual_cv", "median"),
            unstable_asset_rate=("annual_cv_gt_0p30", "mean"),
            mean_k=("k", "mean"),
        )
        .reset_index()
    )

    return pair_agg, side_agg


def select_recommendations(pair_agg: pd.DataFrame, side_agg: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Return-seeking: choose Up label with good class balance/stability.
    up = side_agg[side_agg["side"] == "up"].copy()
    if not up.empty:
        up["objective_score"] = (
            up["mean_side_score"]
            - 0.10 * up["unstable_asset_rate"].fillna(1.0)
            + 0.05 * (up["horizon"] == 20).astype(float)
        )
        best = up.sort_values("objective_score", ascending=False).head(1).iloc[0].to_dict()
        rows.append({
            "objective": "return_seeking",
            "recommended_label_or_pair": best["label_id"],
            "candidate_id": None,
            "method": best["method"],
            "horizon": best["horizon"],
            "target_rate": best["target_rate"],
            "score": best["objective_score"],
            "reason": "Up-touch side label with best balance/stability; H20 receives a small preference.",
        })

    # Defensive: choose Down label with good class balance/stability.
    down = side_agg[side_agg["side"] == "down"].copy()
    if not down.empty:
        down["objective_score"] = (
            down["mean_side_score"]
            - 0.10 * down["unstable_asset_rate"].fillna(1.0)
            + 0.05 * (down["horizon"].isin([20, 40])).astype(float)
        )
        best = down.sort_values("objective_score", ascending=False).head(1).iloc[0].to_dict()
        rows.append({
            "objective": "defensive",
            "recommended_label_or_pair": best["label_id"],
            "candidate_id": None,
            "method": best["method"],
            "horizon": best["horizon"],
            "target_rate": best["target_rate"],
            "score": best["objective_score"],
            "reason": "Down-touch side label with best balance/stability; H20/H40 receive a small preference.",
        })

    # Balanced: choose pair with low up/down rate gap and low annual instability.
    pair = pair_agg.copy()
    if not pair.empty:
        pair["objective_score"] = (
            pair["mean_pair_score"]
            - 0.25 * pair["mean_rate_gap"]
            - 0.10 * pair["unstable_up_asset_rate"].fillna(1.0)
            - 0.10 * pair["unstable_down_asset_rate"].fillna(1.0)
            + 0.05 * (pair["horizon"] == 20).astype(float)
        )
        best = pair.sort_values("objective_score", ascending=False).head(1).iloc[0].to_dict()
        rows.append({
            "objective": "balanced",
            "recommended_label_or_pair": best["candidate_id"],
            "candidate_id": best["candidate_id"],
            "method": best["method"],
            "horizon": best["horizon"],
            "target_rate": best["target_rate"],
            "score": best["objective_score"],
            "reason": "Up/Down pair with good rate balance and annual stability; H20 receives a small preference.",
        })

    return pd.DataFrame(rows)


# ============================================================
# 4. Main Runner
# ============================================================

def build_candidates_for_asset(
    df: pd.DataFrame,
    asset_name: str,
    horizons: List[int],
    fixed_k_values: List[float],
    target_rates: List[float],
    vol_window: int,
    k_min: float,
    k_max: float,
    k_grid_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_k_rows = []
    pair_rows = []
    annual_parts = []
    joint_parts = []
    side_rows = []

    # Fixed candidates
    for horizon in horizons:
        for k in fixed_k_values:
            candidate_id = candidate_pair_id("fixed", horizon, k, k, None)
            labels = make_touch_labels(df, horizon=horizon, vol_window=vol_window, k_up=k, k_down=k)

            pair = summarize_pair(labels, asset_name, candidate_id, "fixed", horizon, k, k, None)
            pair_rows.append(pair)

            annual = annual_positive_rates(labels, asset_name, candidate_id, horizon, "fixed", k, k, None)
            annual_parts.append(annual)

            joint = joint_touch_distribution(labels, asset_name, candidate_id, horizon, "fixed", k, k, None)
            joint_parts.append(joint)

            side_rows.extend(summarize_side_labels(pair, annual))

    # Target-rate candidates
    for horizon in horizons:
        for target_rate in target_rates:
            up_k_info = find_k_for_target_rate_grid(
                df, horizon=horizon, vol_window=vol_window, side="up",
                target_rate=target_rate, k_min=k_min, k_max=k_max, grid_size=k_grid_size,
            )
            down_k_info = find_k_for_target_rate_grid(
                df, horizon=horizon, vol_window=vol_window, side="down",
                target_rate=target_rate, k_min=k_min, k_max=k_max, grid_size=k_grid_size,
            )

            up_k_info["asset_name"] = asset_name
            down_k_info["asset_name"] = asset_name
            target_k_rows.extend([up_k_info, down_k_info])

            k_up = up_k_info["best_k"]
            k_down = down_k_info["best_k"]

            candidate_id = candidate_pair_id("target", horizon, k_up, k_down, target_rate)
            labels = make_touch_labels(df, horizon=horizon, vol_window=vol_window, k_up=k_up, k_down=k_down)

            pair = summarize_pair(labels, asset_name, candidate_id, "target", horizon, k_up, k_down, target_rate)
            pair_rows.append(pair)

            annual = annual_positive_rates(labels, asset_name, candidate_id, horizon, "target", k_up, k_down, target_rate)
            annual_parts.append(annual)

            joint = joint_touch_distribution(labels, asset_name, candidate_id, horizon, "target", k_up, k_down, target_rate)
            joint_parts.append(joint)

            side_rows.extend(summarize_side_labels(pair, annual))

    pair_df = pd.DataFrame(pair_rows)
    annual_df = pd.concat([x for x in annual_parts if not x.empty], ignore_index=True) if annual_parts else pd.DataFrame()
    pair_df = add_annual_stability_to_pair_summary(pair_df, annual_df)

    side_df = pd.DataFrame(side_rows)
    joint_df = pd.concat([x for x in joint_parts if not x.empty], ignore_index=True) if joint_parts else pd.DataFrame()
    target_k_df = pd.DataFrame(target_k_rows)

    return pair_df, side_df, annual_df, joint_df, target_k_df


def run(args) -> Dict[str, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = parse_list(args.inputs)
    asset_names = parse_list(args.asset_names)

    if len(asset_names) == 1 and len(inputs) > 1:
        # Allow asset_names omitted incorrectly by deriving from filenames.
        asset_names = [Path(x).stem.replace("_ohlcv", "").upper() for x in inputs]

    if len(inputs) != len(asset_names):
        raise ValueError(f"inputs count != asset_names count: {len(inputs)} vs {len(asset_names)}")

    horizons = parse_int_list(args.horizons)
    fixed_k_values = parse_float_list(args.fixed_k_values)
    target_rates = parse_float_list(args.target_rates)

    all_pair = []
    all_side = []
    all_annual = []
    all_joint = []
    all_target_k = []
    asset_periods = []

    for input_path, asset_name in zip(inputs, asset_names):
        df = load_ohlcv(input_path)
        if args.start_date:
            df = df[df["date"] >= pd.to_datetime(args.start_date)].copy()
        if args.end_date:
            df = df[df["date"] <= pd.to_datetime(args.end_date)].copy()
        df = df.reset_index(drop=True)

        asset_periods.append({
            "asset_name": asset_name,
            "input": str(input_path),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "rows": int(len(df)),
        })

        pair_df, side_df, annual_df, joint_df, target_k_df = build_candidates_for_asset(
            df=df,
            asset_name=asset_name,
            horizons=horizons,
            fixed_k_values=fixed_k_values,
            target_rates=target_rates,
            vol_window=args.vol_window,
            k_min=args.k_min,
            k_max=args.k_max,
            k_grid_size=args.k_grid_size,
        )

        all_pair.append(pair_df)
        all_side.append(side_df)
        all_annual.append(annual_df)
        all_joint.append(joint_df)
        all_target_k.append(target_k_df)

    pair_df = pd.concat(all_pair, ignore_index=True) if all_pair else pd.DataFrame()
    side_df = pd.concat(all_side, ignore_index=True) if all_side else pd.DataFrame()
    annual_df = pd.concat(all_annual, ignore_index=True) if all_annual else pd.DataFrame()
    joint_df = pd.concat(all_joint, ignore_index=True) if all_joint else pd.DataFrame()
    target_k_df = pd.concat(all_target_k, ignore_index=True) if all_target_k else pd.DataFrame()

    pair_agg, side_agg = aggregate_across_assets(pair_df, side_df)
    recommendations = select_recommendations(pair_agg, side_agg)

    # Sort useful outputs
    if not pair_df.empty:
        pair_df = pair_df.sort_values(["asset_name", "pair_quality_score_with_stability"], ascending=[True, False])
    if not side_df.empty:
        side_df = side_df.sort_values(["asset_name", "side", "side_quality_score"], ascending=[True, True, False])
    if not pair_agg.empty:
        pair_agg = pair_agg.sort_values("mean_pair_score", ascending=False)
    if not side_agg.empty:
        side_agg = side_agg.sort_values(["side", "mean_side_score"], ascending=[True, False])

    # Build best config
    best_config = {
        "experiment": "touch_label_candidate_selector",
        "objective": "label_candidate_selection_without_model_training",
        "asset_periods": asset_periods,
        "config": {
            "horizons": horizons,
            "fixed_k_values": fixed_k_values,
            "target_rates": target_rates,
            "vol_window": args.vol_window,
            "k_search": {
                "k_min": args.k_min,
                "k_max": args.k_max,
                "k_grid_size": args.k_grid_size,
            },
            "note": (
                "Target-rate k values are diagnostic over supplied data. "
                "In model training, recompute k only inside train/calibration windows to avoid leakage."
            ),
        },
        "recommendations": recommendations.to_dict("records") if not recommendations.empty else [],
    }

    # Detailed mapping for each recommendation
    rec_details = {}
    for _, rec in recommendations.iterrows():
        obj = rec["objective"]
        if obj in {"return_seeking", "defensive"}:
            label_id = rec["recommended_label_or_pair"]
            detail = side_df[side_df["label_id"] == label_id].to_dict("records")
        else:
            cid = rec["candidate_id"]
            detail = pair_df[pair_df["candidate_id"] == cid].to_dict("records")
        rec_details[obj] = detail
    best_config["recommendation_details_by_asset"] = rec_details

    summary = {
        "experiment": "touch_label_candidate_selector",
        "asset_count": len(inputs),
        "asset_periods": asset_periods,
        "candidate_count_per_asset": int(len(pair_df) / max(len(inputs), 1)) if not pair_df.empty else 0,
        "side_label_count_per_asset": int(len(side_df) / max(len(inputs), 1)) if not side_df.empty else 0,
        "top_pair_candidates": pair_agg.head(10).to_dict("records") if not pair_agg.empty else [],
        "top_side_candidates": side_agg.head(20).to_dict("records") if not side_agg.empty else [],
        "recommendations": best_config["recommendations"],
        "decision_note": (
            "Proceed to step 3 only after reviewing whether selected label candidates have acceptable positive rate, annual CV, and joint distribution. "
            "This step does not prove predictive performance."
        ),
    }

    outputs = {
        "summary": save_json(out_dir / "label_candidate_selection_summary.json", summary),
        "best_label_config": save_json(out_dir / "best_label_config.json", best_config),
        "candidate_pair_summary": save_csv(out_dir / "candidate_pair_summary.csv", pair_df),
        "side_label_summary": save_csv(out_dir / "side_label_summary.csv", side_df),
        "candidate_pair_aggregate": save_csv(out_dir / "candidate_pair_aggregate.csv", pair_agg),
        "side_label_aggregate": save_csv(out_dir / "side_label_aggregate.csv", side_agg),
        "objective_recommendations": save_csv(out_dir / "objective_recommendations.csv", recommendations),
        "annual_rate_stability": save_csv(out_dir / "annual_rate_stability.csv", annual_df),
        "joint_touch_distribution": save_csv(out_dir / "joint_touch_distribution.csv", joint_df),
        "target_k_search": save_csv(out_dir / "target_k_search.csv", target_k_df),
    }

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputs", required=True, help="Comma-separated OHLCV csv paths")
    parser.add_argument("--asset-names", required=True, help="Comma-separated asset names")
    parser.add_argument("--output-dir", default="touch_label_candidates_output")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")

    parser.add_argument("--horizons", default="10,20,40")
    parser.add_argument("--fixed-k-values", default="1.0")
    parser.add_argument("--target-rates", default="0.30")
    parser.add_argument("--vol-window", type=int, default=60)

    parser.add_argument("--k-min", type=float, default=0.25)
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--k-grid-size", type=int, default=100)

    args = parser.parse_args()
    outputs = run(args)

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    print("[OK] Touch label candidate selection completed.")
    print(json.dumps({
        "asset_count": summary["asset_count"],
        "candidate_count_per_asset": summary["candidate_count_per_asset"],
        "side_label_count_per_asset": summary["side_label_count_per_asset"],
        "recommendations": summary["recommendations"],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
