# -*- coding: utf-8 -*-
"""
touch_label_alignment_validator.py

1단계: Up-touch / Down-touch 라벨 생성 정확성 검증 전용 스크립트.

목적
----
1. future_high_h / future_low_h가 정확히 t+1 ~ t+H 구간을 보는지 검증
2. current_horizon_vol이 t 시점 이전 데이터만 사용하는지 검증
3. Up-touch / Down-touch 라벨 생성 결과 검증
4. 연도별 positive rate, joint touch distribution 확인
5. fixed-k 라벨과 target positive rate 기준 k 탐색 결과를 비교

이 스크립트는 모델 학습과 allocation을 수행하지 않습니다.
오직 라벨 생성의 정렬, 누수 가능성, 분포 안정성만 검증합니다.

실행 예시 - CMD
---------------
python touch_label_alignment_validator.py ^
  --input "QQQ_ohlcv.csv" ^
  --asset-name QQQ ^
  --output-dir "touch_label_validation_QQQ"

여러 horizon/k 조합:
python touch_label_alignment_validator.py ^
  --input "QQQ_ohlcv.csv" ^
  --asset-name QQQ ^
  --horizons "10,20,40" ^
  --k-values "0.5,0.75,1.0" ^
  --output-dir "touch_label_validation_QQQ"

출력 파일
---------
output_dir/
├─ touch_label_validation_summary.json
├─ alignment_checks.csv
├─ label_distribution.csv
├─ annual_positive_rate.csv
├─ joint_touch_distribution.csv
├─ target_rate_k_search.csv
└─ generated_labels_preview.csv

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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 0. IO / Utility
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


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def rolling_min_periods(window: int, min_floor: int = 20, frac: float = 0.5) -> int:
    return max(1, min(int(window), max(int(min_floor), int(window * frac))))


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
# 1. Future Window / Label Logic
# ============================================================

def explicit_future_high_low(
    high: pd.Series,
    low: pd.Series,
    horizon: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    명시적 방식:
    시점 i에서 i+1 ~ i+H 구간의 high max / low min 계산.

    마지막 horizon개 행은 미래 데이터가 부족하므로 NaN.
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    n = len(h)

    future_high = np.full(n, np.nan)
    future_low = np.full(n, np.nan)

    for i in range(0, n - horizon):
        future_high[i] = np.nanmax(h[i + 1 : i + 1 + horizon])
        future_low[i] = np.nanmin(l[i + 1 : i + 1 + horizon])

    return pd.Series(future_high, index=high.index), pd.Series(future_low, index=low.index)


def rolling_future_high_low(
    high: pd.Series,
    low: pd.Series,
    horizon: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    기존 rolling 패턴 검증용.
    high.shift(-1).rolling(H).max().shift(-(H-1))
    """
    future_high = (
        high.shift(-1)
        .rolling(horizon, min_periods=horizon)
        .max()
        .shift(-(horizon - 1))
    )
    future_low = (
        low.shift(-1)
        .rolling(horizon, min_periods=horizon)
        .min()
        .shift(-(horizon - 1))
    )
    return future_high, future_low


def current_horizon_volatility(
    close: pd.Series,
    horizon: int,
    vol_window: int = 60,
    min_periods: int | None = None,
) -> pd.Series:
    """
    current_horizon_vol_t = std(returns up to t-1) * sqrt(horizon)

    close.pct_change()의 returns[i]는 close[i] / close[i-1] - 1.
    shift(1)을 적용하면 시점 i의 vol은 returns up to i-1까지만 사용.
    """
    returns = close.astype(float).pct_change()
    if min_periods is None:
        min_periods = rolling_min_periods(vol_window, min_floor=20, frac=0.5)

    return returns.rolling(vol_window, min_periods=min_periods).std().shift(1) * math.sqrt(horizon)


def make_touch_labels_explicit(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int,
    k_up: float,
    k_down: float,
) -> pd.DataFrame:
    """
    명시적 future window 방식으로 Up/Down touch 라벨 생성.
    """
    out = df.copy()
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

    out["touch_regime"] = np.select(
        [
            (out["y_up_touch"] == 1) & (out["y_down_touch"] == 0),
            (out["y_up_touch"] == 0) & (out["y_down_touch"] == 1),
            (out["y_up_touch"] == 1) & (out["y_down_touch"] == 1),
            (out["y_up_touch"] == 0) & (out["y_down_touch"] == 0),
        ],
        ["up_only", "down_only", "both_touch", "no_touch"],
        default="invalid",
    )

    return out


# ============================================================
# 2. Alignment / Leakage Validation
# ============================================================

def validate_future_window_alignment(
    df: pd.DataFrame,
    horizon: int,
    sample_count: int = 30,
    random_state: int = 42,
    atol: float = 1e-10,
) -> Tuple[bool, pd.DataFrame]:
    """
    future_high_h / future_low_h가 실제로 t+1 ~ t+H 구간인지 수동 검증.
    explicit 방식과 rolling 방식도 비교.
    """
    high = df["high"].astype(float).reset_index(drop=True)
    low = df["low"].astype(float).reset_index(drop=True)
    dates = df["date"].reset_index(drop=True)

    explicit_high, explicit_low = explicit_future_high_low(high, low, horizon)
    rolling_high, rolling_low = rolling_future_high_low(high, low, horizon)

    valid_indices = np.arange(0, max(0, len(df) - horizon))
    if len(valid_indices) == 0:
        raise ValueError(f"not enough rows for horizon={horizon}")

    rng = np.random.default_rng(random_state)
    if len(valid_indices) <= sample_count:
        sample_indices = valid_indices
    else:
        sample_indices = np.sort(rng.choice(valid_indices, size=sample_count, replace=False))

    rows = []
    all_pass = True

    for i in sample_indices:
        expected_high = high.iloc[i + 1 : i + 1 + horizon].max()
        expected_low = low.iloc[i + 1 : i + 1 + horizon].min()

        explicit_high_i = explicit_high.iloc[i]
        explicit_low_i = explicit_low.iloc[i]
        rolling_high_i = rolling_high.iloc[i]
        rolling_low_i = rolling_low.iloc[i]

        explicit_high_pass = bool(np.isclose(expected_high, explicit_high_i, atol=atol, rtol=0))
        explicit_low_pass = bool(np.isclose(expected_low, explicit_low_i, atol=atol, rtol=0))

        # rolling can be nan near boundaries depending on implementation; compare where not nan.
        rolling_high_pass = bool(np.isclose(expected_high, rolling_high_i, atol=atol, rtol=0)) if not pd.isna(rolling_high_i) else False
        rolling_low_pass = bool(np.isclose(expected_low, rolling_low_i, atol=atol, rtol=0)) if not pd.isna(rolling_low_i) else False

        row_pass = explicit_high_pass and explicit_low_pass and rolling_high_pass and rolling_low_pass
        all_pass = all_pass and row_pass

        rows.append({
            "horizon": horizon,
            "index": int(i),
            "date": dates.iloc[i],
            "window_start_index": int(i + 1),
            "window_end_index_inclusive": int(i + horizon),
            "window_start_date": dates.iloc[i + 1],
            "window_end_date": dates.iloc[i + horizon],
            "expected_future_high": float(expected_high),
            "explicit_future_high": float(explicit_high_i),
            "rolling_future_high": float(rolling_high_i) if not pd.isna(rolling_high_i) else np.nan,
            "expected_future_low": float(expected_low),
            "explicit_future_low": float(explicit_low_i),
            "rolling_future_low": float(rolling_low_i) if not pd.isna(rolling_low_i) else np.nan,
            "explicit_high_pass": explicit_high_pass,
            "explicit_low_pass": explicit_low_pass,
            "rolling_high_pass": rolling_high_pass,
            "rolling_low_pass": rolling_low_pass,
            "row_pass": row_pass,
        })

    return all_pass, pd.DataFrame(rows)


def validate_current_vol_no_current_return(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int,
    sample_count: int = 30,
    random_state: int = 42,
    atol: float = 1e-12,
) -> Tuple[bool, pd.DataFrame]:
    """
    current_horizon_vol[i]가 returns up to i-1만 사용하는지 수동 검증.
    """
    close = df["close"].astype(float).reset_index(drop=True)
    dates = df["date"].reset_index(drop=True)
    returns = close.pct_change()
    min_periods = rolling_min_periods(vol_window, min_floor=20, frac=0.5)

    computed = current_horizon_volatility(close, horizon=horizon, vol_window=vol_window, min_periods=min_periods)

    valid_indices = np.arange(vol_window + 2, len(df) - horizon)
    if len(valid_indices) == 0:
        valid_indices = np.arange(min_periods + 2, len(df) - horizon)
    if len(valid_indices) == 0:
        raise ValueError(f"not enough rows for vol validation. rows={len(df)}, horizon={horizon}, vol_window={vol_window}")

    rng = np.random.default_rng(random_state + 777)
    if len(valid_indices) <= sample_count:
        sample_indices = valid_indices
    else:
        sample_indices = np.sort(rng.choice(valid_indices, size=sample_count, replace=False))

    rows = []
    all_pass = True

    for i in sample_indices:
        # rolling std at i after shift(1) uses returns[(i-vol_window) : i], i.e. up to i-1
        start = max(0, i - vol_window)
        window_returns = returns.iloc[start:i].dropna()

        if len(window_returns) < min_periods:
            expected = np.nan
        else:
            # pandas rolling std uses ddof=1 by default
            expected = window_returns.std(ddof=1) * math.sqrt(horizon)

        actual = computed.iloc[i]

        if pd.isna(expected) and pd.isna(actual):
            row_pass = True
        else:
            row_pass = bool(np.isclose(expected, actual, atol=atol, rtol=1e-10))

        all_pass = all_pass and row_pass

        rows.append({
            "horizon": horizon,
            "index": int(i),
            "date": dates.iloc[i],
            "return_window_start_index": int(start),
            "return_window_end_index_inclusive": int(i - 1),
            "uses_return_at_current_index": False,
            "expected_current_horizon_vol": float(expected) if not pd.isna(expected) else np.nan,
            "computed_current_horizon_vol": float(actual) if not pd.isna(actual) else np.nan,
            "row_pass": row_pass,
        })

    return all_pass, pd.DataFrame(rows)


# ============================================================
# 3. Distribution Diagnostics
# ============================================================

def label_distribution(labels: pd.DataFrame, asset_name: str, horizon: int, k_up: float, k_down: float) -> Dict:
    valid = labels.dropna(subset=["y_up_touch", "y_down_touch"]).copy()
    if valid.empty:
        return {
            "asset_name": asset_name,
            "horizon": horizon,
            "k_up": k_up,
            "k_down": k_down,
            "valid_rows": 0,
        }

    out = {
        "asset_name": asset_name,
        "horizon": horizon,
        "k_up": k_up,
        "k_down": k_down,
        "valid_rows": int(len(valid)),
        "up_touch_rate": float(valid["y_up_touch"].mean()),
        "down_touch_rate": float(valid["y_down_touch"].mean()),
        "both_touch_rate": float(((valid["y_up_touch"] == 1) & (valid["y_down_touch"] == 1)).mean()),
        "no_touch_rate": float(((valid["y_up_touch"] == 0) & (valid["y_down_touch"] == 0)).mean()),
        "up_only_rate": float(((valid["y_up_touch"] == 1) & (valid["y_down_touch"] == 0)).mean()),
        "down_only_rate": float(((valid["y_up_touch"] == 0) & (valid["y_down_touch"] == 1)).mean()),
        "avg_current_horizon_vol": float(valid["current_horizon_vol"].mean()),
        "median_current_horizon_vol": float(valid["current_horizon_vol"].median()),
    }
    return out


def annual_positive_rates(labels: pd.DataFrame, asset_name: str, horizon: int, k_up: float, k_down: float) -> pd.DataFrame:
    valid = labels.dropna(subset=["y_up_touch", "y_down_touch"]).copy()
    if valid.empty:
        return pd.DataFrame()

    valid["year"] = pd.to_datetime(valid["date"]).dt.year

    rows = []
    for year, g in valid.groupby("year"):
        rows.append({
            "asset_name": asset_name,
            "year": int(year),
            "horizon": horizon,
            "k_up": k_up,
            "k_down": k_down,
            "rows": int(len(g)),
            "up_touch_rate": float(g["y_up_touch"].mean()),
            "down_touch_rate": float(g["y_down_touch"].mean()),
            "both_touch_rate": float(((g["y_up_touch"] == 1) & (g["y_down_touch"] == 1)).mean()),
            "no_touch_rate": float(((g["y_up_touch"] == 0) & (g["y_down_touch"] == 0)).mean()),
        })

    return pd.DataFrame(rows)


def joint_touch_distribution(labels: pd.DataFrame, asset_name: str, horizon: int, k_up: float, k_down: float) -> pd.DataFrame:
    valid = labels.dropna(subset=["y_up_touch", "y_down_touch"]).copy()
    if valid.empty:
        return pd.DataFrame()

    grouped = (
        valid.groupby(["y_up_touch", "y_down_touch", "touch_regime"])
        .size()
        .reset_index(name="count")
    )
    grouped["asset_name"] = asset_name
    grouped["horizon"] = horizon
    grouped["k_up"] = k_up
    grouped["k_down"] = k_down
    grouped["rate"] = grouped["count"] / grouped["count"].sum()
    return grouped[["asset_name", "horizon", "k_up", "k_down", "y_up_touch", "y_down_touch", "touch_regime", "count", "rate"]]


def coefficient_of_variation(s: pd.Series) -> float:
    x = pd.Series(s).dropna().astype(float)
    if len(x) == 0 or x.mean() == 0:
        return np.nan
    return float(x.std(ddof=1) / x.mean())


def annual_rate_stability(annual_df: pd.DataFrame) -> pd.DataFrame:
    if annual_df.empty:
        return pd.DataFrame()

    rows = []
    key_cols = ["asset_name", "horizon", "k_up", "k_down"]
    for key, g in annual_df.groupby(key_cols):
        row = dict(zip(key_cols, key if isinstance(key, tuple) else (key,)))
        row.update({
            "up_touch_annual_mean": float(g["up_touch_rate"].mean()),
            "up_touch_annual_std": float(g["up_touch_rate"].std(ddof=1)),
            "up_touch_annual_cv": coefficient_of_variation(g["up_touch_rate"]),
            "down_touch_annual_mean": float(g["down_touch_rate"].mean()),
            "down_touch_annual_std": float(g["down_touch_rate"].std(ddof=1)),
            "down_touch_annual_cv": coefficient_of_variation(g["down_touch_rate"]),
            "both_touch_annual_mean": float(g["both_touch_rate"].mean()),
            "no_touch_annual_mean": float(g["no_touch_rate"].mean()),
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 4. Target Positive Rate k Search
# ============================================================

def touch_rate_for_k(
    df: pd.DataFrame,
    horizon: int,
    vol_window: int,
    side: str,
    k: float,
) -> float:
    if side == "up":
        labels = make_touch_labels_explicit(df, horizon=horizon, vol_window=vol_window, k_up=k, k_down=999.0)
        return float(labels["y_up_touch"].dropna().mean())
    if side == "down":
        labels = make_touch_labels_explicit(df, horizon=horizon, vol_window=vol_window, k_up=999.0, k_down=k)
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
    """
    scipy 없이 grid search로 target positive rate에 가장 가까운 k를 찾음.
    k가 커질수록 touch rate는 대체로 감소.
    """
    ks = np.linspace(k_min, k_max, grid_size)
    rows = []

    for k in ks:
        rate = touch_rate_for_k(df, horizon=horizon, vol_window=vol_window, side=side, k=float(k))
        rows.append({"k": float(k), "positive_rate": rate, "abs_error": abs(rate - target_rate)})

    res = pd.DataFrame(rows).sort_values("abs_error").iloc[0].to_dict()

    return {
        "side": side,
        "horizon": horizon,
        "target_rate": target_rate,
        "best_k": float(res["k"]),
        "achieved_positive_rate": float(res["positive_rate"]),
        "abs_error": float(res["abs_error"]),
        "k_min": k_min,
        "k_max": k_max,
        "grid_size": grid_size,
    }


# ============================================================
# 5. Main
# ============================================================

def run(args) -> Dict[str, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv(args.input)
    if args.start_date:
        df = df[df["date"] >= pd.to_datetime(args.start_date)].copy()
    if args.end_date:
        df = df[df["date"] <= pd.to_datetime(args.end_date)].copy()
    df = df.reset_index(drop=True)

    horizons = parse_int_list(args.horizons)
    k_values = parse_float_list(args.k_values)
    target_rates = parse_float_list(args.target_rates)

    alignment_rows = []
    vol_rows = []
    distribution_rows = []
    annual_rows = []
    joint_rows = []
    target_k_rows = []
    preview_parts = []

    overall_pass = True
    horizon_pass_summary = []

    for horizon in horizons:
        future_pass, future_checks = validate_future_window_alignment(
            df,
            horizon=horizon,
            sample_count=args.sample_count,
            random_state=args.random_state,
        )
        future_checks["check_type"] = "future_high_low_alignment"
        alignment_rows.append(future_checks)

        vol_pass, vol_checks = validate_current_vol_no_current_return(
            df,
            horizon=horizon,
            vol_window=args.vol_window,
            sample_count=args.sample_count,
            random_state=args.random_state,
        )
        vol_checks["check_type"] = "current_vol_no_current_return"
        vol_rows.append(vol_checks)

        overall_pass = overall_pass and future_pass and vol_pass
        horizon_pass_summary.append({
            "horizon": horizon,
            "future_window_alignment_pass": bool(future_pass),
            "current_vol_no_current_return_pass": bool(vol_pass),
        })

        for k in k_values:
            labels = make_touch_labels_explicit(
                df,
                horizon=horizon,
                vol_window=args.vol_window,
                k_up=k,
                k_down=k,
            )

            distribution_rows.append(label_distribution(labels, args.asset_name, horizon, k, k))
            annual_rows.append(annual_positive_rates(labels, args.asset_name, horizon, k, k))
            joint_rows.append(joint_touch_distribution(labels, args.asset_name, horizon, k, k))

            # preview only for first few rows of each config
            preview_cols = [
                "date", "close", "high", "low",
                "current_horizon_vol", "upper_barrier", "lower_barrier",
                "future_high_h", "future_low_h",
                "y_up_touch", "y_down_touch", "touch_regime",
            ]
            preview = labels[preview_cols].head(args.preview_rows).copy()
            preview["asset_name"] = args.asset_name
            preview["horizon"] = horizon
            preview["k_up"] = k
            preview["k_down"] = k
            preview_parts.append(preview)

        for target in target_rates:
            target_k_rows.append(find_k_for_target_rate_grid(
                df=df,
                horizon=horizon,
                vol_window=args.vol_window,
                side="up",
                target_rate=target,
                k_min=args.k_min,
                k_max=args.k_max,
                grid_size=args.k_grid_size,
            ))
            target_k_rows.append(find_k_for_target_rate_grid(
                df=df,
                horizon=horizon,
                vol_window=args.vol_window,
                side="down",
                target_rate=target,
                k_min=args.k_min,
                k_max=args.k_max,
                grid_size=args.k_grid_size,
            ))

    alignment_df = pd.concat(alignment_rows + vol_rows, ignore_index=True) if alignment_rows or vol_rows else pd.DataFrame()
    label_dist_df = pd.DataFrame(distribution_rows)
    annual_df = pd.concat([x for x in annual_rows if not x.empty], ignore_index=True) if annual_rows else pd.DataFrame()
    annual_stability_df = annual_rate_stability(annual_df)
    joint_df = pd.concat([x for x in joint_rows if not x.empty], ignore_index=True) if joint_rows else pd.DataFrame()
    target_k_df = pd.DataFrame(target_k_rows)
    preview_df = pd.concat(preview_parts, ignore_index=True) if preview_parts else pd.DataFrame()

    # Flag unstable annual positive rates.
    if not annual_stability_df.empty:
        annual_stability_df["up_rate_cv_gt_0p3"] = annual_stability_df["up_touch_annual_cv"] > 0.3
        annual_stability_df["down_rate_cv_gt_0p3"] = annual_stability_df["down_touch_annual_cv"] > 0.3

    summary = {
        "experiment": "touch_label_alignment_validator",
        "asset_name": args.asset_name,
        "input": str(args.input),
        "period": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "rows": int(len(df)),
        },
        "config": {
            "horizons": horizons,
            "k_values": k_values,
            "vol_window": args.vol_window,
            "target_rates": target_rates,
            "future_window_method": "explicit_loop_i_plus_1_to_i_plus_H",
            "current_horizon_vol": "returns.rolling(vol_window).std().shift(1) * sqrt(horizon)",
        },
        "validation": {
            "overall_pass": bool(overall_pass),
            "horizon_pass_summary": horizon_pass_summary,
            "alignment_checked_rows": int(len(alignment_df)),
            "failed_alignment_rows": int((alignment_df.get("row_pass", pd.Series(dtype=bool)) == False).sum()) if not alignment_df.empty else 0,
        },
        "label_distribution_top": label_dist_df.to_dict("records") if not label_dist_df.empty else [],
        "annual_stability": annual_stability_df.to_dict("records") if not annual_stability_df.empty else [],
        "target_rate_k_search": target_k_df.to_dict("records") if not target_k_df.empty else [],
        "decision_note": (
            "If overall_pass is false, do not use generated labels for model training. "
            "If annual positive-rate CV is high, compare fixed-k with target-rate calibrated-k labels."
        ),
    }

    outputs = {
        "summary": save_json(out_dir / "touch_label_validation_summary.json", summary),
        "alignment_checks": save_csv(out_dir / "alignment_checks.csv", alignment_df),
        "label_distribution": save_csv(out_dir / "label_distribution.csv", label_dist_df),
        "annual_positive_rate": save_csv(out_dir / "annual_positive_rate.csv", annual_df),
        "annual_rate_stability": save_csv(out_dir / "annual_rate_stability.csv", annual_stability_df),
        "joint_touch_distribution": save_csv(out_dir / "joint_touch_distribution.csv", joint_df),
        "target_rate_k_search": save_csv(out_dir / "target_rate_k_search.csv", target_k_df),
        "generated_labels_preview": save_csv(out_dir / "generated_labels_preview.csv", preview_df),
    }

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="OHLCV CSV path")
    parser.add_argument("--asset-name", default="QQQ")
    parser.add_argument("--output-dir", default="touch_label_validation_output")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")

    parser.add_argument("--horizons", default="10,20,40")
    parser.add_argument("--k-values", default="0.5,0.75,1.0")
    parser.add_argument("--target-rates", default="0.25,0.30,0.35")

    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--preview-rows", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--k-min", type=float, default=0.25)
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--k-grid-size", type=int, default=80)

    args = parser.parse_args()

    outputs = run(args)
    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))

    print("[OK] Touch label alignment validation completed.")
    print(json.dumps({
        "asset_name": summary["asset_name"],
        "period": summary["period"],
        "validation": summary["validation"],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
