#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v8.6.40d PH Context EDA

Purpose
-------
Create diagnostics for the next v8.6.40d policy before changing allocation logic.
This script does NOT change model predictions or allocation. It reads existing
*_predictions.csv outputs and generates ph_raw / ph_rank / ph_z diagnostics by
asset class and mid-trend state.

Key design choices
------------------
1. ph_rank is computed from prob_high_vol_raw when available.
2. Rolling rank uses only prior observations: values[t-window:t], then compares current ph[t].
3. min_periods defaults to 504. Before enough history exists, ph_rank is NaN and mode is raw_fallback.
4. EWMA-smoothed prob_high_vol is preserved as ph_ewma for diagnostics only.

Example
-------
python v8_6_40d_ph_context_eda.py ^
  --result-dir results_v8_6_40b_clean_compare ^
  --asset-list QQQ,SPY,AAPL,SOXX,NVDA ^
  --out-dir results_v8_6_40d_eda_clean
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)


DEFAULT_ASSET_CLASS_MAP: Dict[str, str] = {
    "QQQ": "broad_index",
    "SPY": "broad_index",
    "DIA": "broad_index",
    "IWM": "broad_index",
    "AAPL": "mega_cap",
    "MSFT": "mega_cap",
    "GOOGL": "mega_cap",
    "GOOG": "mega_cap",
    "AMZN": "mega_cap",
    "META": "mega_cap",
    "SOXX": "sector_etf",
    "SMH": "sector_etf",
    "XLK": "sector_etf",
    "NVDA": "high_vol_growth",
    "TSLA": "high_vol_growth",
    "AMD": "high_vol_growth",
}

PH_BIN_EDGES = [0.0, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0000001]
PH_BIN_LABELS = ["00_30", "30_50", "50_70", "70_85", "85_95", "95_100"]


@dataclass(frozen=True)
class EDAConfig:
    result_dir: Path
    out_dir: Path
    asset_list: List[str]
    rank_windows: Tuple[int, ...] = (504, 756)
    min_periods: int = 504
    ph_source_preference: Tuple[str, ...] = ("prob_high_vol_raw", "prob_high_vol")
    save_enriched: bool = True


def parse_asset_list(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def infer_asset_class(ticker: str) -> str:
    return DEFAULT_ASSET_CLASS_MAP.get(ticker.upper(), "generic_equity")


def find_predictions_file(result_dir: Path, ticker: str) -> Optional[Path]:
    """Find predictions file in either nested result folders or flat uploads."""
    t = ticker.lower()
    candidates = [
        result_dir / t / f"{t}_xgb_recency_weighted_v8_6_40b_predictions.csv",
        result_dir / t / f"{t}_xgb_recency_weighted_v8_6_40c_predictions.csv",
        result_dir / t / f"{t}_xgb_recency_weighted_v8_6_39_predictions.csv",
        result_dir / f"{t}_xgb_recency_weighted_v8_6_40b_predictions.csv",
        result_dir / f"{t}_xgb_recency_weighted_v8_6_40c_predictions.csv",
        result_dir / f"{t}_xgb_recency_weighted_v8_6_39_predictions.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Generic fallback: any file containing ticker and predictions.
    patterns = [f"**/{t}_*predictions.csv", f"**/{ticker.upper()}_*predictions.csv"]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(result_dir.glob(pat))
    hits = sorted(set(hits), key=lambda x: (len(str(x)), str(x)))
    return hits[0] if hits else None


def choose_ph_source(df: pd.DataFrame, preferences: Sequence[str]) -> str:
    for c in preferences:
        if c in df.columns:
            return c
    raise ValueError(f"No PH source column found. Tried: {preferences}")


def rolling_percentile_rank_prior(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Rank current value against prior window only. No look-ahead.

    rank[t] = mean(history <= values[t]) where history = values[max(0,t-window):t]
    If history length after finite filtering is below min_periods, return NaN.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    for i, v in enumerate(arr):
        if not np.isfinite(v):
            continue
        lo = max(0, i - window)
        hist = arr[lo:i]
        hist = hist[np.isfinite(hist)]
        if hist.size < min_periods:
            continue
        out[i] = float(np.mean(hist <= v))
    return out


def rolling_zscore_prior(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    for i, v in enumerate(arr):
        if not np.isfinite(v):
            continue
        lo = max(0, i - window)
        hist = arr[lo:i]
        hist = hist[np.isfinite(hist)]
        if hist.size < min_periods:
            continue
        std = float(np.std(hist, ddof=1)) if hist.size > 1 else np.nan
        if not np.isfinite(std) or std <= 0:
            continue
        out[i] = float((v - float(np.mean(hist))) / std)
    return out


def annualized_return(ret: pd.Series, periods_per_year: int = 252) -> float:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    if r.empty:
        return np.nan
    gross = float(np.prod(1.0 + r.values))
    if gross <= 0:
        return -1.0
    return gross ** (periods_per_year / len(r)) - 1.0


def max_drawdown(ret: pd.Series) -> float:
    r = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    if r.empty:
        return np.nan
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def safe_mean(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else np.nan


def make_group_metrics(g: pd.DataFrame) -> pd.Series:
    stock_ret_col = "stock_next_return" if "stock_next_return" in g.columns else None
    strategy_col = "strategy_return_net" if "strategy_return_net" in g.columns else None
    gross_col = "strategy_return_gross" if "strategy_return_gross" in g.columns else None

    res: Dict[str, object] = {
        "count": int(len(g)),
        "start_date": str(g["Date"].min()) if "Date" in g.columns else "",
        "end_date": str(g["Date"].max()) if "Date" in g.columns else "",
        "avg_stock_weight": safe_mean(g["stock_weight"]) if "stock_weight" in g.columns else np.nan,
        "avg_base_signal_stock_weight": safe_mean(g["base_signal_stock_weight"]) if "base_signal_stock_weight" in g.columns else np.nan,
        "avg_signal_stock_weight": safe_mean(g["signal_stock_weight"]) if "signal_stock_weight" in g.columns else np.nan,
        "avg_ph_raw": safe_mean(g["ph_raw"]),
        "avg_ph_ewma": safe_mean(g["ph_ewma"]) if "ph_ewma" in g.columns else np.nan,
        "avg_ph_rank_504": safe_mean(g["ph_rank_504"]) if "ph_rank_504" in g.columns else np.nan,
        "avg_ph_rank_756": safe_mean(g["ph_rank_756"]) if "ph_rank_756" in g.columns else np.nan,
        "avg_ph_z_756": safe_mean(g["ph_z_756"]) if "ph_z_756" in g.columns else np.nan,
        "avg_up_strength_score": safe_mean(g["prob_up_strengthening_score"]) if "prob_up_strengthening_score" in g.columns else np.nan,
        "avg_down_strength_score": safe_mean(g["prob_down_strengthening_score"]) if "prob_down_strengthening_score" in g.columns else np.nan,
        "offensive_activation_rate": safe_mean(g["offensive_tier"].astype(float) > 0) if "offensive_tier" in g.columns else np.nan,
        "tier3_rate": safe_mean(g["tier3_signal"].astype(float)) if "tier3_signal" in g.columns else np.nan,
        "ph_context_available_rate_756": safe_mean(g["ph_context_available_756"].astype(float)) if "ph_context_available_756" in g.columns else np.nan,
    }
    if strategy_col:
        res["strategy_ann_return"] = annualized_return(g[strategy_col])
        res["strategy_mdd"] = max_drawdown(g[strategy_col])
    if gross_col:
        res["strategy_gross_ann_return"] = annualized_return(g[gross_col])
    if stock_ret_col:
        res["bh_ann_return"] = annualized_return(g[stock_ret_col])
        res["bh_mdd"] = max_drawdown(g[stock_ret_col])
    if strategy_col and stock_ret_col:
        res["bh_gap_ann_return"] = res.get("strategy_ann_return", np.nan) - res.get("bh_ann_return", np.nan)  # type: ignore[operator]
    return pd.Series(res)


def add_ph_context(df: pd.DataFrame, ticker: str, cfg: EDAConfig) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    ph_src = choose_ph_source(df, cfg.ph_source_preference)
    df["ticker"] = ticker.upper()
    df["asset_class"] = infer_asset_class(ticker)
    df["ph_source"] = ph_src
    df["ph_raw"] = pd.to_numeric(df[ph_src], errors="coerce").clip(0.0, 1.0)
    if "prob_high_vol" in df.columns:
        df["ph_ewma"] = pd.to_numeric(df["prob_high_vol"], errors="coerce").clip(0.0, 1.0)
    else:
        df["ph_ewma"] = df["ph_raw"]

    values = df["ph_raw"].to_numpy(dtype=float)
    for w in cfg.rank_windows:
        rank_col = f"ph_rank_{w}"
        z_col = f"ph_z_{w}"
        avail_col = f"ph_context_available_{w}"
        mode_col = f"ph_context_mode_{w}"
        df[rank_col] = rolling_percentile_rank_prior(values, window=w, min_periods=cfg.min_periods)
        df[z_col] = rolling_zscore_prior(values, window=w, min_periods=cfg.min_periods)
        df[avail_col] = df[rank_col].notna()
        df[mode_col] = np.where(df[avail_col], "rank_policy_candidate", "raw_fallback")
        df[f"ph_rank_bin_{w}"] = pd.cut(
            df[rank_col],
            bins=PH_BIN_EDGES,
            labels=PH_BIN_LABELS,
            include_lowest=True,
            right=False,
        ).astype("object")
        df[f"ph_rank_bin_{w}"] = df[f"ph_rank_bin_{w}"].fillna("raw_fallback")

    if "mid_trend_state" not in df.columns:
        df["mid_trend_state"] = "UNKNOWN"
    if "allocation_regime" not in df.columns:
        df["allocation_regime"] = "UNKNOWN"
    if "signal_regime" not in df.columns:
        df["signal_regime"] = "UNKNOWN"
    if "hold_reason" not in df.columns:
        df["hold_reason"] = "UNKNOWN"

    return df, ph_src


def group_and_save(df: pd.DataFrame, by: List[str], path: Path) -> pd.DataFrame:
    out = df.groupby(by, dropna=False).apply(make_group_metrics).reset_index()
    # Stable useful sort.
    sort_cols = [c for c in by if c in out.columns]
    if "count" in out.columns:
        out = out.sort_values(sort_cols + ["count"], ascending=[True] * len(sort_cols) + [False])
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def summarize_ticker(df: pd.DataFrame, ticker: str, ph_src: str) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "ticker": ticker.upper(),
        "asset_class": infer_asset_class(ticker),
        "rows": int(len(df)),
        "date_start": str(df["Date"].min()) if "Date" in df.columns else "",
        "date_end": str(df["Date"].max()) if "Date" in df.columns else "",
        "ph_source_used": ph_src,
        "ph_raw_mean": safe_mean(df["ph_raw"]),
        "ph_raw_median": float(pd.to_numeric(df["ph_raw"], errors="coerce").median()),
        "ph_raw_p75": float(pd.to_numeric(df["ph_raw"], errors="coerce").quantile(0.75)),
        "ph_raw_p90": float(pd.to_numeric(df["ph_raw"], errors="coerce").quantile(0.90)),
        "ph_raw_p95": float(pd.to_numeric(df["ph_raw"], errors="coerce").quantile(0.95)),
        "ph_ewma_mean": safe_mean(df["ph_ewma"]),
        "ph_ewma_median": float(pd.to_numeric(df["ph_ewma"], errors="coerce").median()),
    }
    for w in [504, 756]:
        if f"ph_context_available_{w}" in df.columns:
            summary[f"ph_context_available_rate_{w}"] = safe_mean(df[f"ph_context_available_{w}"].astype(float))
            summary[f"ph_rank_{w}_mean"] = safe_mean(df[f"ph_rank_{w}"])
            summary[f"ph_rank_{w}_median"] = float(pd.to_numeric(df[f"ph_rank_{w}"], errors="coerce").median())
    if "strategy_return_net" in df.columns:
        summary["strategy_ann_return"] = annualized_return(df["strategy_return_net"])
        summary["strategy_mdd"] = max_drawdown(df["strategy_return_net"])
    if "stock_next_return" in df.columns:
        summary["bh_ann_return"] = annualized_return(df["stock_next_return"])
        summary["bh_mdd"] = max_drawdown(df["stock_next_return"])
    if "stock_weight" in df.columns:
        summary["avg_stock_weight"] = safe_mean(df["stock_weight"])
    return summary


def run_eda(cfg: EDAConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    all_frames: List[pd.DataFrame] = []
    run_summaries: List[Dict[str, object]] = []
    missing: List[str] = []

    for ticker in cfg.asset_list:
        p = find_predictions_file(cfg.result_dir, ticker)
        if p is None:
            missing.append(ticker)
            continue
        df = pd.read_csv(p)
        df, ph_src = add_ph_context(df, ticker, cfg)
        all_frames.append(df)
        run_summaries.append(summarize_ticker(df, ticker, ph_src))

        tdir = cfg.out_dir / ticker.lower()
        tdir.mkdir(parents=True, exist_ok=True)

        # Per-ticker outputs.
        group_and_save(df, ["ticker", "ph_rank_bin_756"], tdir / f"{ticker.lower()}_ph_context_distribution.csv")
        group_and_save(df, ["ticker", "ph_rank_bin_756", "mid_trend_state"], tdir / f"{ticker.lower()}_ph_rank_trend_performance.csv")
        group_and_save(df, ["ticker", "allocation_regime", "mid_trend_state", "ph_rank_bin_756"], tdir / f"{ticker.lower()}_regime_trend_ph_rank_performance.csv")
        group_and_save(df, ["ticker", "hold_reason", "ph_rank_bin_756", "mid_trend_state"], tdir / f"{ticker.lower()}_hold_reason_ph_rank_trend_performance.csv")

        if cfg.save_enriched:
            keep_extra = [
                "Date", "ticker", "asset_class", "ph_source", "ph_raw", "ph_ewma",
                "ph_rank_504", "ph_z_504", "ph_context_available_504", "ph_context_mode_504", "ph_rank_bin_504",
                "ph_rank_756", "ph_z_756", "ph_context_available_756", "ph_context_mode_756", "ph_rank_bin_756",
                "mid_trend_state", "mid_trend_score", "signal_regime", "allocation_regime", "executed_regime",
                "hold_reason", "stock_weight", "signal_stock_weight", "base_signal_stock_weight",
                "strategy_return_net", "strategy_return_gross", "stock_next_return",
                "prob_up_strengthening_score", "prob_down_strengthening_score", "offensive_tier", "tier3_signal",
            ]
            keep_cols = [c for c in keep_extra if c in df.columns]
            df[keep_cols].to_csv(tdir / f"{ticker.lower()}_ph_context_enriched_predictions.csv", index=False, encoding="utf-8-sig")

    if not all_frames:
        raise FileNotFoundError(f"No prediction files found under {cfg.result_dir}. Missing: {missing}")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(cfg.out_dir / "all_assets_ph_context_enriched_predictions_compact.csv", index=False, encoding="utf-8-sig")

    # Combined outputs.
    group_and_save(combined, ["ticker", "asset_class", "ph_rank_bin_756"], cfg.out_dir / "combined_ph_context_distribution.csv")
    group_and_save(combined, ["asset_class", "ph_rank_bin_756", "mid_trend_state"], cfg.out_dir / "asset_class_ph_rank_trend_performance.csv")
    group_and_save(combined, ["ticker", "ph_rank_bin_756", "mid_trend_state"], cfg.out_dir / "ticker_ph_rank_trend_performance.csv")
    group_and_save(combined, ["asset_class", "allocation_regime", "mid_trend_state", "ph_rank_bin_756"], cfg.out_dir / "asset_class_regime_trend_ph_rank_performance.csv")
    group_and_save(combined, ["ticker", "allocation_regime", "mid_trend_state", "ph_rank_bin_756"], cfg.out_dir / "ticker_regime_trend_ph_rank_performance.csv")

    # Summary JSON/CSV.
    summary = {
        "script": "v8_6_40d_ph_context_eda.py",
        "purpose": "EDA-only PH context diagnostics for v8.6.40d policy design",
        "result_dir": str(cfg.result_dir),
        "out_dir": str(cfg.out_dir),
        "asset_list": cfg.asset_list,
        "rank_windows": list(cfg.rank_windows),
        "min_periods": cfg.min_periods,
        "ph_source_preference": list(cfg.ph_source_preference),
        "missing_tickers": missing,
        "tickers": run_summaries,
        "important_notes": [
            "ph_rank is computed from prob_high_vol_raw when available.",
            "rolling rank uses prior observations only; current row is compared to previous window.",
            "rows with insufficient history are marked raw_fallback, not forced into ph_rank policy.",
            "this script does not modify allocation policy or backtest returns.",
        ],
    }
    with open(cfg.out_dir / "ph_context_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(run_summaries).to_csv(cfg.out_dir / "ph_context_ticker_summary.csv", index=False, encoding="utf-8-sig")

    print("[OK] PH context EDA completed")
    print(f"  input : {cfg.result_dir}")
    print(f"  output: {cfg.out_dir}")
    if missing:
        print(f"  missing tickers: {','.join(missing)}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v8.6.40d PH context EDA from existing prediction outputs")
    p.add_argument("--result-dir", required=True, help="Root result directory containing ticker subfolders or flat prediction CSV files")
    p.add_argument("--out-dir", default=None, help="Output directory. Default: <result-dir>_v8_6_40d_eda")
    p.add_argument("--asset-list", default="QQQ,SPY,AAPL,SOXX,NVDA", help="Comma-separated tickers")
    p.add_argument("--rank-windows", default="504,756", help="Comma-separated rolling windows")
    p.add_argument("--min-periods", type=int, default=504, help="Minimum prior observations before ph_rank is available")
    p.add_argument("--no-save-enriched", action="store_true", help="Do not save compact enriched prediction files")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(str(result_dir).rstrip("/\\") + "_v8_6_40d_eda")
    rank_windows = tuple(int(x.strip()) for x in args.rank_windows.split(",") if x.strip())
    cfg = EDAConfig(
        result_dir=result_dir,
        out_dir=out_dir,
        asset_list=parse_asset_list(args.asset_list),
        rank_windows=rank_windows,
        min_periods=int(args.min_periods),
        save_enriched=not bool(args.no_save_enriched),
    )
    run_eda(cfg)


if __name__ == "__main__":
    main()
