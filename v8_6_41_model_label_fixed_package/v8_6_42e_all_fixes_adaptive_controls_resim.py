#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v8.6.42e All-Fixes Adaptive Controls Resimulation

All improvements identified across reviews are applied here.

Fixes vs v8.6.42d
------------------
[HIGH]   add_adaptive_down_weights: bonus sizes now driven by _component_skill()
         rolling skill, not hardcoded rule offsets.
[HIGH]   context score: t-stat confidence weighting added.
         n=40 gave power=5% (noise). Confidence = clip(|t|/2, 0, 1) shrinks
         context_adjust toward 0 when history is statistically unreliable.
         Ref: Bailey et al. (2014) "Probability of Backtest Overfitting".
[HIGH]   context score: excess return (vs cash) replaces raw return.
         Raw return mixes alpha + beta. Using (stock_next_return - cash_return)
         isolates alpha-like contribution.
         Ref: Ang & Bekaert (2004) "How Regimes Affect Asset Allocation".
[MODERATE] context_tail10: now actually computed (rolling 10th-percentile
         return over context window), was np.nan placeholder previously.
[MODERATE] adaptive_risk_weights: dead code removed. Weights are fully
         vectorized in add_adaptive_overall_risk; the standalone function
         was unreachable after 42d refactor.
[MODERATE] full_stock_high_vol_max: now asset_class-aware dict instead of
         single scalar. high_vol_growth can tolerate higher ph_rank at full-stock.
[MINOR]  compute_deflated_sr: Deflated Sharpe Ratio utility added to
         summarize_ticker and multi_asset_summary output.
         Ref: Bailey et al. (2014), Lopez de Prado (2018) Ch.7.

Previously fixed (42c / 42d retained)
---------------------------------------
simulate_execution max_weight_change normalization | w_up 4-weight denom
broad_index_min_stock floor | rolling_signal_skills connected
context_min_rows 120 | sigmoid midpoint rolling | sector_etf max_s 0.88
context fallback fix | context_return_scale 252 | up_tier1 0.40
EWMA vol regime | base_down_rate per AC | ph_rank guard on floors
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Config / asset class
# ============================================================

BROAD_INDEX = {"QQQ", "SPY", "DIA", "IWM", "VTI", "VOO", "IVV", "SCHB", "SPLG"}
MEGA_CAP = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "AVGO", "LLY", "COST", "NFLX"}
SECTOR_ETF = {"SOXX", "SMH", "XLK", "XLY", "XLF", "XLV", "XLE", "XLI", "XLC", "ARKK"}
HIGH_VOL_GROWTH = {"NVDA", "TSLA", "AMD", "PLTR", "MSTR", "COIN", "SMCI"}

# Empirical/structural baseline for daily down-frequency by asset class.
# Used only as a neutral reference in context score, not as a ticker-specific optimizer.
ASSET_CLASS_BASE_DOWN_RATE = {
    "broad_index": 0.48,
    "sector_etf": 0.51,
    "mega_cap": 0.49,
    "high_vol_growth": 0.54,
    "single_stock": 0.50,
    "unknown": 0.50,
}


def infer_asset_class(ticker: str) -> str:
    t = str(ticker).upper().strip()
    if t in BROAD_INDEX:
        return "broad_index"
    if t in SECTOR_ETF:
        return "sector_etf"
    if t in HIGH_VOL_GROWTH:
        return "high_vol_growth"
    if t in MEGA_CAP:
        return "mega_cap"
    return "single_stock"


@dataclass
class AdaptiveControlsConfig:
    initial_capital: float = 100_000_000.0
    transaction_cost_rate: float = 0.001
    rebalance_every_n_days: int = 5
    no_trade_band: float = 0.12
    max_weight_change_per_rebalance: float = 0.20

    # 11. multi-window ph-rank ensemble
    rank_windows: Tuple[int, ...] = (504, 756, 1008)
    rank_min_periods: int = 252

    # 12. rolling context table
    context_window: int = 756
    context_min_rows: int = 120          # n=40→power=5% (noise); 120 gives SE≈0.09%/day (Bailey 2014)
    context_global_min_rows: int = 160   # keep ratio ~4/3 vs context_min_rows
    context_return_scale: float = 252.0
    context_down_penalty: float = 0.70
    context_tail_penalty: float = 4.0
    context_adjust_max: float = 0.08  # kept for backward compatibility
    context_adjust_max_positive: float = 0.03
    context_adjust_max_negative: float = 0.08
    context_positive_score_threshold: float = 0.20
    context_negative_score_threshold: float = -0.15
    broad_index_allow_positive_context_adjust: bool = False

    # 9. adaptive EWMA span
    use_adaptive_ewma: bool = True
    ewma_min_span: int = 3
    ewma_max_span: int = 14

    # 7. adaptive overall risk
    use_adaptive_overall_risk: bool = True
    high_vol_rank_threshold: float = 0.70

    # 8. adaptive down component weight
    use_adaptive_down_components: bool = True
    branch_perf_window: int = 756
    branch_perf_min_rows: int = 252
    branch_softmax_temperature: float = 0.12
    branch_weight_floor: float = 0.05

    # 10. adaptive signal recency/memory control. This is allocation-layer decay, not model training half-life.
    use_adaptive_signal_memory: bool = True
    stale_signal_decay_days_broad: int = 15
    stale_signal_decay_days_other: int = 10

    # offensive overlay controls
    enable_upside_overlay: bool = True
    up_tier1: float = 0.40  # raised from 0.30; weak <0.40 up-strength had little alpha in EDA
    up_tier2: float = 0.46
    up_tier3: float = 0.52
    full_stock_up_score: float = 0.58
    # full_stock_high_vol_max: asset_class-aware dict
    # high_vol_growth can sustain full-stock at higher ph_rank than broad_index
    full_stock_high_vol_max_by_ac: Dict[str, float] = None  # type: ignore  (set in __post_init__)

    # risk / allocation thresholds
    extreme_risk_threshold: float = 0.86
    risk_off_threshold: float = 0.74
    watch_threshold: float = 0.62

    # asset-class guardrails, deliberately broad, not ticker-specific
    broad_index_max_stock: float = 0.86
    broad_index_enable_upside_overlay: bool = False
    mega_cap_bear_cap: float = 0.74
    mega_cap_extreme_bear_cap: float = 0.62
    sector_bull_floor: float = 0.74
    growth_bull_floor: float = 0.70
    unknown_max_stock: float = 0.82

    # v8.6.42d additions
    # --- broad_index floor (fix: 42c avg_stock 63.8% structural collapse) ---
    broad_index_min_stock: float = 0.72

    version: str = "v8.6.42e_all_fixes_adaptive_controls"

    def __post_init__(self):
        # Mutable dict default handled via __post_init__ to avoid dataclass ValueError.
        # Using self.__dict__ assignment is safer than object.__setattr__ for non-frozen classes.
        if self.full_stock_high_vol_max_by_ac is None:
            self.__dict__["full_stock_high_vol_max_by_ac"] = {
                "broad_index":    0.52,
                "sector_etf":     0.58,
                "mega_cap":       0.55,
                "high_vol_growth":0.68,
                "single_stock":   0.58,
                "unknown":        0.55,
            }


def parse_windows(s: str) -> Tuple[int, ...]:
    vals = []
    for x in str(s).split(','):
        x = x.strip()
        if x:
            vals.append(int(x))
    if not vals:
        raise ValueError("rank window list is empty")
    return tuple(vals)


# ============================================================
# Numeric utilities
# ============================================================

def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def rolling_rank_past(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rolling percentile rank using only past observations, excluding current row."""
    shifted = s.shift(1)
    def _rank(arr: np.ndarray) -> float:
        if len(arr) == 0 or np.isnan(arr[-1]):
            return np.nan
        cur = arr[-1]
        hist = arr[:-1]
        hist = hist[np.isfinite(hist)]
        if len(hist) == 0:
            return np.nan
        return float((hist <= cur).mean())
    # combine past history and current value by rolling on original, then internally exclude last element's history?
    # Simpler and correct enough: for each row, use shifted rolling hist and current value.
    out = []
    values = s.to_numpy(dtype=float)
    for i, cur in enumerate(values):
        start = max(0, i - window)
        hist = values[start:i]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(cur) or len(hist) < min_periods:
            out.append(np.nan)
        else:
            out.append(float((hist <= cur).mean()))
    return pd.Series(out, index=s.index, dtype=float)


def expanding_rank_past(s: pd.Series, min_periods: int) -> pd.Series:
    out = []
    vals = s.to_numpy(dtype=float)
    for i, cur in enumerate(vals):
        hist = vals[:i]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(cur) or len(hist) < min_periods:
            out.append(np.nan)
        else:
            out.append(float((hist <= cur).mean()))
    return pd.Series(out, index=s.index, dtype=float)


def adaptive_ewma(raw: pd.Series, span: pd.Series, default_span: int = 7) -> pd.Series:
    vals = raw.astype(float).to_numpy()
    spans = span.fillna(default_span).astype(float).clip(2.0, 60.0).to_numpy()
    out = np.full(len(vals), np.nan, dtype=float)
    prev = np.nan
    for i, x in enumerate(vals):
        if not np.isfinite(x):
            out[i] = prev
            continue
        if not np.isfinite(prev):
            prev = x
        else:
            alpha = 2.0 / (spans[i] + 1.0)
            prev = alpha * x + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=raw.index, dtype=float)


def safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)




def _append_columns(df: pd.DataFrame, columns: Dict[str, object]) -> pd.DataFrame:
    """Append/replace one or more columns at once to reduce pandas fragmentation warnings."""
    add = {}
    for k, v in columns.items():
        if isinstance(v, pd.DataFrame):
            if v.shape[1] < 1:
                add[k] = np.nan
            else:
                add[k] = v.iloc[:, 0].reindex(df.index).to_numpy()
        elif isinstance(v, pd.Series):
            add[k] = v.reindex(df.index).to_numpy()
        elif isinstance(v, np.ndarray):
            arr = np.asarray(v)
            if arr.ndim > 1:
                arr = arr[:, 0]
            add[k] = arr
        elif isinstance(v, list):
            arr = np.asarray(v)
            if arr.ndim > 1:
                arr = arr[:, 0]
            add[k] = arr
        else:
            add[k] = pd.Series(v, index=df.index).to_numpy()
    # Replace existing columns instead of creating duplicate names.
    base = df.drop(columns=[c for c in add.keys() if c in df.columns], errors="ignore")
    return pd.concat([base, pd.DataFrame(add, index=df.index)], axis=1).copy()

def compute_deflated_sr(
    returns: pd.Series,
    n_comparisons: int = 20,
    skew_r: Optional[float] = None,
    kurt_r: Optional[float] = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey et al. 2014).

    Adjusts the observed Sharpe Ratio for:
    1. Non-normality of returns (skewness, excess kurtosis).
    2. Multiple testing bias from iterative strategy development.

    DSR = Φ( (SR - E[SR_max]) / σ_SR )
    where E[SR_max] accounts for the expected best Sharpe from n_comparisons
    independent strategies.

    DSR < 0.5 → result is likely overfitted.
    DSR > 0.9 → strong evidence of genuine alpha.

    Ref: Bailey et al. (2014) "The Probability of Backtest Overfitting",
         Journal of Computational Finance.
         Lopez de Prado (2018) "Advances in Financial Machine Learning", Ch.7.
    """
    from scipy.stats import norm
    r = returns.fillna(0.0).astype(float)
    n = len(r)
    if n < 30:
        return float("nan")
    mean_ann = float(r.mean() * 252.0)
    vol_ann  = float(r.std(ddof=0) * np.sqrt(252.0))
    sr = mean_ann / vol_ann if vol_ann > 1e-9 else 0.0

    gamma1 = float(r.skew())    if skew_r is None else float(skew_r)
    gamma2 = float(r.kurt())    if kurt_r is None else float(kurt_r)  # excess kurtosis

    # Variance of SR estimate: (1 - gamma1*SR + (gamma2+1)/4 * SR^2) / T
    sr_var = max((1.0 - gamma1 * sr + (gamma2 + 1.0) / 4.0 * sr ** 2) / max(n, 1), 1e-12)
    sr_std = float(np.sqrt(sr_var))

    # Expected max SR under n_comparisons (Euler-Mascheroni correction)
    euler_gamma = 0.5772156649
    if n_comparisons > 1:
        z1 = norm.ppf(1.0 - 1.0 / n_comparisons)
        z2 = norm.ppf(1.0 - 1.0 / (n_comparisons * np.e))
        e_max_sr = sr_std * ((1.0 - euler_gamma) * z1 + euler_gamma * z2)
    else:
        e_max_sr = 0.0

    dsr = float(norm.cdf((sr - e_max_sr) / max(sr_std, 1e-9)))
    return round(dsr, 4)


def drawdown_from_equity(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def performance_metrics(returns: pd.Series, initial_capital: float) -> Dict[str, float]:
    r = returns.fillna(0.0).astype(float)
    equity = initial_capital * (1.0 + r).cumprod()
    n = len(r)
    years = max(n / 252.0, 1e-9)
    final_capital = float(equity.iloc[-1]) if n else initial_capital
    cagr = (final_capital / initial_capital) ** (1.0 / years) - 1.0 if final_capital > 0 else -1.0
    dd = drawdown_from_equity(equity)
    mdd = float(dd.min()) if n else 0.0
    vol = float(r.std(ddof=0) * math.sqrt(252.0)) if n else 0.0
    mean = float(r.mean() * 252.0) if n else 0.0
    sharpe = mean / vol if vol > 1e-12 else np.nan
    neg = r[r < 0]
    downside = float(neg.std(ddof=0) * math.sqrt(252.0)) if len(neg) else 0.0
    sortino = mean / downside if downside > 1e-12 else np.nan
    calmar = cagr / abs(mdd) if abs(mdd) > 1e-12 else np.nan
    return {
        "final_capital": final_capital,
        "cagr": float(cagr),
        "mdd": float(mdd),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "calmar": float(calmar) if np.isfinite(calmar) else np.nan,
    }


# ============================================================
# Adaptive controls
# ============================================================

def asset_class_rank_weights(asset_class: str, windows: Tuple[int, ...]) -> Dict[int, float]:
    # If a window is not present, weights are re-normalized later.
    if asset_class == "broad_index":
        base = {504: 0.20, 756: 0.50, 1008: 0.30}
    elif asset_class == "sector_etf":
        base = {504: 0.50, 756: 0.40, 1008: 0.10}
    elif asset_class == "high_vol_growth":
        base = {504: 0.60, 756: 0.35, 1008: 0.05}
    elif asset_class == "mega_cap":
        base = {504: 0.30, 756: 0.60, 1008: 0.10}
    else:
        base = {504: 0.40, 756: 0.50, 1008: 0.10}
    selected = {w: base.get(w, 0.0) for w in windows}
    s = sum(selected.values())
    if s <= 0:
        return {w: 1.0 / len(windows) for w in windows}
    return {w: v / s for w, v in selected.items()}


def add_multi_window_ranks(df: pd.DataFrame, cfg: AdaptiveControlsConfig, asset_class: str) -> pd.DataFrame:
    df = df.copy()
    raw_hv = safe_series(df, "prob_high_vol_raw", safe_series(df, "prob_high_vol", 0.5))
    raw_down = safe_series(df, "prob_down_risk_raw", safe_series(df, "prob_down_strengthening_score", 0.0))
    raw_up = safe_series(df, "prob_up_strengthening_score_raw", safe_series(df, "prob_up_strengthening_score", 0.0))

    for w in cfg.rank_windows:
        df[f"ph_rank_{w}"] = rolling_rank_past(raw_hv, w, cfg.rank_min_periods)
        df[f"down_rank_{w}"] = rolling_rank_past(raw_down, w, cfg.rank_min_periods)
        df[f"up_rank_{w}"] = rolling_rank_past(raw_up, w, cfg.rank_min_periods)

    # v8.6.42d: compute rolling skill weights; falls back to fixed weights if skills unavailable
    df = rolling_signal_skills(df, window=cfg.branch_perf_window,
                                min_rows=cfg.branch_perf_min_rows,
                                temperature=cfg.branch_softmax_temperature)

    fixed_weights = asset_class_rank_weights(asset_class, cfg.rank_windows)

    def _ensemble(prefix: str, fallback: pd.Series) -> pd.Series:
        acc = pd.Series(0.0, index=df.index, dtype=float)
        denom = pd.Series(0.0, index=df.index, dtype=float)
        for w, fixed_wt in fixed_weights.items():
            col = f"{prefix}_{w}"
            if col not in df.columns:
                continue
            s = df[col].astype(float)
            # Use rolling skill weight if available, else fixed asset-class weight
            skill_col = f"ph_rank_weight_{w}"
            if prefix == "ph_rank" and skill_col in df.columns:
                wt_series = df[skill_col].astype(float)
            else:
                wt_series = pd.Series(fixed_wt, index=df.index, dtype=float)
            mask = s.notna()
            acc.loc[mask] += s.loc[mask] * wt_series.loc[mask]
            denom.loc[mask] += wt_series.loc[mask]
        out = acc / denom.replace(0.0, np.nan)
        exp = expanding_rank_past(fallback, cfg.rank_min_periods)
        out = out.fillna(exp).fillna(fallback.clip(0, 1))
        return out.clip(0.0, 1.0)

    df["ph_rank_adaptive"] = _ensemble("ph_rank", raw_hv)
    df["down_rank_adaptive"] = _ensemble("down_rank", raw_down)
    df["up_rank_adaptive"] = _ensemble("up_rank", raw_up)
    return df


def infer_adaptive_ewma_span(df: pd.DataFrame, cfg: AdaptiveControlsConfig, asset_class: str) -> pd.Series:
    """Infer adaptive EWMA span with volatility-regime awareness.

    v8.6.42b used ph_rank + trend only. v8.6.42c adds realized_vol_60 rolling rank:
    - high volatility regime -> longer span to filter noise
    - low volatility regime -> shorter span for quicker reaction
    - extreme ph_rank remains an emergency shortcut to the minimum span
    """
    ph = safe_series(df, "ph_rank_adaptive", 0.5)
    trend = df.get("mid_trend_state", pd.Series("NEUTRAL", index=df.index)).fillna("NEUTRAL").astype(str).str.upper()
    vol60 = safe_series(df, "realized_vol_60", 0.0)
    vol_rank = rolling_rank_past(vol60, 252, 60).fillna(0.5).clip(0, 1)

    base_map = {"broad_index": 10, "mega_cap": 7, "sector_etf": 5, "high_vol_growth": 5}
    base = base_map.get(asset_class, 7)
    span = pd.Series(base, index=df.index, dtype=float)

    # Volatility adjustment: high vol -> smoother/longer span, low vol -> more responsive/shorter span.
    span = span + (vol_rank - 0.5) * 6.0

    # Existing emergency/context overrides, now layered after vol-regime adjustment.
    span = np.where(ph >= 0.85, cfg.ewma_min_span, span)
    span = np.where((ph >= 0.70) & (ph < 0.85), np.minimum(span, max(cfg.ewma_min_span, 5)), span)
    span = np.where((trend == "BEAR") & (ph >= 0.60), cfg.ewma_min_span, span)
    span = np.where((trend == "BULL") & (ph < 0.35) & (asset_class == "broad_index"), min(cfg.ewma_max_span, 14), span)
    return pd.Series(span, index=df.index).clip(cfg.ewma_min_span, cfg.ewma_max_span)


def add_adaptive_smoothing(df: pd.DataFrame, cfg: AdaptiveControlsConfig, asset_class: str) -> pd.DataFrame:
    df = df.copy()
    df["adaptive_ewma_span"] = infer_adaptive_ewma_span(df, cfg, asset_class)
    if not cfg.use_adaptive_ewma:
        df["prob_high_vol_ctrl"] = safe_series(df, "prob_high_vol", 0.5)
        df["prob_up_strength_ctrl"] = safe_series(df, "prob_up_strengthening_score", 0.0)
        df["prob_down_strength_ctrl"] = safe_series(df, "prob_down_strengthening_score", 0.0)
        return df
    df["prob_high_vol_ctrl"] = adaptive_ewma(
        safe_series(df, "prob_high_vol_raw", safe_series(df, "prob_high_vol", 0.5)),
        df["adaptive_ewma_span"],
        default_span=7,
    ).clip(0, 1)
    df["prob_up_strength_ctrl"] = adaptive_ewma(
        safe_series(df, "prob_up_strengthening_score_raw", safe_series(df, "prob_up_strengthening_score", 0.0)),
        df["adaptive_ewma_span"],
        default_span=7,
    ).clip(0, 1)
    df["prob_down_strength_ctrl"] = adaptive_ewma(
        safe_series(df, "prob_down_strengthening_score_raw", safe_series(df, "prob_down_strengthening_score", 0.0)),
        df["adaptive_ewma_span"],
        default_span=7,
    ).clip(0, 1)
    return df


def add_proxy_down_components(df: pd.DataFrame, cfg: AdaptiveControlsConfig) -> pd.DataFrame:
    """Create down-risk component proxies from columns available in predictions.

    The original branch-level model probabilities are not always exported. This proxy layer avoids pretending that
    fixed branch weights are optimal while still staying non-leaky: component ranks use only past observations.
    """
    df = df.copy()
    ret60 = safe_series(df, "return_60d", 0.0)
    ret120 = safe_series(df, "return_120d", 0.0)
    dd60 = safe_series(df, "drawdown_60", 0.0)
    vol60 = safe_series(df, "realized_vol_60", 0.0)
    down_strength = safe_series(df, "prob_down_strength_ctrl", safe_series(df, "prob_down_strengthening_score", 0.0))
    ph = safe_series(df, "ph_rank_adaptive", 0.5)

    # Higher = riskier. Convert trend weakness and drawdown to ranks.
    trend_weak_raw = (-(ret60.fillna(0) * 0.65 + ret120.fillna(0) * 0.35)).astype(float)
    drawdown_raw = (-dd60).clip(lower=0.0)
    vol_raw = vol60.fillna(0.0)

    df["down_component_price_trend"] = rolling_rank_past(trend_weak_raw, 756, cfg.rank_min_periods).fillna(_clip01(trend_weak_raw.rank(pct=True))).clip(0, 1)
    df["down_component_drawdown"] = rolling_rank_past(drawdown_raw, 756, cfg.rank_min_periods).fillna(_clip01(drawdown_raw.rank(pct=True))).clip(0, 1)
    df["down_component_volatility"] = rolling_rank_past(vol_raw, 756, cfg.rank_min_periods).fillna(ph).clip(0, 1)
    df["down_component_strength"] = down_strength.clip(0, 1)
    return df


def _component_skill(pred: pd.Series, target: pd.Series) -> float:
    p = pred.astype(float)
    y = target.astype(float)
    mask = p.notna() & y.notna()
    if mask.sum() < 30 or y[mask].nunique() < 2:
        return 0.0
    p = p[mask]
    y = y[mask]
    # Lightweight rank skill: mean(pred | event) - mean(pred | non-event), penalized by Brier.
    # Gneiting & Raftery (2007): Brier Score is strictly proper — skill > 0 means genuine predictive power.
    event_mean = float(p[y == 1].mean()) if (y == 1).any() else 0.0
    nonevent_mean = float(p[y == 0].mean()) if (y == 0).any() else 0.0
    brier = float(((p - y) ** 2).mean())
    base = float(y.mean())
    base_brier = float(((base - y) ** 2).mean()) if len(y) else 0.25
    brier_skill = (base_brier - brier) / max(base_brier, 1e-9)
    return float(max(0.0, event_mean - nonevent_mean) + max(0.0, brier_skill) * 0.25)


def rolling_signal_skills(
    df: pd.DataFrame,
    window: int = 756,
    min_rows: int = 120,
    temperature: float = 0.12,
) -> pd.DataFrame:
    """Compute rolling Brier-skill for each ph_rank window.

    Lopez de Prado (2018) Ch.14: ensemble weights should reflect each member's
    OOS performance. This replaces the fixed asset_class_rank_weights with
    data-driven skill weights, recomputed at each timestep using only past obs.

    Returns df with columns:
        ph_skill_{w}          : rolling composite skill for window w
        ph_rank_weight_{w}    : softmax-normalized weight for window w
    """
    from collections import deque

    windows = [c.replace("ph_rank_", "") for c in df.columns if c.startswith("ph_rank_") and c != "ph_rank_adaptive"]
    int_windows = []
    for w in windows:
        try:
            int_windows.append(int(w))
        except ValueError:
            pass
    if not int_windows:
        return df

    # Build actual_hv binary target: 1 if next-period realized as high vol
    # Proxy: actual_risk column or forward ph_rank spike
    actual_hv = None
    if "actual_risk" in df.columns:
        actual_hv = df["actual_risk"].astype(str).str.contains("고변동|high", case=False, na=False).astype(float)
    elif "prob_high_vol" in df.columns:
        # Proxy: future-realized high vol ≈ ph_rank > 0.70 in next 5 days
        # Use shift(-1) would leak; instead use realized vol crossing if available
        # Safe fallback: use a hard threshold on prob_high_vol (no future leakage)
        actual_hv = (df["prob_high_vol"].astype(float) > 0.65).astype(float)

    if actual_hv is None:
        return df

    # ph_preds_hist[w]: stores rolling window of past ph_rank_w values (predictions)
    # used as `pred` argument to _component_skill. Name "skills_hist" was misleading.
    ph_preds_hist: dict = {w: deque(maxlen=window) for w in int_windows}
    actual_hist: deque = deque(maxlen=window)

    skill_rows: dict = {w: [] for w in int_windows}
    weight_rows: dict = {w: [] for w in int_windows}

    ph_vals = {w: df[f"ph_rank_{w}"].astype(float).to_numpy() for w in int_windows}
    actual_vals = actual_hv.to_numpy()

    # Performance: recompute skill every `stride` rows (skill changes slowly).
    # Stride of 21 (~monthly) reduces O(N²) to O(N*W/stride).
    stride = 21
    last_weights = {w: 1.0 / len(int_windows) for w in int_windows}

    for i in range(len(df)):
        if i % stride == 0 or i == len(df) - 1:
            n = len(actual_hist)
            row_skills = {}
            for w in int_windows:
                if n >= min_rows:
                    ph_hist_arr = np.array(list(ph_preds_hist[w]), dtype=float)
                    act_arr = np.array(list(actual_hist), dtype=float)
                    row_skills[w] = _component_skill(pd.Series(ph_hist_arr), pd.Series(act_arr))
                else:
                    row_skills[w] = 0.0
            vals_arr = np.array([row_skills[w] for w in int_windows], dtype=float)
            if vals_arr.sum() <= 0 or n < min_rows:
                last_weights = {w: 1.0 / len(int_windows) for w in int_windows}
            else:
                exp_v = np.exp(vals_arr / max(temperature, 1e-9))
                exp_v = exp_v / exp_v.sum()
                last_weights = {w: float(exp_v[j]) for j, w in enumerate(int_windows)}
            # Store skill scores for this stride block
            _block_skills = row_skills.copy()

        for w in int_windows:
            skill_rows[w].append(_block_skills.get(w, 0.0) if i % stride == 0 else skill_rows[w][-1] if skill_rows[w] else 0.0)
            weight_rows[w].append(last_weights[w])

        # Update history with current observation (non-leaky)
        for w in int_windows:
            v = ph_vals[w][i]
            if np.isfinite(v):
                ph_preds_hist[w].append(v)
        av = actual_vals[i]
        if np.isfinite(av):
            actual_hist.append(av)

    add = {}
    for w in int_windows:
        add[f"ph_skill_{w}"] = skill_rows[w]
        add[f"ph_rank_weight_{w}"] = weight_rows[w]
    return _append_columns(df, add)


def add_adaptive_down_weights(df: pd.DataFrame, cfg: AdaptiveControlsConfig) -> pd.DataFrame:
    """Adaptive down-component weights driven by rolling component skill.

    v8.6.42e: bonus sizes for BEAR/high_ph/strong_ds conditions are now
    calibrated by rolling _component_skill() for each proxy component,
    not hardcoded (+0.10, +0.08, +0.06 etc.).

    Design:
      1. Compute rolling skill for each proxy (price_trend, drawdown, vol, strength).
      2. Skill-to-bonus: bonus_i = max_bonus * softmax_skill_i, where max_bonus
         is fixed at 0.15 per component (prevents any single component from
         dominating regardless of skill; Maillard et al. 2010 Risk Parity principle).
      3. Base weights (0.35 / 0.20 / 0.25 / 0.20) remain as prior; skill adjusts
         within those bounds. When skill is unavailable (< min_rows), falls back
         to 42c rule-based adjustments.
    """
    df = df.copy()
    trend = df.get("mid_trend_state", pd.Series("NEUTRAL", index=df.index)).fillna("NEUTRAL").astype(str).str.upper()
    ph = safe_series(df, "ph_rank_adaptive", 0.5)
    ds = safe_series(df, "down_component_strength", 0.0)

    # Attempt skill-driven bonus computation
    components = {
        "price_trend": ("down_component_price_trend", 0.35),
        "drawdown":    ("down_component_drawdown",    0.20),
        "volatility":  ("down_component_volatility",  0.25),
        "strength":    ("down_component_strength",    0.20),
    }
    MAX_BONUS = 0.15
    skill_available = False

    # Build a leak-free proxy target for down-component skill evaluation.
    # Option C: realized_vol_60 > rolling 75th-percentile vol → "high-vol state".
    # This measures whether each component predicts imminent high-vol conditions —
    # a temporally consistent target (no shift(-n) look-ahead).
    proxy_target = None
    if "realized_vol_60" in df.columns:
        vol60 = df["realized_vol_60"].astype(float)
        vol_q75 = vol60.rolling(252, min_periods=60).quantile(0.75).shift(1)  # past only
        proxy_target = (vol60.shift(1) > vol_q75).astype(float)  # shift(1) = past observation

    comp_skills = {}
    if proxy_target is not None and proxy_target.notna().sum() >= int(cfg.branch_perf_min_rows):
        n_total = len(df)
        win = int(cfg.branch_perf_window)
        min_r = int(cfg.branch_perf_min_rows)
        # Performance: compute skill every `stride` rows instead of every row.
        # Skill changes slowly — recomputing every 21 rows (~monthly) is sufficient.
        stride = 21
        last_skill = {name: 0.0 for name in components}
        skills_series = {name: np.zeros(n_total, dtype=float) for name in components}
        for i in range(n_total):
            if i % stride == 0 or i == n_total - 1:
                start = max(0, i - win)
                tgt_slice = proxy_target.iloc[start:i]
                if tgt_slice.notna().sum() >= min_r and tgt_slice.nunique() >= 2:
                    for name, (col, _) in components.items():
                        if col in df.columns:
                            pred_slice = df[col].iloc[start:i]
                            last_skill[name] = _component_skill(pred_slice, tgt_slice)
            for name in components:
                skills_series[name][i] = last_skill[name]
        comp_skills = {name: pd.Series(v, index=df.index) for name, v in skills_series.items()}
        skill_available = True

    bear = trend == "BEAR"
    bull = trend == "BULL"
    high_ph = ph >= 0.70
    extreme_ph = ph >= 0.85
    strong_ds = ds >= 0.60

    w_trend = pd.Series(0.35, index=df.index, dtype=float)
    w_draw  = pd.Series(0.20, index=df.index, dtype=float)
    w_vol   = pd.Series(0.25, index=df.index, dtype=float)
    w_str   = pd.Series(0.20, index=df.index, dtype=float)

    if skill_available:
        # Softmax over component skills to distribute MAX_BONUS
        sk = np.stack([
            comp_skills["price_trend"].to_numpy(),
            comp_skills["drawdown"].to_numpy(),
            comp_skills["volatility"].to_numpy(),
            comp_skills["strength"].to_numpy(),
        ], axis=1)  # shape (N, 4)
        sk = np.maximum(sk, 0.0)
        sk_sum = sk.sum(axis=1, keepdims=True)
        sk_norm = np.where(sk_sum > 1e-9, sk / sk_sum, 0.25)  # equal if no skill
        # bonus proportional to relative skill, capped at MAX_BONUS per component
        bonus = sk_norm * MAX_BONUS
        w_trend = w_trend + pd.Series(bonus[:, 0], index=df.index)
        w_draw  = w_draw  + pd.Series(bonus[:, 1], index=df.index)
        w_vol   = w_vol   + pd.Series(bonus[:, 2], index=df.index)
        w_str   = w_str   + pd.Series(bonus[:, 3], index=df.index)
    else:
        # Fallback: rule-based adjustments (42c behaviour)
        w_trend = w_trend + np.where(bear, 0.10, 0.0) - np.where(bull & ~high_ph, 0.05, 0.0)
        w_draw  = w_draw  + np.where(bear, 0.08, 0.0)
        w_vol   = w_vol   + np.where(high_ph, 0.08, 0.0) + np.where(extreme_ph, 0.06, 0.0)
        w_str   = w_str   + np.where(strong_ds, 0.10, 0.0)

    mat = np.vstack([w_trend, w_draw, w_vol, w_str]).T
    mat = np.maximum(mat, float(cfg.branch_weight_floor))
    mat = mat / mat.sum(axis=1, keepdims=True)

    adaptive_down_risk = (
        mat[:, 0] * safe_series(df, "down_component_price_trend", 0.5)
        + mat[:, 1] * safe_series(df, "down_component_drawdown",    0.5)
        + mat[:, 2] * safe_series(df, "down_component_volatility",  0.5)
        + mat[:, 3] * safe_series(df, "down_component_strength",    0.0)
    ).clip(0, 1)

    df = _append_columns(df, {
        "down_weight_price_trend": mat[:, 0],
        "down_weight_drawdown":    mat[:, 1],
        "down_weight_volatility":  mat[:, 2],
        "down_weight_strength":    mat[:, 3],
        "adaptive_down_risk":      adaptive_down_risk,
        "down_skill_available":    skill_available,
    })
    return df


# adaptive_risk_weights() was removed in v8.6.42d refactor.
# The weight logic is now fully vectorized inside add_adaptive_overall_risk()
# using np.where arrays. Keeping a standalone function that is never called
# is dead code and a maintenance hazard.


def add_adaptive_overall_risk(df: pd.DataFrame, cfg: AdaptiveControlsConfig, asset_class: str) -> pd.DataFrame:
    """Vectorized adaptive risk score.

    v8.6.42c bug (L531): denom excluded w_up, so the w_up subtraction operated on a different
    scale than the weighted average numerator. Fixed here by including w_up in the normalization.

    Formula:
        risk = (w_hv*hv + w_down*down + w_ds*ds - w_up*up) / (w_hv+w_down+w_ds+w_up)
    All four components are now on the same scale. w_up reduces risk when up_strength is high,
    but proportionally to its weight relative to the total.

    The ad-hoc -0.06 correction (L535) is removed: it double-counted w_up and was not
    calibrated to any ticker's actual data. Instead, adaptive_risk_weights gives sector/growth
    BULL a higher w_up already.
    """
    df = df.copy()
    hv = safe_series(df, "ph_rank_adaptive", 0.5).to_numpy()
    down = safe_series(df, "adaptive_down_risk", safe_series(df, "down_rank_adaptive", 0.5)).to_numpy()
    ds = safe_series(df, "prob_down_strength_ctrl", safe_series(df, "prob_down_strengthening_score", 0.0)).to_numpy()
    up = safe_series(df, "prob_up_strength_ctrl", safe_series(df, "prob_up_strengthening_score", 0.0)).to_numpy()
    trend = df.get("mid_trend_state", pd.Series("NEUTRAL", index=df.index)).fillna("NEUTRAL").astype(str).str.upper().to_numpy()

    # Vectorized: build weight arrays per row
    n = len(df)
    w_hv_arr  = np.full(n, 0.40, dtype=float)
    w_dn_arr  = np.full(n, 0.30, dtype=float)
    w_ds_arr  = np.full(n, 0.15, dtype=float)
    w_up_arr  = np.full(n, 0.10, dtype=float)

    is_bear = trend == "BEAR"
    is_bull = trend == "BULL"
    is_neutral = ~is_bear & ~is_bull

    if asset_class == "broad_index":
        w_hv_arr  = np.where(is_bear, 0.45, np.where(is_bull, 0.40, 0.45))
        w_dn_arr  = np.where(is_bear, 0.35, np.where(is_bull, 0.25, 0.30))
        w_ds_arr  = np.where(is_bear, 0.15, np.where(is_bull, 0.10, 0.15))
        w_up_arr  = np.where(is_bear, 0.05, np.where(is_bull, 0.15, 0.10))
    elif asset_class == "sector_etf":
        w_hv_arr  = np.where(is_bear, 0.40, np.where(is_bull, 0.30, 0.35))
        w_dn_arr  = np.where(is_bear, 0.35, np.where(is_bull, 0.20, 0.30))
        w_ds_arr  = np.where(is_bear, 0.20, np.where(is_bull, 0.15, 0.15))
        w_up_arr  = np.where(is_bear, 0.05, np.where(is_bull, 0.25, 0.15))
    elif asset_class == "high_vol_growth":
        w_hv_arr  = np.where(is_bear, 0.35, np.where(is_bull, 0.25, 0.30))
        w_dn_arr  = np.where(is_bear, 0.35, np.where(is_bull, 0.25, 0.30))
        w_ds_arr  = np.where(is_bear, 0.25, np.where(is_bull, 0.15, 0.20))
        w_up_arr  = np.where(is_bear, 0.05, np.where(is_bull, 0.28, 0.15))
    elif asset_class == "mega_cap":
        w_hv_arr  = np.where(is_bear, 0.35, np.where(is_bull, 0.35, 0.35))
        w_dn_arr  = np.where(is_bear, 0.40, np.where(is_bull, 0.25, 0.35))
        w_ds_arr  = np.where(is_bear, 0.20, np.where(is_bull, 0.15, 0.15))
        w_up_arr  = np.where(is_bear, 0.05, np.where(is_bull, 0.20, 0.10))

    # Fixed normalization: all four weights in denominator
    total = w_hv_arr + w_dn_arr + w_ds_arr + w_up_arr
    total = np.maximum(total, 1e-9)
    scores = (w_hv_arr * hv + w_dn_arr * down + w_ds_arr * ds - w_up_arr * up) / total
    scores = np.clip(scores, 0.0, 1.0)

    df = _append_columns(df, {
        "adaptive_overall_risk": scores,
        "adaptive_pred_risk": np.where(scores >= cfg.high_vol_rank_threshold, "위험", "정상"),
    })
    return df


def ph_bin(x: float) -> str:
    if not np.isfinite(x):
        return "unknown"
    if x < 0.35:
        return "00_35"
    if x < 0.50:
        return "35_50"
    if x < 0.65:
        return "50_65"
    if x < 0.75:
        return "65_75"
    if x < 0.85:
        return "75_85"
    if x < 0.95:
        return "85_95"
    return "95_100"


def build_cross_asset_context_scores(combined: pd.DataFrame, cfg: AdaptiveControlsConfig) -> pd.DataFrame:
    """Leakage-guarded rolling context score across tickers.

    v8.6.42b fixes two issues from the first adaptive-controls implementation:
    1) Global fallback now respects context_global_min_rows.
    2) Same-date cross-sectional leakage is blocked. All context statistics use rows with Date < current Date only.

    The policy table is still inferred, not manual, but positive adjustments are deliberately conservative and
    asymmetric: risk cuts can be larger than risk-on boosts.
    """
    from collections import defaultdict, deque

    work = combined.sort_values(["Date", "ticker"]).reset_index(drop=True).copy()
    work["ph_bin"] = work["ph_rank_adaptive"].apply(ph_bin)
    work["context_key"] = (
        work["asset_class"].astype(str)
        + "|" + work["mid_trend_state"].astype(str)
        + "|" + work["ph_bin"].astype(str)
    )

    maxlen = int(cfg.context_window) if int(cfg.context_window) > 0 else None
    ctx_hist = defaultdict(lambda: deque(maxlen=maxlen))
    ac_hist = defaultdict(lambda: deque(maxlen=maxlen))
    global_hist = deque(maxlen=maxlen)

    rows_out = []
    scope_out = []
    mean_out = []
    down_out = []
    score_out = []
    adjust_out = []

    def _stats(buf):
        n = len(buf)
        if n <= 0:
            return 0, 0.0, 0.5, 0.0, 0.0   # confidence=0 when no data (was 1.0 — misleading)
        vals = np.array(list(buf), dtype=float)
        m    = float(np.mean(vals))
        d    = float(np.mean(vals < 0))
        tail = float(np.percentile(vals, 10)) if n >= 10 else m
        std  = float(np.std(vals, ddof=1)) if n > 1 else 1e-9
        t_stat = (m / (std / max(np.sqrt(n), 1e-9))) if std > 1e-9 else 0.0
        confidence = float(np.clip(abs(t_stat) / 2.0, 0.0, 1.0))
        return n, m, d, tail, confidence

    def _score_to_adjust(score: float, asset_class: str, confidence: float) -> float:
        pos_th  = float(cfg.context_positive_score_threshold)
        neg_th  = float(cfg.context_negative_score_threshold)
        pos_max = float(cfg.context_adjust_max_positive)
        neg_max = float(cfg.context_adjust_max_negative)
        if score >= pos_th:
            denom = max(1e-9, 1.0 - pos_th)
            adj = min(1.0, (score - pos_th) / denom) * pos_max
        elif score <= neg_th:
            denom = max(1e-9, 1.0 + neg_th)
            adj = -min(1.0, (neg_th - score) / denom) * neg_max
        else:
            adj = 0.0
        # t-stat confidence shrinkage: weak signal → adj → 0
        # Ref: Bailey et al. (2014), Gneiting & Raftery (2007)
        adj *= confidence
        if asset_class == "broad_index" and not bool(cfg.broad_index_allow_positive_context_adjust):
            adj = min(adj, 0.0)
        return float(adj)

    tail10_out = []
    confidence_out = []

    # Process one Date at a time. Update histories only after all rows for that Date are scored.
    for _, day_df in work.groupby("Date", sort=False):
        pending_updates = []
        for idx, row in day_df.iterrows():
            ac  = str(row.get("asset_class", "unknown"))
            key = str(row.get("context_key", ""))
            n, mean, down_freq, tail10, conf = _stats(ctx_hist[key])
            scope = "asset_class_trend_phbin"
            if n < int(cfg.context_min_rows):
                n_ac, mean_ac, down_ac, tail_ac, conf_ac = _stats(ac_hist[ac])
                if n_ac >= int(cfg.context_min_rows):
                    n, mean, down_freq, tail10, conf = n_ac, mean_ac, down_ac, tail_ac, conf_ac
                    scope = "asset_class"
                else:
                    n_g, mean_g, down_g, tail_g, conf_g = _stats(global_hist)
                    if n_g >= int(cfg.context_global_min_rows):
                        n, mean, down_freq, tail10, conf = n_g, mean_g, down_g, tail_g, conf_g
                        scope = "global"
                    elif n_ac > 0:
                        n, mean, down_freq, tail10, conf = n_ac, mean_ac, down_ac, tail_ac, conf_ac
                        scope = "asset_class_partial"
                    else:
                        n, mean, down_freq, tail10, conf = 0, 0.0, 0.5, 0.0, 1.0
                        scope = "insufficient_history"

            base_down = ASSET_CLASS_BASE_DOWN_RATE.get(ac, 0.50)
            score = (float(cfg.context_return_scale) * float(mean)
                     - float(cfg.context_down_penalty) * (float(down_freq) - float(base_down))
                     - float(cfg.context_tail_penalty) * min(0.0, float(tail10)))
            adj = _score_to_adjust(score, ac, conf) if n > 0 else 0.0

            rows_out.append(int(n))
            scope_out.append(scope)
            mean_out.append(float(mean))
            down_out.append(float(down_freq))
            score_out.append(float(score))
            adjust_out.append(float(adj))
            tail10_out.append(float(tail10))
            confidence_out.append(float(conf))

            # Excess return for context history: (stock_return - cash_return)
            # Ref: Ang & Bekaert (2004) — regime-conditional excess return isolates alpha
            sr = row.get("stock_next_return", 0.0)
            cr = row.get("cash_next_return",  0.0)
            sr = float(sr) if np.isfinite(sr) else 0.0
            cr = float(cr) if np.isfinite(cr) else 0.0
            excess_ret = sr - cr
            pending_updates.append((key, ac, excess_ret))

        for key, ac, ret_val in pending_updates:
            ctx_hist[key].append(ret_val)
            ac_hist[ac].append(ret_val)
            global_hist.append(ret_val)

    work["context_rows"]        = rows_out
    work["context_scope"]       = scope_out
    work["context_mean_return"] = mean_out
    work["context_down_freq"]   = down_out
    work["context_tail10"]      = tail10_out      # was np.nan placeholder — now computed
    work["context_confidence"]  = confidence_out  # new: t-stat based shrinkage factor
    work["context_score"]       = score_out
    work["context_adjust"]      = adjust_out
    return work

def stock_from_risk_score(risk: float, asset_class: str = "broad_index",
                          risk_midpoint: float = 0.50) -> float:
    """Smooth risk-to-stock mapping with calibrated midpoint.

    v8.6.42c used fixed midpoint=0.55. QQQ's actual risk_score median ≈ 0.35–0.45,
    so a fixed 0.55 caused systematic defensive bias. v8.6.42d passes the rolling
    252-day median of adaptive_overall_risk as risk_midpoint, making the curve
    self-calibrating to each asset's typical risk regime.

    Kelly (1956) / MacLean & Thorp (2010): optimal bet size shifts with the distribution
    of expected returns. Anchoring the sigmoid midpoint to the rolling median is a
    lightweight fractional-Kelly-style calibration.

    max_s corrections vs 42c:
      sector_etf  0.90 → 0.88 (SOXX offensive 47% was caused by high ceiling)
      mega_cap    0.88 → 0.86 (consistent with broad_index ceiling)
    """
    max_s = {
        "broad_index":    0.86,
        "sector_etf":     0.88,   # was 0.90 in 42c → SOXX offensive 47% fix
        "high_vol_growth":0.92,
        "mega_cap":       0.86,   # was 0.88 in 42c
        "single_stock":   0.86,
    }.get(str(asset_class), 0.86)
    min_s = 0.28
    k = 6.0
    midpoint = float(np.clip(risk_midpoint, 0.30, 0.65))  # guard rails on the calibrated midpoint
    risk = float(np.clip(risk, 0.0, 1.0))
    stock = min_s + (max_s - min_s) / (1.0 + np.exp(k * (risk - midpoint)))
    return float(np.clip(stock, min_s, max_s))


def bond_cash_from_stock(stock: float) -> Tuple[float, float]:
    defensive = max(0.0, 1.0 - stock)
    return defensive * 0.65, defensive * 0.35


def add_signal_weights(df: pd.DataFrame, cfg: AdaptiveControlsConfig) -> pd.DataFrame:
    """Vectorized signal-to-allocation mapping.

    v8.6.42d changes vs 42c:
    1. Vectorized (no iterrows) — ~20× faster.
    2. broad_index_min_stock floor (fix: 42c QQQ avg_stock 63.8% structural collapse).
    3. stock_from_risk_score receives rolling 252d median risk as midpoint
       (Kelly calibration — avoids systematic defensive bias from fixed 0.55).
    4. sector_etf / growth guardrails use ph_rank_adaptive threshold instead of
       raw up_strength only, reducing false-positive floor activations.
    """
    df = df.copy()
    ac   = df["asset_class"].astype(str)
    trend = df.get("mid_trend_state", pd.Series("NEUTRAL", index=df.index)).fillna("NEUTRAL").astype(str).str.upper()
    risk = df["adaptive_overall_risk"].astype(float)
    ph   = df["ph_rank_adaptive"].astype(float)
    up   = df.get("prob_up_strength_ctrl", df.get("prob_up_strengthening_score", pd.Series(0.0, index=df.index))).astype(float)
    down = df.get("prob_down_strength_ctrl", df.get("prob_down_strengthening_score", pd.Series(0.0, index=df.index))).astype(float)
    ctx  = df.get("context_adjust", pd.Series(0.0, index=df.index)).astype(float)

    # Rolling median risk as sigmoid midpoint (Kelly calibration, window=252)
    risk_median = risk.rolling(252, min_periods=60).median().fillna(0.50)

    # Base stock from smooth sigmoid (vectorized via apply — one call per row is cheap)
    stock = pd.Series(
        [stock_from_risk_score(float(r), str(a), float(m))
         for r, a, m in zip(risk, ac, risk_median)],
        index=df.index, dtype=float
    )

    # Context adjustment
    stock = stock + ctx

    # Upside overlay (non-broad_index only by default)
    overlay_allowed = pd.Series(bool(cfg.enable_upside_overlay), index=df.index)
    if not bool(cfg.broad_index_enable_upside_overlay):
        overlay_allowed = overlay_allowed & (ac != "broad_index")

    tier1 = overlay_allowed & (up >= cfg.up_tier1) & (risk < 0.78)
    tier2 = tier1 & (up >= cfg.up_tier2) & (risk < 0.72)
    tier3 = tier2 & (up >= cfg.up_tier3) & (risk < 0.68)
    full  = tier1 & (up >= cfg.full_stock_up_score) & (
        ph <= pd.Series(
            [cfg.full_stock_high_vol_max_by_ac.get(str(a), 0.58) for a in ac],
            index=df.index, dtype=float
        )
    ) & trend.isin(["BULL","NEUTRAL"])

    tier_target = pd.Series(0.0, index=df.index)
    tier_target = np.where(tier1, 0.82, tier_target)
    tier_target = np.where(tier2, 0.88, tier_target)
    tier_target = np.where(tier3, 0.96, tier_target)
    tier_target = np.where(full,  1.00, tier_target)

    # Down-strength bear cap on overlay
    ds_bear_cap = (down >= 0.60) & (trend == "BEAR")
    tier_target = np.where(ds_bear_cap, np.minimum(tier_target, 0.62), tier_target)

    stock = np.where(tier_target > stock, tier_target, stock)
    stock = pd.Series(stock, index=df.index, dtype=float)

    # ── asset-class guardrails ─────────────────────────────────────────────
    is_broad   = ac == "broad_index"
    is_mega    = ac == "mega_cap"
    is_sector  = ac == "sector_etf"
    is_growth  = ac == "high_vol_growth"
    is_unknown = ~(is_broad | is_mega | is_sector | is_growth)

    # broad_index: max cap + NEW minimum floor
    stock = np.where(is_broad, np.clip(stock, cfg.broad_index_min_stock, cfg.broad_index_max_stock), stock)

    # mega_cap: BEAR risk caps
    mega_bear_cut = is_mega & (trend == "BEAR") & (risk >= 0.55)
    mega_ext_cut  = is_mega & (trend == "BEAR") & (risk >= 0.75)
    stock = np.where(mega_bear_cut, np.minimum(stock, cfg.mega_cap_bear_cap), stock)
    stock = np.where(mega_ext_cut,  np.minimum(stock, cfg.mega_cap_extreme_bear_cap), stock)

    # sector_etf: BULL floor (condition tightened vs 42c — require ph_rank < 0.75 to avoid
    #   activating floor when vol is already elevated, which caused SOXX MDD regression)
    sector_bull_fl = is_sector & (trend == "BULL") & (up >= 0.38) & (risk < 0.82) & (ph < 0.75)
    stock = np.where(sector_bull_fl, np.maximum(stock, cfg.sector_bull_floor), stock)

    # high_vol_growth: BULL floor (tighten condition same way)
    growth_bull_fl = is_growth & (trend == "BULL") & (up >= 0.42) & (risk < 0.80) & (ph < 0.78)
    stock = np.where(growth_bull_fl, np.maximum(stock, cfg.growth_bull_floor), stock)

    # unknown: cap
    stock = np.where(is_unknown, np.minimum(stock, cfg.unknown_max_stock), stock)

    stock = pd.Series(np.clip(stock, 0.0, 1.0), index=df.index, dtype=float)
    bc = [bond_cash_from_stock(float(s)) for s in stock]
    df = _append_columns(df, {
        "adaptive_signal_stock_weight": stock.to_numpy(),
        "adaptive_signal_bond_weight":  np.array([x[0] for x in bc]),
        "adaptive_signal_cash_weight":  np.array([x[1] for x in bc]),
        "adaptive_policy_note":         [f"risk={r:.3f};ctx={c:+.3f}" for r, c in zip(risk, ctx)],
    })
    return df


def simulate_execution(df: pd.DataFrame, cfg: AdaptiveControlsConfig) -> pd.DataFrame:
    """Portfolio execution simulation.

    v8.6.42d fix: max_weight_change_per_rebalance normalization bug.

    42c bug (L790-799):
        change = clip(signal - prev, -cap, +cap)
        new_w  = prev + change          # sum may exceed 1.0
        new_w  = clip(new_w, 0, 1)
        new_w  = new_w / new_w.sum()    # renorm can push stock change past cap
        # Verified: cap=0.20 produced actual change of -0.296

    Fix: clamp stock weight directly, then recompute bond/cash proportionally.
    This guarantees stock change ≤ cap regardless of normalization.
    bond_cash_from_stock() preserves the 65/35 defensive split.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    exec_stock, exec_bond, exec_cash = [], [], []
    turnovers, costs, gross_r, net_r, rebalanced = [], [], [], [], []
    prev = np.array([
        float(df.loc[0, "adaptive_signal_stock_weight"]),
        float(df.loc[0, "adaptive_signal_bond_weight"]),
        float(df.loc[0, "adaptive_signal_cash_weight"]),
    ], dtype=float)
    last_rebalance_i = -(10 ** 9)
    for i, row in df.iterrows():
        signal = np.array([
            float(row["adaptive_signal_stock_weight"]),
            float(row["adaptive_signal_bond_weight"]),
            float(row["adaptive_signal_cash_weight"]),
        ], dtype=float)
        due = (i - last_rebalance_i) >= int(cfg.rebalance_every_n_days)
        distance = float(np.abs(signal - prev).sum())
        force = bool(row.get("adaptive_force_rebalance", False))
        do_trade = bool(force or (due and distance >= float(cfg.no_trade_band)))
        if do_trade:
            max_change = float(cfg.max_weight_change_per_rebalance)
            if np.isfinite(max_change) and max_change > 0:
                # Fix: clamp stock weight first, then rebuild bond/cash.
                # This ensures |new_stock - prev_stock| ≤ max_change regardless
                # of downstream normalization.
                target_stock = float(signal[0])
                capped_stock = float(np.clip(
                    target_stock,
                    prev[0] - max_change,
                    prev[0] + max_change,
                ))
                capped_stock = float(np.clip(capped_stock, 0.0, 1.0))
                b, c = bond_cash_from_stock(capped_stock)  # 65/35 defensive split
                new_w = np.array([capped_stock, b, c], dtype=float)
            else:
                new_w = signal
            turnover = float(np.abs(new_w - prev).sum())
            prev = new_w
            last_rebalance_i = i
            reb = True
        else:
            turnover = 0.0
            reb = False
        sr = float(row.get("stock_next_return", 0.0)) if np.isfinite(row.get("stock_next_return", 0.0)) else 0.0
        br = float(row.get("bond_next_return", 0.0))  if np.isfinite(row.get("bond_next_return",  0.0)) else 0.0
        cr = float(row.get("cash_next_return", 0.0))  if np.isfinite(row.get("cash_next_return",  0.0)) else 0.0
        gr   = float(prev[0] * sr + prev[1] * br + prev[2] * cr)
        cost = turnover * float(cfg.transaction_cost_rate)
        nr   = gr - cost
        exec_stock.append(float(prev[0])); exec_bond.append(float(prev[1])); exec_cash.append(float(prev[2]))
        turnovers.append(turnover); costs.append(cost); gross_r.append(gr); net_r.append(nr); rebalanced.append(reb)
    df = _append_columns(df, {
        "stock_weight": exec_stock,
        "bond_weight":  exec_bond,
        "cash_weight":  exec_cash,
        "turnover":     turnovers,
        "transaction_cost": costs,
        "strategy_return_gross": gross_r,
        "strategy_return_net":   net_r,
        "rebalanced": rebalanced,
    })
    df = _append_columns(df, {
        "strategy_equity_net":   float(cfg.initial_capital) * (1.0 + df["strategy_return_net"].fillna(0.0)).cumprod(),
        "strategy_equity_gross": float(cfg.initial_capital) * (1.0 + df["strategy_return_gross"].fillna(0.0)).cumprod(),
    })
    return df


def process_ticker(ticker: str, pred_path: Path, cfg: AdaptiveControlsConfig) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    if "Date" not in df.columns:
        raise ValueError(f"{pred_path}: Date column not found")
    df["Date"] = pd.to_datetime(df["Date"])
    # copy() de-fragments wide prediction CSVs before adding more columns.
    df = df.sort_values("Date").reset_index(drop=True).copy()
    ac = infer_asset_class(ticker)
    df = _append_columns(df, {"ticker": ticker.upper(), "asset_class": ac})
    df = add_multi_window_ranks(df, cfg, ac)
    df = add_adaptive_smoothing(df, cfg, ac)
    df = add_proxy_down_components(df, cfg)
    df = add_adaptive_down_weights(df, cfg)
    df = add_adaptive_overall_risk(df, cfg, ac)
    return df


def summarize_ticker(df: pd.DataFrame, cfg: AdaptiveControlsConfig,
                     n_comparisons: int = 25) -> Dict[str, object]:
    perf  = performance_metrics(df["strategy_return_net"],   cfg.initial_capital)
    gross = performance_metrics(df["strategy_return_gross"],  cfg.initial_capital)
    dsr   = compute_deflated_sr(df["strategy_return_net"], n_comparisons=n_comparisons)
    return {
        "ticker":      str(df["ticker"].iloc[0]),
        "asset_class": str(df["asset_class"].iloc[0]),
        **perf,
        "gross_cagr":  gross["cagr"],
        "gross_mdd":   gross["mdd"],
        "gross_sharpe": gross["sharpe"],
        "deflated_sr": dsr,           # Bailey et al. 2014 overfitting diagnostic
        "avg_stock_weight":        float(df["stock_weight"].mean()),
        "avg_signal_stock_weight": float(df["adaptive_signal_stock_weight"].mean()),
        "turnover": float(df["turnover"].sum() / max(len(df) / 252.0, 1e-9)),
        "avg_adaptive_overall_risk": float(df["adaptive_overall_risk"].mean()),
        "avg_ph_rank_adaptive":      float(df["ph_rank_adaptive"].mean()),
        "avg_ewma_span":             float(df["adaptive_ewma_span"].mean()),
        "offensive_activation_rate": float((df["adaptive_signal_stock_weight"] >= 0.82).mean()),
        "full_stock_rate":           float((df["adaptive_signal_stock_weight"] >= 0.98).mean()),
        "avg_context_confidence":    float(df["context_confidence"].mean()) if "context_confidence" in df.columns else float("nan"),
        "pct_context_insufficient":  float((df.get("context_scope", pd.Series("")) == "insufficient_history").mean()),
        "down_skill_available":      bool(df.get("down_skill_available", pd.Series(False)).any()),
    }


def find_prediction_file(input_dir: Path, ticker: str, source_tag: str) -> Optional[Path]:
    t = ticker.lower()
    candidates = [
        input_dir / f"{t}_{source_tag}_predictions.csv",
        input_dir / ticker / f"{t}_{source_tag}_predictions.csv",
        input_dir / t / f"{t}_{source_tag}_predictions.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(input_dir.rglob(f"{t}_*predictions.csv"))
    if source_tag:
        matches = [m for m in matches if source_tag in m.name]
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v8.6.42e all-fixes adaptive controls resim")
    p.add_argument("--input-dir", type=str, default=".")
    p.add_argument("--out-dir", type=str, default="results_v8_6_42e_all_fixes_controls")
    p.add_argument("--asset-list", type=str, default="QQQ,SPY,AAPL,SOXX,NVDA")
    p.add_argument("--source-tag", type=str, default="xgb_recency_weighted_v8_6_41_model_label_fixed")
    p.add_argument("--transaction-cost-rate", type=float, default=0.001)
    p.add_argument("--rank-windows", type=str, default="504,756,1008")
    p.add_argument("--rank-min-periods", type=int, default=252)
    p.add_argument("--context-window", type=int, default=756)
    p.add_argument("--max-weight-change-per-rebalance", type=float, default=0.20)
    p.add_argument("--no-trade-band", type=float, default=0.12)
    p.add_argument("--rebalance-every", type=int, default=5)
    p.add_argument("--disable-adaptive-ewma", action="store_true")
    p.add_argument("--disable-upside-overlay", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AdaptiveControlsConfig(
        transaction_cost_rate=float(args.transaction_cost_rate),
        rank_windows=parse_windows(args.rank_windows),
        rank_min_periods=int(args.rank_min_periods),
        context_window=int(args.context_window),
        max_weight_change_per_rebalance=float(args.max_weight_change_per_rebalance),
        no_trade_band=float(args.no_trade_band),
        rebalance_every_n_days=int(args.rebalance_every),
        use_adaptive_ewma=not bool(args.disable_adaptive_ewma),
        enable_upside_overlay=not bool(args.disable_upside_overlay),
    )
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = [x.strip().upper() for x in str(args.asset_list).split(',') if x.strip()]

    processed = []
    missing = []
    for ticker in tickers:
        pred_path = find_prediction_file(input_dir, ticker, args.source_tag)
        if pred_path is None:
            missing.append(ticker)
            continue
        print(f"[LOAD] {ticker}: {pred_path}")
        processed.append(process_ticker(ticker, pred_path, cfg))
    if missing:
        print(f"[WARN] missing prediction files: {missing}")
    if not processed:
        raise SystemExit("No prediction files found")

    combined = pd.concat(processed, ignore_index=True)
    combined_ctx = build_cross_asset_context_scores(combined, cfg)

    summary_rows = []
    for ticker, g in combined_ctx.groupby("ticker", sort=False):
        td = g.sort_values("Date").reset_index(drop=True)
        td = add_signal_weights(td, cfg)
        td = simulate_execution(td, cfg)
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(ticker)).strip("_") or "asset"
        ticker_dir = out_dir / safe
        ticker_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{safe}_xgb_recency_weighted_v8_6_42_adaptive_controls"
        td.to_csv(ticker_dir / f"{prefix}_predictions_resim.csv", index=False, encoding="utf-8-sig")
        summ = summarize_ticker(td, cfg)
        summary_rows.append(summ)
        with open(ticker_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "version": cfg.version,
                "ticker": ticker,
                "asset_class": str(td["asset_class"].iloc[0]),
                "config": asdict(cfg),
                "performance": summ,
                "control_diagnostics": {
                    "rank_windows": list(cfg.rank_windows),
                    "avg_component_weights": {
                        "price_trend": float(td.get("down_weight_price_trend", pd.Series(dtype=float)).mean()),
                        "drawdown": float(td.get("down_weight_drawdown", pd.Series(dtype=float)).mean()),
                        "volatility": float(td.get("down_weight_volatility", pd.Series(dtype=float)).mean()),
                        "strength": float(td.get("down_weight_strength", pd.Series(dtype=float)).mean()),
                    },
                    "context_scope_counts": td["context_scope"].value_counts(dropna=False).to_dict() if "context_scope" in td.columns else {},
                    "policy_note_counts_top20": td["adaptive_policy_note"].value_counts(dropna=False).head(20).to_dict() if "adaptive_policy_note" in td.columns else {},
                },
            }, f, ensure_ascii=False, indent=2)

    multi = pd.DataFrame(summary_rows)
    multi.to_csv(out_dir / "multi_asset_summary.csv", index=False, encoding="utf-8-sig")
    combined_ctx[["Date", "ticker", "asset_class", "mid_trend_state", "ph_bin",
                  "context_scope", "context_rows", "context_mean_return",
                  "context_down_freq", "context_tail10", "context_confidence",
                  "context_score", "context_adjust"]].to_csv(
        out_dir / "adaptive_context_policy_table.csv", index=False, encoding="utf-8-sig"
    )
    with open(out_dir / "adaptive_controls_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    print("\n[SAVED]")
    print(out_dir / "multi_asset_summary.csv")


if __name__ == "__main__":
    main()
