#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v8.6.42 Adaptive Controls Resimulation

Purpose
-------
Post-prediction allocation layer for xgb_recency_weighted v8.6.41 model_label_fixed outputs.
It replaces several fixed controls with adaptive / rolling / asset-class-aware controls:

7. adaptive overall risk score
8. adaptive down-risk component weighting (proxy components from available prediction columns)
9. adaptive EWMA smoothing span
10. adaptive recency-control profile for signal memory / decay diagnostics (training half-life still requires full retrain)
11. multi-window ph_rank ensemble instead of single 756 window
12. rolling context policy table instead of a fully manual asset_class policy table

This script does not retrain the XGBoost heads. It isolates allocation/control effects so the user can compare
v8.6.41_model_label_fixed predictions under an adaptive controls allocation policy.
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
    context_min_rows: int = 40
    context_global_min_rows: int = 80
    context_return_scale: float = 24.0  # approx. 5d monthly-ish scaling for score only
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
    up_tier1: float = 0.30
    up_tier2: float = 0.38
    up_tier3: float = 0.45
    full_stock_up_score: float = 0.50
    full_stock_high_vol_max: float = 0.58

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

    version: str = "v8.6.42b_guarded_adaptive_controls"


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

    weights = asset_class_rank_weights(asset_class, cfg.rank_windows)
    def _ensemble(prefix: str, fallback: pd.Series) -> pd.Series:
        acc = pd.Series(0.0, index=df.index, dtype=float)
        denom = pd.Series(0.0, index=df.index, dtype=float)
        for w, wt in weights.items():
            col = f"{prefix}_{w}"
            if col in df.columns and wt > 0:
                s = df[col].astype(float)
                mask = s.notna()
                acc.loc[mask] += s.loc[mask] * wt
                denom.loc[mask] += wt
        out = acc / denom.replace(0.0, np.nan)
        # fallback: expanding rank, then raw probability clipped
        exp = expanding_rank_past(fallback, cfg.rank_min_periods)
        out = out.fillna(exp).fillna(fallback.clip(0, 1))
        return out.clip(0.0, 1.0)

    df["ph_rank_adaptive"] = _ensemble("ph_rank", raw_hv)
    df["down_rank_adaptive"] = _ensemble("down_rank", raw_down)
    df["up_rank_adaptive"] = _ensemble("up_rank", raw_up)
    return df


def infer_adaptive_ewma_span(df: pd.DataFrame, cfg: AdaptiveControlsConfig, asset_class: str) -> pd.Series:
    ph = safe_series(df, "ph_rank_adaptive", 0.5)
    trend = df.get("mid_trend_state", pd.Series("NEUTRAL", index=df.index)).fillna("NEUTRAL").astype(str)
    if asset_class == "broad_index":
        base = 10
    elif asset_class == "mega_cap":
        base = 7
    elif asset_class == "sector_etf":
        base = 5
    elif asset_class == "high_vol_growth":
        base = 5
    else:
        base = 7
    span = pd.Series(base, index=df.index, dtype=float)
    span = np.where(ph >= 0.85, cfg.ewma_min_span, span)
    span = np.where((ph >= 0.70) & (ph < 0.85), max(cfg.ewma_min_span, 5), span)
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
    event_mean = float(p[y == 1].mean()) if (y == 1).any() else 0.0
    nonevent_mean = float(p[y == 0].mean()) if (y == 0).any() else 0.0
    brier = float(((p - y) ** 2).mean())
    base = float(y.mean())
    base_brier = float(((base - y) ** 2).mean()) if len(y) else 0.25
    brier_skill = (base_brier - brier) / max(base_brier, 1e-9)
    return float(max(0.0, event_mean - nonevent_mean) + max(0.0, brier_skill) * 0.25)


def add_adaptive_down_weights(df: pd.DataFrame, cfg: AdaptiveControlsConfig) -> pd.DataFrame:
    """Adaptive down-component weights, vectorized.

    This removes the fixed 0.40/0.30/0.20 style weighting. When exported branch probabilities are not
    available, it uses condition-dependent proxy component weights:
    - BEAR/weak trend: price_trend and drawdown weights increase.
    - high ph_rank: volatility weight increases.
    - strong down-strength signal: strength weight increases.
    The weights are not optimized to a ticker's final return.
    """
    df = df.copy()
    trend = df.get("mid_trend_state", pd.Series("NEUTRAL", index=df.index)).fillna("NEUTRAL").astype(str).str.upper()
    ph = safe_series(df, "ph_rank_adaptive", 0.5)
    ds = safe_series(df, "down_component_strength", 0.0)

    # Base weights: price_trend, drawdown, volatility, strength.
    w_trend = pd.Series(0.35, index=df.index, dtype=float)
    w_draw = pd.Series(0.20, index=df.index, dtype=float)
    w_vol = pd.Series(0.25, index=df.index, dtype=float)
    w_str = pd.Series(0.20, index=df.index, dtype=float)

    bear = trend == "BEAR"
    bull = trend == "BULL"
    high_ph = ph >= 0.70
    extreme_ph = ph >= 0.85
    strong_ds = ds >= 0.60

    w_trend = w_trend + np.where(bear, 0.10, 0.0) - np.where(bull & ~high_ph, 0.05, 0.0)
    w_draw = w_draw + np.where(bear, 0.08, 0.0)
    w_vol = w_vol + np.where(high_ph, 0.08, 0.0) + np.where(extreme_ph, 0.06, 0.0)
    w_str = w_str + np.where(strong_ds, 0.10, 0.0)

    mat = np.vstack([w_trend, w_draw, w_vol, w_str]).T
    mat = np.maximum(mat, float(cfg.branch_weight_floor))
    mat = mat / mat.sum(axis=1, keepdims=True)

    df["down_weight_price_trend"] = mat[:, 0]
    df["down_weight_drawdown"] = mat[:, 1]
    df["down_weight_volatility"] = mat[:, 2]
    df["down_weight_strength"] = mat[:, 3]
    df["adaptive_down_risk"] = (
        df["down_weight_price_trend"] * safe_series(df, "down_component_price_trend", 0.5)
        + df["down_weight_drawdown"] * safe_series(df, "down_component_drawdown", 0.5)
        + df["down_weight_volatility"] * safe_series(df, "down_component_volatility", 0.5)
        + df["down_weight_strength"] * safe_series(df, "down_component_strength", 0.0)
    ).clip(0, 1)
    return df

def adaptive_risk_weights(asset_class: str, trend: str, ph: float) -> Tuple[float, float, float, float]:
    """Return weights for hv, down, down_strength, up_strength subtraction."""
    trend = str(trend).upper()
    if asset_class == "broad_index":
        if trend == "BEAR":
            return 0.45, 0.35, 0.15, 0.05
        if trend == "BULL":
            return 0.40, 0.25, 0.10, 0.15
        return 0.45, 0.30, 0.15, 0.10
    if asset_class == "sector_etf":
        if trend == "BULL":
            return 0.30, 0.20, 0.15, 0.25
        if trend == "BEAR":
            return 0.40, 0.35, 0.20, 0.05
        return 0.35, 0.30, 0.15, 0.15
    if asset_class == "high_vol_growth":
        if trend == "BULL":
            return 0.25, 0.25, 0.15, 0.25
        if trend == "BEAR":
            return 0.35, 0.35, 0.25, 0.05
        return 0.30, 0.30, 0.20, 0.15
    if asset_class == "mega_cap":
        if trend == "BEAR":
            return 0.35, 0.40, 0.20, 0.05
        if trend == "BULL":
            return 0.35, 0.25, 0.15, 0.20
        return 0.35, 0.35, 0.15, 0.10
    return 0.40, 0.30, 0.15, 0.10


def add_adaptive_overall_risk(df: pd.DataFrame, cfg: AdaptiveControlsConfig, asset_class: str) -> pd.DataFrame:
    df = df.copy()
    hv = safe_series(df, "ph_rank_adaptive", 0.5)
    down = safe_series(df, "adaptive_down_risk", safe_series(df, "down_rank_adaptive", 0.5))
    down_strength = safe_series(df, "prob_down_strength_ctrl", safe_series(df, "prob_down_strengthening_score", 0.0))
    up_strength = safe_series(df, "prob_up_strength_ctrl", safe_series(df, "prob_up_strengthening_score", 0.0))
    trend = df.get("mid_trend_state", pd.Series("NEUTRAL", index=df.index)).fillna("NEUTRAL").astype(str)

    scores = []
    for i in range(len(df)):
        w_hv, w_down, w_ds, w_up = adaptive_risk_weights(asset_class, trend.iloc[i], float(hv.iloc[i]))
        denom = max(w_hv + w_down + w_ds, 1e-9)
        risk = (w_hv * hv.iloc[i] + w_down * down.iloc[i] + w_ds * down_strength.iloc[i]) / denom
        risk = risk - w_up * up_strength.iloc[i]
        # A high ph_rank in BULL for sector/growth is not always defensive risk.
        if trend.iloc[i] == "BULL" and asset_class in {"sector_etf", "high_vol_growth"} and up_strength.iloc[i] >= 0.38:
            risk -= 0.06
        scores.append(float(np.clip(risk, 0.0, 1.0)))
    df["adaptive_overall_risk"] = scores
    df["adaptive_pred_risk"] = np.where(df["adaptive_overall_risk"] >= cfg.high_vol_rank_threshold, "위험", "정상")
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
            return 0, 0.0, 0.5
        vals = list(buf)
        m = float(np.mean(vals))
        d = float(np.mean([1.0 if x < 0 else 0.0 for x in vals]))
        return n, m, d

    def _score_to_adjust(score: float, asset_class: str) -> float:
        pos_th = float(cfg.context_positive_score_threshold)
        neg_th = float(cfg.context_negative_score_threshold)
        pos_max = float(cfg.context_adjust_max_positive)
        neg_max = float(cfg.context_adjust_max_negative)
        if score >= pos_th:
            # Conservative convex-ish risk-on scaling after threshold.
            denom = max(1e-9, 1.0 - pos_th)
            adj = min(1.0, (score - pos_th) / denom) * pos_max
        elif score <= neg_th:
            denom = max(1e-9, 1.0 + neg_th)
            adj = -min(1.0, (neg_th - score) / denom) * neg_max
        else:
            adj = 0.0
        if asset_class == "broad_index" and not bool(cfg.broad_index_allow_positive_context_adjust):
            adj = min(adj, 0.0)
        return float(adj)

    # Process one Date at a time. Update histories only after all rows for that Date are scored.
    for _, day_df in work.groupby("Date", sort=False):
        pending_updates = []
        for idx, row in day_df.iterrows():
            ac = str(row.get("asset_class", "unknown"))
            key = str(row.get("context_key", ""))
            n, mean, down_freq = _stats(ctx_hist[key])
            scope = "asset_class_trend_phbin"
            if n < int(cfg.context_min_rows):
                n, mean, down_freq = _stats(ac_hist[ac])
                scope = "asset_class"
            if n < int(cfg.context_global_min_rows):
                n, mean, down_freq = _stats(global_hist)
                scope = "global"
            if n < int(cfg.context_global_min_rows):
                n, mean, down_freq = 0, 0.0, 0.5
                scope = "insufficient_history"

            score = float(cfg.context_return_scale) * float(mean) - float(cfg.context_down_penalty) * (float(down_freq) - 0.5)
            adj = _score_to_adjust(score, ac) if n > 0 else 0.0

            rows_out.append(int(n))
            scope_out.append(scope)
            mean_out.append(float(mean))
            down_out.append(float(down_freq))
            score_out.append(float(score))
            adjust_out.append(float(adj))

            ret_val = row.get("stock_next_return", 0.0)
            ret_val = float(ret_val) if np.isfinite(ret_val) else 0.0
            pending_updates.append((key, ac, ret_val))

        for key, ac, ret_val in pending_updates:
            ctx_hist[key].append(ret_val)
            ac_hist[ac].append(ret_val)
            global_hist.append(ret_val)

    work["context_rows"] = rows_out
    work["context_scope"] = scope_out
    work["context_mean_return"] = mean_out
    work["context_down_freq"] = down_out
    work["context_tail10"] = np.nan
    work["context_score"] = score_out
    work["context_adjust"] = adjust_out
    return work

def stock_from_risk_score(risk: float) -> float:
    if risk < 0.25:
        return 0.86
    if risk < 0.35:
        return 0.82
    if risk < 0.50:
        return 0.74
    if risk < 0.65:
        return 0.60
    if risk < 0.75:
        return 0.52
    if risk < 0.86:
        return 0.42
    return 0.30


def bond_cash_from_stock(stock: float) -> Tuple[float, float]:
    defensive = max(0.0, 1.0 - stock)
    return defensive * 0.65, defensive * 0.35


def add_signal_weights(df: pd.DataFrame, cfg: AdaptiveControlsConfig) -> pd.DataFrame:
    df = df.copy()
    stocks = []
    notes = []
    for _, row in df.iterrows():
        ac = str(row["asset_class"])
        trend = str(row.get("mid_trend_state", "NEUTRAL"))
        risk = float(row.get("adaptive_overall_risk", 0.5))
        ph = float(row.get("ph_rank_adaptive", 0.5))
        up = float(row.get("prob_up_strength_ctrl", row.get("prob_up_strengthening_score", 0.0)))
        down = float(row.get("prob_down_strength_ctrl", row.get("prob_down_strengthening_score", 0.0)))
        context_adjust = float(row.get("context_adjust", 0.0))

        stock = stock_from_risk_score(risk)
        note = [f"risk_bucket={risk:.3f}"]

        # rolling context policy table adjustment
        stock += context_adjust
        if abs(context_adjust) > 1e-9:
            note.append(f"ctx_adj={context_adjust:+.3f}")

        # upside overlay: adaptive and guarded. Broad-index overlay is disabled by default in v8.6.42b
        # because v8.6.42a over-activated QQQ/SPY offensive exposure.
        upside_allowed = bool(cfg.enable_upside_overlay)
        if ac == "broad_index" and not bool(cfg.broad_index_enable_upside_overlay):
            upside_allowed = False
        if upside_allowed and up >= cfg.up_tier1 and risk < 0.78:
            target = 0.82
            tier = 1
            if up >= cfg.up_tier2 and risk < 0.72:
                target = 0.88
                tier = 2
            if up >= cfg.up_tier3 and risk < 0.68:
                target = 0.96
                tier = 3
            if up >= cfg.full_stock_up_score and ph <= cfg.full_stock_high_vol_max and trend in {"BULL", "NEUTRAL"}:
                target = 1.00
                tier = 4
            # Down-strength blocks aggressive override.
            if down >= 0.60 and trend == "BEAR":
                target = min(target, 0.62)
                note.append("down_strength_bear_cap")
            if target > stock:
                stock = target
                note.append(f"up_overlay_tier={tier}")

        # asset-class guardrails, not ticker-specific
        if ac == "broad_index":
            stock = min(stock, cfg.broad_index_max_stock)
        elif ac == "mega_cap":
            if trend == "BEAR" and risk >= 0.55:
                stock = min(stock, cfg.mega_cap_bear_cap)
                note.append("mega_bear_cap")
            if trend == "BEAR" and risk >= 0.75:
                stock = min(stock, cfg.mega_cap_extreme_bear_cap)
                note.append("mega_extreme_bear_cap")
        elif ac == "sector_etf":
            if trend == "BULL" and up >= 0.38 and risk < 0.82:
                stock = max(stock, cfg.sector_bull_floor)
                note.append("sector_bull_floor")
        elif ac == "high_vol_growth":
            if trend == "BULL" and up >= 0.42 and risk < 0.80:
                stock = max(stock, cfg.growth_bull_floor)
                note.append("growth_bull_floor")
        else:
            stock = min(stock, cfg.unknown_max_stock)

        stocks.append(float(np.clip(stock, 0.0, 1.0)))
        notes.append(";".join(note))
    df["adaptive_signal_stock_weight"] = stocks
    bc = [bond_cash_from_stock(s) for s in stocks]
    df["adaptive_signal_bond_weight"] = [x[0] for x in bc]
    df["adaptive_signal_cash_weight"] = [x[1] for x in bc]
    df["adaptive_policy_note"] = notes
    return df


def simulate_execution(df: pd.DataFrame, cfg: AdaptiveControlsConfig) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    exec_stock, exec_bond, exec_cash = [], [], []
    turnovers, costs, gross_r, net_r, rebalanced = [], [], [], [], []
    prev = np.array([float(df.loc[0, "adaptive_signal_stock_weight"]), float(df.loc[0, "adaptive_signal_bond_weight"]), float(df.loc[0, "adaptive_signal_cash_weight"])], dtype=float)
    last_rebalance_i = -10**9
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
            change = signal - prev
            max_change = float(cfg.max_weight_change_per_rebalance)
            if np.isfinite(max_change) and max_change > 0:
                change = np.clip(change, -max_change, max_change)
                new_w = prev + change
                new_w = np.clip(new_w, 0.0, 1.0)
                if new_w.sum() <= 1e-12:
                    new_w = signal
                else:
                    new_w = new_w / new_w.sum()
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
        br = float(row.get("bond_next_return", 0.0)) if np.isfinite(row.get("bond_next_return", 0.0)) else 0.0
        cr = float(row.get("cash_next_return", 0.0)) if np.isfinite(row.get("cash_next_return", 0.0)) else 0.0
        gr = float(prev[0] * sr + prev[1] * br + prev[2] * cr)
        cost = turnover * float(cfg.transaction_cost_rate)
        nr = gr - cost
        exec_stock.append(float(prev[0])); exec_bond.append(float(prev[1])); exec_cash.append(float(prev[2]))
        turnovers.append(turnover); costs.append(cost); gross_r.append(gr); net_r.append(nr); rebalanced.append(reb)
    df["stock_weight"] = exec_stock
    df["bond_weight"] = exec_bond
    df["cash_weight"] = exec_cash
    df["turnover"] = turnovers
    df["transaction_cost"] = costs
    df["strategy_return_gross"] = gross_r
    df["strategy_return_net"] = net_r
    df["rebalanced"] = rebalanced
    df["strategy_equity_net"] = float(cfg.initial_capital) * (1.0 + df["strategy_return_net"].fillna(0.0)).cumprod()
    df["strategy_equity_gross"] = float(cfg.initial_capital) * (1.0 + df["strategy_return_gross"].fillna(0.0)).cumprod()
    return df


def process_ticker(ticker: str, pred_path: Path, cfg: AdaptiveControlsConfig) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    if "Date" not in df.columns:
        raise ValueError(f"{pred_path}: Date column not found")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["ticker"] = ticker.upper()
    ac = infer_asset_class(ticker)
    df["asset_class"] = ac
    df = add_multi_window_ranks(df, cfg, ac)
    df = add_adaptive_smoothing(df, cfg, ac)
    df = add_proxy_down_components(df, cfg)
    df = add_adaptive_down_weights(df, cfg)
    df = add_adaptive_overall_risk(df, cfg, ac)
    return df


def summarize_ticker(df: pd.DataFrame, cfg: AdaptiveControlsConfig) -> Dict[str, object]:
    perf = performance_metrics(df["strategy_return_net"], cfg.initial_capital)
    gross = performance_metrics(df["strategy_return_gross"], cfg.initial_capital)
    return {
        "ticker": str(df["ticker"].iloc[0]),
        "asset_class": str(df["asset_class"].iloc[0]),
        **perf,
        "gross_cagr": gross["cagr"],
        "gross_mdd": gross["mdd"],
        "gross_sharpe": gross["sharpe"],
        "avg_stock_weight": float(df["stock_weight"].mean()),
        "avg_signal_stock_weight": float(df["adaptive_signal_stock_weight"].mean()),
        "turnover": float(df["turnover"].sum() / max(len(df) / 252.0, 1e-9)),
        "avg_adaptive_overall_risk": float(df["adaptive_overall_risk"].mean()),
        "avg_ph_rank_adaptive": float(df["ph_rank_adaptive"].mean()),
        "avg_ewma_span": float(df["adaptive_ewma_span"].mean()),
        "offensive_activation_rate": float((df["adaptive_signal_stock_weight"] >= 0.82).mean()),
        "full_stock_rate": float((df["adaptive_signal_stock_weight"] >= 0.98).mean()),
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
    p = argparse.ArgumentParser(description="v8.6.42 adaptive controls resim for v8.6.41 predictions")
    p.add_argument("--input-dir", type=str, default=".")
    p.add_argument("--out-dir", type=str, default="results_v8_6_42_adaptive_controls")
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
    combined_ctx[["Date", "ticker", "asset_class", "mid_trend_state", "ph_bin", "context_scope", "context_rows", "context_mean_return", "context_down_freq", "context_tail10", "context_score", "context_adjust"]].to_csv(
        out_dir / "adaptive_context_policy_table.csv", index=False, encoding="utf-8-sig"
    )
    with open(out_dir / "adaptive_controls_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    print("\n[SAVED]")
    print(out_dir / "multi_asset_summary.csv")


if __name__ == "__main__":
    main()
