"""
v8.6.40d_policy1_resim.py
================================

Purpose
- Re-simulate allocation policy from existing v8.6.40b predictions.csv.
- No model retraining.
- Adds ph_rank(raw high-vol probability) + asset_class + mid_trend based base allocation.
- Preserves the previous v8.6.40b offensive overlay approximately by carrying over the old overlay bonus.

Inputs
- A result directory containing ticker subfolders, each with *_predictions.csv; or a flat directory with ticker predictions.
- Existing predictions must contain stock_next_return, bond_next_return, cash_next_return.

Outputs
- Per-ticker enriched predictions and summary.
- Multi-asset summary CSV.

Design constraints
- ph_rank uses prob_high_vol_raw if available, not EWMA-smoothed prob_high_vol.
- ph_rank is computed against past window only: current ph is ranked versus historical ph[t-window:t].
- min_periods defaults to 504. Before enough history exists, policy falls back to the existing v8.6.40b base_signal weights.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import glob

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

@dataclass
class Policy1Config:
    initial_capital: float = 100_000_000.0
    transaction_cost_rate: float = 0.001
    rebalance_every_n_days: int = 5
    no_trade_band: float = 0.12
    emergency_cooldown_days: int = 5

    ph_rank_window: int = 756
    ph_rank_min_periods: int = 504
    ph_rank_fallback: str = "old_base"  # old_base, raw_ph
    bond_ratio_of_defensive: float = 0.65

    # Legacy raw/EWMA risk thresholds used only as fallback/emergency references.
    gate_normal_high_vol_threshold: float = 0.55
    gate_high_vol_threshold: float = 0.74
    extreme_high_vol_threshold: float = 0.86
    raw_extreme_high_vol_threshold: float = 0.88
    emergency_high_vol_threshold: float = 0.88
    emergency_combined_high_vol_threshold: float = 0.78

    use_context_gate: bool = True
    use_trend_context_overlay: bool = True
    preserve_old_offensive_overlay: bool = True
    allow_tier3_full_bypass_context_cap: bool = True

    # If true, old Tier2 overlay bonus is ignored when preserving old overlay.
    disable_tier2_bonus: bool = True

    # Hard limits. These prevent accidental invalid allocations.
    min_stock: float = 0.20
    max_stock: float = 1.00


ASSET_CLASS_PH_RANK_BASE: Dict[str, Dict[str, float]] = {
    "broad_index": {
        "00_30": 0.84,
        "30_50": 0.78,
        "50_70": 0.68,
        "70_85": 0.58,
        "85_95": 0.48,
        "95_100": 0.38,
    },
    "mega_cap": {
        "00_30": 0.86,
        "30_50": 0.80,
        "50_70": 0.72,
        "70_85": 0.62,
        "85_95": 0.50,
        "95_100": 0.38,
    },
    "sector_etf": {
        "00_30": 0.88,
        "30_50": 0.82,
        "50_70": 0.72,
        "70_85": 0.62,
        "85_95": 0.52,
        "95_100": 0.42,
    },
    "high_vol_growth": {
        "00_30": 0.90,
        "30_50": 0.84,
        "50_70": 0.78,
        "70_85": 0.70,
        "85_95": 0.60,
        "95_100": 0.48,
    },
}

TREND_CONTEXT_POLICY: Dict[str, Dict[str, Dict[str, float]]] = {
    "broad_index": {
        "BULL": {"floor_add": 0.04, "cap": 0.88},
        "NEUTRAL": {"floor_add": 0.00, "cap": 0.78},
        "BEAR": {"floor_add": 0.00, "cap": 0.72},
        "UNKNOWN": {"floor_add": 0.00, "cap": 0.78},
    },
    "mega_cap": {
        "BULL": {"floor_add": 0.05, "cap": 0.90},
        "NEUTRAL": {"floor_add": 0.00, "cap": 0.78},
        "BEAR": {"floor_add": 0.00, "cap": 0.72},
        "UNKNOWN": {"floor_add": 0.00, "cap": 0.78},
    },
    "sector_etf": {
        "BULL": {"floor_add": 0.08, "cap": 0.92},
        "NEUTRAL": {"floor_add": 0.00, "cap": 0.78},
        "BEAR": {"floor_add": 0.00, "cap": 0.68},
        "UNKNOWN": {"floor_add": 0.00, "cap": 0.78},
    },
    "high_vol_growth": {
        "BULL": {"floor_add": 0.12, "cap": 0.95},
        "NEUTRAL": {"floor_add": 0.00, "cap": 0.82},
        "BEAR": {"floor_add": 0.00, "cap": 0.75},
        "UNKNOWN": {"floor_add": 0.00, "cap": 0.82},
    },
}

REGIME_TREND_FLOOR_CAP: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {
    "high_vol_growth": {
        ("HIGH_VOL", "BULL"): {"floor": 0.78, "cap": 0.95},
        ("WATCH_BULL_VOL", "BULL"): {"floor": 0.72, "cap": 0.92},
        ("RISK_OFF", "BULL"): {"floor": 0.55, "cap": 0.75},
        ("HIGH_VOL", "BEAR"): {"floor": 0.35, "cap": 0.60},
        ("RISK_OFF", "BEAR"): {"floor": 0.25, "cap": 0.45},
        ("EXTREME_RISK", "BEAR"): {"floor": 0.20, "cap": 0.35},
    },
    "sector_etf": {
        ("HIGH_VOL", "BULL"): {"floor": 0.70, "cap": 0.88},
        ("WATCH_BULL_VOL", "BULL"): {"floor": 0.68, "cap": 0.86},
        ("RISK_OFF", "BULL"): {"floor": 0.50, "cap": 0.70},
        ("HIGH_VOL", "BEAR"): {"floor": 0.30, "cap": 0.55},
        ("RISK_OFF", "BEAR"): {"floor": 0.25, "cap": 0.45},
        ("EXTREME_RISK", "BEAR"): {"floor": 0.20, "cap": 0.35},
    },
    "mega_cap": {
        ("WATCH", "BEAR"): {"floor": 0.35, "cap": 0.70},
        ("WATCH_BULL_VOL", "BEAR"): {"floor": 0.35, "cap": 0.70},
        ("CUSTOM", "BEAR"): {"floor": 0.40, "cap": 0.75},
        ("HIGH_VOL", "BULL"): {"floor": 0.60, "cap": 0.82},
        ("RISK_OFF", "BULL"): {"floor": 0.45, "cap": 0.68},
    },
    "broad_index": {
        ("HIGH_VOL", "BULL"): {"floor": 0.58, "cap": 0.78},
        ("WATCH_BULL_VOL", "BULL"): {"floor": 0.62, "cap": 0.82},
        ("RISK_OFF", "BULL"): {"floor": 0.40, "cap": 0.65},
        ("HIGH_VOL", "BEAR"): {"floor": 0.25, "cap": 0.52},
        ("RISK_OFF", "BEAR"): {"floor": 0.20, "cap": 0.42},
        ("EXTREME_RISK", "BEAR"): {"floor": 0.20, "cap": 0.35},
    },
}


# -----------------------------
# Helpers
# -----------------------------

def normalize_weights(stock: float, bond: float, cash: float) -> Tuple[float, float, float]:
    vals = np.array([stock, bond, cash], dtype=float)
    vals = np.clip(vals, 0.0, 1.0)
    total = float(vals.sum())
    if total <= 0:
        return 1.0, 0.0, 0.0
    vals = vals / total
    return float(vals[0]), float(vals[1]), float(vals[2])


def weights_from_stock(stock: float, cfg: Policy1Config) -> Tuple[float, float, float]:
    stock = float(np.clip(stock, 0.0, 1.0))
    remain = max(0.0, 1.0 - stock)
    br = float(np.clip(cfg.bond_ratio_of_defensive, 0.0, 1.0))
    return normalize_weights(stock, remain * br, remain * (1.0 - br))


def redistribute_after_stock_change(stock: float, old_w: Tuple[float, float, float], cfg: Policy1Config) -> Tuple[float, float, float]:
    stock = float(np.clip(stock, 0.0, 1.0))
    old_bond = float(old_w[1])
    old_cash = float(old_w[2])
    defensive_sum = old_bond + old_cash
    if defensive_sum <= 1e-12:
        return weights_from_stock(stock, cfg)
    remain = max(0.0, 1.0 - stock)
    return normalize_weights(stock, remain * old_bond / defensive_sum, remain * old_cash / defensive_sum)


def infer_asset_class(ticker: str, override: Optional[str] = None) -> str:
    if override:
        val = str(override).strip().lower()
        if val in ASSET_CLASS_PH_RANK_BASE:
            return val
    t = str(ticker).upper().strip()
    if t in {"QQQ", "SPY", "DIA", "IWM", "VTI", "VOO", "IVV", "SCHB", "SPLG"}:
        return "broad_index"
    if t in {"SOXX", "SMH", "XLK", "XLY", "XLF", "XLV", "XLE", "XLI", "XLC", "ARKK"}:
        return "sector_etf"
    if t in {"NVDA", "TSLA", "AMD", "PLTR", "MSTR", "COIN", "SMCI"}:
        return "high_vol_growth"
    if t in {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "AVGO", "LLY", "COST", "NFLX"}:
        return "mega_cap"
    return "mega_cap"


def ph_rank_bin(x: float) -> str:
    if not np.isfinite(x):
        return "raw_fallback"
    if x < 0.30:
        return "00_30"
    if x < 0.50:
        return "30_50"
    if x < 0.70:
        return "50_70"
    if x < 0.85:
        return "70_85"
    if x < 0.95:
        return "85_95"
    return "95_100"


def rolling_percentile_current_vs_past(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    out = np.full(len(vals), np.nan, dtype=float)
    for i, cur in enumerate(vals):
        if not np.isfinite(cur):
            continue
        start = max(0, i - int(window))
        hist = vals[start:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < int(min_periods):
            continue
        # Percentile of current relative to strictly past observations only.
        out[i] = float((np.sum(hist < cur) + 0.5 * np.sum(hist == cur)) / len(hist))
    return pd.Series(out, index=s.index)


def rolling_z_current_vs_past(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    out = np.full(len(vals), np.nan, dtype=float)
    for i, cur in enumerate(vals):
        if not np.isfinite(cur):
            continue
        start = max(0, i - int(window))
        hist = vals[start:i]
        hist = hist[np.isfinite(hist)]
        if len(hist) < int(min_periods):
            continue
        sd = float(np.std(hist, ddof=1)) if len(hist) > 1 else np.nan
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        out[i] = float((cur - float(np.mean(hist))) / sd)
    return pd.Series(out, index=s.index)


def add_ph_context(df: pd.DataFrame, cfg: Policy1Config) -> pd.DataFrame:
    out = df.copy()
    if "prob_high_vol_raw" in out.columns:
        ph_raw = pd.to_numeric(out["prob_high_vol_raw"], errors="coerce").astype(float)
    else:
        ph_raw = pd.to_numeric(out["prob_high_vol"], errors="coerce").astype(float)
    out["ph_raw"] = ph_raw.clip(0.0, 1.0)
    out["ph_ewma"] = pd.to_numeric(out.get("prob_high_vol", out["ph_raw"]), errors="coerce").astype(float).clip(0.0, 1.0)
    out["ph_rank_756"] = rolling_percentile_current_vs_past(out["ph_raw"], cfg.ph_rank_window, cfg.ph_rank_min_periods)
    out["ph_z_756"] = rolling_z_current_vs_past(out["ph_raw"], cfg.ph_rank_window, cfg.ph_rank_min_periods)
    out["ph_context_available_756"] = out["ph_rank_756"].notna()
    out["ph_rank_bin_756"] = out["ph_rank_756"].apply(ph_rank_bin)
    return out


def legacy_base_weight_from_raw_ph(ph: float, cfg: Policy1Config) -> Tuple[float, float, float]:
    ph = float(np.clip(ph, 0.0, 1.0))
    if ph < 0.25:
        stock = 0.86
    elif ph < 0.35:
        stock = 0.82
    elif ph < 0.50:
        stock = 0.74
    elif ph < 0.65:
        stock = 0.60
    elif ph < 0.75:
        stock = 0.52
    elif ph < 0.86:
        stock = 0.42
    else:
        stock = 0.30
    return weights_from_stock(stock, cfg)


def classify_gate_by_ph_context(row: pd.Series, asset_class: str, cfg: Policy1Config) -> str:
    ph_raw = float(row.get("ph_raw", row.get("prob_high_vol_raw", row.get("prob_high_vol", 0.5))))
    ph_ewma = float(row.get("ph_ewma", row.get("prob_high_vol", ph_raw)))
    ph_rank = float(row.get("ph_rank_756", np.nan))
    mid = str(row.get("mid_trend_state", "UNKNOWN")).upper()

    if not np.isfinite(ph_rank):
        # Legacy 40b-compatible fallback; pdn is effectively ph.
        if ph_ewma < cfg.gate_normal_high_vol_threshold:
            return "NORMAL"
        if ph_ewma >= cfg.extreme_high_vol_threshold:
            return "EXTREME_RISK"
        if ph_ewma >= cfg.gate_high_vol_threshold:
            return "RISK_OFF"
        return "WATCH"

    # Raw high-vol jump and relative extreme both count as emergency risk.
    if ph_raw >= cfg.raw_extreme_high_vol_threshold or ph_rank >= 0.95:
        return "EXTREME_RISK"

    if ph_rank >= 0.85:
        if asset_class in {"high_vol_growth", "sector_etf"} and mid == "BULL":
            return "HIGH_VOL"
        return "RISK_OFF"

    if ph_rank >= 0.70:
        if asset_class in {"high_vol_growth", "sector_etf"} and mid == "BULL":
            return "WATCH_BULL_VOL"
        return "HIGH_VOL"

    if ph_rank >= 0.50:
        return "WATCH"
    return "NORMAL"


def base_weight_from_ph_context(row: pd.Series, asset_class: str, cfg: Policy1Config) -> Tuple[Tuple[float, float, float], str, Dict[str, object]]:
    ph_rank = float(row.get("ph_rank_756", np.nan))
    mid = str(row.get("mid_trend_state", "UNKNOWN")).upper()
    bin_name = ph_rank_bin(ph_rank)

    if not np.isfinite(ph_rank):
        if cfg.ph_rank_fallback == "raw_ph":
            w = legacy_base_weight_from_raw_ph(float(row.get("ph_raw", row.get("prob_high_vol", 0.5))), cfg)
            mode = "raw_ph_fallback"
        else:
            # Exact prior result-preserving fallback.
            stock = float(row.get("base_signal_stock_weight", row.get("signal_stock_weight", 0.72)))
            bond = float(row.get("base_signal_bond_weight", (1.0 - stock) * cfg.bond_ratio_of_defensive))
            cash = float(row.get("base_signal_cash_weight", max(0.0, 1.0 - stock - bond)))
            w = normalize_weights(stock, bond, cash)
            mode = "old_base_fallback"
        return w, mode, {"ph_rank_bin": "raw_fallback", "trend_policy_applied": False}

    table = ASSET_CLASS_PH_RANK_BASE.get(asset_class, ASSET_CLASS_PH_RANK_BASE["mega_cap"])
    base_stock = float(table.get(bin_name, 0.72))
    before_trend = base_stock
    trend_policy = TREND_CONTEXT_POLICY.get(asset_class, TREND_CONTEXT_POLICY["mega_cap"]).get(mid, TREND_CONTEXT_POLICY["mega_cap"]["UNKNOWN"])
    if cfg.use_trend_context_overlay:
        base_stock = min(float(trend_policy["cap"]), base_stock + float(trend_policy["floor_add"]))
    base_stock = float(np.clip(base_stock, cfg.min_stock, cfg.max_stock))
    return weights_from_stock(base_stock, cfg), "ph_rank_policy1_base", {
        "ph_rank_bin": bin_name,
        "base_stock_before_trend": before_trend,
        "trend_policy_applied": bool(cfg.use_trend_context_overlay),
        "trend_cap": float(trend_policy.get("cap", np.nan)),
        "trend_floor_add": float(trend_policy.get("floor_add", 0.0)),
    }


def apply_regime_trend_floor_cap(stock: float, regime: str, mid: str, asset_class: str, tier: int, full_stock_signal: bool, cfg: Policy1Config) -> Tuple[float, str, float, float]:
    stock_in = float(stock)
    mid = str(mid).upper()
    policy = REGIME_TREND_FLOOR_CAP.get(asset_class, {}).get((str(regime), mid))
    if not policy:
        return float(np.clip(stock_in, cfg.min_stock, cfg.max_stock)), "none", np.nan, np.nan

    floor = float(policy.get("floor", cfg.min_stock))
    cap = float(policy.get("cap", cfg.max_stock))

    # Strong Tier3/Full can bypass context cap in policy1. Floor still applies.
    if cfg.allow_tier3_full_bypass_context_cap and (int(tier) >= 3 or bool(full_stock_signal)):
        cap = cfg.max_stock

    stock_out = float(np.clip(max(stock_in, floor), cfg.min_stock, cap))
    return stock_out, "regime_trend_floor_cap", floor, cap


def should_override_no_trade(df: pd.DataFrame, i: int, row: pd.Series, cfg: Policy1Config) -> Tuple[bool, str]:
    if i <= 0:
        return False, ""
    prev = df.iloc[i - 1]
    ph = float(row.get("ph_ewma", row.get("prob_high_vol", 0.0)))
    ph_prev = float(prev.get("ph_ewma", prev.get("prob_high_vol", ph)))
    mid = str(row.get("mid_trend_state", "UNKNOWN")).upper()
    mid_prev = str(prev.get("mid_trend_state", "UNKNOWN")).upper()
    if ph_prev < 0.70 <= ph:
        return True, "high_vol_cross_up"
    if ph - ph_prev >= 0.15:
        return True, "high_vol_surge"
    if {mid, mid_prev} == {"BULL", "BEAR"}:
        return True, "major_trend_flip"
    return False, ""


def infer_regime_from_stock(stock: float) -> str:
    s = float(stock)
    if s >= 0.82:
        return "CUSTOM"
    if s >= 0.68:
        return "NORMAL"
    if s >= 0.54:
        return "WATCH"
    if s >= 0.36:
        return "RISK_OFF"
    return "EXTREME_RISK"


def max_drawdown(returns: pd.Series) -> float:
    eq = (1.0 + pd.to_numeric(returns, errors="coerce").fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def annualized_return(returns: pd.Series, dates: Optional[pd.Series] = None) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if len(r) == 0:
        return 0.0
    total = float((1.0 + r).prod())
    if total <= 0:
        return -1.0
    if dates is not None and len(dates) >= 2:
        d0 = pd.to_datetime(dates.iloc[0])
        d1 = pd.to_datetime(dates.iloc[-1])
        years = max((d1 - d0).days / 365.25, len(r) / 252.0)
    else:
        years = len(r) / 252.0
    return float(total ** (1.0 / years) - 1.0) if years > 0 else 0.0


def sharpe_ratio(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    sd = float(r.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(r.mean() / sd * math.sqrt(252.0))


def sortino_ratio(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    downside = r[r < 0]
    sd = float(downside.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(r.mean() / sd * math.sqrt(252.0))


def perf_stats(returns: pd.Series, cfg: Policy1Config, dates: Optional[pd.Series] = None) -> Dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    final_cap = float(cfg.initial_capital * (1.0 + r).cumprod().iloc[-1]) if len(r) else cfg.initial_capital
    cagr = annualized_return(r, dates)
    mdd = max_drawdown(r)
    sh = sharpe_ratio(r)
    so = sortino_ratio(r)
    return {
        "final_capital": final_cap,
        "total_return": final_cap / cfg.initial_capital - 1.0,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sh,
        "sortino": so,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
    }


def simulate_policy1(df_in: pd.DataFrame, ticker: str, cfg: Policy1Config, asset_class_override: Optional[str] = None) -> pd.DataFrame:
    df = df_in.copy().reset_index(drop=True)
    if "Date" not in df.columns:
        raise ValueError("predictions.csv must contain Date column")
    required = ["stock_next_return", "bond_next_return", "cash_next_return"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"predictions.csv missing columns: {missing}")

    df = add_ph_context(df, cfg)
    asset_class = infer_asset_class(ticker, asset_class_override)
    df["asset_class"] = asset_class

    prev_w: Optional[Tuple[float, float, float]] = None
    last_emergency_i = -10**9
    rows: List[Dict[str, object]] = []

    for i, row in df.iterrows():
        signal_regime = classify_gate_by_ph_context(row, asset_class, cfg) if cfg.use_context_gate else str(row.get("signal_regime", "NORMAL"))
        base_w, base_mode, base_meta = base_weight_from_ph_context(row, asset_class, cfg)

        old_base_stock = float(row.get("base_signal_stock_weight", base_w[0]))
        old_signal_stock = float(row.get("signal_stock_weight", old_base_stock))
        old_overlay_bonus = max(0.0, old_signal_stock - old_base_stock)
        tier = int(row.get("offensive_tier", 0) if pd.notna(row.get("offensive_tier", 0)) else 0)
        full_stock_signal = bool(row.get("full_stock_signal", False))
        tier3_signal = bool(row.get("tier3_signal", False))
        original_tier2_signal = bool(row.get("original_tier2_signal", row.get("tier2_signal", False)))

        if cfg.disable_tier2_bonus and tier == 2 and not tier3_signal and not full_stock_signal:
            old_overlay_bonus = 0.0
            tier = 0

        target_stock = float(base_w[0])
        if cfg.preserve_old_offensive_overlay:
            target_stock = max(target_stock, target_stock + old_overlay_bonus)

        mid = str(row.get("mid_trend_state", "UNKNOWN")).upper()
        target_stock, context_action, context_floor, context_cap = apply_regime_trend_floor_cap(
            stock=target_stock,
            regime=signal_regime,
            mid=mid,
            asset_class=asset_class,
            tier=tier,
            full_stock_signal=full_stock_signal,
            cfg=cfg,
        )
        signal_w = redistribute_after_stock_change(target_stock, base_w, cfg)

        ph = float(row.get("ph_ewma", row.get("prob_high_vol", 0.0)))
        ph_rank = float(row.get("ph_rank_756", np.nan))
        raw_emergency = bool(ph >= cfg.emergency_high_vol_threshold or (np.isfinite(ph_rank) and ph_rank >= 0.95) or signal_regime == "EXTREME_RISK")
        emergency = bool(raw_emergency and (i - last_emergency_i >= cfg.emergency_cooldown_days))
        scheduled = (i % cfg.rebalance_every_n_days == 0)
        force_rebalance = bool(row.get("force_rebalance_signal", False) or (tier >= 3))
        rebalance_due = bool(prev_w is None or scheduled or emergency or force_rebalance)

        hold_reason = "rebalanced"
        trade_executed = False
        if prev_w is None:
            w = signal_w
            executed_regime = signal_regime
            hold_reason = "initial"
            trade_executed = True
        elif not rebalance_due:
            override, override_reason = should_override_no_trade(df, i, row, cfg)
            if override:
                w = signal_w
                executed_regime = signal_regime
                hold_reason = f"no_trade_override_{override_reason}"
                trade_executed = True
            else:
                w = prev_w
                executed_regime = infer_regime_from_stock(w[0])
                hold_reason = "not_rebalance_day"
        else:
            total_delta = sum(abs(signal_w[j] - prev_w[j]) for j in range(3))
            if force_rebalance:
                w = signal_w
                executed_regime = signal_regime
                hold_reason = "strong_offensive_override" if tier >= 3 else "force_rebalance"
                trade_executed = True
            elif total_delta < cfg.no_trade_band:
                w = prev_w
                executed_regime = infer_regime_from_stock(w[0])
                hold_reason = "no_trade_band"
            else:
                w = signal_w
                executed_regime = signal_regime
                hold_reason = "emergency" if emergency else "scheduled"
                trade_executed = True

        turnover = 0.0 if prev_w is None else sum(abs(w[j] - prev_w[j]) for j in range(3))
        if turnover > 1e-12:
            trade_executed = True
        gross = float(w[0]) * float(row["stock_next_return"]) + float(w[1]) * float(row["bond_next_return"]) + float(w[2]) * float(row["cash_next_return"])
        cost = cfg.transaction_cost_rate * turnover
        net = gross - cost
        if emergency and rebalance_due:
            last_emergency_i = i

        out = row.to_dict()
        out.update({
            "ticker": ticker,
            "asset_class": asset_class,
            "policy1_signal_regime": signal_regime,
            "policy1_allocation_regime": executed_regime,
            "policy1_base_mode": base_mode,
            "policy1_context_action": context_action,
            "policy1_context_floor": context_floor,
            "policy1_context_cap": context_cap,
            "policy1_base_stock_before_trend": base_meta.get("base_stock_before_trend", np.nan),
            "policy1_trend_cap": base_meta.get("trend_cap", np.nan),
            "policy1_trend_floor_add": base_meta.get("trend_floor_add", np.nan),
            "policy1_old_overlay_bonus": old_overlay_bonus,
            "policy1_original_tier2_signal": original_tier2_signal,
            "policy1_signal_stock_weight": float(signal_w[0]),
            "policy1_signal_bond_weight": float(signal_w[1]),
            "policy1_signal_cash_weight": float(signal_w[2]),
            "policy1_base_signal_stock_weight": float(base_w[0]),
            "policy1_base_signal_bond_weight": float(base_w[1]),
            "policy1_base_signal_cash_weight": float(base_w[2]),
            "policy1_stock_weight": float(w[0]),
            "policy1_bond_weight": float(w[1]),
            "policy1_cash_weight": float(w[2]),
            "policy1_turnover": float(turnover),
            "policy1_transaction_cost": float(cost),
            "policy1_strategy_return_gross": float(gross),
            "policy1_strategy_return_net": float(net),
            "policy1_hold_reason": hold_reason,
            "policy1_rebalance_due": rebalance_due,
            "policy1_trade_executed": trade_executed,
            "policy1_emergency_rebalance": bool(emergency and rebalance_due),
        })
        rows.append(out)
        prev_w = w

    out_df = pd.DataFrame(rows)
    out_df["policy1_strategy_equity_net"] = cfg.initial_capital * (1.0 + out_df["policy1_strategy_return_net"].fillna(0.0)).cumprod()
    out_df["policy1_strategy_equity_gross"] = cfg.initial_capital * (1.0 + out_df["policy1_strategy_return_gross"].fillna(0.0)).cumprod()
    return out_df


def build_regime_trend_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        d: Dict[str, object] = dict(zip(group_cols, keys))
        d.update({
            "count": int(len(g)),
            "start_date": str(g["Date"].iloc[0]),
            "end_date": str(g["Date"].iloc[-1]),
            "avg_stock_weight": float(g["policy1_stock_weight"].mean()),
            "avg_signal_stock_weight": float(g["policy1_signal_stock_weight"].mean()),
            "avg_base_stock_weight": float(g["policy1_base_signal_stock_weight"].mean()),
            "avg_ph_raw": float(g["ph_raw"].mean()),
            "avg_ph_ewma": float(g["ph_ewma"].mean()),
            "avg_ph_rank": float(g["ph_rank_756"].mean()) if g["ph_rank_756"].notna().any() else np.nan,
            "strategy_ann_return": annualized_return(g["policy1_strategy_return_net"], g["Date"]),
            "strategy_mdd": max_drawdown(g["policy1_strategy_return_net"]),
            "bh_ann_return": annualized_return(g["stock_next_return"], g["Date"]),
            "bh_mdd": max_drawdown(g["stock_next_return"]),
        })
        d["bh_gap_ann_return"] = float(d["strategy_ann_return"] - d["bh_ann_return"])
        rows.append(d)
    return pd.DataFrame(rows)


def find_prediction_file(root: Path, ticker: str) -> Path:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in ticker).strip("_") or "asset"
    patterns = [
        root / safe / f"{safe}_*_predictions.csv",
        root / safe / "*_predictions.csv",
        root / f"{safe}_*_predictions.csv",
        root / "*_predictions.csv",
    ]
    for pat in patterns:
        matches = sorted(Path(x) for x in glob.glob(str(pat)))
        if matches:
            return matches[0]
    # Recursive fallback.
    matches = sorted(root.glob(f"**/{safe}_*_predictions.csv"))
    if matches:
        return matches[0]
    matches = sorted(root.glob("**/*_predictions.csv"))
    ticker_upper = ticker.upper()
    for m in matches:
        if ticker_upper.lower() in str(m).lower():
            return m
    raise FileNotFoundError(f"Could not find predictions.csv for ticker={ticker} under {root}")


def build_summary(df: pd.DataFrame, cfg: Policy1Config) -> Dict[str, object]:
    dates = df["Date"] if "Date" in df.columns else None
    strategy = perf_stats(df["policy1_strategy_return_net"], cfg, dates)
    strategy_gross = perf_stats(df["policy1_strategy_return_gross"], cfg, dates)
    stock_bh = perf_stats(df["stock_next_return"], cfg, dates)
    bench6040 = perf_stats(0.60 * df["stock_next_return"] + 0.40 * df["bond_next_return"], cfg, dates)
    turnover = pd.to_numeric(df["policy1_turnover"], errors="coerce").fillna(0.0)
    return {
        "model_type": "v8_6_40d_policy1_resim",
        "config": asdict(cfg),
        "rows": int(len(df)),
        "start_date": str(df["Date"].iloc[0]) if len(df) else "",
        "end_date": str(df["Date"].iloc[-1]) if len(df) else "",
        "performance": {
            "strategy_after_cost": strategy,
            "strategy_gross": strategy_gross,
            "stock_buy_hold": stock_bh,
            "benchmark_60_40": bench6040,
        },
        "average_weights": {
            "avg_stock_weight": float(df["policy1_stock_weight"].mean()),
            "avg_bond_weight": float(df["policy1_bond_weight"].mean()),
            "avg_cash_weight": float(df["policy1_cash_weight"].mean()),
            "min_stock_weight": float(df["policy1_stock_weight"].min()),
            "max_stock_weight": float(df["policy1_stock_weight"].max()),
        },
        "turnover": {
            "avg_daily_trade_ratio": float(turnover.mean()),
            "annual_turnover_estimate": float(turnover.mean() * 252.0),
            "total_transaction_cost_rate_sum": float((turnover * cfg.transaction_cost_rate).sum()),
            "trade_executed_ratio": float(df["policy1_trade_executed"].mean()),
            "emergency_rebalance_ratio": float(df["policy1_emergency_rebalance"].mean()),
        },
        "policy1_diagnostics": {
            "asset_class": str(df["asset_class"].iloc[0]) if len(df) else "",
            "ph_context_available_rate": float(df["ph_context_available_756"].mean()),
            "context_action_counts": df["policy1_context_action"].value_counts(dropna=False).to_dict(),
            "signal_regime_counts": df["policy1_signal_regime"].value_counts(dropna=False).to_dict(),
            "hold_reason_counts": df["policy1_hold_reason"].value_counts(dropna=False).to_dict(),
        },
    }


def process_ticker(root: Path, out_root: Path, ticker: str, cfg: Policy1Config, asset_class_override: Optional[str]) -> Dict[str, object]:
    pred_path = find_prediction_file(root, ticker)
    df = pd.read_csv(pred_path)
    out_df = simulate_policy1(df, ticker=ticker, cfg=cfg, asset_class_override=asset_class_override)

    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in ticker).strip("_") or "asset"
    ticker_dir = out_root / safe
    ticker_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{safe}_xgb_recency_weighted_v8_6_40d_policy1"
    pred_out = ticker_dir / f"{prefix}_predictions.csv"
    summary_out = ticker_dir / f"{prefix}_summary.json"
    regime_out = ticker_dir / f"{prefix}_regime_trend_ph_rank_performance.csv"
    asset_out = ticker_dir / f"{prefix}_asset_class_ph_rank_trend_performance.csv"
    hold_out = ticker_dir / f"{prefix}_hold_reason_ph_rank_trend_performance.csv"

    out_df.to_csv(pred_out, index=False, encoding="utf-8-sig")
    summary = build_summary(out_df, cfg)
    summary["source_predictions"] = str(pred_path)
    summary["ticker"] = ticker
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    build_regime_trend_summary(out_df, ["policy1_signal_regime", "mid_trend_state", "ph_rank_bin_756"]).to_csv(regime_out, index=False, encoding="utf-8-sig")
    build_regime_trend_summary(out_df, ["asset_class", "ph_rank_bin_756", "mid_trend_state"]).to_csv(asset_out, index=False, encoding="utf-8-sig")
    build_regime_trend_summary(out_df, ["policy1_hold_reason", "ph_rank_bin_756", "mid_trend_state"]).to_csv(hold_out, index=False, encoding="utf-8-sig")

    perf = summary["performance"]["strategy_after_cost"]
    row: Dict[str, object] = {
        "ticker": ticker,
        "returncode": 0,
        "source_predictions": str(pred_path),
        "result_dir": str(ticker_dir),
        "final_capital": perf["final_capital"],
        "cagr": perf["cagr"],
        "mdd": perf["mdd"],
        "sharpe": perf["sharpe"],
        "sortino": perf["sortino"],
        "calmar": perf["calmar"],
        "avg_stock_weight": summary["average_weights"]["avg_stock_weight"],
        "turnover": summary["turnover"]["annual_turnover_estimate"],
        "ph_context_available_rate": summary["policy1_diagnostics"]["ph_context_available_rate"],
        "asset_class": summary["policy1_diagnostics"]["asset_class"],
    }
    return row


def parse_asset_class_overrides(text: Optional[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not text:
        return out
    for part in str(text).split(","):
        if not part.strip() or ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip().upper()] = v.strip().lower()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v8.6.40d policy1 resimulation from existing predictions.csv")
    p.add_argument("--result-dir", required=True, help="기존 v8.6.40b result root. 예: results_v8_6_40b_clean_compare")
    p.add_argument("--asset-list", default="QQQ,SPY,AAPL,SOXX,NVDA", help="comma-separated tickers")
    p.add_argument("--out-dir", default="results_v8_6_40d_policy1_resim", help="output root")
    p.add_argument("--transaction-cost-rate", type=float, default=0.001)
    p.add_argument("--rebalance-every", type=int, default=5)
    p.add_argument("--no-trade-band", type=float, default=0.12)
    p.add_argument("--emergency-cooldown", type=int, default=5)
    p.add_argument("--ph-rank-window", type=int, default=756)
    p.add_argument("--ph-rank-min-periods", type=int, default=504)
    p.add_argument("--ph-rank-fallback", choices=["old_base", "raw_ph"], default="old_base")
    p.add_argument("--asset-class-overrides", default=None, help="예: NVDA:high_vol_growth,SOXX:sector_etf")
    p.add_argument("--no-context-gate", action="store_true")
    p.add_argument("--no-trend-context", action="store_true")
    p.add_argument("--keep-tier2-bonus", action="store_true", help="기존 Tier2 overlay bonus도 보존")
    p.add_argument("--no-old-offensive-overlay", action="store_true", help="기존 40b offensive overlay bonus를 보존하지 않음")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.result_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = Policy1Config(
        transaction_cost_rate=float(args.transaction_cost_rate),
        rebalance_every_n_days=int(args.rebalance_every),
        no_trade_band=float(args.no_trade_band),
        emergency_cooldown_days=int(args.emergency_cooldown),
        ph_rank_window=int(args.ph_rank_window),
        ph_rank_min_periods=int(args.ph_rank_min_periods),
        ph_rank_fallback=str(args.ph_rank_fallback),
        use_context_gate=not bool(args.no_context_gate),
        use_trend_context_overlay=not bool(args.no_trend_context),
        preserve_old_offensive_overlay=not bool(args.no_old_offensive_overlay),
        disable_tier2_bonus=not bool(args.keep_tier2_bonus),
    )
    overrides = parse_asset_class_overrides(args.asset_class_overrides)
    tickers = [x.strip().upper() for x in str(args.asset_list).split(",") if x.strip()]

    rows: List[Dict[str, object]] = []
    for ticker in tickers:
        print(f"[POLICY1] {ticker}")
        try:
            row = process_ticker(root, out_root, ticker, cfg, overrides.get(ticker))
        except Exception as exc:
            row = {"ticker": ticker, "returncode": 1, "error": str(exc)}
            print(f"  ERROR: {exc}")
        rows.append(row)

    multi = pd.DataFrame(rows)
    multi_path = out_root / "multi_asset_summary.csv"
    multi.to_csv(multi_path, index=False, encoding="utf-8-sig")
    with open(out_root / "policy1_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    print("\n[DONE]")
    print(f"- {multi_path}")


if __name__ == "__main__":
    main()
