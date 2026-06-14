from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


class AllocationService:
    """Portfolio-level allocation layer.

    The locked v8.6.41 native asset weights are preserved. Program-level portfolio
    adjustment happens through ticker capital weights, not through loss guard/state overrides.
    """

    def capital_weights(self, signals: pd.DataFrame, mode: str = "equal", custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        tickers = [str(t).upper() for t in signals["ticker"].tolist()]
        if not tickers:
            return {}
        if mode == "equal":
            w = 1.0 / len(tickers)
            return {t: w for t in tickers}
        if mode == "custom":
            if not custom_weights:
                raise ValueError("custom_weights required for custom capital mode")
            return {t: float(custom_weights[t]) for t in tickers}
        if mode == "inverse_vol":
            # Use realized_vol_60 if available, otherwise equal fallback.
            vols = []
            for _, row in signals.iterrows():
                vol = row.get("realized_vol_60", row.get("ctx_realized_vol_60", np.nan))
                try:
                    vol = float(vol)
                except Exception:
                    vol = np.nan
                vols.append(vol if np.isfinite(vol) and vol > 0 else np.nan)
            if all(np.isnan(v) for v in vols):
                w = 1.0 / len(tickers)
                return {t: w for t in tickers}
            median_vol = float(np.nanmedian(vols))
            vols = [v if np.isfinite(v) and v > 0 else median_vol for v in vols]
            inv = np.array([1.0 / v for v in vols], dtype=float)
            inv = inv / inv.sum()
            return {t: float(w) for t, w in zip(tickers, inv)}
        raise ValueError(f"Unknown capital mode: {mode}")

    def apply(self, signals: pd.DataFrame, capital_mode: str = "equal", custom_weights: Optional[Dict[str, float]] = None) -> tuple[pd.DataFrame, dict]:
        df = signals.copy()
        cap = self.capital_weights(df, capital_mode, custom_weights)
        df["asset_capital_weight"] = df["ticker"].map(cap)
        for col in ["stock_weight", "bond_weight", "cash_weight"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        df["portfolio_stock_contribution"] = df["asset_capital_weight"] * df["stock_weight"]
        df["portfolio_bond_contribution"] = df["asset_capital_weight"] * df["bond_weight"]
        df["portfolio_cash_contribution"] = df["asset_capital_weight"] * df["cash_weight"]
        totals = {
            "stock": float(df["portfolio_stock_contribution"].sum()),
            "bond": float(df["portfolio_bond_contribution"].sum()),
            "cash": float(df["portfolio_cash_contribution"].sum()),
        }
        s = sum(totals.values())
        if s > 0:
            totals = {k: v / s for k, v in totals.items()}
        elif len(df) > 0:
            # Invalid per-asset weights should not surface as 0/0/0 in the UI.
            # Fall back to neutral cash allocation and mark rows for diagnosis.
            totals = {"stock": 0.0, "bond": 0.0, "cash": 1.0}
            df["allocation_warning"] = "ZERO_TOTAL_WEIGHT_FALLBACK_TO_CASH"
        return df, totals
