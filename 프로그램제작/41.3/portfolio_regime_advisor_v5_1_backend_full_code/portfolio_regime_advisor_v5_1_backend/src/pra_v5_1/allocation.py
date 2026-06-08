from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .schemas import PortfolioAssetInput, PortfolioEvaluateRequest
from .utils import normalize_ticker, safe_float


@dataclass
class AllocationRow:
    ticker: str
    name: str | None
    asset_type: str
    current_weight: float
    model_stock_weight: float
    model_bond_weight: float
    model_cash_weight: float
    recommended_asset_weight: float
    defensive_to_bond: float
    defensive_to_cash: float
    action: str


class PortfolioAllocationService:
    def _initial_weights(self, request: PortfolioEvaluateRequest, risk_assets: List[PortfolioAssetInput]) -> Dict[str, float]:
        mode = request.settings.capital_mode
        if not risk_assets:
            return {}
        if mode == "custom":
            raw = {normalize_ticker(k): float(v) for k, v in request.settings.custom_weights.items()}
            s = sum(max(0.0, raw.get(a.ticker, 0.0)) for a in risk_assets)
            if s <= 0:
                raise ValueError("custom_weights must contain positive weights for risk assets")
            return {a.ticker: max(0.0, raw.get(a.ticker, 0.0)) / s for a in risk_assets}
        if mode == "equal":
            w = 1.0 / len(risk_assets)
            return {a.ticker: w for a in risk_assets}
        # current_weight and inverse_vol fallback to current_weight for local real-portfolio analysis.
        raw_vals = {a.ticker: safe_float(a.current_weight, 0.0) for a in risk_assets}
        s = sum(raw_vals.values())
        if s <= 0:
            w = 1.0 / len(risk_assets)
            return {a.ticker: w for a in risk_assets}
        return {t: v / s for t, v in raw_vals.items()}

    def allocate(self, request: PortfolioEvaluateRequest, latest_predictions: Dict[str, Dict]) -> Dict:
        enabled = [a for a in request.portfolio if a.enabled]
        risk_assets = [a for a in enabled if a.is_risk_asset]
        defensive_bond_current = sum(safe_float(a.current_weight, 0.0) for a in enabled if a.is_bond_bucket or a.is_defensive_etf)
        defensive_cash_current = sum(safe_float(a.current_weight, 0.0) for a in enabled if a.is_cash_bucket)
        total_current = sum(safe_float(a.current_weight, 0.0) for a in enabled if a.current_weight is not None)
        if total_current <= 0:
            # if no current weights, allocate only across risk assets initially.
            risk_base_total = 1.0
            defensive_bond_current = 0.0
            defensive_cash_current = 0.0
        else:
            risk_base_total = sum(safe_float(a.current_weight, 0.0) for a in risk_assets) / total_current
            defensive_bond_current /= total_current
            defensive_cash_current /= total_current
        risk_internal_weights = self._initial_weights(request, risk_assets)
        rows: List[AllocationRow] = []
        total_asset = 0.0
        total_bond = defensive_bond_current
        total_cash = defensive_cash_current
        max_asset = request.settings.max_asset_weight
        min_cash = request.settings.min_cash_weight
        for a in risk_assets:
            pred = latest_predictions.get(a.ticker, {})
            sw = min(max(safe_float(pred.get("stock_weight"), 0.82), 0.0), 1.0)
            bw = min(max(safe_float(pred.get("bond_weight"), 0.117), 0.0), 1.0)
            cw = min(max(safe_float(pred.get("cash_weight"), 0.063), 0.0), 1.0)
            s = sw + bw + cw
            if s <= 0:
                sw, bw, cw = 0.0, 0.65, 0.35
            else:
                sw, bw, cw = sw / s, bw / s, cw / s
            base_abs = risk_base_total * risk_internal_weights.get(a.ticker, 0.0)
            rec_asset = min(base_abs * sw, max_asset)
            to_bond = base_abs * bw
            to_cash = base_abs * cw
            total_asset += rec_asset
            total_bond += to_bond
            total_cash += to_cash
            curr = safe_float(a.current_weight, base_abs) / (total_current if total_current > 0 else 1.0)
            diff = rec_asset - curr
            if abs(diff) < 0.01:
                action = "HOLD"
            elif diff > 0.03:
                action = "INCREASE"
            elif diff > 0:
                action = "SLIGHT_INCREASE"
            elif diff < -0.03:
                action = "REDUCE"
            else:
                action = "SLIGHT_REDUCE"
            rows.append(AllocationRow(a.ticker, a.name, a.asset_type, curr, sw, bw, cw, rec_asset, to_bond, to_cash, action))
        if total_cash < min_cash:
            need = min_cash - total_cash
            # take from bond first, then assets proportionally.
            take_bond = min(total_bond, need)
            total_bond -= take_bond
            total_cash += take_bond
            remain = need - take_bond
            if remain > 0 and total_asset > 0:
                scale = max(0.0, (total_asset - remain) / total_asset)
                for r in rows:
                    r.recommended_asset_weight *= scale
                total_asset *= scale
                total_cash += remain
        total = total_asset + total_bond + total_cash
        if total <= 0:
            total_asset, total_bond, total_cash = 0.0, 0.0, 1.0
            total = 1.0
        payload_rows = []
        for r in rows:
            payload_rows.append({
                "ticker": r.ticker,
                "name": r.name,
                "asset_type": r.asset_type,
                "current_weight": round(r.current_weight, 6),
                "model_stock_weight": round(r.model_stock_weight, 6),
                "model_bond_weight": round(r.model_bond_weight, 6),
                "model_cash_weight": round(r.model_cash_weight, 6),
                "recommended_weight": round(r.recommended_asset_weight / total, 6),
                "defensive_to_bond": round(r.defensive_to_bond / total, 6),
                "defensive_to_cash": round(r.defensive_to_cash / total, 6),
                "action": r.action,
            })
        return {
            "allocation_rows": payload_rows,
            "portfolio_totals": {
                "stock_weight": round(total_asset / total, 6),
                "bond_weight": round(total_bond / total, 6),
                "cash_weight": round(total_cash / total, 6),
            },
            "defensive_buckets_input": {"bond_weight": defensive_bond_current, "cash_weight": defensive_cash_current},
        }
