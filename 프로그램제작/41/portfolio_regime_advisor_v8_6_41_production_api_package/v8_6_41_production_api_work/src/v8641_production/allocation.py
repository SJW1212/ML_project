"""Allocation policy layer.

Important: this layer does not change the v8.6.41 model decision. It only
combines per-ticker native allocations into a portfolio-level allocation.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ProductionConfig
from .schemas import AllocationRow, AssetData, PortfolioTotals, SignalSnapshot, to_plain_dict
from .utils import MathUtils


class AllocationPolicy:
    """Keeps native v8.6.41 per-asset allocation and controls capital distribution."""

    def __init__(self, config: ProductionConfig):
        self.config = config

    def capital_weights(self, assets: Dict[str, AssetData]) -> Dict[str, float]:
        if self.config.capital_mode == "custom":
            return MathUtils.normalize_weights(self.config.custom_capital_weights, self.config.assets)
        if self.config.capital_mode == "inverse_vol":
            return self._inverse_vol_weights(assets)
        return {ticker: 1.0 / len(self.config.assets) for ticker in self.config.assets}

    def _inverse_vol_weights(self, assets: Dict[str, AssetData]) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        for ticker, asset in assets.items():
            df = asset.predictions
            vol = np.nan
            if self.config.inverse_vol_column in df.columns:
                vol = MathUtils.safe_float(df[self.config.inverse_vol_column].iloc[-1], np.nan)
            if not np.isfinite(vol) or vol <= 0:
                ret = pd.to_numeric(df["stock_next_return"], errors="coerce").tail(60).dropna()
                vol = float(ret.std(ddof=0) * math.sqrt(252.0)) if len(ret) >= 20 else np.nan
            raw[ticker] = 1.0 / max(float(vol) if np.isfinite(vol) else 1.0, self.config.inverse_vol_floor)
        return MathUtils.normalize_weights(raw, self.config.assets)

    def build_allocation_rows(
        self,
        snapshots: Dict[str, SignalSnapshot],
        capital_weights: Dict[str, float],
    ) -> List[AllocationRow]:
        rows: List[AllocationRow] = []
        for ticker in self.config.assets:
            snapshot = snapshots[ticker]
            if self.config.allocation_source == "signal":
                stock = snapshot.signal_stock_weight
                bond = snapshot.signal_bond_weight
                cash = snapshot.signal_cash_weight
            else:
                stock = snapshot.executed_stock_weight
                bond = snapshot.executed_bond_weight
                cash = snapshot.executed_cash_weight
            stock, bond, cash = self._normalize_three_way(stock, bond, cash)
            cap_w = capital_weights[ticker]
            rows.append(
                AllocationRow(
                    ticker=ticker,
                    date=snapshot.date,
                    asset_capital_weight=cap_w,
                    asset_stock_weight=stock,
                    asset_bond_weight=bond,
                    asset_cash_weight=cash,
                    portfolio_stock_weight=cap_w * stock,
                    portfolio_bond_weight=cap_w * bond,
                    portfolio_cash_weight=cap_w * cash,
                    portfolio_total_weight=cap_w,
                    allocation_source=self.config.allocation_source,
                    capital_mode=self.config.capital_mode,
                    risk_class=snapshot.risk_class,
                    direction_class=snapshot.direction_class,
                    allocation_class=snapshot.allocation_class,
                    monitoring_note=snapshot.monitoring_note,
                )
            )
        return rows

    @staticmethod
    def totals(rows: List[AllocationRow]) -> PortfolioTotals:
        if not rows:
            return PortfolioTotals("", 0.0, 0.0, 0.0, 0.0, "", "")
        return PortfolioTotals(
            date=rows[0].date,
            portfolio_stock_weight=float(sum(row.portfolio_stock_weight for row in rows)),
            portfolio_bond_weight=float(sum(row.portfolio_bond_weight for row in rows)),
            portfolio_cash_weight=float(sum(row.portfolio_cash_weight for row in rows)),
            portfolio_total_weight=float(sum(row.portfolio_total_weight for row in rows)),
            allocation_source=rows[0].allocation_source,
            capital_mode=rows[0].capital_mode,
        )

    @staticmethod
    def as_ui_list(rows: List[AllocationRow]) -> List[dict]:
        return [to_plain_dict(row) for row in rows]

    @staticmethod
    def _normalize_three_way(stock: float, bond: float, cash: float) -> tuple[float, float, float]:
        values = [stock, bond, cash]
        clean = [0.0 if not np.isfinite(v) else max(0.0, float(v)) for v in values]
        total = sum(clean)
        if total <= 0:
            return 0.0, 0.0, 1.0
        return clean[0] / total, clean[1] / total, clean[2] / total
