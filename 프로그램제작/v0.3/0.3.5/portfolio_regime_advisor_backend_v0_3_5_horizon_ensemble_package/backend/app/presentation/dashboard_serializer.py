from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..schemas import ChartPayload, DashboardPayload, PerformanceItem, PortfolioTotals, SignalItem
from .insight_generator import InsightGenerator


class DashboardSerializer:
    def __init__(self, insight_generator: InsightGenerator):
        self.insight_generator = insight_generator

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    def to_payload(
        self,
        *,
        signals: pd.DataFrame,
        portfolio_totals: Dict[str, float],
        performance_summary: List[dict],
        validation: Dict[str, Any],
        model_version: str,
        model_mode: str,
        user_mode: str,
        preset: Optional[str],
        horizon: str,
        data_source: Dict[str, Any],
    ) -> DashboardPayload:
        latest_items = []
        if not signals.empty:
            for _, row in signals.iterrows():
                latest_items.append(SignalItem(
                    ticker=str(row.get("ticker", "")),
                    date=str(pd.to_datetime(row.get("Date")).date()) if row.get("Date") is not None else "",
                    risk_class=str(row.get("risk_class", row.get("pred_risk", "UNKNOWN"))),
                    direction_class=str(row.get("direction_class", row.get("pred_direction", "UNKNOWN"))),
                    allocation_class=str(row.get("allocation_class", row.get("allocation_regime", "UNKNOWN"))),
                    prob_normal=self._safe_float(row.get("prob_normal")),
                    prob_high_vol=self._safe_float(row.get("prob_high_vol")),
                    prob_overall_risk=self._safe_float(row.get("prob_overall_risk")),
                    prob_up_strengthening_score=self._safe_float(row.get("prob_up_strengthening_score")),
                    prob_down_strengthening_score=self._safe_float(row.get("prob_down_strengthening_score")),
                    selected_horizon=str(row.get("selected_horizon", horizon)),
                    selected_prob_high_vol=self._safe_float(row.get("selected_prob_high_vol")),
                    selected_prob_up_strengthening=self._safe_float(row.get("selected_prob_up_strengthening")),
                    selected_prob_down_strengthening=self._safe_float(row.get("selected_prob_down_strengthening")),
                    stock_weight=self._safe_float(row.get("stock_weight")),
                    bond_weight=self._safe_float(row.get("bond_weight")),
                    cash_weight=self._safe_float(row.get("cash_weight")),
                    recommended_stock_weight=self._safe_float(row.get("recommended_stock_weight", row.get("stock_weight"))),
                    executed_stock_weight=self._safe_float(row.get("executed_stock_weight", row.get("stock_weight"))),
                    comment=str(row.get("comment", "")),
                    warnings=list(row.get("warnings", [])) if isinstance(row.get("warnings", []), list) else [],
                ))
        perf_items = [PerformanceItem(**p) for p in performance_summary]
        as_of_date = None
        if not signals.empty:
            as_of_date = str(pd.to_datetime(signals["Date"]).max().date())
        insights = self.insight_generator.generate(signals, portfolio_totals)
        return DashboardPayload(
            as_of_date=as_of_date,
            model_version=model_version,
            model_mode=model_mode,
            user_mode=user_mode,
            preset=preset,
            horizon=horizon,
            data_source=data_source,
            portfolio_totals=PortfolioTotals(**portfolio_totals),
            latest_signals=latest_items,
            performance_summary=perf_items,
            charts=ChartPayload(),
            validation=validation,
            insights=insights,
        )
