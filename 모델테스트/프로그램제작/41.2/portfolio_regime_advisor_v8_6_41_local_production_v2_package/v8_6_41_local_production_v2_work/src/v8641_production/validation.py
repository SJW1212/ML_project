"""Validation checks for API/UI payload generation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ProductionConfig
from .repository import DataRepository
from .schemas import AllocationRow, AssetData, PortfolioTotals, ValidationCheck, to_plain_dict


class Validator:
    def __init__(self, config: ProductionConfig):
        self.config = config

    def validate(
        self,
        assets: Dict[str, AssetData],
        allocations: List[AllocationRow],
        totals: PortfolioTotals,
        *,
        allocation_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        required = DataRepository.REQUIRED_COLUMNS
        date_ranges: Dict[str, tuple[str, str]] = {}
        for ticker, asset in assets.items():
            df = asset.predictions
            checks.append(self._check(f"{ticker}_rows", len(df) > 0, f"rows={len(df)}"))
            checks.append(self._check(f"{ticker}_date_monotonic", df["Date"].is_monotonic_increasing, "Date is sorted ascending"))
            checks.append(self._check(f"{ticker}_no_duplicate_dates", not df["Date"].duplicated().any(), "No duplicate dates"))
            missing = sorted(required - set(df.columns))
            checks.append(self._check(f"{ticker}_required_columns", not missing, f"missing={missing}"))
            for col in ["prob_high_vol", "prob_normal", "prob_overall_risk", "prob_up_strengthening_score", "prob_down_strengthening_score"]:
                s = pd.to_numeric(df[col], errors="coerce")
                ok = bool(((s >= 0.0) & (s <= 1.0)).all())
                checks.append(self._check(f"{ticker}_{col}_range", ok, "0 <= probability <= 1"))
            latest = df.iloc[-1]
            wsum = float(latest["stock_weight"] + latest["bond_weight"] + latest["cash_weight"])
            checks.append(self._check(f"{ticker}_latest_weight_sum", abs(wsum - 1.0) <= 1e-6, f"sum={wsum:.8f}"))
            if len(df) > 0:
                d = pd.to_datetime(df["Date"], errors="coerce").dropna()
                if not d.empty:
                    date_ranges[ticker] = (str(d.min().date()), str(d.max().date()))

        checks.extend(self._date_range_checks(date_ranges, assets))

        total_sum = totals.portfolio_stock_weight + totals.portfolio_bond_weight + totals.portfolio_cash_weight
        checks.append(self._check("portfolio_total_weight_sum", abs(total_sum - 1.0) <= 1e-6, f"stock+bond+cash={total_sum:.8f}"))
        cap_sum = sum(row.asset_capital_weight for row in allocations)
        checks.append(self._check("asset_capital_weight_sum", abs(cap_sum - 1.0) <= 1e-6, f"sum={cap_sum:.8f}"))

        if allocation_diagnostics:
            fallback = allocation_diagnostics.get("three_way_fallback_tickers") or []
            if fallback:
                checks.append(self._warn(
                    "allocation_three_way_fallback_used",
                    f"all-cash fallback used for tickers={fallback}",
                ))
            vol_meta = allocation_diagnostics.get("inverse_vol_metadata") or {}
            fallback_vol = [t for t, m in vol_meta.items() if m.get("vol_source") != self.config.inverse_vol_column]
            if self.config.capital_mode == "inverse_vol" and fallback_vol:
                checks.append(self._warn(
                    "inverse_vol_fallback_used",
                    f"volatility fallback used for tickers={fallback_vol}; metadata={vol_meta}",
                ))
        return checks

    def _date_range_checks(self, date_ranges: Dict[str, tuple[str, str]], assets: Dict[str, AssetData]) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        if len(date_ranges) <= 1:
            return checks
        starts = {v[0] for v in date_ranges.values()}
        ends = {v[1] for v in date_ranges.values()}
        if len(starts) > 1 or len(ends) > 1:
            checks.append(self._warn(
                "asset_date_range_mismatch",
                f"prediction date ranges differ by asset: {date_ranges}; portfolio returns use outer join and fill missing asset returns with 0.0",
            ))
        # Common range diagnostic: this is not a failure because local mode allows
        # partial history, but it prevents hidden performance cutoffs.
        latest_start = max(v[0] for v in date_ranges.values())
        earliest_end = min(v[1] for v in date_ranges.values())
        if latest_start > earliest_end:
            checks.append(self._warn(
                "asset_common_range_empty",
                f"no common date range across all assets: {date_ranges}",
            ))
        return checks

    @staticmethod
    def as_ui_list(checks: List[ValidationCheck]) -> List[dict]:
        return [to_plain_dict(check) for check in checks]

    @staticmethod
    def _check(name: str, ok: bool, detail: str) -> ValidationCheck:
        return ValidationCheck(check_name=name, status="PASS" if ok else "FAIL", detail=detail)

    @staticmethod
    def _warn(name: str, detail: str) -> ValidationCheck:
        return ValidationCheck(check_name=name, status="WARN", detail=detail)
