"""Validation checks for API/UI payload generation."""
from __future__ import annotations

from typing import Dict, List

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
    ) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        required = DataRepository.REQUIRED_COLUMNS
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
        total_sum = totals.portfolio_stock_weight + totals.portfolio_bond_weight + totals.portfolio_cash_weight
        checks.append(self._check("portfolio_total_weight_sum", abs(total_sum - 1.0) <= 1e-6, f"stock+bond+cash={total_sum:.8f}"))
        cap_sum = sum(row.asset_capital_weight for row in allocations)
        checks.append(self._check("asset_capital_weight_sum", abs(cap_sum - 1.0) <= 1e-6, f"sum={cap_sum:.8f}"))
        return checks

    @staticmethod
    def as_ui_list(checks: List[ValidationCheck]) -> List[dict]:
        return [to_plain_dict(check) for check in checks]

    @staticmethod
    def _check(name: str, ok: bool, detail: str) -> ValidationCheck:
        return ValidationCheck(check_name=name, status="PASS" if ok else "FAIL", detail=detail)
