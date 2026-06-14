from __future__ import annotations

from pathlib import Path

import pandas as pd

from v8641_production.config import ProductionConfig
from v8641_production.performance import PerformanceAnalyzer
from v8641_production.schemas import AllocationRow, AssetData, PortfolioTotals
from v8641_production.validation import Validator


def _df(dates):
    return pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "strategy_return_net": [0.01] * len(dates),
        "prob_high_vol": [0.2] * len(dates),
        "prob_normal": [0.8] * len(dates),
        "prob_overall_risk": [0.2] * len(dates),
        "prob_up_strengthening_score": [0.3] * len(dates),
        "prob_down_strengthening_score": [0.2] * len(dates),
        "stock_weight": [0.8] * len(dates),
        "bond_weight": [0.1] * len(dates),
        "cash_weight": [0.1] * len(dates),
        "stock_next_return": [0.01] * len(dates),
        "signal_stock_weight": [0.8] * len(dates),
        "signal_bond_weight": [0.1] * len(dates),
        "signal_cash_weight": [0.1] * len(dates),
    })


def main() -> None:
    cfg = ProductionConfig(input_dir=Path("."), out_dir=Path("/tmp/v8641_edge"), assets=["AAA", "BBB"])
    assets = {
        "AAA": AssetData("AAA", Path("AAA.csv"), None, _df(["2024-01-01", "2024-01-02", "2024-01-03"]), {}),
        "BBB": AssetData("BBB", Path("BBB.csv"), None, _df(["2024-01-03", "2024-01-04"]), {}),
    }
    perf = PerformanceAnalyzer(cfg).portfolio_daily_returns(assets, {"AAA": 0.5, "BBB": 0.5})
    assert list(pd.to_datetime(perf["Date"]).dt.strftime("%Y-%m-%d")) == [
        "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"
    ]
    allocations = [
        AllocationRow("AAA", "2024-01-03", 0.5, 0.8, 0.1, 0.1, 0.4, 0.05, 0.05, 0.5, "executed", "equal", "NORMAL", "NEUTRAL", "PARTICIPATION", "NORMAL_MONITORING"),
        AllocationRow("BBB", "2024-01-04", 0.5, 0.8, 0.1, 0.1, 0.4, 0.05, 0.05, 0.5, "executed", "equal", "NORMAL", "NEUTRAL", "PARTICIPATION", "NORMAL_MONITORING"),
    ]
    totals = PortfolioTotals("2024-01-03", 0.8, 0.1, 0.1, 1.0, "executed", "equal")
    checks = Validator(cfg).validate(assets, allocations, totals)
    assert any(c.check_name == "asset_date_range_mismatch" and c.status == "WARN" for c in checks)
    print("edge case tests passed")


if __name__ == "__main__":
    main()
