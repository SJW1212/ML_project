from __future__ import annotations

from pathlib import Path

from v8641_production.config import ProductionConfig
from v8641_production.service import ProductionService


def main() -> None:
    weights = {"QQQ": 0.25, "SPY": 0.20, "AAPL": 0.15, "SOXX": 0.20, "NVDA": 0.20}
    config = ProductionConfig(
        input_dir=Path("/mnt/data"),
        out_dir=Path("/tmp/v8_6_41_custom_weight_test_ops"),
        assets=["QQQ", "SPY", "AAPL", "SOXX", "NVDA"],
        allocation_source="executed",
        capital_mode="custom",
        custom_capital_weights=weights,
        export_json=False,
        export_csv=False,
        export_markdown=False,
        make_zip=False,
    )
    payload = ProductionService(config).build_dashboard_payload()
    rows = payload["portfolio_allocation"]
    got = {row["ticker"]: row["asset_capital_weight"] for row in rows}
    for ticker, expected in weights.items():
        assert abs(got[ticker] - expected) < 1e-12, (ticker, got[ticker], expected)
    totals = payload["portfolio_totals"]
    s = totals["portfolio_stock_weight"] + totals["portfolio_bond_weight"] + totals["portfolio_cash_weight"]
    assert abs(s - 1.0) < 1e-8, totals
    print("v8.6.41 custom weight test passed")
    print("portfolio_totals=", totals)


if __name__ == "__main__":
    main()
