from __future__ import annotations

import os
from pathlib import Path

from v8641_production.config import ProductionConfig
from v8641_production.service import ProductionService


def main() -> None:
    input_dir = Path(os.environ.get("V8641_TEST_INPUT_DIR", "/mnt/data"))
    config = ProductionConfig(
        input_dir=input_dir,
        out_dir=Path("/tmp/v8_6_41_api_test_ops"),
        assets=["QQQ", "SPY", "AAPL", "SOXX", "NVDA"],
        allocation_source="executed",
        capital_mode="equal",
        export_json=False,
        export_csv=False,
        export_markdown=False,
        make_zip=False,
    )
    payload = ProductionService(config).build_dashboard_payload()
    assert payload["model_version"] == "v8.6.41_model_label_fixed"
    assert payload["capital_mode"] == "equal"
    assert len(payload["latest_signals"]) == 5
    assert len(payload["portfolio_allocation"]) == 5
    totals = payload["portfolio_totals"]
    s = totals["portfolio_stock_weight"] + totals["portfolio_bond_weight"] + totals["portfolio_cash_weight"]
    assert abs(s - 1.0) < 1e-8, totals
    assert payload["validation"]["fail_count"] == 0, payload["validation"]
    print("v8.6.41 production API payload test passed")
    print("as_of_date=", payload["as_of_date"])
    print("portfolio_totals=", totals)


if __name__ == "__main__":
    main()
