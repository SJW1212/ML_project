"""Optional FastAPI adapter for UI integration.

Install FastAPI separately if you want to run this as an API:
    pip install fastapi uvicorn
    uvicorn v8641_production.api_fastapi:app --reload

This module is optional. The core pipeline does not require FastAPI.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from fastapi import FastAPI
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError("FastAPI is optional. Install fastapi and uvicorn to use api_fastapi.py") from exc

from .cli import parse_assets, parse_custom_weights
from .config import ProductionConfig
from .constants import DEFAULT_ASSETS
from .service import ProductionService

app = FastAPI(title="v8.6.41 Label Fixed Production API")


def _config() -> ProductionConfig:
    return ProductionConfig(
        input_dir=Path(os.getenv("V8641_INPUT_DIR", ".")),
        out_dir=Path(os.getenv("V8641_OUT_DIR", "v8_6_41_ui_modular_ops")),
        assets=parse_assets(os.getenv("V8641_ASSETS", DEFAULT_ASSETS)),
        holdout_start=os.getenv("V8641_HOLDOUT_START", "2024-01-01"),
        allocation_source=os.getenv("V8641_ALLOCATION_SOURCE", "executed"),
        capital_mode=os.getenv("V8641_CAPITAL_MODE", "equal"),
        custom_capital_weights=parse_custom_weights(os.getenv("V8641_CUSTOM_WEIGHTS", "")),
        export_json=False,
        export_csv=False,
        export_markdown=False,
        make_zip=False,
    )


@app.get("/dashboard")
def dashboard():
    return ProductionService(_config()).build_dashboard_payload()


@app.get("/latest")
def latest():
    payload = ProductionService(_config()).build_dashboard_payload()
    return {
        "model_version": payload["model_version"],
        "as_of_date": payload["as_of_date"],
        "latest_signals": payload["latest_signals"],
        "portfolio_totals": payload["portfolio_totals"],
        "portfolio_allocation": payload["portfolio_allocation"],
        "validation": payload["validation"],
    }
