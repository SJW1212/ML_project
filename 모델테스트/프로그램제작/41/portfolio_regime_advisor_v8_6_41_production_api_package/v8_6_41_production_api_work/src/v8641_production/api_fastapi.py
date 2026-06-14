"""FastAPI adapter for the locked v8.6.41 production baseline.

This adapter intentionally uses prediction-file mode. It does not retrain models
and does not use the v0.3.x live runtime, context-head gate, horizon ensemble, or
soft-family gate experiments.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError(
        "FastAPI is optional. Install fastapi and uvicorn to use api_fastapi.py"
    ) from exc

from .cli import parse_assets, parse_custom_weights
from .config import ProductionConfig
from .constants import CANCELLED_LAYERS, DEFAULT_ASSETS, MODEL_VERSION, SOURCE_TAG
from .repository import FileResolver
from .service import ProductionService

app = FastAPI(
    title="Portfolio Regime Advisor - v8.6.41 Production API",
    version="0.4.0-41-production",
    description=(
        "Locked v8.6.41_label_fixed prediction-file API. "
        "No v0.3.x runtime model, context gate, horizon ensemble, or soft-family gate is used."
    ),
)


class CustomWeightsRequest(BaseModel):
    assets: list[str] = Field(default_factory=lambda: parse_assets(DEFAULT_ASSETS))
    weights: Dict[str, float]
    allocation_source: str = "executed"
    holdout_start: str = "2024-01-01"


class DashboardRequest(BaseModel):
    assets: list[str] = Field(default_factory=lambda: parse_assets(DEFAULT_ASSETS))
    allocation_source: str = "executed"
    capital_mode: str = "equal"
    custom_weights: Dict[str, float] = Field(default_factory=dict)
    holdout_start: str = "2024-01-01"


def _env_input_dir() -> Path:
    return Path(os.getenv("V8641_INPUT_DIR", "."))


def _env_out_dir() -> Path:
    return Path(os.getenv("V8641_OUT_DIR", "v8_6_41_ui_modular_ops"))


def _assets_from_query(assets: Optional[str]) -> list[str]:
    return parse_assets(assets or os.getenv("V8641_ASSETS", DEFAULT_ASSETS))


def _config(
    *,
    assets: Optional[list[str]] = None,
    allocation_source: Optional[str] = None,
    capital_mode: Optional[str] = None,
    custom_weights: Optional[Dict[str, float]] = None,
    holdout_start: Optional[str] = None,
) -> ProductionConfig:
    return ProductionConfig(
        input_dir=_env_input_dir(),
        out_dir=_env_out_dir(),
        assets=assets or parse_assets(os.getenv("V8641_ASSETS", DEFAULT_ASSETS)),
        holdout_start=holdout_start or os.getenv("V8641_HOLDOUT_START", "2024-01-01"),
        allocation_source=allocation_source or os.getenv("V8641_ALLOCATION_SOURCE", "executed"),
        capital_mode=capital_mode or os.getenv("V8641_CAPITAL_MODE", "equal"),
        custom_capital_weights=custom_weights
        if custom_weights is not None
        else parse_custom_weights(os.getenv("V8641_CUSTOM_WEIGHTS", "")),
        export_json=False,
        export_csv=False,
        export_markdown=False,
        make_zip=False,
    )


def _build_payload(config: ProductionConfig) -> dict:
    try:
        return ProductionService(config).build_dashboard_payload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {
        "ok": True,
        "api_version": "0.4.0-41-production",
        "model_version": MODEL_VERSION,
        "model_mode": "prediction_file",
        "source_tag": SOURCE_TAG,
        "input_dir": str(_env_input_dir()),
        "out_dir": str(_env_out_dir()),
        "cancelled_layers": CANCELLED_LAYERS,
    }


@app.get("/assets")
def assets():
    cfg = _config()
    resolver = FileResolver(cfg)
    items = []
    for ticker in cfg.assets:
        pred = resolver.prediction_file(ticker)
        summary = resolver.summary_file(ticker)
        items.append(
            {
                "ticker": ticker,
                "prediction_file_found": pred is not None,
                "prediction_path": str(pred) if pred else None,
                "summary_file_found": summary is not None,
                "summary_path": str(summary) if summary else None,
            }
        )
    return {"model_version": MODEL_VERSION, "assets": items}


@app.get("/dashboard")
def dashboard(
    assets: Optional[str] = Query(default=None, description="Comma-separated tickers, e.g. QQQ,SPY,AAPL"),
    allocation_source: str = Query(default="executed", pattern="^(executed|signal)$"),
    capital_mode: str = Query(default="equal", pattern="^(equal|inverse_vol)$"),
    holdout_start: str = Query(default="2024-01-01"),
):
    cfg = _config(
        assets=_assets_from_query(assets),
        allocation_source=allocation_source,
        capital_mode=capital_mode,
        holdout_start=holdout_start,
    )
    return _build_payload(cfg)


@app.post("/dashboard")
def dashboard_post(req: DashboardRequest):
    cfg = _config(
        assets=[ticker.upper() for ticker in req.assets],
        allocation_source=req.allocation_source,
        capital_mode=req.capital_mode,
        custom_weights={k.upper(): float(v) for k, v in req.custom_weights.items()},
        holdout_start=req.holdout_start,
    )
    return _build_payload(cfg)


@app.get("/latest")
def latest(
    assets: Optional[str] = Query(default=None, description="Comma-separated tickers"),
    allocation_source: str = Query(default="executed", pattern="^(executed|signal)$"),
    capital_mode: str = Query(default="equal", pattern="^(equal|inverse_vol)$"),
):
    payload = _build_payload(
        _config(
            assets=_assets_from_query(assets),
            allocation_source=allocation_source,
            capital_mode=capital_mode,
        )
    )
    return {
        "model_version": payload["model_version"],
        "model_mode": "prediction_file",
        "as_of_date": payload["as_of_date"],
        "latest_signals": payload["latest_signals"],
        "portfolio_totals": payload["portfolio_totals"],
        "portfolio_allocation": payload["portfolio_allocation"],
        "validation": payload["validation"],
    }


@app.post("/portfolio/custom-weights")
def portfolio_custom_weights(req: CustomWeightsRequest):
    weights = {k.upper(): float(v) for k, v in req.weights.items()}
    cfg = _config(
        assets=[ticker.upper() for ticker in req.assets],
        allocation_source=req.allocation_source,
        capital_mode="custom",
        custom_weights=weights,
        holdout_start=req.holdout_start,
    )
    payload = _build_payload(cfg)
    return {
        "model_version": payload["model_version"],
        "model_mode": "prediction_file",
        "as_of_date": payload["as_of_date"],
        "capital_mode": payload["capital_mode"],
        "allocation_source": payload["allocation_source"],
        "portfolio_totals": payload["portfolio_totals"],
        "portfolio_allocation": payload["portfolio_allocation"],
        "validation": payload["validation"],
    }


@app.get("/validation")
def validation(
    assets: Optional[str] = Query(default=None, description="Comma-separated tickers"),
):
    payload = _build_payload(_config(assets=_assets_from_query(assets)))
    return {
        "model_version": payload["model_version"],
        "as_of_date": payload["as_of_date"],
        "validation": payload["validation"],
    }


@app.get("/performance")
def performance(
    assets: Optional[str] = Query(default=None, description="Comma-separated tickers"),
    capital_mode: str = Query(default="equal", pattern="^(equal|inverse_vol)$"),
):
    payload = _build_payload(_config(assets=_assets_from_query(assets), capital_mode=capital_mode))
    return {
        "model_version": payload["model_version"],
        "as_of_date": payload["as_of_date"],
        "capital_mode": payload["capital_mode"],
        "performance_summary": payload["performance_summary"],
        "charts": payload["charts"],
    }


@app.get("/schema")
def schema_info():
    return {
        "model_version": MODEL_VERSION,
        "model_mode": "prediction_file",
        "primary_fields": {
            "probabilities": [
                "prob_normal",
                "prob_high_vol",
                "prob_overall_risk",
                "prob_up_strengthening_5d",
                "prob_up_strengthening_10d",
                "prob_up_strengthening_20d",
                "prob_up_strengthening_score",
                "prob_down_strengthening_5d",
                "prob_down_strengthening_10d",
                "prob_down_strengthening_20d",
                "prob_down_strengthening_score",
            ],
            "weights": [
                "signal_stock_weight",
                "signal_bond_weight",
                "signal_cash_weight",
                "executed_stock_weight",
                "executed_bond_weight",
                "executed_cash_weight",
            ],
            "portfolio": [
                "portfolio_stock_weight",
                "portfolio_bond_weight",
                "portfolio_cash_weight",
            ],
        },
        "excluded_experimental_layers": CANCELLED_LAYERS,
    }
