"""FastAPI adapter for the locked v8.6.41 local production baseline.

This adapter intentionally uses prediction-file mode. It does not retrain models
and does not use the v0.3.x live runtime, context-head gate, horizon ensemble, or
soft-family gate experiments.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError(
        "FastAPI is optional. Install fastapi and uvicorn to use api_fastapi.py"
    ) from exc

from .cli import parse_assets, parse_custom_weights
from .config import ProductionConfig
from .constants import (
    CANCELLED_LAYERS,
    DEFAULT_ASSETS,
    EXCLUDED_PRODUCT_SCOPE,
    MODEL_VERSION,
    SOURCE_TAG,
)
from .data_update import DailyMarketDataUpdater, MarketDataCache
from .repository import FileResolver
from .service import ProductionService

LOG_DIR = Path(os.getenv("V8641_LOG_DIR", "storage/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "api_server.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("v8641_production.api")

app = FastAPI(
    title="Portfolio Regime Advisor - v8.6.41 Local Production API",
    version="0.4.1-41-local-production",
    description=(
        "Locked v8.6.41_label_fixed prediction-file API. "
        "No v0.3.x runtime model, context gate, horizon ensemble, or soft-family gate is used. "
        "Local-only scope: no DB, no user accounts, no notifications, no order execution, no realtime streaming."
    ),
)


class CustomWeightsRequest(BaseModel):
    assets: list[str] = Field(default_factory=lambda: parse_assets(DEFAULT_ASSETS))
    weights: Dict[str, float]
    allocation_source: Literal["executed", "signal"] = "executed"
    holdout_start: str = "2024-01-01"


class DailyUpdateRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: parse_assets(DEFAULT_ASSETS))
    provider: Literal["yahoo", "kis"] = "yahoo"
    start: Optional[str] = None
    end: Optional[str] = None
    force: bool = False


class DashboardRequest(BaseModel):
    assets: list[str] = Field(default_factory=lambda: parse_assets(DEFAULT_ASSETS))
    allocation_source: Literal["executed", "signal"] = "executed"
    capital_mode: Literal["equal", "custom", "inverse_vol"] = "equal"
    custom_weights: Dict[str, float] = Field(default_factory=dict)
    holdout_start: str = "2024-01-01"


def _load_config_json() -> dict:
    path = Path(os.getenv("V8641_CONFIG_JSON", "local_app_config.json"))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to load config JSON: {path}: {exc}") from exc


def _env_input_dir(config_json: Optional[dict] = None) -> Path:
    cfg = config_json if config_json is not None else _load_config_json()
    return Path(os.getenv("V8641_INPUT_DIR", cfg.get("input_dir", ".")))


def _env_out_dir(config_json: Optional[dict] = None) -> Path:
    cfg = config_json if config_json is not None else _load_config_json()
    return Path(os.getenv("V8641_OUT_DIR", cfg.get("out_dir", "v8_6_41_local_ops")))


def _assets_from_query(assets: Optional[str]) -> list[str]:
    cfg = _load_config_json()
    default_assets = cfg.get("assets", parse_assets(os.getenv("V8641_ASSETS", DEFAULT_ASSETS)))
    if isinstance(default_assets, str):
        default_assets = parse_assets(default_assets)
    return parse_assets(assets) if assets else [str(x).upper() for x in default_assets]


def _config(
    *,
    assets: Optional[list[str]] = None,
    allocation_source: Optional[str] = None,
    capital_mode: Optional[str] = None,
    custom_weights: Optional[Dict[str, float]] = None,
    holdout_start: Optional[str] = None,
) -> ProductionConfig:
    cfg_json = _load_config_json()
    data_update = cfg_json.get("data_update", {}) if isinstance(cfg_json.get("data_update", {}), dict) else {}
    json_assets = cfg_json.get("assets")
    if isinstance(json_assets, str):
        json_assets = parse_assets(json_assets)
    elif isinstance(json_assets, list):
        json_assets = [str(x).upper() for x in json_assets]
    else:
        json_assets = parse_assets(os.getenv("V8641_ASSETS", DEFAULT_ASSETS))

    return ProductionConfig(
        input_dir=_env_input_dir(cfg_json),
        out_dir=_env_out_dir(cfg_json),
        assets=assets or json_assets,
        holdout_start=holdout_start or os.getenv("V8641_HOLDOUT_START", cfg_json.get("holdout_start", "2024-01-01")),
        allocation_source=allocation_source or os.getenv("V8641_ALLOCATION_SOURCE", cfg_json.get("allocation_source", "executed")),
        capital_mode=capital_mode or os.getenv("V8641_CAPITAL_MODE", cfg_json.get("capital_mode", "equal")),
        custom_capital_weights=custom_weights
        if custom_weights is not None
        else parse_custom_weights(os.getenv("V8641_CUSTOM_WEIGHTS", "")),
        initial_capital=float(os.getenv("V8641_INITIAL_CAPITAL", cfg_json.get("initial_capital", 100_000_000.0))),
        risk_free_rate=float(os.getenv("V8641_RISK_FREE_RATE", cfg_json.get("risk_free_rate", 0.0))),
        export_json=False,
        export_csv=False,
        export_markdown=False,
        make_zip=False,
        cache_dir=Path(os.getenv("V8641_CACHE_DIR", cfg_json.get("cache_dir", "storage/market_cache"))),
        update_provider=os.getenv("V8641_UPDATE_PROVIDER", data_update.get("provider", "yahoo")),
        daily_update_hour_kst=int(os.getenv("V8641_DAILY_UPDATE_HOUR_KST", data_update.get("recommended_hour_kst", 8))),
        daily_freshness_tolerance_days=int(os.getenv("V8641_DAILY_FRESHNESS_TOLERANCE_DAYS", data_update.get("freshness_tolerance_days", 2))),
        default_update_start=os.getenv("V8641_DEFAULT_UPDATE_START", data_update.get("default_update_start", "2013-01-01")),
    )


def _build_payload(config: ProductionConfig) -> dict:
    try:
        return ProductionService(config).build_dashboard_payload()
    except FileNotFoundError as exc:
        logger.exception("prediction file not found")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        logger.exception("invalid request/configuration")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health():
    cfg_json = _load_config_json()
    return {
        "ok": True,
        "api_version": "0.4.1-41-local-production",
        "model_version": MODEL_VERSION,
        "model_mode": "prediction_file",
        "source_tag": SOURCE_TAG,
        "input_dir": str(_env_input_dir(cfg_json)),
        "out_dir": str(_env_out_dir(cfg_json)),
        "log_dir": str(LOG_DIR),
        "cancelled_layers": CANCELLED_LAYERS,
        "excluded_product_scope": EXCLUDED_PRODUCT_SCOPE,
        "local_scope": {
            "database": False,
            "user_accounts": False,
            "notifications": False,
            "pixso_mapping": False,
            "order_execution": False,
            "realtime_streaming": False,
            "daily_data_update": True,
        },
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
    allocation_source: Literal["executed", "signal"] = Query(default="executed"),
    capital_mode: Literal["equal", "inverse_vol"] = Query(default="equal"),
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
    allocation_source: Literal["executed", "signal"] = Query(default="executed"),
    capital_mode: Literal["equal", "inverse_vol"] = Query(default="equal"),
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
    capital_mode: Literal["equal", "inverse_vol"] = Query(default="equal"),
):
    payload = _build_payload(_config(assets=_assets_from_query(assets), capital_mode=capital_mode))
    return {
        "model_version": payload["model_version"],
        "as_of_date": payload["as_of_date"],
        "capital_mode": payload["capital_mode"],
        "performance_summary": payload["performance_summary"],
        "charts": payload["charts"],
    }


@app.get("/data/freshness")
def data_freshness(
    tickers: Optional[str] = Query(default=None, description="Comma-separated tickers"),
    provider: Literal["yahoo", "kis"] = Query(default="yahoo"),
):
    cfg = _config(assets=_assets_from_query(tickers))
    cache = MarketDataCache(cfg)
    items = cache.freshness(cfg.assets, provider)
    return {
        "ok": True,
        "model_version": MODEL_VERSION,
        "model_mode": "prediction_file",
        "update_mode": "daily_cache_only",
        "provider": provider,
        "cache_dir": str(cfg.cache_dir),
        "freshness": [item.__dict__ for item in items],
        "scope_note": "No realtime streaming, no orders, no DB, no user-account storage, no notifications.",
    }


@app.post("/data/update-daily")
def data_update_daily(req: DailyUpdateRequest):
    cfg = _config(assets=[ticker.upper() for ticker in req.tickers])
    updater = DailyMarketDataUpdater(cfg)
    status = updater.update_daily(
        cfg.assets,
        provider=req.provider,
        start=req.start,
        end=req.end,
        force=req.force,
    )
    logger.info("daily update status ok=%s updated=%s skipped=%s errors=%s", status.ok, status.updated, status.skipped, status.errors)
    return status.__dict__


@app.get("/data/update-status")
def data_update_status():
    cfg = _config()
    cache = MarketDataCache(cfg)
    return cache.read_status()


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
        "excluded_product_scope": EXCLUDED_PRODUCT_SCOPE,
        "data_update_policy": {
            "mode": "daily_cache_only",
            "default_hour_kst": 8,
            "providers": ["yahoo", "kis_reserved"],
            "order_api": "excluded",
            "prediction_file_sync": "manual/separate; OHLCV cache update does not regenerate v8.6.41 prediction files",
        },
    }
