from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core.config import get_settings
from ..dependencies import (
    get_allocation_service,
    get_inference_service,
    get_market_data_repository,
    get_model_loader,
    get_model_registry,
)
from ..schemas import InferenceRequest

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/active")
def active_model():
    active = get_model_registry().active()
    settings = get_settings()
    return {
        "active_model": active.get("active_model_version", settings.default_model_version),
        "mode": active.get("active_mode", settings.default_model_mode),
        "available_horizons": ["5D", "10D", "20D"],
        "supported_tickers": settings.default_assets,
        "model": active.get("model"),
    }


@router.get("/registry")
def model_registry():
    return {"models": get_model_registry().list_models()}


@router.get("/artifact-inventory")
def artifact_inventory():
    return get_model_loader().inventory()


@router.get("/runtime-status")
def runtime_status(
    assets: str = "QQQ,SPY,AAPL,SOXX,NVDA",
    horizons: str = "5D,10D,20D",
    model_version: str | None = None,
):
    settings = get_settings()
    active = get_model_registry().active()
    version = model_version or active.get("active_model_version") or settings.default_model_version
    tickers = [a.strip().upper() for a in assets.split(",") if a.strip()]
    horizon_list = [h.strip().upper() for h in horizons.split(",") if h.strip()]
    artifact_status = get_inference_service().artifact_status(version, tickers, horizon_list)
    cache_items = []
    repo = get_market_data_repository()
    for ticker in tickers:
        # auto cache is preferred, but yahoo/kis are included so the UI can guide the user.
        cache_items.append({
            "ticker": ticker,
            "auto": repo.freshness("auto", ticker, "US"),
            "yahoo": repo.freshness("yahoo", ticker, "US"),
            "kis": repo.freshness("kis", ticker, "US"),
        })
    return {
        "model_version": version,
        "active_mode": active.get("active_mode"),
        "runtime_ready": artifact_status.get("complete", False),
        "artifact_status": artifact_status,
        "market_data_cache": cache_items,
        "next_action": "Run POST /training/retrain after collecting OHLCV with POST /market-data/update." if not artifact_status.get("complete") else "Runtime artifacts are complete. You can call POST /models/infer or /dashboard?model_mode=live.",
    }


@router.post("/infer")
def infer(req: InferenceRequest):
    settings = get_settings()
    active = get_model_registry().active()
    version = req.model_version or active.get("active_model_version") or settings.default_model_version
    signals, errors = get_inference_service().infer_from_cache(
        model_version=version,
        tickers=req.tickers,
        horizon=req.horizon,
        provider=req.provider,
        market=req.market,
        repository=get_market_data_repository(),
        horizons_for_model=["5D", "10D", "20D"],
    )
    if signals.empty:
        totals = {"stock": 0.0, "bond": 0.0, "cash": 0.0}
        items = []
    else:
        allocation_df, totals = get_allocation_service().apply(signals, capital_mode="equal")
        items = allocation_df.to_dict("records")
    return {
        "ok": len(errors) == 0 and len(items) > 0,
        "model_version": version,
        "model_mode": "live_inference",
        "horizon": req.horizon,
        "provider": req.provider,
        "market": req.market,
        "portfolio_totals": totals,
        "items": items,
        "errors": errors,
    }


@router.post("/{model_version}/activate")
def activate_model(model_version: str, mode: str = "live_inference", force: bool = False):
    try:
        get_model_registry().activate(model_version, mode=mode, force=force)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "active_model": model_version, "mode": mode, "force": force}
