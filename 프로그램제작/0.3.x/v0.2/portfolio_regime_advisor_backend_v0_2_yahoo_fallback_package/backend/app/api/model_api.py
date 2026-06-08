from __future__ import annotations

from fastapi import APIRouter

from ..core.config import get_settings
from ..dependencies import get_model_registry

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


@router.post("/{model_version}/activate")
def activate_model(model_version: str, mode: str = "live_inference"):
    get_model_registry().activate(model_version, mode=mode)
    return {"ok": True, "active_model": model_version, "mode": mode}
