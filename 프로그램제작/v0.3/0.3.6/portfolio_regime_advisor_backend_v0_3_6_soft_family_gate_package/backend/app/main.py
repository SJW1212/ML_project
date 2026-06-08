from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import credential_api, dashboard_api, market_data_api, model_api, portfolio_api, settings_api, training_api, validation_api
from .core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_api.router)
app.include_router(settings_api.router)
app.include_router(portfolio_api.router)
app.include_router(model_api.router)
app.include_router(credential_api.router)
app.include_router(market_data_api.router)
app.include_router(training_api.router)
app.include_router(validation_api.router)


@app.get("/health")
def health():
    return {"ok": True, "app": settings.app_name, "version": settings.app_version, "default_model": settings.default_model_version}
