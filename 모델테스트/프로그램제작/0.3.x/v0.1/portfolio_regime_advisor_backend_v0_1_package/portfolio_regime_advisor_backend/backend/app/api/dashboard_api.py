from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ..core.config import get_settings
from ..core.user_modes import UserMode
from ..dependencies import get_allocation_service, get_parameter_validator, get_prediction_service, get_serializer

router = APIRouter(prefix="", tags=["dashboard"])


@router.get("/dashboard")
def get_dashboard(
    assets: str = Query(default="QQQ,SPY,AAPL,SOXX,NVDA"),
    horizon: str = Query(default="10D"),
    user_mode: UserMode = Query(default=UserMode.GENERAL),
    preset: Optional[str] = Query(default=None),
    capital_mode: str = Query(default="equal"),
):
    tickers = [a.strip().upper() for a in assets.split(",") if a.strip()]
    validator = get_parameter_validator()
    validation_result = validator.validate_settings(
        user_mode=user_mode,
        horizon=horizon,
        assets=tickers,
        capital_mode=capital_mode,
        require_supported_horizon=True,
    )
    service = get_prediction_service()
    signals, load_errors = service.latest_signals(tickers, horizon=horizon)
    allocation_df, totals = get_allocation_service().apply(signals, capital_mode=capital_mode)
    perf = service.performance_summary(allocation_df["ticker"].tolist())
    validation = {
        "ok": validation_result.ok and not load_errors,
        "fail_count": validation_result.fail_count,
        "warning_count": validation_result.warning_count + len(load_errors),
        "messages": validation_result.messages + [{"level": "WARN", "code": "LOAD_ERROR", "message": str(e), "field": "assets"} for e in load_errors],
    }
    settings = get_settings()
    payload = get_serializer().to_payload(
        signals=allocation_df,
        portfolio_totals=totals,
        performance_summary=perf,
        validation=validation,
        model_version=settings.default_model_version,
        model_mode=settings.default_model_mode,
        user_mode=user_mode.value,
        preset=preset,
        horizon=horizon.upper(),
        data_source={"provider": "prediction_file", "status": "loaded", "input_dir": str(settings.input_dir)},
    )
    return payload


@router.get("/latest")
def get_latest(assets: str = Query(default="QQQ,SPY,AAPL,SOXX,NVDA"), horizon: str = Query(default="10D")):
    tickers = [a.strip().upper() for a in assets.split(",") if a.strip()]
    signals, load_errors = get_prediction_service().latest_signals(tickers, horizon=horizon)
    return {
        "horizon": horizon.upper(),
        "load_errors": load_errors,
        "items": signals.to_dict("records"),
    }
