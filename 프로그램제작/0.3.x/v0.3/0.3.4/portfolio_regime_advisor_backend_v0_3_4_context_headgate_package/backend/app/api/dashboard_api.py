from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Query

from ..core.config import get_settings
from ..core.exceptions import DataNotFoundError
from ..core.user_modes import UserMode
from ..dependencies import (
    get_allocation_service,
    get_inference_service,
    get_market_data_repository,
    get_model_registry,
    get_parameter_validator,
    get_prediction_service,
    get_serializer,
)

router = APIRouter(prefix="", tags=["dashboard"])


def _prediction_file_dashboard(tickers: list[str], horizon: str, validation_result, settings):
    service = get_prediction_service()
    try:
        signals, load_errors = service.latest_signals(tickers, horizon=horizon)
        allocation_df, totals = get_allocation_service().apply(signals, capital_mode="equal")
        perf = service.performance_summary(allocation_df["ticker"].tolist())
        data_source = {"provider": "prediction_file", "status": "loaded", "input_dir": str(settings.input_dir)}
        validation = {
            "ok": validation_result.ok and not load_errors,
            "fail_count": validation_result.fail_count,
            "warning_count": validation_result.warning_count + len(load_errors),
            "messages": validation_result.messages + [
                {"level": "WARN", "code": "LOAD_ERROR", "message": str(e), "field": "assets"} for e in load_errors
            ],
        }
    except DataNotFoundError as exc:
        allocation_df = pd.DataFrame()
        totals = {"stock": 0.0, "bond": 0.0, "cash": 0.0}
        perf = []
        data_source = {
            "provider": "prediction_file",
            "status": "missing_predictions",
            "input_dir": str(settings.input_dir),
            "hint": "Set PRA_INPUT_DIR to prediction CSV folder or use bundled storage/predictions sample files.",
        }
        validation = {
            "ok": False,
            "fail_count": validation_result.fail_count + 1,
            "warning_count": validation_result.warning_count,
            "messages": validation_result.messages + [
                {"level": "ERROR", "code": "PREDICTION_FILES_NOT_FOUND", "message": str(exc), "field": "PRA_INPUT_DIR"}
            ],
        }
    return allocation_df, totals, perf, data_source, validation


def _live_dashboard(tickers: list[str], horizon: str, provider: str, market: str, validation_result, settings, model_version: Optional[str] = None):
    registry = get_model_registry().active()
    active_version = model_version or registry.get("active_model_version", settings.default_model_version)
    signals, errors = get_inference_service().infer_from_cache(
        model_version=active_version,
        tickers=tickers,
        horizon=horizon,
        provider=provider,
        market=market,
        repository=get_market_data_repository(),
        horizons_for_model=["5D", "10D", "20D"],
    )
    if signals.empty:
        allocation_df = pd.DataFrame()
        totals = {"stock": 0.0, "bond": 0.0, "cash": 0.0}
    else:
        allocation_df, totals = get_allocation_service().apply(signals, capital_mode="equal")
    perf = get_prediction_service().performance_summary(allocation_df["ticker"].tolist()) if not allocation_df.empty else []
    data_source = {
        "provider": provider,
        "status": "live_inference" if not signals.empty else "live_inference_failed",
        "market": market,
        "cache_dir": str(settings.cache_dir),
        "model_version": active_version,
    }
    validation = {
        "ok": validation_result.ok and not errors and not signals.empty,
        "fail_count": validation_result.fail_count + (1 if signals.empty else 0),
        "warning_count": validation_result.warning_count + len(errors),
        "messages": validation_result.messages + [
            {"level": "WARN", "code": e.get("code", "INFERENCE_ERROR"), "message": e.get("message", str(e)), "field": e.get("ticker")}
            for e in errors
        ],
    }
    return allocation_df, totals, perf, data_source, validation


@router.get("/dashboard")
def get_dashboard(
    assets: str = Query(default="QQQ,SPY,AAPL,SOXX,NVDA"),
    horizon: str = Query(default="10D"),
    user_mode: UserMode = Query(default=UserMode.GENERAL),
    preset: Optional[str] = Query(default=None),
    capital_mode: str = Query(default="equal"),
    model_mode: str = Query(default="prediction_file", description="prediction_file, live, or auto"),
    provider: str = Query(default="auto", description="market data cache provider for live mode"),
    market: str = Query(default="US"),
    model_version: Optional[str] = Query(default=None, description="Candidate/active runtime model version for live or auto mode"),
    allow_partial_live: bool = Query(default=False, description="If false, auto mode falls back to prediction_file when any ticker fails live inference"),
):
    tickers = [a.strip().upper() for a in assets.split(",") if a.strip()]
    horizon = horizon.upper().replace(" ", "")
    model_mode = model_mode.lower().strip()
    provider = provider.lower().strip()
    validator = get_parameter_validator()
    validation_result = validator.validate_settings(
        user_mode=user_mode,
        horizon=horizon,
        assets=tickers,
        capital_mode=capital_mode,
        require_supported_horizon=True,
    )
    settings = get_settings()

    effective_model_mode = "prediction_file"
    effective_model_version = settings.default_model_version
    if model_mode == "live":
        allocation_df, totals, perf, data_source, validation = _live_dashboard(tickers, horizon, provider, market, validation_result, settings, model_version=model_version)
        effective_model_mode = "live_inference"
        effective_model_version = data_source.get("model_version", settings.default_model_version)
    elif model_mode == "auto":
        allocation_df, totals, perf, data_source, validation = _live_dashboard(tickers, horizon, provider, market, validation_result, settings, model_version=model_version)
        effective_model_mode = "live_inference"
        effective_model_version = data_source.get("model_version", settings.default_model_version)
        live_success_count = 0 if allocation_df.empty else int(len(allocation_df))
        partial_live_failure = live_success_count < len(tickers)
        should_fallback = allocation_df.empty or (partial_live_failure and not allow_partial_live)
        if should_fallback:
            # auto mode must keep the UI usable even when artifacts are not trained yet or only partially available.
            allocation_df, totals, perf, pf_source, pf_validation = _prediction_file_dashboard(tickers, horizon, validation_result, settings)
            pf_source["fallback_reason"] = "live inference unavailable or partial; used prediction file mode"
            pf_source["live_success_count"] = live_success_count
            pf_source["requested_ticker_count"] = len(tickers)
            pf_source["live_attempt"] = data_source
            validation = pf_validation
            effective_model_mode = "prediction_file_fallback"
            effective_model_version = settings.default_model_version
            data_source = pf_source
    else:
        allocation_df, totals, perf, data_source, validation = _prediction_file_dashboard(tickers, horizon, validation_result, settings)
        effective_model_mode = "prediction_file"
        effective_model_version = settings.default_model_version

    # Re-apply requested capital mode after source-specific load. This keeps custom modes consistent.
    if not allocation_df.empty and capital_mode != "equal":
        try:
            allocation_df, totals = get_allocation_service().apply(allocation_df, capital_mode=capital_mode)
        except Exception as exc:
            validation["ok"] = False
            validation["fail_count"] = validation.get("fail_count", 0) + 1
            validation.setdefault("messages", []).append({"level": "ERROR", "code": "CAPITAL_MODE_FAILED", "message": str(exc), "field": "capital_mode"})

    payload = get_serializer().to_payload(
        signals=allocation_df,
        portfolio_totals=totals,
        performance_summary=perf,
        validation=validation,
        model_version=effective_model_version,
        model_mode=effective_model_mode,
        user_mode=user_mode.value,
        preset=preset,
        horizon=horizon.upper(),
        data_source=data_source,
    )
    return payload


@router.get("/latest")
def get_latest(assets: str = Query(default="QQQ,SPY,AAPL,SOXX,NVDA"), horizon: str = Query(default="10D")):
    tickers = [a.strip().upper() for a in assets.split(",") if a.strip()]
    try:
        signals, load_errors = get_prediction_service().latest_signals(tickers, horizon=horizon)
        return {
            "horizon": horizon.upper(),
            "load_errors": load_errors,
            "items": signals.to_dict("records"),
        }
    except DataNotFoundError as exc:
        return {
            "horizon": horizon.upper(),
            "load_errors": [{"ticker": t, "error": str(exc)} for t in tickers],
            "items": [],
        }
