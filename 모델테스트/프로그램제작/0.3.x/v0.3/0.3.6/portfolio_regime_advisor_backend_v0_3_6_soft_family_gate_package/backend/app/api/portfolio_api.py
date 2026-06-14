from __future__ import annotations

from fastapi import APIRouter

from ..dependencies import get_allocation_service, get_parameter_validator, get_prediction_service
from ..schemas import SettingsRequest

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("")
def get_portfolio(assets: str = "QQQ,SPY,AAPL,SOXX,NVDA", horizon: str = "10D", capital_mode: str = "equal"):
    tickers = [a.strip().upper() for a in assets.split(",") if a.strip()]
    signals, errors = get_prediction_service().latest_signals(tickers, horizon=horizon)
    allocation_df, totals = get_allocation_service().apply(signals, capital_mode=capital_mode)
    return {"totals": totals, "items": allocation_df.to_dict("records"), "errors": errors}


@router.post("/custom-weights")
def custom_weights(req: SettingsRequest):
    validation = get_parameter_validator().validate_settings(
        user_mode=req.user_mode,
        horizon=req.horizon,
        assets=req.assets,
        capital_mode="custom",
        custom_weights=req.custom_weights,
    )
    if not validation.ok:
        return {"ok": False, "validation": validation.messages}
    signals, errors = get_prediction_service().latest_signals(req.assets, horizon=req.horizon)
    allocation_df, totals = get_allocation_service().apply(signals, capital_mode="custom", custom_weights={k.upper(): v for k, v in (req.custom_weights or {}).items()})
    return {"ok": True, "totals": totals, "items": allocation_df.to_dict("records"), "errors": errors}
