from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..dependencies import (
    build_kis_client,
    build_yahoo_client,
    get_credential_manager,
    get_market_data_repository,
    get_market_data_service,
)
from ..schemas import MarketDataUpdateRequest

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.post("/update")
def update_market_data(req: MarketDataUpdateRequest):
    """Update OHLCV cache from an external provider.

    provider modes:
    - auto: KIS if credentials exist, otherwise Yahoo; if KIS fails, Yahoo; if all fail, cache fallback.
    - kis: KIS only, then cache fallback.
    - yahoo: Yahoo Finance only, then cache fallback.
    """
    provider = req.provider.lower().strip()
    if provider not in {"auto", "kis", "yahoo"}:
        raise HTTPException(status_code=400, detail="provider must be one of: auto, kis, yahoo")

    cm = get_credential_manager()
    kis_available = bool(cm.load_credentials("kis"))
    kis_client = None
    yahoo_client = None

    if provider in {"auto", "kis"} and kis_available:
        try:
            kis_client = build_kis_client(req.environment)
        except Exception:
            kis_client = None
    if provider in {"auto", "yahoo"}:
        yahoo_client = build_yahoo_client()

    service = get_market_data_service()
    results = []
    for ticker in req.tickers:
        result = service.update_one(
            ticker=ticker,
            start_date=req.start_date,
            end_date=req.end_date,
            market=req.market,
            provider_mode=provider,
            kis_client=kis_client,
            yahoo_client=yahoo_client,
            kis_available=kis_available,
        )
        results.append(result)

    return {
        "provider_requested": provider,
        "environment": req.environment,
        "market": req.market,
        "kis_credentials_registered": kis_available,
        "results": results,
    }


@router.get("/freshness")
def freshness(provider: str = "auto", market: str = "US", tickers: str = "QQQ"):
    repo = get_market_data_repository()
    items = [repo.freshness(provider, t.strip(), market) for t in tickers.split(",") if t.strip()]
    return {"items": items}


@router.get("/{ticker}")
def get_cached_market_data(ticker: str, provider: str = "auto", market: str = "US", limit: int = 200):
    df = get_market_data_repository().load_ohlcv(provider, ticker, market)
    if df is None:
        raise HTTPException(status_code=404, detail="Cached market data not found")
    df = df.tail(limit).copy()
    df["Date"] = df["Date"].astype(str)
    return {"ticker": ticker.upper(), "provider": provider, "market": market, "rows": df.to_dict("records")}
