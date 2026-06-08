from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..dependencies import build_kis_client, get_market_data_repository
from ..schemas import MarketDataUpdateRequest

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.post("/update")
def update_market_data(req: MarketDataUpdateRequest):
    if req.provider.lower() != "kis":
        raise HTTPException(status_code=400, detail="Only kis provider is implemented in MVP.")
    client = build_kis_client(req.environment)
    repo = get_market_data_repository()
    results = []
    for ticker in req.tickers:
        try:
            df = client.get_daily_ohlcv(ticker, req.start_date, req.end_date, market=req.market)
            path = repo.save_ohlcv(df, provider=req.provider, ticker=ticker, market=req.market)
            results.append({"ticker": ticker.upper(), "ok": True, "rows": len(df), "cache_path": str(path)})
        except Exception as exc:
            cached = repo.freshness(req.provider, ticker, req.market)
            results.append({"ticker": ticker.upper(), "ok": False, "error": str(exc), "cache_fallback": cached})
    return {"provider": req.provider, "environment": req.environment, "results": results}


@router.get("/freshness")
def freshness(provider: str = "kis", market: str = "KR", tickers: str = "005930"):
    repo = get_market_data_repository()
    items = [repo.freshness(provider, t.strip(), market) for t in tickers.split(",") if t.strip()]
    return {"items": items}


@router.get("/{ticker}")
def get_cached_market_data(ticker: str, provider: str = "kis", market: str = "KR", limit: int = 200):
    df = get_market_data_repository().load_ohlcv(provider, ticker, market)
    if df is None:
        raise HTTPException(status_code=404, detail="Cached market data not found")
    df = df.tail(limit).copy()
    df["Date"] = df["Date"].astype(str)
    return {"ticker": ticker.upper(), "provider": provider, "market": market, "rows": df.to_dict("records")}
