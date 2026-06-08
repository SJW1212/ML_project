from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..core.exceptions import ProviderError
from ..integrations.base_provider import MarketDataProvider
from .market_data_repository import MarketDataRepository


@dataclass
class ProviderAttempt:
    provider: str
    ok: bool
    message: str = ""
    rows: int = 0
    cache_path: Optional[str] = None
    used_cache: bool = False


class MarketDataService:
    """Provider orchestration with KIS -> Yahoo -> cache fallback.

    Provider modes
    --------------
    - kis: use KIS only, fallback to cache if request fails.
    - yahoo: use Yahoo Finance only, fallback to cache if request fails.
    - auto: try KIS first when credentials are present, then Yahoo, then cache.
    """

    def __init__(self, repository: MarketDataRepository):
        self.repository = repository

    def update_one(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        market: str,
        provider_mode: str,
        kis_client: Optional[MarketDataProvider] = None,
        yahoo_client: Optional[MarketDataProvider] = None,
        kis_available: bool = False,
    ) -> dict:
        provider_mode = provider_mode.lower().strip()
        ticker_norm = ticker.upper().strip()
        attempts: List[ProviderAttempt] = []

        provider_order: List[tuple[str, Optional[MarketDataProvider]]] = []
        if provider_mode == "kis":
            provider_order = [("kis", kis_client)]
        elif provider_mode == "yahoo":
            provider_order = [("yahoo", yahoo_client)]
        elif provider_mode == "auto":
            # KIS has priority only when credentials exist; otherwise skip directly to Yahoo.
            if kis_available:
                provider_order.append(("kis", kis_client))
            else:
                attempts.append(ProviderAttempt("kis", False, "KIS credentials unavailable; skipped."))
            provider_order.append(("yahoo", yahoo_client))
        else:
            raise ProviderError("provider must be one of: auto, kis, yahoo")

        for provider_name, client in provider_order:
            if client is None:
                attempts.append(ProviderAttempt(provider_name, False, f"{provider_name} client is not configured."))
                continue
            try:
                df = client.get_daily_ohlcv(ticker_norm, start_date, end_date, market=market)
                if df is None or df.empty:
                    raise ProviderError(f"{provider_name} returned empty data.")
                path = self.repository.save_ohlcv(df, provider=provider_name, ticker=ticker_norm, market=market)
                # Also save the winning provider result under provider=auto to simplify default UI cache reads.
                if provider_mode == "auto":
                    self.repository.save_ohlcv(df, provider="auto", ticker=ticker_norm, market=market)
                attempts.append(ProviderAttempt(provider_name, True, "success", len(df), str(path)))
                return {
                    "ticker": ticker_norm,
                    "ok": True,
                    "provider_requested": provider_mode,
                    "provider_used": provider_name,
                    "rows": len(df),
                    "cache_path": str(path),
                    "attempts": [a.__dict__ for a in attempts],
                    "cache_fallback": None,
                }
            except Exception as exc:
                attempts.append(ProviderAttempt(provider_name, False, str(exc)))

        # Last resort: cache fallback. Prefer the requested mode cache, then auto, then provider-specific cache.
        cache_candidates = []
        if provider_mode != "auto":
            cache_candidates.append(provider_mode)
        cache_candidates.append("auto")
        cache_candidates.extend(["kis", "yahoo"])
        seen = set()
        for cache_provider in cache_candidates:
            if cache_provider in seen:
                continue
            seen.add(cache_provider)
            freshness = self.repository.freshness(cache_provider, ticker_norm, market)
            if freshness.get("exists"):
                attempts.append(ProviderAttempt(cache_provider, True, "cache fallback", int(freshness.get("rows") or 0), used_cache=True))
                return {
                    "ticker": ticker_norm,
                    "ok": True,
                    "provider_requested": provider_mode,
                    "provider_used": cache_provider,
                    "rows": int(freshness.get("rows") or 0),
                    "cache_path": None,
                    "attempts": [a.__dict__ for a in attempts],
                    "cache_fallback": freshness,
                    "warning": "Used cached market data because live provider requests failed or were unavailable.",
                }

        return {
            "ticker": ticker_norm,
            "ok": False,
            "provider_requested": provider_mode,
            "provider_used": None,
            "rows": 0,
            "cache_path": None,
            "attempts": [a.__dict__ for a in attempts],
            "cache_fallback": None,
        }
