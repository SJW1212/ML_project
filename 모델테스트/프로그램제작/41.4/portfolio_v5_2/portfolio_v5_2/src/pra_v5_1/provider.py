from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .cache import MarketDataCache
from .utils import normalize_ticker


@dataclass
class UpdateStatus:
    provider: str
    updated: List[str]
    skipped: List[str]
    errors: Dict[str, str]


class YahooMarketDataProvider:
    def download_ohlcv(self, ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as exc:
            raise RuntimeError("yfinance is not installed. Run: python -m pip install yfinance") from exc
        t = normalize_ticker(ticker)
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            raise RuntimeError(f"No OHLCV data returned for {t}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.reset_index()


class DailyMarketDataUpdater:
    def __init__(self, cache: MarketDataCache):
        self.cache = cache

    def update_daily(self, tickers: Iterable[str], provider: str = "yahoo", start: str = "2013-01-01", end: Optional[str] = None, force: bool = False) -> UpdateStatus:
        provider = provider.lower()
        updated: List[str] = []
        skipped: List[str] = []
        errors: Dict[str, str] = {}
        if provider == "kis":
            return UpdateStatus(provider=provider, updated=[], skipped=[], errors={"kis": "KIS data adapter is reserved for quotation/daily data only and is not implemented in this local package."})
        if provider != "yahoo":
            return UpdateStatus(provider=provider, updated=[], skipped=[], errors={"provider": f"Unsupported provider: {provider}"})
        yf_provider = YahooMarketDataProvider()
        for raw in tickers:
            t = normalize_ticker(raw)
            try:
                if not force and self.cache.has_ohlcv(t, provider):
                    fresh = self.cache.freshness([t], provider)[0]
                    if fresh.get("is_fresh_daily"):
                        skipped.append(t)
                        continue
                df = yf_provider.download_ohlcv(t, start=start, end=end)
                self.cache.write_ohlcv(t, df, provider)
                updated.append(t)
            except Exception as exc:
                errors[t] = str(exc)
        return UpdateStatus(provider=provider, updated=updated, skipped=skipped, errors=errors)
