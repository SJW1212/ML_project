from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def test_connection(self, ticker: str = "005930", market: str = "KR") -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_daily_ohlcv(self, ticker: str, start_date: str, end_date: str, market: str = "KR") -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_current_price(self, ticker: str, market: str = "KR") -> dict:
        raise NotImplementedError

    def get_multiple_daily_ohlcv(self, tickers: List[str], start_date: str, end_date: str, market: str = "KR") -> dict:
        results = {}
        for ticker in tickers:
            try:
                results[ticker.upper()] = {"ok": True, "data": self.get_daily_ohlcv(ticker, start_date, end_date, market)}
            except Exception as exc:
                results[ticker.upper()] = {"ok": False, "error": str(exc)}
        return results
