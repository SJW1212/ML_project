from __future__ import annotations

from typing import Optional

import pandas as pd

from ..core.exceptions import ProviderError
from ..data.data_normalizer import DataNormalizer
from .base_provider import MarketDataProvider


class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance data provider via yfinance.

    Notes
    -----
    - This provider is designed as a fallback when authenticated broker APIs are unavailable.
    - yfinance is not an official Yahoo product. Use it as a convenience/research provider,
      not as the only production-grade data source.
    """

    provider_name = "yahoo"

    def __init__(self, auto_adjust: bool = False):
        self.auto_adjust = auto_adjust

    def _yf(self):
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:
            raise ProviderError(
                "yfinance is not installed. Install it with `pip install yfinance` "
                "or use another data provider."
            ) from exc
        return yf

    @staticmethod
    def _to_yahoo_symbol(ticker: str, market: str = "US") -> str:
        """Map internal ticker/market to Yahoo symbol.

        - US tickers generally use the raw symbol: QQQ, AAPL, SPY.
        - Korean listed stocks usually use .KS for KOSPI and .KQ for KOSDAQ. Because the
          program currently receives only `market`, KR defaults to .KS unless the user
          already supplied a suffix.
        """
        ticker = ticker.strip().upper()
        market = market.upper()
        if "." in ticker:
            return ticker
        if market in {"KR", "KOR", "KS", "KOSPI"}:
            return f"{ticker}.KS"
        if market in {"KQ", "KOSDAQ"}:
            return f"{ticker}.KQ"
        return ticker

    def test_connection(self, ticker: str = "QQQ", market: str = "US") -> dict:
        price = self.get_current_price(ticker=ticker, market=market)
        return {"ok": True, "provider": self.provider_name, "sample": price}

    def get_current_price(self, ticker: str, market: str = "US") -> dict:
        yf = self._yf()
        symbol = self._to_yahoo_symbol(ticker, market)
        try:
            hist = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=self.auto_adjust)
        except Exception as exc:
            raise ProviderError(f"Yahoo Finance current price request failed for {symbol}: {exc}") from exc
        if hist is None or hist.empty:
            raise ProviderError(f"Yahoo Finance returned no current price data for {symbol}.")
        row = hist.dropna(how="all").tail(1)
        if row.empty:
            raise ProviderError(f"Yahoo Finance returned only empty rows for {symbol}.")
        last = row.iloc[0]
        date_value = row.index[-1]
        return {
            "ticker": ticker.upper(),
            "symbol": symbol,
            "market": market.upper(),
            "provider": self.provider_name,
            "date": str(pd.to_datetime(date_value).date()),
            "open": float(last.get("Open")) if pd.notna(last.get("Open")) else None,
            "high": float(last.get("High")) if pd.notna(last.get("High")) else None,
            "low": float(last.get("Low")) if pd.notna(last.get("Low")) else None,
            "close": float(last.get("Close")) if pd.notna(last.get("Close")) else None,
            "volume": int(last.get("Volume")) if pd.notna(last.get("Volume")) else None,
        }

    def get_daily_ohlcv(self, ticker: str, start_date: str, end_date: str, market: str = "US") -> pd.DataFrame:
        yf = self._yf()
        symbol = self._to_yahoo_symbol(ticker, market)
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=self.auto_adjust,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise ProviderError(f"Yahoo Finance daily OHLCV request failed for {symbol}: {exc}") from exc
        return DataNormalizer.normalize_yahoo_daily(df, ticker=ticker, symbol=symbol, market=market)
