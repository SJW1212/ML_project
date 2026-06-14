from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class MarketDataRepository:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, provider: str, ticker: str, market: str) -> Path:
        safe = f"{provider.lower()}_{market.upper()}_{ticker.upper()}_ohlcv.parquet"
        return self.cache_dir / safe

    def save_ohlcv(self, df: pd.DataFrame, provider: str, ticker: str, market: str) -> Path:
        path = self._path(provider, ticker, market)
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date", keep="last")
        try:
            df.to_parquet(path, index=False)
        except Exception:
            # fallback if pyarrow is unavailable
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)
        return path

    def load_ohlcv(self, provider: str, ticker: str, market: str) -> Optional[pd.DataFrame]:
        path = self._path(provider, ticker, market)
        if path.exists():
            return pd.read_parquet(path)
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path, parse_dates=["Date"])
        return None

    def freshness(self, provider: str, ticker: str, market: str) -> dict:
        df = self.load_ohlcv(provider, ticker, market)
        if df is None or df.empty:
            return {"ticker": ticker.upper(), "provider": provider, "market": market, "exists": False, "last_date": None, "rows": 0}
        dates = pd.to_datetime(df["Date"])
        return {"ticker": ticker.upper(), "provider": provider, "market": market, "exists": True, "last_date": str(dates.max().date()), "rows": int(len(df))}
