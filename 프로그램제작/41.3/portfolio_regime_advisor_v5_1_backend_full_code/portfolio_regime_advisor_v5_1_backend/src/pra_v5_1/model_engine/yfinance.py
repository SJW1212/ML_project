"""Local yfinance shim for the external v8.6.41 script.

When the original script imports yfinance from this directory, download() reads
cached OHLCV CSV files from PRA_MARKET_CACHE_DIR instead of making network calls.
This keeps the architecture provider/cache-first.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd


def download(ticker: str, start: Optional[str] = None, end: Optional[str] = None, auto_adjust: bool = True, progress: bool = False, threads: bool = False, **kwargs):
    root = Path(os.environ.get("PRA_MARKET_CACHE_DIR", "storage/market_cache"))
    provider = os.environ.get("PRA_MARKET_CACHE_PROVIDER", "yahoo")
    path = root / "ohlcv" / provider / f"{ticker.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cached OHLCV not found for {ticker}: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] < pd.to_datetime(end)]
    out = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index.name = "Date"
    return out
