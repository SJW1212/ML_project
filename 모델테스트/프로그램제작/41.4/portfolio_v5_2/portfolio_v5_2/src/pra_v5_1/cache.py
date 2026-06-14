from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .utils import atomic_write_csv, ensure_dir, normalize_ticker


class MarketDataCache:
    def __init__(self, root: Path):
        self.root = root

    def ohlcv_path(self, ticker: str, provider: str = "yahoo") -> Path:
        t = normalize_ticker(ticker)
        return self.root / "ohlcv" / provider / f"{t}.csv"

    def has_ohlcv(self, ticker: str, provider: str = "yahoo") -> bool:
        return self.ohlcv_path(ticker, provider).exists()

    def write_ohlcv(self, ticker: str, df: pd.DataFrame, provider: str = "yahoo") -> Path:
        path = self.ohlcv_path(ticker, provider)
        out = self.normalize_ohlcv(df)
        atomic_write_csv(path, out)
        return path

    def read_ohlcv(self, ticker: str, provider: str = "yahoo") -> pd.DataFrame:
        path = self.ohlcv_path(ticker, provider)
        if not path.exists():
            raise FileNotFoundError(f"OHLCV cache not found: {path}")
        df = pd.read_csv(path)
        return self.normalize_ohlcv(df)

    def latest_date(self, ticker: str, provider: str = "yahoo") -> Optional[str]:
        try:
            df = self.read_ohlcv(ticker, provider)
        except FileNotFoundError:
            return None
        if df.empty:
            return None
        return str(pd.to_datetime(df["Date"]).max().date())

    def freshness(self, tickers: Iterable[str], provider: str = "yahoo") -> List[Dict]:
        rows = []
        today = pd.Timestamp.utcnow().date()
        for raw in tickers:
            t = normalize_ticker(raw)
            latest = self.latest_date(t, provider)
            age_days = None
            is_fresh_daily = False
            if latest:
                d = pd.to_datetime(latest).date()
                age_days = (today - d).days
                is_fresh_daily = age_days <= 4
            rows.append({"ticker": t, "provider": provider, "latest_date": latest, "age_days": age_days, "is_fresh_daily": is_fresh_daily})
        return rows

    @staticmethod
    def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        out = df.copy()
        rename = {c: str(c).strip().title() for c in out.columns}
        out = out.rename(columns=rename)
        if "Adj Close" in out.columns and "Close" not in out.columns:
            out["Close"] = out["Adj Close"]
        if "Date" not in out.columns:
            if isinstance(out.index, pd.DatetimeIndex):
                out = out.reset_index().rename(columns={out.index.name or "index": "Date"})
            else:
                raise ValueError("OHLCV DataFrame must include Date column or DatetimeIndex")
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in out.columns:
                if col == "Volume":
                    out[col] = 0
                else:
                    raise ValueError(f"Missing OHLCV column: {col}")
        out = out[required].copy()
        out["Date"] = pd.to_datetime(out["Date"]).dt.date.astype(str)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"]).drop_duplicates("Date").sort_values("Date")
        return out.reset_index(drop=True)
