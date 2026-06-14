from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


class DataNormalizer:
    """Converts external provider responses to internal OHLCV schema."""

    INTERNAL_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "Provider", "Market", "Currency", "UpdatedAt"]

    @staticmethod
    def normalize_kis_domestic_daily(rows: Iterable[Dict[str, Any]], ticker: str, provider: str = "kis") -> pd.DataFrame:
        out: List[Dict[str, Any]] = []
        for r in rows:
            date = r.get("stck_bsop_date") or r.get("date") or r.get("Date")
            if not date:
                continue
            out.append({
                "Date": pd.to_datetime(str(date), format="%Y%m%d", errors="coerce"),
                "Ticker": ticker.upper(),
                "Open": pd.to_numeric(r.get("stck_oprc") or r.get("open") or r.get("Open"), errors="coerce"),
                "High": pd.to_numeric(r.get("stck_hgpr") or r.get("high") or r.get("High"), errors="coerce"),
                "Low": pd.to_numeric(r.get("stck_lwpr") or r.get("low") or r.get("Low"), errors="coerce"),
                "Close": pd.to_numeric(r.get("stck_clpr") or r.get("close") or r.get("Close"), errors="coerce"),
                "Volume": pd.to_numeric(r.get("acml_vol") or r.get("volume") or r.get("Volume"), errors="coerce"),
                "Provider": provider,
                "Market": "KR",
                "Currency": "KRW",
                "UpdatedAt": pd.Timestamp.utcnow().isoformat(),
            })
        df = pd.DataFrame(out)
        if df.empty:
            return pd.DataFrame(columns=DataNormalizer.INTERNAL_COLUMNS)
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        df = df.sort_values("Date")
        return df[DataNormalizer.INTERNAL_COLUMNS]

    @staticmethod
    def normalize_kis_overseas_daily(rows: Iterable[Dict[str, Any]], ticker: str, market: str = "US", provider: str = "kis") -> pd.DataFrame:
        out: List[Dict[str, Any]] = []
        for r in rows:
            date = r.get("xymd") or r.get("stck_bsop_date") or r.get("date") or r.get("Date")
            out.append({
                "Date": pd.to_datetime(str(date), format="%Y%m%d", errors="coerce"),
                "Ticker": ticker.upper(),
                "Open": pd.to_numeric(r.get("open") or r.get("ovrs_nmix_oprc") or r.get("Open"), errors="coerce"),
                "High": pd.to_numeric(r.get("high") or r.get("ovrs_nmix_hgpr") or r.get("High"), errors="coerce"),
                "Low": pd.to_numeric(r.get("low") or r.get("ovrs_nmix_lwpr") or r.get("Low"), errors="coerce"),
                "Close": pd.to_numeric(r.get("clos") or r.get("close") or r.get("ovrs_nmix_prpr") or r.get("Close"), errors="coerce"),
                "Volume": pd.to_numeric(r.get("tvol") or r.get("volume") or r.get("Volume"), errors="coerce"),
                "Provider": provider,
                "Market": market.upper(),
                "Currency": "USD",
                "UpdatedAt": pd.Timestamp.utcnow().isoformat(),
            })
        df = pd.DataFrame(out)
        if df.empty:
            return pd.DataFrame(columns=DataNormalizer.INTERNAL_COLUMNS)
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        df = df.sort_values("Date")
        return df[DataNormalizer.INTERNAL_COLUMNS]

    @staticmethod
    def normalize_yahoo_daily(df: pd.DataFrame, ticker: str, symbol: str | None = None, market: str = "US", provider: str = "yahoo") -> pd.DataFrame:
        """Normalize yfinance daily OHLCV data to the internal schema."""
        if df is None or df.empty:
            return pd.DataFrame(columns=DataNormalizer.INTERNAL_COLUMNS)
        data = df.copy()
        # yfinance can return MultiIndex columns when a single symbol is downloaded in newer versions.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        data = data.reset_index()
        date_col = "Date" if "Date" in data.columns else "Datetime" if "Datetime" in data.columns else data.columns[0]
        out = pd.DataFrame({
            "Date": pd.to_datetime(data[date_col], errors="coerce").dt.tz_localize(None),
            "Ticker": ticker.upper(),
            "Open": pd.to_numeric(data.get("Open"), errors="coerce"),
            "High": pd.to_numeric(data.get("High"), errors="coerce"),
            "Low": pd.to_numeric(data.get("Low"), errors="coerce"),
            "Close": pd.to_numeric(data.get("Close"), errors="coerce"),
            "Volume": pd.to_numeric(data.get("Volume"), errors="coerce"),
            "Provider": provider,
            "Market": market.upper(),
            "Currency": "USD" if market.upper() in {"US", "USA", "NASDAQ", "NYSE"} else "KRW" if market.upper() in {"KR", "KOR", "KS", "KQ", "KOSPI", "KOSDAQ"} else None,
            "UpdatedAt": pd.Timestamp.utcnow().isoformat(),
        })
        out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"]).copy()
        out = out.sort_values("Date").drop_duplicates("Date", keep="last")
        return out[DataNormalizer.INTERNAL_COLUMNS]

