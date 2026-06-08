from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..data.market_data_repository import MarketDataRepository


CONTEXT_TICKERS: Dict[str, List[str]] = {
    "market": ["SPY", "QQQ", "IWM", "RSP"],
    "volatility": ["^VIX"],
    "credit": ["HYG", "LQD"],
    "rates": ["TLT", "IEF", "SHY"],
    "rates_raw": ["^TNX"],
    "dollar": ["UUP"],
    "defensive": ["GLD"],
    "sectors": ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLB", "XLRE"],
}


def flatten_context_tickers(groups: Optional[Iterable[str]] = None) -> List[str]:
    selected = list(groups or CONTEXT_TICKERS.keys())
    out: List[str] = []
    for group in selected:
        for ticker in CONTEXT_TICKERS.get(group, []):
            if ticker not in out:
                out.append(ticker)
    return out


@dataclass
class ContextAssetUniverse:
    """Point-in-time aligned context asset cache.

    The universe is loaded once at training/inference start from MarketDataRepository.
    Folds should call ``slice_fold`` instead of downloading or recomputing context
    assets repeatedly. Each context series is aligned to the target ticker's trading
    calendar and forward-filled with a small limit to avoid cross-asset holiday gaps.
    """

    repository: MarketDataRepository
    provider: str
    market: str
    target_index: pd.DatetimeIndex
    tickers: Optional[Iterable[str]] = None
    ffill_limit: int = 2
    strict: bool = False
    _cache: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    missing_tickers: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.provider = (self.provider or "auto").lower().strip()
        self.market = (self.market or "US").upper().strip()
        idx = pd.DatetimeIndex(pd.to_datetime(self.target_index)).sort_values().drop_duplicates()
        self.target_index = idx
        self._load_all()

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame, ticker: str, target_index: pd.DatetimeIndex, ffill_limit: int) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(index=target_index, columns=["close"])
        work = df.copy()
        rename = {c: c.title() for c in work.columns if c.lower() in {"date", "open", "high", "low", "close", "volume"}}
        work = work.rename(columns=rename)
        if "Date" not in work.columns or "Close" not in work.columns:
            return pd.DataFrame(index=target_index, columns=["close"])
        work["Date"] = pd.to_datetime(work["Date"])
        work = work.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
        close = pd.to_numeric(work["Close"], errors="coerce").rename("close")
        out = close.to_frame().reindex(target_index).ffill(limit=ffill_limit)
        if ticker.upper() == "^TNX":
            # yfinance returns ^TNX as percent points, e.g. 4.5 for 4.5%.
            # Use decimal yield for z-score/level and diff() for changes.
            out["close"] = out["close"] / 100.0
        return out

    def _load_one(self, ticker: str) -> Optional[pd.DataFrame]:
        candidates = [self.provider]
        if self.provider != "auto":
            candidates.append("auto")
        candidates.extend(["yahoo", "kis"])
        for provider in dict.fromkeys([c.lower() for c in candidates]):
            df = self.repository.load_ohlcv(provider, ticker, self.market)
            if df is not None and not df.empty:
                return self._normalize_ohlcv(df, ticker, self.target_index, self.ffill_limit)
        return None

    def _load_all(self) -> None:
        for ticker in self.tickers or flatten_context_tickers():
            ticker = ticker.upper()
            df = self._load_one(ticker)
            if df is None or df.empty or df["close"].isna().all():
                self.missing_tickers.append(ticker)
                if self.strict:
                    raise FileNotFoundError(f"Context OHLCV cache not found or empty for {ticker}")
                continue
            self._cache[ticker] = df

    def get(self, ticker: str) -> Optional[pd.DataFrame]:
        return self._cache.get(ticker.upper())

    def close(self, ticker: str) -> Optional[pd.Series]:
        df = self.get(ticker)
        if df is None or "close" not in df:
            return None
        return df["close"]

    def slice_fold(self, ticker: str, fold_index: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        df = self.get(ticker)
        if df is None:
            return None
        return df.reindex(pd.DatetimeIndex(fold_index))

    def summary(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "market": self.market,
            "loaded_tickers": sorted(self._cache.keys()),
            "missing_tickers": sorted(set(self.missing_tickers)),
            "ffill_limit": self.ffill_limit,
            "target_rows": int(len(self.target_index)),
        }
