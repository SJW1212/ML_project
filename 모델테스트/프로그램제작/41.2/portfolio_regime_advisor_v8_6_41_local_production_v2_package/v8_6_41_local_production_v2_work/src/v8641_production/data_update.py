"""Daily market data cache update layer.

Scope is intentionally narrow:
- daily OHLCV/cache update only
- no realtime streaming
- no order execution
- no account storage
- no database dependency
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import os
import tempfile

import pandas as pd

from .config import ProductionConfig


@dataclass
class CacheFreshness:
    ticker: str
    provider: str
    cache_path: str
    exists: bool
    latest_date: Optional[str]
    age_days: Optional[int]
    is_fresh_daily: bool
    row_count: int
    message: str


@dataclass
class UpdateStatus:
    ok: bool
    provider: str
    updated_at_utc: str
    tickers: List[str]
    updated: List[str]
    skipped: List[str]
    errors: Dict[str, str]
    cache_dir: str
    note: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class MarketDataCache:
    """File-based cache for local-only operation."""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.status_path = self.cache_dir / "daily_update_status.json"

    def ohlcv_path(self, ticker: str, provider: Optional[str] = None) -> Path:
        p = (provider or self.config.update_provider).lower()
        return self.cache_dir / "ohlcv" / p / f"{ticker.upper()}.csv"

    def freshness(self, tickers: Iterable[str], provider: Optional[str] = None) -> List[CacheFreshness]:
        p = (provider or self.config.update_provider).lower()
        today = pd.Timestamp.utcnow().normalize().tz_localize(None)
        items: List[CacheFreshness] = []
        for ticker in [t.upper() for t in tickers]:
            path = self.ohlcv_path(ticker, p)
            if not path.exists():
                items.append(CacheFreshness(
                    ticker=ticker,
                    provider=p,
                    cache_path=str(path),
                    exists=False,
                    latest_date=None,
                    age_days=None,
                    is_fresh_daily=False,
                    row_count=0,
                    message="cache file not found",
                ))
                continue
            try:
                df = pd.read_csv(path)
                if "Date" not in df.columns or df.empty:
                    raise ValueError("missing Date column or empty file")
                dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
                if dates.empty:
                    raise ValueError("no valid Date values")
                latest = dates.max().normalize().tz_localize(None) if getattr(dates.max(), "tzinfo", None) else dates.max().normalize()
                age_days = int((today - latest).days)
                is_fresh = age_days <= int(self.config.daily_freshness_tolerance_days)
                items.append(CacheFreshness(
                    ticker=ticker,
                    provider=p,
                    cache_path=str(path),
                    exists=True,
                    latest_date=str(latest.date()),
                    age_days=age_days,
                    is_fresh_daily=is_fresh,
                    row_count=int(len(df)),
                    message="fresh" if is_fresh else "stale",
                ))
            except Exception as exc:
                items.append(CacheFreshness(
                    ticker=ticker,
                    provider=p,
                    cache_path=str(path),
                    exists=True,
                    latest_date=None,
                    age_days=None,
                    is_fresh_daily=False,
                    row_count=0,
                    message=f"invalid cache file: {exc}",
                ))
        return items

    def write_status(self, status: UpdateStatus) -> None:
        self._ensure_writable_dir(self.status_path.parent)
        self._atomic_write_text(self.status_path, json.dumps(asdict(status), ensure_ascii=False, indent=2))

    @staticmethod
    def _ensure_writable_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        test_path = path / ".write_test"
        try:
            test_path.write_text("ok", encoding="utf-8")
            test_path.unlink(missing_ok=True)
        except PermissionError as exc:
            raise RuntimeError(f"cache directory is not writable: {path}") from exc
        except OSError as exc:
            raise RuntimeError(f"cache directory write check failed: {path}: {exc}") from exc

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @staticmethod
    def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
        MarketDataCache._ensure_writable_dir(path.parent)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
        try:
            os.close(fd)
            df.to_csv(tmp_name, index=False, encoding="utf-8-sig")
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def read_status(self) -> Dict[str, Any]:
        if not self.status_path.exists():
            return {
                "ok": False,
                "message": "daily update has not been executed yet",
                "status_path": str(self.status_path),
            }
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "ok": False,
                "message": f"failed to read update status: {exc}",
                "status_path": str(self.status_path),
            }


class DailyMarketDataUpdater:
    """Run one daily OHLCV cache update.

    yfinance is optional. The API remains usable without it; update endpoints then
    return a clear installation error instead of failing at import time.
    """

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.cache = MarketDataCache(config)

    def update_daily(
        self,
        tickers: Iterable[str],
        *,
        provider: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force: bool = False,
    ) -> UpdateStatus:
        provider_name = (provider or self.config.update_provider).lower()
        tickers_list = [t.upper() for t in tickers]
        if provider_name == "kis":
            status = UpdateStatus(
                ok=False,
                provider=provider_name,
                updated_at_utc=_utc_now_iso(),
                tickers=tickers_list,
                updated=[],
                skipped=tickers_list,
                errors={"kis": "KIS data-query adapter is reserved for a later local-data patch; no order API is implemented."},
                cache_dir=str(self.cache.cache_dir),
                note="daily cache update only; realtime/order features are excluded",
            )
            self.cache.write_status(status)
            return status
        if provider_name != "yahoo":
            status = UpdateStatus(
                ok=False,
                provider=provider_name,
                updated_at_utc=_utc_now_iso(),
                tickers=tickers_list,
                updated=[],
                skipped=tickers_list,
                errors={provider_name: "unsupported provider; use yahoo or kis"},
                cache_dir=str(self.cache.cache_dir),
                note="daily cache update only; realtime/order features are excluded",
            )
            self.cache.write_status(status)
            return status
        return self._update_yahoo(tickers_list, start=start, end=end, force=force)

    def _update_yahoo(self, tickers: List[str], *, start: Optional[str], end: Optional[str], force: bool) -> UpdateStatus:
        updated: List[str] = []
        skipped: List[str] = []
        errors: Dict[str, str] = {}
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:
            status = UpdateStatus(
                ok=False,
                provider="yahoo",
                updated_at_utc=_utc_now_iso(),
                tickers=tickers,
                updated=[],
                skipped=tickers,
                errors={"yfinance": f"yfinance is not installed or unavailable: {exc}"},
                cache_dir=str(self.cache.cache_dir),
                note="install yfinance for daily cache update; API/dashboard still work from prediction files",
            )
            self.cache.write_status(status)
            return status

        for ticker in tickers:
            path = self.cache.ohlcv_path(ticker, "yahoo")
            try:
                if not force and path.exists():
                    fresh = self.cache.freshness([ticker], "yahoo")[0]
                    if fresh.is_fresh_daily:
                        skipped.append(ticker)
                        continue
                df = yf.download(
                    ticker,
                    start=start or self.config.default_update_start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
                if df is None or df.empty:
                    raise ValueError("no data returned")
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                rename = {"Adj Close": "Adj_Close"}
                df = df.rename(columns=rename)
                required = ["Date", "Open", "High", "Low", "Close", "Volume"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    raise ValueError(f"downloaded data missing columns: {missing}")
                keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume"] if c in df.columns]
                out = df[keep_cols].copy()
                out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
                self.cache._atomic_write_csv(out, path)
                updated.append(ticker)
            except Exception as exc:
                errors[ticker] = str(exc)
        status = UpdateStatus(
            ok=len(errors) == 0,
            provider="yahoo",
            updated_at_utc=_utc_now_iso(),
            tickers=tickers,
            updated=updated,
            skipped=skipped,
            errors=errors,
            cache_dir=str(self.cache.cache_dir),
            note="daily cache update only; realtime streaming and orders are excluded",
        )
        self.cache.write_status(status)
        return status
