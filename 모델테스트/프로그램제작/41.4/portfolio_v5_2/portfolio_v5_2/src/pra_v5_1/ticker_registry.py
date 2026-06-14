from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .utils import atomic_write_json, load_json, normalize_ticker, utc_now_iso


@dataclass
class TickerRecord:
    ticker: str
    asset_type: str = "stock"
    market: str = "US"
    enabled: bool = True
    note: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class LocalTickerRegistry:
    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> Dict[str, TickerRecord]:
        raw = load_json(self.path, default={}) or {}
        records = raw.get("tickers", raw if isinstance(raw, dict) else {})
        out: Dict[str, TickerRecord] = {}
        for ticker, item in records.items():
            if not isinstance(item, dict):
                continue
            t = normalize_ticker(item.get("ticker", ticker))
            out[t] = TickerRecord(
                ticker=t,
                asset_type=item.get("asset_type", "stock"),
                market=item.get("market", "US"),
                enabled=bool(item.get("enabled", True)),
                note=item.get("note"),
                created_at=item.get("created_at", ""),
                updated_at=item.get("updated_at", ""),
            )
        return out

    def _save(self, records: Dict[str, TickerRecord]) -> None:
        payload = {
            "schema_version": "v1",
            "updated_at": utc_now_iso(),
            "tickers": {t: asdict(r) for t, r in sorted(records.items())},
        }
        atomic_write_json(self.path, payload)

    def add_or_update(self, tickers: Iterable[str], asset_type: str = "stock", market: str = "US", note: Optional[str] = None) -> List[TickerRecord]:
        records = self._load()
        now = utc_now_iso()
        changed: List[TickerRecord] = []
        for raw in tickers:
            t = normalize_ticker(raw)
            old = records.get(t)
            rec = TickerRecord(
                ticker=t,
                asset_type=asset_type,
                market=market,
                enabled=True,
                note=note if note is not None else (old.note if old else None),
                created_at=old.created_at if old else now,
                updated_at=now,
            )
            records[t] = rec
            changed.append(rec)
        self._save(records)
        return changed

    def replace_enabled(self, tickers: Iterable[str], asset_type: str = "stock", market: str = "US") -> List[TickerRecord]:
        records: Dict[str, TickerRecord] = {}
        now = utc_now_iso()
        for raw in tickers:
            t = normalize_ticker(raw)
            records[t] = TickerRecord(ticker=t, asset_type=asset_type, market=market, enabled=True, created_at=now, updated_at=now)
        self._save(records)
        return list(records.values())

    def disable(self, tickers: Iterable[str]) -> None:
        records = self._load()
        now = utc_now_iso()
        for raw in tickers:
            t = normalize_ticker(raw)
            if t in records:
                records[t].enabled = False
                records[t].updated_at = now
        self._save(records)

    def list(self, enabled_only: bool = False) -> List[TickerRecord]:
        records = self._load()
        values = list(records.values())
        if enabled_only:
            values = [x for x in values if x.enabled]
        return sorted(values, key=lambda x: x.ticker)

    def enabled_tickers(self) -> List[str]:
        return [r.ticker for r in self.list(enabled_only=True)]
