from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pandas as pd

from .cache import MarketDataCache
from .performance import PerformanceAnalyzer


class BenchmarkService:
    """Lightweight benchmark calculator for UI comparison.

    It intentionally does not download data. It only uses prediction frames already produced
    for the user's risk assets and optional cached OHLCV for standard tickers.
    """

    def __init__(self, cache: MarketDataCache, performance: Optional[PerformanceAnalyzer] = None):
        self.cache = cache
        self.performance = performance or PerformanceAnalyzer()

    @staticmethod
    def _empty(name: str, reason: str) -> Dict[str, Any]:
        return {"name": name, "status": "unavailable", "reason": reason, "metrics": None}

    @staticmethod
    def _returns_from_ohlcv(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        close = pd.to_numeric(df["Close"], errors="coerce")
        return close.pct_change().fillna(0.0)

    @staticmethod
    def _frame_from_ohlcv(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df is None or df.empty or "Date" not in df.columns:
            return pd.DataFrame(columns=["Date", col])
        out = df[["Date", "Close"]].copy()
        out["Date"] = pd.to_datetime(out["Date"])
        out[col] = pd.to_numeric(out["Close"], errors="coerce").pct_change().fillna(0.0)
        return out[["Date", col]]

    def _metrics_payload(self, name: str, returns: pd.Series, note: str | None = None) -> Dict[str, Any]:
        if returns is None or len(returns) == 0:
            return self._empty(name, "return series is empty")
        return {"name": name, "status": "ok", "note": note, "metrics": self.performance.metrics(returns)}

    def _current_portfolio_returns(self, prediction_frames: Mapping[str, pd.DataFrame], current_weights: Mapping[str, float]) -> pd.Series:
        frames = []
        for ticker, df in prediction_frames.items():
            if df is None or df.empty or "stock_next_return" not in df.columns:
                continue
            part = df[["Date", "stock_next_return"]].copy()
            part["Date"] = pd.to_datetime(part["Date"])
            part = part.rename(columns={"stock_next_return": ticker})
            frames.append(part)
        if not frames:
            return pd.Series(dtype=float)
        merged = frames[0]
        for f in frames[1:]:
            merged = pd.merge(merged, f, on="Date", how="outer")
        merged = merged.sort_values("Date")
        ret = pd.Series(0.0, index=merged.index)
        for ticker, w in current_weights.items():
            if ticker in merged.columns:
                ret += pd.to_numeric(merged[ticker], errors="coerce").fillna(0.0) * float(w)
        return ret

    def _cached_single_asset(self, ticker: str, provider: str = "yahoo") -> Optional[pd.Series]:
        try:
            df = self.cache.read_ohlcv(ticker, provider=provider)
        except Exception:
            return None
        ret = self._returns_from_ohlcv(df)
        return ret if len(ret) else None

    def _cached_6040(self, provider: str = "yahoo") -> Optional[pd.Series]:
        try:
            spy = self._frame_from_ohlcv(self.cache.read_ohlcv("SPY", provider=provider), "SPY")
            ief = self._frame_from_ohlcv(self.cache.read_ohlcv("IEF", provider=provider), "IEF")
        except Exception:
            return None
        if spy.empty or ief.empty:
            return None
        merged = pd.merge(spy, ief, on="Date", how="inner").sort_values("Date")
        if merged.empty:
            return None
        return pd.to_numeric(merged["SPY"], errors="coerce").fillna(0.0) * 0.60 + pd.to_numeric(merged["IEF"], errors="coerce").fillna(0.0) * 0.40

    def build(self, prediction_frames: Mapping[str, pd.DataFrame], current_weights: Mapping[str, float], recommended_returns: pd.Series, provider: str = "yahoo") -> Dict[str, Any]:
        items: Dict[str, Dict[str, Any]] = {}
        items["model_recommended"] = self._metrics_payload("모델 추천 포트폴리오", recommended_returns, "recommended risk-asset weights; latest pending return is included as 0 only for metric continuity")
        cur = self._current_portfolio_returns(prediction_frames, current_weights)
        items["current_portfolio"] = self._metrics_payload("현재 포트폴리오 유지", cur, "bond/cash bucket returns are treated as 0 in this lightweight comparison")

        qqq = None
        if "QQQ" in prediction_frames and prediction_frames["QQQ"] is not None and not prediction_frames["QQQ"].empty:
            qqq = pd.to_numeric(prediction_frames["QQQ"].get("stock_next_return"), errors="coerce").fillna(0.0)
        if qqq is None or len(qqq) == 0:
            qqq = self._cached_single_asset("QQQ", provider=provider)
        items["qqq_buy_hold"] = self._metrics_payload("QQQ Buy&Hold", qqq) if qqq is not None else self._empty("QQQ Buy&Hold", "QQQ cache/prediction data is unavailable")

        spy = self._cached_single_asset("SPY", provider=provider)
        items["spy_buy_hold"] = self._metrics_payload("SPY Buy&Hold", spy) if spy is not None else self._empty("SPY Buy&Hold", "SPY cache data is unavailable")

        r6040 = self._cached_6040(provider=provider)
        items["static_60_40"] = self._metrics_payload("60/40 Static", r6040) if r6040 is not None else self._empty("60/40 Static", "SPY/IEF cache data is unavailable")

        # 85/10/5 lightweight benchmark: same risk asset mix as current portfolio, 10% bond and 5% cash as zero-return buckets.
        risk_sum = sum(max(0.0, float(w)) for w in current_weights.values())
        if risk_sum > 0:
            scaled = {t: 0.85 * float(w) / risk_sum for t, w in current_weights.items()}
            r85105 = self._current_portfolio_returns(prediction_frames, scaled)
            items["static_85_10_5"] = self._metrics_payload("85/10/5 Static", r85105, "10% bond and 5% cash buckets are treated as 0-return placeholders unless full defensive price data is connected")
        else:
            items["static_85_10_5"] = self._empty("85/10/5 Static", "current risk-asset weights are empty")

        return {
            "items": items,
            "notes": [
                "This benchmark module is lightweight and does not download missing benchmark tickers.",
                "Unavailable benchmark rows mean the required cache file was not present.",
                "Bond/cash bucket returns are placeholders until dedicated defensive asset return series are connected.",
            ],
        }
