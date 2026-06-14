"""Performance analytics layer."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ProductionConfig
from .schemas import AssetData, PerformanceRow, to_plain_dict
from .utils import MathUtils


class PerformanceAnalyzer:
    """Computes metrics for UI cards and charts."""

    def __init__(self, config: ProductionConfig):
        self.config = config

    def summarize_all(self, assets: Dict[str, AssetData]) -> List[PerformanceRow]:
        rows: List[PerformanceRow] = []
        for ticker, asset in assets.items():
            df = asset.predictions
            rows.append(self._summarize(ticker, df, "full_period"))
            holdout = df[df["Date"] >= pd.to_datetime(self.config.holdout_start)]
            rows.append(self._summarize(ticker, holdout, f"holdout_{self.config.holdout_start}"))
        return rows

    def portfolio_daily_returns(self, assets: Dict[str, AssetData], capital_weights: Dict[str, float]) -> pd.DataFrame:
        merged = None
        for ticker, asset in assets.items():
            df = asset.predictions[["Date", "strategy_return_net"]].copy()
            df = df.rename(columns={"strategy_return_net": ticker})
            merged = df if merged is None else pd.merge(merged, df, on="Date", how="inner")
        if merged is None:
            return pd.DataFrame(columns=["Date", "portfolio_return", "portfolio_equity", "portfolio_drawdown"])
        ret = pd.Series(0.0, index=merged.index)
        for ticker in assets:
            ret = ret + pd.to_numeric(merged[ticker], errors="coerce").fillna(0.0) * capital_weights[ticker]
        equity, dd = MathUtils.equity_and_drawdown(ret, self.config.initial_capital)
        return pd.DataFrame({
            "Date": merged["Date"],
            "portfolio_return": ret.astype(float),
            "portfolio_equity": equity.astype(float),
            "portfolio_drawdown": dd.astype(float),
        })

    def annual_returns(self, portfolio_returns: pd.DataFrame) -> List[dict]:
        if portfolio_returns.empty:
            return []
        df = portfolio_returns.copy()
        df["year"] = pd.to_datetime(df["Date"]).dt.year
        out = []
        for year, g in df.groupby("year"):
            r = pd.to_numeric(g["portfolio_return"], errors="coerce").fillna(0.0)
            out.append({"year": int(year), "return": float((1.0 + r).prod() - 1.0), "n_days": int(len(r))})
        return out

    def monthly_returns(self, portfolio_returns: pd.DataFrame) -> List[dict]:
        if portfolio_returns.empty:
            return []
        df = portfolio_returns.copy()
        dt = pd.to_datetime(df["Date"])
        df["month"] = dt.dt.to_period("M").astype(str)
        out = []
        for month, g in df.groupby("month"):
            r = pd.to_numeric(g["portfolio_return"], errors="coerce").fillna(0.0)
            out.append({"month": str(month), "return": float((1.0 + r).prod() - 1.0), "n_days": int(len(r))})
        return out

    @staticmethod
    def as_ui_list(rows: List[PerformanceRow]) -> List[dict]:
        return [to_plain_dict(row) for row in rows]

    def _summarize(self, ticker: str, df: pd.DataFrame, scope: str) -> PerformanceRow:
        returns = df["strategy_return_net"] if len(df) else pd.Series(dtype=float)
        metrics = self.performance_metrics(returns)
        return PerformanceRow(
            ticker=ticker,
            scope=scope,
            n_days=int(metrics["n_days"]),
            final_capital=metrics["final_capital"],
            cagr=metrics["cagr"],
            mdd=metrics["mdd"],
            sharpe=metrics["sharpe"],
            sortino=metrics["sortino"],
            calmar=metrics["calmar"],
            annual_vol=metrics["annual_vol"],
            win_rate=metrics["win_rate"],
            avg_stock_weight=float(pd.to_numeric(df.get("stock_weight"), errors="coerce").mean()) if len(df) else np.nan,
            avg_bond_weight=float(pd.to_numeric(df.get("bond_weight"), errors="coerce").mean()) if len(df) else np.nan,
            avg_cash_weight=float(pd.to_numeric(df.get("cash_weight"), errors="coerce").mean()) if len(df) else np.nan,
        )

    def performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
        n = int(len(r))
        if n == 0:
            return {"n_days": 0, "final_capital": self.config.initial_capital, "cagr": np.nan, "mdd": np.nan, "sharpe": np.nan, "sortino": np.nan, "calmar": np.nan, "annual_vol": np.nan, "win_rate": np.nan}
        equity, dd = MathUtils.equity_and_drawdown(r, self.config.initial_capital)
        years = max(n / 252.0, 1e-12)
        final_capital = float(equity.iloc[-1])
        cagr = (final_capital / self.config.initial_capital) ** (1.0 / years) - 1.0 if final_capital > 0 else -1.0
        mdd = float(dd.min())
        annual_mean = float(r.mean() * 252.0)
        annual_vol = float(r.std(ddof=0) * math.sqrt(252.0))
        sharpe = annual_mean / annual_vol if annual_vol > 1e-12 else np.nan
        downside_vol = float(r[r < 0].std(ddof=0) * math.sqrt(252.0)) if (r < 0).any() else np.nan
        sortino = annual_mean / downside_vol if np.isfinite(downside_vol) and downside_vol > 1e-12 else np.nan
        calmar = cagr / abs(mdd) if abs(mdd) > 1e-12 else np.nan
        return {"n_days": n, "final_capital": final_capital, "cagr": float(cagr), "mdd": mdd, "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan, "sortino": float(sortino) if np.isfinite(sortino) else np.nan, "calmar": float(calmar) if np.isfinite(calmar) else np.nan, "annual_vol": annual_vol, "win_rate": float((r > 0).mean())}
