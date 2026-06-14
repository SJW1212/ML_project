from __future__ import annotations

import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    @staticmethod
    def equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
        return initial_capital * (1.0 + returns.fillna(0)).cumprod()

    @staticmethod
    def drawdown(equity: pd.Series) -> pd.Series:
        peak = equity.cummax()
        return equity / peak - 1.0

    @staticmethod
    def metrics_from_returns(returns: pd.Series, periods_per_year: int = 252) -> dict:
        r = returns.dropna()
        if r.empty:
            return {"cagr": None, "mdd": None, "sharpe": None, "sortino": None, "calmar": None}
        equity = PerformanceAnalyzer.equity_curve(r)
        years = len(r) / periods_per_year
        cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else None
        dd = PerformanceAnalyzer.drawdown(equity)
        mdd = float(dd.min())
        vol = float(r.std() * np.sqrt(periods_per_year)) if r.std() else None
        sharpe = float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() and r.std() > 0 else None
        downside = r.where(r < 0, 0)
        sortino = float(r.mean() / downside.std() * np.sqrt(periods_per_year)) if downside.std() and downside.std() > 0 else None
        calmar = float(cagr / abs(mdd)) if cagr is not None and mdd < 0 else None
        return {"cagr": cagr, "mdd": mdd, "sharpe": sharpe, "sortino": sortino, "calmar": calmar, "volatility": vol}
