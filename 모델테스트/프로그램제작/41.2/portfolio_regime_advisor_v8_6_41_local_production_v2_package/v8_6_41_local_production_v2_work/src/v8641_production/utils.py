"""Utility functions."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class MathUtils:
    @staticmethod
    def safe_float(x, default: float = np.nan) -> float:
        try:
            if pd.isna(x):
                return default
            return float(x)
        except Exception:
            return default

    @staticmethod
    def normalize_weights(weights: Dict[str, float], assets: List[str]) -> Dict[str, float]:
        clean = {ticker: max(0.0, float(weights.get(ticker, 0.0))) for ticker in assets}
        total = sum(clean.values())
        if total <= 0:
            return {ticker: 1.0 / len(assets) for ticker in assets}
        return {ticker: value / total for ticker, value in clean.items()}

    @staticmethod
    def equity_and_drawdown(returns: pd.Series, initial: float = 1.0) -> Tuple[pd.Series, pd.Series]:
        r = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
        equity = initial * (1.0 + r).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        return equity, drawdown


class DateUtils:
    @staticmethod
    def ensure_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
        out = df.copy()
        out[col] = pd.to_datetime(out[col], errors="coerce")
        out = out.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
        return out
