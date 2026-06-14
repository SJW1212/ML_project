from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    @staticmethod
    def metrics(returns: pd.Series, initial_capital: float = 100_000_000.0, risk_free_rate: float = 0.0) -> Dict[str, float]:
        r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
        n = len(r)
        if n == 0:
            return {"n_days": 0, "final_capital": initial_capital, "cagr": None, "mdd": None, "sharpe": None, "sortino": None, "calmar": None}
        equity = initial_capital * (1 + r).cumprod()
        dd = equity / equity.cummax() - 1
        years = max(n / 252.0, 1e-12)
        final = float(equity.iloc[-1])
        cagr = (final / initial_capital) ** (1 / years) - 1 if final > 0 else -1.0
        ann_mean = float(r.mean() * 252)
        ann_vol = float(r.std(ddof=0) * math.sqrt(252))
        downside = float(r[r < 0].std(ddof=0) * math.sqrt(252)) if (r < 0).any() else np.nan
        mdd = float(dd.min())
        return {
            "n_days": int(n),
            "final_capital": final,
            "cagr": cagr,
            "mdd": mdd,
            "sharpe": (ann_mean - risk_free_rate) / ann_vol if ann_vol > 1e-12 else None,
            "sortino": (ann_mean - risk_free_rate) / downside if np.isfinite(downside) and downside > 1e-12 else None,
            "calmar": cagr / abs(mdd) if abs(mdd) > 1e-12 else None,
            "annual_vol": ann_vol,
            "win_rate": float((r > 0).mean()),
        }

    def portfolio_returns(self, prediction_frames: Dict[str, pd.DataFrame], weights: Dict[str, float], missing_asset_policy: str = "cash_fallback") -> pd.DataFrame:
        frames = []
        for ticker, df in prediction_frames.items():
            if df.empty:
                continue
            part = df[["Date", "stock_next_return"]].copy()
            part["Date"] = pd.to_datetime(part["Date"])
            part = part.rename(columns={"stock_next_return": ticker})
            frames.append(part)
        if not frames:
            return pd.DataFrame(columns=["Date", "portfolio_return", "equity", "drawdown"])
        if missing_asset_policy == "common_range_only":
            merged = frames[0]
            for f in frames[1:]:
                merged = pd.merge(merged, f, on="Date", how="inner")
            ret = sum(pd.to_numeric(merged[t], errors="coerce").fillna(0.0) * weights.get(t, 0.0) for t in weights if t in merged.columns)
        else:
            merged = frames[0]
            for f in frames[1:]:
                merged = pd.merge(merged, f, on="Date", how="outer")
            merged = merged.sort_values("Date")
            if missing_asset_policy == "active_weight_renormalize":
                cols = [t for t in weights if t in merged.columns]
                weighted = pd.Series(0.0, index=merged.index)
                active_w = pd.Series(0.0, index=merged.index)
                for t in cols:
                    rr = pd.to_numeric(merged[t], errors="coerce")
                    mask = rr.notna()
                    weighted += rr.fillna(0.0) * weights.get(t, 0.0)
                    active_w += mask.astype(float) * weights.get(t, 0.0)
                ret = weighted / active_w.replace(0, np.nan)
                ret = ret.fillna(0.0)
            else:
                ret = sum(pd.to_numeric(merged[t], errors="coerce").fillna(0.0) * weights.get(t, 0.0) for t in weights if t in merged.columns)
        out = pd.DataFrame({"Date": merged["Date"], "portfolio_return": ret})
        out["equity"] = (1 + out["portfolio_return"].fillna(0.0)).cumprod()
        out["drawdown"] = out["equity"] / out["equity"].cummax() - 1
        return out
