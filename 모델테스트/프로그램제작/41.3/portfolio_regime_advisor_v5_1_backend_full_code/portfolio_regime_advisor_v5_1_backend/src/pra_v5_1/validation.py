from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from .feature_schema import REQUIRED_PREDICTION_COLUMNS


class ValidationService:
    def validate_predictions(self, prediction_frames: Dict[str, pd.DataFrame], cache_freshness: List[Dict]) -> List[Dict]:
        checks: List[Dict] = []
        cache_latest = {x["ticker"]: x.get("latest_date") for x in cache_freshness}
        for ticker, df in prediction_frames.items():
            if df is None or df.empty:
                checks.append({"check_name": "prediction_not_empty", "ticker": ticker, "status": "FAIL", "detail": "Prediction frame is empty"})
                continue
            missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
            checks.append({"check_name": "required_prediction_columns", "ticker": ticker, "status": "FAIL" if missing else "PASS", "detail": f"missing={missing}"})
            pred_latest = str(pd.to_datetime(df["Date"]).max().date()) if "Date" in df.columns else None
            c_latest = cache_latest.get(ticker)
            if c_latest and pred_latest and c_latest > pred_latest:
                checks.append({"check_name": "prediction_date_vs_cache_date", "ticker": ticker, "status": "WARN", "detail": f"cache_latest={c_latest}, prediction_latest={pred_latest}"})
            else:
                checks.append({"check_name": "prediction_date_vs_cache_date", "ticker": ticker, "status": "PASS", "detail": f"cache_latest={c_latest}, prediction_latest={pred_latest}"})
        ranges = {}
        for ticker, df in prediction_frames.items():
            if df is not None and not df.empty and "Date" in df.columns:
                d = pd.to_datetime(df["Date"])
                ranges[ticker] = (str(d.min().date()), str(d.max().date()))
        if len(set(ranges.values())) > 1:
            checks.append({"check_name": "asset_date_range_mismatch", "status": "WARN", "detail": str(ranges)})
        else:
            checks.append({"check_name": "asset_date_range_mismatch", "status": "PASS", "detail": str(ranges)})
        return checks

    def validate_allocation(self, allocation_payload: Dict) -> List[Dict]:
        totals = allocation_payload.get("portfolio_totals", {})
        s = float(totals.get("stock_weight", 0)) + float(totals.get("bond_weight", 0)) + float(totals.get("cash_weight", 0))
        return [{"check_name": "portfolio_weight_sum", "status": "PASS" if abs(s - 1.0) <= 1e-3 else "FAIL", "detail": f"sum={s:.6f}"}]

    @staticmethod
    def summarize(checks: List[Dict]) -> Dict:
        return {
            "fail_count": sum(1 for c in checks if c.get("status") == "FAIL"),
            "warn_count": sum(1 for c in checks if c.get("status") == "WARN"),
            "pass_count": sum(1 for c in checks if c.get("status") == "PASS"),
            "checks": checks,
        }
