from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..core.exceptions import DataNotFoundError


class PredictionRepository:
    """Loads v8.6.41 prediction/result files.

    MVP uses Prediction File Mode. Live inference can later write the same schema into this repository.
    """

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)

    def _prediction_candidates(self, ticker: str) -> List[Path]:
        t = ticker.lower()
        patterns = [
            f"{t}_xgb_recency_weighted_v8_6_41_model_label_fixed_predictions.csv",
            f"{t}_*v8_6_41_model_label_fixed_predictions.csv",
            f"{t}_*predictions.csv",
        ]
        paths: List[Path] = []
        for pattern in patterns:
            paths.extend(sorted(self.input_dir.glob(pattern)))
        # prefer explicit v8_6_41 files
        uniq = []
        seen = set()
        for p in paths:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq

    def find_prediction_file(self, ticker: str) -> Path:
        candidates = self._prediction_candidates(ticker)
        if not candidates:
            raise DataNotFoundError(f"Prediction file not found for ticker={ticker} in {self.input_dir}")
        candidates.sort(key=lambda p: ("v8_6_41_model_label_fixed" not in p.name, len(p.name)))
        return candidates[0]

    def load_predictions(self, ticker: str) -> pd.DataFrame:
        path = self.find_prediction_file(ticker)
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            raise DataNotFoundError(f"Prediction file has no Date column: {path}")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date", keep="last")
        df["ticker"] = ticker.upper()
        return df

    def load_latest(self, tickers: Iterable[str]) -> pd.DataFrame:
        rows = []
        errors = []
        for ticker in tickers:
            try:
                df = self.load_predictions(ticker)
                latest = df.tail(1).copy()
                rows.append(latest)
            except Exception as exc:  # partial success required
                errors.append({"ticker": ticker.upper(), "error": str(exc)})
        if not rows:
            raise DataNotFoundError(f"No prediction files loaded. errors={errors}")
        latest_df = pd.concat(rows, ignore_index=True)
        latest_df.attrs["load_errors"] = errors
        return latest_df

    def load_summary(self, ticker: str) -> Dict:
        t = ticker.lower()
        candidates = sorted(self.input_dir.glob(f"{t}_xgb_recency_weighted_v8_6_41_model_label_fixed_summary.json"))
        if not candidates:
            return {}
        try:
            with candidates[0].open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def load_many_predictions(self, tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
        out = {}
        for ticker in tickers:
            try:
                out[ticker.upper()] = self.load_predictions(ticker)
            except Exception:
                continue
        return out
