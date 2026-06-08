"""Model-output loading layer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import ProductionConfig
from .schemas import AssetData
from .utils import DateUtils


class FileResolver:
    """Find v8.6.41 prediction and summary files."""

    def __init__(self, config: ProductionConfig):
        self.config = config

    def prediction_file(self, ticker: str) -> Optional[Path]:
        t = ticker.lower()
        base = self.config.input_dir
        candidates = [
            base / f"{t}_{self.config.source_tag}_predictions.csv",
            base / ticker / f"{t}_{self.config.source_tag}_predictions.csv",
            base / t / f"{t}_{self.config.source_tag}_predictions.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        matches = sorted(base.rglob(f"{t}_*model_label_fixed_predictions.csv"))
        return matches[0] if matches else None

    def summary_file(self, ticker: str) -> Optional[Path]:
        t = ticker.lower()
        base = self.config.input_dir
        candidates = [
            base / f"{t}_{self.config.source_tag}_summary.json",
            base / ticker / f"{t}_{self.config.source_tag}_summary.json",
            base / t / f"{t}_{self.config.source_tag}_summary.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        matches = sorted(base.rglob(f"{t}_*model_label_fixed_summary.json"))
        return matches[0] if matches else None


class DataRepository:
    """Load and validate final model outputs."""

    REQUIRED_COLUMNS = {
        "Date",
        "stock_next_return",
        "bond_next_return",
        "cash_next_return",
        "prob_high_vol",
        "prob_normal",
        "prob_overall_risk",
        "prob_up_strengthening_score",
        "prob_down_strengthening_score",
        "pred_risk",
        "pred_direction",
        "signal_stock_weight",
        "signal_bond_weight",
        "signal_cash_weight",
        "stock_weight",
        "bond_weight",
        "cash_weight",
        "strategy_return_net",
        "strategy_equity_net",
    }

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.resolver = FileResolver(config)

    def load_all(self) -> Dict[str, AssetData]:
        return {ticker: self.load_asset(ticker) for ticker in self.config.assets}

    def load_asset(self, ticker: str) -> AssetData:
        pred_path = self.resolver.prediction_file(ticker)
        if pred_path is None:
            raise FileNotFoundError(f"Prediction CSV not found for {ticker}")
        df = pd.read_csv(pred_path)
        df = DateUtils.ensure_datetime(df, "Date")
        missing = sorted(self.REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(f"{ticker} prediction file is missing required columns: {missing}")

        summary_path = self.resolver.summary_file(ticker)
        summary = {}
        if summary_path and summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
        return AssetData(
            ticker=ticker,
            prediction_path=pred_path,
            summary_path=summary_path,
            predictions=df,
            summary=summary,
        )
