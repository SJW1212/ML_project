from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import pandas as pd


REQUIRED_PREDICTION_COLUMNS = [
    "Date",
    "prob_normal",
    "prob_high_vol",
    "prob_overall_risk",
    "prob_up_strengthening_5d",
    "prob_up_strengthening_10d",
    "prob_up_strengthening_20d",
    "prob_up_strengthening_score",
    "prob_down_strengthening_5d",
    "prob_down_strengthening_10d",
    "prob_down_strengthening_20d",
    "prob_down_strengthening_score",
    "stock_weight",
    "bond_weight",
    "cash_weight",
    "stock_next_return",
]


@dataclass(frozen=True)
class FeatureSchema:
    allowed_raw_columns: List[str] = field(default_factory=lambda: ["Date", "Open", "High", "Low", "Close", "Volume"])
    allowed_feature_prefixes: List[str] = field(default_factory=lambda: ["ret_", "vol_", "mom_", "ma_", "drawdown_", "range_", "volume_"])
    forbidden_prefixes: List[str] = field(default_factory=lambda: ["y_", "future_", "label_", "target_"])
    forbidden_suffixes: List[str] = field(default_factory=lambda: ["_future", "_label", "_target"])

    def validate_raw(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.allowed_raw_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Raw OHLCV missing columns: {missing}")

    def validate_features(self, df: pd.DataFrame) -> None:
        bad = []
        for col in df.columns:
            c = str(col)
            if c in {"Date", "Close"}:
                continue
            if any(c.startswith(p) for p in self.forbidden_prefixes) or any(c.endswith(s) for s in self.forbidden_suffixes):
                bad.append(c)
            elif not any(c.startswith(p) for p in self.allowed_feature_prefixes):
                bad.append(c)
        if bad:
            raise ValueError(f"Feature leakage/schema violation columns: {bad[:20]}")


def validate_prediction_output(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction output missing columns: {missing}")
