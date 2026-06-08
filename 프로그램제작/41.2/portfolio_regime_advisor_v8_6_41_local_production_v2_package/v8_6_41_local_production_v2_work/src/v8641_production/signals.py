"""Signal classification layer."""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import numpy as np
import pandas as pd

from .config import ProductionConfig
from .schemas import AssetData, SignalSnapshot, to_plain_dict
from .constants import (
    PRED_DIRECTION_DOWN_KEYWORDS,
    PRED_DIRECTION_UP_KEYWORDS,
    PRED_RISK_HIGH_KEYWORDS,
    SIGNAL_THRESHOLDS,
)
from .utils import MathUtils


class SignalClassifier:
    """Convert latest model outputs into UI-friendly classes."""

    def __init__(self, config: ProductionConfig):
        self.config = config

    def latest_snapshots(self, assets: Dict[str, AssetData]) -> Dict[str, SignalSnapshot]:
        return {ticker: self.latest_snapshot(asset) for ticker, asset in assets.items()}

    def latest_snapshot(self, asset: AssetData) -> SignalSnapshot:
        row = asset.predictions.iloc[-1]
        return SignalSnapshot(
            ticker=asset.ticker,
            date=str(pd.to_datetime(row["Date"]).date()),
            model_version=self.config.model_version,
            pred_risk=str(row.get("pred_risk", "")),
            pred_direction=str(row.get("pred_direction", "")),
            pred_overall_risk=str(row.get("pred_overall_risk", "")),
            signal_regime=str(row.get("signal_regime", "")),
            allocation_regime=str(row.get("allocation_regime", "")),
            executed_regime=str(row.get("executed_regime", "")),
            prob_normal=MathUtils.safe_float(row.get("prob_normal")),
            prob_high_vol=MathUtils.safe_float(row.get("prob_high_vol")),
            prob_overall_risk=MathUtils.safe_float(row.get("prob_overall_risk")),
            prob_up_strengthening_5d=MathUtils.safe_float(row.get("prob_up_strengthening_5d")),
            prob_up_strengthening_10d=MathUtils.safe_float(row.get("prob_up_strengthening_10d")),
            prob_up_strengthening_20d=MathUtils.safe_float(row.get("prob_up_strengthening_20d")),
            prob_up_strengthening_score=MathUtils.safe_float(row.get("prob_up_strengthening_score")),
            prob_down_strengthening_5d=MathUtils.safe_float(row.get("prob_down_strengthening_5d")),
            prob_down_strengthening_10d=MathUtils.safe_float(row.get("prob_down_strengthening_10d")),
            prob_down_strengthening_20d=MathUtils.safe_float(row.get("prob_down_strengthening_20d")),
            prob_down_strengthening_score=MathUtils.safe_float(row.get("prob_down_strengthening_score")),
            signal_stock_weight=MathUtils.safe_float(row.get("signal_stock_weight")),
            signal_bond_weight=MathUtils.safe_float(row.get("signal_bond_weight")),
            signal_cash_weight=MathUtils.safe_float(row.get("signal_cash_weight")),
            executed_stock_weight=MathUtils.safe_float(row.get("stock_weight")),
            executed_bond_weight=MathUtils.safe_float(row.get("bond_weight")),
            executed_cash_weight=MathUtils.safe_float(row.get("cash_weight")),
            offensive_active=self._as_bool(row.get("offensive_active")),
            offensive_tier=MathUtils.safe_float(row.get("offensive_tier")),
            full_stock_signal=self._as_bool(row.get("full_stock_signal")),
            risk_class=self._risk_class(row),
            direction_class=self._direction_class(row),
            allocation_class=self._allocation_class(row),
            monitoring_note=self._monitoring_note(row),
        )

    @staticmethod
    def as_ui_list(snapshots: Dict[str, SignalSnapshot]) -> list[dict]:
        return [to_plain_dict(snapshot) for snapshot in snapshots.values()]

    @staticmethod
    def _as_bool(x) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)) and not pd.isna(x):
            return bool(x)
        if isinstance(x, str):
            return x.strip().lower() in {"true", "1", "yes", "y"}
        return False

    @staticmethod
    def _risk_class(row: pd.Series) -> str:
        pred = str(row.get("pred_risk", ""))
        phv = MathUtils.safe_float(row.get("prob_high_vol"), np.nan)
        pdown = MathUtils.safe_float(row.get("prob_down_strengthening_score"), np.nan)
        if pred in PRED_RISK_HIGH_KEYWORDS or (
            np.isfinite(phv) and phv >= SIGNAL_THRESHOLDS["high_risk_prob_high_vol"]
        ):
            return "HIGH_RISK"
        if (
            np.isfinite(phv) and phv >= SIGNAL_THRESHOLDS["watch_prob_high_vol"]
        ) or (
            np.isfinite(pdown) and pdown >= SIGNAL_THRESHOLDS["watch_prob_down_strengthening_score"]
        ):
            return "WATCH"
        return "NORMAL"

    @staticmethod
    def _direction_class(row: pd.Series) -> str:
        pred = str(row.get("pred_direction", ""))
        pup = MathUtils.safe_float(row.get("prob_up_strengthening_score"), np.nan)
        pdown = MathUtils.safe_float(row.get("prob_down_strengthening_score"), np.nan)
        if pred in PRED_DIRECTION_UP_KEYWORDS or (
            np.isfinite(pup)
            and pup >= SIGNAL_THRESHOLDS["up_strength_score"]
            and (not np.isfinite(pdown) or pup > pdown)
        ):
            return "UP_STRENGTH"
        if pred in PRED_DIRECTION_DOWN_KEYWORDS or (
            np.isfinite(pdown)
            and pdown >= SIGNAL_THRESHOLDS["down_strength_score"]
            and (not np.isfinite(pup) or pdown > pup)
        ):
            return "DOWN_STRENGTH"
        return "NEUTRAL"

    @staticmethod
    def _allocation_class(row: pd.Series) -> str:
        stock = MathUtils.safe_float(row.get("stock_weight"), np.nan)
        if not np.isfinite(stock):
            return "UNKNOWN"
        if stock >= 0.90:
            return "AGGRESSIVE"
        if stock >= 0.75:
            return "PARTICIPATION"
        if stock >= 0.55:
            return "BALANCED"
        return "DEFENSIVE"

    @staticmethod
    def _monitoring_note(row: pd.Series) -> str:
        phv = MathUtils.safe_float(row.get("prob_high_vol"), np.nan)
        pup = MathUtils.safe_float(row.get("prob_up_strengthening_score"), np.nan)
        pdown = MathUtils.safe_float(row.get("prob_down_strengthening_score"), np.nan)
        if np.isfinite(pdown) and pdown >= SIGNAL_THRESHOLDS["watch_prob_down_strengthening_score"]:
            return "DOWN_STRENGTH_WATCH"
        if np.isfinite(phv) and phv >= SIGNAL_THRESHOLDS["watch_prob_high_vol"]:
            return "HIGH_VOL_WATCH"
        if np.isfinite(pup) and pup >= SIGNAL_THRESHOLDS["up_strength_score"]:
            return "UP_STRENGTH_PARTICIPATION"
        return "NORMAL_MONITORING"
