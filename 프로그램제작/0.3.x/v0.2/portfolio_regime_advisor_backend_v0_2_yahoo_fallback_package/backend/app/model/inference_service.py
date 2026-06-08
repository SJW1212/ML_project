from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ..features.feature_pipeline import FeaturePipeline
from .model_loader import ModelLoader


class InferenceService:
    """Live inference using saved model artifacts.

    This is intentionally separate from PredictionService. MVP can run without artifacts,
    while this service enables future API-data based inference.
    """

    HEADS = ["highvol", "up_strength", "down_strength"]

    def __init__(self, model_loader: ModelLoader, feature_pipeline: FeaturePipeline):
        self.model_loader = model_loader
        self.feature_pipeline = feature_pipeline

    @staticmethod
    def _predict_proba_one(model, X: pd.DataFrame) -> float:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] > 1:
                return float(proba[:, 1][0])
            return float(proba[:, 0][0])
        pred = model.predict(X)
        return float(np.asarray(pred).ravel()[0])

    def infer_ticker(self, model_version: str, ticker: str, ohlcv: pd.DataFrame, horizons: Iterable[str]) -> Dict:
        feature_row = self.feature_pipeline.latest_feature_row(ohlcv)
        X = feature_row[self.feature_pipeline.FEATURE_COLUMNS]
        result = {"ticker": ticker.upper(), "Date": str(pd.to_datetime(feature_row["Date"].iloc[0]).date())}
        for horizon in horizons:
            h = horizon.lower()
            for head in self.HEADS:
                model = self.model_loader.load(model_version, ticker, head, horizon)
                result[f"prob_{head}_{h}"] = self._predict_proba_one(model, X)
        # aggregate to v8.6.41-compatible shape
        vals_high = [result[f"prob_highvol_{h.lower()}"] for h in horizons if f"prob_highvol_{h.lower()}" in result]
        vals_up = [result[f"prob_up_strength_{h.lower()}"] for h in horizons if f"prob_up_strength_{h.lower()}" in result]
        vals_down = [result[f"prob_down_strength_{h.lower()}"] for h in horizons if f"prob_down_strength_{h.lower()}" in result]
        result["prob_high_vol"] = float(np.nanmean(vals_high)) if vals_high else np.nan
        result["prob_normal"] = 1.0 - result["prob_high_vol"] if np.isfinite(result["prob_high_vol"]) else np.nan
        result["prob_overall_risk"] = result["prob_high_vol"]
        result["prob_up_strengthening_score"] = float(np.nanmean(vals_up)) if vals_up else np.nan
        result["prob_down_strengthening_score"] = float(np.nanmean(vals_down)) if vals_down else np.nan
        return result
