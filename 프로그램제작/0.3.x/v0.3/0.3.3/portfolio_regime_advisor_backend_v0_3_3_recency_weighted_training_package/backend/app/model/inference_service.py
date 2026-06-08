from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..data.market_data_repository import MarketDataRepository
from ..features.feature_pipeline import FeaturePipeline
from ..portfolio.allocation_policy_engine import AllocationPolicyEngine
from .model_loader import ModelLoader
from .prediction_service import PredictionService


class InferenceService:
    """Live inference using saved runtime model artifacts.

    This service is intentionally separate from PredictionService.
    - PredictionService: load precomputed v8.6.41 prediction CSV files.
    - InferenceService: load OHLCV cache, build features, load model artifacts, predict latest probabilities.
    """

    HEADS = ["highvol", "up_strength", "down_strength"]

    def __init__(self, model_loader: ModelLoader, feature_pipeline: FeaturePipeline, policy_engine: AllocationPolicyEngine | None = None):
        self.model_loader = model_loader
        self.feature_pipeline = feature_pipeline
        self.policy_engine = policy_engine or AllocationPolicyEngine()

    @staticmethod
    def _predict_proba_one(model, X: pd.DataFrame) -> float:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] > 1:
                return float(proba[:, 1][0])
            return float(proba[:, 0][0])
        pred = model.predict(X)
        return float(np.asarray(pred).ravel()[0])

    @staticmethod
    def _clip01(value: float) -> float:
        try:
            if not np.isfinite(value):
                return float("nan")
            return float(np.clip(value, 0.0, 1.0))
        except Exception:
            return float("nan")

    def artifact_status(self, model_version: str, tickers: Iterable[str], horizons: Iterable[str]) -> Dict:
        return self.model_loader.artifact_status(model_version, tickers, horizons)

    def infer_ticker(
        self,
        model_version: str,
        ticker: str,
        ohlcv: pd.DataFrame,
        horizons: Iterable[str],
        selected_horizon: str = "10D",
    ) -> Dict:
        horizons = [h.upper() for h in horizons]
        selected_horizon = selected_horizon.upper()
        missing = self.model_loader.missing_artifacts(model_version, ticker, horizons)
        if missing:
            raise FileNotFoundError(f"Missing model artifacts for {ticker}: {missing[:3]}{'...' if len(missing) > 3 else ''}")

        feature_row = self.feature_pipeline.latest_feature_row(ohlcv)
        X = feature_row[self.feature_pipeline.FEATURE_COLUMNS]
        result = {"ticker": ticker.upper(), "Date": str(pd.to_datetime(feature_row["Date"].iloc[0]).date())}

        for horizon in horizons:
            hnum = horizon.upper().replace("D", "")
            for head in self.HEADS:
                model = self.model_loader.load(model_version, ticker, head, horizon)
                prob = self._clip01(self._predict_proba_one(model, X))
                if head == "highvol":
                    result[f"prob_high_vol_{hnum}d"] = prob
                    result[f"prob_high_vol_h{hnum}"] = prob
                elif head == "up_strength":
                    result[f"prob_up_strengthening_{hnum}d"] = prob
                elif head == "down_strength":
                    result[f"prob_down_strengthening_{hnum}d"] = prob

        # selected horizon values first, average fallback second
        hsel = selected_horizon.replace("D", "")
        high_values = [result[k] for k in result if k.startswith("prob_high_vol_") and k.endswith("d")]
        up_values = [result[k] for k in result if k.startswith("prob_up_strengthening_") and k.endswith("d")]
        down_values = [result[k] for k in result if k.startswith("prob_down_strengthening_") and k.endswith("d")]

        fallback_warnings = []
        if f"prob_high_vol_{hsel}d" in result:
            ph = result[f"prob_high_vol_{hsel}d"]
        else:
            ph = float(np.nanmean(high_values)) if high_values else np.nan
            fallback_warnings.append("HIGH_VOL_SELECTED_HORIZON_MISSING_AVERAGE_USED")
        if f"prob_up_strengthening_{hsel}d" in result:
            pu = result[f"prob_up_strengthening_{hsel}d"]
        else:
            pu = float(np.nanmean(up_values)) if up_values else np.nan
            fallback_warnings.append("UP_SELECTED_HORIZON_MISSING_AVERAGE_USED")
        if f"prob_down_strengthening_{hsel}d" in result:
            pdn = result[f"prob_down_strengthening_{hsel}d"]
        else:
            pdn = float(np.nanmean(down_values)) if down_values else np.nan
            fallback_warnings.append("DOWN_SELECTED_HORIZON_MISSING_AVERAGE_USED")

        result["prob_high_vol"] = self._clip01(ph)
        result["prob_normal"] = self._clip01(1.0 - result["prob_high_vol"])
        result["prob_overall_risk"] = result["prob_high_vol"]
        result["prob_up_strengthening_score"] = self._clip01(pu)
        result["prob_down_strengthening_score"] = self._clip01(pdn)
        result["selected_horizon"] = selected_horizon
        result["selected_prob_high_vol"] = result["prob_high_vol"]
        result["selected_prob_up_strengthening"] = result["prob_up_strengthening_score"]
        result["selected_prob_down_strengthening"] = result["prob_down_strengthening_score"]
        # Use the shared allocation policy. This keeps live inference and prediction_file
        # routes comparable instead of maintaining a second hidden policy here.
        result.update(self.policy_engine.apply_row(result, preserve_existing=False))

        # Reuse current classification/comment policy for dashboard compatibility.
        row = pd.Series(result)
        risk, direction, allocation, warnings, comment = PredictionService.classify(row)
        result.update({
            "risk_class": risk,
            "direction_class": direction,
            "allocation_class": allocation,
            "warnings": list(dict.fromkeys((warnings or []) + fallback_warnings)),
            "comment": comment,
        })
        return result

    def infer_from_cache(
        self,
        *,
        model_version: str,
        tickers: Iterable[str],
        horizon: str,
        provider: str,
        market: str,
        repository: MarketDataRepository,
        horizons_for_model: Optional[Iterable[str]] = None,
    ) -> tuple[pd.DataFrame, List[Dict]]:
        horizons_for_model = list(horizons_for_model or ["5D", "10D", "20D"])
        provider_candidates = [provider]
        if provider != "auto":
            provider_candidates.append("auto")
        provider_candidates.extend(["yahoo", "kis"])

        items: List[Dict] = []
        errors: List[Dict] = []
        for ticker in tickers:
            ticker = ticker.upper()
            df = None
            used_provider = None
            for p in dict.fromkeys(provider_candidates):
                df = repository.load_ohlcv(p, ticker, market)
                if df is not None and not df.empty:
                    used_provider = p
                    break
            if df is None or df.empty:
                errors.append({"ticker": ticker, "code": "MARKET_DATA_NOT_FOUND", "message": f"Cached OHLCV not found for {ticker}. Run POST /market-data/update first."})
                continue
            try:
                result = self.infer_ticker(model_version, ticker, df, horizons=horizons_for_model, selected_horizon=horizon)
                result["market_data_provider"] = used_provider
                result["model_mode"] = "live_inference"
                items.append(result)
            except Exception as exc:
                errors.append({"ticker": ticker, "code": "INFERENCE_FAILED", "message": str(exc)})

        return pd.DataFrame(items), errors
