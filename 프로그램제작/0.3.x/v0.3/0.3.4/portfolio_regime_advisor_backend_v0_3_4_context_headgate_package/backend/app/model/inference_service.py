from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..data.market_data_repository import MarketDataRepository
from ..features.context_asset_universe import ContextAssetUniverse
from ..features.feature_pipeline import FeaturePipeline
from ..features.market_context_feature_builder import MarketContextFeatureBuilder
from ..portfolio.allocation_policy_engine import AllocationPolicyEngine
from .model_loader import ModelLoader
from .prediction_service import PredictionService
from .selective_inference_service import SelectiveInferenceService


class InferenceService:
    """Live inference using saved runtime model artifacts.

    This service is intentionally separate from PredictionService.
    - PredictionService: load precomputed v8.6.41 prediction CSV files.
    - InferenceService: load OHLCV cache, build features, load model artifacts, predict latest probabilities.
    """

    HEADS = ["highvol", "up_strength", "down_strength"]

    def __init__(self, model_loader: ModelLoader, feature_pipeline: FeaturePipeline, policy_engine: AllocationPolicyEngine | None = None, selective_service: SelectiveInferenceService | None = None):
        self.model_loader = model_loader
        self.feature_pipeline = feature_pipeline
        self.policy_engine = policy_engine or AllocationPolicyEngine()
        self.selective_service = selective_service or SelectiveInferenceService()

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

        expected_cols = self.model_loader.read_feature_columns(model_version, ticker) or list(self.feature_pipeline.FEATURE_COLUMNS)
        features = self.feature_pipeline.build(ohlcv, include_labels=False)
        if any(str(c).startswith("ctx_") for c in expected_cols):
            # Context features are loaded from the same repository/provider cache in infer_from_cache.
            # infer_ticker receives repository/provider through temporary attributes set by infer_from_cache.
            repo = getattr(self, "_current_repository", None)
            provider = getattr(self, "_current_provider", "auto")
            market = getattr(self, "_current_market", "US")
            if repo is not None:
                try:
                    target_index = pd.DatetimeIndex(pd.to_datetime(features["Date"]))
                    universe = ContextAssetUniverse(repo, provider=provider, market=market, target_index=target_index, strict=False)
                    raw = ohlcv.copy()
                    raw = raw.rename(columns={c: c.title() for c in raw.columns if c.lower() in {"date", "close"}})
                    raw["Date"] = pd.to_datetime(raw["Date"])
                    close_map = raw.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")["Close"]
                    features = features.copy()
                    features["Close"] = pd.to_datetime(features["Date"]).map(close_map)
                    features = MarketContextFeatureBuilder().build(features, universe)
                except Exception:
                    pass
        missing_cols = [c for c in expected_cols if c not in features.columns]
        for c in missing_cols:
            features[c] = np.nan
        row = features.dropna(subset=[c for c in expected_cols if c in features.columns], how="all").tail(1)
        if row.empty:
            raise ValueError("Not enough OHLCV/context history to build a valid latest feature row.")
        feature_row = row[["Date"] + expected_cols].copy()
        X = feature_row[expected_cols]
        result = {"ticker": ticker.upper(), "Date": str(pd.to_datetime(feature_row["Date"].iloc[0]).date())}
        for c in [c for c in features.columns if str(c).startswith("ctx_")]:
            try:
                result[c] = float(row[c].iloc[0]) if pd.notna(row[c].iloc[0]) else None
            except Exception:
                result[c] = None

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

        manifest = self.model_loader.read_manifest(model_version, ticker)
        metrics = manifest.get("metrics") or {}
        hsel_key = selected_horizon.upper()
        live_probs = {
            f"highvol_{hsel_key}": result["prob_high_vol"],
            f"up_strength_{hsel_key}": result["prob_up_strengthening_score"],
            f"down_strength_{hsel_key}": result["prob_down_strengthening_score"],
        }
        selective = self.selective_service.adjust(
            live_probs=live_probs,
            metrics=metrics,
            baseline_probs=None,
            context_features={k: v for k, v in result.items() if str(k).startswith("ctx_")},
        )
        adjusted = selective.get("adjusted_probs", {})
        result["prob_high_vol"] = self._clip01(adjusted.get(f"highvol_{hsel_key}", result["prob_high_vol"]))
        result["prob_normal"] = self._clip01(1.0 - result["prob_high_vol"])
        result["prob_overall_risk"] = result["prob_high_vol"]
        result["prob_up_strengthening_score"] = self._clip01(adjusted.get(f"up_strength_{hsel_key}", result["prob_up_strengthening_score"]))
        result["prob_down_strengthening_score"] = self._clip01(adjusted.get(f"down_strength_{hsel_key}", result["prob_down_strengthening_score"]))
        if adjusted.get("risk_override") == "WATCH":
            result["risk_override"] = "WATCH"
        result["head_gates"] = selective.get("head_gates", {})
        result["selective_warnings"] = selective.get("selective_warnings", [])
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
            "warnings": list(dict.fromkeys((warnings or []) + fallback_warnings + result.get("selective_warnings", []))),
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
                self._current_repository = repository
                self._current_provider = used_provider or provider
                self._current_market = market
                result = self.infer_ticker(model_version, ticker, df, horizons=horizons_for_model, selected_horizon=horizon)
                result["market_data_provider"] = used_provider
                result["model_mode"] = "live_inference"
                items.append(result)
            except Exception as exc:
                errors.append({"ticker": ticker, "code": "INFERENCE_FAILED", "message": str(exc)})

        return pd.DataFrame(items), errors
