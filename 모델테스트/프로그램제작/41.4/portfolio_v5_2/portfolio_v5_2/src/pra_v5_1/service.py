from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .allocation import PortfolioAllocationService
from .benchmarks import BenchmarkService
from .cache import MarketDataCache
from .config import AppConfig
from .logging_utils import JsonlLogger
from .performance import PerformanceAnalyzer
from .prediction_engine import PredictionGenerationService, PredictionRepository
from .provider import DailyMarketDataUpdater
from .schemas import DailyUpdateRequest, PortfolioAssetInput, PortfolioEvaluateRequest, PredictionGenerateRequest
from .signals import SignalClassifier
from .ticker_registry import LocalTickerRegistry
from .utils import config_hash, ensure_dir, normalize_ticker, utc_now_iso
from .ui_contract import postprocess_dashboard_payload
from .validation import ValidationService


class PortfolioRegimeAdvisorService:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.from_json()
        ensure_dir(self.config.storage_root)
        self.registry = LocalTickerRegistry(self.config.registry_path)
        self.cache = MarketDataCache(self.config.cache_dir)
        self.updater = DailyMarketDataUpdater(self.cache)
        self.pred_repo = PredictionRepository(self.config.prediction_dir)
        self.pred_gen = PredictionGenerationService(self.cache, self.pred_repo, self.config.run_dir)
        self.allocator = PortfolioAllocationService()
        self.performance = PerformanceAnalyzer()
        self.benchmarks = BenchmarkService(self.cache, self.performance)
        self.validator = ValidationService()
        self.signals = SignalClassifier()
        self.logger = JsonlLogger(self.config.log_dir / "events.jsonl")

    def add_tickers(self, tickers: List[str], asset_type: str = "stock", market: str = "US", note: str | None = None) -> Dict:
        records = self.registry.add_or_update(tickers, asset_type=asset_type, market=market, note=note)
        return {"ok": True, "tickers": [r.__dict__ for r in records]}

    def update_daily(self, req: DailyUpdateRequest) -> Dict:
        tickers = req.tickers or self.registry.enabled_tickers()
        status = self.updater.update_daily(tickers, provider=req.provider, start=req.start, end=req.end, force=req.force)
        payload = {"provider": status.provider, "updated": status.updated, "skipped": status.skipped, "errors": status.errors}
        self.logger.write("daily_update", **payload)
        return payload

    def generate_predictions(self, req: PredictionGenerateRequest, risk_sensitivity: float = 1.0) -> Dict:
        tickers = req.tickers or self.registry.enabled_tickers()
        if req.mode == "external_v8641_xgb":
            return {"ok": False, "error": "external_v8641_xgb wrapper is reserved. Use reference_v8641_compatible in this package or plug the original script through model_engine/."}
        results = self.pred_gen.generate_reference(tickers, provider="yahoo", force=req.force, risk_sensitivity=risk_sensitivity)
        payload = {"ok": all(not r.errors for r in results), "results": [r.__dict__ | {"path": str(r.path)} for r in results]}
        self.logger.write("prediction_generation", **payload)
        return payload

    @staticmethod
    def _risk_assets(portfolio: List[PortfolioAssetInput]) -> List[PortfolioAssetInput]:
        return [a for a in portfolio if a.is_risk_asset]

    def evaluate(self, request: PortfolioEvaluateRequest) -> Dict:
        request_id = config_hash(request.model_dump())
        self.logger.write("portfolio_evaluate_started", request_id=request_id, user_level=request.user_level)
        risk_assets = self._risk_assets(request.portfolio)
        risk_tickers = [a.ticker for a in risk_assets]
        if not risk_tickers:
            raise ValueError("At least one risk asset ticker is required")
        # Registry update is local JSON only, not user account storage.
        self.registry.add_or_update(risk_tickers, asset_type="risk_asset", market="US", note="from_evaluate_request")
        # Data update includes risk assets only. Defensive bucket and CASH are not downloaded.
        if request.settings.update_data:
            update_req = DailyUpdateRequest(tickers=risk_tickers, provider=request.settings.provider, start=request.settings.start_date, end=request.settings.end_date, force=request.settings.force_update)
            update_status = self.update_daily(update_req)
            if update_status.get("errors"):
                # Continue only for tickers that already have cache.
                pass
        else:
            update_status = {"provider": request.settings.provider, "updated": [], "skipped": [], "errors": {}}
        if request.settings.generate_predictions:
            gen_req = PredictionGenerateRequest(tickers=risk_tickers, start=request.settings.start_date, end=request.settings.end_date, mode=request.settings.prediction_engine_mode, force=request.settings.force_prediction)
            gen_status = self.generate_predictions(gen_req, risk_sensitivity=request.settings.risk_sensitivity)
        else:
            gen_status = {"ok": True, "results": []}
        frames: Dict[str, pd.DataFrame] = {}
        latest_predictions: Dict[str, Dict] = {}
        latest_signals: List[Dict] = []
        for t in risk_tickers:
            df = self.pred_repo.read(t)
            frames[t] = df
            latest = df.iloc[-1].to_dict()
            latest["Date"] = str(latest.get("Date"))
            cls = self.signals.classify(latest)
            latest.update(cls)
            latest_predictions[t] = latest
            latest_signals.append({k: latest.get(k) for k in [
                "ticker", "Date", "prob_normal", "prob_high_vol", "prob_overall_risk",
                "prob_up_strengthening_5d", "prob_up_strengthening_10d", "prob_up_strengthening_20d", "prob_up_strengthening_score",
                "prob_down_strengthening_5d", "prob_down_strengthening_10d", "prob_down_strengthening_20d", "prob_down_strengthening_score",
                "risk_class", "direction_class", "allocation_class", "stock_weight", "bond_weight", "cash_weight"
            ]})
        allocation = self.allocator.allocate(request, latest_predictions)
        perf_weights = {row["ticker"]: row["recommended_weight"] for row in allocation["allocation_rows"]}
        perf_df = self.performance.portfolio_returns(frames, perf_weights, missing_asset_policy=request.settings.missing_asset_policy)
        recommended_returns = perf_df["portfolio_return"] if not perf_df.empty else pd.Series(dtype=float)
        metrics = self.performance.metrics(recommended_returns)
        current_weights = {a.ticker: float(a.current_weight or 0.0) for a in risk_assets}
        benchmarks = self.benchmarks.build(frames, current_weights, recommended_returns, provider=request.settings.provider)
        freshness = self.cache.freshness(risk_tickers, provider=request.settings.provider)
        checks = []
        checks.extend(self.validator.validate_predictions(frames, freshness))
        checks.extend(self.validator.validate_allocation(allocation))
        validation = self.validator.summarize(checks)
        payload = {
            "ok": validation["fail_count"] == 0,
            "request_id": request_id,
            "model_version": "v8.6.41_model_label_fixed_compatible_reference",
            "engine_mode": request.settings.prediction_engine_mode,
            "as_of_date": max((s.get("Date") for s in latest_signals if s.get("Date")), default=None),
            "portfolio_input": [a.model_dump() for a in request.portfolio],
            "latest_signals": latest_signals,
            "allocation": allocation,
            "performance": {
                "metrics": metrics,
                "equity_curve_tail": perf_df.tail(20).assign(Date=lambda x: x["Date"].astype(str)).to_dict(orient="records") if not perf_df.empty else [],
            },
            "benchmarks": benchmarks,
            "data_update_status": update_status,
            "prediction_generation_status": gen_status,
            "cache_freshness": freshness,
            "validation": validation,
        }
        payload = postprocess_dashboard_payload(payload, perf_df=perf_df)
        self.logger.write("portfolio_evaluate_finished", request_id=request_id, ok=payload["ok"], fail_count=validation["fail_count"], warn_count=validation["warn_count"])
        return payload
