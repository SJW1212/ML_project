"""Application service for UI/backend integration."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import pandas as pd

from .allocation import AllocationPolicy
from .config import ProductionConfig
from .constants import CANCELLED_LAYERS
from .performance import PerformanceAnalyzer
from .repository import DataRepository
from .schemas import to_plain_dict
from .serializer import OutputWriter
from .signals import SignalClassifier
from .validation import Validator


class ProductionService:
    """Main façade. UI backends should call build_dashboard_payload()."""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.config.validate()
        self.repository = DataRepository(config)
        self.signal_classifier = SignalClassifier(config)
        self.allocation_policy = AllocationPolicy(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.validator = Validator(config)

    def build_dashboard_payload(self) -> Dict[str, Any]:
        assets = self.repository.load_all()
        snapshots = self.signal_classifier.latest_snapshots(assets)
        capital_weights = self.allocation_policy.capital_weights(assets)
        allocation_rows = self.allocation_policy.build_allocation_rows(snapshots, capital_weights)
        totals = self.allocation_policy.totals(allocation_rows)
        performance_rows = self.performance_analyzer.summarize_all(assets)
        portfolio_daily = self.performance_analyzer.portfolio_daily_returns(assets, capital_weights)
        annual = self.performance_analyzer.annual_returns(portfolio_daily)
        monthly = self.performance_analyzer.monthly_returns(portfolio_daily)
        checks = self.validator.validate(
            assets,
            allocation_rows,
            totals,
            allocation_diagnostics={
                "three_way_fallback_tickers": self.allocation_policy.three_way_fallback_tickers,
                "inverse_vol_metadata": self.allocation_policy.inverse_vol_metadata,
            },
        )
        fail_count = sum(1 for check in checks if check.status == "FAIL")
        warn_count = sum(1 for check in checks if check.status == "WARN")

        as_of_date = totals.date or max(snapshot.date for snapshot in snapshots.values())
        # UI payload: compact, explicit, no filesystem-coupled CSV dependency.
        payload = {
            "model_version": self.config.model_version,
            "as_of_date": as_of_date,
            "allocation_source": self.config.allocation_source,
            "capital_mode": self.config.capital_mode,
            "cancelled_layers": CANCELLED_LAYERS,
            "latest_signals": self.signal_classifier.as_ui_list(snapshots),
            "portfolio_allocation": self.allocation_policy.as_ui_list(allocation_rows),
            "portfolio_totals": to_plain_dict(totals),
            "performance_summary": self.performance_analyzer.as_ui_list(performance_rows),
            "charts": {
                "portfolio_equity_curve": self._tail_records(portfolio_daily, 756),
                "annual_returns": annual,
                "monthly_returns": monthly[-36:],
            },
            "validation": {
                "fail_count": fail_count,
                "warn_count": warn_count,
                "pass_count": sum(1 for check in checks if check.status == "PASS"),
                "checks": self.validator.as_ui_list(checks),
            },
            "ui_notes": {
                "output_policy": "JSON payload first. CSV is optional debug/export only.",
                "base_model_policy": "v8.6.41_label_fixed native allocations only. No loss guard or confidence/logit overlay.",
                "local_runtime_scope": "Local API/dashboard only. No DB, no user-account portfolio storage, no notifications, no Pixso-specific mapping, no realtime streaming, no order execution.",
                "data_update_policy": "Daily OHLCV/cache update only. Prediction files remain the source of allocation decisions. Cache and prediction dates may differ by design.",
                "portfolio_return_join_policy": "Portfolio return charts use outer join across assets; missing per-asset returns are filled with 0.0 and date mismatches are surfaced as WARN validations.",
            },
        }
        return payload

    def run_and_write(self) -> Dict[str, Any]:
        payload = self.build_dashboard_payload()
        writer = OutputWriter(self.config.out_dir)
        if self.config.export_json:
            writer.write_json("dashboard_payload.json", payload)
            writer.write_json("latest_state.json", {
                "model_version": payload["model_version"],
                "as_of_date": payload["as_of_date"],
                "latest_signals": payload["latest_signals"],
                "portfolio_totals": payload["portfolio_totals"],
                "portfolio_allocation": payload["portfolio_allocation"],
                "validation": payload["validation"],
            })
        if self.config.export_csv:
            writer.write_csv_bundle(payload)
        if self.config.export_markdown:
            writer.write_markdown_report(payload)
        if self.config.make_zip:
            writer.make_zip()
        return payload

    @staticmethod
    def _tail_records(df: pd.DataFrame, n: int) -> list[dict]:
        if df.empty:
            return []
        out = df.tail(n).copy()
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
        return out.to_dict(orient="records")
