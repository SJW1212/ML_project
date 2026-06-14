"""Runtime configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal

from .constants import MODEL_VERSION, SOURCE_TAG


@dataclass(frozen=True)
class ProductionConfig:
    """Configuration for UI-ready production inference/reporting.

    Output policy:
    - JSON is the default because UI/UX should consume structured payloads.
    - CSV export is optional and disabled unless export_csv=True.
    """

    input_dir: Path
    out_dir: Path
    assets: List[str]
    source_tag: str = SOURCE_TAG
    model_version: str = MODEL_VERSION
    initial_capital: float = 100_000_000.0
    risk_free_rate: float = 0.0
    holdout_start: str = "2024-01-01"
    allocation_source: Literal["executed", "signal"] = "executed"
    capital_mode: Literal["equal", "custom", "inverse_vol"] = "equal"
    custom_capital_weights: Dict[str, float] = field(default_factory=dict)
    inverse_vol_column: str = "realized_vol_60"
    inverse_vol_floor: float = 1e-6
    export_json: bool = True
    export_csv: bool = False
    export_markdown: bool = False
    make_zip: bool = True

    # Local-only daily data cache settings. These do not trigger realtime streaming
    # and do not execute any broker/order workflow.
    cache_dir: Path = Path("storage/market_cache")
    update_provider: Literal["yahoo", "kis"] = "yahoo"
    daily_update_hour_kst: int = 8
    daily_freshness_tolerance_days: int = 2
    default_update_start: str = "2013-01-01"

    def validate(self) -> None:
        if self.allocation_source not in {"executed", "signal"}:
            raise ValueError("allocation_source must be 'executed' or 'signal'.")
        if self.capital_mode not in {"equal", "custom", "inverse_vol"}:
            raise ValueError("capital_mode must be 'equal', 'custom', or 'inverse_vol'.")
        if not self.assets:
            raise ValueError("At least one asset ticker is required.")
        if self.update_provider not in {"yahoo", "kis"}:
            raise ValueError("update_provider must be 'yahoo' or 'kis'.")
        if self.daily_update_hour_kst < 0 or self.daily_update_hour_kst > 23:
            raise ValueError("daily_update_hour_kst must be between 0 and 23.")
        if self.daily_freshness_tolerance_days < 0:
            raise ValueError("daily_freshness_tolerance_days must be non-negative.")
        if self.risk_free_rate < -1.0 or self.risk_free_rate > 1.0:
            raise ValueError("risk_free_rate should be expressed as a decimal annual rate between -1.0 and 1.0.")
        if self.capital_mode == "custom":
            missing = [ticker for ticker in self.assets if ticker not in self.custom_capital_weights]
            if missing:
                raise ValueError(f"Missing custom capital weights for: {missing}")
            total = sum(float(self.custom_capital_weights[ticker]) for ticker in self.assets)
            if total <= 0:
                raise ValueError("Sum of custom capital weights must be positive.")
