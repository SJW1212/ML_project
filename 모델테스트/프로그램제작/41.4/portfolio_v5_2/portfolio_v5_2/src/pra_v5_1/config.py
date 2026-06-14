from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import load_json


@dataclass(frozen=True)
class AppConfig:
    storage_root: Path = Path("storage")
    default_provider: str = "yahoo"
    default_bond_ticker: str = "IEF"
    default_cash_ticker: str = "BIL"
    daily_update_hour_kst: int = 8
    prediction_engine_mode: str = "reference_v8641_compatible"
    external_v8641_script: Path = Path("src/pra_v5_1/model_engine/xgb_recency_weighted_v8_6_41_model_label_fixed.py")
    transaction_cost_bps: float = 10.0
    min_cash_weight: float = 0.0
    max_asset_weight: float = 1.0
    risk_sensitivity: float = 1.0
    missing_asset_policy: str = "cash_fallback"
    cors_allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.1:8080"])

    @property
    def config_dir(self) -> Path:
        return self.storage_root / "config"

    @property
    def registry_path(self) -> Path:
        return self.config_dir / "tickers.json"

    @property
    def cache_dir(self) -> Path:
        return self.storage_root / "market_cache"

    @property
    def prediction_dir(self) -> Path:
        return self.storage_root / "predictions" / "v8_6_41"

    @property
    def run_dir(self) -> Path:
        return self.storage_root / "runs"

    @property
    def log_dir(self) -> Path:
        return self.storage_root / "logs"

    @classmethod
    def from_json(cls, path: Optional[Path] = None) -> "AppConfig":
        if path is None:
            path = Path("local_app_config.json")
        data: Dict[str, Any] = load_json(path, default={}) or {}
        adv = data.get("advanced_defaults", {}) or {}
        api = data.get("api", {}) or {}
        return cls(
            storage_root=Path(data.get("storage_root", "storage")),
            default_provider=data.get("default_provider", "yahoo"),
            default_bond_ticker=data.get("default_bond_ticker", "IEF"),
            default_cash_ticker=data.get("default_cash_ticker", "BIL"),
            daily_update_hour_kst=int(data.get("daily_update_hour_kst", 8)),
            prediction_engine_mode=data.get("prediction_engine_mode", "reference_v8641_compatible"),
            external_v8641_script=Path(data.get("external_v8641_script", "src/pra_v5_1/model_engine/xgb_recency_weighted_v8_6_41_model_label_fixed.py")),
            transaction_cost_bps=float(adv.get("transaction_cost_bps", data.get("transaction_cost_bps", 10.0))),
            min_cash_weight=float(adv.get("min_cash_weight", data.get("min_cash_weight", 0.0))),
            max_asset_weight=float(adv.get("max_asset_weight", data.get("max_asset_weight", 1.0))),
            risk_sensitivity=float(adv.get("risk_sensitivity", data.get("risk_sensitivity", 1.0))),
            missing_asset_policy=adv.get("missing_asset_policy", data.get("missing_asset_policy", "cash_fallback")),
            cors_allowed_origins=list(api.get("cors_allowed_origins", ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.1:8080"])),
        )
