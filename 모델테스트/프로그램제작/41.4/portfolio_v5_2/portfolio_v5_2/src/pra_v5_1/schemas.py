from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .utils import normalize_ticker


UserLevel = Literal["general", "advanced", "expert", "developer"]
RiskProfile = Literal["conservative", "balanced", "aggressive"]
AssetType = Literal["stock", "etf", "risk_asset", "bond", "bond_etf", "bond_bucket", "cash", "cash_bucket"]
CapitalMode = Literal["current_weight", "equal", "custom", "inverse_vol"]
MissingAssetPolicy = Literal["cash_fallback", "active_weight_renormalize", "common_range_only"]
PredictionEngineMode = Literal["reference_v8641_compatible", "external_v8641_xgb"]


class PortfolioAssetInput(BaseModel):
    name: Optional[str] = None
    ticker: str
    asset_type: AssetType = "stock"
    current_weight: Optional[float] = Field(default=None, ge=0.0)
    current_value: Optional[float] = Field(default=None, ge=0.0)
    enabled: bool = True

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, v: str) -> str:
        return normalize_ticker(v)

    @property
    def is_cash_bucket(self) -> bool:
        return self.asset_type in {"cash", "cash_bucket"} or self.ticker in {"CASH", "CASH_BUCKET"}

    @property
    def is_bond_bucket(self) -> bool:
        return self.asset_type == "bond_bucket" or self.ticker == "BOND_BUCKET"

    @property
    def is_risk_asset(self) -> bool:
        return self.enabled and not self.is_cash_bucket and not self.is_bond_bucket and self.asset_type in {"stock", "etf", "risk_asset"}

    @property
    def is_defensive_etf(self) -> bool:
        return self.enabled and self.asset_type in {"bond", "bond_etf"} and not self.is_bond_bucket


class EvaluateSettings(BaseModel):
    start_date: str = "2013-01-01"
    end_date: Optional[str] = None
    holdout_start: str = "2024-01-01"
    update_data: bool = True
    generate_predictions: bool = True
    force_update: bool = False
    force_prediction: bool = False
    provider: Literal["yahoo", "kis"] = "yahoo"
    prediction_engine_mode: PredictionEngineMode = "reference_v8641_compatible"
    capital_mode: CapitalMode = "current_weight"
    custom_weights: Dict[str, float] = Field(default_factory=dict)
    missing_asset_policy: MissingAssetPolicy = "cash_fallback"
    transaction_cost_bps: float = Field(default=10.0, ge=0.0, le=200.0)
    min_cash_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    max_asset_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    risk_sensitivity: float = Field(default=1.0, ge=0.1, le=3.0)
    speed_profile: Literal["fast", "balanced", "full"] = "balanced"

    @field_validator("custom_weights")
    @classmethod
    def _normalize_custom_keys(cls, v: Dict[str, float]) -> Dict[str, float]:
        return {normalize_ticker(k): float(val) for k, val in (v or {}).items()}


class PortfolioEvaluateRequest(BaseModel):
    portfolio: List[PortfolioAssetInput]
    risk_profile: RiskProfile = "balanced"
    user_level: UserLevel = "general"
    settings: EvaluateSettings = Field(default_factory=EvaluateSettings)

    @model_validator(mode="after")
    def _has_assets(self):
        if not self.portfolio:
            raise ValueError("portfolio must not be empty")
        return self


class TickerAddRequest(BaseModel):
    tickers: List[str]
    asset_type: AssetType = "stock"
    market: str = "US"
    note: Optional[str] = None

    @field_validator("tickers")
    @classmethod
    def _tickers(cls, v: List[str]) -> List[str]:
        return [normalize_ticker(x) for x in v]


class DailyUpdateRequest(BaseModel):
    tickers: List[str] = Field(default_factory=list)
    provider: Literal["yahoo", "kis"] = "yahoo"
    start: str = "2013-01-01"
    end: Optional[str] = None
    force: bool = False

    @field_validator("tickers")
    @classmethod
    def _tickers(cls, v: List[str]) -> List[str]:
        return [normalize_ticker(x) for x in v]


class PredictionGenerateRequest(BaseModel):
    tickers: List[str] = Field(default_factory=list)
    start: str = "2013-01-01"
    end: Optional[str] = None
    mode: PredictionEngineMode = "reference_v8641_compatible"
    force: bool = False

    @field_validator("tickers")
    @classmethod
    def _tickers(cls, v: List[str]) -> List[str]:
        return [normalize_ticker(x) for x in v]
