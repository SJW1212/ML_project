from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .core.user_modes import UserMode


class ValidationMessage(BaseModel):
    level: str
    code: str
    message: str
    field: Optional[str] = None


class SettingsRequest(BaseModel):
    user_mode: UserMode = UserMode.GENERAL
    preset: Optional[str] = None
    horizon: str = "10D"
    assets: List[str] = Field(default_factory=lambda: ["QQQ", "SPY", "AAPL", "SOXX", "NVDA"])
    capital_mode: str = "equal"
    custom_weights: Optional[Dict[str, float]] = None
    oos_start: Optional[str] = None
    benchmark: str = "60_40"

    @validator("horizon")
    def normalize_horizon(cls, value: str) -> str:
        return value.upper().replace(" ", "")

    @validator("assets", each_item=True)
    def normalize_assets(cls, value: str) -> str:
        return value.upper().strip()


class SignalItem(BaseModel):
    ticker: str
    date: str
    risk_class: str
    direction_class: str
    allocation_class: str
    prob_normal: float
    prob_high_vol: float
    prob_overall_risk: float
    prob_up_strengthening_score: float
    prob_down_strengthening_score: float
    selected_horizon: str
    selected_prob_high_vol: Optional[float] = None
    selected_prob_up_strengthening: Optional[float] = None
    selected_prob_down_strengthening: Optional[float] = None
    stock_weight: float
    bond_weight: float
    cash_weight: float
    recommended_stock_weight: Optional[float] = None
    executed_stock_weight: Optional[float] = None
    comment: str
    warnings: List[str] = Field(default_factory=list)


class PortfolioTotals(BaseModel):
    stock: float
    bond: float
    cash: float


class PerformanceItem(BaseModel):
    ticker: str
    cagr: Optional[float] = None
    mdd: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    calmar: Optional[float] = None
    source: str = "summary"


class ChartPayload(BaseModel):
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    drawdown: List[Dict[str, Any]] = Field(default_factory=list)
    annual_returns: List[Dict[str, Any]] = Field(default_factory=list)
    monthly_returns: List[Dict[str, Any]] = Field(default_factory=list)


class DashboardPayload(BaseModel):
    as_of_date: Optional[str]
    model_version: str
    model_mode: str
    user_mode: str
    preset: Optional[str]
    horizon: str
    data_source: Dict[str, Any]
    portfolio_totals: PortfolioTotals
    latest_signals: List[SignalItem]
    performance_summary: List[PerformanceItem]
    charts: ChartPayload
    validation: Dict[str, Any]
    insights: List[str]


class CredentialPayload(BaseModel):
    environment: str = Field(default="mock", description="mock or real")
    app_key: str
    app_secret: str
    account_no: Optional[str] = None
    account_product_code: Optional[str] = None


class ProviderTestRequest(BaseModel):
    environment: str = "mock"
    ticker: str = "QQQ"
    market: str = "US"


class MarketDataUpdateRequest(BaseModel):
    provider: str = "auto"
    environment: str = "mock"
    tickers: List[str]
    market: str = "US"
    start_date: str
    end_date: str

    @validator("provider")
    def normalize_provider(cls, value: str) -> str:
        return value.lower().strip()

    @validator("tickers", each_item=True)
    def normalize_tickers(cls, value: str) -> str:
        return value.upper().strip()


class InferenceRequest(BaseModel):
    tickers: List[str]
    horizon: str = "10D"
    provider: str = "auto"
    market: str = "US"
    model_version: Optional[str] = None
    allow_prediction_file_fallback: bool = True

    @validator("horizon")
    def normalize_horizon(cls, value: str) -> str:
        return value.upper().replace(" ", "")

    @validator("provider")
    def normalize_provider(cls, value: str) -> str:
        return value.lower().strip()

    @validator("tickers", each_item=True)
    def normalize_infer_tickers(cls, value: str) -> str:
        return value.upper().strip()


class TrainingRequest(BaseModel):
    tickers: List[str]
    horizons: List[str] = Field(default_factory=lambda: ["5D", "10D", "20D"])
    train_start: str
    train_end: str
    preset: str = "balanced"
    data_source: str = "cache"  # cache, auto, yahoo, kis
    market: str = "US"
    model_version: Optional[str] = None

    # v0.3.3 runtime training controls. Defaults intentionally match the
    # v8.6.41 recency-weighted philosophy while keeping validation leak-safe.
    sample_weight_mode: str = "recency"  # recency, equal
    walk_forward_mode: str = "expanding"  # expanding, rolling
    rolling_train_rows: Optional[int] = None
    recency_half_life_by_horizon: Optional[Dict[str, int]] = None

    @validator("data_source")
    def normalize_data_source(cls, value: str) -> str:
        return value.lower().strip()

    @validator("sample_weight_mode")
    def normalize_sample_weight_mode(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"recency", "equal"}:
            raise ValueError("sample_weight_mode must be 'recency' or 'equal'")
        return value

    @validator("walk_forward_mode")
    def normalize_walk_forward_mode(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"expanding", "rolling"}:
            raise ValueError("walk_forward_mode must be 'expanding' or 'rolling'")
        return value

    @validator("horizons", each_item=True)
    def normalize_train_horizons(cls, value: str) -> str:
        return value.upper().replace(" ", "")

    @validator("tickers", each_item=True)
    def normalize_train_tickers(cls, value: str) -> str:
        return value.upper().strip()


class TrainingJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
