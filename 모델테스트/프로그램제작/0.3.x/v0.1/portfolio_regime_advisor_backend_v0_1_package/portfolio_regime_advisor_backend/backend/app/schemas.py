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
    ticker: str = "005930"
    market: str = "KR"


class MarketDataUpdateRequest(BaseModel):
    provider: str = "kis"
    environment: str = "mock"
    tickers: List[str]
    market: str = "KR"
    start_date: str
    end_date: str


class TrainingRequest(BaseModel):
    tickers: List[str]
    horizons: List[str] = Field(default_factory=lambda: ["5D", "10D", "20D"])
    train_start: str
    train_end: str
    preset: str = "balanced"
    data_source: str = "cache"


class TrainingJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
