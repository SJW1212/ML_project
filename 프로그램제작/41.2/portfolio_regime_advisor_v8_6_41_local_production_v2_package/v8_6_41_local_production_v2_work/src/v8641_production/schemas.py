"""Dataclasses used by the production pipeline and UI serializer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def to_plain_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass to JSON-serializable dict."""
    raw = asdict(obj)
    return _make_json_safe(raw)


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    # pandas/numpy scalars are handled through item() if present.
    try:
        import numpy as np
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str, bytes, bool)) else False:
        return None
    return value


@dataclass
class AssetData:
    ticker: str
    prediction_path: Path
    summary_path: Optional[Path]
    predictions: pd.DataFrame
    summary: Dict[str, Any]


@dataclass
class SignalSnapshot:
    ticker: str
    date: str
    model_version: str
    pred_risk: str
    pred_direction: str
    pred_overall_risk: str
    signal_regime: str
    allocation_regime: str
    executed_regime: str
    prob_normal: float
    prob_high_vol: float
    prob_overall_risk: float
    prob_up_strengthening_5d: float
    prob_up_strengthening_10d: float
    prob_up_strengthening_20d: float
    prob_up_strengthening_score: float
    prob_down_strengthening_5d: float
    prob_down_strengthening_10d: float
    prob_down_strengthening_20d: float
    prob_down_strengthening_score: float
    signal_stock_weight: float
    signal_bond_weight: float
    signal_cash_weight: float
    executed_stock_weight: float
    executed_bond_weight: float
    executed_cash_weight: float
    offensive_active: bool
    offensive_tier: float
    full_stock_signal: bool
    risk_class: str
    direction_class: str
    allocation_class: str
    monitoring_note: str


@dataclass
class AllocationRow:
    ticker: str
    date: str
    asset_capital_weight: float
    asset_stock_weight: float
    asset_bond_weight: float
    asset_cash_weight: float
    portfolio_stock_weight: float
    portfolio_bond_weight: float
    portfolio_cash_weight: float
    portfolio_total_weight: float
    allocation_source: str
    capital_mode: str
    risk_class: str
    direction_class: str
    allocation_class: str
    monitoring_note: str


@dataclass
class PortfolioTotals:
    date: str
    portfolio_stock_weight: float
    portfolio_bond_weight: float
    portfolio_cash_weight: float
    portfolio_total_weight: float
    allocation_source: str
    capital_mode: str


@dataclass
class PerformanceRow:
    ticker: str
    scope: str
    n_days: int
    final_capital: float
    cagr: float
    mdd: float
    sharpe: float
    sortino: float
    calmar: float
    annual_vol: float
    win_rate: float
    avg_stock_weight: float
    avg_bond_weight: float
    avg_cash_weight: float


@dataclass
class ValidationCheck:
    check_name: str
    status: str
    detail: str
