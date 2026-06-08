from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


class MissingFeatureError(ValueError):
    pass


@dataclass(frozen=True)
class AllocationPolicyConfig:
    """Shared signal-to-weight policy used by prediction_file and live inference paths.

    The locked v8.6.41 prediction files already contain validated stock/bond/cash
    weights. Therefore, for prediction_file rows this engine preserves existing
    native weights by default. For live inference rows without native weights, it
    applies the same deterministic probability-to-weight fallback in one shared
    place instead of duplicating policy inside InferenceService.
    """

    defensive_bond_ratio: float = 0.67
    normal_stock: float = 0.82
    watch_stock: float = 0.74
    high_vol_stock: float = 0.62
    risk_off_stock: float = 0.45
    extreme_risk_stock: float = 0.30
    up_participation_stock: float = 0.86

    watch_high_vol_threshold: float = 0.35
    high_vol_threshold: float = 0.55
    risk_off_high_vol_threshold: float = 0.70
    extreme_high_vol_threshold: float = 0.82

    watch_down_threshold: float = 0.50
    high_down_threshold: float = 0.55
    risk_off_down_threshold: float = 0.65
    extreme_down_threshold: float = 0.78

    up_threshold: float = 0.55
    up_margin: float = 0.10


class AllocationPolicyEngine:
    """Single source of truth for per-ticker stock/bond/cash weights.

    Scope:
    - Keeps v8.6.41 prediction-file weights intact when present.
    - Computes runtime/candidate live inference weights when only probabilities are present.
    - Normalizes every row so stock + bond + cash = 1 unless all values are invalid.
    """

    def __init__(self, config: Optional[AllocationPolicyConfig] = None):
        self.config = config or AllocationPolicyConfig()

    @staticmethod
    def _float(value: Any, default: float = np.nan) -> float:
        try:
            if value is None or pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _row_float(row: pd.Series | Dict[str, Any], key: str, default: float | None = None, required: bool = False) -> float | None:
        if key not in row or row.get(key) is None:
            if required:
                raise MissingFeatureError(f"Required allocation feature missing: {key}")
            return default
        try:
            value = float(row.get(key))
            if not np.isfinite(value):
                if required:
                    raise MissingFeatureError(f"Required allocation feature invalid: {key}={row.get(key)}")
                return default
            return value
        except Exception:
            if required:
                raise MissingFeatureError(f"Required allocation feature invalid: {key}={row.get(key)}")
            return default

    @staticmethod
    def merge_prediction_features(prediction_df: pd.DataFrame, feature_df: pd.DataFrame | None, on: Iterable[str] = ("Date", "ticker")) -> pd.DataFrame:
        """Left-join raw feature columns into prediction rows before allocation.

        This prevents the previous silent-dead-feature failure mode where
        prediction rows did not contain context features and threshold logic read
        missing columns as 0.0. Existing prediction columns take precedence when
        duplicate names exist.
        """
        out = prediction_df.copy()
        if feature_df is None or feature_df.empty or out.empty:
            return out
        join_cols = [c for c in on if c in out.columns and c in feature_df.columns]
        if not join_cols:
            return out
        feat = feature_df.copy()
        extra_cols = [c for c in feat.columns if c in join_cols or c not in out.columns]
        return out.merge(feat[extra_cols], how="left", on=join_cols)

    @staticmethod
    def _clip01(value: float, default: float = 0.0) -> float:
        try:
            if not np.isfinite(value):
                return default
            return float(np.clip(value, 0.0, 1.0))
        except Exception:
            return default

    def _split_defensive_assets(self, stock: float) -> Dict[str, float]:
        stock = self._clip01(stock)
        defensive = max(0.0, 1.0 - stock)
        bond = defensive * self.config.defensive_bond_ratio
        cash = defensive - bond
        return self._normalize({"stock_weight": stock, "bond_weight": bond, "cash_weight": cash})

    def _normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        stock = self._clip01(weights.get("stock_weight", np.nan), default=np.nan)
        bond = self._clip01(weights.get("bond_weight", np.nan), default=np.nan)
        cash = self._clip01(weights.get("cash_weight", np.nan), default=np.nan)
        vals = np.array([stock, bond, cash], dtype=float)
        if not np.isfinite(vals).all() or vals.sum() <= 0:
            return {"stock_weight": 0.0, "bond_weight": 0.0, "cash_weight": 0.0}
        vals = vals / vals.sum()
        return {
            "stock_weight": float(vals[0]),
            "bond_weight": float(vals[1]),
            "cash_weight": float(vals[2]),
        }

    def native_weights_available(self, row: pd.Series | Dict[str, Any]) -> bool:
        return all(k in row and np.isfinite(self._float(row.get(k))) for k in ["stock_weight", "bond_weight", "cash_weight"])

    def from_existing(self, row: pd.Series | Dict[str, Any]) -> Dict[str, float]:
        return self._normalize({
            "stock_weight": self._float(row.get("stock_weight")),
            "bond_weight": self._float(row.get("bond_weight")),
            "cash_weight": self._float(row.get("cash_weight")),
        })

    def from_probabilities(self, row: pd.Series | Dict[str, Any]) -> Dict[str, float]:
        ph = self._clip01(self._float(row.get("prob_high_vol"), 0.50), default=0.50)
        pu = self._clip01(self._float(row.get("prob_up_strengthening_score"), 0.0), default=0.0)
        pdn = self._clip01(self._float(row.get("prob_down_strengthening_score"), 0.0), default=0.0)

        c = self.config
        allocation_warnings = []
        spy_dd = self._row_float(row, "ctx_spy_drawdown_252", default=None)
        vix_z = self._row_float(row, "ctx_vix_z_63", default=None)
        risk_override = row.get("risk_override") if isinstance(row, dict) else row.get("risk_override")

        if ph >= c.extreme_high_vol_threshold or pdn >= c.extreme_down_threshold:
            stock = c.extreme_risk_stock
            reason = "EXTREME_RISK_PROBABILITY_POLICY"
        elif ph >= c.risk_off_high_vol_threshold or pdn >= c.risk_off_down_threshold:
            stock = c.risk_off_stock
            reason = "RISK_OFF_PROBABILITY_POLICY"
        elif ph >= c.high_vol_threshold or pdn >= c.high_down_threshold:
            stock = c.high_vol_stock
            reason = "HIGH_VOL_PROBABILITY_POLICY"
        elif ph >= c.watch_high_vol_threshold or pdn >= c.watch_down_threshold:
            stock = c.watch_stock
            reason = "WATCH_PROBABILITY_POLICY"
        elif risk_override == "WATCH":
            stock = min(c.watch_stock, c.normal_stock)
            reason = "RULE_BASED_WATCH_OVERRIDE_POLICY"
        elif spy_dd is not None and spy_dd < -0.15 and ph >= 0.30:
            stock = min(c.watch_stock, c.normal_stock)
            reason = "CTX_SPY_DRAWDOWN_WATCH_POLICY"
            allocation_warnings.append("CTX_SPY_DRAWDOWN_252_WATCH")
        elif vix_z is not None and vix_z > 2.0 and ph >= 0.30:
            stock = min(c.watch_stock, c.normal_stock)
            reason = "CTX_VIX_SPIKE_WATCH_POLICY"
            allocation_warnings.append("CTX_VIX_Z_63_WATCH")
        elif pu >= c.up_threshold and (pu - pdn) >= c.up_margin:
            stock = c.up_participation_stock
            reason = "UP_STRENGTH_PARTICIPATION_POLICY"
        else:
            stock = c.normal_stock
            reason = "NORMAL_PARTICIPATION_POLICY"
        out = self._split_defensive_assets(stock)
        out["allocation_policy_reason"] = reason
        if allocation_warnings:
            out["allocation_warning"] = ";".join(allocation_warnings)
        return out

    def apply_row(self, row: pd.Series | Dict[str, Any], preserve_existing: bool = True) -> Dict[str, float]:
        if preserve_existing and self.native_weights_available(row):
            out = self.from_existing(row)
            out["allocation_policy_reason"] = "PRESERVED_NATIVE_WEIGHTS"
            return out
        return self.from_probabilities(row)

    def apply_dataframe(self, df: pd.DataFrame, preserve_existing: bool = True, feature_df: pd.DataFrame | None = None) -> pd.DataFrame:
        out = self.merge_prediction_features(df, feature_df) if feature_df is not None else df.copy()
        if out.empty:
            return out
        rows = []
        for _, row in out.iterrows():
            rows.append(self.apply_row(row, preserve_existing=preserve_existing))
        policy_df = pd.DataFrame(rows, index=out.index)
        for col in ["stock_weight", "bond_weight", "cash_weight", "allocation_policy_reason", "allocation_warning"]:
            if col in policy_df.columns:
                out[col] = policy_df[col]
        out["recommended_stock_weight"] = out["stock_weight"]
        out["executed_stock_weight"] = out["stock_weight"]
        return out
