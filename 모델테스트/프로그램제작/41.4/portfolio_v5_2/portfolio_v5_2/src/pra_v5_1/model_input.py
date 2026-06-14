from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_schema import FeatureSchema


class ModelInputBuilder:
    """Builds leakage-safe features from cached OHLCV only."""

    def __init__(self, schema: FeatureSchema | None = None):
        self.schema = schema or FeatureSchema()

    def build(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        self.schema.validate_raw(ohlcv)
        df = ohlcv.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        close = pd.to_numeric(df["Close"], errors="coerce")
        high = pd.to_numeric(df["High"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")
        volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)

        feat = pd.DataFrame({"Date": df["Date"], "Close": close})
        for w in [1, 3, 5, 10, 20, 60, 120, 252]:
            feat[f"ret_{w}d"] = close.pct_change(w)
        ret1 = close.pct_change()
        for w in [5, 10, 20, 60, 120]:
            feat[f"vol_{w}d"] = ret1.rolling(w).std().shift(1) * np.sqrt(252.0)
        for w in [5, 10, 20, 60, 120, 200]:
            ma = close.rolling(w).mean().shift(1)
            feat[f"ma_{w}_ratio"] = close / ma - 1.0
        for w in [20, 60, 120, 252]:
            peak = close.rolling(w).max().shift(1)
            feat[f"drawdown_{w}d"] = close / peak - 1.0
        tr = (high - low) / close.replace(0, np.nan)
        for w in [5, 20, 60]:
            feat[f"range_{w}d"] = tr.rolling(w).mean().shift(1)
        vol_ma = volume.rolling(20).mean().shift(1)
        feat["volume_20d_ratio"] = volume / vol_ma.replace(0, np.nan)
        # no future/label columns here. actual next return is created in PredictionEngine after probabilities.
        feat = feat.replace([np.inf, -np.inf], np.nan)
        self.schema.validate_features(feat.drop(columns=["Close"]))
        return feat
