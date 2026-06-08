from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    horizons: List[int]
    vol_window: int = 60
    min_abs_threshold: float = 0.003
    max_abs_threshold: float = 0.040
    k_direction: float = 0.30


class FeaturePipeline:
    """Feature generation shared by training and live inference.

    This is the production-safe subset. It mirrors the v8.6.41 philosophy:
    use returns, ratios, moving-average gaps, volume ratios, volatility and drawdown,
    not raw prices as model features.
    """

    FEATURE_COLUMNS = [
        "return_5d", "return_10d", "return_20d", "return_60d", "return_120d",
        "return_5d_minus_20d", "return_10d_minus_20d",
        "price_ma_20_gap", "price_ma_60_gap", "price_ma_120_gap", "price_ma_200_gap",
        "ma_gap_5_20", "ma_gap_20_60", "ma_gap_60_120", "ma_gap_50_200",
        "trend_slope_5", "trend_slope_10", "trend_slope_20", "trend_slope_60", "ma200_slope_60",
        "drawdown_5", "drawdown_10", "drawdown_20", "drawdown_60", "drawdown_120",
        "price_position_10", "price_position_20", "price_position_60",
        "close_to_10d_high", "close_to_20d_high", "close_to_60d_high",
        "volume_ratio_20", "volume_zscore_20", "down_volume_ratio_10", "down_volume_ratio_20",
        "high_volume_down_ratio_10", "high_volume_down_ratio_20", "volume_shock_rank_252",
        "atr_pct_5", "atr_pct_10", "atr_pct_14", "atr_pct_20", "atr_pct_60",
        "realized_vol_10", "realized_vol_20", "realized_vol_60",
        "ewma_vol_20", "ewma_vol_60",
        "downside_vol_10", "downside_vol_20", "downside_vol_60",
        "ulcer_index_20", "ulcer_index_60", "bb_width_20", "vol_of_vol_20",
    ]

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig(horizons=[5, 10, 20])

    @staticmethod
    def _slope(series: pd.Series, window: int) -> pd.Series:
        idx = np.arange(window)
        denom = ((idx - idx.mean()) ** 2).sum()

        def calc(x: np.ndarray) -> float:
            y = np.asarray(x, dtype=float)
            if np.isnan(y).any():
                return np.nan
            return float(((idx - idx.mean()) * (y - y.mean())).sum() / denom)

        return series.rolling(window).apply(calc, raw=True)

    def build(self, ohlcv: pd.DataFrame, include_labels: bool = False) -> pd.DataFrame:
        df = ohlcv.copy()
        rename = {c: c.title() for c in df.columns if c.lower() in {"open", "high", "low", "close", "volume", "date"}}
        df = df.rename(columns=rename)
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"OHLCV missing required columns: {sorted(missing)}")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date", keep="last")
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Volume"] = df["Volume"].replace(0, np.nan)
        close = df["Close"]
        ret1 = close.pct_change()

        for w in [5, 10, 20, 60, 120]:
            df[f"return_{w}d"] = close.pct_change(w)
        df["return_5d_minus_20d"] = df["return_5d"] - df["return_20d"]
        df["return_10d_minus_20d"] = df["return_10d"] - df["return_20d"]

        for w in [5, 20, 50, 60, 120, 200]:
            ma = close.rolling(w).mean()
            df[f"ma_{w}"] = ma
            if w in [20, 60, 120, 200]:
                df[f"price_ma_{w}_gap"] = close / ma - 1.0
        df["ma_gap_5_20"] = df["ma_5"] / df["ma_20"] - 1.0
        df["ma_gap_20_60"] = df["ma_20"] / df["ma_60"] - 1.0
        df["ma_gap_60_120"] = df["ma_60"] / df["ma_120"] - 1.0
        df["ma_gap_50_200"] = df["ma_50"] / df["ma_200"] - 1.0

        log_close = np.log(close.replace(0, np.nan))
        for w in [5, 10, 20, 60]:
            df[f"trend_slope_{w}"] = self._slope(log_close, w)
        df["ma200_slope_60"] = self._slope(np.log(df["ma_200"].replace(0, np.nan)), 60)

        for w in [5, 10, 20, 60, 120]:
            roll_max = close.rolling(w).max()
            roll_min = close.rolling(w).min()
            df[f"drawdown_{w}"] = close / roll_max - 1.0
            if w in [10, 20, 60]:
                df[f"price_position_{w}"] = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)
                df[f"close_to_{w}d_high"] = close / roll_max - 1.0

        vol20 = df["Volume"].rolling(20).mean()
        vol20_std = df["Volume"].rolling(20).std()
        df["volume_ratio_20"] = df["Volume"] / vol20
        df["volume_zscore_20"] = (df["Volume"] - vol20) / vol20_std.replace(0, np.nan)
        down_day = ret1 < 0
        for w in [10, 20]:
            df[f"down_volume_ratio_{w}"] = (df["Volume"].where(down_day, 0).rolling(w).sum() / df["Volume"].rolling(w).sum())
            high_vol = df["Volume"] > df["Volume"].rolling(w).mean()
            df[f"high_volume_down_ratio_{w}"] = (high_vol & down_day).rolling(w).mean()
        df["volume_shock_rank_252"] = df["volume_ratio_20"].rolling(252).rank(pct=True)

        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        for w in [5, 10, 14, 20, 60]:
            df[f"atr_pct_{w}"] = tr.rolling(w).mean() / close
        for w in [10, 20, 60]:
            df[f"realized_vol_{w}"] = ret1.rolling(w).std()
            df[f"downside_vol_{w}"] = ret1.where(ret1 < 0, 0).rolling(w).std()
        df["ewma_vol_20"] = ret1.ewm(span=20, adjust=False).std()
        df["ewma_vol_60"] = ret1.ewm(span=60, adjust=False).std()
        for w in [20, 60]:
            dd = df[f"drawdown_{w}"]
            df[f"ulcer_index_{w}"] = np.sqrt((dd.clip(upper=0) ** 2).rolling(w).mean())
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        df["bb_width_20"] = (4 * std) / mid
        df["vol_of_vol_20"] = df["realized_vol_20"].rolling(20).std()

        # volatility-scaled future labels for improved/candidate training
        if include_labels:
            current_vol = ret1.rolling(self.config.vol_window).std().shift(1)
            for h in self.config.horizons:
                future_ret = close.shift(-h) / close - 1.0
                threshold = (self.config.k_direction * current_vol * np.sqrt(h)).clip(
                    lower=self.config.min_abs_threshold,
                    upper=self.config.max_abs_threshold,
                )
                df[f"future_return_{h}d"] = future_ret
                df[f"threshold_{h}d"] = threshold
                df[f"y_up_strength_{h}d"] = (future_ret > threshold).astype(float)
                df[f"y_down_strength_{h}d"] = (future_ret < -threshold).astype(float)
                future_abs = future_ret.abs()
                high_vol_threshold = future_abs.rolling(252, min_periods=60).quantile(0.75).shift(1)
                df[f"y_high_vol_{h}d"] = (future_abs > high_vol_threshold).astype(float)

        keep = ["Date", "Ticker"] if "Ticker" in df.columns else ["Date"]
        label_cols = [c for c in df.columns if c.startswith("future_") or c.startswith("threshold_") or c.startswith("y_")]
        return df[keep + self.FEATURE_COLUMNS + label_cols].replace([np.inf, -np.inf], np.nan)

    def latest_feature_row(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        features = self.build(ohlcv, include_labels=False)
        row = features.dropna(subset=self.FEATURE_COLUMNS).tail(1)
        if row.empty:
            raise ValueError("Not enough OHLCV history to build a valid latest feature row.")
        return row[["Date"] + self.FEATURE_COLUMNS]
