from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .context_asset_universe import ContextAssetUniverse


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def _rolling_pct_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).rank(pct=True)


def _drawdown(series: pd.Series, window: int) -> pd.Series:
    high = series.rolling(window).max()
    return series / high - 1.0


@dataclass
class MarketContextFeatureBuilder:
    """Build leak-safe market-context features aligned to target feature rows.

    All external/context features are shifted by one row so a prediction for date T
    only uses context information available at T-1 or earlier. Target-derived
    rolling features are also shifted when used as context overlays.
    """

    include_volatility: bool = True
    include_market_structure: bool = True
    include_breadth: bool = True
    include_cross_asset: bool = True
    prefix: str = "ctx_"
    warnings: List[str] = field(default_factory=list)

    FEATURE_GROUPS: Dict[str, List[str]] = field(default_factory=lambda: {
        "volatility": [
            "ctx_vix_close", "ctx_vix_ret_5d", "ctx_vix_z_63", "ctx_vix_pct_252", "ctx_vix_spike_5d",
            "ctx_realized_vol_5d", "ctx_realized_vol_20d", "ctx_realized_vol_60d",
            "ctx_vol_ratio_5_20", "ctx_vol_ratio_20_60", "ctx_vol_of_vol_20d", "ctx_market_vol_spread_spy",
        ],
        "market_structure": [
            "ctx_spy_drawdown_63", "ctx_spy_drawdown_252", "ctx_qqq_drawdown_63", "ctx_qqq_drawdown_252",
            "ctx_target_drawdown_63", "ctx_target_drawdown_252", "ctx_recovery_from_low_63",
            "ctx_lower_high_count_20d", "ctx_lower_low_count_20d",
        ],
        "breadth": [
            "ctx_rsp_spy_rel_20d", "ctx_iwm_spy_rel_20d", "ctx_qqq_spy_rel_20d",
            "ctx_sector_positive_ratio_20d", "ctx_sector_above_sma50_ratio", "ctx_sector_dispersion_20d",
        ],
        "cross_asset": [
            "ctx_hyg_lqd_rel_20d", "ctx_hyg_lqd_rel_60d", "ctx_hyg_lqd_z_63",
            "ctx_tlt_shy_rel_20d", "ctx_ief_shy_rel_20d",
            "ctx_tnx_change_5d", "ctx_tnx_change_20d", "ctx_tnx_z_252",
            "ctx_uup_ret_20d", "ctx_gld_spy_rel_20d",
        ],
    })

    @property
    def feature_columns(self) -> List[str]:
        cols: List[str] = []
        for group in ["volatility", "market_structure", "breadth", "cross_asset"]:
            cols.extend(self.FEATURE_GROUPS[group])
        return cols

    def _close(self, universe: ContextAssetUniverse, ticker: str) -> Optional[pd.Series]:
        s = universe.close(ticker)
        if s is None:
            self.warnings.append(f"CONTEXT_TICKER_MISSING:{ticker.upper()}")
        return s

    def _assign(self, df: pd.DataFrame, name: str, values: pd.Series) -> None:
        df[name] = values.reindex(df.index).shift(1)

    def build(self, features: pd.DataFrame, universe: ContextAssetUniverse) -> pd.DataFrame:
        df = features.copy()
        if "Date" not in df.columns:
            raise ValueError("MarketContextFeatureBuilder requires a Date column in feature DataFrame")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date", drop=False)

        close_col = "Close" if "Close" in df.columns else None
        # FeaturePipeline output usually omits Close. If Close is absent, target-only context
        # features that require raw close are skipped instead of filled with silent zeros.
        if close_col is not None:
            target_close = pd.to_numeric(df[close_col], errors="coerce")
        else:
            target_close = None

        if self.include_volatility:
            self._build_volatility(df, universe, target_close)
        if self.include_market_structure:
            self._build_market_structure(df, universe, target_close)
        if self.include_breadth:
            self._build_breadth(df, universe)
        if self.include_cross_asset:
            self._build_cross_asset(df, universe)

        # Ensure every declared context feature exists. Missing optional features remain NaN,
        # not 0.0; downstream imputation can handle model inputs, while allocation can warn.
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        df["ctx_missing_count"] = df[self.feature_columns].isna().sum(axis=1)
        return df.reset_index(drop=True)

    def _build_volatility(self, df: pd.DataFrame, universe: ContextAssetUniverse, target_close: Optional[pd.Series]) -> None:
        vix = self._close(universe, "^VIX")
        spy = self._close(universe, "SPY")
        if vix is not None:
            self._assign(df, "ctx_vix_close", vix)
            self._assign(df, "ctx_vix_ret_5d", vix.pct_change(5))
            self._assign(df, "ctx_vix_z_63", _zscore(vix, 63))
            self._assign(df, "ctx_vix_pct_252", _rolling_pct_rank(vix, 252))
            vix_ret5 = vix.pct_change(5)
            spike = (vix_ret5 > (vix_ret5.rolling(252).mean() + 2.0 * vix_ret5.rolling(252).std())).astype(float)
            self._assign(df, "ctx_vix_spike_5d", spike)
        if target_close is not None:
            ret = target_close.pct_change()
            rv5 = ret.rolling(5).std() * np.sqrt(252)
            rv20 = ret.rolling(20).std() * np.sqrt(252)
            rv60 = ret.rolling(60).std() * np.sqrt(252)
            df["ctx_realized_vol_5d"] = rv5.shift(1)
            df["ctx_realized_vol_20d"] = rv20.shift(1)
            df["ctx_realized_vol_60d"] = rv60.shift(1)
            df["ctx_vol_ratio_5_20"] = (rv5 / rv20.replace(0, np.nan)).shift(1)
            df["ctx_vol_ratio_20_60"] = (rv20 / rv60.replace(0, np.nan)).shift(1)
            df["ctx_vol_of_vol_20d"] = (ret.rolling(5).std().rolling(20).std() * np.sqrt(252)).shift(1)
            if spy is not None:
                spy_vol20 = spy.pct_change().rolling(20).std() * np.sqrt(252)
                df["ctx_market_vol_spread_spy"] = (rv20 - spy_vol20.reindex(df.index)).shift(1)

    def _build_market_structure(self, df: pd.DataFrame, universe: ContextAssetUniverse, target_close: Optional[pd.Series]) -> None:
        for ticker, name in [("SPY", "spy"), ("QQQ", "qqq")]:
            s = self._close(universe, ticker)
            if s is not None:
                self._assign(df, f"ctx_{name}_drawdown_63", _drawdown(s, 63))
                self._assign(df, f"ctx_{name}_drawdown_252", _drawdown(s, 252))
        if target_close is not None:
            roll63 = target_close.rolling(63).max()
            roll252 = target_close.rolling(252).max()
            low63 = target_close.rolling(63).min()
            df["ctx_target_drawdown_63"] = (target_close / roll63 - 1.0).shift(1)
            df["ctx_target_drawdown_252"] = (target_close / roll252 - 1.0).shift(1)
            df["ctx_recovery_from_low_63"] = ((target_close - low63) / (roll63 - low63 + 1e-8)).shift(1)
            highs = target_close.rolling(5).max()
            lows = target_close.rolling(5).min()
            df["ctx_lower_high_count_20d"] = (highs.diff() < 0).rolling(20).sum().shift(1)
            df["ctx_lower_low_count_20d"] = (lows.diff() < 0).rolling(20).sum().shift(1)

    def _build_breadth(self, df: pd.DataFrame, universe: ContextAssetUniverse) -> None:
        spy = self._close(universe, "SPY")
        qqq = self._close(universe, "QQQ")
        rsp = self._close(universe, "RSP")
        iwm = self._close(universe, "IWM")
        if spy is not None and rsp is not None:
            self._assign(df, "ctx_rsp_spy_rel_20d", rsp.pct_change(20) - spy.pct_change(20))
        if spy is not None and iwm is not None:
            self._assign(df, "ctx_iwm_spy_rel_20d", iwm.pct_change(20) - spy.pct_change(20))
        if spy is not None and qqq is not None:
            self._assign(df, "ctx_qqq_spy_rel_20d", qqq.pct_change(20) - spy.pct_change(20))
        sectors = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLU", "XLB", "XLRE"]
        sector_closes = []
        for ticker in sectors:
            s = self._close(universe, ticker)
            if s is not None:
                sector_closes.append(s.rename(ticker))
        if sector_closes:
            closes = pd.concat(sector_closes, axis=1).reindex(df.index)
            rets20 = closes.pct_change(20)
            df["ctx_sector_positive_ratio_20d"] = (rets20 > 0).mean(axis=1).shift(1)
            df["ctx_sector_above_sma50_ratio"] = (closes > closes.rolling(50).mean()).mean(axis=1).shift(1)
            df["ctx_sector_dispersion_20d"] = rets20.std(axis=1).shift(1)

    def _build_cross_asset(self, df: pd.DataFrame, universe: ContextAssetUniverse) -> None:
        hyg = self._close(universe, "HYG")
        lqd = self._close(universe, "LQD")
        tlt = self._close(universe, "TLT")
        ief = self._close(universe, "IEF")
        shy = self._close(universe, "SHY")
        tnx = self._close(universe, "^TNX")
        uup = self._close(universe, "UUP")
        gld = self._close(universe, "GLD")
        spy = self._close(universe, "SPY")
        if hyg is not None and lqd is not None:
            rel20 = hyg.pct_change(20) - lqd.pct_change(20)
            rel60 = hyg.pct_change(60) - lqd.pct_change(60)
            self._assign(df, "ctx_hyg_lqd_rel_20d", rel20)
            self._assign(df, "ctx_hyg_lqd_rel_60d", rel60)
            self._assign(df, "ctx_hyg_lqd_z_63", _zscore(rel20, 63))
        if tlt is not None and shy is not None:
            self._assign(df, "ctx_tlt_shy_rel_20d", tlt.pct_change(20) - shy.pct_change(20))
        if ief is not None and shy is not None:
            self._assign(df, "ctx_ief_shy_rel_20d", ief.pct_change(20) - shy.pct_change(20))
        if tnx is not None:
            # ^TNX is already normalized to decimal yield in ContextAssetUniverse.
            self._assign(df, "ctx_tnx_change_5d", tnx.diff(5))
            self._assign(df, "ctx_tnx_change_20d", tnx.diff(20))
            self._assign(df, "ctx_tnx_z_252", _zscore(tnx, 252))
        if uup is not None:
            self._assign(df, "ctx_uup_ret_20d", uup.pct_change(20))
        if gld is not None and spy is not None:
            self._assign(df, "ctx_gld_spy_rel_20d", gld.pct_change(20) - spy.pct_change(20))
