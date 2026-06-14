from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .cache import MarketDataCache
from .feature_schema import validate_prediction_output
from .model_input import ModelInputBuilder
from .utils import atomic_write_csv, atomic_write_json, config_hash, ensure_dir, normalize_ticker, safe_float, utc_now_iso

SOURCE_TAG = "xgb_recency_weighted_v8_6_41_model_label_fixed"


def sigmoid(x: pd.Series | float) -> pd.Series | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


@dataclass
class PredictionResult:
    ticker: str
    path: Path
    mode: str
    latest_date: Optional[str]
    rows: int
    errors: List[str]


class LocalRunTransaction:
    def __init__(self, root: Path, run_type: str):
        self.root = root
        self.run_type = run_type
        self.run_id = f"{run_type}_{utc_now_iso().replace(':', '').replace('.', '')}_{uuid.uuid4().hex[:8]}"
        self.staging_dir = ensure_dir(root / "staging" / self.run_id)
        self.active_dir = ensure_dir(root / "active")
        self.manifest: Dict = {"run_id": self.run_id, "run_type": run_type, "started_at": utc_now_iso(), "items": []}

    def add_item(self, **item) -> None:
        self.manifest["items"].append(item)

    def commit(self) -> Path:
        self.manifest["committed_at"] = utc_now_iso()
        manifest_path = self.staging_dir / "manifest.json"
        atomic_write_json(manifest_path, self.manifest)
        active_manifest = self.active_dir / f"{self.run_id}_manifest.json"
        atomic_write_json(active_manifest, self.manifest)
        return active_manifest


class ReferenceV8641CompatibleEngine:
    """Cache-based reference engine that emits the v8.6.41 probability schema.

    This is deterministic and leakage-safe. It is not a trained XGBoost artifact.
    Use ExternalV8641XgbEngine for the original script when xgboost is available.
    """

    def __init__(self, builder: ModelInputBuilder | None = None):
        self.builder = builder or ModelInputBuilder()

    def predict(self, ticker: str, ohlcv: pd.DataFrame, risk_sensitivity: float = 1.0) -> pd.DataFrame:
        t = normalize_ticker(ticker)
        raw = ohlcv.copy().sort_values("Date").reset_index(drop=True)
        features = self.builder.build(raw)
        close = pd.to_numeric(raw["Close"], errors="coerce").reset_index(drop=True)
        feat = features.copy().reset_index(drop=True)

        vol20 = feat["vol_20d"].fillna(feat["vol_60d"]).fillna(0.20)
        vol_rank = vol20.rolling(504, min_periods=80).rank(pct=True).fillna(0.5)
        mom5 = feat["ret_5d"].fillna(0.0)
        mom10 = feat["ret_10d"].fillna(0.0)
        mom20 = feat["ret_20d"].fillna(0.0)
        dd60 = feat["drawdown_60d"].fillna(0.0)
        dd252 = feat["drawdown_252d"].fillna(0.0)
        range20 = feat["range_20d"].fillna(feat["range_60d"]).fillna(0.02)
        range_rank = range20.rolling(504, min_periods=80).rank(pct=True).fillna(0.5)
        vol_boost = ((vol_rank + range_rank) / 2.0 - 0.5) * 3.0 * risk_sensitivity
        prob_high_vol = pd.Series(sigmoid(vol_boost), index=feat.index).clip(0.01, 0.99)

        scale = vol20.replace(0, np.nan).fillna(0.20) / math.sqrt(252.0)
        up5 = pd.Series(sigmoid((mom5 / (scale * math.sqrt(5) + 1e-9)) * 0.45), index=feat.index).clip(0.01, 0.99)
        up10 = pd.Series(sigmoid((mom10 / (scale * math.sqrt(10) + 1e-9)) * 0.38), index=feat.index).clip(0.01, 0.99)
        up20 = pd.Series(sigmoid((mom20 / (scale * math.sqrt(20) + 1e-9)) * 0.30), index=feat.index).clip(0.01, 0.99)
        down5 = pd.Series(sigmoid((-mom5 / (scale * math.sqrt(5) + 1e-9)) * 0.45 + (-dd60).clip(0, 1) * 1.5), index=feat.index).clip(0.01, 0.99)
        down10 = pd.Series(sigmoid((-mom10 / (scale * math.sqrt(10) + 1e-9)) * 0.38 + (-dd60).clip(0, 1) * 1.3), index=feat.index).clip(0.01, 0.99)
        down20 = pd.Series(sigmoid((-mom20 / (scale * math.sqrt(20) + 1e-9)) * 0.30 + (-dd252).clip(0, 1) * 1.2), index=feat.index).clip(0.01, 0.99)
        up_score = (up5 * 0.25 + up10 * 0.35 + up20 * 0.40).clip(0.01, 0.99)
        down_score = (down5 * 0.25 + down10 * 0.35 + down20 * 0.40).clip(0.01, 0.99)
        prob_overall_risk = (prob_high_vol * 0.55 + down_score * 0.45).clip(0.01, 0.99)
        prob_normal = (1.0 - prob_overall_risk).clip(0.01, 0.99)

        base_stock = 0.82 + (up_score - 0.50) * 0.20 - (prob_overall_risk - 0.50).clip(lower=0) * 0.60 - (down_score - 0.50).clip(lower=0) * 0.35
        stock_w = base_stock.clip(0.15, 1.0)
        defensive = (1.0 - stock_w).clip(0.0, 1.0)
        bond_w = defensive * 0.65
        cash_w = defensive * 0.35
        next_ret = close.pct_change().shift(-1).fillna(0.0)

        out = pd.DataFrame({
            "Date": pd.to_datetime(feat["Date"]).dt.date.astype(str),
            "ticker": t,
            "prob_normal": prob_normal,
            "prob_high_vol": prob_high_vol,
            "prob_overall_risk": prob_overall_risk,
            "prob_up_strengthening_5d": up5,
            "prob_up_strengthening_10d": up10,
            "prob_up_strengthening_20d": up20,
            "prob_up_strengthening_score": up_score,
            "prob_down_strengthening_5d": down5,
            "prob_down_strengthening_10d": down10,
            "prob_down_strengthening_20d": down20,
            "prob_down_strengthening_score": down_score,
            "stock_weight": stock_w,
            "bond_weight": bond_w,
            "cash_weight": cash_w,
            "stock_next_return": next_ret,
        })
        out = out.dropna(subset=["Date"]).reset_index(drop=True)
        validate_prediction_output(out)
        return out


class PredictionRepository:
    def __init__(self, prediction_dir: Path):
        self.prediction_dir = prediction_dir

    def prediction_path(self, ticker: str) -> Path:
        t = normalize_ticker(ticker).lower()
        return self.prediction_dir / f"{t}_{SOURCE_TAG}_predictions.csv"

    def summary_path(self, ticker: str) -> Path:
        t = normalize_ticker(ticker).lower()
        return self.prediction_dir / f"{t}_{SOURCE_TAG}_summary.json"

    def exists(self, ticker: str) -> bool:
        return self.prediction_path(ticker).exists()

    def read(self, ticker: str) -> pd.DataFrame:
        path = self.prediction_path(ticker)
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found for {ticker}: {path}")
        df = pd.read_csv(path)
        validate_prediction_output(df)
        return df

    def latest_date(self, ticker: str) -> Optional[str]:
        try:
            df = self.read(ticker)
        except FileNotFoundError:
            return None
        if df.empty:
            return None
        return str(pd.to_datetime(df["Date"]).max().date())


class PredictionGenerationService:
    def __init__(self, cache: MarketDataCache, repo: PredictionRepository, run_root: Path):
        self.cache = cache
        self.repo = repo
        self.run_root = run_root

    def generate_reference(self, tickers: Iterable[str], provider: str = "yahoo", force: bool = False, risk_sensitivity: float = 1.0) -> List[PredictionResult]:
        tx = LocalRunTransaction(self.run_root, "prediction_generation")
        engine = ReferenceV8641CompatibleEngine()
        results: List[PredictionResult] = []
        for raw in tickers:
            t = normalize_ticker(raw)
            path = self.repo.prediction_path(t)
            errors: List[str] = []
            try:
                if path.exists() and not force:
                    df_old = self.repo.read(t)
                    results.append(PredictionResult(t, path, "reference_v8641_compatible", self.repo.latest_date(t), len(df_old), []))
                    continue
                ohlcv = self.cache.read_ohlcv(t, provider)
                pred = engine.predict(t, ohlcv, risk_sensitivity=risk_sensitivity)
                atomic_write_csv(path, pred)
                summary = {
                    "ticker": t,
                    "model_version": "v8.6.41_model_label_fixed_compatible_reference",
                    "engine_mode": "reference_v8641_compatible",
                    "created_at": utc_now_iso(),
                    "rows": int(len(pred)),
                    "latest_date": str(pred["Date"].iloc[-1]) if len(pred) else None,
                    "source_cache_provider": provider,
                }
                atomic_write_json(self.repo.summary_path(t), summary)
                tx.add_item(ticker=t, prediction_path=str(path), rows=len(pred), latest_date=summary["latest_date"])
                results.append(PredictionResult(t, path, "reference_v8641_compatible", summary["latest_date"], len(pred), []))
            except Exception as exc:
                errors.append(str(exc))
                results.append(PredictionResult(t, path, "reference_v8641_compatible", None, 0, errors))
        tx.commit()
        return results
