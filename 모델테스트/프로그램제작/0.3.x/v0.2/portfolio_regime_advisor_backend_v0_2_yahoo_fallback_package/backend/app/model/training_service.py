from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
    from sklearn.ensemble import HistGradientBoostingClassifier

from ..features.feature_pipeline import FeaturePipeline
from .model_loader import ModelLoader
from .model_registry import ModelRegistry


class TrainingService:
    """Candidate model trainer.

    It trains a compact multi-head model compatible with the backend architecture.
    It does not replace the locked v8.6.41 baseline unless explicitly activated.
    """

    def __init__(self, feature_pipeline: FeaturePipeline, model_loader: ModelLoader, registry: ModelRegistry):
        self.feature_pipeline = feature_pipeline
        self.model_loader = model_loader
        self.registry = registry

    @staticmethod
    def _make_model():
        if XGBClassifier is not None:
            return XGBClassifier(
                n_estimators=160,
                learning_rate=0.025,
                max_depth=2,
                min_child_weight=8,
                subsample=0.90,
                colsample_bytree=0.85,
                reg_lambda=10.0,
                reg_alpha=0.2,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=2,
                random_state=42,
            )
        return HistGradientBoostingClassifier(max_iter=160, learning_rate=0.025, max_leaf_nodes=7, random_state=42)

    @staticmethod
    def _safe_metrics(y_true, y_prob) -> Dict[str, Optional[float]]:
        out = {}
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None
        except Exception:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else None
        except Exception:
            out["pr_auc"] = None
        try:
            out["brier"] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            out["brier"] = None
        out["positive_rate"] = float(np.mean(y_true)) if len(y_true) else None
        return out

    def train_ticker_candidate(self, ticker: str, ohlcv: pd.DataFrame, horizons: Iterable[str], model_version: Optional[str] = None) -> Dict:
        ticker = ticker.upper()
        model_version = model_version or f"candidate_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
        features = self.feature_pipeline.build(ohlcv, include_labels=True)
        X_cols = self.feature_pipeline.FEATURE_COLUMNS
        metrics: Dict[str, dict] = {}
        artifacts: List[str] = []
        for horizon in horizons:
            hnum = int(horizon.upper().replace("D", ""))
            head_targets = {
                "highvol": f"y_high_vol_{hnum}d",
                "up_strength": f"y_up_strength_{hnum}d",
                "down_strength": f"y_down_strength_{hnum}d",
            }
            for head, y_col in head_targets.items():
                train_df = features.dropna(subset=X_cols + [y_col]).copy()
                if len(train_df) < 500:
                    metrics[f"{head}_{horizon}"] = {"status": "SKIPPED", "reason": "insufficient_rows", "rows": len(train_df)}
                    continue
                y = train_df[y_col].astype(int)
                if y.nunique() < 2:
                    metrics[f"{head}_{horizon}"] = {"status": "SKIPPED", "reason": "single_class", "rows": len(train_df)}
                    continue
                split = int(len(train_df) * 0.8)
                X_train, X_valid = train_df.iloc[:split][X_cols], train_df.iloc[split:][X_cols]
                y_train, y_valid = y.iloc[:split], y.iloc[split:]
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", self._make_model()),
                ])
                pipe.fit(X_train, y_train)
                if hasattr(pipe, "predict_proba"):
                    y_prob = pipe.predict_proba(X_valid)[:, 1]
                else:
                    y_prob = pipe.predict(X_valid)
                key = f"{head}_{horizon}"
                metrics[key] = {"status": "OK", "rows": int(len(train_df)), "valid_rows": int(len(X_valid)), **self._safe_metrics(y_valid, y_prob)}
                path = self.model_loader.save(pipe, model_version, ticker, head, horizon)
                artifacts.append(str(path))
        metadata = {
            "model_id": str(uuid.uuid4()),
            "model_version": model_version,
            "ticker": ticker,
            "horizons": list(horizons),
            "feature_set_version": "feature_pipeline_v1_core_subset",
            "label_mode": "volatility_scaled_multi_head_candidate",
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "metrics": metrics,
            "artifacts": artifacts,
        }
        self.registry.register(metadata, status="CANDIDATE")
        # Save feature columns/config near artifacts
        meta_dir = self.model_loader.model_dir / model_version / ticker
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "feature_columns.json").write_text(json.dumps(X_cols, indent=2), encoding="utf-8")
        return metadata
