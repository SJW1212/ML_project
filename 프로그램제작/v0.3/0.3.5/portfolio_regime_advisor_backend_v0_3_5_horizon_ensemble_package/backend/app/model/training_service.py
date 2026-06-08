from __future__ import annotations

import json
import uuid
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

from ..data.market_data_repository import MarketDataRepository
from ..features.context_asset_universe import ContextAssetUniverse
from ..features.feature_pipeline import FeatureConfig, FeaturePipeline
from ..features.market_context_feature_builder import MarketContextFeatureBuilder
from .model_artifact_store import ModelArtifactStore
from .model_loader import ModelLoader
from .model_registry import ModelRegistry


class TrainingService:
    """Candidate model trainer with time-series validation and recency weighting.

    v0.3.3 keeps the locked v8.6.41 prediction-file baseline as the operating
    default, but brings runtime candidate training closer to the v8.6.41
    recency-weighted philosophy. The default validation remains leak-safe
    walk-forward, while training rows inside each fold are exponentially weighted
    by recency.
    """

    DEFAULT_HALF_LIFE_BY_HORIZON = {5: 126, 10: 252, 20: 504}
    MIN_TRAIN_ROWS = 500
    MIN_VALID_ROWS = 60
    WALK_FORWARD_SPLITS = 3

    def __init__(self, feature_pipeline: FeaturePipeline, model_loader: ModelLoader, registry: ModelRegistry, market_data_repository: MarketDataRepository | None = None):
        self.feature_pipeline = feature_pipeline
        self.model_loader = model_loader
        self.registry = registry
        self.market_data_repository = market_data_repository
        self.artifact_store = ModelArtifactStore(model_loader.model_dir)

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
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        out: Dict[str, Optional[float]] = {}
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
        out["sample_count"] = int(len(y_true))
        if out["pr_auc"] is not None and out["positive_rate"] not in (None, 0.0):
            out["pr_auc_lift"] = float(out["pr_auc"] / out["positive_rate"])
        else:
            out["pr_auc_lift"] = None
        return out

    @staticmethod
    def _predict_prob(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)
            if getattr(proba, "ndim", 1) == 2 and proba.shape[1] > 1:
                return np.asarray(proba[:, 1], dtype=float)
            return np.asarray(proba).ravel().astype(float)
        return np.asarray(pipe.predict(X), dtype=float).ravel()

    @classmethod
    def _half_life_for_horizon(cls, horizon_days: int, overrides: Optional[Dict[str, int]] = None) -> int:
        if overrides:
            for key in (str(horizon_days), f"{horizon_days}D", f"{horizon_days}d"):
                if key in overrides and overrides[key]:
                    return int(overrides[key])
        return int(cls.DEFAULT_HALF_LIFE_BY_HORIZON.get(horizon_days, max(126, horizon_days * 25)))

    @staticmethod
    def _recency_sample_weight(n: int, half_life: int) -> np.ndarray:
        """Exponential row weights normalized to mean 1.

        The newest row receives the highest weight. A row ``half_life`` samples
        older receives roughly half of the newest row's unnormalized weight.
        Normalizing to mean 1 keeps regularization behavior more stable across
        folds and train-window lengths.
        """
        if n <= 0:
            return np.asarray([], dtype=float)
        half_life = max(int(half_life), 1)
        idx = np.arange(n, dtype=float)
        weights = np.power(0.5, (n - 1 - idx) / half_life)
        mean = float(np.mean(weights)) if len(weights) else 1.0
        if mean <= 0 or not np.isfinite(mean):
            return np.ones(n, dtype=float)
        return weights / mean

    @staticmethod
    def _sample_weight_summary(weights: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        if weights is None or len(weights) == 0:
            return {
                "sample_weight_used": False,
                "sample_weight_min": None,
                "sample_weight_max": None,
                "sample_weight_mean": None,
                "effective_sample_size": None,
            }
        weights = np.asarray(weights, dtype=float)
        denom = float(np.sum(weights ** 2))
        ess = float((np.sum(weights) ** 2) / denom) if denom > 0 else None
        return {
            "sample_weight_used": True,
            "sample_weight_min": float(np.min(weights)),
            "sample_weight_max": float(np.max(weights)),
            "sample_weight_mean": float(np.mean(weights)),
            "effective_sample_size": ess,
        }

    @staticmethod
    def _fit_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> Pipeline:
        if sample_weight is None:
            pipe.fit(X, y)
            return pipe
        try:
            pipe.fit(X, y, model__sample_weight=sample_weight)
        except TypeError:
            # Fallback for unexpected estimator versions. This keeps the endpoint
            # usable, but metadata still records whether sample weights were intended.
            pipe.fit(X, y)
        return pipe

    @staticmethod
    def _walk_forward_slices(
        n: int,
        n_splits: int = 3,
        min_train_rows: int = 500,
        min_valid_rows: int = 60,
        mode: str = "expanding",
        rolling_train_rows: Optional[int] = None,
    ):
        """Leak-safe walk-forward folds.

        mode="expanding": train_start is always 0, train_end expands.
        mode="rolling": train_start moves forward and caps train length. This is
        useful for regime-sensitive experiments, but the default remains
        expanding + recency sample weights for v8.6.41 alignment.
        """
        if n < min_train_rows + min_valid_rows:
            return []
        mode = (mode or "expanding").lower().strip()
        remaining = n - min_train_rows
        valid_size = max(min_valid_rows, remaining // (n_splits + 1))
        folds = []
        train_end = min_train_rows
        while len(folds) < n_splits:
            valid_start = train_end
            valid_end = min(n, valid_start + valid_size)
            if valid_end - valid_start < min_valid_rows:
                break
            if mode == "rolling":
                window = int(rolling_train_rows or min_train_rows)
                window = max(window, min_train_rows)
                train_start = max(0, train_end - window)
            else:
                train_start = 0
            if train_end - train_start < min_train_rows:
                break
            folds.append((train_start, train_end, valid_start, valid_end))
            train_end = valid_end
            if n - train_end < min_valid_rows:
                break
        return folds

    def _walk_forward_metrics(
        self,
        train_df: pd.DataFrame,
        X_cols: List[str],
        y: pd.Series,
        horizon_days: int,
        sample_weight_mode: str = "recency",
        walk_forward_mode: str = "expanding",
        rolling_train_rows: Optional[int] = None,
        recency_half_life_by_horizon: Optional[Dict[str, int]] = None,
    ) -> Dict:
        half_life = self._half_life_for_horizon(horizon_days, recency_half_life_by_horizon)
        folds = self._walk_forward_slices(
            len(train_df),
            n_splits=self.WALK_FORWARD_SPLITS,
            min_train_rows=self.MIN_TRAIN_ROWS,
            min_valid_rows=self.MIN_VALID_ROWS,
            mode=walk_forward_mode,
            rolling_train_rows=rolling_train_rows,
        )
        fold_rows = []
        for fold_id, (tr0, tr1, va0, va1) in enumerate(folds, start=1):
            X_train = train_df.iloc[tr0:tr1][X_cols]
            y_train = y.iloc[tr0:tr1]
            X_valid = train_df.iloc[va0:va1][X_cols]
            y_valid = y.iloc[va0:va1]
            if y_train.nunique() < 2 or y_valid.nunique() < 2:
                fold_rows.append({
                    "fold": fold_id,
                    "status": "SKIPPED",
                    "reason": "single_class_train_or_validation",
                    "train_start_idx": int(tr0),
                    "train_end_idx": int(tr1),
                    "valid_start_idx": int(va0),
                    "valid_end_idx": int(va1),
                    "train_rows": int(len(X_train)),
                    "valid_rows": int(len(X_valid)),
                    "positive_rate": float(y_valid.mean()) if len(y_valid) else None,
                })
                continue
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", self._make_model()),
            ])
            weights = None
            if sample_weight_mode == "recency":
                weights = self._recency_sample_weight(len(X_train), half_life)
            self._fit_pipeline(pipe, X_train, y_train, sample_weight=weights)
            y_prob = self._predict_prob(pipe, X_valid)
            fold_metrics = self._safe_metrics(y_valid, y_prob)
            fold_metrics.update({
                "fold": fold_id,
                "status": "OK",
                "train_start_idx": int(tr0),
                "train_end_idx": int(tr1),
                "valid_start_idx": int(va0),
                "valid_end_idx": int(va1),
                "train_rows": int(len(X_train)),
                "valid_rows": int(len(X_valid)),
                "train_rows_actual": int(len(X_train)),
                "valid_rows_actual": int(len(X_valid)),
                "train_start_date": str(pd.to_datetime(train_df.iloc[tr0]["Date"]).date()) if "Date" in train_df.columns and len(X_train) else None,
                "train_end_date": str(pd.to_datetime(train_df.iloc[tr1 - 1]["Date"]).date()) if "Date" in train_df.columns and len(X_train) else None,
                "valid_start_date": str(pd.to_datetime(train_df.iloc[va0]["Date"]).date()) if "Date" in train_df.columns and len(X_valid) else None,
                "valid_end_date": str(pd.to_datetime(train_df.iloc[va1 - 1]["Date"]).date()) if "Date" in train_df.columns and len(X_valid) else None,
                "sample_weight_mode": sample_weight_mode,
                "recency_half_life": int(half_life) if sample_weight_mode == "recency" else None,
            })
            fold_metrics.update(self._sample_weight_summary(weights))
            fold_rows.append(fold_metrics)

        ok_folds = [r for r in fold_rows if r.get("status") == "OK"]
        aggregate: Dict[str, Optional[float] | int | str] = {
            "fold_count": int(len(fold_rows)),
            "ok_fold_count": int(len(ok_folds)),
            "validation_method": f"{walk_forward_mode}_walk_forward",
            "sample_weight_mode": sample_weight_mode,
            "recency_half_life": int(half_life) if sample_weight_mode == "recency" else None,
            "rolling_train_rows": int(rolling_train_rows) if rolling_train_rows else None,
        }
        for metric in ["roc_auc", "pr_auc", "pr_auc_lift", "brier", "positive_rate", "effective_sample_size"]:
            vals = [r.get(metric) for r in ok_folds if r.get(metric) is not None]
            aggregate[f"{metric}_mean"] = float(np.mean(vals)) if vals else None
            aggregate[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else None
            aggregate[f"{metric}_worst"] = float(np.max(vals)) if vals and metric == "brier" else (float(np.min(vals)) if vals else None)
        return {"folds": fold_rows, "aggregate": aggregate}

    def _fit_final(
        self,
        train_df: pd.DataFrame,
        X_cols: List[str],
        y: pd.Series,
        horizon_days: int,
        sample_weight_mode: str = "recency",
        recency_half_life_by_horizon: Optional[Dict[str, int]] = None,
    ) -> Pipeline:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", self._make_model()),
        ])
        weights = None
        if sample_weight_mode == "recency":
            half_life = self._half_life_for_horizon(horizon_days, recency_half_life_by_horizon)
            weights = self._recency_sample_weight(len(train_df), half_life)
        return self._fit_pipeline(pipe, train_df[X_cols], y, sample_weight=weights)

    def train_ticker_candidate(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        horizons: Iterable[str],
        model_version: Optional[str] = None,
        sample_weight_mode: str = "recency",
        walk_forward_mode: str = "expanding",
        rolling_train_rows: Optional[int] = None,
        recency_half_life_by_horizon: Optional[Dict[str, int]] = None,
        use_context_features: bool = False,
        context_provider: str = "auto",
        market: str = "US",
    ) -> Dict:
        ticker = ticker.upper()
        horizons = [h.upper().replace(" ", "") for h in horizons]
        requested_horizon_ints = [int(h.replace("D", "")) for h in horizons]
        sample_weight_mode = (sample_weight_mode or "recency").lower().strip()
        if sample_weight_mode not in {"recency", "equal"}:
            raise ValueError("sample_weight_mode must be 'recency' or 'equal'")
        walk_forward_mode = (walk_forward_mode or "expanding").lower().strip()
        if walk_forward_mode not in {"expanding", "rolling"}:
            raise ValueError("walk_forward_mode must be 'expanding' or 'rolling'")

        # Use a local pipeline instance to avoid mutating the shared singleton while
        # background training and live inference may run concurrently.
        base_cfg = self.feature_pipeline.config
        local_pipeline = FeaturePipeline(FeatureConfig(
            horizons=requested_horizon_ints,
            vol_window=base_cfg.vol_window,
            min_abs_threshold=base_cfg.min_abs_threshold,
            max_abs_threshold=base_cfg.max_abs_threshold,
            k_direction=base_cfg.k_direction,
            k_high_vol=base_cfg.k_high_vol,
        ))
        features = local_pipeline.build(ohlcv, include_labels=True)
        context_summary = None
        context_warnings: List[str] = []
        if use_context_features:
            if self.market_data_repository is None:
                context_warnings.append("CONTEXT_FEATURES_REQUESTED_BUT_REPOSITORY_UNAVAILABLE")
            else:
                try:
                    target_index = pd.DatetimeIndex(pd.to_datetime(features["Date"]))
                    universe = ContextAssetUniverse(
                        repository=self.market_data_repository,
                        provider=context_provider,
                        market=market,
                        target_index=target_index,
                        strict=False,
                    )
                    # Reattach raw close so target-derived context overlays can be built.
                    raw = ohlcv.copy()
                    raw = raw.rename(columns={c: c.title() for c in raw.columns if c.lower() in {"date", "close"}})
                    raw["Date"] = pd.to_datetime(raw["Date"])
                    close_map = raw.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")["Close"]
                    features = features.copy()
                    features["Close"] = pd.to_datetime(features["Date"]).map(close_map)
                    builder = MarketContextFeatureBuilder()
                    features = builder.build(features, universe)
                    context_summary = universe.summary()
                    context_summary["builder_warnings"] = sorted(set(builder.warnings))
                    context_warnings.extend(builder.warnings)
                except Exception as exc:
                    context_warnings.append(f"CONTEXT_FEATURE_BUILD_FAILED:{exc}")

        model_version = model_version or f"candidate_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
        base_cols = list(local_pipeline.FEATURE_COLUMNS)
        context_cols = [c for c in features.columns if c.startswith("ctx_")] if use_context_features else []
        X_cols = base_cols + [c for c in context_cols if c not in base_cols]
        metrics: Dict[str, dict] = {}
        artifacts: List[str] = []
        training_config = {
            "sample_weight_mode": sample_weight_mode,
            "walk_forward_mode": walk_forward_mode,
            "rolling_train_rows": rolling_train_rows,
            "recency_half_life_by_horizon": recency_half_life_by_horizon or {
                f"{h}D": self.DEFAULT_HALF_LIFE_BY_HORIZON.get(h, max(126, h * 25)) for h in requested_horizon_ints
            },
            "min_train_rows": self.MIN_TRAIN_ROWS,
            "min_valid_rows": self.MIN_VALID_ROWS,
            "walk_forward_splits": self.WALK_FORWARD_SPLITS,
            "use_context_features": bool(use_context_features),
            "context_provider": context_provider,
            "market": market,
        }

        for horizon in horizons:
            hnum = int(horizon.replace("D", ""))
            head_targets = {
                "highvol": f"y_high_vol_{hnum}d",
                "up_strength": f"y_up_strength_{hnum}d",
                "down_strength": f"y_down_strength_{hnum}d",
            }
            for head, y_col in head_targets.items():
                key = f"{head}_{horizon}"
                train_df = features.dropna(subset=base_cols + [y_col]).copy()
                if len(train_df) < self.MIN_TRAIN_ROWS + self.MIN_VALID_ROWS:
                    metrics[key] = {"status": "SKIPPED", "reason": "insufficient_rows", "rows": int(len(train_df))}
                    continue
                y = train_df[y_col].astype(int)
                if y.nunique() < 2:
                    metrics[key] = {"status": "SKIPPED", "reason": "single_class", "rows": int(len(train_df)), "positive_rate": float(y.mean())}
                    continue

                wf = self._walk_forward_metrics(
                    train_df,
                    X_cols,
                    y,
                    horizon_days=hnum,
                    sample_weight_mode=sample_weight_mode,
                    walk_forward_mode=walk_forward_mode,
                    rolling_train_rows=rolling_train_rows,
                    recency_half_life_by_horizon=recency_half_life_by_horizon,
                )
                ok_fold_count = int(wf.get("aggregate", {}).get("ok_fold_count", 0) or 0)
                if ok_fold_count == 0:
                    metrics[key] = {
                        "status": "SKIPPED",
                        "reason": "no_valid_walk_forward_fold",
                        "rows": int(len(train_df)),
                        "walk_forward": wf,
                    }
                    continue

                final_model = self._fit_final(
                    train_df,
                    X_cols,
                    y,
                    horizon_days=hnum,
                    sample_weight_mode=sample_weight_mode,
                    recency_half_life_by_horizon=recency_half_life_by_horizon,
                )
                path = self.model_loader.save(final_model, model_version, ticker, head, horizon)
                artifacts.append(str(path))
                agg = wf["aggregate"]
                metrics[key] = {
                    "status": "OK",
                    "rows": int(len(train_df)),
                    "validation_method": agg.get("validation_method"),
                    "sample_weight_mode": sample_weight_mode,
                    "recency_half_life": agg.get("recency_half_life"),
                    "walk_forward": wf,
                    "roc_auc": agg.get("roc_auc_mean"),
                    "roc_auc_worst": agg.get("roc_auc_worst"),
                    "pr_auc": agg.get("pr_auc_mean"),
                    "pr_auc_worst": agg.get("pr_auc_worst"),
                    "pr_auc_lift": agg.get("pr_auc_lift_mean"),
                    "pr_auc_lift_worst": agg.get("pr_auc_lift_worst"),
                    "brier": agg.get("brier_mean"),
                    "brier_worst": agg.get("brier_worst"),
                    "positive_rate": agg.get("positive_rate_mean"),
                    "positive_rate_worst": agg.get("positive_rate_worst"),
                    "effective_sample_size": agg.get("effective_sample_size_mean"),
                    "ok_fold_count": ok_fold_count,
                }

        gate = self.registry.evaluate_activation_gate({"metrics": metrics})
        metadata = {
            "model_id": str(uuid.uuid4()),
            "model_version": model_version,
            "ticker": ticker,
            "horizons": list(horizons),
            "feature_set_version": "feature_pipeline_v1_core_subset_vol_scaled_labels" + ("_with_context_v0_3_5" if use_context_features else ""),
            "label_mode": "volatility_scaled_multi_head_candidate_v2",
            "training_method": "context_aware_recency_weighted_candidate_training_v0_3_5" if use_context_features else "recency_weighted_candidate_training_v0_3_3",
            "training_config": training_config,
            "context_summary": context_summary,
            "context_warnings": sorted(set(context_warnings)),
            "created_at": pd.Timestamp.utcnow().isoformat(),
            "metrics": metrics,
            "artifacts": artifacts,
            "activation_gate_passed": gate.get("passed", False),
            "activation_gate": gate,
        }
        self.registry.register(metadata, status="CANDIDATE")

        meta_dir = self.model_loader.model_dir / model_version / ticker
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "feature_columns.json").write_text(json.dumps(X_cols, indent=2), encoding="utf-8")
        manifest = self.artifact_store.build_manifest(model_version, ticker, horizons, X_cols, metrics)
        manifest["activation_gate"] = metadata["activation_gate"]
        manifest["training_config"] = training_config
        self.artifact_store.write_manifest(model_version, ticker, manifest)
        return metadata
