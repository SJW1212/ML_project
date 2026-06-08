from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

try:
    from filelock import FileLock
except Exception:  # pragma: no cover
    FileLock = None


@contextmanager
def _optional_lock(lock_path: Path):
    if FileLock is None:
        yield
        return
    with FileLock(str(lock_path), timeout=10):
        yield


class ModelRegistry:
    """File-backed model registry with activation gate and file locking."""

    def __init__(self, registry_dir: Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.registry_dir / "model_registry.json"
        self.lock_path = self.registry_dir / "model_registry.json.lock"
        if not self.path.exists():
            self._save_locked({"models": [], "active_model_version": "v8.6.41_label_fixed", "active_mode": "prediction_file"})

    def _load_unlocked(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save_unlocked(self, data: Dict[str, Any]) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _load_locked(self) -> Dict[str, Any]:
        with _optional_lock(self.lock_path):
            return self._load_unlocked()

    def _save_locked(self, data: Dict[str, Any]) -> None:
        with _optional_lock(self.lock_path):
            self._save_unlocked(data)

    def active(self) -> Dict[str, Any]:
        data = self._load_locked()
        version = data.get("active_model_version", "v8.6.41_label_fixed")
        model = next((m for m in data.get("models", []) if m.get("model_version") == version and m.get("status") == "ACTIVE"), None)
        return {
            "active_model_version": version,
            "active_mode": data.get("active_mode", "prediction_file"),
            "model": model,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return self._load_locked().get("models", [])

    def register(self, metadata: Dict[str, Any], status: str = "CANDIDATE") -> None:
        with _optional_lock(self.lock_path):
            data = self._load_unlocked()
            metadata = dict(metadata)
            metadata["status"] = status
            gate = self.evaluate_activation_gate(metadata)
            metadata.setdefault("activation_gate", gate)
            metadata.setdefault("activation_gate_passed", gate.get("passed", False))
            models = [m for m in data.get("models", []) if m.get("model_id") != metadata.get("model_id")]
            models.append(metadata)
            data["models"] = models
            self._save_unlocked(data)

    @staticmethod
    def _metric_value(item: Dict[str, Any], metric_name: str):
        """Read a metric from the flattened item or from walk_forward.aggregate.

        TrainingService v0.3.1 stores mean metrics at the top level and full
        fold aggregates under item["walk_forward"]["aggregate"]. The activation
        gate must inspect both mean and worst-fold values, so this helper keeps
        the registry tolerant to either schema.
        """
        if metric_name in item:
            return item.get(metric_name)
        aggregate = ((item.get("walk_forward") or {}).get("aggregate") or {})
        return aggregate.get(metric_name)

    @staticmethod
    def evaluate_activation_gate(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Safety gate before a candidate can become ACTIVE.

        The gate is deliberately conservative. It rejects models that have no
        measurable discrimination even when their Brier score looks acceptable.
        This is still not a profitability guarantee; portfolio-level OOS tests
        and prediction_file-vs-live comparisons remain mandatory before operation.
        """
        metrics = metadata.get("metrics") or {}
        thresholds = {
            "min_ok_folds": 2,
            "max_brier_mean": 0.35,
            "max_brier_worst": 0.45,
            "min_positive_rate": 0.02,
            "max_positive_rate": 0.98,
            "min_positive_rate_worst": 0.01,
            "max_positive_rate_worst": 0.99,
            "min_roc_auc_mean": 0.52,
            "min_roc_auc_worst": 0.48,
            "min_pr_auc_lift_mean": 1.02,
            "min_pr_auc_lift_worst": 0.95,
        }
        ok = []
        failed_reasons = []
        warning_reasons = []

        for key, item in metrics.items():
            if not isinstance(item, dict) or item.get("status") != "OK":
                continue
            ok.append(key)

            ok_folds = item.get("ok_fold_count")
            if ok_folds is not None and int(ok_folds) < thresholds["min_ok_folds"]:
                failed_reasons.append(
                    f"{key}: fewer than {thresholds['min_ok_folds']} valid walk-forward folds"
                )

            brier = item.get("brier")
            if brier is not None and float(brier) > thresholds["max_brier_mean"]:
                failed_reasons.append(
                    f"{key}: brier {float(brier):.4f} > {thresholds['max_brier_mean']:.2f}"
                )
            brier_worst = ModelRegistry._metric_value(item, "brier_worst")
            if brier_worst is not None and float(brier_worst) > thresholds["max_brier_worst"]:
                failed_reasons.append(
                    f"{key}: brier_worst {float(brier_worst):.4f} > {thresholds['max_brier_worst']:.2f} "
                    "(unstable calibration in worst fold)"
                )

            pos = item.get("positive_rate")
            if pos is not None and not (thresholds["min_positive_rate"] <= float(pos) <= thresholds["max_positive_rate"]):
                failed_reasons.append(
                    f"{key}: positive_rate {float(pos):.4f} outside "
                    f"[{thresholds['min_positive_rate']:.2f}, {thresholds['max_positive_rate']:.2f}]"
                )
            pos_worst = ModelRegistry._metric_value(item, "positive_rate_worst")
            if pos_worst is not None and not (thresholds["min_positive_rate_worst"] <= float(pos_worst) <= thresholds["max_positive_rate_worst"]):
                failed_reasons.append(
                    f"{key}: positive_rate_worst {float(pos_worst):.4f} outside "
                    f"[{thresholds['min_positive_rate_worst']:.2f}, {thresholds['max_positive_rate_worst']:.2f}]"
                )

            roc = item.get("roc_auc")
            pr = item.get("pr_auc")
            if roc is None and pr is None:
                failed_reasons.append(f"{key}: both roc_auc and pr_auc are missing")
            elif roc is None:
                failed_reasons.append(f"{key}: roc_auc is missing; discrimination cannot be verified")
            else:
                roc_float = float(roc)
                if roc_float < thresholds["min_roc_auc_mean"]:
                    failed_reasons.append(
                        f"{key}: roc_auc {roc_float:.4f} < {thresholds['min_roc_auc_mean']:.2f} "
                        "(insufficient discrimination)"
                    )

            roc_worst = ModelRegistry._metric_value(item, "roc_auc_worst")
            if roc_worst is None:
                warning_reasons.append(f"{key}: roc_auc_worst missing; fold stability not fully checked")
            elif float(roc_worst) < thresholds["min_roc_auc_worst"]:
                failed_reasons.append(
                    f"{key}: roc_auc_worst {float(roc_worst):.4f} < {thresholds['min_roc_auc_worst']:.2f} "
                    "(unstable worst fold)"
                )

            pr_lift = ModelRegistry._metric_value(item, "pr_auc_lift")
            pr_lift_worst = ModelRegistry._metric_value(item, "pr_auc_lift_worst")
            if pr_lift is None:
                warning_reasons.append(f"{key}: pr_auc_lift missing; PR quality vs base rate not checked")
            elif float(pr_lift) < thresholds["min_pr_auc_lift_mean"]:
                failed_reasons.append(
                    f"{key}: pr_auc_lift {float(pr_lift):.4f} < {thresholds['min_pr_auc_lift_mean']:.2f} "
                    "(insufficient PR lift over positive-rate baseline)"
                )
            if pr_lift_worst is None:
                warning_reasons.append(f"{key}: pr_auc_lift_worst missing; worst-fold PR stability not checked")
            elif float(pr_lift_worst) < thresholds["min_pr_auc_lift_worst"]:
                failed_reasons.append(
                    f"{key}: pr_auc_lift_worst {float(pr_lift_worst):.4f} < {thresholds['min_pr_auc_lift_worst']:.2f} "
                    "(weak worst-fold PR lift)"
                )

        if not ok:
            failed_reasons.append("no OK head metrics")

        return {
            "passed": len(failed_reasons) == 0,
            "ok_metric_count": len(ok),
            "failed_reasons": failed_reasons,
            "warning_reasons": warning_reasons,
            "thresholds": thresholds,
            "required_note": "This gate is necessary but not sufficient. Run prediction_file vs live and portfolio OOS checks before production use.",
        }

    def activate(self, model_version: str, mode: str = "live_inference", force: bool = False) -> None:
        with _optional_lock(self.lock_path):
            data = self._load_unlocked()
            found = False
            selected = None
            for m in data.get("models", []):
                if m.get("model_version") == model_version:
                    selected = m
                    found = True
                    break
            if not found and model_version != "v8.6.41_label_fixed":
                raise ValueError(f"Model version not found: {model_version}")
            if model_version != "v8.6.41_label_fixed" and not force:
                gate = self.evaluate_activation_gate(selected or {})
                if not gate.get("passed"):
                    raise ValueError(f"Activation gate failed for {model_version}: {gate.get('failed_reasons')}")
                selected["activation_gate"] = gate
                selected["activation_gate_passed"] = True

            for m in data.get("models", []):
                if m.get("model_version") == model_version:
                    m["status"] = "ACTIVE"
                elif m.get("status") == "ACTIVE":
                    m["status"] = "ARCHIVED"
            data["active_model_version"] = model_version
            data["active_mode"] = mode
            self._save_unlocked(data)
