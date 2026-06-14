from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.model.training_service import TrainingService
from backend.app.model.model_registry import ModelRegistry


def assert_close(a, b, tol=1e-6):
    if abs(a - b) > tol:
        raise AssertionError(f"expected {a} ~= {b}")


def test_recency_weights():
    w = TrainingService._recency_sample_weight(253, half_life=252)
    if len(w) != 253:
        raise AssertionError("weight length mismatch")
    if not all(w[i] <= w[i + 1] for i in range(len(w) - 1)):
        raise AssertionError("recency weights must be monotonic increasing")
    assert_close(float(w.mean()), 1.0, tol=1e-10)
    ratio = float(w[-253] / w[-1])
    if not (0.49 <= ratio <= 0.51):
        raise AssertionError(f"half-life ratio unexpected: {ratio}")


def test_walk_forward_modes():
    exp = TrainingService._walk_forward_slices(1500, mode="expanding")
    if not exp or any(f[0] != 0 for f in exp):
        raise AssertionError(f"expanding folds must start at zero: {exp}")
    roll = TrainingService._walk_forward_slices(1500, mode="rolling", rolling_train_rows=600)
    if not roll:
        raise AssertionError("rolling folds empty")
    if not any(f[0] > 0 for f in roll[1:]):
        raise AssertionError(f"rolling folds should move train_start after first fold: {roll}")
    for tr0, tr1, va0, va1 in roll:
        if not (tr0 < tr1 <= va0 < va1):
            raise AssertionError(f"leak or bad ordering in fold: {(tr0, tr1, va0, va1)}")


def test_activation_gate_extra_stability_checks():
    ok_item = {
        "status": "OK",
        "ok_fold_count": 3,
        "roc_auc": 0.56,
        "roc_auc_worst": 0.50,
        "pr_auc": 0.25,
        "positive_rate": 0.20,
        "positive_rate_worst": 0.18,
        "pr_auc_lift": 1.25,
        "pr_auc_lift_worst": 1.05,
        "brier": 0.20,
        "brier_worst": 0.30,
    }
    gate = ModelRegistry.evaluate_activation_gate({"metrics": {"head_10D": dict(ok_item)}})
    if not gate["passed"]:
        raise AssertionError(f"valid gate unexpectedly failed: {gate}")

    low_pr = dict(ok_item)
    low_pr["pr_auc_lift"] = 1.00
    gate = ModelRegistry.evaluate_activation_gate({"metrics": {"head_10D": low_pr}})
    if gate["passed"]:
        raise AssertionError("low PR lift should fail gate")

    bad_brier = dict(ok_item)
    bad_brier["brier_worst"] = 0.50
    gate = ModelRegistry.evaluate_activation_gate({"metrics": {"head_10D": bad_brier}})
    if gate["passed"]:
        raise AssertionError("bad brier worst should fail gate")

    bad_pos = dict(ok_item)
    bad_pos["positive_rate_worst"] = 0.995
    gate = ModelRegistry.evaluate_activation_gate({"metrics": {"head_10D": bad_pos}})
    if gate["passed"]:
        raise AssertionError("extreme positive_rate_worst should fail gate")


if __name__ == "__main__":
    test_recency_weights()
    test_walk_forward_modes()
    test_activation_gate_extra_stability_checks()
    print("v0.3.3 recency training tests passed")
