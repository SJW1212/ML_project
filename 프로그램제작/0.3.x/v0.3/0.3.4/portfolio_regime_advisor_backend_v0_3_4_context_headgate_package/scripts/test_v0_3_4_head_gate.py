from __future__ import annotations

from backend.app.model.head_gate import GateStatus, HeadLevelGate, apply_fallback_probability
from backend.app.model.selective_inference_service import SelectiveInferenceService


def test_head_gate_pass_uncertain_fail():
    gate = HeadLevelGate()
    assert gate.evaluate("highvol_5D", {"roc_auc_worst": 0.55, "pr_auc_lift_worst": 1.1, "brier_worst": 0.22}).status == GateStatus.PASS
    assert gate.evaluate("up_strength_5D", {"roc_auc_worst": 0.505, "pr_auc_lift_worst": 1.0, "brier_worst": 0.30}).status == GateStatus.UNCERTAIN
    assert gate.evaluate("down_strength_20D", {"roc_auc_worst": 0.42, "pr_auc_lift_worst": 0.8, "brier_worst": 0.30}).status == GateStatus.FAIL


def test_softened_probability():
    gate = HeadLevelGate().evaluate("up_strength_5D", {"roc_auc_worst": 0.505, "pr_auc_lift_worst": 1.0, "brier_worst": 0.30})
    adjusted = apply_fallback_probability(0.9, gate)
    assert 0.5 < adjusted < 0.9


def test_selective_down_fail_highvol_pass_watch():
    svc = SelectiveInferenceService()
    out = svc.adjust(
        live_probs={"highvol_10D": 0.7, "down_strength_10D": 0.8},
        metrics={
            "highvol_10D": {"roc_auc_worst": 0.60, "pr_auc_lift_worst": 1.20, "brier_worst": 0.20},
            "down_strength_10D": {"roc_auc_worst": 0.30, "pr_auc_lift_worst": 0.70, "brier_worst": 0.20},
        },
        context_features={"ctx_spy_drawdown_252": -0.10},
    )
    assert "DOWN_STRENGTH_FAIL_HIGHVOL_PASS_DEFENSIVE_WATCH_ALLOWED" in out["selective_warnings"]
    assert out["head_gates"]["highvol_10D"]["status"] == "PASS"
    assert out["head_gates"]["down_strength_10D"]["status"] == "FAIL"


if __name__ == "__main__":
    test_head_gate_pass_uncertain_fail()
    test_softened_probability()
    test_selective_down_fail_highvol_pass_watch()
    print("v0.3.4 head gate tests passed")
