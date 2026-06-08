from backend.app.model.horizon_ensemble import HorizonEnsemble
from backend.app.model.soft_family_gate import SoftFamilyGate


def test_shrink_to_neutral():
    gate = SoftFamilyGate()
    assert abs(gate.shrink_to_neutral(0.80, 0.30) - 0.59) < 1e-9
    assert abs(gate.shrink_to_neutral(0.20, 0.50) - 0.35) < 1e-9
    assert abs(gate.shrink_to_neutral(0.80, 0.00) - 0.50) < 1e-9


def test_family_fail_is_reduced_not_dropped_when_not_inverted():
    ensemble = HorizonEnsemble()
    probs = {"highvol_5D": 0.80, "highvol_10D": 0.70, "highvol_20D": 0.65}
    head_gates = {
        "highvol_5D": {"status": "FAIL"},
        "highvol_10D": {"status": "FAIL"},
        "highvol_20D": {"status": "FAIL"},
    }
    metrics = {
        "highvol_5D": {"roc_auc": 0.50, "roc_auc_worst": 0.45, "pr_auc_lift": 1.00, "pr_auc_lift_worst": 0.90},
        "highvol_10D": {"roc_auc": 0.51, "roc_auc_worst": 0.46, "pr_auc_lift": 1.01, "pr_auc_lift_worst": 0.90},
        "highvol_20D": {"roc_auc": 0.49, "roc_auc_worst": 0.46, "pr_auc_lift": 0.99, "pr_auc_lift_worst": 0.90},
    }
    er = ensemble.combine_family("highvol", probs, head_gates)
    fg = SoftFamilyGate(ensemble=ensemble).evaluate_family("highvol", er, head_gates, metrics)
    assert fg.confidence > 0.0
    assert fg.effective_probability > 0.5
    assert fg.effective_probability < fg.raw_probability
    assert fg.status in {"WEAK_FAIL", "FAIL", "UNCERTAIN"}


def test_inverted_head_goes_neutral():
    ensemble = HorizonEnsemble()
    probs = {"up_strength_5D": 0.90, "up_strength_10D": 0.85, "up_strength_20D": 0.80}
    head_gates = {
        "up_strength_5D": {"status": "FAIL"},
        "up_strength_10D": {"status": "FAIL"},
        "up_strength_20D": {"status": "FAIL"},
    }
    metrics = {
        "up_strength_5D": {"roc_auc": 0.45, "roc_auc_worst": 0.30, "pr_auc_lift": 0.90, "pr_auc_lift_worst": 0.70},
        "up_strength_10D": {"roc_auc": 0.45, "roc_auc_worst": 0.30, "pr_auc_lift": 0.90, "pr_auc_lift_worst": 0.70},
        "up_strength_20D": {"roc_auc": 0.45, "roc_auc_worst": 0.30, "pr_auc_lift": 0.90, "pr_auc_lift_worst": 0.70},
    }
    er = ensemble.combine_family("up_strength", probs, head_gates)
    fg = SoftFamilyGate(ensemble=ensemble).evaluate_family("up_strength", er, head_gates, metrics)
    assert fg.confidence == 0.0
    assert fg.effective_probability == 0.5
    assert fg.status == "INVERTED"


if __name__ == "__main__":
    test_shrink_to_neutral()
    test_family_fail_is_reduced_not_dropped_when_not_inverted()
    test_inverted_head_goes_neutral()
    print("v0.3.6 soft family gate tests passed")
