from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.model.model_registry import ModelRegistry


def gate(metrics):
    return ModelRegistry.evaluate_activation_gate({"metrics": metrics})


def assert_false(result, text):
    assert result["passed"] is False, result
    joined = " | ".join(result.get("failed_reasons", []))
    assert text in joined, result


def test_rejects_low_mean_roc_auc():
    result = gate({
        "highvol_20D": {
            "status": "OK",
            "ok_fold_count": 3,
            "roc_auc": 0.40,
            "roc_auc_worst": 0.39,
            "pr_auc": 0.30,
            "brier": 0.20,
            "positive_rate": 0.30,
        }
    })
    assert_false(result, "roc_auc")


def test_rejects_low_worst_fold_roc_auc():
    result = gate({
        "highvol_20D": {
            "status": "OK",
            "ok_fold_count": 3,
            "roc_auc": 0.56,
            "roc_auc_worst": 0.42,
            "pr_auc": 0.30,
            "brier": 0.20,
            "positive_rate": 0.30,
        }
    })
    assert_false(result, "roc_auc_worst")


def test_accepts_minimal_valid_metrics():
    result = gate({
        "highvol_20D": {
            "status": "OK",
            "ok_fold_count": 3,
            "roc_auc": 0.53,
            "roc_auc_worst": 0.49,
            "pr_auc": 0.30,
            "brier": 0.20,
            "positive_rate": 0.30,
        }
    })
    assert result["passed"] is True, result


if __name__ == "__main__":
    test_rejects_low_mean_roc_auc()
    test_rejects_low_worst_fold_roc_auc()
    test_accepts_minimal_valid_metrics()
    print("v0.3.2 activation gate tests passed")
