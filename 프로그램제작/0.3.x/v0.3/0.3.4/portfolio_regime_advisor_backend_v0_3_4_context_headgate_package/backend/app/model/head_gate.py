from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Optional


class GateStatus(str, Enum):
    PASS = "PASS"
    UNCERTAIN = "UNCERTAIN"
    FAIL = "FAIL"


FallbackMode = Literal["live", "softened", "baseline", "neutral"]


@dataclass(frozen=True)
class HeadGateResult:
    head_name: str
    status: GateStatus
    roc_auc_worst: Optional[float]
    pr_lift_worst: Optional[float]
    brier_worst: Optional[float]
    fallback_mode: FallbackMode
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "head_name": self.head_name,
            "status": self.status.value,
            "roc_auc_worst": self.roc_auc_worst,
            "pr_lift_worst": self.pr_lift_worst,
            "brier_worst": self.brier_worst,
            "fallback_mode": self.fallback_mode,
            "reason": self.reason,
        }


class HeadLevelGate:
    """Head-level validation gate for selective live inference.

    This does not replace the conservative full-model activation gate. It lets
    diagnostic/experimental live inference use strong heads while falling back
    for weak heads.
    """

    PASS_THRESHOLDS = {
        "roc_auc_worst": 0.52,
        "pr_lift_worst": 1.05,
        "brier_worst": 0.40,
    }
    UNCERTAIN_THRESHOLDS = {
        "roc_auc_worst": 0.50,
        "pr_lift_worst": 0.98,
        "brier_worst": 0.45,
    }

    @staticmethod
    def _float(value) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def evaluate(self, head_name: str, metrics: Dict) -> HeadGateResult:
        roc = self._float(metrics.get("roc_auc_worst"))
        lift = self._float(metrics.get("pr_auc_lift_worst") or metrics.get("pr_lift_worst"))
        brier = self._float(metrics.get("brier_worst"))

        if roc is None or lift is None:
            return HeadGateResult(head_name, GateStatus.FAIL, roc, lift, brier, "baseline", "missing worst-fold discrimination metric")

        pass_ok = (
            roc >= self.PASS_THRESHOLDS["roc_auc_worst"] and
            lift >= self.PASS_THRESHOLDS["pr_lift_worst"] and
            (brier is None or brier <= self.PASS_THRESHOLDS["brier_worst"])
        )
        if pass_ok:
            return HeadGateResult(head_name, GateStatus.PASS, roc, lift, brier, "live", "worst-fold metrics passed")

        uncertain_ok = (
            roc >= self.UNCERTAIN_THRESHOLDS["roc_auc_worst"] and
            lift >= self.UNCERTAIN_THRESHOLDS["pr_lift_worst"] and
            (brier is None or brier <= self.UNCERTAIN_THRESHOLDS["brier_worst"])
        )
        if uncertain_ok:
            return HeadGateResult(head_name, GateStatus.UNCERTAIN, roc, lift, brier, "softened", "borderline worst-fold metrics; shrink probability")

        return HeadGateResult(head_name, GateStatus.FAIL, roc, lift, brier, "baseline", "worst-fold metrics failed")


def apply_fallback_probability(prob: float, gate: HeadGateResult, baseline_prob: Optional[float] = None) -> float:
    prob = float(prob)
    if gate.fallback_mode == "live":
        return prob
    if gate.fallback_mode == "baseline" and baseline_prob is not None:
        return float(baseline_prob)
    if gate.fallback_mode == "softened":
        roc = gate.roc_auc_worst if gate.roc_auc_worst is not None else 0.50
        alpha = max(0.0, min(1.0, (float(roc) - 0.50) / 0.10))
        return float(alpha * prob + (1.0 - alpha) * 0.50)
    return 0.50
