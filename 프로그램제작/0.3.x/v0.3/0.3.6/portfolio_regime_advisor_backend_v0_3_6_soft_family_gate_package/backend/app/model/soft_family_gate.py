from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

from .horizon_ensemble import HorizonEnsembleResult, HorizonEnsemble


@dataclass(frozen=True)
class FamilyGateConfig:
    """Soft family gate configuration.

    The gate does not hard-drop a weak family. Instead it estimates a confidence
    multiplier and shrinks the family ensemble probability toward the neutral
    0.5 value. This prevents an all-or-nothing ticker activation gate from
    discarding usable partial signals, while still limiting noisy families.
    """

    neutral_probability: float = 0.50
    pass_confidence: float = 1.00
    uncertain_confidence: float = 0.55
    weak_fail_confidence: float = 0.35
    fail_confidence_by_family: Dict[str, float] = field(default_factory=lambda: {
        "highvol": 0.20,
        "up_strength": 0.10,
        "down_strength": 0.20,
    })
    inverted_confidence: float = 0.00

    # Metric boundaries. These are deliberately conservative defaults and should
    # later be tuned with OOS portfolio performance, not only classification metrics.
    inverted_roc_auc_worst: float = 0.40
    inverted_pr_lift_worst: float = 0.80
    weak_fail_roc_auc_mean: float = 0.49
    weak_fail_pr_lift_mean: float = 0.98

    strong_family_confidence: float = 0.85
    pass_family_confidence: float = 0.70
    uncertain_family_confidence: float = 0.45
    weak_fail_family_confidence: float = 0.20


@dataclass(frozen=True)
class HeadSoftGateResult:
    head: str
    status: str
    confidence: float
    reason: str
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "head": self.head,
            "status": self.status,
            "confidence": self.confidence,
            "reason": self.reason,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class FamilySoftGateResult:
    family: str
    status: str
    confidence: float
    raw_probability: float
    effective_probability: float
    active_weight: float
    shrink_method: str
    head_results: Dict[str, HeadSoftGateResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "status": self.status,
            "confidence": self.confidence,
            "raw_probability": self.raw_probability,
            "effective_probability": self.effective_probability,
            "active_weight": self.active_weight,
            "shrink_method": self.shrink_method,
            "head_results": {k: v.to_dict() for k, v in self.head_results.items()},
            "warnings": self.warnings,
        }


class SoftFamilyGate:
    """Confidence-weighted gate over horizon ensembles.

    v0.3.5 introduced horizon ensembles. v0.3.6 changes how weak families are
    handled: a family no longer becomes useless simply because one or more heads
    fail the strict activation gate. Instead, every head contributes a confidence
    score, and the family probability is shrunk toward 0.5 according to the
    family-level confidence.
    """

    def __init__(self, config: Optional[FamilyGateConfig] = None, ensemble: Optional[HorizonEnsemble] = None):
        self.config = config or FamilyGateConfig()
        self.ensemble = ensemble or HorizonEnsemble()

    @staticmethod
    def _float(value, default: float = np.nan) -> float:
        try:
            value = float(value)
            if not np.isfinite(value):
                return default
            return value
        except Exception:
            return default

    @staticmethod
    def _status_from_payload(payload: Mapping[str, object] | None) -> str:
        if not payload:
            return "FAIL"
        status = payload.get("status")
        if hasattr(status, "value"):
            status = status.value
        status = str(status or "FAIL").upper()
        return status if status in {"PASS", "UNCERTAIN", "FAIL"} else "FAIL"

    @staticmethod
    def shrink_to_neutral(probability: float, confidence: float, neutral: float = 0.50) -> float:
        p = float(np.clip(probability, 0.0, 1.0))
        c = float(np.clip(confidence, 0.0, 1.0))
        n = float(np.clip(neutral, 0.0, 1.0))
        return float(np.clip(n + (p - n) * c, 0.0, 1.0))

    def _head_metrics(self, metrics: Mapping[str, Mapping[str, object]], head: str) -> Dict[str, Optional[float]]:
        raw = metrics.get(head, {}) or {}
        return {
            "roc_auc": self._float(raw.get("roc_auc"), default=np.nan),
            "roc_auc_worst": self._float(raw.get("roc_auc_worst"), default=np.nan),
            "pr_auc_lift": self._float(raw.get("pr_auc_lift"), default=np.nan),
            "pr_auc_lift_worst": self._float(raw.get("pr_auc_lift_worst"), default=np.nan),
            "brier": self._float(raw.get("brier"), default=np.nan),
            "brier_worst": self._float(raw.get("brier_worst"), default=np.nan),
        }

    def _confidence_for_head(self, family: str, head: str, status: str, head_metrics: Dict[str, Optional[float]]) -> HeadSoftGateResult:
        c = self.config
        roc = head_metrics.get("roc_auc")
        roc_worst = head_metrics.get("roc_auc_worst")
        lift = head_metrics.get("pr_auc_lift")
        lift_worst = head_metrics.get("pr_auc_lift_worst")

        if status == "PASS":
            return HeadSoftGateResult(head, "PASS", c.pass_confidence, "head gate PASS", head_metrics)
        if status == "UNCERTAIN":
            return HeadSoftGateResult(head, "UNCERTAIN", c.uncertain_confidence, "head gate UNCERTAIN; shrink toward neutral", head_metrics)

        # FAIL does not automatically mean zero. Only obviously inverted/unstable
        # folds are neutralized completely.
        inverted = False
        if np.isfinite(roc_worst) and roc_worst < c.inverted_roc_auc_worst:
            inverted = True
        if np.isfinite(lift_worst) and lift_worst < c.inverted_pr_lift_worst:
            inverted = True
        if inverted:
            return HeadSoftGateResult(head, "INVERTED", c.inverted_confidence, "worst-fold metric indicates inverted/unsafe signal", head_metrics)

        weak_fail = False
        if np.isfinite(roc) and roc >= c.weak_fail_roc_auc_mean:
            weak_fail = True
        if np.isfinite(lift) and lift >= c.weak_fail_pr_lift_mean:
            weak_fail = True
        if weak_fail:
            return HeadSoftGateResult(head, "WEAK_FAIL", c.weak_fail_confidence, "strict gate failed but mean signal is not unusable", head_metrics)

        return HeadSoftGateResult(
            head,
            "FAIL",
            c.fail_confidence_by_family.get(family, 0.20),
            "strict gate failed; keep only a small neutral-shrunk contribution",
            head_metrics,
        )

    def _family_status(self, confidence: float) -> str:
        c = self.config
        if confidence >= c.strong_family_confidence:
            return "STRONG_PASS"
        if confidence >= c.pass_family_confidence:
            return "PASS"
        if confidence >= c.uncertain_family_confidence:
            return "UNCERTAIN"
        if confidence >= c.weak_fail_family_confidence:
            return "WEAK_FAIL"
        if confidence > 0.0:
            return "FAIL"
        return "INVERTED"

    def evaluate_family(
        self,
        family: str,
        ensemble_result: HorizonEnsembleResult,
        head_gates: Mapping[str, Mapping[str, object]],
        metrics: Mapping[str, Mapping[str, object]],
    ) -> FamilySoftGateResult:
        family = family.lower()
        cfg = self.ensemble.configs.get(family)
        if cfg is None:
            raise ValueError(f"Unknown family: {family}")

        head_results: Dict[str, HeadSoftGateResult] = {}
        weighted_conf = 0.0
        base_weight_total = 0.0
        warnings: list[str] = []

        for horizon in cfg.horizons:
            h = horizon.upper()
            head = f"{family}_{h}"
            base_w = float(cfg.base_weights.get(h, 0.0))
            if base_w <= 0:
                continue
            status = self._status_from_payload(head_gates.get(head))
            hm = self._head_metrics(metrics, head)
            hr = self._confidence_for_head(family, head, status, hm)
            head_results[head] = hr
            weighted_conf += base_w * hr.confidence
            base_weight_total += base_w
            if hr.status in {"FAIL", "WEAK_FAIL", "INVERTED"}:
                warnings.append(f"{head}:{hr.status}")

        confidence = float(weighted_conf / base_weight_total) if base_weight_total > 0 else 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # v0.3.5 HorizonEnsemble drops strict FAIL heads before combining. For
        # v0.3.6 soft gating, that would make an all-FAIL-but-not-inverted family
        # look exactly neutral before the confidence shrink can do its job. Use
        # the raw horizon probabilities for the raw family signal, then apply
        # confidence shrinkage here.
        raw_weighted = 0.0
        raw_weight_total = 0.0
        for horizon in cfg.horizons:
            h = horizon.upper()
            if h not in ensemble_result.raw_probabilities:
                continue
            bw = float(cfg.base_weights.get(h, 0.0))
            if bw <= 0:
                continue
            raw_weighted += bw * float(np.clip(ensemble_result.raw_probabilities[h], 0.0, 1.0))
            raw_weight_total += bw
        raw_p = float(raw_weighted / raw_weight_total) if raw_weight_total > 0 else float(np.clip(ensemble_result.probability, 0.0, 1.0))
        raw_p = float(np.clip(raw_p, 0.0, 1.0))
        eff_p = self.shrink_to_neutral(raw_p, confidence, neutral=self.config.neutral_probability)
        status = self._family_status(confidence)
        # Active weight is not a hard gate; it is the amount of non-neutral signal left.
        active_weight = confidence

        return FamilySoftGateResult(
            family=family,
            status=status,
            confidence=confidence,
            raw_probability=raw_p,
            effective_probability=eff_p,
            active_weight=active_weight,
            shrink_method="neutral_shrinkage",
            head_results=head_results,
            warnings=warnings,
        )

    def evaluate_all(
        self,
        ensemble_results: Mapping[str, HorizonEnsembleResult],
        head_gates: Mapping[str, Mapping[str, object]],
        metrics: Mapping[str, Mapping[str, object]],
    ) -> Dict[str, FamilySoftGateResult]:
        return {
            family: self.evaluate_family(family, er, head_gates=head_gates, metrics=metrics)
            for family, er in ensemble_results.items()
        }
