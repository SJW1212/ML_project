from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import math
import numpy as np


@dataclass(frozen=True)
class HorizonFamilyConfig:
    """Configuration for combining horizon-specific probabilities.

    The ensemble is deliberately gate-aware. A weak 20D model must not pull down
    a strong 5D/10D signal unless its own head-level validation is at least
    UNCERTAIN/PASS.
    """

    family: str
    horizons: List[str]
    base_weights: Dict[str, float]
    mode: str = "weighted_average"  # weighted_average | logit_average
    fail_policy: str = "drop"        # drop | neutral
    neutral_probability: float = 0.50


@dataclass(frozen=True)
class HorizonEnsembleResult:
    family: str
    probability: float
    state: str
    term_slope_5_20: Optional[float]
    persistence_10_20: Optional[float]
    used_heads: List[str] = field(default_factory=list)
    uncertain_heads: List[str] = field(default_factory=list)
    fallback_heads: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    raw_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "probability": self.probability,
            "state": self.state,
            "term_slope_5_20": self.term_slope_5_20,
            "persistence_10_20": self.persistence_10_20,
            "used_heads": self.used_heads,
            "uncertain_heads": self.uncertain_heads,
            "fallback_heads": self.fallback_heads,
            "weights": self.weights,
            "raw_probabilities": self.raw_probabilities,
        }


class HorizonEnsemble:
    """Gate-aware horizon ensemble for highvol/up_strength/down_strength heads.

    Families are intentionally configured differently:
    - highvol: 5D/10D/20D, 10D-centered with 20D persistence contribution.
    - up/down strength: 5D/10D are primary; 20D is only a light persistence head.

    This class does not train models. It combines already predicted probabilities
    after head-level gate adjustment.
    """

    DEFAULT_CONFIGS: Dict[str, HorizonFamilyConfig] = {
        "highvol": HorizonFamilyConfig(
            family="highvol",
            horizons=["5D", "10D", "20D"],
            base_weights={"5D": 0.35, "10D": 0.45, "20D": 0.20},
        ),
        "up_strength": HorizonFamilyConfig(
            family="up_strength",
            horizons=["5D", "10D", "20D"],
            # 20D is deliberately light; it often fails worst-fold stability.
            base_weights={"5D": 0.50, "10D": 0.40, "20D": 0.10},
        ),
        "down_strength": HorizonFamilyConfig(
            family="down_strength",
            horizons=["5D", "10D", "20D"],
            # Down-risk is regime-dependent. Keep 20D small and gate-aware.
            base_weights={"5D": 0.45, "10D": 0.45, "20D": 0.10},
        ),
    }

    GATE_MULTIPLIERS = {
        "PASS": 1.00,
        "UNCERTAIN": 0.50,
        "FAIL": 0.00,
    }

    def __init__(self, configs: Optional[Mapping[str, HorizonFamilyConfig]] = None):
        self.configs = dict(configs or self.DEFAULT_CONFIGS)

    @staticmethod
    def _clip01(value: float, default: float = 0.50) -> float:
        try:
            value = float(value)
            if not np.isfinite(value):
                return default
            return float(np.clip(value, 0.0, 1.0))
        except Exception:
            return default

    @staticmethod
    def _head_name(family: str, horizon: str) -> str:
        return f"{family}_{horizon.upper()}"

    @staticmethod
    def _logit(p: float, eps: float = 1e-6) -> float:
        p = min(max(float(p), eps), 1.0 - eps)
        return math.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _gate_status(head_gate_payload: Mapping[str, object] | None) -> str:
        if not head_gate_payload:
            return "FAIL"
        status = head_gate_payload.get("status")
        if hasattr(status, "value"):
            status = status.value
        status = str(status or "FAIL").upper()
        if status not in {"PASS", "UNCERTAIN", "FAIL"}:
            return "FAIL"
        return status

    def combine_family(
        self,
        family: str,
        probabilities: Mapping[str, float],
        head_gates: Mapping[str, Mapping[str, object]],
    ) -> HorizonEnsembleResult:
        family = family.lower()
        if family not in self.configs:
            raise ValueError(f"Unknown ensemble family: {family}")
        cfg = self.configs[family]

        raw_probs: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        used_heads: List[str] = []
        uncertain_heads: List[str] = []
        fallback_heads: List[str] = []

        for horizon in cfg.horizons:
            h = horizon.upper()
            head = self._head_name(family, h)
            p = self._clip01(probabilities.get(head, cfg.neutral_probability), cfg.neutral_probability)
            raw_probs[h] = p
            status = self._gate_status(head_gates.get(head))
            multiplier = self.GATE_MULTIPLIERS.get(status, 0.0)

            if status == "PASS":
                used_heads.append(head)
            elif status == "UNCERTAIN":
                used_heads.append(head)
                uncertain_heads.append(head)
            else:
                fallback_heads.append(head)

            if status == "FAIL" and cfg.fail_policy == "neutral":
                # Include a weak neutral vote only if explicitly requested.
                p = cfg.neutral_probability
                multiplier = 0.25
            weights[h] = float(cfg.base_weights.get(h, 0.0) * multiplier)

        total_weight = float(sum(weights.values()))
        if total_weight <= 0:
            probability = cfg.neutral_probability
        elif cfg.mode == "logit_average":
            z = sum(weights[h] * self._logit(raw_probs[h]) for h in weights) / total_weight
            probability = self._clip01(self._sigmoid(z), cfg.neutral_probability)
        else:
            probability = self._clip01(sum(weights[h] * raw_probs[h] for h in weights) / total_weight, cfg.neutral_probability)

        normalized_weights = {h: (w / total_weight if total_weight > 0 else 0.0) for h, w in weights.items()}
        p5 = raw_probs.get("5D")
        p10 = raw_probs.get("10D")
        p20 = raw_probs.get("20D")
        state = self.classify_state(family, p5, p10, p20, probability, used_heads)
        term_slope = None if p5 is None or p20 is None else float(p5 - p20)
        persistence = None if p10 is None or p20 is None else float(min(p10, p20))

        return HorizonEnsembleResult(
            family=family,
            probability=probability,
            state=state,
            term_slope_5_20=term_slope,
            persistence_10_20=persistence,
            used_heads=used_heads,
            uncertain_heads=uncertain_heads,
            fallback_heads=fallback_heads,
            weights=normalized_weights,
            raw_probabilities=raw_probs,
        )

    def combine_all(
        self,
        probabilities: Mapping[str, float],
        head_gates: Mapping[str, Mapping[str, object]],
        families: Optional[Iterable[str]] = None,
    ) -> Dict[str, HorizonEnsembleResult]:
        families = list(families or ["highvol", "up_strength", "down_strength"])
        return {family: self.combine_family(family, probabilities, head_gates) for family in families}

    @staticmethod
    def classify_state(
        family: str,
        p5: Optional[float],
        p10: Optional[float],
        p20: Optional[float],
        ensemble: float,
        used_heads: List[str],
    ) -> str:
        p5 = 0.50 if p5 is None else float(p5)
        p10 = 0.50 if p10 is None else float(p10)
        p20 = 0.50 if p20 is None else float(p20)
        used_count = len(used_heads)

        if used_count == 0:
            return f"{family.upper()}_NEUTRAL_NO_PASSING_HEAD"

        if family == "highvol":
            if ensemble < 0.35 and p5 < 0.40 and p10 < 0.40:
                return "NORMAL_VOL"
            if p5 >= 0.60 and p10 < 0.50 and p20 < 0.50:
                return "ACUTE_SPIKE"
            if p5 >= 0.55 and p10 >= 0.50 and p20 < 0.50:
                return "RISING_VOL"
            if p5 >= 0.50 and p10 >= 0.55 and p20 >= 0.45:
                return "CONFIRMED_HIGH_VOL"
            if p10 >= 0.55 and p20 >= 0.55:
                return "PERSISTENT_HIGH_VOL"
            if p5 < 0.40 and p10 >= 0.50 and p20 >= 0.50:
                return "VOL_COMPRESSION"
            return "MIXED_VOL"

        if family == "up_strength":
            if ensemble >= 0.60 and p5 >= p10:
                return "ACUTE_UP_STRENGTH"
            if ensemble >= 0.58 and p10 >= 0.55:
                return "CONFIRMED_UP_STRENGTH"
            if ensemble >= 0.52:
                return "MILD_UP_STRENGTH"
            if p5 < 0.45 and p10 < 0.45:
                return "NO_UP_STRENGTH"
            return "MIXED_UP_STRENGTH"

        if family == "down_strength":
            if ensemble >= 0.62 and p5 >= 0.55 and p10 >= 0.55:
                return "CONFIRMED_DOWN_STRENGTH"
            if p5 >= 0.60 and p10 < 0.50:
                return "ACUTE_DOWN_SPIKE"
            if ensemble >= 0.55:
                return "MILD_DOWN_STRENGTH"
            if p5 < 0.45 and p10 < 0.45:
                return "NO_DOWN_STRENGTH"
            return "MIXED_DOWN_STRENGTH"

        return "MIXED"
