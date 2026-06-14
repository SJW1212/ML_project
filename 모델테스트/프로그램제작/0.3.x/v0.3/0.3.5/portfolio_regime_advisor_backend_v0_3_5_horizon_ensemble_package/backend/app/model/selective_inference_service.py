from __future__ import annotations

from typing import Dict, Optional

from .head_gate import GateStatus, HeadLevelGate, apply_fallback_probability


class SelectiveInferenceService:
    """Adjust per-head live probabilities according to head-level gates.

    Fallback hierarchy:
    - PASS: use live probability.
    - UNCERTAIN: shrink live probability toward 0.5.
    - FAIL: use prediction-file baseline if supplied, otherwise neutral 0.5.

    Down-strength FAIL does not remove defense entirely. If high-vol passes, the
    adjusted signal can still keep WATCH behavior through high-vol probability.
    If all ML heads fail, the caller can use context rule overrides such as
    spy_drawdown_252 < -0.15 to maintain minimum WATCH.
    """

    def __init__(self, gate: Optional[HeadLevelGate] = None):
        self.gate = gate or HeadLevelGate()

    def adjust(
        self,
        *,
        live_probs: Dict[str, float],
        metrics: Dict[str, Dict],
        baseline_probs: Optional[Dict[str, float]] = None,
        context_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        baseline_probs = baseline_probs or {}
        context_features = context_features or {}
        gate_results = {}
        adjusted = {}
        for head, prob in live_probs.items():
            g = self.gate.evaluate(head, metrics.get(head, {}))
            gate_results[head] = g
            adjusted[head] = apply_fallback_probability(prob, g, baseline_probs.get(head))

        warnings = []
        down_gates = [g for h, g in gate_results.items() if h.startswith("down_strength")]
        highvol_gates = [g for h, g in gate_results.items() if h.startswith("highvol")]
        if down_gates and all(g.status == GateStatus.FAIL for g in down_gates):
            if any(g.status == GateStatus.PASS for g in highvol_gates):
                warnings.append("DOWN_STRENGTH_FAIL_HIGHVOL_PASS_DEFENSIVE_WATCH_ALLOWED")
            elif any(g.status == GateStatus.UNCERTAIN for g in highvol_gates):
                warnings.append("DOWN_AND_HIGHVOL_UNCERTAIN_SOFTENED_WATCH")
            else:
                warnings.append("DOWN_AND_HIGHVOL_FAIL_NEUTRAL_HOLD_POSITION")

        spy_dd = context_features.get("ctx_spy_drawdown_252")
        try:
            if spy_dd is not None and float(spy_dd) < -0.15:
                warnings.append("RULE_BASED_WATCH_OVERRIDE_SPY_DRAWDOWN_252")
                adjusted["risk_override"] = "WATCH"
        except Exception:
            pass

        return {
            "adjusted_probs": adjusted,
            "head_gates": {k: v.to_dict() for k, v in gate_results.items()},
            "selective_warnings": warnings,
        }
