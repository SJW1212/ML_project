from __future__ import annotations

from typing import Dict

from .utils import safe_float


class SignalClassifier:
    def classify(self, row: Dict) -> Dict:
        phv = safe_float(row.get("prob_high_vol"), 0.5)
        prisk = safe_float(row.get("prob_overall_risk"), 0.5)
        pup = safe_float(row.get("prob_up_strengthening_score"), 0.5)
        pdown = safe_float(row.get("prob_down_strengthening_score"), 0.5)
        if prisk >= 0.65 or phv >= 0.65:
            risk_class = "HIGH_RISK"
        elif prisk >= 0.50 or phv >= 0.40 or pdown >= 0.50:
            risk_class = "WATCH"
        else:
            risk_class = "NORMAL"
        if pdown >= 0.55 and pdown > pup:
            direction_class = "DOWN_STRENGTH"
        elif pup >= 0.55 and pup > pdown:
            direction_class = "UP_STRENGTH"
        else:
            direction_class = "NEUTRAL"
        if risk_class == "HIGH_RISK":
            allocation_class = "DEFENSIVE"
        elif direction_class == "UP_STRENGTH" and risk_class == "NORMAL":
            allocation_class = "PARTICIPATION"
        else:
            allocation_class = "BALANCED"
        return {"risk_class": risk_class, "direction_class": direction_class, "allocation_class": allocation_class}
