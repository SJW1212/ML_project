from __future__ import annotations

from typing import Dict

import pandas as pd

from .allocation_service import AllocationService


class ScenarioComparator:
    def __init__(self, allocation_service: AllocationService):
        self.allocation_service = allocation_service

    def compare(self, signals: pd.DataFrame, base_mode: str, next_mode: str, base_weights=None, next_weights=None) -> dict:
        _, base = self.allocation_service.apply(signals, base_mode, base_weights)
        _, nxt = self.allocation_service.apply(signals, next_mode, next_weights)
        return {
            "base": base,
            "next": nxt,
            "delta": {k: nxt[k] - base[k] for k in base.keys()},
        }
