from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Preset:
    name: str
    display_name: str
    description: str
    horizon: str
    rebalance_frequency: str
    capital_mode: str
    risk_alert_sensitivity: str
    benchmark: str


class PresetManager:
    PRESETS: Dict[str, Preset] = {
        "stable": Preset(
            name="stable",
            display_name="안정형",
            description="MDD 완화와 변동성 관리 우선",
            horizon="20D",
            rebalance_frequency="monthly",
            capital_mode="inverse_vol",
            risk_alert_sensitivity="high",
            benchmark="60_40",
        ),
        "balanced": Preset(
            name="balanced",
            display_name="균형형",
            description="수익성과 위험 관리 균형",
            horizon="10D",
            rebalance_frequency="monthly",
            capital_mode="equal",
            risk_alert_sensitivity="medium",
            benchmark="60_40",
        ),
        "aggressive": Preset(
            name="aggressive",
            display_name="공격형",
            description="단기 수익 기회 참여 우선",
            horizon="5D",
            rebalance_frequency="weekly",
            capital_mode="custom",
            risk_alert_sensitivity="low",
            benchmark="buy_hold",
        ),
        "etf_core": Preset(
            name="etf_core",
            display_name="ETF 중심",
            description="ETF 중심의 중기 포트폴리오",
            horizon="10D",
            rebalance_frequency="monthly",
            capital_mode="equal",
            risk_alert_sensitivity="medium",
            benchmark="60_40",
        ),
        "single_stock": Preset(
            name="single_stock",
            display_name="개별주 중심",
            description="개별주 포함, 더 잦은 모니터링",
            horizon="5D",
            rebalance_frequency="biweekly",
            capital_mode="custom",
            risk_alert_sensitivity="medium",
            benchmark="buy_hold",
        ),
    }

    @classmethod
    def list_presets(cls) -> List[Preset]:
        return list(cls.PRESETS.values())

    @classmethod
    def get(cls, name: str) -> Preset:
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {name}")
        return cls.PRESETS[name]
