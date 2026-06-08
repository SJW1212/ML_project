from __future__ import annotations

from enum import Enum
from typing import Dict, Set


class UserMode(str, Enum):
    GENERAL = "GENERAL"
    ADVANCED = "ADVANCED"
    EXPERT = "EXPERT"
    ADMIN = "ADMIN"


class SettingMode(str, Enum):
    PRESET = "PRESET"
    GUIDED = "GUIDED"
    DIRECT = "DIRECT"
    DEVELOPER = "DEVELOPER"


class UserModePolicy:
    """Mode-based capability guard.

    UI can hide features, but backend must still validate capabilities.
    """

    CAPABILITIES: Dict[UserMode, Set[str]] = {
        UserMode.GENERAL: {
            "select_preset",
            "select_assets",
            "select_horizon",
            "view_dashboard",
            "view_basic_performance",
        },
        UserMode.ADVANCED: {
            "select_preset",
            "select_assets",
            "select_horizon",
            "view_dashboard",
            "view_basic_performance",
            "edit_capital_weights",
            "select_benchmark",
            "set_oos_start",
            "scenario_compare",
            "select_data_source",
        },
        UserMode.EXPERT: {
            "select_preset",
            "select_assets",
            "select_horizon",
            "view_dashboard",
            "view_basic_performance",
            "edit_capital_weights",
            "select_benchmark",
            "set_oos_start",
            "scenario_compare",
            "select_data_source",
            "set_train_window",
            "set_train_period",
            "set_recency_half_life",
            "run_retraining",
            "compare_model_versions",
        },
        UserMode.ADMIN: {
            "select_preset",
            "select_assets",
            "select_horizon",
            "view_dashboard",
            "view_basic_performance",
            "edit_capital_weights",
            "select_benchmark",
            "set_oos_start",
            "scenario_compare",
            "select_data_source",
            "set_train_window",
            "set_train_period",
            "set_recency_half_life",
            "run_retraining",
            "compare_model_versions",
            "manage_credentials",
            "manage_providers",
            "manage_registry",
            "view_system_logs",
            "clear_cache",
        },
    }

    @classmethod
    def allowed(cls, mode: UserMode, capability: str) -> bool:
        return capability in cls.CAPABILITIES.get(mode, set())

    @classmethod
    def require(cls, mode: UserMode, capability: str) -> None:
        if not cls.allowed(mode, capability):
            raise PermissionError(f"{mode.value} mode cannot use capability: {capability}")
