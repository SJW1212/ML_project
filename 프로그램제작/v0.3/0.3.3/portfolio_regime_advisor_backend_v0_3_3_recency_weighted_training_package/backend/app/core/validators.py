from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .user_modes import UserMode, UserModePolicy


SUPPORTED_HORIZONS = {"5D", "10D", "20D"}
EXPERIMENTAL_HORIZONS = {"1D", "3D", "30D", "60D"}
CAPITAL_MODES = {"equal", "custom", "inverse_vol"}


@dataclass
class ValidationResult:
    ok: bool = True
    messages: List[Dict[str, str]] = field(default_factory=list)

    @property
    def fail_count(self) -> int:
        return sum(1 for m in self.messages if m.get("level") == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for m in self.messages if m.get("level") == "WARN")

    def add(self, level: str, code: str, message: str, field: Optional[str] = None) -> None:
        self.messages.append({"level": level, "code": code, "message": message, "field": field or ""})
        if level == "ERROR":
            self.ok = False


class ParameterValidator:
    def validate_settings(
        self,
        *,
        user_mode: UserMode,
        horizon: str,
        assets: Iterable[str],
        capital_mode: str,
        custom_weights: Optional[Dict[str, float]] = None,
        require_supported_horizon: bool = True,
    ) -> ValidationResult:
        result = ValidationResult()
        h = horizon.upper().strip()
        assets_list = [a.upper().strip() for a in assets]

        if not assets_list:
            result.add("ERROR", "NO_ASSETS", "최소 1개 이상의 종목이 필요합니다.", "assets")

        if h not in SUPPORTED_HORIZONS:
            if h in EXPERIMENTAL_HORIZONS and user_mode in {UserMode.EXPERT, UserMode.ADMIN}:
                result.add("WARN", "EXPERIMENTAL_HORIZON", f"{h}는 실험 모드입니다. 새 라벨/재학습이 필요할 수 있습니다.", "horizon")
            elif require_supported_horizon:
                result.add("ERROR", "UNSUPPORTED_HORIZON", f"지원 horizon은 {sorted(SUPPORTED_HORIZONS)}입니다.", "horizon")
            else:
                result.add("WARN", "UNSUPPORTED_HORIZON", f"{h}는 기본 지원 horizon이 아닙니다.", "horizon")

        if capital_mode not in CAPITAL_MODES:
            result.add("ERROR", "BAD_CAPITAL_MODE", f"capital_mode는 {sorted(CAPITAL_MODES)} 중 하나여야 합니다.", "capital_mode")

        if capital_mode == "custom":
            if not UserModePolicy.allowed(user_mode, "edit_capital_weights"):
                result.add("ERROR", "MODE_PERMISSION", "현재 사용자 모드에서는 사용자 지정 비중을 사용할 수 없습니다.", "user_mode")
            if not custom_weights:
                result.add("ERROR", "MISSING_CUSTOM_WEIGHTS", "custom mode에는 custom_weights가 필요합니다.", "custom_weights")
            else:
                weight_assets = {k.upper().strip() for k in custom_weights.keys()}
                missing = set(assets_list) - weight_assets
                if missing:
                    result.add("ERROR", "CUSTOM_WEIGHT_MISSING_TICKER", f"비중이 없는 종목: {sorted(missing)}", "custom_weights")
                total = sum(float(v) for v in custom_weights.values())
                if abs(total - 1.0) > 0.005:
                    result.add("ERROR", "CUSTOM_WEIGHT_SUM", f"종목 비중 합계는 1.0이어야 합니다. 현재 {total:.4f}", "custom_weights")
                for ticker, weight in custom_weights.items():
                    if weight < 0:
                        result.add("ERROR", "NEGATIVE_WEIGHT", f"{ticker} 비중은 음수일 수 없습니다.", "custom_weights")
                    if weight > 0.6:
                        result.add("WARN", "CONCENTRATED_WEIGHT", f"{ticker} 비중이 60%를 초과합니다.", "custom_weights")

        return result

    def validate_training_config(
        self,
        *,
        user_mode: UserMode,
        horizons: Iterable[str],
        train_rows_by_horizon: Optional[Dict[str, int]] = None,
    ) -> ValidationResult:
        result = ValidationResult()
        if not UserModePolicy.allowed(user_mode, "run_retraining"):
            result.add("ERROR", "MODE_PERMISSION", "재학습은 전문가 모드 이상에서만 가능합니다.", "user_mode")
        minimums = {"5D": 750, "10D": 1000, "20D": 1260}
        for h in horizons:
            h = h.upper()
            if h not in SUPPORTED_HORIZONS:
                result.add("WARN", "EXPERIMENTAL_HORIZON", f"{h}는 기본 모델 horizon이 아닙니다.", "horizons")
            if train_rows_by_horizon and h in minimums:
                rows = train_rows_by_horizon.get(h, 0)
                if rows < minimums[h]:
                    result.add("ERROR", "INSUFFICIENT_TRAIN_ROWS", f"{h} 학습에는 최소 {minimums[h]} rows가 권장됩니다. 현재 {rows}", "train_period")
        return result
