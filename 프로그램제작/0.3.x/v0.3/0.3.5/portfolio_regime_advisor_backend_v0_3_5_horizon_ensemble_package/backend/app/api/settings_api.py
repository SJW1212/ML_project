from __future__ import annotations

from fastapi import APIRouter

from ..core.presets import PresetManager
from ..dependencies import get_parameter_validator
from ..schemas import SettingsRequest

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/presets")
def list_presets():
    return {"presets": [p.__dict__ for p in PresetManager.list_presets()]}


@router.post("/apply-preset")
def apply_preset(name: str):
    preset = PresetManager.get(name)
    return {"preset": preset.__dict__}


@router.post("/validate")
def validate_settings(req: SettingsRequest):
    result = get_parameter_validator().validate_settings(
        user_mode=req.user_mode,
        horizon=req.horizon,
        assets=req.assets,
        capital_mode=req.capital_mode,
        custom_weights=req.custom_weights,
    )
    return {"ok": result.ok, "fail_count": result.fail_count, "warning_count": result.warning_count, "messages": result.messages}
