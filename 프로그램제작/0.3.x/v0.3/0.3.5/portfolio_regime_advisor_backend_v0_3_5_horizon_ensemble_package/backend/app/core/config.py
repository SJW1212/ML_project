from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    app_name: str = "Portfolio Regime Advisor Backend"
    app_version: str = "0.3.5"
    default_model_version: str = "v8.6.41_label_fixed"
    default_model_mode: str = "prediction_file"
    default_assets: List[str] = Field(default_factory=lambda: ["QQQ", "SPY", "AAPL", "SOXX", "NVDA"])
    default_horizon: str = "10D"
    default_data_provider: str = "auto"
    input_dir: Path = Path("storage/predictions")
    storage_dir: Path = Path("storage")
    model_dir: Path = Path("storage/models")
    cache_dir: Path = Path("storage/cache")
    registry_dir: Path = Path("storage/registry")
    secrets_dir: Path = Path("storage/secrets")
    cors_allow_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"])

    # KIS base URLs are configurable because KIS can change endpoints or separate mock/real hosts.
    kis_real_base_url: str = "https://openapi.koreainvestment.com:9443"
    kis_mock_base_url: str = "https://openapivts.koreainvestment.com:29443"

    @classmethod
    def from_env(cls) -> "AppSettings":
        import os

        data = {}
        mapping = {
            "PRA_INPUT_DIR": ("input_dir", Path),
            "PRA_STORAGE_DIR": ("storage_dir", Path),
            "PRA_MODEL_DIR": ("model_dir", Path),
            "PRA_CACHE_DIR": ("cache_dir", Path),
            "PRA_REGISTRY_DIR": ("registry_dir", Path),
            "PRA_SECRETS_DIR": ("secrets_dir", Path),
            "PRA_DEFAULT_HORIZON": ("default_horizon", str),
            "PRA_KIS_REAL_BASE_URL": ("kis_real_base_url", str),
            "PRA_KIS_MOCK_BASE_URL": ("kis_mock_base_url", str),
        }
        env_present = {}
        for env_key, (field_name, caster) in mapping.items():
            value = os.getenv(env_key)
            if value:
                data[field_name] = caster(value)
                env_present[field_name] = True
        settings = cls(**data)
        storage_overridden = env_present.get("storage_dir", False)
        if storage_overridden:
            if not env_present.get("cache_dir", False):
                settings.cache_dir = settings.storage_dir / "cache"
            if not env_present.get("registry_dir", False):
                settings.registry_dir = settings.storage_dir / "registry"
            if not env_present.get("secrets_dir", False):
                settings.secrets_dir = settings.storage_dir / "secrets"
            if not env_present.get("model_dir", False):
                settings.model_dir = settings.storage_dir / "models"
        return settings

    def ensure_dirs(self) -> None:
        for path in [self.storage_dir, self.cache_dir, self.registry_dir, self.secrets_dir, self.model_dir]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings.from_env()
    settings.ensure_dirs()
    return settings
