from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    app_name: str = "Portfolio Regime Advisor Backend"
    app_version: str = "0.2.0"
    default_model_version: str = "v8.6.41_label_fixed"
    default_model_mode: str = "prediction_file"
    default_assets: List[str] = Field(default_factory=lambda: ["QQQ", "SPY", "AAPL", "SOXX", "NVDA"])
    default_horizon: str = "10D"
    default_data_provider: str = "auto"
    input_dir: Path = Path("/mnt/data")
    storage_dir: Path = Path("storage")
    model_dir: Path = Path("models")
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
            "PRA_DEFAULT_HORIZON": ("default_horizon", str),
            "PRA_KIS_REAL_BASE_URL": ("kis_real_base_url", str),
            "PRA_KIS_MOCK_BASE_URL": ("kis_mock_base_url", str),
        }
        for env_key, (field_name, caster) in mapping.items():
            value = os.getenv(env_key)
            if value:
                data[field_name] = caster(value)
        settings = cls(**data)
        # derive storage subdirs if storage_dir was overridden and subdir env not set
        settings.cache_dir = settings.storage_dir / "cache"
        settings.registry_dir = settings.storage_dir / "registry"
        settings.secrets_dir = settings.storage_dir / "secrets"
        return settings

    def ensure_dirs(self) -> None:
        for path in [self.storage_dir, self.cache_dir, self.registry_dir, self.secrets_dir, self.model_dir]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings.from_env()
    settings.ensure_dirs()
    return settings
