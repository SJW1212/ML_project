from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


class TokenStore:
    """Stores token metadata. Token itself is stored locally for MVP; do not log it."""

    def __init__(self, secrets_dir: Path):
        self.path = Path(secrets_dir) / "tokens.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_all(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_all(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            self.path.chmod(0o600)
        except Exception:
            pass

    def save_token(self, provider: str, environment: str, access_token: str, expires_at: str) -> None:
        data = self._load_all()
        key = f"{provider.lower()}:{environment.lower()}"
        data[key] = {"access_token": access_token, "expires_at": expires_at}
        self._save_all(data)

    def get_token(self, provider: str, environment: str) -> Optional[dict]:
        return self._load_all().get(f"{provider.lower()}:{environment.lower()}")

    def is_valid(self, provider: str, environment: str, buffer_minutes: int = 10) -> bool:
        token = self.get_token(provider, environment)
        if not token:
            return False
        try:
            expires = pd.to_datetime(token.get("expires_at"), utc=True)
            return pd.Timestamp.utcnow() + pd.Timedelta(minutes=buffer_minutes) < expires
        except Exception:
            return False

    def clear(self, provider: str, environment: str) -> bool:
        data = self._load_all()
        key = f"{provider.lower()}:{environment.lower()}"
        existed = key in data
        data.pop(key, None)
        self._save_all(data)
        return existed
