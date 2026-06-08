from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from cryptography.fernet import Fernet


class CredentialManager:
    """Encrypted local credential storage.

    This is suitable for local development/MVP. For deployed web services, replace with
    cloud Secret Manager or a server-side vault.
    """

    def __init__(self, secrets_dir: Path):
        self.secrets_dir = Path(secrets_dir)
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        self.key_path = self.secrets_dir / "pra_fernet.key"
        self.cred_path = self.secrets_dir / "credentials.enc"
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes()
        key = Fernet.generate_key()
        self.key_path.write_bytes(key)
        try:
            self.key_path.chmod(0o600)
        except Exception:
            pass
        return key

    def _load_all(self) -> Dict[str, dict]:
        if not self.cred_path.exists():
            return {}
        encrypted = self.cred_path.read_bytes()
        if not encrypted:
            return {}
        data = self._fernet.decrypt(encrypted)
        return json.loads(data.decode("utf-8"))

    def _save_all(self, payload: Dict[str, dict]) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.cred_path.write_bytes(self._fernet.encrypt(data))
        try:
            self.cred_path.chmod(0o600)
        except Exception:
            pass

    def save_credentials(self, provider: str, credentials: dict) -> None:
        provider = provider.lower()
        all_credentials = self._load_all()
        all_credentials[provider] = credentials
        self._save_all(all_credentials)

    def load_credentials(self, provider: str) -> Optional[dict]:
        return self._load_all().get(provider.lower())

    def delete_credentials(self, provider: str) -> bool:
        provider = provider.lower()
        all_credentials = self._load_all()
        existed = provider in all_credentials
        all_credentials.pop(provider, None)
        self._save_all(all_credentials)
        return existed

    @staticmethod
    def _mask(value: Optional[str], visible: int = 4) -> Optional[str]:
        if value is None:
            return None
        if len(value) <= visible:
            return "*" * len(value)
        return value[:visible] + "*" * max(4, len(value) - visible)

    def status(self, provider: str) -> dict:
        creds = self.load_credentials(provider)
        if not creds:
            return {"provider": provider.lower(), "exists": False}
        return {
            "provider": provider.lower(),
            "exists": True,
            "environment": creds.get("environment"),
            "app_key": self._mask(creds.get("app_key")),
            "app_secret": "********" if creds.get("app_secret") else None,
            "account_no": self._mask(creds.get("account_no"), visible=4),
            "account_product_code": self._mask(creds.get("account_product_code"), visible=0),
        }
