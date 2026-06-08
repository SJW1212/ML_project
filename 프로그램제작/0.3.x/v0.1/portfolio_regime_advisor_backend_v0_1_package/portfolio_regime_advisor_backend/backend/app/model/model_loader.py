from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib


class ModelLoader:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)

    def artifact_path(self, model_version: str, ticker: str, head: str, horizon: str) -> Path:
        return self.model_dir / model_version / ticker.upper() / f"{head}_{horizon.lower()}.joblib"

    def exists(self, model_version: str, ticker: str, head: str, horizon: str) -> bool:
        return self.artifact_path(model_version, ticker, head, horizon).exists()

    def load(self, model_version: str, ticker: str, head: str, horizon: str) -> Any:
        path = self.artifact_path(model_version, ticker, head, horizon)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        return joblib.load(path)

    def save(self, model: Any, model_version: str, ticker: str, head: str, horizon: str) -> Path:
        path = self.artifact_path(model_version, ticker, head, horizon)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return path
