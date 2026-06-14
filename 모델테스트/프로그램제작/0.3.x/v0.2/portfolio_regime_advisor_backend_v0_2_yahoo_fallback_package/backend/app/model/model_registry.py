from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelRegistry:
    def __init__(self, registry_dir: Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.registry_dir / "model_registry.json"
        if not self.path.exists():
            self._save({"models": [], "active_model_version": "v8.6.41_label_fixed", "active_mode": "prediction_file"})

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def active(self) -> Dict[str, Any]:
        data = self._load()
        version = data.get("active_model_version", "v8.6.41_label_fixed")
        model = next((m for m in data.get("models", []) if m.get("model_version") == version and m.get("status") == "ACTIVE"), None)
        return {
            "active_model_version": version,
            "active_mode": data.get("active_mode", "prediction_file"),
            "model": model,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return self._load().get("models", [])

    def register(self, metadata: Dict[str, Any], status: str = "CANDIDATE") -> None:
        data = self._load()
        metadata = dict(metadata)
        metadata["status"] = status
        models = [m for m in data.get("models", []) if m.get("model_id") != metadata.get("model_id")]
        models.append(metadata)
        data["models"] = models
        self._save(data)

    def activate(self, model_version: str, mode: str = "live_inference") -> None:
        data = self._load()
        found = False
        for m in data.get("models", []):
            if m.get("model_version") == model_version:
                m["status"] = "ACTIVE"
                found = True
            elif m.get("status") == "ACTIVE":
                m["status"] = "ARCHIVED"
        if not found and model_version != "v8.6.41_label_fixed":
            raise ValueError(f"Model version not found: {model_version}")
        data["active_model_version"] = model_version
        data["active_mode"] = mode
        self._save(data)
