from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class ModelArtifactStore:
    """Metadata helper for runtime model artifacts.

    The actual binary load/save is handled by ModelLoader. This class stores human-readable
    artifact manifests beside model files so UI/Admin screens can audit what exists.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def ticker_dir(self, model_version: str, ticker: str) -> Path:
        path = self.model_dir / model_version / ticker.upper()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_manifest(self, model_version: str, ticker: str, manifest: Dict[str, Any]) -> Path:
        path = self.ticker_dir(model_version, ticker) / "artifact_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_manifest(self, model_version: str, ticker: str) -> Dict[str, Any]:
        path = self.ticker_dir(model_version, ticker) / "artifact_manifest.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def build_manifest(self, model_version: str, ticker: str, horizons: Iterable[str], feature_columns: List[str], metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model_version": model_version,
            "ticker": ticker.upper(),
            "horizons": list(horizons),
            "feature_columns": feature_columns,
            "metrics": metrics,
        }
