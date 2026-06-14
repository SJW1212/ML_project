from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib


class ModelLoader:
    """Load/save runtime model artifacts.

    Artifact layout:
      models/{model_version}/{TICKER}/{head}_{horizon}.joblib
    """

    HEADS = ["highvol", "up_strength", "down_strength"]

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

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

    def missing_artifacts(self, model_version: str, ticker: str, horizons: Iterable[str]) -> List[str]:
        missing: List[str] = []
        for horizon in horizons:
            for head in self.HEADS:
                path = self.artifact_path(model_version, ticker, head, horizon)
                if not path.exists():
                    missing.append(str(path))
        return missing

    def artifact_status(self, model_version: str, tickers: Iterable[str], horizons: Iterable[str]) -> Dict:
        horizons = list(horizons)
        items = []
        total_required = 0
        total_exists = 0
        for ticker in tickers:
            ticker = ticker.upper()
            missing = self.missing_artifacts(model_version, ticker, horizons)
            required = len(horizons) * len(self.HEADS)
            exists_count = required - len(missing)
            total_required += required
            total_exists += exists_count
            items.append({
                "ticker": ticker,
                "required": required,
                "exists": exists_count,
                "complete": len(missing) == 0,
                "missing": missing,
            })
        return {
            "model_version": model_version,
            "required": total_required,
            "exists": total_exists,
            "complete": total_required > 0 and total_exists == total_required,
            "items": items,
        }

    def inventory(self) -> Dict:
        versions = []
        if not self.model_dir.exists():
            return {"model_dir": str(self.model_dir), "versions": []}
        for version_dir in sorted([p for p in self.model_dir.iterdir() if p.is_dir()]):
            tickers = []
            for ticker_dir in sorted([p for p in version_dir.iterdir() if p.is_dir()]):
                artifacts = sorted([p.name for p in ticker_dir.glob("*.joblib")])
                tickers.append({"ticker": ticker_dir.name, "artifact_count": len(artifacts), "artifacts": artifacts})
            versions.append({"model_version": version_dir.name, "tickers": tickers})
        return {"model_dir": str(self.model_dir), "versions": versions}
