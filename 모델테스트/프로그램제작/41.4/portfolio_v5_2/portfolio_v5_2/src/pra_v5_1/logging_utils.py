from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ensure_dir, utc_now_iso


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)

    def write(self, event_type: str, level: str = "INFO", **context: Any) -> None:
        record = {
            "@timestamp": utc_now_iso(),
            "level": level.upper(),
            "event_type": event_type,
            **context,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def setup_python_logging(log_file: Path) -> None:
    ensure_dir(log_file.parent)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
