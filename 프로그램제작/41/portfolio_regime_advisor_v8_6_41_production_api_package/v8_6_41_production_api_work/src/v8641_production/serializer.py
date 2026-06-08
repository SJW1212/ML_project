"""Output writers.

CSV is intentionally optional. For UI/UX, consume the dashboard payload JSON or
call ProductionService.build_dashboard_payload() directly from a backend route.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd


class OutputWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self.out_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def write_csv_bundle(self, payload: Dict[str, Any]) -> list[Path]:
        """Optional debug export. Not required for UI."""
        files: list[Path] = []
        mapping = {
            "latest_signals.csv": payload.get("latest_signals", []),
            "portfolio_allocation.csv": payload.get("portfolio_allocation", []),
            "performance_summary.csv": payload.get("performance_summary", []),
            "annual_returns.csv": payload.get("charts", {}).get("annual_returns", []),
            "monthly_returns.csv": payload.get("charts", {}).get("monthly_returns", []),
            "validation_checks.csv": payload.get("validation", {}).get("checks", []),
        }
        for filename, data in mapping.items():
            path = self.out_dir / filename
            pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig")
            files.append(path)
        return files

    def write_markdown_report(self, payload: Dict[str, Any]) -> Path:
        totals = payload.get("portfolio_totals", {})
        validation = payload.get("validation", {})
        lines = [
            "# v8.6.41 UI-ready Production Report",
            "",
            f"- model_version: `{payload.get('model_version')}`",
            f"- as_of_date: `{payload.get('as_of_date')}`",
            f"- allocation_source: `{payload.get('allocation_source')}`",
            f"- capital_mode: `{payload.get('capital_mode')}`",
            f"- validation_fail_count: `{validation.get('fail_count')}`",
            "",
            "## Portfolio Totals",
            "",
            f"- stock: {totals.get('portfolio_stock_weight', 0):.2%}",
            f"- bond: {totals.get('portfolio_bond_weight', 0):.2%}",
            f"- cash: {totals.get('portfolio_cash_weight', 0):.2%}",
        ]
        path = self.out_dir / "ui_ready_report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def make_zip(self, zip_name: str = "v8_6_41_ui_modular_outputs.zip") -> Path:
        path = self.out_dir.parent / zip_name
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in self.out_dir.rglob("*"):
                if item.is_file():
                    zf.write(item, arcname=str(item.relative_to(self.out_dir)))
        return path
