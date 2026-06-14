"""CLI entry point.

Default behavior writes JSON only. Use --export-csv only for debugging or handoff.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .config import ProductionConfig
from .constants import DEFAULT_ASSETS
from .service import ProductionService


def parse_assets(value: str) -> List[str]:
    return [x.strip().upper() for x in value.split(",") if x.strip()]


def parse_custom_weights(value: str) -> Dict[str, float]:
    if not value:
        return {}
    result: Dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Invalid custom weight item: {item}")
        ticker, raw = item.split(":", 1)
        result[ticker.strip().upper()] = float(raw)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v8.6.41 label_fixed UI-ready modular production pipeline")
    p.add_argument("--input-dir", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("v8_6_41_ui_modular_ops"))
    p.add_argument("--assets", type=str, default=DEFAULT_ASSETS)
    p.add_argument("--holdout-start", type=str, default="2024-01-01")
    p.add_argument("--allocation-source", choices=["executed", "signal"], default="executed")
    p.add_argument("--capital-mode", choices=["equal", "custom", "inverse_vol"], default="equal")
    p.add_argument("--custom-capital-weights", type=str, default="")
    p.add_argument("--no-json", action="store_true", help="Do not write dashboard_payload.json")
    p.add_argument("--export-csv", action="store_true", help="Optional debug CSV export. Not needed for UI.")
    p.add_argument("--export-markdown", action="store_true", help="Optional markdown report export.")
    p.add_argument("--no-zip", action="store_true", help="Do not zip output folder.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = ProductionConfig(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        assets=parse_assets(args.assets),
        holdout_start=args.holdout_start,
        allocation_source=args.allocation_source,
        capital_mode=args.capital_mode,
        custom_capital_weights=parse_custom_weights(args.custom_capital_weights),
        export_json=not args.no_json,
        export_csv=args.export_csv,
        export_markdown=args.export_markdown,
        make_zip=not args.no_zip,
    )
    payload = ProductionService(config).run_and_write()
    totals = payload["portfolio_totals"]
    validation = payload["validation"]
    print("[DONE] v8.6.41 UI-ready modular payload generated")
    print(f"as_of_date={payload['as_of_date']}")
    print(f"stock={totals['portfolio_stock_weight']:.4f}, bond={totals['portfolio_bond_weight']:.4f}, cash={totals['portfolio_cash_weight']:.4f}")
    print(f"validation_fail_count={validation['fail_count']}")
    print(f"out_dir={config.out_dir}")


if __name__ == "__main__":
    main()
