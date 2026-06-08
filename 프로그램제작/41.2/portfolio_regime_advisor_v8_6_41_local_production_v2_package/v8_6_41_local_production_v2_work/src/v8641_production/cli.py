"""CLI entry point.

Default behavior writes JSON only. Use --export-csv only for debugging or handoff.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from .config import ProductionConfig
from .data_update import DailyMarketDataUpdater
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
    p.add_argument("--cache-dir", type=Path, default=Path("storage/market_cache"), help="Local OHLCV cache directory for daily data updates.")
    p.add_argument("--update-provider", choices=["yahoo", "kis"], default="yahoo", help="Daily cache update provider. KIS adapter is reserved; no order flow is implemented.")
    p.add_argument("--daily-update-hour-kst", type=int, default=8, help="Recommended local daily update hour in KST.")
    p.add_argument("--daily-freshness-tolerance-days", type=int, default=2, help="Daily cache freshness tolerance.")
    p.add_argument("--default-update-start", type=str, default="2013-01-01", help="Default start date for OHLCV cache update.")
    p.add_argument("--update-cache-only", action="store_true", help="Run one daily OHLCV cache update and exit. No realtime streaming and no orders.")
    p.add_argument("--update-force", action="store_true", help="Force cache update even if cache is fresh.")
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
        cache_dir=args.cache_dir,
        update_provider=args.update_provider,
        daily_update_hour_kst=args.daily_update_hour_kst,
        daily_freshness_tolerance_days=args.daily_freshness_tolerance_days,
        default_update_start=args.default_update_start,
    )
    if args.update_cache_only:
        status = DailyMarketDataUpdater(config).update_daily(
            config.assets,
            provider=config.update_provider,
            start=config.default_update_start,
            end=None,
            force=args.update_force,
        )
        print("[DONE] daily cache update attempted")
        print(f"ok={status.ok}, updated={status.updated}, skipped={status.skipped}, errors={status.errors}")
        return

    payload = ProductionService(config).run_and_write()
    totals = payload["portfolio_totals"]
    validation = payload["validation"]
    print("[DONE] v8.6.41 local production payload generated")
    print(f"as_of_date={payload['as_of_date']}")
    print(f"stock={totals['portfolio_stock_weight']:.4f}, bond={totals['portfolio_bond_weight']:.4f}, cash={totals['portfolio_cash_weight']:.4f}")
    print(f"validation_fail_count={validation['fail_count']}")
    print(f"validation_warn_count={validation.get('warn_count', 0)}")
    print(f"out_dir={config.out_dir}")


if __name__ == "__main__":
    main()
