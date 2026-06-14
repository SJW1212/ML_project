from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schemas import DailyUpdateRequest, PortfolioEvaluateRequest, PredictionGenerateRequest
from .service import PortfolioRegimeAdvisorService
from .utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio Regime Advisor v5.1 local CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--request", required=True)
    p_eval.add_argument("--out", default="storage/latest_dashboard_payload.json")
    p_up = sub.add_parser("update-daily")
    p_up.add_argument("--tickers", default="")
    p_up.add_argument("--start", default="2013-01-01")
    p_up.add_argument("--force", action="store_true")
    p_gen = sub.add_parser("generate-predictions")
    p_gen.add_argument("--tickers", default="")
    p_gen.add_argument("--force", action="store_true")
    args = parser.parse_args()
    service = PortfolioRegimeAdvisorService()
    if args.cmd == "evaluate":
        with open(args.request, "r", encoding="utf-8") as f:
            req = PortfolioEvaluateRequest.model_validate(json.load(f))
        payload = service.evaluate(req)
        atomic_write_json(Path(args.out), payload)
        print(json.dumps({"ok": payload.get("ok"), "out": args.out, "as_of_date": payload.get("as_of_date")}, ensure_ascii=False, indent=2))
    elif args.cmd == "update-daily":
        tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]
        payload = service.update_daily(DailyUpdateRequest(tickers=tickers, start=args.start, force=args.force))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.cmd == "generate-predictions":
        tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]
        payload = service.generate_predictions(PredictionGenerateRequest(tickers=tickers, force=args.force))
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
