from __future__ import annotations

from fastapi import APIRouter

from ..core.config import get_settings
from ..dependencies import get_prediction_repository

router = APIRouter(prefix="/validation", tags=["validation"])


@router.get("")
def validation(assets: str = "QQQ,SPY,AAPL,SOXX,NVDA"):
    repo = get_prediction_repository()
    messages = []
    fail_count = 0
    warning_count = 0
    for ticker in [a.strip().upper() for a in assets.split(",") if a.strip()]:
        try:
            path = repo.find_prediction_file(ticker)
            df = repo.load_predictions(ticker)
            required = ["Date", "prob_high_vol", "prob_up_strengthening_score", "prob_down_strengthening_score", "stock_weight", "bond_weight", "cash_weight"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                fail_count += 1
                messages.append({"level": "ERROR", "code": "MISSING_COLUMNS", "ticker": ticker, "message": f"Missing {missing}"})
            else:
                messages.append({"level": "INFO", "code": "OK", "ticker": ticker, "message": f"Loaded {len(df)} rows from {path.name}"})
        except Exception as exc:
            warning_count += 1
            messages.append({"level": "WARN", "code": "LOAD_FAILED", "ticker": ticker, "message": str(exc)})
    return {"ok": fail_count == 0, "fail_count": fail_count, "warning_count": warning_count, "messages": messages, "input_dir": str(get_settings().input_dir)}
