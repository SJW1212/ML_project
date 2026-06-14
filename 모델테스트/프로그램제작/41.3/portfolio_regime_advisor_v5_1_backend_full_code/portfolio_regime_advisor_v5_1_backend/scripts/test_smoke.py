from pathlib import Path
import shutil
import tempfile

import pandas as pd

from pra_v5_1.cache import MarketDataCache
from pra_v5_1.config import AppConfig
from pra_v5_1.schemas import PortfolioEvaluateRequest
from pra_v5_1.service import PortfolioRegimeAdvisorService


def make_ohlcv(n=800):
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = 100 * (1 + pd.Series(range(n)) * 0.0005)
    return pd.DataFrame({"Date": dates, "Open": price, "High": price*1.01, "Low": price*0.99, "Close": price, "Volume": 1000000})


def main():
    root = Path(tempfile.mkdtemp()) / "storage"
    cfg = AppConfig(storage_root=root)
    svc = PortfolioRegimeAdvisorService(cfg)
    for ticker in ["AAPL", "QQQ", "NVDA", "LLY"]:
        svc.cache.write_ohlcv(ticker, make_ohlcv(), provider="yahoo")
    req = PortfolioEvaluateRequest.model_validate({
        "portfolio": [
            {"name": "애플", "ticker": "AAPL", "asset_type": "stock", "current_weight": 0.25},
            {"name": "QQQ", "ticker": "QQQ", "asset_type": "etf", "current_weight": 0.25},
            {"name": "엔비디아", "ticker": "NVDA", "asset_type": "stock", "current_weight": 0.20},
            {"name": "일라이릴리", "ticker": "LLY", "asset_type": "stock", "current_weight": 0.15},
            {"name": "채권", "ticker": "BOND_BUCKET", "asset_type": "bond_bucket", "current_weight": 0.10},
            {"name": "현금", "ticker": "CASH", "asset_type": "cash", "current_weight": 0.05}
        ],
        "settings": {"update_data": False, "generate_predictions": True, "force_prediction": True}
    })
    payload = svc.evaluate(req)
    assert payload["ok"] is True, payload["validation"]
    assert len(payload["latest_signals"]) == 4
    totals = payload["allocation"]["portfolio_totals"]
    assert abs(totals["stock_weight"] + totals["bond_weight"] + totals["cash_weight"] - 1.0) < 1e-6
    print("PASS", payload["as_of_date"], totals)


if __name__ == "__main__":
    main()
