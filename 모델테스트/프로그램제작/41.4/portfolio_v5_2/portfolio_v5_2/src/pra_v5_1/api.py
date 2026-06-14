from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import AppConfig
from .schemas import DailyUpdateRequest, PortfolioEvaluateRequest, PredictionGenerateRequest, TickerAddRequest
from .service import PortfolioRegimeAdvisorService
from .ui_contract import build_ui_contract

config = AppConfig.from_json()
service = PortfolioRegimeAdvisorService(config)

app = FastAPI(
    title="Portfolio Regime Advisor v5.2 Stitch UI Backend",
    version=__version__,
    description="Local single-user backend: Stitch UI input -> OHLCV cache -> model input -> probabilities -> portfolio allocation -> dashboard payload.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def handle(fn):
    try:
        return fn()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": __version__,
        "scope": {
            "local_only": True,
            "database": False,
            "user_accounts": False,
            "notifications": False,
            "order_execution": False,
            "realtime_streaming": False,
            "daily_ohlcv_update": True,
        },
        "paths": {
            "storage_root": str(config.storage_root),
            "ticker_registry": str(config.registry_path),
            "cache_dir": str(config.cache_dir),
            "prediction_dir": str(config.prediction_dir),
            "log_dir": str(config.log_dir),
        },
    }




@app.get("/ui/contract")
def ui_contract():
    return build_ui_contract()


@app.get("/ui/mock-request")
def ui_mock_request():
    return {
        "portfolio": [
            {"name": "애플", "ticker": "AAPL", "asset_type": "stock", "current_weight": 0.25},
            {"name": "QQQ", "ticker": "QQQ", "asset_type": "etf", "current_weight": 0.25},
            {"name": "엔비디아", "ticker": "NVDA", "asset_type": "stock", "current_weight": 0.20},
            {"name": "일라이릴리", "ticker": "LLY", "asset_type": "stock", "current_weight": 0.15},
            {"name": "채권", "ticker": "BOND_BUCKET", "asset_type": "bond_bucket", "current_weight": 0.10},
            {"name": "현금", "ticker": "CASH", "asset_type": "cash", "current_weight": 0.05}
        ],
        "risk_profile": "balanced",
        "user_level": "general",
        "settings": {
            "start_date": "2013-01-01",
            "update_data": True,
            "generate_predictions": True,
            "prediction_engine_mode": "reference_v8641_compatible",
            "capital_mode": "current_weight",
            "missing_asset_policy": "cash_fallback"
        }
    }

@app.get("/tickers")
def tickers(enabled_only: bool = False):
    return {"tickers": [r.__dict__ for r in service.registry.list(enabled_only=enabled_only)]}


@app.post("/tickers/add")
def add_tickers(req: TickerAddRequest):
    return handle(lambda: service.add_tickers(req.tickers, req.asset_type, req.market, req.note))


@app.get("/data/freshness")
def data_freshness(provider: str = "yahoo"):
    tickers = service.registry.enabled_tickers()
    return {"provider": provider, "freshness": service.cache.freshness(tickers, provider=provider)}


@app.post("/data/update-daily")
def update_daily(req: DailyUpdateRequest):
    return handle(lambda: service.update_daily(req))


@app.post("/predictions/generate")
def generate_predictions(req: PredictionGenerateRequest):
    return handle(lambda: service.generate_predictions(req))


@app.get("/predictions/status")
def prediction_status():
    rows = []
    for t in service.registry.enabled_tickers():
        rows.append({"ticker": t, "exists": service.pred_repo.exists(t), "latest_date": service.pred_repo.latest_date(t), "path": str(service.pred_repo.prediction_path(t))})
    return {"predictions": rows}


@app.post("/portfolio/evaluate")
def portfolio_evaluate(req: PortfolioEvaluateRequest):
    return handle(lambda: service.evaluate(req))
