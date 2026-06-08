from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..dependencies import get_market_data_repository, get_training_job_manager, get_training_service
from ..schemas import TrainingRequest

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/retrain")
def retrain(req: TrainingRequest):
    """Create a candidate retraining job from cached API data.

    This is expert-mode backend infrastructure. UI should call settings validation first.
    """
    manager = get_training_job_manager()
    repo = get_market_data_repository()
    trainer = get_training_service()

    def target(job):
        results = []
        tickers = [t.upper() for t in req.tickers]
        for i, ticker in enumerate(tickers, start=1):
            # MVP assumes US tickers cached under KIS/US or KR codes under KIS/KR; try US first, then KR.
            df = repo.load_ohlcv(req.data_source if req.data_source != "cache" else "kis", ticker, "US")
            if df is None:
                df = repo.load_ohlcv(req.data_source if req.data_source != "cache" else "kis", ticker, "KR")
            if df is None:
                results.append({"ticker": ticker, "ok": False, "error": "cached market data not found"})
                continue
            mask = (df["Date"] >= req.train_start) & (df["Date"] <= req.train_end)
            train_df = df.loc[mask].copy()
            if train_df.empty:
                results.append({"ticker": ticker, "ok": False, "error": "no rows in selected train period"})
                continue
            metadata = trainer.train_ticker_candidate(ticker, train_df, req.horizons)
            results.append({"ticker": ticker, "ok": True, "metadata": metadata})
            job.progress = 0.05 + 0.90 * i / max(len(tickers), 1)
            job.message = f"Trained {i}/{len(tickers)} tickers"
        return {"results": results}

    job = manager.create_job(target)
    return {"job_id": job.job_id, "status": job.status, "progress": job.progress, "message": job.message}


@router.get("/jobs/{job_id}")
def get_training_job(job_id: str):
    job = get_training_job_manager().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job.job_id, "status": job.status, "progress": job.progress, "message": job.message, "result": job.result}
