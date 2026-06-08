from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..dependencies import get_market_data_repository, get_training_job_manager, get_training_service
from ..schemas import TrainingRequest

router = APIRouter(prefix="/training", tags=["training"])


def _load_training_cache(repo, ticker: str, data_source: str, market: str):
    """Load OHLCV cache with explicit fallback rules.

    data_source:
      - cache/auto: auto -> yahoo -> kis
      - yahoo: yahoo -> auto
      - kis: kis -> auto
    """
    data_source = data_source.lower().strip()
    if data_source in {"cache", "auto"}:
        candidates = ["auto", "yahoo", "kis"]
    elif data_source == "yahoo":
        candidates = ["yahoo", "auto"]
    elif data_source == "kis":
        candidates = ["kis", "auto"]
    else:
        candidates = [data_source, "auto", "yahoo", "kis"]
    for provider in candidates:
        df = repo.load_ohlcv(provider, ticker, market)
        if df is not None and not df.empty:
            return df, provider
    return None, None


@router.post("/retrain")
def retrain(req: TrainingRequest):
    """Create candidate runtime model artifacts from cached API data.

    This endpoint does not overwrite the locked v8.6.41 prediction-file baseline.
    It registers trained artifacts as CANDIDATE. Use /models/{model_version}/activate only after validation.
    """
    manager = get_training_job_manager()
    repo = get_market_data_repository()
    trainer = get_training_service()

    def target(job):
        results = []
        tickers = [t.upper() for t in req.tickers]
        for i, ticker in enumerate(tickers, start=1):
            df, used_provider = _load_training_cache(repo, ticker, req.data_source, req.market)
            if df is None:
                results.append({"ticker": ticker, "ok": False, "error": "cached market data not found", "data_source": req.data_source})
                continue
            date_series = df["Date"].astype(str)
            mask = (date_series >= req.train_start) & (date_series <= req.train_end)
            train_df = df.loc[mask].copy()
            if train_df.empty:
                results.append({"ticker": ticker, "ok": False, "error": "no rows in selected train period", "provider_used": used_provider})
                continue
            metadata = trainer.train_ticker_candidate(
                ticker,
                train_df,
                req.horizons,
                model_version=req.model_version,
                sample_weight_mode=req.sample_weight_mode,
                walk_forward_mode=req.walk_forward_mode,
                rolling_train_rows=req.rolling_train_rows,
                recency_half_life_by_horizon=req.recency_half_life_by_horizon,
                use_context_features=req.use_context_features,
                context_provider=req.context_provider,
                market=req.market,
            )
            metadata["data_provider_used"] = used_provider
            metadata["train_start"] = req.train_start
            metadata["train_end"] = req.train_end
            metadata["market"] = req.market
            results.append({"ticker": ticker, "ok": True, "provider_used": used_provider, "metadata": metadata})
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
