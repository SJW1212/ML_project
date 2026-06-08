from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..dependencies import build_kis_client, build_yahoo_client, get_credential_manager
from ..schemas import CredentialPayload, ProviderTestRequest

router = APIRouter(prefix="", tags=["credentials/providers"])


@router.post("/credentials/{provider}")
def save_credentials(provider: str, payload: CredentialPayload):
    provider = provider.lower().strip()
    if provider != "kis":
        raise HTTPException(status_code=400, detail="Only kis credentials are stored. Yahoo fallback requires no API key.")
    get_credential_manager().save_credentials(provider, payload.dict())
    return {"ok": True, "status": get_credential_manager().status(provider)}


@router.get("/credentials/{provider}/status")
def credential_status(provider: str):
    provider = provider.lower().strip()
    if provider == "yahoo":
        return {"provider": "yahoo", "exists": True, "requires_credentials": False, "message": "Yahoo fallback uses yfinance and does not require an API key."}
    return get_credential_manager().status(provider)


@router.delete("/credentials/{provider}")
def delete_credentials(provider: str):
    provider = provider.lower().strip()
    if provider == "yahoo":
        return {"ok": True, "deleted": False, "message": "Yahoo fallback has no stored credentials."}
    existed = get_credential_manager().delete_credentials(provider)
    return {"ok": True, "deleted": existed}


@router.get("/providers")
def providers():
    cm = get_credential_manager()
    return {
        "providers": [
            {
                "name": "auto",
                "display_name": "자동 선택: KIS → Yahoo → Cache",
                "requires_credentials": False,
                "status": {"provider": "auto", "exists": True},
            },
            {
                "name": "kis",
                "display_name": "한국투자증권 Open API",
                "requires_credentials": True,
                "status": cm.status("kis"),
            },
            {
                "name": "yahoo",
                "display_name": "Yahoo Finance fallback",
                "requires_credentials": False,
                "status": {"provider": "yahoo", "exists": True},
            },
        ]
    }


@router.post("/providers/{provider}/test-connection")
def test_provider(provider: str, payload: ProviderTestRequest):
    provider = provider.lower().strip()
    try:
        if provider == "kis":
            client = build_kis_client(payload.environment)
            return client.test_connection(ticker=payload.ticker, market=payload.market)
        if provider == "yahoo":
            client = build_yahoo_client()
            return client.test_connection(ticker=payload.ticker, market=payload.market)
        if provider == "auto":
            cm = get_credential_manager()
            if cm.load_credentials("kis"):
                kis_result = test_provider("kis", payload)
                if kis_result.get("ok"):
                    kis_result["provider_requested"] = "auto"
                    return kis_result
            yahoo_result = test_provider("yahoo", payload)
            yahoo_result["provider_requested"] = "auto"
            return yahoo_result
        raise HTTPException(status_code=400, detail="provider must be one of: auto, kis, yahoo")
    except HTTPException:
        raise
    except Exception as exc:
        # Do not expose credential values.
        return {"ok": False, "provider": provider, "environment": payload.environment, "message": str(exc)}
