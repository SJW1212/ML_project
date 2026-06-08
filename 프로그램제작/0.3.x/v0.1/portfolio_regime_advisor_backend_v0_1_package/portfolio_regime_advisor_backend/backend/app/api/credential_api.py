from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..dependencies import build_kis_client, get_credential_manager
from ..schemas import CredentialPayload, ProviderTestRequest

router = APIRouter(prefix="", tags=["credentials/providers"])


@router.post("/credentials/{provider}")
def save_credentials(provider: str, payload: CredentialPayload):
    if provider.lower() != "kis":
        raise HTTPException(status_code=400, detail="Only kis provider is implemented in MVP.")
    get_credential_manager().save_credentials(provider, payload.dict())
    return {"ok": True, "status": get_credential_manager().status(provider)}


@router.get("/credentials/{provider}/status")
def credential_status(provider: str):
    return get_credential_manager().status(provider)


@router.delete("/credentials/{provider}")
def delete_credentials(provider: str):
    existed = get_credential_manager().delete_credentials(provider)
    return {"ok": True, "deleted": existed}


@router.get("/providers")
def providers():
    return {"providers": [{"name": "kis", "display_name": "한국투자증권 Open API", "status": get_credential_manager().status("kis")}]} 


@router.post("/providers/{provider}/test-connection")
def test_provider(provider: str, payload: ProviderTestRequest):
    if provider.lower() != "kis":
        raise HTTPException(status_code=400, detail="Only kis provider is implemented in MVP.")
    client = build_kis_client(payload.environment)
    try:
        result = client.test_connection(ticker=payload.ticker, market=payload.market)
        return result
    except Exception as exc:
        # Do not expose credential values.
        return {"ok": False, "provider": provider.lower(), "environment": payload.environment, "message": str(exc)}
