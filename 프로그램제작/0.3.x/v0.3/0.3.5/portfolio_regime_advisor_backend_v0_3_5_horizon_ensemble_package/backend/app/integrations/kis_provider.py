from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import requests

from ..core.config import AppSettings
from ..core.exceptions import ProviderError
from ..data.data_normalizer import DataNormalizer
from ..security.credential_manager import CredentialManager
from ..security.token_store import TokenStore
from .base_provider import MarketDataProvider


@dataclass
class KisEndpointConfig:
    """KIS endpoint/tr_id constants.

    Keep these configurable. KIS endpoint paths/TR IDs should be verified against the
    current KIS Developers documentation before production deployment.
    """

    oauth_token_path: str = "/oauth2/tokenP"
    domestic_current_price_path: str = "/uapi/domestic-stock/v1/quotations/inquire-price"
    domestic_daily_price_path: str = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    overseas_current_price_path: str = "/uapi/overseas-price/v1/quotations/price"
    overseas_daily_price_path: str = "/uapi/overseas-price/v1/quotations/dailyprice"

    domestic_current_price_tr_id: str = "FHKST01010100"
    domestic_daily_price_tr_id: str = "FHKST03010100"
    overseas_current_price_tr_id: str = "HHDFS00000300"
    overseas_daily_price_tr_id: str = "HHDFS76240000"


class KisAuthClient:
    def __init__(
        self,
        settings: AppSettings,
        credential_manager: CredentialManager,
        token_store: TokenStore,
        endpoint_config: Optional[KisEndpointConfig] = None,
    ):
        self.settings = settings
        self.credential_manager = credential_manager
        self.token_store = token_store
        self.endpoint_config = endpoint_config or KisEndpointConfig()

    def base_url(self, environment: str) -> str:
        return self.settings.kis_mock_base_url if environment.lower() == "mock" else self.settings.kis_real_base_url

    def issue_access_token(self, environment: str = "mock") -> str:
        creds = self.credential_manager.load_credentials("kis")
        if not creds:
            raise ProviderError("KIS credentials are not registered.")
        url = self.base_url(environment) + self.endpoint_config.oauth_token_path
        body = {
            "grant_type": "client_credentials",
            "appkey": creds["app_key"],
            "appsecret": creds["app_secret"],
        }
        try:
            resp = requests.post(url, json=body, timeout=10)
        except Exception as exc:
            raise ProviderError(f"KIS token request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise ProviderError(f"KIS token request returned HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        token = data.get("access_token") or data.get("accessToken")
        if not token:
            raise ProviderError(f"KIS token response has no access_token: {data}")
        # Common KIS responses include access_token_token_expired, but keep fallback robust.
        expires_at = data.get("access_token_token_expired")
        if not expires_at:
            expires_at = (pd.Timestamp.utcnow() + pd.Timedelta(hours=23)).isoformat()
        self.token_store.save_token("kis", environment, token, str(expires_at))
        return token

    def get_valid_access_token(self, environment: str = "mock") -> str:
        if self.token_store.is_valid("kis", environment):
            token = self.token_store.get_token("kis", environment)
            assert token is not None
            return token["access_token"]
        return self.issue_access_token(environment)


class KisMarketDataClient(MarketDataProvider):
    def __init__(
        self,
        settings: AppSettings,
        credential_manager: CredentialManager,
        token_store: TokenStore,
        environment: str = "mock",
        endpoint_config: Optional[KisEndpointConfig] = None,
    ):
        self.settings = settings
        self.credential_manager = credential_manager
        self.token_store = token_store
        self.environment = environment
        self.endpoint_config = endpoint_config or KisEndpointConfig()
        self.auth = KisAuthClient(settings, credential_manager, token_store, self.endpoint_config)

    def base_url(self) -> str:
        return self.settings.kis_mock_base_url if self.environment.lower() == "mock" else self.settings.kis_real_base_url

    def _headers(self, tr_id: str) -> Dict[str, str]:
        creds = self.credential_manager.load_credentials("kis")
        if not creds:
            raise ProviderError("KIS credentials are not registered.")
        token = self.auth.get_valid_access_token(self.environment)
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": creds["app_key"],
            "appsecret": creds["app_secret"],
            "tr_id": tr_id,
        }

    def _get(self, path: str, params: dict, tr_id: str) -> dict:
        url = self.base_url() + path
        try:
            resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        except Exception as exc:
            raise ProviderError(f"KIS request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise ProviderError(f"KIS request returned HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        rt_cd = str(data.get("rt_cd", "0"))
        if rt_cd not in {"0", "None", ""}:
            raise ProviderError(f"KIS business error: {data.get('msg_cd')} {data.get('msg1')}")
        return data

    def test_connection(self, ticker: str = "005930", market: str = "KR") -> dict:
        price = self.get_current_price(ticker=ticker, market=market)
        return {"ok": True, "provider": "kis", "environment": self.environment, "sample": price}

    def get_current_price(self, ticker: str, market: str = "KR") -> dict:
        market = market.upper()
        if market == "KR":
            params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
            data = self._get(
                self.endpoint_config.domestic_current_price_path,
                params,
                self.endpoint_config.domestic_current_price_tr_id,
            )
            return data.get("output") or data
        # US market example. EXCD can be NAS/NYSE/AMS depending on ticker; expose it later as a UI setting.
        params = {"AUTH": "", "EXCD": "NAS", "SYMB": ticker.upper()}
        data = self._get(
            self.endpoint_config.overseas_current_price_path,
            params,
            self.endpoint_config.overseas_current_price_tr_id,
        )
        return data.get("output") or data

    def get_daily_ohlcv(self, ticker: str, start_date: str, end_date: str, market: str = "KR") -> pd.DataFrame:
        market = market.upper()
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        if market == "KR":
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
                "FID_INPUT_DATE_1": start,
                "FID_INPUT_DATE_2": end,
                "FID_PERIOD_DIV_CODE": "D",
                "FID_ORG_ADJ_PRC": "1",
            }
            data = self._get(
                self.endpoint_config.domestic_daily_price_path,
                params,
                self.endpoint_config.domestic_daily_price_tr_id,
            )
            rows = data.get("output2") or data.get("output") or []
            return DataNormalizer.normalize_kis_domestic_daily(rows, ticker)
        params = {"AUTH": "", "EXCD": "NAS", "SYMB": ticker.upper(), "GUBN": "0", "BYMD": end, "MODP": "1"}
        data = self._get(
            self.endpoint_config.overseas_daily_price_path,
            params,
            self.endpoint_config.overseas_daily_price_tr_id,
        )
        rows = data.get("output2") or data.get("output") or []
        df = DataNormalizer.normalize_kis_overseas_daily(rows, ticker, market=market)
        if not df.empty:
            mask = (pd.to_datetime(df["Date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(df["Date"]) <= pd.to_datetime(end_date))
            df = df.loc[mask].copy()
        return df
