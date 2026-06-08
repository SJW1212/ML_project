from __future__ import annotations

from functools import lru_cache

from .analytics.performance_analyzer import PerformanceAnalyzer
from .core.config import get_settings
from .core.validators import ParameterValidator
from .data.market_data_repository import MarketDataRepository
from .data.market_data_service import MarketDataService
from .data.prediction_repository import PredictionRepository
from .features.feature_pipeline import FeaturePipeline
from .integrations.kis_provider import KisMarketDataClient
from .integrations.yahoo_provider import YahooFinanceProvider
from .model.inference_service import InferenceService
from .model.model_loader import ModelLoader
from .model.model_registry import ModelRegistry
from .model.prediction_service import PredictionService
from .model.training_job_manager import TrainingJobManager
from .model.training_service import TrainingService
from .portfolio.allocation_policy_engine import AllocationPolicyEngine
from .portfolio.allocation_service import AllocationService
from .portfolio.scenario_comparator import ScenarioComparator
from .presentation.dashboard_serializer import DashboardSerializer
from .presentation.insight_generator import InsightGenerator
from .security.credential_manager import CredentialManager
from .security.token_store import TokenStore


@lru_cache(maxsize=1)
def get_prediction_repository() -> PredictionRepository:
    return PredictionRepository(get_settings().input_dir)


@lru_cache(maxsize=1)
def get_allocation_policy_engine() -> AllocationPolicyEngine:
    return AllocationPolicyEngine()


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    return PredictionService(get_prediction_repository(), get_allocation_policy_engine())


@lru_cache(maxsize=1)
def get_allocation_service() -> AllocationService:
    return AllocationService()


@lru_cache(maxsize=1)
def get_serializer() -> DashboardSerializer:
    return DashboardSerializer(InsightGenerator())


@lru_cache(maxsize=1)
def get_parameter_validator() -> ParameterValidator:
    return ParameterValidator()


@lru_cache(maxsize=1)
def get_credential_manager() -> CredentialManager:
    return CredentialManager(get_settings().secrets_dir)


@lru_cache(maxsize=1)
def get_token_store() -> TokenStore:
    return TokenStore(get_settings().secrets_dir)


@lru_cache(maxsize=1)
def get_market_data_repository() -> MarketDataRepository:
    return MarketDataRepository(get_settings().cache_dir)


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    return ModelRegistry(get_settings().registry_dir)


@lru_cache(maxsize=1)
def get_model_loader() -> ModelLoader:
    return ModelLoader(get_settings().model_dir)


@lru_cache(maxsize=1)
def get_feature_pipeline() -> FeaturePipeline:
    return FeaturePipeline()


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService(get_model_loader(), get_feature_pipeline(), get_allocation_policy_engine())


@lru_cache(maxsize=1)
def get_training_service() -> TrainingService:
    return TrainingService(get_feature_pipeline(), get_model_loader(), get_model_registry(), get_market_data_repository())


@lru_cache(maxsize=1)
def get_training_job_manager() -> TrainingJobManager:
    return TrainingJobManager()


@lru_cache(maxsize=1)
def get_market_data_service() -> MarketDataService:
    return MarketDataService(get_market_data_repository())


def build_yahoo_client() -> YahooFinanceProvider:
    return YahooFinanceProvider()


def build_kis_client(environment: str = "mock") -> KisMarketDataClient:
    return KisMarketDataClient(get_settings(), get_credential_manager(), get_token_store(), environment=environment)
