# 모듈/클래스 설계 검토 v5.1

## schemas.py

- `PortfolioEvaluateRequest`: UI에서 들어오는 통합 요청 모델.
- `PortfolioAssetInput`: 종목/ETF/채권 bucket/현금 bucket 구분.
- `EvaluateSettings`: 사용자 수준별 조정값의 기본 컨테이너.

## ticker_registry.py

- `LocalTickerRegistry`: DB 없이 `storage/config/tickers.json`에 종목 저장.
- 사용자 계정 저장이 아니라 로컬 실행 상태 저장입니다.

## provider.py

- `YahooMarketDataProvider`: yfinance로 OHLCV 조회.
- `DailyMarketDataUpdater`: 하루 1회 캐시 갱신용.
- KIS는 주문이 아니라 시세/일봉 조회 adapter 후보입니다.

## cache.py

- `MarketDataCache`: OHLCV CSV 저장/읽기/최신성 확인.
- 경로: `storage/market_cache/ohlcv/{provider}/{TICKER}.csv`.

## model_input.py

- `ModelInputBuilder`: cache OHLCV만 읽어 feature 생성.
- future/label/target 컬럼을 만들지 않습니다.

## feature_schema.py

- `FeatureSchema`: leakage guard.
- prefix 제외만 하지 않고 allowed feature prefix와 required prediction columns를 검증합니다.

## prediction_engine.py

- `ReferenceV8641CompatibleEngine`: v8.6.41 출력 스키마와 동일한 확률 컬럼 생성.
- `PredictionGenerationService`: 종목별 prediction artifact 저장.
- `LocalRunTransaction`: staging/manifest 기반 로컬 transaction-like 기록.

## allocation.py

- `PortfolioAllocationService`: 현재 보유 비중과 모델 stock/bond/cash 비중을 결합.
- risk asset에서 줄인 비중은 채권/현금 bucket으로 이동.

## performance.py

- `PerformanceAnalyzer`: CAGR/MDD/Sharpe/Sortino/Calmar 및 equity curve 계산.
- missing_asset_policy 지원: cash_fallback / active_weight_renormalize / common_range_only.

## validation.py

- `ValidationService`: prediction schema, prediction/cache date mismatch, 자산별 기간 불일치, 비중 합계 검증.

## service.py

- `PortfolioRegimeAdvisorService`: 전체 파이프라인 orchestration.

## api.py

- FastAPI 진입점.
- 핵심 API: `POST /portfolio/evaluate`.
