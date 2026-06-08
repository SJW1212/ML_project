# 구현 요약: 개선 모델 + 백엔드

## 1. 구현한 것

### 백엔드

- FastAPI 기반 백엔드 서버
- `/dashboard` UI payload API
- `/latest` 최신 신호 API
- `/portfolio` 포트폴리오 비중 API
- `/settings/*` 프리셋/설정 검증 API
- `/models/*` 모델 registry API
- `/credentials/*` API key 저장/조회/삭제 API
- `/providers/kis/test-connection` KIS 연결 테스트 API
- `/market-data/*` 외부 데이터 업데이트/조회 API
- `/training/*` 후보 모델 재학습 job API

### 모델 계층

- `PredictionService`: 기존 `v8.6.41_label_fixed` prediction 파일 기반 운영
- `FeaturePipeline`: Live inference / 재학습용 feature 생성
- `InferenceService`: 저장된 model artifact 기반 예측 구조
- `TrainingService`: 후보 모델 재학습 구조
- `ModelRegistry`: active/candidate/archived 모델 관리
- `ModelLoader`: model artifact 저장/로드

### 외부 API 계층

- `CredentialManager`: API Key 암호화 저장
- `TokenStore`: Access Token 저장/만료 관리
- `KisAuthClient`: KIS token 발급
- `KisMarketDataClient`: KIS 현재가/일봉 조회 구조
- `MarketDataRepository`: API 수집 데이터 캐시 저장

## 2. 기본 운영 방식

MVP 기본값은 안전하게 `Prediction File Mode`입니다.

```text
v8.6.41_label_fixed prediction files
→ PredictionRepository
→ PredictionService
→ AllocationService
→ DashboardSerializer
→ FastAPI JSON
→ UI
```

## 3. 개선 모델 구조

실제 운영 baseline은 `v8.6.41_label_fixed`를 유지합니다.

다만 신규 종목/외부 API/재학습을 위해 다음 확장 구조를 포함했습니다.

```text
OHLCV
→ FeaturePipeline
→ TrainingService
→ CANDIDATE model artifact
→ ModelRegistry
→ 전문가 검증 후 ACTIVE 전환
```

후보 모델은 `highvol`, `up_strength`, `down_strength` 3개 head와 `5D/10D/20D` horizon을 지원합니다.

## 4. UI 연결 원칙

CSV는 UI에 직접 연결하지 않습니다.

```text
UI → FastAPI → JSON payload
```

프론트엔드는 `/dashboard`를 기준으로 먼저 연결하면 됩니다.

## 5. 테스트 결과

Smoke test 통과:

```text
/health 200
/models/active 200
/settings/presets 200
/validation 200
/dashboard 200
as_of_date 2026-05-07
portfolio_totals stock=82.0%, bond=11.7%, cash=6.3%
signals=5
```

## 6. 다음 구현 순서

1. Next.js UI 생성
2. `/dashboard` payload 기반 메인 대시보드 연결
3. 포트폴리오 빌더 화면 연결
4. KIS API Key 등록/연결 테스트 화면 연결
5. API 수집 데이터 캐시 검증
6. 전문가 모드 재학습 job 화면 연결

---

## v0.2 구현 메모: Yahoo Finance fallback

### 추가된 모듈

```text
backend/app/integrations/yahoo_provider.py
backend/app/data/market_data_service.py
```

### 수정된 모듈

```text
backend/app/api/market_data_api.py
backend/app/api/credential_api.py
backend/app/data/data_normalizer.py
backend/app/dependencies.py
backend/app/schemas.py
backend/app/core/config.py
requirements.txt
```

### 설계 의도

KIS API Key가 없는 사용자는 첫 실행부터 데이터 갱신 기능을 쓸 수 있어야 한다. 따라서 외부 데이터 provider를 다음 우선순위로 분리했다.

```text
auto mode:
KIS credentials 존재 → KIS 시도 → 실패 시 Yahoo → 실패 시 cache
KIS credentials 없음 → Yahoo → 실패 시 cache
```

### 운영 주의

- Yahoo fallback은 `yfinance`를 사용한다.
- `yfinance`는 Yahoo 공식 제품이 아니므로 실운영 유일 데이터 소스로 고정하지 않는다.
- 데이터 소스는 dashboard payload와 model registry에 기록해야 한다.
- KIS와 Yahoo의 수정주가/거래일/통화/market suffix 차이가 있으므로 재학습 또는 검증 시 provider 정보를 반드시 보존한다.


## v0.3.6 implementation note

SoftFamilyGate was added to avoid over-strict ticker-level rejection. Family FAIL no longer implies full exclusion. Instead, each family receives a confidence score derived from head-level gate status and validation metrics. The family ensemble probability is then shrunk toward 0.5.

Key safety rule:

```text
weak signal -> shrink to neutral
inverted signal -> neutralize
```

Allocation uses effective probabilities only. Raw ensemble probabilities remain available for audit.
