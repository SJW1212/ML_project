# Portfolio Regime Advisor Backend v0.1

`v8.6.41_label_fixed`를 기본 운영 모델로 두고, UI/UX용 FastAPI 백엔드를 제공하는 구현 패키지입니다.

## 포함 범위

- Prediction File Mode: 기존 `v8.6.41_label_fixed` prediction 파일 로딩
- Dashboard API: UI용 JSON payload 생성
- 사용자 모드/프리셋/설정 검증
- 다중 종목 포트폴리오 비중 계산
- 모델 Registry / Live Inference / Retraining 구조
- 한국투자증권(KIS) Open API credential/token/market-data client 구조
- API 키 암호화 저장 MVP 구현
- CSV 직접 UI 연결 대신 JSON/API 중심 구조

## 기본 실행

```bash
cd portfolio_regime_advisor_backend
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Windows:

```bat
run_backend.bat
```

API 문서:

```text
http://127.0.0.1:8000/docs
```

## 주요 API

```text
GET  /health
GET  /dashboard
GET  /latest
GET  /portfolio
POST /portfolio/custom-weights
GET  /settings/presets
POST /settings/validate
GET  /models/active
GET  /models/registry
GET  /validation
```

외부 API/KIS:

```text
POST   /credentials/kis
GET    /credentials/kis/status
DELETE /credentials/kis
POST   /providers/kis/test-connection
POST   /market-data/update
GET    /market-data/freshness
GET    /market-data/{ticker}
```

재학습/후보 모델:

```text
POST /training/retrain
GET  /training/jobs/{job_id}
POST /models/{model_version}/activate
```

## 모델 위치

모델은 UI가 아니라 백엔드의 `backend/app/model/` 계층에 있습니다.

```text
backend/app/model/
  model_registry.py
  model_loader.py
  prediction_service.py
  inference_service.py
  training_service.py
  training_job_manager.py
```

실제 모델 artifact는 코드와 분리해 아래에 저장됩니다.

```text
models/{model_version}/{ticker}/
```

## 운영 모드

### 1. Prediction File Mode

MVP 기본 모드입니다. 기존 prediction 파일을 읽어 UI payload를 생성합니다.

```text
v8.6.41 prediction CSV
→ PredictionRepository
→ PredictionService
→ AllocationService
→ DashboardSerializer
→ FastAPI response
```

### 2. Live Inference Mode

확장 모드입니다. 외부 API 데이터에서 feature를 생성하고 저장된 모델 artifact로 예측합니다.

```text
KIS API / cache
→ FeaturePipeline
→ ModelLoader
→ InferenceService
→ AllocationService
→ Dashboard API
```

## KIS API 연동

MVP에서는 데이터 조회 기능을 위해 KIS Open API 구조를 추가했습니다.

주의:

- 주문 API는 포함하지 않았습니다.
- endpoint path / tr_id는 `kis_provider.py`에서 관리합니다.
- 운영 전 반드시 현재 KIS Developers 공식 문서와 GitHub 샘플로 확인해야 합니다.
- API key와 secret은 암호화 저장되며 로그에 출력하지 않습니다.

Credential 등록 예시:

```bash
curl -X POST http://127.0.0.1:8000/credentials/kis \
  -H "Content-Type: application/json" \
  -d '{
    "environment": "mock",
    "app_key": "YOUR_APP_KEY",
    "app_secret": "YOUR_APP_SECRET",
    "account_no": "12345678",
    "account_product_code": "01"
  }'
```

연결 테스트:

```bash
curl -X POST http://127.0.0.1:8000/providers/kis/test-connection \
  -H "Content-Type: application/json" \
  -d '{"environment":"mock", "ticker":"005930", "market":"KR"}'
```

## UI 연결 방식

프론트엔드는 CSV를 읽지 말고 `/dashboard` JSON을 사용합니다.

```ts
const response = await fetch("http://127.0.0.1:8000/dashboard?assets=QQQ,SPY,AAPL&horizon=10D");
const payload = await response.json();
```

권장 UI 스택:

```text
Next.js + React + TypeScript + Tailwind CSS
Charts: Recharts / ECharts / Lightweight Charts
```

## Smoke Test

```bash
python scripts/test_backend_smoke.py
```

## 현재 한계

- `v8.6.41_label_fixed`의 실제 학습 artifact가 없는 경우, 기본 동작은 prediction 파일 기반입니다.
- Live Inference와 Retraining은 구조와 후보 모델 생성 기능까지 포함했지만, 최종 운영 활성화 전 별도 검증이 필요합니다.
- 사용자 지정 horizon은 기본 안정 지원 대상이 아닙니다. 5D/10D/20D만 안정 지원합니다.
