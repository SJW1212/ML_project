# Portfolio Regime Advisor v8.6.41 Production API

## 목적

이 패키지는 `v8.6.41_model_label_fixed`를 운영 기준선으로 고정하고, 기존 prediction CSV를 API/UI에서 바로 사용할 수 있는 JSON payload로 변환합니다.

이 패키지는 아래 실험 레이어를 기본 운영에 사용하지 않습니다.

- v0.3.x live runtime model
- context head gate
- horizon ensemble
- soft family gate
- confidence weighted v1/v2
- accuracy benchmark v3
- logit calibrated v4
- loss guard
- v8.6.42 adaptive controls

## 구조

```text
src/v8641_production/
  api_fastapi.py      # FastAPI endpoints
  service.py          # dashboard payload facade
  repository.py       # v8.6.41 prediction/summary file loader
  signals.py          # latest signal normalization
  allocation.py       # native 41 allocation aggregation
  performance.py      # performance/chart payload
  validation.py       # payload validation checks
  serializer.py       # JSON/CSV/MD output writer
  cli.py              # CLI payload generator
```

## 설치

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install pandas numpy fastapi uvicorn pydantic
```

## 전제

`V8641_INPUT_DIR`에는 아래 파일들이 있어야 합니다.

```text
qqq_xgb_recency_weighted_v8_6_41_model_label_fixed_predictions.csv
qqq_xgb_recency_weighted_v8_6_41_model_label_fixed_summary.json
spy_xgb_recency_weighted_v8_6_41_model_label_fixed_predictions.csv
...
```

현재 작업 폴더에 파일이 있으면 별도 설정 없이 실행할 수 있습니다.

## API 실행

```bat
set V8641_INPUT_DIR=C:\path\to\prediction_files
set V8641_ASSETS=QQQ,SPY,AAPL,SOXX,NVDA
run_v8_6_41_production_api.bat
```

또는 직접 실행:

```bat
set PYTHONPATH=src
python -m uvicorn v8641_production.api_fastapi:app --host 127.0.0.1 --port 8000 --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## 주요 API

```text
GET  /health
GET  /assets
GET  /dashboard
POST /dashboard
GET  /latest
POST /portfolio/custom-weights
GET  /validation
GET  /performance
GET  /schema
```

## 예시

### 전체 dashboard

```text
GET http://127.0.0.1:8000/dashboard?assets=QQQ,SPY,AAPL,SOXX,NVDA&allocation_source=executed&capital_mode=equal
```

### 최신 상태만

```text
GET http://127.0.0.1:8000/latest?assets=QQQ,SPY,AAPL,SOXX,NVDA
```

### 사용자 비중

```json
POST /portfolio/custom-weights
{
  "assets": ["QQQ", "SPY", "AAPL", "SOXX", "NVDA"],
  "weights": {
    "QQQ": 0.25,
    "SPY": 0.20,
    "AAPL": 0.15,
    "SOXX": 0.20,
    "NVDA": 0.20
  },
  "allocation_source": "executed"
}
```

## CLI로 JSON 생성

```bat
set PYTHONPATH=src
python -m v8641_production.cli ^
  --input-dir C:\path\to\prediction_files ^
  --out-dir v8_6_41_ui_modular_ops ^
  --assets QQQ,SPY,AAPL,SOXX,NVDA ^
  --allocation-source executed ^
  --capital-mode equal ^
  --no-zip
```

## 운영 원칙

- 모델은 재학습하지 않습니다.
- 입력은 v8.6.41 prediction CSV입니다.
- 최종 UI/API는 JSON payload를 기본으로 사용합니다.
- CSV는 디버깅/엑셀 공유용 옵션입니다.
- v0.3.x 실험 구조는 운영 기준선에서 제외합니다.
