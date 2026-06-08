# Portfolio Regime Advisor v5.1 Local Backend

## 목적

로컬 단일 사용자 환경에서 다음 흐름을 수행합니다.

```text
UI/UX 사용자 입력
→ Input Normalizer
→ Ticker Registry(JSON)
→ Daily OHLCV Cache
→ Model Input Builder
→ v8.6.41-compatible Prediction Engine
→ Probability / Signal Output
→ Portfolio Allocation Module
→ Dashboard Payload
→ UI 반환
```

## 제외 범위

- DB 저장 없음
- 사용자 계정별 포트폴리오 저장 없음
- 알림 없음
- Pixso 화면 설계 반영 없음
- 주문/자동매매 없음
- 실시간 스트리밍 없음

## 실행

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
run_api.bat
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## 핵심 API

- `POST /portfolio/evaluate`: 통합 평가 API
- `POST /tickers/add`: 로컬 ticker registry 저장
- `POST /data/update-daily`: 하루 1회 OHLCV 캐시 갱신
- `POST /predictions/generate`: 캐시 기반 prediction 생성
- `GET /predictions/status`: prediction artifact 상태
- `GET /data/freshness`: cache 최신성

## 포트폴리오 입력 예시

```json
{
  "portfolio": [
    {"name": "애플", "ticker": "AAPL", "asset_type": "stock", "current_weight": 0.25},
    {"name": "QQQ", "ticker": "QQQ", "asset_type": "etf", "current_weight": 0.25},
    {"name": "엔비디아", "ticker": "NVDA", "asset_type": "stock", "current_weight": 0.20},
    {"name": "일라이릴리", "ticker": "LLY", "asset_type": "stock", "current_weight": 0.15},
    {"name": "채권", "ticker": "BOND_BUCKET", "asset_type": "bond_bucket", "current_weight": 0.10},
    {"name": "현금", "ticker": "CASH", "asset_type": "cash", "current_weight": 0.05}
  ],
  "risk_profile": "balanced",
  "user_level": "general",
  "settings": {
    "start_date": "2013-01-01",
    "update_data": true,
    "generate_predictions": true,
    "prediction_engine_mode": "reference_v8641_compatible",
    "capital_mode": "current_weight",
    "missing_asset_policy": "cash_fallback"
  }
}
```

## Prediction Engine Mode

### reference_v8641_compatible

- 기본값입니다.
- OHLCV 캐시만 읽어서 leakage-safe feature를 만들고, v8.6.41 출력 스키마와 동일한 확률 컬럼을 생성합니다.
- 외부 네트워크를 직접 호출하지 않습니다.

### external_v8641_xgb

- 원본 `xgb_recency_weighted_v8_6_41_model_label_fixed.py`를 연결하기 위한 reserved mode입니다.
- 원본 스크립트는 `model_engine/`에 포함되어 있고, 같은 폴더의 `yfinance.py` shim은 캐시 CSV를 읽도록 되어 있습니다.
- 전체 XGBoost 재학습/생성은 시간이 오래 걸릴 수 있습니다.

## 테스트

```bat
set PYTHONPATH=%CD%\src
python -m compileall src
python scripts\test_smoke.py
```
