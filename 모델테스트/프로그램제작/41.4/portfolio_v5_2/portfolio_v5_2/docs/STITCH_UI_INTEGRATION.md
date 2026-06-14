# Stitch UI Integration Guide

## 1. 결론

이 백엔드는 Stitch로 생성한 웹 UI와 REST API(JSON)로 연결하는 구조를 전제로 한다.
투자 판단, 확률 생성, 포트폴리오 비중 계산, 검증 로직은 프론트에 넣지 않고 백엔드에 둔다.

## 2. 로컬 실행

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
run_api.bat
```

Swagger 확인:

```text
http://127.0.0.1:8000/docs
```

Stitch/프론트에서 호출할 기본 엔드포인트:

```text
POST http://127.0.0.1:8000/portfolio/evaluate
```

UI 계약 확인:

```text
GET http://127.0.0.1:8000/ui/contract
GET http://127.0.0.1:8000/ui/mock-request
```

## 3. 요청 JSON

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

## 4. Stitch UI에서 반드시 표시할 응답 필드

| 섹션 | 필드 | UI 의미 |
|---|---|---|
| 엔진 상태 | `engine_status.engine_mode` | 현재 엔진 모드 |
| 엔진 상태 | `engine_status.production_ready_for_investment` | 실전 투자판단 가능 여부 |
| 엔진 상태 | `engine_status.warning` | reference engine 경고 |
| 요약 | `ui.current_weights_pct` | 현재 주식/채권/현금 비중 |
| 요약 | `ui.recommended_weights_pct` | 추천 주식/채권/현금 비중 |
| 요약 | `ui.delta_pct_points` | 현재 대비 변화폭 |
| 종목 판단 | `allocation.allocation_rows` | 종목별 현재/추천 비중, 액션, 근거 |
| 신호 | `latest_signals` | 종목별 확률, risk/direction/allocation class |
| 성과 | `performance.metrics` | CAGR, MDD, Sharpe 등 |
| 벤치마크 | `benchmarks.items` | 모델 추천 vs 현재 유지 vs QQQ/SPY/60-40 등 |
| 검증 | `validation` | fail/warn/pass 및 UI 경고 |

## 5. 안전 규칙

1. `production_ready_for_investment=false`이면 투자 추천/실전 운용 가능 문구를 쓰지 않는다.
2. `engine_mode`는 화면 상단 상태 배너에 항상 노출한다.
3. `reference_v8641_compatible` 결과는 “백엔드/UI 흐름 검증용”으로만 표시한다.
4. `performance.equity_curve_tail[-1].return_status="pending"`이면 해당 날짜 수익률은 실현 수익률로 표시하지 않는다.
5. 영어 enum은 내부 디버그에 남기고, 사용자 화면에는 `*_ko` 필드를 우선 표시한다.

## 6. Stitch 프롬프트

```text
Create a Korean desktop-first web dashboard UI for a portfolio regime advisor.

The app receives a user portfolio and displays model-based recommended allocation across stocks, bonds, and cash.

Design style:
- Clean financial dashboard
- Korean language UI
- Desktop-first responsive layout
- Light mode
- Clear risk warning banner
- Data-card based layout
- Suitable for individual investors

Main screens:
1. Portfolio Input Screen
- Table input for asset name, ticker, asset type, current weight
- Add row and remove row buttons
- Total weight validation indicator
- Submit button: "포트폴리오 분석하기"

2. Dashboard Result Screen
- Top engine status banner
- If production_ready_for_investment is false, show a yellow warning:
  "현재 결과는 reference-compatible engine 기반이며, 실제 locked v8.6.41 trained model 결과가 아닙니다."
- Summary cards:
  현재 주식 비중, 추천 주식 비중, 추천 채권 비중, 추천 현금 비중, 위험 상태
- Allocation comparison bar chart:
  현재 비중 vs 추천 비중
- Asset decision table:
  종목명, 티커, 현재 비중, 추천 비중, 위험 판단, 방향 판단, 액션, 판단 근거
- Performance cards:
  CAGR, MDD, Sharpe, Sortino, Calmar
- Benchmark comparison table:
  모델 추천, 현재 포트폴리오 유지, QQQ Buy&Hold, SPY Buy&Hold, 60/40, 85/10/5
- Validation panel:
  pass/fail status, warning count, error count

API integration:
- POST http://127.0.0.1:8000/portfolio/evaluate
- GET http://127.0.0.1:8000/ui/contract
- GET http://127.0.0.1:8000/ui/mock-request

Important behavior:
- Never hide engine_mode
- Never present reference engine output as production investment advice
- Use Korean labels for user-facing text
- Keep English enum values available only in a technical detail area
```
