# Portfolio Regime Advisor Backend v0.3.4

## 목적

v0.3.4는 v0.3.3의 recency-weighted candidate training 위에 다음 운영 리스크를 보완한다.

1. `apply_allocation()` 계층의 dead feature column 문제 수정
2. Market Context Feature scaffolding 추가
3. Head-Level Gate + Selective Inference 추가
4. context data 정렬/캐시/`^TNX` 단위 처리 기준 추가
5. 실험 로그에 fold별 실제 train/test 날짜 및 row 수 기록

## 핵심 변경

### Phase 0: allocation feature join 안전장치

- `AllocationPolicyEngine.merge_prediction_features()` 추가
- `_row_float()`가 missing feature를 silent 0으로 바꾸지 않도록 수정
- context 기반 WATCH override는 `None`일 때 평가 제외
- `ctx_spy_drawdown_252`, `ctx_vix_z_63`, `risk_override`를 policy overlay로 사용 가능

### Phase 1: context cache

- `backend/app/features/context_asset_universe.py`
- `CONTEXT_TICKERS` 정의
- repository cache에서 context assets 1회 로드
- target ticker Date index에 `reindex(...).ffill(limit=2)` 적용
- `^TNX`는 `/100`으로 decimal yield 정규화

### Phase 2: context features

- `backend/app/features/market_context_feature_builder.py`
- volatility, market structure, breadth, cross-asset feature group 추가
- 모든 context features는 `.shift(1)` 적용
- 누락 context ticker는 warning으로 기록하고 NaN 유지

### Phase 3: head-level gate

- `backend/app/model/head_gate.py`
- PASS / UNCERTAIN / FAIL 3단계
- fallback: live / softened / baseline / neutral

### Phase 4: selective inference

- `backend/app/model/selective_inference_service.py`
- selected horizon head별 gate 평가
- PASS는 live, UNCERTAIN은 0.5 방향 shrinkage, FAIL은 baseline 또는 neutral fallback
- down_strength FAIL & highvol PASS 시 방어적 WATCH 가능
- `ctx_spy_drawdown_252 < -0.15`이면 rule-based WATCH override 허용

## TrainingRequest 추가 필드

```json
{
  "use_context_features": true,
  "context_provider": "auto"
}
```

기본값은 `false`다. v0.3.3 결과 재현성을 유지하기 위해 context는 명시적으로 켜야 한다.

## 권장 실행 순서

1. 기존 5개 ticker 캐시 수집
2. context universe ticker 캐시 수집
3. `use_context_features=false`로 v0.3.3 baseline 재현
4. `use_context_features=true`로 v0.3.4 context 실험
5. `/models/infer`에서 `head_gates`, `selective_warnings` 확인

## 주의

- v0.3.4도 운영 기준선이 아니다.
- 운영 기준은 여전히 `v8.6.41 prediction_file`이다.
- context 피처가 늘어난 만큼 SHAP picogroup importance와 OOS portfolio validation이 필요하다.
