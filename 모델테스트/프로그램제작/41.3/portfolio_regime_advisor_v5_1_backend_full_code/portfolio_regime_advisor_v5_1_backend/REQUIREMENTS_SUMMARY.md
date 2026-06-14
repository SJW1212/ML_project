# 요구사항 통합 요약 v5.1

## 운영 기준

- v8.6.41 계열 판단 출력 구조 유지
- 사용자가 입력한 종목을 기준으로 동작
- OHLCV는 provider layer에서만 수집
- model input은 local cache에서만 생성
- 각 종목별 확률 출력 후 포트폴리오 모듈로 전달
- 최종 dashboard payload를 UI에 반환

## 사용자 포트폴리오 지원

다음과 같은 입력을 지원합니다.

```text
AAPL / QQQ / NVDA / LLY / 채권 / 현금
```

- AAPL, QQQ, NVDA, LLY: OHLCV 기반 모델 분석 대상
- 채권 bucket: 방어 비중으로 처리
- 현금 bucket: 방어 비중으로 처리
- 채권을 IEF/TLT/BIL 등 ETF로 넣으면 별도 자산으로 확장 가능

## 사용자 수준별 설정

### 일반사용자
- 종목 입력
- 투자금/비중 입력
- 위험 성향 선택
- 대시보드 확인
- 내부 threshold 직접 조정 없음

### 고급사용자
- capital_mode
- custom_weights
- holdout_start
- transaction_cost_bps
- min_cash_weight
- max_asset_weight
- risk_sensitivity
- missing_asset_policy

### 전문가
- prediction generation
- speed_profile
- 학습/검증 기간
- threshold set
- feature/probability/성과 진단

### 개발자
- config path
- cache path
- prediction path
- model engine path
- log level
- API schema
- tests

## 제외 항목

- DB
- 계정 저장
- 알림
- Pixso 설계 반영
- 주문/자동매매
- 실시간 스트리밍
