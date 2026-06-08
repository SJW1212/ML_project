# v8.6.42 Adaptive Controls

## 목적

`v8.6.42_adaptive_controls`는 `v8.6.41_model_label_fixed`의 예측 결과를 입력으로 받아, 아직 고정값으로 남아 있던 7~12번 control layer를 adaptive 구조로 재시뮬레이션하는 버전입니다.

대상 개선 항목:

1. `overall risk` 고정 가중치 → `asset_class × mid_trend_state × ph_rank` 기반 adaptive risk weight
2. `down-risk branch` 고정 가중치 → 조건부 proxy component weight
3. `EWMA span=7` 고정 → 위험/자산군/추세 기반 adaptive span
4. `recency half-life` 고정 → 이 스크립트에서는 allocation-layer signal memory만 반영. 학습 half-life 자체는 full retrain 패치 필요
5. `ph_rank window=756` 고정 → `504/756/1008` multi-window ensemble
6. `asset_class policy table` 수동 고정 → rolling context table 기반 `context_adjust`

## 입력 파일

기본적으로 아래 파일명을 찾습니다.

```text
{ticker}_xgb_recency_weighted_v8_6_41_model_label_fixed_predictions.csv
```

예:

```text
qqq_xgb_recency_weighted_v8_6_41_model_label_fixed_predictions.csv
spy_xgb_recency_weighted_v8_6_41_model_label_fixed_predictions.csv
```

## 실행

```bat
run_v8_6_42_adaptive_controls_resim.bat
```

또는 직접 실행:

```bat
python v8_6_42_adaptive_controls_resim.py ^
  --input-dir . ^
  --out-dir results_v8_6_42_adaptive_controls_from_label_fixed ^
  --asset-list QQQ,SPY,AAPL,SOXX,NVDA ^
  --source-tag xgb_recency_weighted_v8_6_41_model_label_fixed ^
  --transaction-cost-rate 0.001 ^
  --rank-windows 504,756,1008 ^
  --rank-min-periods 252 ^
  --context-window 756 ^
  --rebalance-every 5 ^
  --no-trade-band 0.12 ^
  --max-weight-change-per-rebalance 0.20
```

## 출력

```text
results_v8_6_42_adaptive_controls_from_label_fixed/multi_asset_summary.csv
results_v8_6_42_adaptive_controls_from_label_fixed/adaptive_context_policy_table.csv
results_v8_6_42_adaptive_controls_from_label_fixed/adaptive_controls_config.json
results_v8_6_42_adaptive_controls_from_label_fixed/{ticker}/{ticker}_xgb_recency_weighted_v8_6_42_adaptive_controls_predictions_resim.csv
results_v8_6_42_adaptive_controls_from_label_fixed/{ticker}/{ticker}_xgb_recency_weighted_v8_6_42_adaptive_controls_summary.json
```

## 중요 해석

이 버전은 XGBoost 모델을 재학습하지 않습니다. 목적은 분류 모델 변경이 아니라 allocation/control layer의 고정값 제거 효과를 분리 검증하는 것입니다.

`recency half-life`는 원래 학습 sample weight에 영향을 주므로, 완전한 개선은 full retrain 코드에 asset-class recency profile을 추가해야 합니다. 이 resim 버전에서는 signal memory/control layer만 adaptive하게 처리합니다.

## 현재 smoke test 메모

제공된 5개 ticker 예측 파일 기준으로 구문 검사와 재시뮬레이션 저장은 통과했습니다. 다만 초기 adaptive controls는 공격 신호가 강해져 QQQ/SPY MDD가 커질 수 있습니다. 이 경우 다음 단계는 broad_index guard를 더 강하게 하고, context_adjust 및 upside overlay를 완화하는 것입니다.
