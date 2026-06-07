# v8.6.40d_policy1 Resimulation

## 목적

기존 `v8.6.40b_clean` 또는 `v8.6.40b_riskoff30`의 `*_predictions.csv`를 재사용해, 모델 재학습 없이 `ph_rank × mid_trend_state × asset_class` 기반 배분 정책만 재시뮬레이션합니다.

## 핵심 설계

- `ph_rank`는 `prob_high_vol_raw` 기준으로 계산합니다.
- EWMA 이후 `prob_high_vol`은 `ph_ewma`로 보존합니다.
- `ph_rank_756`은 현재 값을 과거 window와만 비교합니다.
- `min_periods=504` 미만 구간은 기존 `base_signal_*` 비중으로 fallback합니다.
- 기존 v8.6.40b의 offensive overlay는 기본적으로 `old signal - old base` bonus 형태로 보존합니다.
- 기본값에서는 Tier2 bonus를 보존하지 않습니다. 필요하면 `--keep-tier2-bonus`를 사용합니다.

## 실행

```bat
run_v8_6_40d_policy1_resim.bat
```

또는 단독 실행:

```bat
python v8_6_40d_policy1_resim.py ^
  --result-dir results_v8_6_40b_clean_compare ^
  --asset-list QQQ,SPY,AAPL,SOXX,NVDA ^
  --out-dir results_v8_6_40d_policy1_from_clean ^
  --transaction-cost-rate 0.001 ^
  --ph-rank-window 756 ^
  --ph-rank-min-periods 504 ^
  --ph-rank-fallback old_base
```

## 주요 출력

루트:

- `multi_asset_summary.csv`
- `policy1_config.json`

종목별 폴더:

- `<ticker>_xgb_recency_weighted_v8_6_40d_policy1_predictions.csv`
- `<ticker>_xgb_recency_weighted_v8_6_40d_policy1_summary.json`
- `<ticker>_xgb_recency_weighted_v8_6_40d_policy1_regime_trend_ph_rank_performance.csv`
- `<ticker>_xgb_recency_weighted_v8_6_40d_policy1_asset_class_ph_rank_trend_performance.csv`
- `<ticker>_xgb_recency_weighted_v8_6_40d_policy1_hold_reason_ph_rank_trend_performance.csv`

## 해석 순서

1. `multi_asset_summary.csv`로 clean/riskoff30 대비 성과 비교
2. 종목별 `summary.json`의 평균 주식비중과 turnover 확인
3. `regime_trend_ph_rank_performance.csv`에서 HIGH_VOL+BULL / WATCH+BEAR 구간 확인
4. 성과가 악화되면 `--no-context-gate`, `--no-trend-context`, `--keep-tier2-bonus`를 각각 분리 실험

## 주의

이 파일은 full training source가 아니라 **policy resim harness**입니다. 성능이 좋아지면 그때 full source에 이식하는 순서가 안전합니다.
