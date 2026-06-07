# v8.6.41_adaptive_trend

## 목적

`v8.6.41_model_label_fixed`에서 아직 남아 있던 `mid_trend_state` 고정 0 기준 문제를 줄이는 패치입니다.

기존 mid trend는 아래처럼 모든 종목에 같은 조건을 적용했습니다.

```text
return_60d > 0
return_120d > 0
price_ma_60_gap > 0
price_ma_120_gap > 0
ma_gap_20_60 > 0
trend_slope_60 > 0
```

이 방식은 SPY의 +1%와 NVDA의 +1%를 같은 추세 강도로 보는 문제가 있습니다.

## 변경 사항

### 1. adaptive trend context feature 추가

`build_features()`에서 아래 컬럼을 추가합니다.

```text
trend_horizon_vol_60d
trend_horizon_vol_120d
return_60d_vol_scaled
return_120d_vol_scaled
return_60d_rank_756
return_120d_rank_756
price_ma_60_gap_rank_756
price_ma_120_gap_rank_756
ma_gap_20_60_rank_756
trend_slope_60_rank_756
```

### 2. Direction Strength trend score 개선

`strength_trend_score_components()`가 기존 0 기준 대신 다음 구조를 사용합니다.

```text
방향은 양수여야 함
+ 변동성 스케일 기준 또는 rolling rank 기준을 통과해야 함
```

예:

```text
return_60d > 0
AND (return_60d_vol_scaled >= 0.20 OR return_60d_rank_756 >= 0.55)
```

### 3. mid_trend_state 계산 개선

`compute_mid_trend_score()`가 bull_score와 bear_score를 별도로 계산합니다.

BULL:

```text
bull_score >= 4
AND bull_score >= bear_score + 2
```

BEAR:

```text
bear_score >= 4
AND bear_score >= bull_score + 2
```

그 외는 NEUTRAL입니다.

초기 구간 또는 rank 값이 부족한 경우 기존 방식으로 fallback합니다.

## 실행

```bat
run_v8_6_41_adaptive_trend_compare.bat
```

직접 실행:

```bat
python xgb_recency_weighted_v8_6_41_adaptive_trend.py ^
  --asset-list QQQ,SPY,AAPL,SOXX,NVDA ^
  --speed-profile fast ^
  --h10-down-only ^
  --disable-tier2 ^
  --allocation-downrisk-weight 0 ^
  --result-dir results_v8_6_41_adaptive_trend_compare ^
  --transaction-cost-rate 0.001 ^
  --execution-lag-days 1 ^
  --direction-label-mode vol_scaled ^
  --direction-vol-window 60 ^
  --direction-vol-k 0.30 ^
  --direction-min-abs-threshold 0.003 ^
  --direction-max-abs-threshold 0.040 ^
  --direction-strength-eps 0.25 ^
  --direction-margin-rank-threshold 0.65 ^
  --direction-margin-abs-floor 0.03
```

## 확인할 파일

```text
results_v8_6_41_adaptive_trend_compare/multi_asset_summary.csv
```

종목별:

```text
*_xgb_recency_weighted_v8_6_41_adaptive_trend_predictions.csv
*_xgb_recency_weighted_v8_6_41_adaptive_trend_summary.json
*_xgb_recency_weighted_v8_6_41_adaptive_trend_probability_bins.csv
*_xgb_recency_weighted_v8_6_41_adaptive_trend_threshold_diagnostics.csv
```

## 판정 기준

- QQQ/SPY: 기존 41_label_fixed 또는 40b_clean 대비 성과가 크게 무너지면 실패
- AAPL: WATCH/CUSTOM + BEAR 구간 손실 완화 가능성 확인
- SOXX/NVDA: HIGH_VOL + BULL 참여가 과도하게 줄어드는지 확인
- 전체: `mid_trend_state` 분포가 BULL/BEAR 한쪽으로 과도하게 몰리지 않는지 확인

## 주의

이 버전은 `policy1b` 통합 버전이 아닙니다.  
`mid_trend_state`와 Direction Strength 라벨에 쓰이는 trend score를 adaptive화한 모델 개선 패치입니다.
