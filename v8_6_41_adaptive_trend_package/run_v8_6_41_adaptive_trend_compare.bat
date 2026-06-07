@echo off
setlocal

REM v8.6.41_adaptive_trend
REM - Direction label: volatility-scaled
REM - Direction Strength trend_delta: minimum eps
REM - mid_trend_state: adaptive vol/rank-aware trend score
REM - Allocation policy: existing v8.6.41 label-fixed policy, no policy1b integration

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

endlocal
