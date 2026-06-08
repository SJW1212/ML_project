@echo off
setlocal

REM Full retrain variant: removes fixed recency half-life by applying asset-class recency profiles.
REM Requires the same environment used to run v8.6.41 model_label_fixed.

python xgb_recency_weighted_v8_6_42_adaptive_recency_profile.py ^
  --asset-list QQQ,SPY,AAPL,SOXX,NVDA ^
  --speed-profile fast ^
  --h10-down-only ^
  --disable-tier2 ^
  --allocation-downrisk-weight 0 ^
  --result-dir results_v8_6_42_adaptive_recency_profile_compare ^
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

pause
