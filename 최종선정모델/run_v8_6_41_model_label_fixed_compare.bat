@echo off
setlocal

REM v8.6.41 model-label improvement run
REM - Classification model labels are changed from fixed +/-0.5%% to volatility-scaled thresholds.
REM - Allocation settings are kept close to v8.6.40b_clean for controlled comparison.

python xgb_recency_weighted_v8_6_41_model_label_fixed.py ^
  --asset-list QQQ,SPY,AAPL,SOXX,NVDA ^
  --speed-profile fast ^
  --h10-down-only ^
  --disable-tier2 ^
  --allocation-downrisk-weight 0 ^
  --result-dir results_v8_6_41_model_label_fixed_compare ^
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

if errorlevel 1 (
  echo [ERROR] v8.6.41 run failed.
  exit /b 1
)

echo [OK] v8.6.41 model-label comparison completed.
echo Output: results_v8_6_41_model_label_fixed_compare\multi_asset_summary.csv
endlocal
