@echo off
setlocal

REM v8.6.40d_policy1 resimulation from existing v8.6.40b predictions.
REM Put this .bat in the same folder as v8_6_40d_policy1_resim.py.

set ASSETS=QQQ,SPY,AAPL,SOXX,NVDA
set COST=0.001

python v8_6_40d_policy1_resim.py ^
  --result-dir results_v8_6_40b_clean_compare ^
  --asset-list %ASSETS% ^
  --out-dir results_v8_6_40d_policy1_from_clean ^
  --transaction-cost-rate %COST% ^
  --rebalance-every 5 ^
  --no-trade-band 0.12 ^
  --emergency-cooldown 5 ^
  --ph-rank-window 756 ^
  --ph-rank-min-periods 504 ^
  --ph-rank-fallback old_base

python v8_6_40d_policy1_resim.py ^
  --result-dir results_v8_6_40b_riskoff30_compare ^
  --asset-list %ASSETS% ^
  --out-dir results_v8_6_40d_policy1_from_riskoff30 ^
  --transaction-cost-rate %COST% ^
  --rebalance-every 5 ^
  --no-trade-band 0.12 ^
  --emergency-cooldown 5 ^
  --ph-rank-window 756 ^
  --ph-rank-min-periods 504 ^
  --ph-rank-fallback old_base

echo.
echo Done.
echo - results_v8_6_40d_policy1_from_clean\multi_asset_summary.csv
echo - results_v8_6_40d_policy1_from_riskoff30\multi_asset_summary.csv
pause
