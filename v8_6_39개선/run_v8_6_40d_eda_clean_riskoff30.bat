@echo off
chcp 65001 > nul

set ASSETS=QQQ,SPY,AAPL,SOXX,NVDA

echo ========================================
echo v8.6.40d EDA - clean baseline
echo ========================================
python v8_6_40d_ph_context_eda.py ^
  --result-dir results_v8_6_40b_clean_compare ^
  --asset-list %ASSETS% ^
  --out-dir results_v8_6_40d_eda_clean

if errorlevel 1 (
  echo [ERROR] clean EDA failed.
  pause
  exit /b 1
)

echo.
echo ========================================
echo v8.6.40d EDA - riskoff30 comparison
echo ========================================
python v8_6_40d_ph_context_eda.py ^
  --result-dir results_v8_6_40b_riskoff30_compare ^
  --asset-list %ASSETS% ^
  --out-dir results_v8_6_40d_eda_riskoff30

if errorlevel 1 (
  echo [ERROR] riskoff30 EDA failed.
  pause
  exit /b 1
)

echo.
echo ========================================
echo Done. Check these folders:
echo   results_v8_6_40d_eda_clean
echo   results_v8_6_40d_eda_riskoff30
echo ========================================
pause
