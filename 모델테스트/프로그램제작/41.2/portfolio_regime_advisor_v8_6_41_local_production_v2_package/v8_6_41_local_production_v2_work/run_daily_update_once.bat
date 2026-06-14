@echo off
setlocal
set PYTHONPATH=src
REM One-shot daily OHLCV cache update. This is not realtime streaming and does not call any order API.
if "%V8641_ASSETS%"=="" set V8641_ASSETS=QQQ,SPY,AAPL,SOXX,NVDA
if "%V8641_INPUT_DIR%"=="" set V8641_INPUT_DIR=.
if "%V8641_CACHE_DIR%"=="" set V8641_CACHE_DIR=storage\market_cache
if not exist storage\logs mkdir storage\logs
python -m v8641_production.cli ^
  --input-dir "%V8641_INPUT_DIR%" ^
  --assets "%V8641_ASSETS%" ^
  --cache-dir "%V8641_CACHE_DIR%" ^
  --update-provider yahoo ^
  --update-cache-only >> storage\logs\daily_update.log 2>&1
