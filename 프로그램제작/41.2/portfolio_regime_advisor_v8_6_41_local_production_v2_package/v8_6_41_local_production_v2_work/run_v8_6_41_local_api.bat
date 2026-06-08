@echo off
setlocal
set PYTHONPATH=src
REM Local-only v8.6.41 production API.
REM No DB, no account storage, no notifications, no Pixso mapping, no orders.

if "%V8641_INPUT_DIR%"=="" set V8641_INPUT_DIR=.
if "%V8641_OUT_DIR%"=="" set V8641_OUT_DIR=v8_6_41_local_ops
if "%V8641_ASSETS%"=="" set V8641_ASSETS=QQQ,SPY,AAPL,SOXX,NVDA
if "%V8641_CACHE_DIR%"=="" set V8641_CACHE_DIR=storage\market_cache
if "%V8641_UPDATE_PROVIDER%"=="" set V8641_UPDATE_PROVIDER=yahoo
if "%V8641_DAILY_UPDATE_HOUR_KST%"=="" set V8641_DAILY_UPDATE_HOUR_KST=8

python -m uvicorn v8641_production.api_fastapi:app --host 127.0.0.1 --port 8000 --reload
