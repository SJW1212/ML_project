@echo off
setlocal
set PYTHONPATH=src
if "%V8641_INPUT_DIR%"=="" set V8641_INPUT_DIR=%cd%
if "%V8641_OUT_DIR%"=="" set V8641_OUT_DIR=%cd%\v8_6_41_ui_modular_ops
if "%V8641_ASSETS%"=="" set V8641_ASSETS=QQQ,SPY,AAPL,SOXX,NVDA
python -m uvicorn v8641_production.api_fastapi:app --host 127.0.0.1 --port 8000 --reload
