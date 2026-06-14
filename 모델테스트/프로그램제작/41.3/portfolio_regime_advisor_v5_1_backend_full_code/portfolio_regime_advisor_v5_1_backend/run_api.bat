@echo off
setlocal
cd /d %~dp0
set PYTHONPATH=%CD%\src
if not exist storage\logs mkdir storage\logs
python -m uvicorn pra_v5_1.api:app --host 127.0.0.1 --port 8000 --reload
