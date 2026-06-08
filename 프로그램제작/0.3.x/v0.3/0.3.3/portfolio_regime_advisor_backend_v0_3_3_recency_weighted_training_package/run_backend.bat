@echo off
setlocal
set PYTHONPATH=%CD%
set PRA_INPUT_DIR=%CD%\storage\predictions
set PRA_STORAGE_DIR=%CD%\storage
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
