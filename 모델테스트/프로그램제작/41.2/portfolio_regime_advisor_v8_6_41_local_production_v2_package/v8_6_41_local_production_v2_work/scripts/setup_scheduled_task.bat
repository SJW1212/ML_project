@echo off
setlocal
REM Registers a local Windows Task Scheduler job for daily OHLCV cache update at 08:00.
REM Scope: daily cache update only. No realtime streaming, no DB, no account storage, no orders.
set SCRIPT_DIR=%~dp0..
if not exist "%SCRIPT_DIR%\storage\logs" mkdir "%SCRIPT_DIR%\storage\logs"
schtasks /create ^
  /tn "PortfolioRegimeAdvisor_v8641_DailyCacheUpdate" ^
  /tr "cmd /c cd /d \"%SCRIPT_DIR%\" && run_daily_update_once.bat" ^
  /sc daily ^
  /st 08:00 ^
  /f
if errorlevel 1 (
  echo Failed to create scheduled task.
  exit /b 1
)
echo Daily cache update task registered at 08:00 local time.
