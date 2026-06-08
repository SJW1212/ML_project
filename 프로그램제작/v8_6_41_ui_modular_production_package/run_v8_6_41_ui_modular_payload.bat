@echo off
setlocal
set PYTHONPATH=%~dp0src
python -m v8641_production.cli ^
  --input-dir /mnt/data ^
  --out-dir /mnt/data/v8_6_41_ui_modular_ops ^
  --assets QQQ,SPY,AAPL,SOXX,NVDA ^
  --allocation-source executed ^
  --capital-mode equal
endlocal
