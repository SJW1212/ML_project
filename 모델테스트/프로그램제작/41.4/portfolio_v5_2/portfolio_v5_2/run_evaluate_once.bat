@echo off
setlocal
cd /d %~dp0
set PYTHONPATH=%CD%\src
python -m pra_v5_1.local_cli evaluate --request examples\portfolio_request_example.json
