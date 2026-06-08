#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD"
export PRA_INPUT_DIR="$PWD/storage/predictions"
export PRA_STORAGE_DIR="$PWD/storage"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
