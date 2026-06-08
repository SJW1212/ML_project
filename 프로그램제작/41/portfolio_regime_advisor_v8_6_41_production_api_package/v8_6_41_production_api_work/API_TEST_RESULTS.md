# API / Payload Test Results

Environment: local container validation using `/mnt/data` prediction files.

## Commands

```bash
PYTHONPATH=src python -m compileall -q src scripts
PYTHONPATH=src python scripts/test_v8_6_41_production_api_payload.py
PYTHONPATH=src python scripts/test_v8_6_41_custom_weights.py
```

## Results

- compileall: PASS
- equal-weight payload test: PASS
- custom-weight payload test: PASS
- FastAPI smoke test `/health`, `/latest`, `/portfolio/custom-weights`: PASS

## Baseline equal-weight latest portfolio

```text
as_of_date = 2026-05-07
stock      = 0.8200
bond       = 0.1170
cash       = 0.0630
validation = 0 FAIL
```

## Custom-weight smoke portfolio

Weights:

```json
{"QQQ": 0.25, "SPY": 0.20, "AAPL": 0.15, "SOXX": 0.20, "NVDA": 0.20}
```

Result:

```text
stock = 0.8260
bond  = 0.1131
cash  = 0.0609
```
