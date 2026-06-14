# API_TEST_RESULTS

## Package

- Package: `portfolio_regime_advisor_v8_6_41_local_production_v2_package.zip`
- Model: `v8.6.41_model_label_fixed`
- Mode: `prediction_file`
- Scope: local-only, no DB, no user accounts, no notifications, no Pixso-specific mapping, no order execution, no realtime streaming

## Regression Tests

```text
compileall: PASS
daily cache freshness test: PASS
custom weight portfolio test: PASS
production API payload test: PASS
edge case test: PASS
FastAPI app import: PASS
```

## Important Fixes Verified

```text
portfolio_daily_returns_join_policy: OUTER_JOIN_FILL_MISSING_ASSET_RETURNS_0
asset_date_range_mismatch_validation: WARN
allocation_three_way_fallback_tracking: implemented
inverse_vol_fallback_metadata: implemented
provider request validation: Literal[yahoo,kis]
API duplicate provider parameter: fixed
API log file: storage/logs/api_server.log
one-shot daily update CLI: implemented
Windows scheduled task helper: implemented
```

## Latest Baseline Payload Test

Using available v8.6.41 prediction files under `/mnt/data`:

```text
as_of_date = 2026-05-07
portfolio stock = 82.00%
portfolio bond  = 11.70%
portfolio cash  = 6.30%
validation fail count = 0
```

## Notes

- `WARN` validation checks are intentionally not counted as failures.
- Daily OHLCV cache update does not regenerate v8.6.41 prediction files.
- Prediction files remain the only allocation decision source.
