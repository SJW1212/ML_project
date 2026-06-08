# v8.6.41 Model Label Fixed

## Purpose

This version modifies the **classification model / label layer**, not the policy1b allocation layer.

The main target is the structural problem where fixed thresholds mean different things for low-volatility assets and high-volatility assets.

## Base file

- Base: `xgb_recency_weighted_v8_6_40b_cli_fixed.py`
- New file: `xgb_recency_weighted_v8_6_41_model_label_fixed.py`

## Main changes

### 1. Direction labels: fixed threshold -> volatility-scaled threshold

Legacy:

```text
future_return_h > +0.5% -> UP
future_return_h < -0.5% -> DOWN
else -> NEUTRAL
```

New default:

```text
current_horizon_vol = returns.rolling(60).std().shift(1) * sqrt(horizon)
threshold_h = clip(0.30 * current_horizon_vol, 0.003, 0.040)

future_return_h > +threshold_h -> UP
future_return_h < -threshold_h -> DOWN
else -> NEUTRAL
```

Relevant config / CLI:

```text
direction_label_mode = vol_scaled
direction_vol_window = 60
direction_vol_k = 0.30
direction_min_abs_threshold = 0.003
direction_max_abs_threshold = 0.040
```

### 2. Direction Strength label: tiny trend changes filtered

Legacy:

```text
trend_delta > 0  -> UP_STRENGTHENING candidate
trend_delta < 0  -> DOWN_STRENGTHENING candidate
```

New default:

```text
trend_delta > +0.25 -> UP_STRENGTHENING candidate
trend_delta < -0.25 -> DOWN_STRENGTHENING candidate
```

Because the default method is `score_delta`, this effectively requires a meaningful trend-score change rather than a tiny positive/negative change.

Relevant CLI:

```text
--direction-strength-eps 0.25
```

### 3. Prediction direction decision: fixed margin -> rolling rank confirmation

Legacy:

```text
direction_score = prob_up - prob_down
abs(direction_score) >= 0.05 -> direction signal
```

New default:

```text
abs(direction_score) >= 0.03
and abs(direction_score) is high enough versus its own rolling history
```

Relevant config / CLI:

```text
use_dynamic_direction_margin = True
direction_margin_window = 756
direction_margin_min_periods = 252
direction_margin_rank_threshold = 0.65
direction_margin_abs_floor = 0.03
```

The legacy decision is preserved in:

```text
pred_direction_legacy_fixed_margin
```

### 4. Reporting-only probability rank diagnostics

The model also adds:

```text
prob_high_vol_rank_756
pred_risk_ranked
pred_risk_rank_source
```

This does not replace legacy `pred_risk`; it provides rank-based diagnostics for later policy work.

## Run command

```bat
run_v8_6_41_model_label_fixed_compare.bat
```

Or directly:

```bat
python xgb_recency_weighted_v8_6_41_model_label_fixed.py ^
  --asset-list QQQ,SPY,AAPL,SOXX,NVDA ^
  --speed-profile fast ^
  --h10-down-only ^
  --disable-tier2 ^
  --allocation-downrisk-weight 0 ^
  --result-dir results_v8_6_41_model_label_fixed_compare ^
  --transaction-cost-rate 0.001 ^
  --execution-lag-days 1 ^
  --direction-label-mode vol_scaled ^
  --direction-vol-window 60 ^
  --direction-vol-k 0.30 ^
  --direction-min-abs-threshold 0.003 ^
  --direction-max-abs-threshold 0.040 ^
  --direction-strength-eps 0.25
```

## What to compare

Compare against v8.6.40b_clean:

```text
multi_asset_summary.csv
*_summary.json
*_probability_bins.csv
*_threshold_diagnostics.csv
*_predictions.csv
```

Primary checks:

```text
1. Direction label distribution by ticker
2. prob_up/prob_down calibration or bins
3. Up/down strengthening class distribution
4. Strategy CAGR/MDD/Sharpe after cost
5. QQQ/SPY degradation risk
6. SOXX/NVDA improvement or MDD explosion risk
```

## Important warning

This changes the classification labels, so policy1b validation results from v8.6.40b predictions are no longer automatically valid. After running this version, policy1b must be re-tested against the new predictions.
