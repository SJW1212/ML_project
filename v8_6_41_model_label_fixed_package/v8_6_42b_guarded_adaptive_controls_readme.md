# v8.6.42b_guarded_adaptive_controls

## Purpose
This patch fixes the first v8.6.42 adaptive-controls implementation after result review.

## Main fixes
1. Blocks same-date cross-sectional context leakage.
2. Enforces `context_global_min_rows` before using global fallback statistics.
3. Makes context adjustment asymmetric and conservative:
   - positive max: 0.03
   - negative max: 0.08
   - positive threshold: 0.20
   - negative threshold: -0.15
4. Disables positive context adjustment for `broad_index` by default.
5. Disables broad-index upside overlay by default.
6. Reduces broad-index max stock guardrail from 1.00 to 0.86.

## Run
```bat
run_v8_6_42b_guarded_adaptive_controls_resim.bat
```

Or:
```bat
python v8_6_42b_guarded_adaptive_controls_resim.py ^
  --input-dir . ^
  --out-dir results_v8_6_42b_guarded_adaptive_controls_from_label_fixed ^
  --asset-list QQQ,SPY,AAPL,SOXX,NVDA ^
  --source-tag xgb_recency_weighted_v8_6_41_model_label_fixed ^
  --transaction-cost-rate 0.001 ^
  --rank-windows 504,756,1008 ^
  --rank-min-periods 252 ^
  --context-window 756 ^
  --rebalance-every 5 ^
  --no-trade-band 0.12 ^
  --max-weight-change-per-rebalance 0.20
```

## Interpretation
This patch is not the final version. It confirms that the original v8.6.42 adaptive-controls layer was too aggressive and had context-history defects. It should be used to test safer controls before integrating into the main model.
