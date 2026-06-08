@echo off
setlocal
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
endlocal
