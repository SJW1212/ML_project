"""Constants for the locked v8.6.41 local production pipeline."""

MODEL_VERSION = "v8.6.41_model_label_fixed"
SOURCE_TAG = "xgb_recency_weighted_v8_6_41_model_label_fixed"
DEFAULT_ASSETS = "QQQ,SPY,AAPL,SOXX,NVDA"

CANCELLED_LAYERS = [
    "drawdown_loss_guard",
    "confidence_weighted_v1",
    "sensitive_v2",
    "accuracy_benchmark_v3",
    "logit_calibrated_v4",
    "v8.6.42_adaptive_controls",
    "v0.3.x_live_runtime_context_gate_ensemble",
]

EXCLUDED_PRODUCT_SCOPE = [
    "realtime_order_execution",
    "automated_trading",
    "database_storage",
    "user_account_portfolio_storage",
    "notifications",
    "pixso_screen_design_mapping",
    "realtime_market_streaming",
]

# UI classification thresholds only. These do not change the native v8.6.41
# portfolio allocation columns in prediction files.
SIGNAL_THRESHOLDS = {
    "high_risk_prob_high_vol": 0.65,
    "watch_prob_high_vol": 0.40,
    "watch_prob_down_strengthening_score": 0.50,
    "up_strength_score": 0.45,
    "down_strength_score": 0.45,
}

PRED_RISK_HIGH_KEYWORDS = {"고변동", "위험", "HIGH_VOL", "RISK_OFF"}
PRED_DIRECTION_UP_KEYWORDS = {"상승", "UP"}
PRED_DIRECTION_DOWN_KEYWORDS = {"하락", "DOWN"}
