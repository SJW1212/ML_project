"""Constants for the locked v8.6.41 UI-ready production pipeline."""

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
]
