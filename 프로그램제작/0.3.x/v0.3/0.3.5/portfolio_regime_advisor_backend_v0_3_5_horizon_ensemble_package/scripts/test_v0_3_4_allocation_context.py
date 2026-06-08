from __future__ import annotations

import pandas as pd

from backend.app.portfolio.allocation_policy_engine import AllocationPolicyEngine


def test_feature_join_and_context_overlay():
    pred = pd.DataFrame([
        {"Date": "2026-05-07", "ticker": "QQQ", "prob_high_vol": 0.31, "prob_up_strengthening_score": 0.1, "prob_down_strengthening_score": 0.1},
    ])
    feat = pd.DataFrame([
        {"Date": "2026-05-07", "ticker": "QQQ", "ctx_spy_drawdown_252": -0.20, "ctx_vix_z_63": 0.5},
    ])
    engine = AllocationPolicyEngine()
    out = engine.apply_dataframe(pred, preserve_existing=False, feature_df=feat)
    row = out.iloc[0]
    assert row["allocation_policy_reason"] == "CTX_SPY_DRAWDOWN_WATCH_POLICY"
    assert abs(float(row["stock_weight"]) - 0.74) < 1e-9
    assert "ctx_spy_drawdown_252" in out.columns


def test_missing_context_is_not_silent_zero():
    pred = pd.DataFrame([
        {"Date": "2026-05-07", "ticker": "QQQ", "prob_high_vol": 0.31, "prob_up_strengthening_score": 0.1, "prob_down_strengthening_score": 0.1},
    ])
    engine = AllocationPolicyEngine()
    out = engine.apply_dataframe(pred, preserve_existing=False)
    assert out.iloc[0]["allocation_policy_reason"] == "NORMAL_PARTICIPATION_POLICY"


if __name__ == "__main__":
    test_feature_join_and_context_overlay()
    test_missing_context_is_not_silent_zero()
    print("v0.3.4 allocation context tests passed")
