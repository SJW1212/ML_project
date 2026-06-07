"""
v8.6.40 Integration Guide
=========================
v8.6.39 소스코드에 패치를 적용하는 구체적인 위치와 방법.

각 수정은 독립적이며, 우선순위 순서대로 적용 가능.

## 최우선 (P0) — mid_trend context 버그

### 수정 위치: run_walk_forward() 내 prediction row 조립 블록
```python
# ─── BEFORE (v8.6.39) ───────────────────────────────────
out.update({
    ...
    "stock_next_return": float(all_df.iloc[pos]["stock_next_return"]),
    "bond_next_return": float(all_df.iloc[pos]["bond_next_return"]),
    "cash_next_return": float(all_df.iloc[pos]["cash_next_return"]),
})
prediction_rows.append(out)

# ─── AFTER (v8.6.40) ────────────────────────────────────
out.update({
    ...
    "stock_next_return": float(all_df.iloc[pos]["stock_next_return"]),
    "bond_next_return": float(all_df.iloc[pos]["bond_next_return"]),
    "cash_next_return": float(all_df.iloc[pos]["cash_next_return"]),
})
# [v8.6.40 PATCH P0] raw feature context 보존
out = patch_prediction_row_with_features(out, all_df.iloc[pos], feature_cols)
prediction_rows.append(out)
```

### 수정 위치: compute_mid_trend_score() 함수 교체
```python
# ─── BEFORE (v8.6.39) ───────────────────────────────────
def compute_mid_trend_score(row: pd.Series) -> Tuple[int, str]:
    checks = [
        _row_float(row, "return_60d") > 0.0,
        _row_float(row, "return_120d") > 0.0,
        _row_float(row, "price_ma_60_gap") > 0.0,
        _row_float(row, "price_ma_120_gap") > 0.0,
        _row_float(row, "ma_gap_20_60") > 0.0,
        _row_float(row, "trend_slope_60") > 0.0,
    ]
    score = int(sum(bool(x) for x in checks))
    if score >= 4:
        state = "BULL"
    elif score <= 2:
        state = "BEAR"
    else:
        state = "NEUTRAL"
    return score, state

# ─── AFTER (v8.6.40) ────────────────────────────────────
# → compute_mid_trend_score_safe()로 교체 (v8_6_40_patch_core.py 참조)
# 함수명을 유지하려면:
def compute_mid_trend_score(row: pd.Series) -> Tuple[int, str]:
    return compute_mid_trend_score_safe(row)  # from patch_core
```

---

## P0 — Down-risk allocation 제거 확인

### 수정 위치: apply_direction_strength_specialist_policy()
```python
# ─── BEFORE (v8.6.39) ───────────────────────────────────
pdn_alloc = ph  # 이미 ph로 치환되어 있음 (사실상 제거)

# ─── AFTER (v8.6.40) ────────────────────────────────────
# 명시적 제거 강제 (코드 의도 명확화)
pdn_alloc = allocation_downrisk_score_v40(ph, 0.0, cfg)  # always 0.0
```

### 수정 위치: apply_allocation() — pdn_raw 계산 부분
```python
# ─── BEFORE (v8.6.39) ───────────────────────────────────
pdn_raw = float(row.get("prob_down", row.get("prob_down_risk", 0.0)))
pdn = allocation_downrisk_score(ph, pdn_raw, cfg)

# ─── AFTER (v8.6.40) ────────────────────────────────────
pdn = 0.0  # [v8.6.40] Down-risk allocation 영향 완전 제거
```

---

## P0 — allocation_trace 추가

### 수정 위치: apply_direction_strength_specialist_policy() 반환 meta dict
```python
# ─── BEFORE (v8.6.39) ───────────────────────────────────
meta = {
    "policy_overlay": float(stock - base_stock),
    "mid_trend_score": trend_score,
    "mid_trend_state": trend_state,
    ...
}
return w, meta

# ─── AFTER (v8.6.40) ────────────────────────────────────
meta = {
    "policy_overlay": float(stock - base_stock),
    "mid_trend_score": trend_score,
    "mid_trend_state": trend_state,
    ...
    # [v8.6.40 PATCH] allocation_trace
    "allocation_trace": build_allocation_trace(
        base_stock=base_stock,
        up_bonus=float(up_bonus if 'up_bonus' in dir() else 0.0),
        trend_cut=float(cut if cut < 0 else 0.0),
        drawdown_cut=0.0,
        no_trade_band_hold=False,   # apply_allocation에서 채워짐
        final_stock=float(stock),
        mid_trend_state=trend_state,
        ph=ph,
        pus_score=pus_score,
        tier=offensive_tier,
        regime=regime,
    ),
}
return w, meta
```

---

## P0 — ConflictResolver 적용

### 수정 위치: apply_allocation() — 최종 w 결정 이후
```python
# ─── AFTER (v8.6.40) — apply_allocation() 내부 ───────────
# ... w, executed_regime, hold_reason 결정 이후 ...
conflicts = check_allocation_conflict(
    stock=float(w[0]),
    ph=ph,
    mid_trend_state=str(policy_meta.get("mid_trend_state", "UNKNOWN")),
)
if conflicts:
    # WARNING은 로깅만, HARD_CAP은 강제 상한
    for c in conflicts:
        if c["level"] == "HARD_CAP":
            capped_stock = min(float(w[0]), 0.70)
            w = _redistribute_after_stock_change(capped_stock, w)
            hold_reason = f"conflict_cap_{c['rule']}"
```

---

## P1 — no_trade_band 예외 조건

### 수정 위치: apply_allocation() — not rebalance_due 분기
```python
# ─── BEFORE (v8.6.39) ───────────────────────────────────
elif not rebalance_due:
    stale_gap = float(prev_w[0] - signal_w[0])
    stale_up_prob = ...
    stale_decay = (...)
    if stale_decay:
        ...
    else:
        w = prev_w

# ─── AFTER (v8.6.40) ────────────────────────────────────
elif not rebalance_due:
    # [v8.6.40 PATCH P1] no_trade_band 긴급 override
    ph_prev = float(pred_df.iloc[max(0, i-1)].get("prob_high_vol", ph)) if i > 0 else ph
    mt_prev = str(pred_df.iloc[max(0, i-1)].get("mid_trend_state", "UNKNOWN")) if i > 0 else "UNKNOWN"
    override, override_reason = should_override_no_trade_band(
        ph=ph, ph_prev=ph_prev,
        mid_trend_state=str(policy_meta.get("mid_trend_state", "UNKNOWN")),
        mid_trend_state_prev=mt_prev,
        drawdown_guard=0.0, drawdown_guard_prev=0.0,
        cfg=cfg,
    )
    if override:
        w = signal_w
        executed_regime = signal_regime
        hold_reason = f"no_trade_override_{override_reason[:30]}"
        trade_executed = True
    else:
        stale_gap = float(prev_w[0] - signal_w[0])
        ...
```

---

## P1 — UpStrength score 최솟값 필터

### 수정 위치: apply_direction_strength_specialist_policy() — tier1 조건
```python
# ─── AFTER (v8.6.40) ────────────────────────────────────
# [v8.6.40] score < 0.40인 경우 tier1/2/3 모두 비활성
up_score_min = float(getattr(cfg, "up_strength_allocation_score_minimum", 0.40))
if pus_score < up_score_min:
    # tier signal 모두 False로 강제
    tier1_signal = tier2_signal = tier3_signal = full_stock_signal = False
    offensive_tier = 0
```

근거:
    probability_bins.csv에서 score 0.2~0.3 구간:
        actual UP rate: 12.8%, ann_return: -0.13% (사실상 무알파)
    score 0.3~0.4 구간:
        actual UP rate: 28.4%, ann_return: 3.8% (미미)
    score 0.4~0.5 구간:
        actual UP rate: 39.5%, ann_return: 41.5% ← 급격한 전환점

---

## P2 — Dead branch 제거 목록

다음 클래스/함수는 v8.6.40에서 제거 또는 stub으로 교체 권고:

1. PortfolioPolicyModel 관련:
   - build_portfolio_policy_model()
   - run_portfolio_policy_model()
   - apply_portfolio_policy_model()
   - portfolio_policy_summary()
   - portfolio_candidate_weights()
   근거: portfolio_model_enabled=false 고정, 실제 allocation에 미기여

2. TierWeightOptimizer:
   - run_tier_weight_optimizer()
   - simulate_tier_weight_strategy()
   - _evaluate_tier_candidate()
   - _candidate_weight_grid()
   근거: 운용 모델 본체에서 과최적화 위험. research script로 분리

3. ConditionSearch:
   - run_condition_search()
   - make_condition_candidate_configs()
   근거: 동일

4. Down-risk 모델 (진단 목적으로만 유지 가능):
   - make_xgb_downrisk() → 진단용 deprecated 표시
   - build_downrisk_feature_sets() → 유지 (진단 출력에 사용)
   - apply_aggressive_dynamic_policy() → 미사용 시 제거
   - apply_return_seeking_policy() → 미사용 시 제거

5. apply_defensive_risk_policy() → 검토 후 결정
   (방어성은 base_weight_from_vol_probability로 이미 처리)

제거 방법:
    즉시 삭제 대신 함수 앞에 다음 decorator 추가:
    
    import warnings
    def deprecated(reason):
        def decorator(fn):
            def wrapper(*a, **k):
                warnings.warn(f"{fn.__name__} is deprecated: {reason}", DeprecationWarning, stacklevel=2)
                return fn(*a, **k)
            return wrapper
        return decorator

    @deprecated("v8.6.40: PortfolioPolicyModel removed from allocation path")
    def build_portfolio_policy_model(...):
        ...

---

## 검증 체크리스트

v8.6.40 패치 적용 후 다음을 확인:

[ ] mid_trend_score 분포 확인 (0 고착 해소)
    → pred_df['mid_trend_score'].value_counts() 에 0 이외 값 존재해야 함

[ ] mid_trend_state BULL 비율 > 30% (정상 시장에서)
    → pred_df['mid_trend_state'].value_counts(normalize=True)

[ ] allocation_trace가 latest.json에 존재

[ ] conflict 발생건수 출력 확인

[ ] hold_reason_diagnostics.csv 생성 확인

[ ] QQQ CAGR ≥ 16.46% (v8.6.39 기준선 유지 또는 개선)

[ ] QQQ MDD ≤ 27.04% (v8.6.39 기준선 유지)

[ ] QQQ Sharpe ≥ 1.20 (허용 범위)
"""

print("Integration guide loaded.")
