# v8.6.40 개선 전략 보고서
**대상 모델**: `xgb_recency_weighted_v8_6_39` (QQQ 기준)  
**분석 데이터**: predictions (3,357행), annual_returns, regime_analysis, threshold_diagnostics, probability_bins, turnover_diagnostics, drawdown_episodes, feature_importance

---

## 핵심 요약

v8.6.39는 구조적으로 가장 균형 잡힌 버전이지만, 실측 데이터를 직접 분석한 결과 **4개의 명확한 수정 대상**이 확인됐다. 아래 문제들은 추정이 아니라 파일에서 직접 계측된 수치로 뒷받침된다.

---

## P0 (최우선): mid_trend_score 버그 — 전수 0 고착

### 실측 근거
```
pred_df['mid_trend_score'].value_counts()
→ 0: 3357  (전체 100%)

pred_df['mid_trend_state'].value_counts()
→ BEAR: 3357  (전체 100%)
```
전체 백테스트 기간(2013~2026)에 걸쳐 단 한 번도 BULL/NEUTRAL이 없다. 명백한 버그.

### 원인 (소스코드 확인)
`compute_mid_trend_score(row)`가 읽어야 하는 6개 컬럼:
```
return_60d, return_120d, price_ma_60_gap,
price_ma_120_gap, ma_gap_20_60, trend_slope_60
```
이 컬럼들이 **prediction DataFrame에 전혀 존재하지 않는다.**

```python
# predictions.csv 컬럼 점검 결과:
trend_cols = ['return_60d', 'return_120d', 'price_ma_60_gap',
              'price_ma_120_gap', 'ma_gap_20_60', 'trend_slope_60']
→ 전부 missing
```

`_row_float(row, col, default=0.0)` 함수는 키 부재 시 `0.0`을 반환하므로,
모든 check가 `0.0 > 0.0 = False` → `score = 0` → `state = "BEAR"`.

### 실제 피해
`apply_direction_strength_specialist_policy()` 내에서:
```python
trend_score, trend_state = compute_mid_trend_score(row)
```
이 결과가 BULL bonus overlay 조건으로 사용된다. BULL 판정이 불가능하므로
**상승장 참여율 향상 로직이 3,357일 전 기간 동안 단 한 번도 작동하지 않았다.**

ConflictResolver 분석 결과:
```
midtrend_bear_high_stock conflicts: 1,908건 (56.8%)
→ mid_trend=BEAR인데 stock >= 80%인 모순 상황이 절반 이상
```

### 수정

**Step 1.** `run_walk_forward()` 내 prediction row 조립 블록 마지막에 추가:
```python
# raw feature context 보존 (mid_trend 등 policy 계산에 필요)
MID_TREND_FEATURES = [
    "return_60d", "return_120d", "price_ma_60_gap",
    "price_ma_120_gap", "ma_gap_20_60", "trend_slope_60",
    "drawdown_60", "realized_vol_60",
]
for col in MID_TREND_FEATURES:
    if col not in out and col in feature_cols:
        val = all_df.iloc[pos].get(col, np.nan)
        if not pd.isna(val):
            out[col] = float(val)
prediction_rows.append(out)
```

**Step 2.** `compute_mid_trend_score()` 안전 버전으로 교체:
```python
def compute_mid_trend_score(row: pd.Series) -> Tuple[int, str]:
    available = [c for c in MID_TREND_FEATURES[:6]
                 if c in row.index and not pd.isna(row.get(c))]
    if len(available) < 3:
        return 0, "UNKNOWN"   # ← "BEAR"가 아닌 "UNKNOWN"
    checks = [float(row[c]) > 0.0 for c in available]
    score = sum(checks)
    total = len(checks)
    if score >= max(4, round(total * 0.67)):
        return score, "BULL"
    elif score <= round(total * 0.33):
        return score, "BEAR"
    else:
        return score, "NEUTRAL"
```

UNKNOWN 상태에서는 overlay를 중립으로 처리 (BULL bonus 적용 안 함,
BEAR cut도 적용 안 함). 이것이 "0.0 → BEAR → overlay 막힘"보다 훨씬 안전하다.

---

## P0: Down-risk allocation 영향 확인 및 명시적 제거

### 실측 근거
`threshold_diagnostics.csv`에서 down_strength_score:

| threshold | precision | recall | F1 |
|-----------|-----------|--------|----|
| 0.10 | 0.173 | 0.946 | 0.292 |
| 0.30 | 0.204 | 0.641 | 0.309 |
| 0.50 | 0.137 | 0.149 | 0.143 |
| 0.70 | 0.090 | 0.012 | 0.021 |

어떤 threshold를 써도 precision이 0.20 수준이다.
base rate (DOWN_STRENGTHENING 비율) = 577 / 3357 = **17.2%**이므로,
threshold 0.50에서 precision 13.7%는 **base rate보다 낮다** — 역방향.

`overall_risk_down_weight = 0.0`으로 이미 비활성이지만,
`allocation_downrisk_score()` 호출 경로가 일부 남아 있고
`apply_direction_strength_specialist_policy()`에서 `pdn_alloc = ph`로
수동 치환하는 구조가 의도의 명확성을 해친다.

### 수정
`apply_allocation()` 내에서:
```python
# BEFORE
pdn_raw = float(row.get("prob_down", row.get("prob_down_risk", 0.0)))
pdn = allocation_downrisk_score(ph, pdn_raw, cfg)

# AFTER — [v8.6.40] down-risk allocation 완전 제거
pdn = 0.0
```

`use_multi_branch_downrisk = False`로 Config 변경 권고 (3개 branch F1 < 0.31).
Down-risk 모델은 진단 출력용으로는 유지, allocation path에서만 단절.

---

## P0: allocation_trace 추가 + ConflictResolver

### 현황
QQQ latest.json에서 stock=86%, mid_trend=BEAR(버그), up_bonus=0.0이 공존한다.
어떤 경로로 86%가 나왔는지 trace가 없어 디버깅 불가.

### 수정
`apply_direction_strength_specialist_policy()` meta dict에 추가:
```python
meta["allocation_trace"] = {
    "base_stock_from_highvol": base_stock,
    "up_bonus": up_bonus,
    "trend_cut": 0.0,
    "drawdown_cut": 0.0,
    "no_trade_band_hold": False,  # apply_allocation에서 덮어쓰기
    "final_stock": stock,
    "mid_trend_state": trend_state,
    "prob_high_vol": ph,
    "up_strength_score": pus_score,
    "offensive_tier": offensive_tier,
}
```

`apply_allocation()` 말미에 ConflictResolver 호출:
```python
conflicts = check_allocation_conflict(w[0], ph, mid_trend_state)
for c in conflicts:
    if c["level"] == "HARD_CAP":
        w = _redistribute_after_stock_change(min(w[0], 0.70), w)
```

---

## P1: no_trade_band 예외 조건

### 실측 근거

hold_reason별 성과 분석:

| hold_reason | count | ann_return | BH_gap | avg_stock |
|-------------|-------|------------|--------|-----------|
| not_rebalance_day | 2,269 | 13.69% | **-7.59%** | 69.4% |
| no_trade_band | 506 | 8.65% | **+14.38%** | 64.4% |
| strong_offensive_override | 331 | 44.85% | -2.64% | 98.3% |
| scheduled | 161 | 5.69% | **-22.55%** | 68.0% |
| emergency | 40 | 26.09% | -85.95% | 48.5% |

`no_trade_band` 구간에서 BH_gap = +14.38%라는 결과는 흥미롭다 — 이 구간 자체에서는
방어가 잘 됐다는 뜻이다. 그러나 `scheduled` 구간의 BH_gap = -22.55%는 심각하다.
10일 주기 리밸런싱에서 risk 급등 직후 타이밍 문제가 발생하고 있음을 시사한다.

`not_rebalance_day` 69%가 13.7% 수익이지만 BH 대비 -7.6pp 뒤처진다.
이 구간에서 risk 변화에 반응하지 못하는 것이 기회비용의 핵심 원인이다.

### 수정
```python
# prob_high_vol이 0.70 돌파 시 즉시 리밸런싱
if ph >= 0.70 and ph_prev < 0.70:
    rebalance_due = True
    hold_reason = "hv_threshold_override"

# ph 급등 (+0.15 이상)
if ph - ph_prev >= 0.15 and ph >= 0.55:
    rebalance_due = True
    hold_reason = "hv_surge_override"

# mid_trend BULL↔BEAR 전환 (버그 수정 후 활성화)
if {trend_state, trend_state_prev} == {"BULL", "BEAR"}:
    rebalance_due = True
    hold_reason = "trend_flip_override"
```

근거: threshold_diagnostics에서 prob_high_vol >= 0.70의 precision = 0.811, recall = 0.466.
이 임계값 이상에서는 진짜 고변동 확률이 81% — 즉시 리밸런싱이 정당하다.

---

## P1: UpStrength score 최솟값 필터 (0.40)

### 실측 근거

`probability_bins.csv` (prob_up_strengthening_score 구간별):

| score 구간 | actual UP rate | ann_return |
|-----------|---------------|------------|
| 0.0~0.1 | 2.8% | 11.1% |
| 0.1~0.2 | 9.4% | 11.6% |
| 0.2~0.3 | 12.8% | **-0.13%** |
| 0.3~0.4 | 28.4% | **3.8%** |
| **0.4~0.5** | **39.5%** | **41.5%** ← 임계점 |
| 0.5~0.6 | 61.3% | 56.8% |
| 0.6~0.7 | 59.8% | 46.4% |
| 0.7~0.8 | 46.6% | **-10.2%** ← 역전 주의 |
| 0.8~0.9 | 92.7% | 80.6% |

0.40 미만 구간은 실질 알파가 없다 (ann_return -0.13% ~ 3.8%).
0.40~0.50 이상에서 급격한 전환점이 존재한다.
0.70~0.80 구간의 역전 현상은 EWMA 스무딩 후 score 과평가 가능성을 시사한다.

### 수정
```python
# apply_direction_strength_specialist_policy() 상단
up_score_min = float(getattr(cfg, "up_strength_allocation_score_minimum", 0.40))
if pus_score < up_score_min:
    tier1_signal = tier2_signal = tier3_signal = full_stock_signal = False
    offensive_tier = 0
    # base_weight_from_vol_probability 결과만 사용
```

이 변경으로 0.3~0.4 구간에서 잘못된 tier 신호가 발생하던 것을 차단한다.

---

## P2: NORMAL regime 성과 저조 분석

### 실측 근거

```
NORMAL regime:
  count: 595 (17.7%)
  ann_return: 9.20%
  avg_stock: 82.4%
  avg_prob_high_vol: 0.387
```

CUSTOM regime(44.6%, ann_return 20.4%)과 비교하면 NORMAL은 심각히 저조하다.
avg_stock이 82.4%로 충분히 공격적인데도 수익이 낮다는 것은
mid_trend BEAR 버그로 BULL bonus가 차단됐기 때문일 가능성이 높다.

P0 버그 수정 후 NORMAL regime 성과가 크게 개선될 것으로 예상된다.

---

## P2: UpStrength 5D 제거 확인

이미 `up_strength_disable_5d_trigger = True`로 비활성화되어 있다.
다만 5D 관련 계산 코드(모델 학습, 확률 산출)는 여전히 실행되어 compute 낭비가 있다.

`use_multi_strength_horizons`에서 5를 제거하면:
- training 시간 단축
- prediction row에서 `prob_up_strengthening_5d` 컬럼 불필요
- 코드 명확성 향상

단, 5D label은 진단 목적으로 유지하는 것이 좋다
(어느 구간에서 5D만 신호가 나오는지 파악에 유용).

---

## P3: asset_class별 policy 분리 준비

현재 SPY, QQQ, NVDA에 동일한 `stock_at_risk` table을 적용한다.
이것이 Buy & Hold 대비 기회비용의 구조적 원인이다:
- NVDA gap = -20.3%p: 고변동 성장주에 broad index 방어 로직 적용
- SOXX gap = -8.4%p: sector ETF에 과도한 방어

### 설계안
```python
ASSET_CLASS_CONFIG = {
    "broad_index":    {"target_mdd": 0.27, "hv_threshold_override": 0.70},
    "sector_etf":     {"target_mdd": 0.36, "hv_threshold_override": 0.74},
    "mega_cap":       {"target_mdd": 0.32, "hv_threshold_override": 0.72},
    "high_vol_growth":{"target_mdd": 0.52, "hv_threshold_override": 0.82},
}

ASSET_CLASS_MAP = {
    "QQQ": "broad_index", "SPY": "broad_index",
    "SOXX": "sector_etf", "SMH": "sector_etf",
    "AAPL": "mega_cap", "MSFT": "mega_cap",
    "NVDA": "high_vol_growth", "TSLA": "high_vol_growth",
}
```

target_mdd를 `base_weight_from_vol_probability()`의 lower floor로 연결:
- NVDA의 경우 극단 고변동 시에도 주식 비중 최솟값을 20%로 유지 (현재 30% 고정값)
- SPY는 EXTREME_RISK 시 주식 20~25% 유지 (현재 30% 고정 → 충분)

---

## P3: Dead Code 제거 목록

다음을 v8.6.40에서 제거 또는 deprecated 처리:

| 대상 | 근거 | 조치 |
|------|------|------|
| `PortfolioPolicyModel` 관련 5개 함수 | `portfolio_model_enabled=false` 고착 | 제거 |
| `TierWeightOptimizer` 관련 3개 함수 | 본체 과최적화 위험 | research script 분리 |
| `ConditionSearch` 관련 2개 함수 | 동일 | research script 분리 |
| `apply_return_seeking_policy()` | 미사용 policy mode | deprecated |
| `apply_aggressive_dynamic_policy()` | 미사용 | deprecated |
| `prob_down_price_trend/volume/volatility` 컬럼 | down-risk 분리 후 불필요 | 진단 전용 |
| `gate_rolling_optimization` 분기 | 실효성 검증 미완 | 비활성 표시 |

제거 전 각 함수가 실제로 호출되는지 grep 확인 권장:
```bash
grep -n "run_portfolio_policy_model\|run_tier_weight_optimizer\|run_condition_search" *.py
```

---

## P4: 검증 구조 보완

### CPCV / Deflated Sharpe Ratio 필요성

현재 v8.6.39는 walk-forward만 사용한다.
버전 반복(v8.6.x ~ v8.8.x)이 20회 이상 축적되면 best result 선택 편향이 발생한다.

최소 조치:
```python
# fold별 Sharpe 안정성 점검
annual_df['sharpe_est'] = annual_df['strategy_net'] / annual_df['strategy_net'].expanding().std()
fold_sharpe_std = annual_df['sharpe_est'].std()  # < 0.5이면 안정적
```

DSR 계산 (Marcos Lopez de Prado, 2014):
```
DSR = SR * sqrt(1 - skewness(r)*SR + (kurtosis(r)-1)/4 * SR^2)
      × N_comparisons^(-0.5)  # multiple testing 조정
```

---

## 우선순위별 실행 계획

### 즉시 적용 (P0) — 소요: 1~2시간

1. `run_walk_forward()`: prediction row에 raw feature 6개 보존
2. `compute_mid_trend_score()`: UNKNOWN 반환 추가
3. `apply_allocation()`: `pdn = 0.0` 명시
4. meta dict에 `allocation_trace` 추가
5. ConflictResolver 경고 로깅

**기대 효과**: mid_trend BEAR 고착 해소 → NORMAL regime BULL bonus 복원
→ 2019, 2020, 2023 같은 강한 상승년도의 gap 일부 회복.

### 단기 (P1) — 소요: 반나절

6. no_trade_band 예외 조건 (hv >= 0.70 돌파, surge >= 0.15)
7. UpStrength score 최솟값 0.40 필터
8. `hold_reason_diagnostics.csv` 출력 추가
9. `auto_review.md` 자동 생성

### 중기 (P2) — 소요: 1~2일

10. Dead code deprecated/제거
11. Down-risk 학습 코드 비활성화 (`use_multi_branch_downrisk = False`)
12. Config 6개 클래스로 분리 (CoreConfig/ModelConfig/LabelConfig/AllocationConfig/DiagnosticsConfig/ExperimentConfig)
13. `latest_context_backtest` 추가 (현재 조건과 유사한 과거 구간의 5/10/20일 성과)

### 장기 (P3/P4) — 소요: 3~5일

14. asset_class별 policy 분리 (4개 클래스)
15. Relative strength feature 추가 (target/SPY 20D, 60D)
16. CPCV 또는 DSR 계산 추가
17. 코드 모듈화 (8개 파일 분리)

---

## 수정 후 예상 성과 (QQQ 기준)

| 지표 | v8.6.39 현재 | P0 수정 후 예상 | 근거 |
|------|-------------|----------------|------|
| CAGR | 16.46% | 17.5~18.5% | mid_trend BULL bonus 복원으로 NORMAL regime +4~8%p 개선 |
| MDD | -27.04% | -25~-28% | 큰 변화 없음 (HighVol head는 유효하므로) |
| Sharpe | 1.24 | 1.25~1.35 | 수익 개선 + volatility 유지 |
| BH gap | -3.99%p | -2~-3%p | 상승장 참여율 소폭 개선 |

> 주의: 이 예상은 mid_trend score가 실제로 BULL을 올바르게 식별했을 때만 유효하다.
> 버그 수정 후 재실행하여 실측값과 대조 필요.

---

## 자기 비판

1. P0 버그 수정의 효과가 얼마나 클지는 실행 전에는 정확히 알 수 없다.
   mid_trend BULL bonus의 구체적인 크기는 `apply_direction_strength_specialist_policy()`
   코드 내 `up_bonus` 계산 로직에 달려 있다.

2. `no_trade_band` 구간의 BH_gap = +14.38%는 직관과 다르다.
   (이 구간에서 방어가 오히려 잘 됐다는 의미) — scheduled 구간(-22.55%)과 대비된다.
   따라서 no_trade_band 완화가 오히려 해가 될 가능성도 있다.
   P1 수정은 hv >= 0.70 돌파에만 제한적으로 적용하는 것이 안전하다.

3. UpStrength 0.7~0.8 구간의 ann_return = -10.2% 역전은 EWMA 스무딩 효과일 수 있다.
   (score가 높게 평가된 시점이 실제로는 peak 근처인 경우)
   이것은 별도 분석이 필요하다.
