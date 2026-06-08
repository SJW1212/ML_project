# Portfolio Regime Advisor v0.3.5

## 목적

v0.3.4는 head-level gate와 selective inference를 추가했지만, live inference에서 선택된 horizon 하나만 대표 확률로 사용했다. v0.3.5는 `highvol`, `up_strength`, `down_strength` 각각에 대해 5D/10D/20D horizon probability를 gate-aware ensemble로 결합한다.

## 핵심 변경

### 1. Horizon Ensemble 추가

신규 파일:

```text
backend/app/model/horizon_ensemble.py
```

추가 출력:

```text
prob_high_vol_ensemble
prob_up_strengthening_ensemble
prob_down_strengthening_ensemble
highvol_state
up_strength_state
down_strength_state
horizon_ensembles
ensemble_used_heads
ensemble_fallback_heads
```

### 2. Family별 기본 앙상블 구조

```text
highvol:
  highvol_5D  weight 0.35
  highvol_10D weight 0.45
  highvol_20D weight 0.20

up_strength:
  up_strength_5D  weight 0.50
  up_strength_10D weight 0.40
  up_strength_20D weight 0.10

down_strength:
  down_strength_5D  weight 0.45
  down_strength_10D weight 0.45
  down_strength_20D weight 0.10
```

20D strength head는 기존 실험에서 불안정한 경우가 많았기 때문에 가중치를 낮게 둔다. PASS/UNCERTAIN/FAIL gate에 따라 실제 weight가 다시 조정된다.

### 3. Gate-aware weight

```text
PASS      -> base weight × 1.00
UNCERTAIN -> base weight × 0.50
FAIL      -> base weight × 0.00
```

모든 head가 FAIL이면 neutral probability 0.50을 반환한다.

### 4. InferenceService 수정

v0.3.4:

```text
selected horizon head만 selective inference 적용
prob_high_vol = selected highvol horizon
prob_up_strengthening_score = selected up horizon
prob_down_strengthening_score = selected down horizon
```

v0.3.5:

```text
모든 horizon head에 selective inference 적용
각 family별 horizon ensemble 생성
prob_high_vol = prob_high_vol_ensemble
prob_up_strengthening_score = prob_up_strengthening_ensemble
prob_down_strengthening_score = prob_down_strengthening_ensemble
```

selected horizon 값은 audit/backward compatibility 용도로 유지한다.

### 5. State 분류

HighVol:

```text
NORMAL_VOL
ACUTE_SPIKE
RISING_VOL
CONFIRMED_HIGH_VOL
PERSISTENT_HIGH_VOL
VOL_COMPRESSION
MIXED_VOL
```

UpStrength:

```text
ACUTE_UP_STRENGTH
CONFIRMED_UP_STRENGTH
MILD_UP_STRENGTH
NO_UP_STRENGTH
MIXED_UP_STRENGTH
```

DownStrength:

```text
CONFIRMED_DOWN_STRENGTH
ACUTE_DOWN_SPIKE
MILD_DOWN_STRENGTH
NO_DOWN_STRENGTH
MIXED_DOWN_STRENGTH
```

### 6. 테스트

추가:

```text
scripts/test_v0_3_5_horizon_ensemble.py
scripts/test_v0_3_5_runtime_smoke.py
```

통과 확인:

```text
v0.3.5 horizon ensemble tests passed
v0.3.4 head gate tests passed
v0.3.4 allocation context tests passed
v0.3.5 runtime smoke test passed
compile_ok
```

## 운영상 주의

v0.3.5는 구조 개선 패치다. 전체 activation gate를 통과하지 못한 candidate를 강제로 ACTIVE 전환하면 안 된다. v0.3.5는 다음 비교 실험을 위한 기반이다.

필수 비교:

```text
A. v0.3.4 selected-horizon 방식
B. v0.3.5 family ensemble 방식
C. v0.3.5 ensemble + rolling750
D. v0.3.5 ensemble + walk_forward_splits=5
E. prediction_file v8.6.41 baseline 대비 allocation/OOS 비교
```
