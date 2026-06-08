# STRUCTURE_CHANGES_V0_3_6

## Theme

v0.3.6 converts the strict ticker-level activation interpretation into a **Soft Family Gate + Confidence-Weighted Ensemble Allocation** layer.

v0.3.5 already combines 5D/10D/20D heads into horizon ensembles. However, a strict ticker-level gate still rejects the whole ticker if several heads fail. This is too harsh for practical allocation because a ticker can have usable highvol or down_strength information even when another family is weak.

## Main additions

### 1. SoftFamilyGate

New module:

```text
backend/app/model/soft_family_gate.py
```

Responsibilities:

```text
- evaluate highvol / up_strength / down_strength families separately
- convert head-level gate outputs into per-head confidence
- aggregate head confidence into family confidence
- shrink raw family ensemble probability toward neutral 0.5
- return raw probability, effective probability, confidence, status, warnings
```

### 2. Neutral shrinkage instead of hard exclusion

Formula:

```text
effective_prob = 0.5 + (raw_prob - 0.5) * family_confidence
```

This avoids the unsafe behavior of `prob * confidence`, which can wrongly push a weak positive signal below neutral.

### 3. Family statuses

```text
STRONG_PASS
PASS
UNCERTAIN
WEAK_FAIL
FAIL
INVERTED
```

`FAIL` is no longer automatically dropped. Only `INVERTED` gets confidence 0.

### 4. InferenceService integration

v0.3.6 now emits:

```text
raw_prob_high_vol_ensemble
raw_prob_up_strengthening_ensemble
raw_prob_down_strengthening_ensemble

prob_high_vol_ensemble                  # effective/shrunk
prob_up_strengthening_ensemble          # effective/shrunk
prob_down_strengthening_ensemble        # effective/shrunk

effective_prob_high_vol
effective_prob_up_strengthening
effective_prob_down_strengthening

family_gates
family_confidence
family_gate_status
```

Allocation uses the effective probabilities, not the raw ensemble probabilities.

## Why this is safer

```text
- useful partial signals are not discarded by ticker-level gate failure
- weak families contribute only a small neutral-shrunk signal
- obviously inverted signals are neutralized
- up_strength failures do not create aggressive overweight
- down_strength failures can still keep softened defense through highvol/drawdown context
```

## Current production stance

```text
v8.6.41 prediction_file remains production baseline.
v0.3.6 is an experimental runtime-candidate branch.
Do not promote v0.3.6 to production before OOS portfolio comparison.
```
