# Portfolio Regime Advisor Backend v0.3.3 — Recency-Weighted Training

## 목적

v0.3.2에서 남아 있던 두 가지 검증 지적을 처리한다.

1. walk-forward가 pure expanding window라 v8.6.41의 recency-weighted 철학과 완전히 일치하지 않는 문제
2. activation gate가 ROC-AUC worst-fold만 반영하고 PR-AUC/Brier/positive-rate worst-fold 안정성을 충분히 보지 않는 문제

v0.3.3은 운영 기본값을 변경하지 않는다. 기본 운영은 여전히 `prediction_file` + `v8.6.41_label_fixed`다. 새 runtime candidate는 CANDIDATE로만 등록된다.

---

## 핵심 변경

### 1. Recency sample weight 추가

`TrainingService`에 horizon별 half-life 기반 sample weight를 추가했다.

기본값:

```text
5D  -> 126 rows
10D -> 252 rows
20D -> 504 rows
```

가중치 공식:

```text
weight_i = 0.5 ** ((n - 1 - i) / half_life)
```

가중치는 평균 1로 정규화된다. 최신 row가 가장 큰 가중치를 받고, half-life만큼 과거 row는 최신 row 대비 약 절반의 비정규화 가중치를 갖는다.

### 2. Expanding + recency weighting을 기본값으로 유지

기본 walk-forward 구조는 leak-safe expanding window를 유지한다.

```text
fold 1: train[0:t1] -> valid[t1:v1]
fold 2: train[0:t2] -> valid[t2:v2]
fold 3: train[0:t3] -> valid[t3:v3]
```

다만 train 내부에서 오래된 데이터는 낮은 sample_weight를 받는다. 즉, 데이터 누출 방지와 최근 regime 반영을 동시에 겨냥한다.

### 3. Rolling walk-forward 옵션 추가

실험용으로 `walk_forward_mode="rolling"`과 `rolling_train_rows`를 추가했다.

```json
{
  "walk_forward_mode": "rolling",
  "rolling_train_rows": 1260
}
```

기본값은 여전히 `expanding`이다.

### 4. TrainingRequest 확장

`POST /training/retrain` body에 아래 필드를 추가했다.

```json
{
  "sample_weight_mode": "recency",
  "walk_forward_mode": "expanding",
  "rolling_train_rows": null,
  "recency_half_life_by_horizon": {
    "5D": 126,
    "10D": 252,
    "20D": 504
  }
}
```

### 5. Activation gate 보강

v0.3.2 조건에 아래 조건을 추가했다.

```text
brier_worst <= 0.45
positive_rate_worst within [0.01, 0.99]
pr_auc_lift >= 1.02
pr_auc_lift_worst >= 0.95
```

`pr_auc_lift = pr_auc / positive_rate`로 계산한다. 이는 PR-AUC가 단순 positive-rate baseline보다 실질적으로 나은지 보기 위한 최소 방어 조건이다.

### 6. Metadata/manifest 저장

후보 모델 metadata와 manifest에 아래 정보가 저장된다.

```text
training_method = recency_weighted_candidate_training_v0_3_3
training_config.sample_weight_mode
training_config.walk_forward_mode
training_config.rolling_train_rows
training_config.recency_half_life_by_horizon
```

---

## 권장 실행 예시

### 기본 권장값: expanding + recency

```json
{
  "tickers": ["QQQ", "SPY", "AAPL", "SOXX", "NVDA"],
  "horizons": ["5D", "10D", "20D"],
  "train_start": "2020-01-01",
  "train_end": "2026-05-07",
  "data_source": "cache",
  "market": "US",
  "model_version": "candidate_runtime_v0_3_3_recency_test",
  "sample_weight_mode": "recency",
  "walk_forward_mode": "expanding"
}
```

### rolling window 실험

```json
{
  "tickers": ["QQQ", "SPY", "AAPL", "SOXX", "NVDA"],
  "horizons": ["5D", "10D", "20D"],
  "train_start": "2020-01-01",
  "train_end": "2026-05-07",
  "data_source": "cache",
  "market": "US",
  "model_version": "candidate_runtime_v0_3_3_rolling1260_test",
  "sample_weight_mode": "recency",
  "walk_forward_mode": "rolling",
  "rolling_train_rows": 1260
}
```

---

## 해석 주의

- v0.3.3은 candidate training을 v8.6.41 철학에 더 맞춘 것이지, 수익성을 보장하지 않는다.
- activation gate 통과는 최소 차단선일 뿐 최종 운영 승인 기준이 아니다.
- prediction_file vs live 비교와 portfolio OOS 검증은 여전히 필수다.
