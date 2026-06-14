# Portfolio Regime Advisor Backend v0.3.3 Recency-Weighted Training

v0.3.3은 v0.3.2의 activation gate 강화 위에 recency-weighted candidate training을 추가한 버전입니다. 기본 운영은 여전히 `prediction_file` 모드이며, live inference는 candidate 검증용입니다.



## v0.3.3 추가 변경

- TrainingService 기본 학습을 `sample_weight_mode=recency`로 변경
- horizon별 recency half-life 기본값 추가: 5D=126, 10D=252, 20D=504
- leak-safe expanding walk-forward 안에서 train row에 지수 감쇠 sample_weight 적용
- 선택 옵션으로 `walk_forward_mode=rolling` 및 `rolling_train_rows` 추가
- Activation gate에 PR-AUC lift, brier worst-fold, positive-rate worst-fold 조건 추가
- `training_config`를 model metadata/manifest에 저장

## v0.3.2 추가 변경

- activation gate에 `roc_auc >= 0.52` 최소 기준 추가
- activation gate에 `roc_auc_worst >= 0.48` worst-fold 안정성 기준 추가
- `TrainingService`가 `roc_auc_worst`, `pr_auc_worst`, `brier_worst`, `positive_rate_worst`를 top-level metric으로 저장
- `AllocationService._weight_col()` dead code 제거
- 구버전 run script 정리

## 핵심 변경

- 기존 `Prediction File Mode` 유지
- `Live Inference Mode` 추가
- `FeaturePipeline` 기반 OHLCV -> feature 변환 구조 추가
- `ModelLoader`/`ModelArtifactStore`/`ModelRegistry` 기반 모델 artifact 관리 추가
- `TrainingService`로 cached OHLCV 기반 candidate 모델 학습 구조 추가
- `/models/infer`, `/models/runtime-status`, `/models/artifact-inventory` 추가
- `/dashboard?model_mode=prediction_file|live|auto` 지원
- `auto` dashboard는 live inference 실패 시 prediction file mode로 fallback

## 실행

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
set PYTHONPATH=%CD%
set PRA_INPUT_DIR=%CD%\storage\predictions
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

브라우저:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/dashboard
```

## 주요 확인 순서

### 1. 기존 대시보드 확인

```text
GET /dashboard
```

### 2. Yahoo 데이터 수집

```json
POST /market-data/update
{
  "provider": "yahoo",
  "tickers": ["QQQ", "SPY", "AAPL"],
  "market": "US",
  "start_date": "2020-01-01",
  "end_date": "2026-05-07"
}
```

### 3. Runtime 상태 확인

```text
GET /models/runtime-status?assets=QQQ,SPY,AAPL
```

초기 상태에서는 model artifact가 없으므로 `runtime_ready=false`가 정상입니다.

### 4. Candidate 모델 학습

```json
POST /training/retrain
{
  "tickers": ["QQQ", "SPY", "AAPL"],
  "horizons": ["5D", "10D", "20D"],
  "train_start": "2020-01-01",
  "train_end": "2026-05-07",
  "data_source": "cache",
  "market": "US",
  "model_version": "candidate_runtime_v0_3_test"
}
```

작업 상태:

```text
GET /training/jobs/{job_id}
```

### 5. 모델 artifact 확인

```text
GET /models/artifact-inventory
GET /models/runtime-status?assets=QQQ,SPY,AAPL&model_version=candidate_runtime_v0_3_test
```

### 6. Live inference 실행

```json
POST /models/infer
{
  "tickers": ["QQQ", "SPY", "AAPL"],
  "horizon": "10D",
  "provider": "auto",
  "market": "US",
  "model_version": "candidate_runtime_v0_3_test"
}
```

### 7. Dashboard live mode

```text
GET /dashboard?model_mode=live&assets=QQQ,SPY,AAPL&horizon=10D&provider=auto
```

안전하게 fallback까지 쓰려면:

```text
GET /dashboard?model_mode=auto&assets=QQQ,SPY,AAPL&horizon=10D&provider=auto
```

## 운영 원칙

- `v8.6.41_label_fixed`는 기본 기준선입니다.
- 새로 학습한 모델은 `CANDIDATE`로 등록됩니다.
- 검증 없이 active 전환하지 않는 것이 원칙입니다.
- `live_inference`는 연구 모델 개선이 아니라 프로그램 운영 구조입니다.


## v0.3.5 Horizon Ensemble Update

v0.3.5 adds gate-aware horizon ensembles for `highvol`, `up_strength`, and `down_strength`.

The live inference path now combines 5D/10D/20D probabilities per family after head-level gate adjustment. Main probability fields used by allocation are set to ensemble values:

```text
prob_high_vol = prob_high_vol_ensemble
prob_up_strengthening_score = prob_up_strengthening_ensemble
prob_down_strengthening_score = prob_down_strengthening_ensemble
```

Additional audit fields are returned:

```text
horizon_ensembles
highvol_state
up_strength_state
down_strength_state
ensemble_used_heads
ensemble_fallback_heads
```

This is experimental and must be compared against the locked v8.6.41 prediction_file baseline before operational use.

## v0.3.6 note

v0.3.6 adds Soft Family Gate. The strict ticker-level activation gate is still reported for audit, but runtime allocation uses family-level confidence-shrunk ensemble probabilities:

```text
raw family ensemble probability -> neutral shrinkage -> effective probability -> allocation
```

Do not treat v0.3.6 as production replacement for v8.6.41 until OOS portfolio validation is complete.
