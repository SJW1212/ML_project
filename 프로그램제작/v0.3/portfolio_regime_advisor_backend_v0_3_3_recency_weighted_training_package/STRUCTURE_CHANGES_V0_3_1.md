# Portfolio Regime Advisor Backend v0.3.1 — Runtime Stabilized Patch

## 목적

v0.3 검증에서 확인된 운영 승격 차단 이슈를 우선 수정한 안정화 패치입니다.
기본 운영은 여전히 `prediction_file` 모드이며, live inference는 CANDIDATE 검증 경로로 유지합니다.

## 핵심 수정

### 1. Secret key 패키지 제거

- `storage/secrets/pra_fernet.key`를 패키지에서 제거했습니다.
- `CredentialManager`는 최초 실행 시 로컬 런타임에서 키를 자동 생성합니다.
- `.gitignore`, `.packageignore`에 `storage/secrets/`를 추가했습니다.

### 2. 공통 AllocationPolicyEngine 추가

추가 파일:

```text
backend/app/portfolio/allocation_policy_engine.py
```

역할:

- prediction_file 경로: 기존 v8.6.41 native `stock/bond/cash_weight` 보존
- live inference 경로: 동일한 policy engine으로 확률 기반 fallback 비중 산출
- InferenceService 내부의 `_weights_from_probs()` 제거

### 3. FeaturePipeline high_vol label 안정화

변경 전:

```text
future_abs.rolling(252).quantile(0.75).shift(1)
```

변경 후:

```text
future_abs_return_h > k_high_vol * current_vol * sqrt(horizon)
```

즉, `y_high_vol_{h}d`도 up/down strength와 같은 volatility-scaled 계열로 맞췄습니다.

### 4. TrainingService walk-forward 검증 추가

변경 전:

```text
단순 80/20 holdout
```

변경 후:

```text
expanding walk-forward validation
```

저장 지표:

- fold별 ROC-AUC
- fold별 PR-AUC
- fold별 Brier
- fold별 positive rate
- 평균/표준편차/worst fold

최종 artifact는 walk-forward 검증 후 전체 train window로 재학습하여 저장합니다.

### 5. ModelRegistry 안정화

- `filelock` 기반 파일 잠금 추가
- atomic save 방식 적용
- `activate()` 전에 activation gate 적용
- gate 실패 시 400 응답 반환

### 6. Dashboard live/auto 개선

- `/dashboard`에 `model_version` 파라미터 추가
- `/dashboard`에 `allow_partial_live` 파라미터 추가
- auto 모드에서 일부 ticker만 live 성공할 경우 기본적으로 prediction_file fallback

### 7. 기타 수정

- `performance_summary()`의 `0.0` falsy 처리 수정
- CORS wildcard 제거
- `PRA_STORAGE_DIR` override 시 `model_dir/cache_dir/registry_dir/secrets_dir` 파생 정리
- training job 동시 실행 제한
- `run_backend.bat` 스크립트 통합

## 사용 순서

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
run_backend.bat
```

확인:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/dashboard
http://127.0.0.1:8000/models/runtime-status
```

## 남은 검증

v0.3.1도 live inference를 즉시 운영으로 쓰면 안 됩니다.
아래 검증이 필요합니다.

```text
1. Yahoo/KIS cache 수집
2. candidate 학습
3. walk-forward metrics 확인
4. /models/infer 결과 확인
5. prediction_file vs live 결과 비교
6. 포트폴리오 OOS 성과 비교
7. 통과 시 실험 모드로 UI 노출
```
