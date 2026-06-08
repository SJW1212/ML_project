# Portfolio Regime Advisor Backend v0.3.2 — Activation Gate Hardened

## 목적

v0.3.1 검증에서 남은 핵심 결함인 activation gate의 ROC-AUC 하한 부재를 수정했다. v0.3.2는 live inference candidate 모델을 ACTIVE로 승격하기 전, 최소 판별력과 fold 안정성을 코드 레벨에서 검사한다.

## 주요 변경

### 1. Activation Gate 강화

`backend/app/model/model_registry.py`

추가 기준:

- `min_roc_auc_mean = 0.52`
- `min_roc_auc_worst = 0.48`
- `min_ok_folds = 2`
- `max_brier_mean = 0.35`
- `positive_rate in [0.02, 0.98]`

기존 문제:

- v0.3.1은 ROC-AUC가 존재하기만 하면 gate를 통과할 수 있었다.
- AUC 0.40 수준의 무판별 모델도 통과 가능했다.

수정 후:

- 평균 ROC-AUC가 0.52 미만이면 실패한다.
- worst-fold ROC-AUC가 0.48 미만이면 실패한다.
- ROC-AUC가 없으면 실패한다.

### 2. TrainingService metric flatten 보강

`backend/app/model/training_service.py`

`metrics[key]`에 아래 값을 top-level로 저장한다.

- `roc_auc_worst`
- `pr_auc_worst`
- `brier_worst`
- `positive_rate_worst`

`walk_forward.aggregate` 내부 값도 그대로 보존한다.

### 3. LOW 이슈 정리

- `AllocationService._weight_col()` dead code 제거
- 구버전 배치 스크립트 `run_backend_v0_3.bat`, `run_backend_v0_3_1.bat` 제거
- `run_backend.bat`을 단일 Windows 실행 스크립트로 유지

## 운영 판단

v0.3.2도 live 모델을 곧바로 운영 ACTIVE로 쓰는 버전은 아니다. 다만 v0.3.1보다 candidate 승격 차단 장치가 실질적으로 강화되었으므로, 다음 단계인 실데이터 candidate 학습과 prediction_file vs live 비교 검증으로 넘어갈 수 있다.

## 필수 후속 검증

1. Yahoo/KIS cache 생성
2. candidate 학습
3. activation gate 결과 확인
4. `/models/infer` live inference 확인
5. `/dashboard?model_mode=prediction_file` vs `/dashboard?model_mode=live` 비교
6. 포트폴리오 OOS 성과 검증
