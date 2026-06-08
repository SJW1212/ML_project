# v0.3 구조 변경 및 추가 사항

## 1. 변경 목적

기존 v0.2.1은 `/dashboard`에서 prediction CSV를 읽는 구조였습니다. v0.3은 여기에 실제 운영 모델 구조를 추가합니다.

```text
Yahoo/KIS OHLCV cache
→ FeaturePipeline
→ ModelLoader
→ InferenceService
→ AllocationService
→ DashboardSerializer
→ UI
```

## 2. 추가된 모델 계층

```text
backend/app/model/
  model_loader.py
  model_artifact_store.py
  inference_service.py
  training_service.py
  training_job_manager.py
  model_registry.py
```

## 3. 추가된 API

```text
GET  /models/runtime-status
GET  /models/artifact-inventory
POST /models/infer
POST /training/retrain
GET  /training/jobs/{job_id}
GET  /dashboard?model_mode=live
GET  /dashboard?model_mode=auto
```

## 4. Dashboard mode

```text
prediction_file:
  기존 CSV 기반 대시보드

live:
  cached OHLCV + model artifact 기반 실시간 추론

auto:
  live inference 시도 후 실패하면 prediction_file로 fallback
```

## 5. 아직 의도적으로 하지 않은 것

```text
- v8.6.41 기준 모델 교체
- 자동매매
- 무검증 active 전환
- 사용자 지정 horizon 안정 지원
```

## 6. 다음 작업

```text
1. Yahoo/KIS 데이터 수집
2. candidate_runtime_v0_3_test 학습
3. /models/infer 결과 확인
4. prediction_file 결과와 live 결과 비교
5. 오차/성능 검증 후 active 전환 여부 판단
```
