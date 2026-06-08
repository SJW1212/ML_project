# v8.6.41 Local Production Requirements Summary

## Fixed Scope

```text
운영 모델: v8.6.41_model_label_fixed
실행 모드: prediction_file
운영 환경: 로컬 실행
데이터 갱신: 하루 1회 일봉/OHLCV 캐시 업데이트
```

## Included

```text
- v8.6.41 prediction file 로딩
- latest signal JSON 생성
- portfolio allocation JSON 생성
- equal/custom/inverse-vol 자본 배분
- performance/charts payload 생성
- validation checks
- local FastAPI API
- Yahoo/yfinance daily OHLCV cache update
- KIS data-query adapter placeholder only
- cache freshness endpoint
- one-shot daily update batch file
- Windows Task Scheduler helper script
- file logging
```

## Excluded

```text
- 실시간 주문
- 자동매매
- DB 저장
- 사용자 계정별 포트폴리오 저장
- 알림 기능
- Pixso 화면 설계 반영
- 실시간 데이터 스트리밍
- v0.3.x runtime/context/head gate/horizon ensemble/soft family gate
```

## Review-driven Fixes Applied

```text
1. performance.py inner join silent cutoff 제거
   - outer join 사용
   - missing per-asset returns = 0.0
   - date mismatch validation WARN 추가

2. prediction file과 cache 역할 분리 명시
   - OHLCV cache는 데이터 참고/향후 입력 갱신용
   - allocation 결정은 prediction file 기준

3. yfinance 의존성 명확화
   - requirements 주석 추가
   - 런타임 graceful error 유지

4. API provider 검증 강화
   - Literal['yahoo','kis'] 사용
   - 중복 provider parameter 제거

5. 로컬 운영성 보강
   - storage/logs/api_server.log
   - run_daily_update_once.bat
   - scripts/setup_scheduled_task.bat
   - scripts/remove_scheduled_task.bat

6. validation 확장
   - WARN 상태 추가
   - asset date range mismatch
   - allocation all-cash fallback tracking
   - inverse-vol fallback metadata

7. config 확장
   - risk_free_rate 추가
   - local_app_config.json 로딩 지원
```

## Daily Update Policy

```text
- 기본 권장 시간: 매일 08:00 KST
- 실시간/분봉/스트리밍 없음
- 주문 API 없음
- 캐시 업데이트와 prediction file 갱신은 별도 프로세스
```
