# v8.6.41 Label Fixed UI-ready Modular Production

## 목적

- 최종 운영 모델은 `v8.6.41_model_label_fixed`로 고정합니다.
- Loss Guard, confidence layer, v2/v3/v4 보정 레이어, 42계열 adaptive controls는 적용하지 않습니다.
- 코드는 실제 패키지 구조로 분리했습니다.
- UI/UX 연동을 전제로 하므로 **CSV 출력은 기본값에서 제외**하고, JSON/API payload를 기본 산출물로 사용합니다.

## 디렉터리 구조

```text
src/v8641_production/
  __init__.py
  constants.py       # 모델명, 기본 assets, 제외 레이어
  config.py          # ProductionConfig
  schemas.py         # dataclass schema
  utils.py           # 수치/날짜 유틸
  repository.py      # prediction/summary 파일 로딩
  signals.py         # latest signal class 분류
  allocation.py      # native 41 allocation + 자본비중 집계
  performance.py     # 성과/차트용 데이터 계산
  validation.py      # 검증 체크
  serializer.py      # JSON/선택적 CSV/Markdown 저장
  service.py         # UI backend에서 호출할 facade
  cli.py             # CLI entry point
  api_fastapi.py     # 선택적 FastAPI adapter
```

## 실행

```bash
set PYTHONPATH=src
python -m v8641_production.cli ^
  --input-dir /mnt/data ^
  --out-dir /mnt/data/v8_6_41_ui_modular_ops ^
  --assets QQQ,SPY,AAPL,SOXX,NVDA ^
  --allocation-source executed ^
  --capital-mode equal
```

## 기본 출력

```text
v8_6_41_ui_modular_ops/
  dashboard_payload.json   # UI 화면 전체 payload
  latest_state.json        # 최신 상태만 압축한 payload
```

CSV가 필요할 때만 다음 옵션을 추가합니다.

```bash
--export-csv
```

## UI 연동 방식

백엔드에서 직접 호출:

```python
from pathlib import Path
from v8641_production import ProductionConfig, ProductionService

config = ProductionConfig(
    input_dir=Path('/mnt/data'),
    out_dir=Path('/mnt/data/v8_6_41_ui_modular_ops'),
    assets=['QQQ', 'SPY', 'AAPL', 'SOXX', 'NVDA'],
    export_json=False,
    export_csv=False,
)

payload = ProductionService(config).build_dashboard_payload()
```

FastAPI 옵션:

```bash
pip install fastapi uvicorn
set PYTHONPATH=src
set V8641_INPUT_DIR=/mnt/data
uvicorn v8641_production.api_fastapi:app --reload
```

- `GET /dashboard`: 전체 화면 payload
- `GET /latest`: 최신 signal/allocation/validation만 반환

## 설계 원칙

- 입력은 기존 v8.6.41 prediction CSV를 사용합니다. 이건 모델 산출물 소스입니다.
- 출력은 UI payload JSON/API 중심입니다.
- 최종 사용자 화면에는 CSV를 직접 노출할 필요가 없습니다.
- CSV export는 디버깅/검증/엑셀 공유가 필요할 때만 사용합니다.

---

## v0.4.1 Review Fix Patch Notes

사용자가 업로드한 검증 보고서의 핵심 지적 사항을 반영했습니다.

### 반영된 수정

```text
- portfolio_daily_returns inner join 제거
- outer join + missing asset return 0.0 처리
- asset date range mismatch를 WARN validation으로 노출
- yfinance 의존성 설명 정리
- /data/freshness provider 중복 parameter 제거
- DailyUpdateRequest provider 검증 강화
- local_app_config.json 로딩 지원
- storage/logs/api_server.log 파일 로깅 추가
- run_daily_update_once.bat를 서버 의존 curl 방식에서 CLI 방식으로 변경
- Windows Task Scheduler 등록/삭제 스크립트 추가
- allocation all-cash fallback 추적
- inverse_vol fallback metadata 추적
- risk_free_rate 설정 추가
```

### 중요 정책

```text
OHLCV cache update는 prediction file을 자동 재생성하지 않습니다.
포트폴리오 비중 결정의 source of truth는 여전히 v8.6.41 prediction file입니다.
```

### 하루 1회 업데이트 등록

```bat
scripts\setup_scheduled_task.bat
```

삭제:

```bat
scripts\remove_scheduled_task.bat
```

수동 1회 실행:

```bat
run_daily_update_once.bat
```
