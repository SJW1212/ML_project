# -*- coding: utf-8 -*-
"""
make_ohlcv_source_data.py

Portfolio Regime Advisor / RiskOff-HighVol 실험용 원천 OHLCV CSV 생성 스크립트.

목적
----
이전 검증 코드인 riskoff_highvol_walkforward_experiment.py에 입력할 수 있는
원천 데이터 파일을 생성합니다.

생성 파일
---------
1. 티커별 OHLCV CSV
   예: ohlcv_source/QQQ_ohlcv.csv

2. 전체 티커 통합 CSV
   예: ohlcv_source/ohlcv_all_tickers.csv

3. 종가 패널 CSV
   예: ohlcv_source/close_panel.csv

4. 데이터 품질 리포트 JSON
   예: ohlcv_source/data_quality_report.json

필수 컬럼
---------
- date
- open
- high
- low
- close
- adj_close
- volume
- ticker
- source

riskoff_highvol_walkforward_experiment.py는 최소한 date, close만 있으면 실행되지만,
open/high/low/volume까지 함께 저장하는 것을 권장합니다.

설치
----
pip install yfinance pandas numpy

실행 예시
---------
# QQQ 단일 원천 데이터 생성
python make_ohlcv_source_data.py --tickers QQQ --start 2013-01-01 --output-dir ohlcv_source

# 여러 티커 생성
python make_ohlcv_source_data.py --tickers QQQ,SPY,IEF,BIL,SOXX,XLK --start 2013-01-01 --output-dir ohlcv_source

# 특정 종료일 지정
python make_ohlcv_source_data.py --tickers QQQ --start 2013-01-01 --end 2026-06-01 --output-dir ohlcv_source

# 인터넷 없이 동작 확인용 synthetic 데이터 생성
python make_ohlcv_source_data.py --synthetic --tickers QQQ,SPY --output-dir ohlcv_source_synthetic

주의
----
- yfinance 데이터는 연구/교육용으로 적합하지만, 실거래/정산용 원천으로 쓰면 안 됩니다.
- 한국 종목은 yfinance 형식에 맞춰 005930.KS, 035720.KQ처럼 입력해야 합니다.
- CSV 파일명에서는 특수문자를 안전하게 치환합니다.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 0. 유틸
# ============================================================

def parse_tickers(value: str) -> List[str]:
    if not value:
        return ["QQQ"]
    return [x.strip().upper() for x in value.split(",") if x.strip()]


def safe_filename_ticker(ticker: str) -> str:
    """
    파일명에 안전한 티커 문자열로 변환.
    예:
    005930.KS -> 005930_KS
    BRK-B -> BRK_B
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", ticker.replace(".", "_").replace("-", "_"))


def save_json(path: str | Path, data: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def normalize_ohlcv_columns(df: pd.DataFrame, ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """
    yfinance 또는 synthetic 결과를 표준 OHLCV schema로 정규화.
    """
    out = df.copy()

    # MultiIndex column 방어
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([str(x) for x in col if str(x) != ""])
            for col in out.columns
        ]

    # Reset index 후 date 컬럼 정리
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()

    # 컬럼명 소문자/스네이크 유사 정리
    rename_map = {}
    for c in out.columns:
        c_str = str(c).strip()
        c_low = c_str.lower().replace(" ", "_")

        if c_low in {"date", "datetime"}:
            rename_map[c] = "date"
        elif c_low.startswith("open"):
            rename_map[c] = "open"
        elif c_low.startswith("high"):
            rename_map[c] = "high"
        elif c_low.startswith("low"):
            rename_map[c] = "low"
        elif c_low.startswith("close") and "adj" not in c_low:
            rename_map[c] = "close"
        elif c_low in {"adj_close", "adjclose"} or "adj_close" in c_low:
            rename_map[c] = "adj_close"
        elif c_low.startswith("volume"):
            rename_map[c] = "volume"

    out = out.rename(columns=rename_map)

    if "date" not in out.columns:
        raise ValueError("date column not found after normalization")

    # yfinance auto_adjust=True이면 adj_close가 없을 수 있음
    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"] if "close" in out.columns else np.nan

    required_price_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for c in required_price_cols:
        if c not in out.columns:
            out[c] = np.nan

    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = ticker
    out["source"] = source

    keep_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
    out = out[keep_cols].copy()

    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values("date").drop_duplicates(subset=["date", "ticker"], keep="last").reset_index(drop=True)

    # 가격이 전부 비어있는 행 제거
    out = out.dropna(subset=["close"]).reset_index(drop=True)

    return out


# ============================================================
# 1. yfinance 다운로드
# ============================================================

def download_yfinance_ohlcv(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    yfinance에서 티커별 OHLCV 다운로드.

    auto_adjust=False:
    - close와 adj_close를 둘 다 확보하기 위함.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance가 설치되어 있지 않습니다. 먼저 `pip install yfinance`를 실행하세요."
        ) from e

    data = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=False,
    )

    if data is None or data.empty:
        raise ValueError(f"No data downloaded for ticker={ticker}")

    return normalize_ohlcv_columns(data, ticker=ticker, source="yfinance")


# ============================================================
# 2. Synthetic 데이터 생성
# ============================================================

def make_synthetic_ohlcv(
    ticker: str,
    start: str = "2013-01-01",
    rows: int = 3373,
    seed: int = 42,
) -> pd.DataFrame:
    """
    인터넷 없이 코드 동작 확인용 synthetic OHLCV 생성.

    실제 투자/모델 성능 평가에는 사용하면 안 됨.
    """
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32) + seed)
    dates = pd.bdate_range(start=start, periods=rows)

    # 간단한 regime 구조
    ret = rng.normal(0.00035, 0.012, size=rows)

    # 위기 구간 몇 개 삽입
    crisis_slices = [
        slice(int(rows * 0.25), int(rows * 0.30)),
        slice(int(rows * 0.55), int(rows * 0.60)),
        slice(int(rows * 0.78), int(rows * 0.82)),
    ]

    for s in crisis_slices:
        ret[s] = rng.normal(-0.0012, 0.028, size=len(range(*s.indices(rows))))

    close = 100 * np.cumprod(1 + ret)
    open_ = close * (1 + rng.normal(0, 0.0025, size=rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.000, 0.010, size=rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.000, 0.010, size=rows))
    adj_close = close.copy()
    volume = rng.integers(1_000_000, 10_000_000, size=rows)

    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": adj_close,
            "volume": volume,
            "source": "synthetic",
        }
    )
    return df


# ============================================================
# 3. 품질 검사
# ============================================================

@dataclass
class DataQualityReport:
    ticker: str
    rows: int
    start_date: Optional[str]
    end_date: Optional[str]
    missing_close: int
    missing_volume: int
    duplicate_dates: int
    non_positive_close: int
    max_date_gap_days: Optional[int]
    median_date_gap_days: Optional[float]
    status: str
    warning: List[str]


def check_data_quality(df: pd.DataFrame, ticker: str) -> DataQualityReport:
    warnings: List[str] = []

    if df.empty:
        return DataQualityReport(
            ticker=ticker,
            rows=0,
            start_date=None,
            end_date=None,
            missing_close=0,
            missing_volume=0,
            duplicate_dates=0,
            non_positive_close=0,
            max_date_gap_days=None,
            median_date_gap_days=None,
            status="failed",
            warning=["empty dataframe"],
        )

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")

    missing_close = int(d["close"].isna().sum())
    missing_volume = int(d["volume"].isna().sum()) if "volume" in d.columns else len(d)
    duplicate_dates = int(d.duplicated(subset=["date"]).sum())
    non_positive_close = int((d["close"] <= 0).sum())

    gaps = d["date"].diff().dt.days.dropna()
    max_gap = int(gaps.max()) if len(gaps) else None
    median_gap = float(gaps.median()) if len(gaps) else None

    if len(d) < 756:
        warnings.append("rows < 756: walk-forward minimum training rows may be insufficient")
    if missing_close > 0:
        warnings.append("missing close values detected")
    if duplicate_dates > 0:
        warnings.append("duplicate dates detected")
    if non_positive_close > 0:
        warnings.append("non-positive close values detected")
    if max_gap is not None and max_gap > 14:
        warnings.append(f"large date gap detected: max_gap_days={max_gap}")

    status = "ok" if not warnings else "warning"

    return DataQualityReport(
        ticker=ticker,
        rows=int(len(d)),
        start_date=str(d["date"].min().date()),
        end_date=str(d["date"].max().date()),
        missing_close=missing_close,
        missing_volume=missing_volume,
        duplicate_dates=duplicate_dates,
        non_positive_close=non_positive_close,
        max_date_gap_days=max_gap,
        median_date_gap_days=median_gap,
        status=status,
        warning=warnings,
    )


# ============================================================
# 4. 저장 로직
# ============================================================

def build_close_panel(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    티커별 close를 wide panel로 변환.
    """
    panel = all_df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    panel = panel.sort_index().reset_index()
    return panel


def create_source_data(
    tickers: Sequence[str],
    start: str,
    end: Optional[str],
    output_dir: str | Path,
    synthetic: bool = False,
    synthetic_rows: int = 3373,
    auto_adjust: bool = False,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ticker_frames: List[pd.DataFrame] = []
    reports: List[DataQualityReport] = []
    output_paths: Dict[str, Path] = {}

    for ticker in tickers:
        print(f"[INFO] building source data: {ticker}")

        try:
            if synthetic:
                df = make_synthetic_ohlcv(ticker=ticker, start=start, rows=synthetic_rows)
            else:
                df = download_yfinance_ohlcv(
                    ticker=ticker,
                    start=start,
                    end=end,
                    auto_adjust=auto_adjust,
                )

            report = check_data_quality(df, ticker)
            ticker_frames.append(df)

            file_ticker = safe_filename_ticker(ticker)
            ticker_path = output_dir / f"{file_ticker}_ohlcv.csv"
            df.to_csv(ticker_path, index=False, encoding="utf-8-sig")
            output_paths[f"{ticker}_ohlcv"] = ticker_path

            print(
                f"[OK] {ticker}: rows={len(df)}, "
                f"start={df['date'].min().date()}, end={df['date'].max().date()}, "
                f"status={report.status}"
            )

        except Exception as e:
            report = DataQualityReport(
                ticker=ticker,
                rows=0,
                start_date=None,
                end_date=None,
                missing_close=0,
                missing_volume=0,
                duplicate_dates=0,
                non_positive_close=0,
                max_date_gap_days=None,
                median_date_gap_days=None,
                status="failed",
                warning=[str(e)],
            )
            print(f"[ERROR] {ticker}: {e}")

        reports.append(report)

    if ticker_frames:
        all_df = pd.concat(ticker_frames, axis=0, ignore_index=True)
        all_df = all_df.sort_values(["ticker", "date"]).reset_index(drop=True)

        all_path = output_dir / "ohlcv_all_tickers.csv"
        all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
        output_paths["all_tickers"] = all_path

        close_panel = build_close_panel(all_df)
        panel_path = output_dir / "close_panel.csv"
        close_panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
        output_paths["close_panel"] = panel_path

    report_data = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "synthetic" if synthetic else "yfinance",
        "start": start,
        "end": end,
        "tickers": list(tickers),
        "auto_adjust": auto_adjust,
        "reports": [asdict(r) for r in reports],
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "usage_example": {
            "riskoff_highvol": (
                "python riskoff_highvol_walkforward_experiment.py "
                "--input <TICKER>_ohlcv.csv --ticker <TICKER> "
                "--output-dir riskoff_highvol_results"
            )
        },
    }

    report_path = output_dir / "data_quality_report.json"
    save_json(report_path, report_data)
    output_paths["data_quality_report"] = report_path

    return output_paths


# ============================================================
# 5. Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default="QQQ", help="comma-separated tickers, e.g. QQQ,SPY,IEF,BIL")
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default=None, help="exclusive end date for yfinance, e.g. 2026-06-02")
    parser.add_argument("--output-dir", default="ohlcv_source")
    parser.add_argument("--auto-adjust", action="store_true", help="yfinance auto_adjust=True")
    parser.add_argument("--synthetic", action="store_true", help="create synthetic OHLCV without internet")
    parser.add_argument("--synthetic-rows", type=int, default=3373)

    args = parser.parse_args()

    tickers = parse_tickers(args.tickers)

    paths = create_source_data(
        tickers=tickers,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        synthetic=args.synthetic,
        synthetic_rows=args.synthetic_rows,
        auto_adjust=args.auto_adjust,
    )

    print()
    print("[DONE] Source data generation completed.")
    print(f"[DONE] Output dir: {Path(args.output_dir).resolve()}")
    print("[DONE] Files:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
