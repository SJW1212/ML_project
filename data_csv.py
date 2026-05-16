# Python 3.10+
# 필요 패키지:
# pip install yfinance pandas numpy matplotlib seaborn scikit-learn

import os
import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# 1. 기본 설정
# =========================

TICKER = "QQQ"
START_DATE = "2000-01-01"
END_DATE = None  # None이면 가능한 최신 데이터까지 수집

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "qqq_features_labeled.csv")


# =========================
# 2. 데이터 수집
# =========================

def download_price_data(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """
    yfinance를 사용해 OHLCV 데이터를 수집한다.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError("데이터 수집 실패: 반환된 데이터가 비어 있습니다.")

    df = df.reset_index()

    # yfinance 버전에 따라 컬럼이 MultiIndex로 나오는 경우 방어
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")

    return df[required_cols].copy()


# =========================
# 3. 피처 생성
# =========================

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    주가 원천 데이터에서 머신러닝 입력 피처를 생성한다.
    가격 자체보다 수익률, 비율, 변동성 중심으로 변환한다.
    """
    df = df.copy()

    # 날짜 정렬
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 수익률
    df["daily_return"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # 로그 수익률
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # 이동평균
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()
    df["ma_60"] = df["Close"].rolling(window=60).mean()

    # 이동평균 괴리율
    df["ma_gap_5_20"] = df["ma_5"] / df["ma_20"] - 1
    df["ma_gap_20_60"] = df["ma_20"] / df["ma_60"] - 1

    # 변동성
    df["volatility_5d"] = df["daily_return"].rolling(window=5).std()
    df["volatility_20d"] = df["daily_return"].rolling(window=20).std()
    df["volatility_60d"] = df["daily_return"].rolling(window=60).std()

    # 거래량 변화율
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ma_20"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma_20"]

    # 고점 대비 하락률, Drawdown
    df["cummax_close"] = df["Close"].cummax()
    df["drawdown"] = df["Close"] / df["cummax_close"] - 1

    # 미래 수익률
    df["future_return_20d"] = df["Close"].shift(-20) / df["Close"] - 1

    # 미래 변동성
    df["future_volatility_20d"] = (
        df["daily_return"]
        .shift(-20)
        .rolling(window=20)
        .std()
    )

    return df


# =========================
# 4. 라벨 생성
# =========================

def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    미래 20거래일 수익률과 미래 변동성을 기준으로 시장 국면 라벨을 만든다.

    라벨:
    - 상승
    - 하락
    - 횡보
    - 고변동
    - 불확실
    """
    df = df.copy()

    # 고변동 기준: 미래 20일 변동성 상위 20%
    high_vol_threshold = df["future_volatility_20d"].quantile(0.80)

    def classify_market(row):
        future_return = row["future_return_20d"]
        future_vol = row["future_volatility_20d"]

        if pd.isna(future_return) or pd.isna(future_vol):
            return np.nan

        # 고변동을 가장 먼저 판단
        # 이유: 수익률 방향보다 위험도 자체가 중요한 구간이기 때문
        if future_vol >= high_vol_threshold:
            return "고변동"

        if future_return >= 0.03:
            return "상승"

        if future_return <= -0.03:
            return "하락"

        if -0.015 <= future_return <= 0.015:
            return "횡보"

        return "불확실"

    df["label"] = df.apply(classify_market, axis=1)

    return df


# =========================
# 5. 데이터 정리 및 저장
# =========================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 학습에 사용할 수 있도록 결측값과 무한값을 정리한다.
    """
    df = df.copy()

    # inf 값 제거
    df = df.replace([np.inf, -np.inf], np.nan)

    # 피처 생성 과정에서 생기는 앞부분 NaN, 미래수익률 때문에 생기는 뒷부분 NaN 제거
    df = df.dropna().reset_index(drop=True)

    return df


# =========================
# 6. 실행 함수
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1] QQQ 데이터 수집 중...")
    raw_df = download_price_data(TICKER, START_DATE, END_DATE)
    print(f"수집 완료: {len(raw_df):,}개 행")

    print("[2] 피처 생성 중...")
    feature_df = make_features(raw_df)

    print("[3] 라벨 생성 중...")
    labeled_df = make_labels(feature_df)

    print("[4] 결측값 정리 중...")
    final_df = clean_dataset(labeled_df)

    print("[5] CSV 저장 중...")
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n저장 완료:")
    print(OUTPUT_FILE)

    print("\n데이터 크기:")
    print(final_df.shape)

    print("\n컬럼 목록:")
    print(final_df.columns.tolist())

    print("\n라벨 분포:")
    print(final_df["label"].value_counts())

    print("\n라벨 비율:")
    print((final_df["label"].value_counts(normalize=True) * 100).round(2))


if __name__ == "__main__":
    main()