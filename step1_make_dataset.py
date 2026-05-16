# Python 3.10+
# 필요 패키지:
# pip install yfinance pandas numpy

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

# 라벨 기준
PREDICT_HORIZON = 20              # 미래 20거래일 기준
UP_THRESHOLD = 0.02               # 미래 20일 수익률 +2% 이상 → 상승
DOWN_THRESHOLD = -0.02            # 미래 20일 수익률 -2% 이하 → 하락
HIGH_VOL_QUANTILE = 0.80          # 미래 변동성 상위 20% → 고변동


# =========================
# 2. 데이터 수집
# =========================

def download_price_data(
    ticker: str,
    start: str,
    end: str | None = None
) -> pd.DataFrame:
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
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")

    df = df[required_cols].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# =========================
# 3. 보조 지표 함수
# =========================

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    0으로 나누기와 무한대 값을 방지하는 나눗셈 함수.
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    rolling z-score 계산.
    현재 값이 최근 window 평균 대비 얼마나 벗어나 있는지 측정한다.
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    return safe_divide(series - rolling_mean, rolling_std)


def rolling_log_slope(close: pd.Series, window: int) -> pd.Series:
    """
    로그 가격 기준 rolling 추세 기울기.
    값이 양수면 상승 추세, 음수면 하락 추세로 해석 가능하다.
    """
    log_price = np.log(close)

    def calc_slope(values: np.ndarray) -> float:
        x = np.arange(len(values))

        if np.isnan(values).any():
            return np.nan

        slope = np.polyfit(x, values, 1)[0]
        return slope

    return log_price.rolling(window=window).apply(calc_slope, raw=True)


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI 계산.
    과매수/과매도 상태를 나타내는 대표적인 모멘텀 지표.
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = safe_divide(avg_gain, avg_loss)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD 계산.
    추세 전환과 모멘텀을 확인하기 위한 지표.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    return macd, macd_signal, macd_hist


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    ATR 계산.
    장중 변동폭과 갭 변동을 함께 반영하는 변동성 지표.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()

    return atr


# =========================
# 4. 피처 생성
# =========================

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    주가 원천 데이터에서 머신러닝 입력 피처를 생성한다.

    설계 원칙:
    - 가격 수준 자체보다 수익률, 비율, 위치, 변동성 중심으로 변환한다.
    - 미래 정보는 라벨 생성에만 사용한다.
    """
    df = df.copy()

    # -------------------------
    # 수익률 피처
    # -------------------------
    df["daily_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)
    df["return_60d"] = df["Close"].pct_change(60)

    # -------------------------
    # 이동평균 기반 피처
    # -------------------------
    ma_5 = df["Close"].rolling(window=5).mean()
    ma_20 = df["Close"].rolling(window=20).mean()
    ma_60 = df["Close"].rolling(window=60).mean()

    df["price_ma_5_gap"] = safe_divide(df["Close"], ma_5) - 1
    df["price_ma_20_gap"] = safe_divide(df["Close"], ma_20) - 1
    df["price_ma_60_gap"] = safe_divide(df["Close"], ma_60) - 1

    df["ma_gap_5_20"] = safe_divide(ma_5, ma_20) - 1
    df["ma_gap_20_60"] = safe_divide(ma_20, ma_60) - 1

    # -------------------------
    # 변동성 피처
    # -------------------------
    df["volatility_5d"] = df["daily_return"].rolling(window=5).std()
    df["volatility_20d"] = df["daily_return"].rolling(window=20).std()
    df["volatility_60d"] = df["daily_return"].rolling(window=60).std()

    df["volatility_ratio_5_20"] = safe_divide(
        df["volatility_5d"],
        df["volatility_20d"]
    )

    df["volatility_ratio_20_60"] = safe_divide(
        df["volatility_20d"],
        df["volatility_60d"]
    )

    # 하락 수익률만 대상으로 한 변동성
    downside_return = df["daily_return"].where(df["daily_return"] < 0, 0)
    df["downside_volatility_20d"] = downside_return.rolling(window=20).std()

    # -------------------------
    # 거래량 피처
    # -------------------------
    volume_ma_20 = df["Volume"].rolling(window=20).mean()
    volume_std_20 = df["Volume"].rolling(window=20).std()

    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ratio_20"] = safe_divide(df["Volume"], volume_ma_20)
    df["volume_zscore_20"] = safe_divide(df["Volume"] - volume_ma_20, volume_std_20)

    # -------------------------
    # Drawdown / 가격 위치 피처
    # -------------------------
    cumulative_max_close = df["Close"].cummax()
    df["drawdown"] = safe_divide(df["Close"], cumulative_max_close) - 1

    high_20 = df["High"].rolling(window=20).max()
    low_20 = df["Low"].rolling(window=20).min()

    high_60 = df["High"].rolling(window=60).max()
    low_60 = df["Low"].rolling(window=60).min()

    # 최근 고점/저점 구간 내 위치: 0에 가까우면 저점, 1에 가까우면 고점
    df["price_position_20"] = safe_divide(df["Close"] - low_20, high_20 - low_20)
    df["price_position_60"] = safe_divide(df["Close"] - low_60, high_60 - low_60)

    df["close_to_20d_high"] = safe_divide(df["Close"], high_20) - 1
    df["close_to_60d_high"] = safe_divide(df["Close"], high_60) - 1

    # -------------------------
    # RSI
    # -------------------------
    df["rsi_14"] = calculate_rsi(df["Close"], window=14)
    df["rsi_14_scaled"] = df["rsi_14"] / 100

    # -------------------------
    # MACD
    # -------------------------
    macd, macd_signal, macd_hist = calculate_macd(df["Close"])

    # 가격 단위 의존성을 줄이기 위해 Close 대비 비율로 변환
    df["macd_pct"] = safe_divide(macd, df["Close"])
    df["macd_signal_pct"] = safe_divide(macd_signal, df["Close"])
    df["macd_hist_pct"] = safe_divide(macd_hist, df["Close"])

    # -------------------------
    # Bollinger Band
    # -------------------------
    bb_mid = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()

    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    df["bollinger_width_20"] = safe_divide(bb_upper - bb_lower, bb_mid)
    df["bollinger_position_20"] = safe_divide(df["Close"] - bb_lower, bb_upper - bb_lower)

    # -------------------------
    # ATR
    # -------------------------
    atr_14 = calculate_atr(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    df["atr_14_pct"] = safe_divide(atr_14, df["Close"])

    # -------------------------
    # z-score / 추세 기울기
    # -------------------------
    df["return_5d_zscore"] = rolling_zscore(df["return_5d"], window=60)
    df["return_20d_zscore"] = rolling_zscore(df["return_20d"], window=60)

    df["trend_slope_20"] = rolling_log_slope(df["Close"], window=20)
    df["trend_slope_60"] = rolling_log_slope(df["Close"], window=60)

    # 최근 20일 중 상승일 비율
    df["positive_return_ratio_20"] = (
        (df["daily_return"] > 0)
        .rolling(window=20)
        .mean()
    )

    # -------------------------
    # 미래 라벨 생성용 컬럼
    # 주의: 모델 입력 피처로 사용하면 안 됨
    # -------------------------
    df["future_return_20d"] = (
        df["Close"].shift(-PREDICT_HORIZON) / df["Close"] - 1
    )

    # 현재 시점 t에서 미래 t+1 ~ t+20 일간수익률의 표준편차
    df["future_volatility_20d"] = (
        df["daily_return"]
        .shift(-PREDICT_HORIZON)
        .rolling(window=PREDICT_HORIZON)
        .std()
    )

    return df


# =========================
# 5. 라벨 생성
# =========================

def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    방향성 라벨과 위험도 라벨을 분리해서 생성한다.

    direction_label:
    - 상승
    - 하락
    - 횡보

    risk_label:
    - 정상
    - 고변동

    label:
    - 기존 step2 코드 호환을 위해 direction_label과 동일하게 둔다.
    """
    df = df.copy()

    valid_df = df.dropna(
        subset=["future_return_20d", "future_volatility_20d"]
    ).copy()

    if valid_df.empty:
        raise ValueError("라벨 생성에 사용할 유효 데이터가 없습니다.")

    high_vol_threshold = valid_df["future_volatility_20d"].quantile(
        HIGH_VOL_QUANTILE
    )

    # -------------------------
    # 방향성 라벨
    # -------------------------
    conditions_direction = [
        df["future_return_20d"] >= UP_THRESHOLD,
        df["future_return_20d"] <= DOWN_THRESHOLD
    ]

    choices_direction = ["상승", "하락"]

    df["direction_label"] = np.select(
        conditions_direction,
        choices_direction,
        default="횡보"
    )

    # 미래 수익률이 없는 마지막 구간은 NaN 처리
    df.loc[df["future_return_20d"].isna(), "direction_label"] = np.nan

    # -------------------------
    # 위험도 라벨
    # -------------------------
    df["risk_label"] = np.where(
        df["future_volatility_20d"] >= high_vol_threshold,
        "고변동",
        "정상"
    )

    df.loc[df["future_volatility_20d"].isna(), "risk_label"] = np.nan

    # 기존 코드 호환용
    df["label"] = df["direction_label"]

    return df


# =========================
# 6. 데이터 정리 및 저장
# =========================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 학습에 사용할 수 있도록 결측값과 무한값을 정리한다.
    """
    df = df.copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    # 피처 생성 과정의 앞부분 NaN, 미래 라벨 생성 때문에 생기는 뒷부분 NaN 제거
    df = df.dropna().reset_index(drop=True)

    return df


# =========================
# 7. 검증 출력
# =========================

def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    생성된 데이터셋의 기본 상태를 출력한다.
    """
    print("\n데이터 크기:")
    print(df.shape)

    print("\n기간:")
    print(f"{df['Date'].min()} ~ {df['Date'].max()}")

    print("\n컬럼 목록:")
    print(df.columns.tolist())

    print("\n방향성 라벨 분포:")
    print(df["direction_label"].value_counts())

    print("\n방향성 라벨 비율:")
    print((df["direction_label"].value_counts(normalize=True) * 100).round(2))

    print("\n위험도 라벨 분포:")
    print(df["risk_label"].value_counts())

    print("\n위험도 라벨 비율:")
    print((df["risk_label"].value_counts(normalize=True) * 100).round(2))

    print("\n기존 호환용 label 분포:")
    print(df["label"].value_counts())


# =========================
# 8. 실행 함수
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

    print_dataset_summary(final_df)


if __name__ == "__main__":
    main()