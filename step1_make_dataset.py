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

TICKER = "VTI"
TICKER_LOWER = TICKER.lower()

START_DATE = "2001-01-01"
END_DATE = None

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{TICKER_LOWER}_features_labeled.csv")

# =========================
# 라벨 기준
# =========================

# 방향성 라벨은 60거래일, 약 3개월 기준
DIRECTION_HORIZON = 60

# 위험도 라벨은 20거래일, 약 1개월 기준
RISK_HORIZON = 20

# 60일 미래 수익률 기준
UP_THRESHOLD = 0.05        # +5% 이상이면 상승
DOWN_THRESHOLD = -0.05     # -5% 이하이면 하락

# 미래 변동성 상위 20%를 고변동으로 판단
HIGH_VOL_QUANTILE = 0.80


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

def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    ADX, +DI, -DI 계산.

    ADX:
    - 추세 강도 지표
    - 방향 자체가 아니라 추세가 강한지 약한지를 나타낸다.

    +DI:
    - 상승 방향성 강도

    -DI:
    - 하락 방향성 강도
    """
    high = high.copy()
    low = low.copy()
    close = close.copy()

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0
    )

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing에 가까운 ewm 방식 사용
    atr = true_range.ewm(alpha=1 / window, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * safe_divide(plus_dm_smooth, atr)
    minus_di = 100 * safe_divide(minus_dm_smooth, atr)

    dx = 100 * safe_divide(
        (plus_di - minus_di).abs(),
        plus_di + minus_di
    )

    adx = dx.ewm(alpha=1 / window, adjust=False).mean()

    return adx, plus_di, minus_di


def calculate_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    CCI 계산.

    CCI:
    - 현재 가격이 일정 기간 평균 가격에서 얼마나 벗어나 있는지 측정
    - +100 이상: 강한 상승 모멘텀 또는 과열 가능성
    - -100 이하: 강한 하락 모멘텀 또는 과매도 가능성
    """
    typical_price = (high + low + close) / 3
    tp_ma = typical_price.rolling(window=window).mean()

    mean_deviation = typical_price.rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )

    cci = safe_divide(
        typical_price - tp_ma,
        0.015 * mean_deviation
    )

    return cci


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
    
        # MACD Histogram 변화율 / 기울기
    df["macd_hist_diff"] = macd_hist.diff()
    df["macd_hist_slope_5"] = macd_hist.rolling(window=5).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if not np.isnan(x).any() else np.nan,
        raw=True
    )

    # MACD와 Signal의 교차 상태
    # 1  : MACD > Signal, 상승 모멘텀 우위
    # -1 : MACD < Signal, 하락 모멘텀 우위
    df["macd_cross_signal"] = np.where(macd > macd_signal, 1, -1)

    # MACD 교차 변화 감지
    # 1  : 하락 상태에서 상승 상태로 전환
    # -1 : 상승 상태에서 하락 상태로 전환
    # 0  : 변화 없음
    df["macd_cross_event"] = df["macd_cross_signal"].diff().fillna(0)

    df["macd_bullish_cross"] = np.where(df["macd_cross_event"] == 2, 1, 0)
    df["macd_bearish_cross"] = np.where(df["macd_cross_event"] == -2, 1, 0)

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
    
        # -------------------------
    # ADX / DI
    # -------------------------
    adx_14, plus_di_14, minus_di_14 = calculate_adx(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    df["adx_14"] = adx_14
    df["plus_di_14"] = plus_di_14
    df["minus_di_14"] = minus_di_14

    # +DI와 -DI의 차이
    # 양수면 상승 방향성 우위, 음수면 하락 방향성 우위
    df["di_gap_14"] = df["plus_di_14"] - df["minus_di_14"]

    # 방향성 비율
    df["di_ratio_14"] = safe_divide(
        df["plus_di_14"],
        df["minus_di_14"]
    )

    # ADX 기준 추세 강도 플래그
    df["adx_trend_strength"] = np.where(df["adx_14"] >= 25, 1, 0)

    # 상승 추세 조건성 피처
    df["adx_bullish_trend"] = np.where(
        (df["adx_14"] >= 25) & (df["plus_di_14"] > df["minus_di_14"]),
        1,
        0
    )

    # 하락 추세 조건성 피처
    df["adx_bearish_trend"] = np.where(
        (df["adx_14"] >= 25) & (df["minus_di_14"] > df["plus_di_14"]),
        1,
        0
    )

    # -------------------------
    # CCI
    # -------------------------
    cci_20 = calculate_cci(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=20
    )

    df["cci_20"] = cci_20

    # CCI는 값이 커질 수 있으므로 범위를 제한한 뒤 스케일링
    df["cci_20_scaled"] = df["cci_20"].clip(-300, 300) / 300

    # CCI 모멘텀 변화
    df["cci_20_diff"] = df["cci_20"].diff()

    # CCI 구간 플래그
    df["cci_bullish"] = np.where(df["cci_20"] >= 100, 1, 0)
    df["cci_bearish"] = np.where(df["cci_20"] <= -100, 1, 0)
    df["cci_neutral"] = np.where(
        (df["cci_20"] > -100) & (df["cci_20"] < 100),
        1,
        0
    )
    
        # -------------------------
    # 기술적 지표 기반 보조 추세 점수
    # -------------------------

    # 상승 점수
    bullish_score = (
        (df["adx_bullish_trend"] == 1).astype(int)
        + (df["macd_hist_pct"] > 0).astype(int)
        + (df["macd_cross_signal"] == 1).astype(int)
        + (df["cci_20"] > 0).astype(int)
        + (df["rsi_14"] > 50).astype(int)
        + (df["price_ma_20_gap"] > 0).astype(int)
        + (df["ma_gap_20_60"] > 0).astype(int)
    )

    # 하락 점수
    bearish_score = (
        (df["adx_bearish_trend"] == 1).astype(int)
        + (df["macd_hist_pct"] < 0).astype(int)
        + (df["macd_cross_signal"] == -1).astype(int)
        + (df["cci_20"] < 0).astype(int)
        + (df["rsi_14"] < 50).astype(int)
        + (df["price_ma_20_gap"] < 0).astype(int)
        + (df["ma_gap_20_60"] < 0).astype(int)
    )

    # 횡보 점수
    sideways_score = (
        (df["adx_14"] < 20).astype(int)
        + (df["cci_neutral"] == 1).astype(int)
        + (df["bollinger_width_20"] < df["bollinger_width_20"].rolling(window=60).median()).astype(int)
        + (df["ma_gap_20_60"].abs() < 0.02).astype(int)
        + (df["return_20d"].abs() < 0.03).astype(int)
    )

    df["technical_bullish_score"] = bullish_score
    df["technical_bearish_score"] = bearish_score
    df["technical_sideways_score"] = sideways_score

    # 보조 추세 판단
    technical_scores = pd.concat(
        [
            df["technical_bullish_score"],
            df["technical_bearish_score"],
            df["technical_sideways_score"]
        ],
        axis=1
    )

    technical_scores.columns = ["상승", "하락", "횡보"]

    df["technical_trend_label"] = technical_scores.idxmax(axis=1)

    # 점수 차이: 클수록 기술적 판단 확신이 강함
    sorted_scores = np.sort(technical_scores.values, axis=1)

    df["technical_trend_margin"] = (
        sorted_scores[:, -1] - sorted_scores[:, -2]
    )

    # 기술적 상승/하락 균형값
    df["technical_trend_balance"] = (
        df["technical_bullish_score"] - df["technical_bearish_score"]
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

    # 방향성 판단용: 미래 60거래일 수익률
    df["future_return_60d"] = (
        df["Close"].shift(-DIRECTION_HORIZON) / df["Close"] - 1
    )

    # 참고용: 기존 20일 미래 수익률도 저장
    df["future_return_20d"] = (
        df["Close"].shift(-RISK_HORIZON) / df["Close"] - 1
    )

    # 위험도 판단용: 미래 20거래일 변동성
    df["future_volatility_20d"] = (
        df["daily_return"]
        .shift(-RISK_HORIZON)
        .rolling(window=RISK_HORIZON)
        .std()
    )

    # 참고용: 미래 60거래일 변동성
    df["future_volatility_60d"] = (
        df["daily_return"]
        .shift(-DIRECTION_HORIZON)
        .rolling(window=DIRECTION_HORIZON)
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
    - 상승: 미래 60거래일 수익률이 +5% 이상
    - 하락: 미래 60거래일 수익률이 -5% 이하
    - 횡보: 그 외 구간

    risk_label:
    - 고변동: 미래 20거래일 변동성이 전체 상위 20%
    - 정상: 그 외 구간
    """
    df = df.copy()

    required_cols = [
        "future_return_60d",
        "future_volatility_20d"
    ]

    valid_df = df.dropna(subset=required_cols).copy()

    if valid_df.empty:
        raise ValueError("라벨 생성에 사용할 유효 데이터가 없습니다.")

    high_vol_threshold = valid_df["future_volatility_20d"].quantile(
        HIGH_VOL_QUANTILE
    )

    # -------------------------
    # 방향성 라벨: 60거래일 기준
    # -------------------------
    conditions_direction = [
        df["future_return_60d"] >= UP_THRESHOLD,
        df["future_return_60d"] <= DOWN_THRESHOLD
    ]

    choices_direction = ["상승", "하락"]

    df["direction_label"] = np.select(
        conditions_direction,
        choices_direction,
        default="횡보"
    )

    df.loc[df["future_return_60d"].isna(), "direction_label"] = np.nan

    # -------------------------
    # 위험도 라벨: 20거래일 변동성 기준
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