# Python 3.10+
# 필요 패키지:
# pip install yfinance pandas numpy scikit-learn matplotlib joblib

import os

# Windows/Python 3.13 환경에서 Tkinter 백엔드 오류 방지
os.environ["MPLBACKEND"] = "Agg"

import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# =========================
# 1. 기본 설정
# =========================

TARGET_TICKER = "VTI"
TARGET_TICKER_LOWER = TARGET_TICKER.lower()

DATA_PATH = f"data/{TARGET_TICKER_LOWER}_features_labeled.csv"

MODEL_PATH = f"models/{TARGET_TICKER_LOWER}_two_stage_model.pkl"
META_PATH = f"models/{TARGET_TICKER_LOWER}_two_stage_metadata.json"

RESULT_DIR = "results"

BACKTEST_RESULT_PATH = os.path.join(
    RESULT_DIR,
    f"{TARGET_TICKER_LOWER}_multi_asset_two_stage_backtest_result.csv"
)

BACKTEST_SUMMARY_PATH = os.path.join(
    RESULT_DIR,
    f"{TARGET_TICKER_LOWER}_multi_asset_two_stage_backtest_summary.json"
)

EQUITY_CURVE_PATH = os.path.join(
    RESULT_DIR,
    f"{TARGET_TICKER_LOWER}_multi_asset_two_stage_equity_curve.png"
)

DRAWDOWN_PATH = os.path.join(
    RESULT_DIR,
    f"{TARGET_TICKER_LOWER}_multi_asset_two_stage_drawdown_curve.png"
)

ALLOCATION_COUNT_PATH = os.path.join(
    RESULT_DIR,
    f"{TARGET_TICKER_LOWER}_multi_asset_two_stage_allocation_count.csv"
)

TEST_SIZE = 0.5
BACKTEST_START_DATE = None

INITIAL_CAPITAL = 100_000_000

STOCK_TICKER = TARGET_TICKER
BOND_TICKER = "IEF"
CASH_TICKER = "BIL"

DIRECTION_MIN_CONFIDENCE = 30.0
DIRECTION_MIN_MARGIN = 3.0
DOWN_ACCEPT_THRESHOLD = 50.0
HIGH_VOL_PROBA_THRESHOLD = 45.0


# =========================
# 2. 데이터 로드
# =========================

def load_dataset(path: str) -> pd.DataFrame:
    """
    step1_make_dataset.py에서 생성한 QQQ 피처/라벨 데이터셋을 불러온다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {path}\n"
            "먼저 step1_make_dataset.py를 실행하세요."
        )

    df = pd.read_csv(path)

    required_cols = [
        "Date",
        "Close",
        "direction_label",
        "risk_label"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"필수 컬럼 누락: {missing_cols}\n"
            "수정된 step1_make_dataset.py를 먼저 실행해야 합니다."
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_model_package(model_path: str) -> dict:
    """
    step2_train_two_models.py에서 저장한 2단계 모델 패키지를 불러온다.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {model_path}\n"
            "먼저 step2_train_two_models.py를 실행하세요."
        )

    package = joblib.load(model_path)

    required_keys = [
        "direction_model",
        "direction_imputer",
        "direction_encoder",
        "direction_feature_cols",
        "risk_model",
        "risk_imputer",
        "risk_encoder",
        "risk_feature_cols"
    ]

    missing_keys = [key for key in required_keys if key not in package]

    if missing_keys:
        raise ValueError(
            f"2단계 모델 패키지 필수 항목 누락: {missing_keys}\n"
            "qqq_two_stage_model.pkl을 사용해야 합니다."
        )

    return package


def load_metadata(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 3. 자산 가격 데이터 다운로드
# =========================

def download_asset_monthly_returns(
    ticker: str,
    start_date: str,
    end_date: str | None = None
) -> pd.DataFrame:
    """
    yfinance로 자산 일봉 데이터를 받고,
    월말 종가 기준 다음 달 수익률을 계산한다.

    반환 컬럼:
    - year_month
    - {ticker}_next_month_return
    """
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"{ticker} 데이터 다운로드 실패")

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"{ticker} 데이터에 Date 또는 Close 컬럼이 없습니다.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["year_month"] = df["Date"].dt.to_period("M")

    month_end = (
        df.groupby("year_month", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    month_end[f"{ticker}_next_month_return"] = (
        month_end["Close"].shift(-1) / month_end["Close"] - 1
    )

    result = month_end[["year_month", f"{ticker}_next_month_return"]].copy()
    result = result.dropna().reset_index(drop=True)

    return result


# =========================
# 4. 백테스트 월말 데이터 생성
# =========================

def restrict_to_backtest_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    백테스트 구간 설정.

    BACKTEST_START_DATE가 있으면 날짜 기준 사용.
    없으면 TEST_SIZE 기준으로 뒤 구간 사용.
    """
    if BACKTEST_START_DATE is not None:
        test_df = df[df["Date"] >= BACKTEST_START_DATE].copy().reset_index(drop=True)

        if test_df.empty:
            raise ValueError(
                f"BACKTEST_START_DATE 이후 데이터가 없습니다: {BACKTEST_START_DATE}"
            )

        return test_df

    split_index = int(len(df) * (1 - TEST_SIZE))
    return df.iloc[split_index:].copy().reset_index(drop=True)


def get_month_end_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    QQQ 피처 데이터에서 각 월의 마지막 거래일 행을 추출한다.
    """
    temp = df.copy()
    temp["year_month"] = temp["Date"].dt.to_period("M")

    month_end_df = (
        temp.groupby("year_month", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    return month_end_df


def add_multi_asset_returns(month_end_df: pd.DataFrame) -> pd.DataFrame:
    """
    STOCK_TICKER/IEF/BIL의 다음 달 수익률을 월말 데이터에 병합한다.
    """
    df = month_end_df.copy()

    # STOCK_TICKER는 기존 피처 데이터의 Close를 사용
    df[f"{STOCK_TICKER}_next_month_return"] = df["Close"].shift(-1) / df["Close"] - 1

    start_date = str(df["Date"].min().date())
    end_date = None

    print(f"\n자산 수익률 다운로드 시작일: {start_date}")
    print(f"채권 ETF: {BOND_TICKER}")
    print(f"현금성 ETF: {CASH_TICKER}")

    bond_returns = download_asset_monthly_returns(
        ticker=BOND_TICKER,
        start_date=start_date,
        end_date=end_date
    )

    cash_returns = download_asset_monthly_returns(
        ticker=CASH_TICKER,
        start_date=start_date,
        end_date=end_date
    )

    df = df.merge(
        bond_returns,
        on="year_month",
        how="left"
    )

    df = df.merge(
        cash_returns,
        on="year_month",
        how="left"
    )

    df = df.rename(columns={
        f"{BOND_TICKER}_next_month_return": "IEF_next_month_return",
        f"{CASH_TICKER}_next_month_return": "BIL_next_month_return"
    })

    required_return_cols = [
        f"{STOCK_TICKER}_next_month_return",
        "IEF_next_month_return",
        "BIL_next_month_return"
    ]

    missing_return_cols = [
        col for col in required_return_cols
        if col not in df.columns
    ]

    if missing_return_cols:
        raise ValueError(f"수익률 컬럼 생성 실패: {missing_return_cols}")

    before_rows = len(df)

    df = df.dropna(subset=required_return_cols).reset_index(drop=True)

    after_rows = len(df)

    if after_rows < before_rows:
        print(f"수익률 결측으로 제거된 월 수: {before_rows - after_rows}")

    return df


# =========================
# 5. 예측 보조 함수
# =========================

def make_probability_dict(model, label_encoder, X_processed) -> dict[str, float]:
    proba = model.predict_proba(X_processed)[0]

    return {
        label_encoder.classes_[i]: round(float(proba[i]) * 100, 4)
        for i in range(len(label_encoder.classes_))
    }


def apply_direction_decision_rule(
    pred_label: str,
    proba_result: dict[str, float],
    min_confidence: float = DIRECTION_MIN_CONFIDENCE,
    min_margin: float = DIRECTION_MIN_MARGIN,
    down_accept_threshold: float = DOWN_ACCEPT_THRESHOLD
) -> str:
    """
    방향성 모델 예측 확률을 기준으로 최종 방향성을 판단한다.

    규칙:
    - 최고 확률이 낮으면 불확실
    - 1등과 2등 차이가 작으면 불확실
    - 하락은 확률이 충분히 높을 때만 인정
    """
    sorted_probs = sorted(
        proba_result.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if len(sorted_probs) < 2:
        return pred_label

    top_label, top_prob = sorted_probs[0]
    second_label, second_prob = sorted_probs[1]

    if top_prob < min_confidence:
        return "불확실"

    if top_prob - second_prob < min_margin:
        return "불확실"

    if top_label == "하락" and top_prob < down_accept_threshold:
        return "불확실"

    return top_label


def apply_high_vol_threshold(
    risk_proba: dict[str, float],
    threshold: float = HIGH_VOL_PROBA_THRESHOLD
) -> str:
    """
    고변동 확률 기준 보정.
    """
    high_vol_prob = risk_proba.get("고변동", 0.0)

    if high_vol_prob >= threshold:
        return "고변동"

    return "정상"


def predict_one_row(row: pd.DataFrame, model_package: dict) -> dict:
    """
    월말 1개 행에 대해 방향성/위험도 예측.
    """
    direction_model = model_package["direction_model"]
    direction_imputer = model_package["direction_imputer"]
    direction_encoder = model_package["direction_encoder"]
    direction_feature_cols = model_package["direction_feature_cols"]

    risk_model = model_package["risk_model"]
    risk_imputer = model_package["risk_imputer"]
    risk_encoder = model_package["risk_encoder"]
    risk_feature_cols = model_package["risk_feature_cols"]

    missing_direction = [
        col for col in direction_feature_cols
        if col not in row.columns
    ]

    missing_risk = [
        col for col in risk_feature_cols
        if col not in row.columns
    ]

    if missing_direction:
        raise ValueError(f"방향성 모델 입력 피처 누락: {missing_direction}")

    if missing_risk:
        raise ValueError(f"위험도 모델 입력 피처 누락: {missing_risk}")

    # 방향성
    X_direction = row[direction_feature_cols]
    X_direction_processed = direction_imputer.transform(X_direction)

    direction_pred_encoded = direction_model.predict(X_direction_processed)[0]
    direction_pred_label = direction_encoder.inverse_transform([direction_pred_encoded])[0]

    direction_proba = make_probability_dict(
        model=direction_model,
        label_encoder=direction_encoder,
        X_processed=X_direction_processed
    )

    final_direction = apply_direction_decision_rule(
        pred_label=direction_pred_label,
        proba_result=direction_proba
    )

    # 위험도
    X_risk = row[risk_feature_cols]
    X_risk_processed = risk_imputer.transform(X_risk)

    risk_pred_encoded = risk_model.predict(X_risk_processed)[0]
    risk_pred_label = risk_encoder.inverse_transform([risk_pred_encoded])[0]

    risk_proba = make_probability_dict(
        model=risk_model,
        label_encoder=risk_encoder,
        X_processed=X_risk_processed
    )

    final_risk = apply_high_vol_threshold(
        risk_proba=risk_proba,
        threshold=HIGH_VOL_PROBA_THRESHOLD
    )

    return {
        "direction_pred_label": direction_pred_label,
        "final_direction": final_direction,
        "direction_proba": direction_proba,
        "risk_pred_label": risk_pred_label,
        "final_risk": final_risk,
        "risk_proba": risk_proba
    }


# =========================
# 6. 자산배분 규칙
# =========================

def normalize_allocation(allocation: dict[str, float]) -> dict[str, float]:
    """
    자산 비중 합계를 1로 정규화한다.
    """
    total = sum(allocation.values())

    if total <= 0:
        return {
            "stock": 0.50,
            "bond": 0.30,
            "cash": 0.20
        }

    return {
        key: value / total
        for key, value in allocation.items()
    }


def clip_allocation(
    allocation: dict[str, float],
    min_stock: float = 0.20,
    max_stock: float = 0.90,
    min_bond: float = 0.05,
    max_bond: float = 0.60,
    min_cash: float = 0.05,
    max_cash: float = 0.50
) -> dict[str, float]:
    """
    극단적인 비중을 방지하기 위해 최소/최대 비중을 제한한다.
    """
    clipped = {
        "stock": min(max(allocation["stock"], min_stock), max_stock),
        "bond": min(max(allocation["bond"], min_bond), max_bond),
        "cash": min(max(allocation["cash"], min_cash), max_cash)
    }

    return normalize_allocation(clipped)


def recommend_allocation(
    final_direction: str,
    direction_proba: dict[str, float],
    final_risk: str,
    risk_proba: dict[str, float]
) -> tuple[dict[str, float], str]:
    """
    방향성 확률과 고변동 확률을 기반으로 주식/채권/현금 비중을 연속적으로 계산한다.

    핵심:
    - 상승/하락/횡보를 고정 라벨로만 쓰지 않음
    - 각 추세 확률을 가중 평균하여 기본 비중 계산
    - 고변동 확률이 높을수록 주식 비중을 줄이고 채권/현금 비중을 높임
    """

    # -------------------------
    # 1. 방향성 확률
    # -------------------------
    p_up = direction_proba.get("상승", 0.0) / 100
    p_down = direction_proba.get("하락", 0.0) / 100
    p_sideways = direction_proba.get("횡보", 0.0) / 100

    direction_sum = p_up + p_down + p_sideways

    if direction_sum <= 0:
        p_up = 1 / 3
        p_down = 1 / 3
        p_sideways = 1 / 3
    else:
        p_up /= direction_sum
        p_down /= direction_sum
        p_sideways /= direction_sum

    # -------------------------
    # 2. 각 추세별 기준 포트폴리오
    # -------------------------
    up_portfolio = {
        "stock": 0.95,
        "bond": 0.03,
        "cash": 0.02
    }

    down_portfolio = {
        "stock": 0.40,
        "bond": 0.40,
        "cash": 0.20
    }

    sideways_portfolio = {
        "stock": 0.70,
        "bond": 0.20,
        "cash": 0.10
    }
    
    # up_portfolio = {
    #     "stock": 0.85,
    #     "bond": 0.10,
    #     "cash": 0.05
    # }

    # down_portfolio = {
    #     "stock": 0.35,
    #     "bond": 0.40,
    #     "cash": 0.25
    # }

    # sideways_portfolio = {
    #     "stock": 0.60,
    #     "bond": 0.25,
    #     "cash": 0.15
    # }

    # -------------------------
    # 3. 방향성 확률 기반 기본 비중
    # -------------------------
    stock_weight = (
        p_up * up_portfolio["stock"]
        + p_down * down_portfolio["stock"]
        + p_sideways * sideways_portfolio["stock"]
    )

    bond_weight = (
        p_up * up_portfolio["bond"]
        + p_down * down_portfolio["bond"]
        + p_sideways * sideways_portfolio["bond"]
    )

    cash_weight = (
        p_up * up_portfolio["cash"]
        + p_down * down_portfolio["cash"]
        + p_sideways * sideways_portfolio["cash"]
    )

    # -------------------------
    # 4. 고변동 확률 기반 위험 조정
    # -------------------------
    p_high_vol = risk_proba.get("고변동", 0.0) / 100

    # 고변동 확률이 높을수록 주식 비중을 줄임
    # 최대 25%p까지 주식 비중 축소
    stock_reduction = 0.18 * p_high_vol

    stock_weight -= stock_reduction

    # 줄어든 주식 비중을 채권과 현금으로 배분
    bond_weight += stock_reduction * 0.45
    cash_weight += stock_reduction * 0.55

    # -------------------------
    # 5. 하락 확률 추가 방어 보정
    # -------------------------
    # 하락 확률이 높을수록 주식 비중을 소폭 더 줄임
    # 단, 이미 방향성 가중 평균에 반영되어 있으므로 과도하게 줄이지 않음
    extra_down_risk = max(0.0, p_down - 0.35)

    extra_stock_reduction = 0.08 * extra_down_risk

    stock_weight -= extra_stock_reduction
    bond_weight += extra_stock_reduction * 0.60
    cash_weight += extra_stock_reduction * 0.40

    # -------------------------
    # 6. 불확실성 보정
    # -------------------------
    # 방향성 확률이 서로 비슷하면 중립 포트폴리오 쪽으로 이동
    sorted_direction_probs = sorted(
        [p_up, p_down, p_sideways],
        reverse=True
    )

    top_prob = sorted_direction_probs[0]
    second_prob = sorted_direction_probs[1]
    margin = top_prob - second_prob

    # margin이 낮을수록 불확실성이 크다고 판단
    uncertainty_strength = max(0.0, min(1.0, (0.15 - margin) / 0.15))
    
    neutral_portfolio = {
        "stock": 0.65,
        "bond": 0.25,
        "cash": 0.10
    }
    
    # neutral_portfolio = {
    #     "stock": 0.55,
    #     "bond": 0.30,
    #     "cash": 0.15
    # }

    stock_weight = (
        stock_weight * (1 - uncertainty_strength)
        + neutral_portfolio["stock"] * uncertainty_strength
    )

    bond_weight = (
        bond_weight * (1 - uncertainty_strength)
        + neutral_portfolio["bond"] * uncertainty_strength
    )

    cash_weight = (
        cash_weight * (1 - uncertainty_strength)
        + neutral_portfolio["cash"] * uncertainty_strength
    )

    # -------------------------
    # 7. 정규화 및 제한
    # -------------------------
    allocation = {
        "stock": stock_weight,
        "bond": bond_weight,
        "cash": cash_weight
    }

    allocation = normalize_allocation(allocation)
    allocation = clip_allocation(allocation)

    reason = (
        "확률 기반 동적 배분: "
        f"상승 {p_up * 100:.1f}%, "
        f"하락 {p_down * 100:.1f}%, "
        f"횡보 {p_sideways * 100:.1f}%, "
        f"고변동 {p_high_vol * 100:.1f}% 반영"
    )

    return allocation, reason


# =========================
# 7. 백테스트 실행
# =========================

def run_backtest(
    df: pd.DataFrame,
    model_package: dict,
    initial_capital: float
) -> pd.DataFrame:
    """
    월말마다:
    1. 방향성/위험도 예측
    2. 자산배분 결정
    3. QQQ/IEF/BIL 실제 다음 달 수익률 적용
    4. 벤치마크와 비교
    """
    test_df = restrict_to_backtest_period(df)
    month_end_df = get_month_end_rows(test_df)
    month_end_df = add_multi_asset_returns(month_end_df)

    if len(month_end_df) < 3:
        raise ValueError("백테스트 가능한 월별 데이터가 너무 적습니다.")

    records = []

    strategy_capital = initial_capital
    stock_capital = initial_capital
    benchmark_60_40_capital = initial_capital
    static_50_30_20_capital = initial_capital

    for i in range(len(month_end_df)):
        current_row = month_end_df.iloc[i:i + 1].copy()

        date = current_row["Date"].iloc[0]
        close = float(current_row["Close"].iloc[0])

        stock_return = float(current_row[f"{STOCK_TICKER}_next_month_return"].iloc[0])
        bond_return = float(current_row[f"{BOND_TICKER}_next_month_return"].iloc[0])
        cash_return = float(current_row[f"{CASH_TICKER}_next_month_return"].iloc[0])

        prediction = predict_one_row(
            row=current_row,
            model_package=model_package
        )

        allocation, allocation_reason = recommend_allocation(
            final_direction=prediction["final_direction"],
            direction_proba=prediction["direction_proba"],
            final_risk=prediction["final_risk"],
            risk_proba=prediction["risk_proba"]
        )

        stock_weight = allocation["stock"]
        bond_weight = allocation["bond"]
        cash_weight = allocation["cash"]

        strategy_return = (
            stock_weight * stock_return
            + bond_weight * bond_return
            + cash_weight * cash_return
        )

        stock_benchmark_return = stock_return

        benchmark_60_40_return = (
            0.60 * stock_return
            + 0.40 * bond_return
        )

        static_50_30_20_return = (
            0.50 * stock_return
            + 0.30 * bond_return
            + 0.20 * cash_return
        )

        strategy_capital *= (1 + strategy_return)
        stock_capital *= (1 + stock_benchmark_return)
        benchmark_60_40_capital *= (1 + benchmark_60_40_return)
        static_50_30_20_capital *= (1 + static_50_30_20_return)

        direction_proba = prediction["direction_proba"]
        risk_proba = prediction["risk_proba"]

        record = {
            "Date": date,
            "Close": close,

            "direction_pred_label": prediction["direction_pred_label"],
            "final_direction": prediction["final_direction"],
            "risk_pred_label": prediction["risk_pred_label"],
            "final_risk": prediction["final_risk"],

            "prob_상승": direction_proba.get("상승", 0.0),
            "prob_하락": direction_proba.get("하락", 0.0),
            "prob_횡보": direction_proba.get("횡보", 0.0),

            "prob_고변동": risk_proba.get("고변동", 0.0),
            "prob_정상": risk_proba.get("정상", 0.0),

            "stock_weight": stock_weight,
            "bond_weight": bond_weight,
            "cash_weight": cash_weight,
            "allocation_reason": allocation_reason,

            f"{STOCK_TICKER}_next_month_return": stock_return,
            f"{BOND_TICKER}_next_month_return": bond_return,
            f"{CASH_TICKER}_next_month_return": cash_return,

            "strategy_return": strategy_return,
            "stock_benchmark_return": stock_benchmark_return,
            "benchmark_60_40_return": benchmark_60_40_return,
            "static_50_30_20_return": static_50_30_20_return,

            "strategy_capital": strategy_capital,
            "stock_capital": stock_capital,
            "benchmark_60_40_capital": benchmark_60_40_capital,
            "static_50_30_20_capital": static_50_30_20_capital
        }

        records.append(record)

    result_df = pd.DataFrame(records)

    return result_df


# =========================
# 8. 성과 지표
# =========================

def calculate_cagr(final_capital: float, initial_capital: float, months: int) -> float:
    if months <= 0:
        return np.nan

    if initial_capital <= 0 or final_capital <= 0:
        return np.nan

    years = months / 12

    return (final_capital / initial_capital) ** (1 / years) - 1


def calculate_mdd(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1

    return drawdown.min()


def calculate_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0
) -> float:
    excess_return = returns - (risk_free_rate / periods_per_year)
    std = excess_return.std()

    if std == 0 or np.isnan(std):
        return np.nan

    return (excess_return.mean() / std) * np.sqrt(periods_per_year)


def make_metric_block(
    result_df: pd.DataFrame,
    capital_col: str,
    return_col: str,
    initial_capital: float
) -> dict:
    months = len(result_df)
    final_capital = float(result_df[capital_col].iloc[-1])

    total_return = final_capital / initial_capital - 1
    cagr = calculate_cagr(final_capital, initial_capital, months)
    mdd = calculate_mdd(result_df[capital_col])
    sharpe = calculate_sharpe_ratio(result_df[return_col])

    return {
        "final_capital": round(final_capital, 2),
        "total_return": round(float(total_return), 6),
        "cagr": round(float(cagr), 6),
        "mdd": round(float(mdd), 6),
        "sharpe": round(float(sharpe), 6)
    }


def summarize_performance(result_df: pd.DataFrame, initial_capital: float) -> dict:
    allocation_counts = (
        result_df[["final_direction", "final_risk"]]
        .value_counts()
        .reset_index(name="count")
    )
    
    average_weights = {
        "avg_stock_weight": round(float(result_df["stock_weight"].mean()), 6),
        "avg_bond_weight": round(float(result_df["bond_weight"].mean()), 6),
        "avg_cash_weight": round(float(result_df["cash_weight"].mean()), 6),
        "min_stock_weight": round(float(result_df["stock_weight"].min()), 6),
        "max_stock_weight": round(float(result_df["stock_weight"].max()), 6)
    }

    summary = {
        "backtest_start": str(result_df["Date"].iloc[0]),
        "backtest_end": str(result_df["Date"].iloc[-1]),
        "months": int(len(result_df)),
        "initial_capital": int(initial_capital),

        "assets": {
            "stock": STOCK_TICKER,
            "bond": BOND_TICKER,
            "cash": CASH_TICKER
        },
        
        "average_weights": average_weights,

        "strategy": make_metric_block(
            result_df,
            capital_col="strategy_capital",
            return_col="strategy_return",
            initial_capital=initial_capital
        ),

        "stock_buy_hold": make_metric_block(
            result_df,
            capital_col="stock_capital",
            return_col="stock_benchmark_return",
            initial_capital=initial_capital
        ),

        "benchmark_60_40": make_metric_block(
            result_df,
            capital_col="benchmark_60_40_capital",
            return_col="benchmark_60_40_return",
            initial_capital=initial_capital
        ),

        "static_50_30_20": make_metric_block(
            result_df,
            capital_col="static_50_30_20_capital",
            return_col="static_50_30_20_return",
            initial_capital=initial_capital
        ),

        "direction_min_confidence": DIRECTION_MIN_CONFIDENCE,
        "direction_min_margin": DIRECTION_MIN_MARGIN,
        "down_accept_threshold": DOWN_ACCEPT_THRESHOLD,
        "high_vol_proba_threshold": HIGH_VOL_PROBA_THRESHOLD,

        "allocation_count": allocation_counts.to_dict(orient="records"),

        "note": (
            f"{STOCK_TICKER}/{BOND_TICKER}/{CASH_TICKER} 실제 월별 수익률을 반영한 "
            "2단계 자산배분 백테스트입니다. "
            "전략은 방향성 모델과 위험도 모델의 예측 결과를 기반으로 "
            "주식/채권/현금 비중을 조절합니다."
        )
    }

    return summary


# =========================
# 9. 그래프 저장
# =========================

def save_equity_curve(result_df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(result_df["Date"], result_df["strategy_capital"], label="Two-Stage Strategy")
    ax.plot(
        result_df["Date"],
        result_df["stock_capital"],
        label=f"{STOCK_TICKER} Buy & Hold"
    )
    ax.plot(result_df["Date"], result_df["benchmark_60_40_capital"], label=f"60/40 {STOCK_TICKER}-{BOND_TICKER}")
    ax.plot(result_df["Date"], result_df["static_50_30_20_capital"], label="Static 50/30/20")

    ax.set_title("Multi-Asset Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    plt.close("all")

    print(f"자산 곡선 저장 완료: {save_path}")


def save_drawdown_curve(result_df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    curves = {
        "Two-Stage Strategy": "strategy_capital",
        f"{STOCK_TICKER} Buy & Hold": "stock_capital",
        f"60/40 {STOCK_TICKER}-{BOND_TICKER}": "benchmark_60_40_capital",
        "Static 50/30/20": "static_50_30_20_capital"
    }

    for label, col in curves.items():
        running_max = result_df[col].cummax()
        drawdown = result_df[col] / running_max - 1
        ax.plot(result_df["Date"], drawdown, label=label)

    ax.set_title("Multi-Asset Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    plt.close("all")

    print(f"Drawdown 그래프 저장 완료: {save_path}")


# =========================
# 10. 저장 및 출력
# =========================

def save_results(result_df: pd.DataFrame, summary: dict):
    os.makedirs(RESULT_DIR, exist_ok=True)

    result_df.to_csv(BACKTEST_RESULT_PATH, index=False, encoding="utf-8-sig")

    with open(BACKTEST_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    allocation_count_df = (
        result_df[["final_direction", "final_risk"]]
        .value_counts()
        .reset_index(name="count")
    )

    allocation_count_df.to_csv(
        ALLOCATION_COUNT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"백테스트 상세 결과 저장 완료: {BACKTEST_RESULT_PATH}")
    print(f"백테스트 요약 저장 완료: {BACKTEST_SUMMARY_PATH}")
    print(f"판단 조합 빈도 저장 완료: {ALLOCATION_COUNT_PATH}")


def print_metric_block(name: str, block: dict):
    print(f"\n[{name}]")
    print(f"최종 자산: {block['final_capital']:,.0f}원")
    print(f"총수익률: {block['total_return'] * 100:.2f}%")
    print(f"CAGR: {block['cagr'] * 100:.2f}%")
    print(f"MDD: {block['mdd'] * 100:.2f}%")
    print(f"Sharpe: {block['sharpe']:.4f}")


def print_summary(summary: dict):
    print("\n==============================")
    print("Multi-Asset Two-Stage 백테스트 요약")
    print("==============================")

    print(f"기간: {summary['backtest_start']} ~ {summary['backtest_end']}")
    print(f"개월 수: {summary['months']}")
    print(f"초기 자본: {summary['initial_capital']:,}원")

    print("\n사용 자산:")
    print(f"주식: {summary['assets']['stock']}")
    print(f"채권: {summary['assets']['bond']}")
    print(f"현금성: {summary['assets']['cash']}")
    
    if "average_weights" in summary:
        weights = summary["average_weights"]

        print("\n[평균 자산 비중]")
        print(f"평균 주식 비중: {weights['avg_stock_weight'] * 100:.2f}%")
        print(f"평균 채권 비중: {weights['avg_bond_weight'] * 100:.2f}%")
        print(f"평균 현금 비중: {weights['avg_cash_weight'] * 100:.2f}%")
        print(f"최소 주식 비중: {weights['min_stock_weight'] * 100:.2f}%")
        print(f"최대 주식 비중: {weights['max_stock_weight'] * 100:.2f}%")

    print_metric_block("Two-Stage Strategy", summary["strategy"])
    print_metric_block(f"{STOCK_TICKER} Buy & Hold", summary["stock_buy_hold"])
    print_metric_block(f"60/40 {STOCK_TICKER}-{BOND_TICKER}", summary["benchmark_60_40"])
    print_metric_block("Static 50/30/20", summary["static_50_30_20"])

    print(f"방향성 최소 확신도: {summary['direction_min_confidence']}%")
    print(f"방향성 1등-2등 최소 차이: {summary['direction_min_margin']}%p")
    print(f"하락 인정 최소 확률: {summary['down_accept_threshold']}%")
    print(f"고변동 판단 기준: {summary['high_vol_proba_threshold']}%")

    print("\n[판단 조합 빈도]")
    for item in summary["allocation_count"]:
        print(
            f"- 방향성={item['final_direction']}, "
            f"위험도={item['final_risk']}: {item['count']}회"
        )

    print("\n주의:")
    print(summary["note"])


def print_recent_results(result_df: pd.DataFrame, n: int = 10):
    print(f"\n최근 {n}개 월별 결과:")

    display_cols = [
        "Date",
        "final_direction",
        "final_risk",
        "prob_상승",
        "prob_하락",
        "prob_횡보",
        "prob_고변동",
        "stock_weight",
        "bond_weight",
        "cash_weight",
        "stock_next_month_return",
        "bond_next_month_return",
        "cash_next_month_return",
        "strategy_return",
        "stock_benchmark_return",
        "benchmark_60_40_return",
        "static_50_30_20_return",
        "strategy_capital",
        "stock_capital",
        "benchmark_60_40_capital",
        "static_50_30_20_capital"
    ]

    existing_cols = [col for col in display_cols if col in result_df.columns]
    print(result_df[existing_cols].tail(n).to_string(index=False))


# =========================
# 11. 메인
# =========================

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("[1] 데이터 불러오기")
    df = load_dataset(DATA_PATH)
    print(f"데이터 크기: {df.shape}")
    print(f"기간: {df['Date'].min()} ~ {df['Date'].max()}")

    print("\n[2] 2단계 모델 불러오기")
    model_package = load_model_package(MODEL_PATH)
    metadata = load_metadata(META_PATH)

    if metadata:
        print("모델 메타데이터 로드 완료")
        print(f"모델 구조: {metadata.get('model_structure', 'unknown')}")
        print(f"방향성 라벨: {metadata.get('direction_labels', [])}")
        print(f"위험도 라벨: {metadata.get('risk_labels', [])}")

    print("\n[3] Multi-Asset 백테스트 실행")
    result_df = run_backtest(
        df=df,
        model_package=model_package,
        initial_capital=INITIAL_CAPITAL
    )

    print("\n[4] 성과 지표 계산")
    summary = summarize_performance(
        result_df=result_df,
        initial_capital=INITIAL_CAPITAL
    )

    print_summary(summary)

    print("\n[5] 결과 저장")
    save_results(result_df, summary)

    print("\n[6] 그래프 저장")
    save_equity_curve(result_df, EQUITY_CURVE_PATH)
    save_drawdown_curve(result_df, DRAWDOWN_PATH)

    print_recent_results(result_df, n=10)

    print("\n완료")


if __name__ == "__main__":
    main()