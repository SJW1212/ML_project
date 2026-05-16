# Python 3.10+
# 필요 패키지:
# pip install pandas numpy scikit-learn matplotlib joblib

import os

# Matplotlib GUI 백엔드 비활성화
# Windows + Python 3.13 + TkAgg 환경에서 Tcl/Tk 스레드 오류 방지
os.environ["MPLBACKEND"] = "Agg"

import json
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# =========================
# 1. 기본 설정
# =========================

DATA_PATH = "data/qqq_features_labeled.csv"

# step2_train_two_models.py에서 저장한 모델
MODEL_PATH = "models/qqq_two_stage_model.pkl"
META_PATH = "models/qqq_two_stage_metadata.json"

RESULT_DIR = "results"

BACKTEST_RESULT_PATH = os.path.join(RESULT_DIR, "two_stage_backtest_result.csv")
BACKTEST_SUMMARY_PATH = os.path.join(RESULT_DIR, "two_stage_backtest_summary.json")
EQUITY_CURVE_PATH = os.path.join(RESULT_DIR, "two_stage_equity_curve.png")
DRAWDOWN_PATH = os.path.join(RESULT_DIR, "two_stage_drawdown_curve.png")
ALLOCATION_COUNT_PATH = os.path.join(RESULT_DIR, "two_stage_allocation_count.csv")

# step2와 동일하게 뒤 20% 테스트 구간만 백테스트
TEST_SIZE = 0.2

# 초기 투자금
INITIAL_CAPITAL = 100_000_000

# 현재는 QQQ만 실제 수익률로 사용
# 이후 IEF/BIL 데이터를 붙이면 실제 채권/현금성 ETF 수익률로 교체
DEFAULT_BOND_MONTHLY_RETURN = 0.0
DEFAULT_CASH_MONTHLY_RETURN = 0.0

# 방향성 불확실 판단 기준
DIRECTION_MIN_CONFIDENCE = 40.0
DIRECTION_MIN_MARGIN = 8.0

# 고변동 판단 기준
# risk_model.predict()만 사용하면 고변동을 많이 놓칠 수 있으므로 확률 기준 사용
HIGH_VOL_PROBA_THRESHOLD = 35.0


# =========================
# 2. 파일 로드
# =========================

def load_dataset(path: str) -> pd.DataFrame:
    """
    step1_make_dataset.py에서 생성한 데이터셋을 불러온다.
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
            "기존 단일 모델 파일이 아니라 qqq_two_stage_model.pkl을 사용해야 합니다."
        )

    return package


def load_metadata(meta_path: str) -> dict:
    """
    모델 메타데이터를 불러온다.
    없으면 빈 dict를 반환한다.
    """
    if not os.path.exists(meta_path):
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 3. 월말 리밸런싱 데이터 생성
# =========================

def restrict_to_test_period(df: pd.DataFrame, test_size: float) -> pd.DataFrame:
    """
    step2와 동일하게 뒤 20% 구간만 백테스트에 사용한다.

    이유:
    - 앞 80%는 모델 학습 구간
    - 뒤 20%는 테스트 구간
    - 학습 구간을 백테스트에 포함하면 성과가 과대평가될 수 있음
    """
    split_index = int(len(df) * (1 - test_size))
    return df.iloc[split_index:].copy().reset_index(drop=True)


def get_month_end_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 월의 마지막 거래일 행만 추출한다.
    """
    temp = df.copy()
    temp["year_month"] = temp["Date"].dt.to_period("M")

    month_end_df = (
        temp
        .groupby("year_month", as_index=False)
        .tail(1)
        .drop(columns=["year_month"])
        .reset_index(drop=True)
    )

    return month_end_df


def add_next_month_return(month_end_df: pd.DataFrame) -> pd.DataFrame:
    """
    월말 Close 기준 다음 월말까지의 QQQ 수익률을 계산한다.

    예:
    2024-01 월말 Close → 2024-02 월말 Close 수익률
    """
    df = month_end_df.copy()

    df["next_close"] = df["Close"].shift(-1)
    df["qqq_next_month_return"] = df["next_close"] / df["Close"] - 1

    # 마지막 월은 다음 월 수익률이 없으므로 제거
    df = df.dropna(subset=["qqq_next_month_return"]).reset_index(drop=True)

    return df


# =========================
# 4. 예측 보조 함수
# =========================

def make_probability_dict(model, label_encoder, X_processed) -> dict[str, float]:
    """
    predict_proba 결과를 {라벨: 확률%} 형태로 변환한다.
    """
    proba = model.predict_proba(X_processed)[0]

    return {
        label_encoder.classes_[i]: round(float(proba[i]) * 100, 4)
        for i in range(len(label_encoder.classes_))
    }


def apply_uncertainty_rule(
    pred_label: str,
    proba_result: dict[str, float],
    min_confidence: float = DIRECTION_MIN_CONFIDENCE,
    min_margin: float = DIRECTION_MIN_MARGIN
) -> str:
    """
    방향성 모델 예측 확률을 기준으로 불확실 여부를 판단한다.

    조건:
    - 최고 확률이 min_confidence 미만이면 불확실
    - 1등과 2등 확률 차이가 min_margin 미만이면 불확실
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

    return pred_label


def apply_high_vol_threshold(
    risk_proba: dict[str, float],
    threshold: float = HIGH_VOL_PROBA_THRESHOLD
) -> str:
    """
    고변동 판단을 확률 기준으로 보정한다.

    이유:
    - risk_model.predict()는 정상 쪽으로 치우칠 수 있음
    - 자산배분에서는 고변동을 놓치는 것이 더 위험할 수 있음
    """
    high_vol_prob = risk_proba.get("고변동", 0.0)

    if high_vol_prob >= threshold:
        return "고변동"

    return "정상"


def predict_one_row(row: pd.DataFrame, model_package: dict) -> dict:
    """
    월말 1개 행에 대해 방향성/위험도 예측을 수행한다.
    """
    direction_model = model_package["direction_model"]
    direction_imputer = model_package["direction_imputer"]
    direction_encoder = model_package["direction_encoder"]
    direction_feature_cols = model_package["direction_feature_cols"]

    risk_model = model_package["risk_model"]
    risk_imputer = model_package["risk_imputer"]
    risk_encoder = model_package["risk_encoder"]
    risk_feature_cols = model_package["risk_feature_cols"]

    # 피처 존재 확인
    missing_direction = [col for col in direction_feature_cols if col not in row.columns]
    missing_risk = [col for col in risk_feature_cols if col not in row.columns]

    if missing_direction:
        raise ValueError(f"방향성 모델 입력 피처 누락: {missing_direction}")

    if missing_risk:
        raise ValueError(f"위험도 모델 입력 피처 누락: {missing_risk}")

    # -------------------------
    # 방향성 예측
    # -------------------------
    X_direction = row[direction_feature_cols]
    X_direction_processed = direction_imputer.transform(X_direction)

    direction_pred_encoded = direction_model.predict(X_direction_processed)[0]
    direction_pred_label = direction_encoder.inverse_transform([direction_pred_encoded])[0]

    direction_proba = make_probability_dict(
        model=direction_model,
        label_encoder=direction_encoder,
        X_processed=X_direction_processed
    )

    final_direction = apply_uncertainty_rule(
        pred_label=direction_pred_label,
        proba_result=direction_proba
    )

    # -------------------------
    # 위험도 예측
    # -------------------------
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
# 5. 자산배분 규칙
# =========================

def recommend_allocation(
    final_direction: str,
    direction_proba: dict[str, float],
    final_risk: str,
    risk_proba: dict[str, float]
) -> tuple[dict[str, float], str]:
    """
    방향성 판단 + 위험도 판단을 결합해 주식/채권/현금 비중을 결정한다.

    출력 비중 단위:
    - 0~1
    """
    up = direction_proba.get("상승", 0.0)
    down = direction_proba.get("하락", 0.0)
    sideways = direction_proba.get("횡보", 0.0)
    high_vol = risk_proba.get("고변동", 0.0)

    # 1. 고변동이면 우선 방어
    if final_risk == "고변동" and high_vol >= 60:
        allocation = {
            "stock": 0.25,
            "bond": 0.35,
            "cash": 0.40
        }
        reason = "고변동 확률 60% 이상: 강한 방어 배분"

    elif final_risk == "고변동":
        allocation = {
            "stock": 0.35,
            "bond": 0.35,
            "cash": 0.30
        }
        reason = "고변동 확률 기준 초과: 방어 배분"

    # 2. 하락 판단이면 방어
    elif final_direction == "하락" or down >= 45:
        allocation = {
            "stock": 0.25,
            "bond": 0.45,
            "cash": 0.30
        }
        reason = "하락 가능성 우세: 주식 비중 축소"

    # 3. 방향성이 불확실하면 중립 방어
    elif final_direction == "불확실":
        allocation = {
            "stock": 0.40,
            "bond": 0.30,
            "cash": 0.30
        }
        reason = "방향성 확신 부족: 중립 방어 배분"

    # 4. 상승 확률이 높고 고변동이 낮으면 공격
    elif final_direction == "상승" and up >= 55 and high_vol < 35:
        allocation = {
            "stock": 0.70,
            "bond": 0.20,
            "cash": 0.10
        }
        reason = "상승 우세 및 고변동 위험 낮음: 주식 비중 확대"

    # 5. 상승이지만 확신이 강하지 않으면 중간 배분
    elif final_direction == "상승":
        allocation = {
            "stock": 0.60,
            "bond": 0.25,
            "cash": 0.15
        }
        reason = "상승 우세: 중간 수준 주식 비중"

    # 6. 횡보면 균형
    elif final_direction == "횡보" or sideways >= 40:
        allocation = {
            "stock": 0.50,
            "bond": 0.30,
            "cash": 0.20
        }
        reason = "횡보 가능성 우세: 균형 배분"

    # 7. 그 외 기본
    else:
        allocation = {
            "stock": 0.50,
            "bond": 0.30,
            "cash": 0.20
        }
        reason = "명확한 우세 조건 없음: 기본 배분"

    return allocation, reason


# =========================
# 6. 백테스트 실행
# =========================

def run_backtest(
    df: pd.DataFrame,
    model_package: dict,
    initial_capital: float
) -> pd.DataFrame:
    """
    월말마다:
    1. 방향성 확률 예측
    2. 위험도 확률 예측
    3. 불확실/고변동 기준 적용
    4. 자산배분 결정
    5. 다음 달 수익률 적용
    """
    test_df = restrict_to_test_period(df, TEST_SIZE)
    month_end_df = get_month_end_rows(test_df)
    month_end_df = add_next_month_return(month_end_df)

    if len(month_end_df) < 3:
        raise ValueError(
            "백테스트 가능한 월별 데이터가 너무 적습니다. "
            "데이터 기간을 늘리거나 TEST_SIZE를 조정하세요."
        )

    records = []

    strategy_capital = initial_capital
    benchmark_capital = initial_capital

    for i in range(len(month_end_df)):
        current_row = month_end_df.iloc[i:i + 1].copy()

        date = current_row["Date"].iloc[0]
        close = float(current_row["Close"].iloc[0])
        qqq_return = float(current_row["qqq_next_month_return"].iloc[0])

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

        bond_return = DEFAULT_BOND_MONTHLY_RETURN
        cash_return = DEFAULT_CASH_MONTHLY_RETURN

        strategy_return = (
            stock_weight * qqq_return
            + bond_weight * bond_return
            + cash_weight * cash_return
        )

        benchmark_return = qqq_return

        strategy_capital *= (1 + strategy_return)
        benchmark_capital *= (1 + benchmark_return)

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

            "qqq_next_month_return": qqq_return,
            "bond_monthly_return": bond_return,
            "cash_monthly_return": cash_return,

            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,

            "strategy_capital": strategy_capital,
            "benchmark_capital": benchmark_capital
        }

        records.append(record)

    result_df = pd.DataFrame(records)

    result_df["strategy_cumulative_return"] = (
        result_df["strategy_capital"] / initial_capital - 1
    )

    result_df["benchmark_cumulative_return"] = (
        result_df["benchmark_capital"] / initial_capital - 1
    )

    return result_df


# =========================
# 7. 성과 지표 계산
# =========================

def calculate_cagr(
    final_capital: float,
    initial_capital: float,
    months: int
) -> float:
    """
    월별 백테스트 기준 CAGR 계산.
    """
    if months <= 0:
        return np.nan

    if initial_capital <= 0 or final_capital <= 0:
        return np.nan

    years = months / 12

    return (final_capital / initial_capital) ** (1 / years) - 1


def calculate_mdd(equity_curve: pd.Series) -> float:
    """
    최대 낙폭 MDD 계산.
    """
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1

    return drawdown.min()


def calculate_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0
) -> float:
    """
    월별 수익률 기준 Sharpe Ratio 계산.
    """
    excess_return = returns - (risk_free_rate / periods_per_year)
    std = excess_return.std()

    if std == 0 or np.isnan(std):
        return np.nan

    return (excess_return.mean() / std) * np.sqrt(periods_per_year)


def summarize_performance(
    result_df: pd.DataFrame,
    initial_capital: float
) -> dict:
    """
    전략과 QQQ 단순 보유 성과를 요약한다.
    """
    months = len(result_df)

    strategy_final = float(result_df["strategy_capital"].iloc[-1])
    benchmark_final = float(result_df["benchmark_capital"].iloc[-1])

    strategy_total_return = strategy_final / initial_capital - 1
    benchmark_total_return = benchmark_final / initial_capital - 1

    strategy_cagr = calculate_cagr(strategy_final, initial_capital, months)
    benchmark_cagr = calculate_cagr(benchmark_final, initial_capital, months)

    strategy_mdd = calculate_mdd(result_df["strategy_capital"])
    benchmark_mdd = calculate_mdd(result_df["benchmark_capital"])

    strategy_sharpe = calculate_sharpe_ratio(result_df["strategy_return"])
    benchmark_sharpe = calculate_sharpe_ratio(result_df["benchmark_return"])

    allocation_counts = (
        result_df[["final_direction", "final_risk"]]
        .value_counts()
        .reset_index(name="count")
    )

    summary = {
        "backtest_start": str(result_df["Date"].iloc[0]),
        "backtest_end": str(result_df["Date"].iloc[-1]),
        "months": int(months),
        "initial_capital": int(initial_capital),

        "strategy_final_capital": round(strategy_final, 2),
        "benchmark_final_capital": round(benchmark_final, 2),

        "strategy_total_return": round(float(strategy_total_return), 6),
        "benchmark_total_return": round(float(benchmark_total_return), 6),

        "strategy_cagr": round(float(strategy_cagr), 6),
        "benchmark_cagr": round(float(benchmark_cagr), 6),

        "strategy_mdd": round(float(strategy_mdd), 6),
        "benchmark_mdd": round(float(benchmark_mdd), 6),

        "strategy_sharpe": round(float(strategy_sharpe), 6),
        "benchmark_sharpe": round(float(benchmark_sharpe), 6),

        "direction_min_confidence": DIRECTION_MIN_CONFIDENCE,
        "direction_min_margin": DIRECTION_MIN_MARGIN,
        "high_vol_proba_threshold": HIGH_VOL_PROBA_THRESHOLD,

        "allocation_count": allocation_counts.to_dict(orient="records"),

        "note": (
            "2단계 모델 기반 간이 백테스트입니다. "
            "방향성 모델은 상승/하락/횡보를 예측하고, "
            "위험도 모델은 정상/고변동을 예측합니다. "
            "현재 채권/현금 수익률은 0으로 가정했습니다. "
            "실제 검증에서는 IEF/BIL 또는 SHV 데이터를 붙여야 합니다."
        )
    }

    return summary


# =========================
# 8. 그래프 저장
# =========================

def save_equity_curve(result_df: pd.DataFrame, save_path: str):
    """
    전략 vs QQQ 단순 보유 자산 곡선 저장.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(result_df["Date"], result_df["strategy_capital"], label="Two-Stage Strategy")
    ax.plot(result_df["Date"], result_df["benchmark_capital"], label="QQQ Buy & Hold")

    ax.set_title("Two-Stage Equity Curve")
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
    """
    전략 vs QQQ 단순 보유 Drawdown 그래프 저장.
    """
    strategy_running_max = result_df["strategy_capital"].cummax()
    benchmark_running_max = result_df["benchmark_capital"].cummax()

    strategy_drawdown = result_df["strategy_capital"] / strategy_running_max - 1
    benchmark_drawdown = result_df["benchmark_capital"] / benchmark_running_max - 1

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(result_df["Date"], strategy_drawdown, label="Two-Stage Strategy Drawdown")
    ax.plot(result_df["Date"], benchmark_drawdown, label="QQQ Drawdown")

    ax.set_title("Two-Stage Drawdown Curve")
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
# 9. 결과 저장 / 출력
# =========================

def save_results(result_df: pd.DataFrame, summary: dict):
    """
    백테스트 상세 결과와 요약 결과를 저장한다.
    """
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


def print_summary(summary: dict):
    """
    콘솔에 성과 요약을 출력한다.
    """
    print("\n==============================")
    print("2단계 모델 백테스트 성과 요약")
    print("==============================")

    print(f"기간: {summary['backtest_start']} ~ {summary['backtest_end']}")
    print(f"개월 수: {summary['months']}")
    print(f"초기 자본: {summary['initial_capital']:,}원")

    print("\n[2단계 전략]")
    print(f"최종 자산: {summary['strategy_final_capital']:,.0f}원")
    print(f"총수익률: {summary['strategy_total_return'] * 100:.2f}%")
    print(f"CAGR: {summary['strategy_cagr'] * 100:.2f}%")
    print(f"MDD: {summary['strategy_mdd'] * 100:.2f}%")
    print(f"Sharpe: {summary['strategy_sharpe']:.4f}")

    print("\n[QQQ 단순 보유]")
    print(f"최종 자산: {summary['benchmark_final_capital']:,.0f}원")
    print(f"총수익률: {summary['benchmark_total_return'] * 100:.2f}%")
    print(f"CAGR: {summary['benchmark_cagr'] * 100:.2f}%")
    print(f"MDD: {summary['benchmark_mdd'] * 100:.2f}%")
    print(f"Sharpe: {summary['benchmark_sharpe']:.4f}")

    print("\n[판단 기준]")
    print(f"방향성 최소 확신도: {summary['direction_min_confidence']}%")
    print(f"방향성 1등-2등 최소 차이: {summary['direction_min_margin']}%p")
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
    """
    최근 n개 월별 결과를 출력한다.
    """
    print(f"\n최근 {n}개 월별 결과:")

    display_cols = [
        "Date",
        "direction_pred_label",
        "final_direction",
        "risk_pred_label",
        "final_risk",
        "prob_상승",
        "prob_하락",
        "prob_횡보",
        "prob_고변동",
        "prob_정상",
        "stock_weight",
        "bond_weight",
        "cash_weight",
        "qqq_next_month_return",
        "strategy_return",
        "benchmark_return",
        "strategy_capital",
        "benchmark_capital"
    ]

    existing_cols = [col for col in display_cols if col in result_df.columns]
    print(result_df[existing_cols].tail(n).to_string(index=False))


# =========================
# 10. 메인 실행
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

    print("\n[3] 월말 리밸런싱 백테스트 실행")
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