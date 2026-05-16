# Python 3.10+
# 필요 패키지:
# pip install pandas numpy scikit-learn matplotlib joblib

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 기본 설정
# =========================

DATA_PATH = "data/qqq_features_labeled.csv"
MODEL_PATH = "models/qqq_regime_randomforest.pkl"
META_PATH = "models/qqq_regime_metadata.json"

RESULT_DIR = "results"

BACKTEST_RESULT_PATH = os.path.join(RESULT_DIR, "backtest_allocation_result.csv")
BACKTEST_SUMMARY_PATH = os.path.join(RESULT_DIR, "backtest_summary.json")
EQUITY_CURVE_PATH = os.path.join(RESULT_DIR, "equity_curve.png")
DRAWDOWN_PATH = os.path.join(RESULT_DIR, "drawdown_curve.png")

# step2_train_model.py와 동일하게 테스트 구간을 뒤 20%로 가정
TEST_SIZE = 0.2

# 초기 투자금
INITIAL_CAPITAL = 100_000_000

# 채권/현금 월수익률 기본값
# 현재는 QQQ 데이터만 있으므로 보수적으로 0으로 둔다.
# 이후 IEF/BIL 데이터를 붙이면 실제 월수익률로 교체해야 한다.
DEFAULT_BOND_MONTHLY_RETURN = 0.0
DEFAULT_CASH_MONTHLY_RETURN = 0.0


# =========================
# 2. 파일 로드
# =========================

def load_dataset(path: str) -> pd.DataFrame:
    """
    라벨링된 QQQ 데이터셋을 불러온다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {path}\n"
            "먼저 step1_make_dataset.py를 실행하세요."
        )

    df = pd.read_csv(path)

    required_cols = ["Date", "Close", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def load_model_package(model_path: str) -> dict:
    """
    step2에서 저장한 모델 패키지를 불러온다.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {model_path}\n"
            "먼저 step2_train_model.py를 실행하세요."
        )

    package = joblib.load(model_path)

    required_keys = ["model", "imputer", "label_encoder", "feature_cols"]
    missing_keys = [key for key in required_keys if key not in package]

    if missing_keys:
        raise ValueError(f"모델 패키지 필수 항목 누락: {missing_keys}")

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
# 3. 자산배분 규칙
# =========================

def recommend_allocation(proba_result: dict[str, float]) -> tuple[dict[str, float], str]:
    """
    국면별 확률을 바탕으로 주식/채권/현금 비율을 결정한다.

    입력 확률 단위:
    - 0~100

    출력 비중 단위:
    - 0~1
    """
    up = proba_result.get("상승", 0.0)
    down = proba_result.get("하락", 0.0)
    sideways = proba_result.get("횡보", 0.0)
    high_vol = proba_result.get("고변동", 0.0)
    uncertain = proba_result.get("불확실", 0.0)

    if down >= 50:
        allocation = {
            "stock": 0.20,
            "bond": 0.45,
            "cash": 0.35
        }
        reason = "하락 확률 50% 이상: 방어적 배분"

    elif high_vol >= 50:
        allocation = {
            "stock": 0.30,
            "bond": 0.30,
            "cash": 0.40
        }
        reason = "고변동 확률 50% 이상: 현금 비중 확대"

    elif up >= 60 and down < 25:
        allocation = {
            "stock": 0.75,
            "bond": 0.15,
            "cash": 0.10
        }
        reason = "상승 확률 60% 이상, 하락 확률 25% 미만: 주식 비중 확대"

    elif up >= 45:
        allocation = {
            "stock": 0.60,
            "bond": 0.25,
            "cash": 0.15
        }
        reason = "상승 우세: 중간 수준 주식 비중"

    elif uncertain >= 40:
        allocation = {
            "stock": 0.40,
            "bond": 0.30,
            "cash": 0.30
        }
        reason = "불확실 확률 40% 이상: 중립 방어 배분"

    elif sideways >= 40:
        allocation = {
            "stock": 0.50,
            "bond": 0.30,
            "cash": 0.20
        }
        reason = "횡보 확률 40% 이상: 균형 배분"

    else:
        allocation = {
            "stock": 0.50,
            "bond": 0.30,
            "cash": 0.20
        }
        reason = "명확한 우세 국면 없음: 기본 배분"

    return allocation, reason


# =========================
# 4. 월말 리밸런싱 날짜 생성
# =========================

def get_month_end_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 월의 마지막 거래일 데이터를 추출한다.
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


def restrict_to_test_period(df: pd.DataFrame, test_size: float) -> pd.DataFrame:
    """
    step2의 시계열 분할 방식과 동일하게 뒤 20% 구간만 백테스트에 사용한다.

    이유:
    - step2 모델은 앞 80%로 학습했으므로,
      전체 기간에 대해 백테스트하면 학습 구간 성과가 섞인다.
    - 뒤 20%만 쓰면 최소한 테스트 구간 기반 평가가 된다.
    """
    split_index = int(len(df) * (1 - test_size))
    test_df = df.iloc[split_index:].copy().reset_index(drop=True)

    return test_df


# =========================
# 5. 월별 수익률 계산
# =========================

def add_next_month_return(month_end_df: pd.DataFrame) -> pd.DataFrame:
    """
    월말 가격 기준 다음 월말까지의 QQQ 수익률을 계산한다.

    예:
    2024-01 월말 Close → 2024-02 월말 Close 수익률
    """
    df = month_end_df.copy()

    df["next_close"] = df["Close"].shift(-1)
    df["qqq_next_month_return"] = df["next_close"] / df["Close"] - 1

    # 마지막 월은 다음 월 수익률을 알 수 없으므로 제거
    df = df.dropna(subset=["qqq_next_month_return"]).reset_index(drop=True)

    return df


# =========================
# 6. 국면 확률 예측
# =========================

def predict_regime_probabilities(
    model,
    imputer,
    label_encoder,
    row: pd.DataFrame,
    feature_cols: list[str]
) -> tuple[str, dict[str, float]]:
    """
    특정 월말 1개 행에 대해 국면과 확률을 예측한다.
    """
    X = row[feature_cols]

    X_imputed = imputer.transform(X)

    pred_encoded = model.predict(X_imputed)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    probabilities = model.predict_proba(X_imputed)[0]

    proba_result = {
        label_encoder.classes_[i]: round(float(probabilities[i]) * 100, 4)
        for i in range(len(label_encoder.classes_))
    }

    return pred_label, proba_result


# =========================
# 7. 백테스트 실행
# =========================

def run_backtest(
    df: pd.DataFrame,
    model_package: dict,
    initial_capital: float
) -> pd.DataFrame:
    """
    월말마다 모델 예측 → 자산배분 → 다음 달 수익률 적용 방식으로 백테스트한다.
    """
    model = model_package["model"]
    imputer = model_package["imputer"]
    label_encoder = model_package["label_encoder"]
    feature_cols = model_package["feature_cols"]

    # 피처 컬럼 존재 확인
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"데이터셋에 모델 입력 피처가 없습니다: {missing_features}")

    # 테스트 구간만 사용
    test_df = restrict_to_test_period(df, TEST_SIZE)

    # 월말 데이터 추출
    month_end_df = get_month_end_rows(test_df)

    # 다음 달 QQQ 수익률 계산
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

        pred_label, proba_result = predict_regime_probabilities(
            model=model,
            imputer=imputer,
            label_encoder=label_encoder,
            row=current_row,
            feature_cols=feature_cols
        )

        allocation, reason = recommend_allocation(proba_result)

        stock_weight = allocation["stock"]
        bond_weight = allocation["bond"]
        cash_weight = allocation["cash"]

        # 현재는 채권/현금 실제 데이터가 없으므로 기본값 사용
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

        record = {
            "Date": date,
            "Close": close,
            "pred_label": pred_label,

            "prob_상승": proba_result.get("상승", 0.0),
            "prob_하락": proba_result.get("하락", 0.0),
            "prob_횡보": proba_result.get("횡보", 0.0),
            "prob_고변동": proba_result.get("고변동", 0.0),
            "prob_불확실": proba_result.get("불확실", 0.0),

            "stock_weight": stock_weight,
            "bond_weight": bond_weight,
            "cash_weight": cash_weight,
            "allocation_reason": reason,

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
# 8. 성과 지표 계산
# =========================

def calculate_cagr(equity_curve: pd.Series, periods_per_year: int = 12) -> float:
    """
    CAGR 계산.
    월별 데이터 기준 periods_per_year=12.
    """
    if len(equity_curve) < 2:
        return np.nan

    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]

    if start_value <= 0 or end_value <= 0:
        return np.nan

    years = len(equity_curve) / periods_per_year

    return (end_value / start_value) ** (1 / years) - 1


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
    Sharpe Ratio 계산.

    현재 risk_free_rate는 단순화를 위해 0으로 둔다.
    """
    excess_return = returns - (risk_free_rate / periods_per_year)

    std = excess_return.std()

    if std == 0 or np.isnan(std):
        return np.nan

    return (excess_return.mean() / std) * np.sqrt(periods_per_year)


def summarize_performance(result_df: pd.DataFrame) -> dict:
    """
    전략과 벤치마크의 성과를 요약한다.
    """
    strategy_total_return = result_df["strategy_capital"].iloc[-1] / result_df["strategy_capital"].iloc[0] - 1
    benchmark_total_return = result_df["benchmark_capital"].iloc[-1] / result_df["benchmark_capital"].iloc[0] - 1

    strategy_cagr = calculate_cagr(result_df["strategy_capital"])
    benchmark_cagr = calculate_cagr(result_df["benchmark_capital"])

    strategy_mdd = calculate_mdd(result_df["strategy_capital"])
    benchmark_mdd = calculate_mdd(result_df["benchmark_capital"])

    strategy_sharpe = calculate_sharpe_ratio(result_df["strategy_return"])
    benchmark_sharpe = calculate_sharpe_ratio(result_df["benchmark_return"])

    summary = {
        "backtest_start": str(result_df["Date"].iloc[0]),
        "backtest_end": str(result_df["Date"].iloc[-1]),
        "months": int(len(result_df)),

        "initial_capital": INITIAL_CAPITAL,

        "strategy_final_capital": round(float(result_df["strategy_capital"].iloc[-1]), 2),
        "benchmark_final_capital": round(float(result_df["benchmark_capital"].iloc[-1]), 2),

        "strategy_total_return": round(float(strategy_total_return), 6),
        "benchmark_total_return": round(float(benchmark_total_return), 6),

        "strategy_cagr": round(float(strategy_cagr), 6),
        "benchmark_cagr": round(float(benchmark_cagr), 6),

        "strategy_mdd": round(float(strategy_mdd), 6),
        "benchmark_mdd": round(float(benchmark_mdd), 6),

        "strategy_sharpe": round(float(strategy_sharpe), 6),
        "benchmark_sharpe": round(float(benchmark_sharpe), 6),

        "note": (
            "간이 백테스트입니다. 현재 채권/현금 수익률은 0으로 가정했습니다. "
            "실제 검증에서는 IEF/BIL 또는 SHV 데이터를 붙여야 합니다."
        )
    }

    return summary


# =========================
# 9. 그래프 저장
# =========================

def save_equity_curve(result_df: pd.DataFrame, save_path: str):
    """
    전략 vs QQQ 벤치마크 누적 자산 그래프 저장.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(result_df["Date"], result_df["strategy_capital"], label="Strategy")
    ax.plot(result_df["Date"], result_df["benchmark_capital"], label="QQQ Buy & Hold")

    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"자산 곡선 저장 완료: {save_path}")


def save_drawdown_curve(result_df: pd.DataFrame, save_path: str):
    """
    전략 vs 벤치마크 Drawdown 그래프 저장.
    """
    strategy_running_max = result_df["strategy_capital"].cummax()
    benchmark_running_max = result_df["benchmark_capital"].cummax()

    strategy_drawdown = result_df["strategy_capital"] / strategy_running_max - 1
    benchmark_drawdown = result_df["benchmark_capital"] / benchmark_running_max - 1

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(result_df["Date"], strategy_drawdown, label="Strategy Drawdown")
    ax.plot(result_df["Date"], benchmark_drawdown, label="QQQ Drawdown")

    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Drawdown 그래프 저장 완료: {save_path}")


# =========================
# 10. 결과 저장
# =========================

def save_results(result_df: pd.DataFrame, summary: dict):
    """
    백테스트 상세 결과와 요약 결과를 저장한다.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)

    result_df.to_csv(BACKTEST_RESULT_PATH, index=False, encoding="utf-8-sig")

    with open(BACKTEST_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"백테스트 상세 결과 저장 완료: {BACKTEST_RESULT_PATH}")
    print(f"백테스트 요약 저장 완료: {BACKTEST_SUMMARY_PATH}")


# =========================
# 11. 출력 보조
# =========================

def print_summary(summary: dict):
    """
    콘솔에 성과 요약을 출력한다.
    """
    print("\n==============================")
    print("백테스트 성과 요약")
    print("==============================")

    print(f"기간: {summary['backtest_start']} ~ {summary['backtest_end']}")
    print(f"개월 수: {summary['months']}")
    print(f"초기 자본: {summary['initial_capital']:,}원")

    print("\n[전략]")
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

    print("\n주의:")
    print(summary["note"])


# =========================
# 12. 메인 실행
# =========================

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("[1] 데이터 불러오기")
    df = load_dataset(DATA_PATH)
    print(f"데이터 크기: {df.shape}")
    print(f"기간: {df['Date'].min()} ~ {df['Date'].max()}")

    print("\n[2] 모델 불러오기")
    model_package = load_model_package(MODEL_PATH)
    metadata = load_metadata(META_PATH)

    if metadata:
        print("모델 메타데이터 로드 완료")
        print(f"모델 타입: {metadata.get('model_type', 'unknown')}")
        print(f"라벨 목록: {metadata.get('labels', [])}")

    print("\n[3] 월말 리밸런싱 백테스트 실행")
    result_df = run_backtest(
        df=df,
        model_package=model_package,
        initial_capital=INITIAL_CAPITAL
    )

    print("\n[4] 성과 지표 계산")
    summary = summarize_performance(result_df)

    print_summary(summary)

    print("\n[5] 결과 저장")
    save_results(result_df, summary)

    print("\n[6] 그래프 저장")
    save_equity_curve(result_df, EQUITY_CURVE_PATH)
    save_drawdown_curve(result_df, DRAWDOWN_PATH)

    print("\n최근 10개 월별 결과:")
    display_cols = [
        "Date",
        "pred_label",
        "prob_상승",
        "prob_하락",
        "prob_횡보",
        "prob_고변동",
        "prob_불확실",
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
    print(result_df[existing_cols].tail(10).to_string(index=False))

    print("\n완료")


if __name__ == "__main__":
    main()