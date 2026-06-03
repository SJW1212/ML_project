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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# =========================
# 1. 기본 설정
# =========================

TARGET_TICKER = "TQQQ"
TARGET_TICKER_LOWER = TARGET_TICKER.lower()

DATA_PATH = f"data/{TARGET_TICKER_LOWER}_features_labeled.csv"

MODEL_DIR = "models"
RESULT_DIR = "results"

MODEL_PATH = os.path.join(MODEL_DIR, f"{TARGET_TICKER_LOWER}_two_stage_model.pkl")
META_PATH = os.path.join(MODEL_DIR, f"{TARGET_TICKER_LOWER}_two_stage_metadata.json")

DIRECTION_CM_PATH = os.path.join(RESULT_DIR, f"{TARGET_TICKER_LOWER}_direction_confusion_matrix.png")
RISK_CM_PATH = os.path.join(RESULT_DIR, f"{TARGET_TICKER_LOWER}_risk_confusion_matrix.png")

DIRECTION_FEATURE_IMPORTANCE_PATH = os.path.join(RESULT_DIR, f"{TARGET_TICKER_LOWER}_direction_feature_importance.png")
RISK_FEATURE_IMPORTANCE_PATH = os.path.join(RESULT_DIR, f"{TARGET_TICKER_LOWER}_risk_feature_importance.png")

DIRECTION_FEATURE_IMPORTANCE_CSV_PATH = os.path.join(RESULT_DIR, f"{TARGET_TICKER_LOWER}_direction_feature_importance.csv")
RISK_FEATURE_IMPORTANCE_CSV_PATH = os.path.join(RESULT_DIR, f"{TARGET_TICKER_LOWER}_risk_feature_importance.csv")

EVALUATION_SUMMARY_PATH = os.path.join(
    RESULT_DIR,
    f"{TARGET_TICKER_LOWER}_step2_two_stage_evaluation_summary.json"
)

TEST_SIZE = 0.5
RANDOM_STATE = 42


# =========================
# 2. 데이터 불러오기
# =========================

def load_dataset(path: str) -> pd.DataFrame:
    """
    step1_make_dataset.py에서 생성한 데이터셋을 불러온다.

    필수 컬럼:
    - direction_label: 상승 / 하락 / 횡보
    - risk_label: 정상 / 고변동
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {path}\n"
            "먼저 step1_make_dataset.py를 실행하세요."
        )

    df = pd.read_csv(path)

    required_cols = ["Date", "direction_label", "risk_label"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing_cols}\n"
            "수정된 step1_make_dataset.py를 먼저 실행해야 합니다."
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# =========================
# 3. 피처 선택
# =========================

def get_base_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    모델 입력 후보 피처를 선택한다.

    제외 대상:
    - 날짜
    - 원시 가격/거래량
    - 미래 정보
    - 정답 라벨
    """
    exclude_cols = {
        "Date",

        # 원시 가격/거래량
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",

        # 라벨
        "label",
        "direction_label",
        "risk_label"
    }

    feature_cols = []

    for col in df.columns:
        # 미래 정보는 전부 제외
        if col.startswith("future_"):
            continue

        if col in exclude_cols:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("사용 가능한 피처 컬럼이 없습니다.")

    return feature_cols

def get_direction_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    러프 버전 방향성 모델용 피처.

    의도:
    - ADX, MACD, CCI, RSI, Bollinger, ATR 제거
    - 가격과 거래량에서 직접 파생된 단순 피처만 사용
    - 모델이 복잡한 후행 지표에 끌려가지 않도록 제한
    """
    preferred_cols = [
        # 수익률
        "daily_return",
        "log_return",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_60d",
        "return_5d_zscore",
        "return_20d_zscore",

        # 가격-이동평균 괴리율
        "price_ma_5_gap",
        "price_ma_20_gap",
        "price_ma_60_gap",
        "ma_gap_5_20",
        "ma_gap_20_60",

        # 추세 기울기
        "trend_slope_20",
        "trend_slope_60",
        "positive_return_ratio_20",

        # 가격 위치
        "drawdown",
        "price_position_20",
        "price_position_60",
        "close_to_20d_high",
        "close_to_60d_high",

        # 변동성
        "volatility_5d",
        "volatility_20d",
        "volatility_60d",
        "volatility_ratio_5_20",
        "volatility_ratio_20_60",
        "downside_volatility_20d",

        # 거래량
        "volume_change",
        "volume_ratio_20",
        "volume_zscore_20",
    ]

    feature_cols = [
        col for col in preferred_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("러프 방향성 모델에 사용할 피처가 없습니다.")

    return feature_cols

# def get_direction_feature_columns(df: pd.DataFrame) -> list[str]:
#     """
#     방향성 모델용 피처를 선택한다.

#     방향성 모델은 상승/하락/횡보를 구분해야 하므로,
#     변동성 자체보다 수익률, 추세, 가격 위치, 모멘텀 피처를 우선 사용한다.
#     """
#     preferred_cols = [
#         # 수익률 / 모멘텀
#         "daily_return",
#         "log_return",
#         "return_3d",
#         "return_5d",
#         "return_10d",
#         "return_20d",
#         "return_60d",
#         "return_5d_zscore",
#         "return_20d_zscore",

#         # 이동평균 / 추세
#         "price_ma_5_gap",
#         "price_ma_20_gap",
#         "price_ma_60_gap",
#         "ma_gap_5_20",
#         "ma_gap_20_60",
#         "trend_slope_20",
#         "trend_slope_60",
#         "positive_return_ratio_20",

#         # 가격 위치
#         "drawdown",
#         "price_position_20",
#         "price_position_60",
#         "close_to_20d_high",
#         "close_to_60d_high",

#         # RSI / MACD / Bollinger 위치
#         "rsi_14_scaled",
#         "macd_pct",
#         "macd_signal_pct",
#         "macd_hist_pct",
#         "bollinger_position_20",
        
#                 # ADX / DI
#         "adx_14",
#         "plus_di_14",
#         "minus_di_14",
#         "di_gap_14",
#         "di_ratio_14",
#         "adx_trend_strength",
#         "adx_bullish_trend",
#         "adx_bearish_trend",

#         # CCI
#         "cci_20_scaled",
#         "cci_20_diff",
#         "cci_bullish",
#         "cci_bearish",
#         "cci_neutral",

#         # MACD 추가 피처
#         "macd_hist_diff",
#         "macd_hist_slope_5",
#         "macd_cross_signal",
#         "macd_bullish_cross",
#         "macd_bearish_cross",

#         # 기술적 보조 점수
#         "technical_bullish_score",
#         "technical_bearish_score",
#         "technical_sideways_score",
#         "technical_trend_margin",
#         "technical_trend_balance",

#         # 거래량 보조
#         "volume_change",
#         "volume_ratio_20",
#         "volume_zscore_20",

#         # 변동성은 보조로 일부만 사용
#         "volatility_ratio_5_20",
#         "volatility_ratio_20_60",
#         "downside_volatility_20d"
#     ]

#     feature_cols = [
#         col for col in preferred_cols
#         if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
#     ]

#     if not feature_cols:
#         feature_cols = get_base_feature_columns(df)

#     return feature_cols

def get_risk_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    러프 버전 위험도 모델용 피처.

    의도:
    - 고변동 판단도 가격/거래량 기반으로만 수행
    - ATR, ADX, Bollinger 같은 기술적 지표 제거
    """
    preferred_cols = [
        # 변동성
        "volatility_5d",
        "volatility_20d",
        "volatility_60d",
        "volatility_ratio_5_20",
        "volatility_ratio_20_60",
        "downside_volatility_20d",

        # 수익률 급변
        "daily_return",
        "log_return",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "return_60d",
        "return_5d_zscore",
        "return_20d_zscore",

        # 낙폭 / 가격 위치
        "drawdown",
        "price_position_20",
        "price_position_60",
        "close_to_20d_high",
        "close_to_60d_high",

        # 거래량
        "volume_change",
        "volume_ratio_20",
        "volume_zscore_20",

        # 추세 기울기
        "trend_slope_20",
        "trend_slope_60",
        "positive_return_ratio_20",
    ]

    feature_cols = [
        col for col in preferred_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("러프 위험도 모델에 사용할 피처가 없습니다.")

    return feature_cols

# def get_risk_feature_columns(df: pd.DataFrame) -> list[str]:
#     """
#     위험도 모델용 피처를 선택한다.

#     위험도 모델은 정상/고변동을 구분하므로,
#     변동성, ATR, Bollinger 폭, 거래량 급증, 낙폭 관련 피처를 우선 사용한다.
#     """
#     preferred_cols = [
#         # 변동성
#         "volatility_5d",
#         "volatility_20d",
#         "volatility_60d",
#         "volatility_ratio_5_20",
#         "volatility_ratio_20_60",
#         "downside_volatility_20d",
#         "atr_14_pct",
#         "bollinger_width_20",

#         # 낙폭 / 위치
#         "drawdown",
#         "close_to_20d_high",
#         "close_to_60d_high",
#         "price_position_20",
#         "price_position_60",

#         # 수익률 급변
#         "daily_return",
#         "log_return",
#         "return_3d",
#         "return_5d",
#         "return_20d",
#         "return_5d_zscore",
#         "return_20d_zscore",

#         # 거래량
#         "volume_change",
#         "volume_ratio_20",
#         "volume_zscore_20",

#         # 추세 보조
#         "trend_slope_20",
#         "trend_slope_60"
        
#         # ADX / 추세 강도
#         "adx_14",
#         "di_gap_14",
#         "di_ratio_14",
#         "adx_trend_strength",

#         # CCI 급변 / 과열
#         "cci_20_scaled",
#         "cci_20_diff",
#         "cci_bullish",
#         "cci_bearish",

#         # MACD 변동성 보조
#         "macd_hist_diff",
#         "macd_hist_slope_5",

#         # 기술적 보조 점수
#         "technical_bullish_score",
#         "technical_bearish_score",
#         "technical_sideways_score",
#         "technical_trend_margin",
#         "technical_trend_balance",
#     ]

#     feature_cols = [
#         col for col in preferred_cols
#         if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
#     ]

#     if not feature_cols:
#         feature_cols = get_base_feature_columns(df)

#     return feature_cols


# =========================
# 4. 시계열 학습/테스트 분리
# =========================

def time_series_train_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float = 0.2
):
    """
    금융 시계열 데이터이므로 랜덤 셔플 없이 시간 순서대로 분리한다.

    앞 80%: 학습
    뒤 20%: 테스트
    """
    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    return train_df, test_df, X_train, X_test, y_train, y_test


# =========================
# 5. 전처리 / 라벨 인코딩
# =========================

def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    결측값을 중앙값으로 대체한다.

    RandomForest 계열은 스케일링이 필수는 아니므로,
    우선 imputation만 수행한다.
    """
    imputer = SimpleImputer(strategy="median")

    X_train_processed = imputer.fit_transform(X_train)
    X_test_processed = imputer.transform(X_test)

    return X_train_processed, X_test_processed, imputer


def encode_labels(y_train: pd.Series, y_test: pd.Series):
    """
    문자열 라벨을 숫자로 변환한다.
    """
    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    return y_train_encoded, y_test_encoded, encoder


# =========================
# 6. 모델 생성 / 학습
# =========================

def create_direction_model() -> RandomForestClassifier:
    """
    방향성 모델.

    max_features='sqrt':
    - 특정 피처군에 과도하게 의존하는 것을 줄인다.

    class_weight='balanced_subsample':
    - 상승/하락/횡보 라벨 불균형을 일부 보정한다.
    """
    return RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def create_risk_model() -> RandomForestClassifier:
    """
    위험도 모델.

    정상/고변동 이진 분류이지만, 고변동 라벨이 적을 수 있으므로
    class_weight를 사용한다.
    """
    return RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def train_single_model(
    X_train,
    y_train,
    model_type: str
) -> RandomForestClassifier:
    """
    target 종류에 따라 모델을 생성하고 학습한다.
    """
    if model_type == "direction":
        model = create_direction_model()
    elif model_type == "risk":
        model = create_risk_model()
    else:
        raise ValueError(f"지원하지 않는 model_type입니다: {model_type}")

    model.fit(X_train, y_train)

    return model


# =========================
# 7. 모델 평가
# =========================

def evaluate_model(
    model,
    X_test,
    y_test,
    label_encoder: LabelEncoder,
    title: str
) -> dict:
    """
    모델 성능을 평가하고 평가 결과 dict를 반환한다.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n==============================")
    print(f"{title} 평가 결과")
    print("==============================")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    print("\n분류 리포트:")
    print(report_text)

    metrics = {
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "classification_report": report_dict
    }

    return y_pred, metrics


# =========================
# 8. 시각화 저장
# =========================

def save_confusion_matrix(
    y_test,
    y_pred,
    label_encoder: LabelEncoder,
    save_path: str,
    title: str
):
    """
    Confusion Matrix 이미지를 저장한다.
    """
    labels = list(range(len(label_encoder.classes_)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(np.arange(len(label_encoder.classes_)))
    ax.set_yticks(np.arange(len(label_encoder.classes_)))
    ax.set_xticklabels(label_encoder.classes_, rotation=45, ha="right")
    ax.set_yticklabels(label_encoder.classes_)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    plt.close("all")

    print(f"{title} 저장 완료: {save_path}")


def save_feature_importance(
    model,
    feature_cols: list[str],
    save_img_path: str,
    save_csv_path: str,
    title: str
):
    """
    RandomForest 피처 중요도를 저장한다.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"{title}: feature_importances_가 없어 저장을 건너뜁니다.")
        return None

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")

    top_n = min(15, len(importance_df))
    top_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_df["feature"], top_df["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    plt.savefig(save_img_path, dpi=150)
    plt.close(fig)
    plt.close("all")

    print(f"{title} 이미지 저장 완료: {save_img_path}")
    print(f"{title} CSV 저장 완료: {save_csv_path}")

    print(f"\n{title} 상위 피처 중요도:")
    print(importance_df.head(10))

    return importance_df


# =========================
# 9. 불확실 판단 / 자산배분
# =========================

def apply_direction_decision_rule(
    pred_label: str,
    proba_result: dict[str, float],
    min_confidence: float = 35.0,
    min_margin: float = 5.0,
    down_accept_threshold: float = 45.0
) -> str:
    """
    방향성 모델 예측 확률을 기준으로 최종 방향성을 판단한다.

    수정 목적:
    - 기존 모델이 하락을 과도하게 판단하는 문제를 완화
    - 상승/횡보 판단이 불확실로 너무 많이 밀리는 문제 완화

    규칙:
    1. 최고 확률이 min_confidence 미만이면 불확실
    2. 1등과 2등 차이가 min_margin 미만이면 불확실
    3. 하락은 확률이 down_accept_threshold 이상일 때만 인정
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

    # 하락 과대 판단 방지
    if top_label == "하락" and top_prob < down_accept_threshold:
        return "불확실"

    return top_label


def make_probability_dict(
    model,
    label_encoder: LabelEncoder,
    X_processed
) -> dict[str, float]:
    """
    모델의 predict_proba 결과를 {라벨: 확률%} dict로 변환한다.
    """
    proba = model.predict_proba(X_processed)[0]

    return {
        label_encoder.classes_[i]: round(float(proba[i]) * 100, 2)
        for i in range(len(label_encoder.classes_))
    }


def recommend_allocation(
    final_direction: str,
    direction_proba: dict[str, float],
    risk_proba: dict[str, float]
) -> tuple[dict[str, int], str]:
    """
    방향성 확률 + 위험도 확률을 바탕으로 자산배분을 추천한다.

    주의:
    - 실제 투자 조언이 아니라 프로젝트용 의사결정 규칙이다.
    """
    up = direction_proba.get("상승", 0)
    down = direction_proba.get("하락", 0)
    sideways = direction_proba.get("횡보", 0)

    high_vol = risk_proba.get("고변동", 0)
    normal = risk_proba.get("정상", 0)

    # 1. 고변동이 강하면 우선 방어
    if high_vol >= 60:
        allocation = {
            "주식": 30,
            "채권": 35,
            "현금": 35
        }
        reason = "고변동 확률이 높아 방어적 배분"

    # 2. 하락 가능성이 강하면 방어
    elif final_direction == "하락" or down >= 45:
        allocation = {
            "주식": 25,
            "채권": 45,
            "현금": 30
        }
        reason = "하락 가능성이 높아 주식 비중 축소"

    # 3. 불확실이면 중립 방어
    elif final_direction == "불확실":
        allocation = {
            "주식": 40,
            "채권": 30,
            "현금": 30
        }
        reason = "방향성 확신이 낮아 중립 방어 배분"

    # 4. 상승 확률이 높고 고변동이 낮으면 공격
    elif final_direction == "상승" and up >= 55 and high_vol < 50:
        allocation = {
            "주식": 70,
            "채권": 20,
            "현금": 10
        }
        reason = "상승 우세이며 고변동 위험이 낮아 주식 비중 확대"

    # 5. 상승이지만 확신이 약하면 중간
    elif final_direction == "상승":
        allocation = {
            "주식": 60,
            "채권": 25,
            "현금": 15
        }
        reason = "상승 우세이나 확신이 강하지 않아 중간 수준 주식 비중"

    # 6. 횡보면 균형
    elif final_direction == "횡보" or sideways >= 40:
        allocation = {
            "주식": 50,
            "채권": 30,
            "현금": 20
        }
        reason = "횡보 가능성이 높아 균형 배분"

    else:
        allocation = {
            "주식": 50,
            "채권": 30,
            "현금": 20
        }
        reason = "명확한 우세 조건이 없어 기본 배분"

    return allocation, reason


# =========================
# 10. 최신 데이터 예측
# =========================

def predict_latest_sample(
    direction_model,
    direction_imputer,
    direction_encoder: LabelEncoder,
    direction_feature_cols: list[str],
    risk_model,
    risk_imputer,
    risk_encoder: LabelEncoder,
    risk_feature_cols: list[str],
    df: pd.DataFrame
):
    """
    최신 데이터 1건에 대해:
    - 방향성 모델 확률
    - 위험도 모델 확률
    - 최종 판단
    - 자산배분
    을 출력한다.
    """
    latest_row = df.iloc[-1:].copy()

    # 방향성 예측
    X_direction = latest_row[direction_feature_cols]
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
        proba_result=direction_proba,
        min_confidence=35.0,
        min_margin=5.0,
        down_accept_threshold=45.0
    )

    # 위험도 예측
    X_risk = latest_row[risk_feature_cols]
    X_risk_processed = risk_imputer.transform(X_risk)

    risk_pred_encoded = risk_model.predict(X_risk_processed)[0]
    risk_pred_label = risk_encoder.inverse_transform([risk_pred_encoded])[0]

    risk_proba = make_probability_dict(
        model=risk_model,
        label_encoder=risk_encoder,
        X_processed=X_risk_processed
    )

    allocation, allocation_reason = recommend_allocation(
        final_direction=final_direction,
        direction_proba=direction_proba,
        risk_proba=risk_proba
    )

    print("\n==============================")
    print("최신 데이터 기준 예측 결과")
    print("==============================")

    print(f"기준일: {latest_row['Date'].iloc[0]}")
    print(f"방향성 모델 예측: {direction_pred_label}")
    print(f"최종 방향성 판단: {final_direction}")
    print(f"위험도 모델 예측: {risk_pred_label}")

    print("\n방향성 확률:")
    for label, prob in sorted(direction_proba.items(), key=lambda x: x[1], reverse=True):
        print(f"- {label}: {prob:.2f}%")

    print("\n위험도 확률:")
    for label, prob in sorted(risk_proba.items(), key=lambda x: x[1], reverse=True):
        print(f"- {label}: {prob:.2f}%")

    print("\n자산배분 추천:")
    print(f"추천 사유: {allocation_reason}")
    print(f"주식: {allocation['주식']}%")
    print(f"채권: {allocation['채권']}%")
    print(f"현금: {allocation['현금']}%")

    latest_result = {
        "date": str(latest_row["Date"].iloc[0]),
        "direction_pred_label": direction_pred_label,
        "final_direction": final_direction,
        "direction_proba": direction_proba,
        "risk_pred_label": risk_pred_label,
        "risk_proba": risk_proba,
        "allocation": allocation,
        "allocation_reason": allocation_reason
    }

    return latest_result


# =========================
# 11. 저장
# =========================

def save_model_package(
    direction_model,
    direction_imputer,
    direction_encoder,
    direction_feature_cols,
    risk_model,
    risk_imputer,
    risk_encoder,
    risk_feature_cols,
    metrics: dict
):
    """
    두 모델과 전처리 객체를 하나의 패키지로 저장한다.
    """
    model_package = {
        "direction_model": direction_model,
        "direction_imputer": direction_imputer,
        "direction_encoder": direction_encoder,
        "direction_feature_cols": direction_feature_cols,

        "risk_model": risk_model,
        "risk_imputer": risk_imputer,
        "risk_encoder": risk_encoder,
        "risk_feature_cols": risk_feature_cols,

        "model_structure": "two_stage_direction_and_risk"
    }

    joblib.dump(model_package, MODEL_PATH)

    metadata = {
        "model_structure": "two_stage_direction_and_risk",
        "direction_model_type": "RandomForestClassifier",
        "risk_model_type": "RandomForestClassifier",
        "direction_labels": direction_encoder.classes_.tolist(),
        "risk_labels": risk_encoder.classes_.tolist(),
        "direction_feature_cols": direction_feature_cols,
        "risk_feature_cols": risk_feature_cols,
        "metrics": metrics,
        "note": (
            "Two-stage model. Direction model predicts 상승/하락/횡보. "
            "Risk model predicts 정상/고변동. Time-series split used. "
            "Future columns excluded to avoid leakage."
        )
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    with open(EVALUATION_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print("\n모델 저장 완료:")
    print(MODEL_PATH)

    print("\n메타데이터 저장 완료:")
    print(META_PATH)

    print("\n평가 요약 저장 완료:")
    print(EVALUATION_SUMMARY_PATH)


# =========================
# 12. 메인 실행
# =========================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("[1] 데이터 불러오기")
    df = load_dataset(DATA_PATH)
    print(f"데이터 크기: {df.shape}")
    print(f"기간: {df['Date'].min()} ~ {df['Date'].max()}")

    print("\n[2] 라벨 분포 확인")
    print("\n방향성 라벨 분포:")
    print(df["direction_label"].value_counts())
    print("\n방향성 라벨 비율:")
    print((df["direction_label"].value_counts(normalize=True) * 100).round(2))

    print("\n위험도 라벨 분포:")
    print(df["risk_label"].value_counts())
    print("\n위험도 라벨 비율:")
    print((df["risk_label"].value_counts(normalize=True) * 100).round(2))

    print("\n[3] 피처 컬럼 선택")
    direction_feature_cols = get_direction_feature_columns(df)
    risk_feature_cols = get_risk_feature_columns(df)

    print(f"방향성 피처 수: {len(direction_feature_cols)}")
    print(direction_feature_cols)

    print(f"\n위험도 피처 수: {len(risk_feature_cols)}")
    print(risk_feature_cols)

    # =========================
    # 방향성 모델
    # =========================

    print("\n[4] 방향성 모델 데이터 분리")
    (
        direction_train_df,
        direction_test_df,
        X_direction_train,
        X_direction_test,
        y_direction_train,
        y_direction_test
    ) = time_series_train_test_split(
        df=df,
        feature_cols=direction_feature_cols,
        target_col="direction_label",
        test_size=TEST_SIZE
    )

    print(f"방향성 학습 데이터: {X_direction_train.shape}")
    print(f"방향성 테스트 데이터: {X_direction_test.shape}")
    print(f"방향성 학습 기간: {direction_train_df['Date'].min()} ~ {direction_train_df['Date'].max()}")
    print(f"방향성 테스트 기간: {direction_test_df['Date'].min()} ~ {direction_test_df['Date'].max()}")

    print("\n[5] 방향성 모델 전처리")
    X_direction_train_processed, X_direction_test_processed, direction_imputer = preprocess_features(
        X_direction_train,
        X_direction_test
    )

    print("\n[6] 방향성 라벨 인코딩")
    y_direction_train_encoded, y_direction_test_encoded, direction_encoder = encode_labels(
        y_direction_train,
        y_direction_test
    )
    print("방향성 라벨 목록:", direction_encoder.classes_.tolist())

    print("\n[7] 방향성 모델 학습")
    direction_model = train_single_model(
        X_train=X_direction_train_processed,
        y_train=y_direction_train_encoded,
        model_type="direction"
    )

    print("\n[8] 방향성 모델 평가")
    direction_pred, direction_metrics = evaluate_model(
        model=direction_model,
        X_test=X_direction_test_processed,
        y_test=y_direction_test_encoded,
        label_encoder=direction_encoder,
        title="방향성 모델"
    )

    save_confusion_matrix(
        y_test=y_direction_test_encoded,
        y_pred=direction_pred,
        label_encoder=direction_encoder,
        save_path=DIRECTION_CM_PATH,
        title="Direction Confusion Matrix"
    )

    save_feature_importance(
        model=direction_model,
        feature_cols=direction_feature_cols,
        save_img_path=DIRECTION_FEATURE_IMPORTANCE_PATH,
        save_csv_path=DIRECTION_FEATURE_IMPORTANCE_CSV_PATH,
        title="Direction Feature Importance"
    )

    # =========================
    # 위험도 모델
    # =========================

    print("\n[9] 위험도 모델 데이터 분리")
    (
        risk_train_df,
        risk_test_df,
        X_risk_train,
        X_risk_test,
        y_risk_train,
        y_risk_test
    ) = time_series_train_test_split(
        df=df,
        feature_cols=risk_feature_cols,
        target_col="risk_label",
        test_size=TEST_SIZE
    )

    print(f"위험도 학습 데이터: {X_risk_train.shape}")
    print(f"위험도 테스트 데이터: {X_risk_test.shape}")
    print(f"위험도 학습 기간: {risk_train_df['Date'].min()} ~ {risk_train_df['Date'].max()}")
    print(f"위험도 테스트 기간: {risk_test_df['Date'].min()} ~ {risk_test_df['Date'].max()}")

    print("\n[10] 위험도 모델 전처리")
    X_risk_train_processed, X_risk_test_processed, risk_imputer = preprocess_features(
        X_risk_train,
        X_risk_test
    )

    print("\n[11] 위험도 라벨 인코딩")
    y_risk_train_encoded, y_risk_test_encoded, risk_encoder = encode_labels(
        y_risk_train,
        y_risk_test
    )
    print("위험도 라벨 목록:", risk_encoder.classes_.tolist())

    print("\n[12] 위험도 모델 학습")
    risk_model = train_single_model(
        X_train=X_risk_train_processed,
        y_train=y_risk_train_encoded,
        model_type="risk"
    )

    print("\n[13] 위험도 모델 평가")
    risk_pred, risk_metrics = evaluate_model(
        model=risk_model,
        X_test=X_risk_test_processed,
        y_test=y_risk_test_encoded,
        label_encoder=risk_encoder,
        title="위험도 모델"
    )

    save_confusion_matrix(
        y_test=y_risk_test_encoded,
        y_pred=risk_pred,
        label_encoder=risk_encoder,
        save_path=RISK_CM_PATH,
        title="Risk Confusion Matrix"
    )

    save_feature_importance(
        model=risk_model,
        feature_cols=risk_feature_cols,
        save_img_path=RISK_FEATURE_IMPORTANCE_PATH,
        save_csv_path=RISK_FEATURE_IMPORTANCE_CSV_PATH,
        title="Risk Feature Importance"
    )

    # =========================
    # 최신 데이터 예측
    # =========================

    print("\n[14] 최신 데이터 예측")
    latest_result = predict_latest_sample(
        direction_model=direction_model,
        direction_imputer=direction_imputer,
        direction_encoder=direction_encoder,
        direction_feature_cols=direction_feature_cols,
        risk_model=risk_model,
        risk_imputer=risk_imputer,
        risk_encoder=risk_encoder,
        risk_feature_cols=risk_feature_cols,
        df=df
    )

    # =========================
    # 저장
    # =========================

    print("\n[15] 모델 저장")
    metrics = {
        "direction_metrics": direction_metrics,
        "risk_metrics": risk_metrics,
        "latest_result": latest_result
    }

    save_model_package(
        direction_model=direction_model,
        direction_imputer=direction_imputer,
        direction_encoder=direction_encoder,
        direction_feature_cols=direction_feature_cols,
        risk_model=risk_model,
        risk_imputer=risk_imputer,
        risk_encoder=risk_encoder,
        risk_feature_cols=risk_feature_cols,
        metrics=metrics
    )

    print("\n완료")


if __name__ == "__main__":
    main()