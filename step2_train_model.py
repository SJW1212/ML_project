# Python 3.10+
# 필요 패키지:
# pip install pandas numpy scikit-learn matplotlib joblib

import os
import json
import joblib
import numpy as np
import pandas as pd
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

DATA_PATH = "data/qqq_features_labeled.csv"

MODEL_DIR = "models"
RESULT_DIR = "results"

MODEL_PATH = os.path.join(MODEL_DIR, "qqq_regime_randomforest.pkl")
META_PATH = os.path.join(MODEL_DIR, "qqq_regime_metadata.json")

CONFUSION_MATRIX_PATH = os.path.join(RESULT_DIR, "confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = os.path.join(RESULT_DIR, "feature_importance.png")
FEATURE_IMPORTANCE_CSV_PATH = os.path.join(RESULT_DIR, "feature_importance.csv")

TEST_SIZE = 0.2
RANDOM_STATE = 42


# =========================
# 2. 데이터 불러오기
# =========================

def load_dataset(path: str) -> pd.DataFrame:
    """
    이전 단계에서 생성한 라벨링된 QQQ 데이터셋을 불러온다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {path}\n"
            "먼저 step1_make_dataset.py를 실행해 qqq_features_labeled.csv를 생성하세요."
        )

    df = pd.read_csv(path)

    if "label" not in df.columns:
        raise ValueError("데이터셋에 label 컬럼이 없습니다.")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    return df


# =========================
# 3. 피처 선택
# =========================

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    모델 학습에 사용할 피처 컬럼을 선택한다.

    주의:
    - future_return_20d, future_volatility_20d는 미래 정보이므로 반드시 제외한다.
    - label은 정답이므로 제외한다.
    - Date는 날짜 식별자이므로 제외한다.
    - Open, High, Low, Close, Volume 같은 원시 가격은 일단 제외한다.
      이유: 가격 단위에 의존하면 다른 종목 확장성이 낮아질 수 있기 때문이다.
    """

    exclude_cols = {
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "cummax_close",
        "future_return_20d",
        "future_volatility_20d",
        "label"
    }

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("사용 가능한 피처 컬럼이 없습니다.")

    return feature_cols


# =========================
# 4. 시계열 기준 학습/테스트 분리
# =========================

def time_series_train_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "label",
    test_size: float = 0.2
):
    """
    금융 시계열 데이터이므로 랜덤 셔플을 하지 않고 시간 순서대로 분리한다.

    앞 80%: 학습 데이터
    뒤 20%: 테스트 데이터
    """
    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return train_df, test_df, X_train, X_test, y_train, y_test


# =========================
# 5. 결측값 처리
# =========================

def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    결측값을 중앙값으로 대체한다.

    RandomForest는 스케일링이 필수는 아니므로 여기서는 스케일러를 사용하지 않는다.
    이유:
    - 트리 기반 모델은 변수 스케일에 민감하지 않다.
    - 불필요한 전처리를 줄여 해석 가능성을 높인다.
    """
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    return X_train_imputed, X_test_imputed, imputer


# =========================
# 6. 라벨 인코딩
# =========================

def encode_labels(y_train: pd.Series, y_test: pd.Series):
    """
    문자열 라벨을 숫자 라벨로 변환한다.
    예: 상승, 하락, 횡보, 고변동, 불확실 → 0, 1, 2, 3, 4
    """
    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    return y_train_encoded, y_test_encoded, encoder


# =========================
# 7. 모델 학습
# =========================

def train_model(X_train, y_train):
    """
    RandomForestClassifier 학습.

    class_weight='balanced_subsample':
    - 라벨 분포가 불균형할 때 일부 보정 효과가 있다.
    - 금융 데이터는 상승/횡보/고변동 등 라벨 비율이 균등하지 않을 수 있으므로 사용한다.
    """
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# =========================
# 8. 모델 평가
# =========================

def evaluate_model(model, X_test, y_test, label_encoder: LabelEncoder):
    """
    테스트 데이터 기준으로 분류 성능을 평가한다.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    target_names = label_encoder.classes_

    print("\n==============================")
    print("모델 평가 결과")
    print("==============================")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    print("\n분류 리포트:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    ))

    return y_pred, accuracy, macro_f1, weighted_f1


# =========================
# 9. Confusion Matrix 저장
# =========================

def save_confusion_matrix(y_test, y_pred, label_encoder: LabelEncoder, save_path: str):
    """
    Confusion Matrix 이미지를 저장한다.
    """
    labels = list(range(len(label_encoder.classes_)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
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
    plt.close()

    print(f"\nConfusion Matrix 저장 완료: {save_path}")


# =========================
# 10. Feature Importance 저장
# =========================

def save_feature_importance(model, feature_cols: list[str], save_img_path: str, save_csv_path: str):
    """
    RandomForest의 피처 중요도를 저장한다.
    """
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
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    plt.savefig(save_img_path, dpi=150)
    plt.close()

    print(f"Feature Importance 이미지 저장 완료: {save_img_path}")
    print(f"Feature Importance CSV 저장 완료: {save_csv_path}")

    print("\n상위 피처 중요도:")
    print(importance_df.head(10))


# =========================
# 11. 최신 데이터 1건 예측
# =========================
def apply_uncertainty_rule(
    pred_label: str,
    proba_result: dict[str, float],
    min_confidence: float = 40.0,
    min_margin: float = 8.0
) -> str:
    """
    모델 예측 확률을 기준으로 '불확실' 여부를 사후 판단한다.

    조건:
    1. 최고 확률이 min_confidence보다 낮으면 불확실
    2. 1등 확률과 2등 확률 차이가 min_margin보다 작으면 불확실
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

def predict_latest_sample(
    model,
    imputer,
    label_encoder: LabelEncoder,
    df: pd.DataFrame,
    feature_cols: list[str]
):
    """
    가장 최근 데이터 1건에 대해 시장 국면 확률을 출력한다.
    """
    latest_row = df.iloc[-1:].copy()
    latest_features = latest_row[feature_cols]

    latest_features_imputed = imputer.transform(latest_features)

    pred_encoded = model.predict(latest_features_imputed)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    proba = model.predict_proba(latest_features_imputed)[0]
    proba_result = {
        label_encoder.classes_[i]: round(float(proba[i]) * 100, 2)
        for i in range(len(label_encoder.classes_))
    }

    print("\n==============================")
    print("최신 데이터 기준 예측 결과")
    print("==============================")

    if "Date" in latest_row.columns:
        print(f"기준일: {latest_row['Date'].iloc[0]}")

    print(f"예측 국면: {pred_label}")

    print("\n국면별 확률:")
    for label, prob in sorted(proba_result.items(), key=lambda x: x[1], reverse=True):
        print(f"- {label}: {prob:.2f}%")

    return pred_label, proba_result


# =========================
# 12. 자산배분 규칙
# =========================

def recommend_allocation(proba_result: dict[str, float]) -> dict[str, int]:
    """
    국면별 확률을 바탕으로 주식/채권/현금 비율을 추천한다.

    주의:
    - 실제 투자 추천이 아니라 프로젝트용 의사결정 규칙이다.
    - 확률값은 모델이 산출한 분류 확률이지, 실제 미래 확률을 보장하지 않는다.
    """
    up = proba_result.get("상승", 0)
    down = proba_result.get("하락", 0)
    sideways = proba_result.get("횡보", 0)
    high_vol = proba_result.get("고변동", 0)
    uncertain = proba_result.get("불확실", 0)

    if down >= 50:
        allocation = {
            "주식": 20,
            "채권": 45,
            "현금": 35
        }
        reason = "하락 확률이 높아 방어적 배분"

    elif high_vol >= 50:
        allocation = {
            "주식": 30,
            "채권": 30,
            "현금": 40
        }
        reason = "고변동 확률이 높아 현금 비중 확대"

    elif up >= 60 and down < 25:
        allocation = {
            "주식": 75,
            "채권": 15,
            "현금": 10
        }
        reason = "상승 확률이 높고 하락 확률이 낮아 주식 비중 확대"

    elif up >= 45:
        allocation = {
            "주식": 60,
            "채권": 25,
            "현금": 15
        }
        reason = "상승 우세지만 확신이 강하지 않아 중간 수준 주식 비중"

    elif uncertain >= 40:
        allocation = {
            "주식": 40,
            "채권": 30,
            "현금": 30
        }
        reason = "불확실 확률이 높아 중립적 배분"

    elif sideways >= 40:
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
        reason = "명확한 우세 국면이 없어 기본 배분"

    print("\n==============================")
    print("자산배분 추천 결과")
    print("==============================")
    print(f"추천 사유: {reason}")
    print(f"주식: {allocation['주식']}%")
    print(f"채권: {allocation['채권']}%")
    print(f"현금: {allocation['현금']}%")

    return allocation


# =========================
# 13. 모델 저장
# =========================

def save_model_and_metadata(
    model,
    imputer,
    label_encoder: LabelEncoder,
    feature_cols: list[str],
    metrics: dict,
    model_path: str,
    meta_path: str
):
    """
    학습된 모델과 부가 정보를 저장한다.

    저장 구성:
    - model
    - imputer
    - label_encoder
    - feature_cols
    """
    model_package = {
        "model": model,
        "imputer": imputer,
        "label_encoder": label_encoder,
        "feature_cols": feature_cols
    }

    joblib.dump(model_package, model_path)

    metadata = {
        "model_type": "RandomForestClassifier",
        "feature_cols": feature_cols,
        "labels": label_encoder.classes_.tolist(),
        "metrics": metrics,
        "note": "Time-series split used. No shuffle. Future columns excluded to avoid leakage."
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"\n모델 저장 완료: {model_path}")
    print(f"메타데이터 저장 완료: {meta_path}")


# =========================
# 14. 메인 실행
# =========================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("[1] 데이터 불러오기")
    df = load_dataset(DATA_PATH)
    print(f"데이터 크기: {df.shape}")

    print("\n[2] 피처 컬럼 선택")
    feature_cols = get_feature_columns(df)
    print(f"사용 피처 수: {len(feature_cols)}")
    print(feature_cols)

    print("\n[3] 라벨 분포 확인")
    print(df["label"].value_counts())
    print("\n라벨 비율:")
    print((df["label"].value_counts(normalize=True) * 100).round(2))

    print("\n[4] 시간 순서 기준 학습/테스트 분리")
    train_df, test_df, X_train, X_test, y_train, y_test = time_series_train_test_split(
        df=df,
        feature_cols=feature_cols,
        target_col="label",
        test_size=TEST_SIZE
    )

    print(f"학습 데이터: {X_train.shape}")
    print(f"테스트 데이터: {X_test.shape}")

    if "Date" in train_df.columns and "Date" in test_df.columns:
        print(f"학습 기간: {train_df['Date'].min()} ~ {train_df['Date'].max()}")
        print(f"테스트 기간: {test_df['Date'].min()} ~ {test_df['Date'].max()}")

    print("\n[5] 결측값 처리")
    X_train_processed, X_test_processed, imputer = preprocess_features(X_train, X_test)

    print("\n[6] 라벨 인코딩")
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)
    print("라벨 목록:", label_encoder.classes_.tolist())

    print("\n[7] 모델 학습")
    model = train_model(X_train_processed, y_train_encoded)

    print("\n[8] 모델 평가")
    y_pred, accuracy, macro_f1, weighted_f1 = evaluate_model(
        model=model,
        X_test=X_test_processed,
        y_test=y_test_encoded,
        label_encoder=label_encoder
    )

    print("\n[9] 결과 이미지 저장")
    save_confusion_matrix(
        y_test=y_test_encoded,
        y_pred=y_pred,
        label_encoder=label_encoder,
        save_path=CONFUSION_MATRIX_PATH
    )

    save_feature_importance(
        model=model,
        feature_cols=feature_cols,
        save_img_path=FEATURE_IMPORTANCE_PATH,
        save_csv_path=FEATURE_IMPORTANCE_CSV_PATH
    )

    print("\n[10] 최신 데이터 예측")
    pred_label, proba_result = predict_latest_sample(
        model=model,
        imputer=imputer,
        label_encoder=label_encoder,
        df=df,
        feature_cols=feature_cols
    )

    print("\n[11] 자산배분 추천")
    allocation = recommend_allocation(proba_result)

    print("\n[12] 모델 저장")
    metrics = {
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "latest_pred_label": pred_label,
        "latest_proba": proba_result,
        "latest_allocation": allocation
    }

    save_model_and_metadata(
        model=model,
        imputer=imputer,
        label_encoder=label_encoder,
        feature_cols=feature_cols,
        metrics=metrics,
        model_path=MODEL_PATH,
        meta_path=META_PATH
    )

    print("\n완료")


if __name__ == "__main__":
    main()