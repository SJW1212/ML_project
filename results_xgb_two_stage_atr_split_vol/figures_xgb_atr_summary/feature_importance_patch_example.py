# ============================================================
# [선택] XGBoost walk-forward 내부에 feature importance 저장 추가 예시
# ============================================================
# 사용 위치:
# - stage1_model.fit(...) 직후
# - stage2_model.fit(...) 직후
#
# 전제:
# - feature_cols: 학습에 사용한 피처명 리스트
# - stage1_model, stage2_model: XGBClassifier
# - stage1_importance_history, stage2_importance_history: list

# 루프 바깥에 추가
stage1_importance_history = []
stage2_importance_history = []

# stage1_model.fit(...) 직후에 추가
stage1_importance_history.append(
    dict(zip(feature_cols, stage1_model.feature_importances_.tolist()))
)

# stage2_model.fit(...) 직후에 추가
stage2_importance_history.append(
    dict(zip(feature_cols, stage2_model.feature_importances_.tolist()))
)

# walk-forward 종료 후 summary 저장 전에 추가
import pandas as pd

def mean_importance(history):
    if len(history) == 0:
        return {}
    imp_df = pd.DataFrame(history).fillna(0.0)
    return imp_df.mean(axis=0).sort_values(ascending=False).to_dict()

summary["stage1_feature_importance_mean"] = mean_importance(stage1_importance_history)
summary["stage2_feature_importance_mean"] = mean_importance(stage2_importance_history)
