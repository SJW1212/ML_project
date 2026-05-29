from __future__ import annotations

"""
XGBoost ATR-aware Two-stage Split-vol summary visualizer

사용 방법:
1. 이 파일을 JSON 파일과 같은 폴더에 저장합니다.
2. 아래처럼 실행합니다.

   python xgb_atr_summary_visualizer.py --json "Pasted code.json"

출력:
- 기본 출력 폴더: figures_xgb_atr_summary/
- PNG 이미지와 CSV 요약 파일을 저장합니다.

주의:
- 현재 summary JSON에 실제 XGBoost feature importance 값이 없으면,
  피처 중요도 순위 그래프는 생성할 수 없습니다.
- 이 경우 피처 그룹 구성 그래프와 안내 텍스트를 생성합니다.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 기본 설정
# =========================

def setup_korean_font() -> None:
    """Windows/Colab/Linux 환경에서 한글 표시를 최대한 안정적으로 처리합니다."""
    candidates = [
        "Malgun Gothic",      # Windows
        "AppleGothic",        # macOS
        "NanumGothic",        # Linux/Colab if installed
        "Noto Sans CJK KR",
        "DejaVu Sans",
    ]
    plt.rcParams["font.family"] = candidates
    plt.rcParams["axes.unicode_minus"] = False


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_current_figure(out_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


# =========================
# 2. 유틸 함수
# =========================

def to_percent(x: float) -> float:
    return float(x) * 100.0


def safe_get(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def annotate_bars(ax, fmt: str = "{:.2f}", y_offset_ratio: float = 0.01) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * y_offset_ratio
    for bar in ax.patches:
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, height + offset, fmt.format(height), ha="center", va="bottom", fontsize=9)


def sanitize_filename(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


# =========================
# 3. 성능 시각화
# =========================

def plot_strategy_cagr_mdd(summary: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    performance = summary.get("performance", {})
    rows = []
    for model_name, metrics in performance.items():
        rows.append({
            "model": model_name,
            "CAGR(%)": to_percent(metrics.get("cagr", np.nan)),
            "MDD(%)": to_percent(metrics.get("mdd", np.nan)),
            "Sharpe": metrics.get("sharpe", np.nan),
            "Sortino": metrics.get("sortino", np.nan),
            "Calmar": metrics.get("calmar", np.nan),
            "Final Capital": metrics.get("final_capital", np.nan),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df.to_csv(out_dir / "performance_summary.csv", index=False, encoding="utf-8-sig")

    # CAGR
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.bar(df["model"], df["CAGR(%)"])
    ax.set_title("전략별 CAGR 비교")
    ax.set_ylabel("CAGR (%)")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    annotate_bars(ax, "{:.2f}")
    save_current_figure(out_dir / "01_strategy_cagr.png")

    # MDD
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.bar(df["model"], df["MDD(%)"])
    ax.set_title("전략별 MDD 비교")
    ax.set_ylabel("MDD (%)")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    for bar in ax.patches:
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, height, f"{height:.2f}", ha="center", va="top", fontsize=9)
    save_current_figure(out_dir / "02_strategy_mdd.png")

    # Sharpe
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.bar(df["model"], df["Sharpe"])
    ax.set_title("전략별 Sharpe 비교")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    annotate_bars(ax, "{:.3f}")
    save_current_figure(out_dir / "03_strategy_sharpe.png")

    # Calmar
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.bar(df["model"], df["Calmar"])
    ax.set_title("전략별 Calmar 비교")
    ax.set_ylabel("Calmar")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    annotate_bars(ax, "{:.3f}")
    save_current_figure(out_dir / "04_strategy_calmar.png")

    return df


def plot_stage1_classification(summary: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    s1 = summary.get("stage1_risk_classification", {})
    metrics = {
        "Accuracy": s1.get("accuracy", np.nan),
        "Macro F1": s1.get("macro_f1", np.nan),
        "High-vol Precision": s1.get("high_vol_precision", np.nan),
        "High-vol Recall": s1.get("high_vol_recall", np.nan),
        "High-vol F1": s1.get("high_vol_f1", np.nan),
        "ROC-AUC": s1.get("roc_auc", np.nan),
        "PR-AUC": s1.get("pr_auc", np.nan),
        "Brier": s1.get("brier", np.nan),
    }
    df = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    df.to_csv(out_dir / "stage1_classification_metrics.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.bar(df["metric"], df["value"])
    ax.set_title("1단계 Risk 분류 성능")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(1.0, float(np.nanmax(df["value"])) * 1.15))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    annotate_bars(ax, "{:.3f}")
    save_current_figure(out_dir / "05_stage1_classification_metrics.png")

    return df


def plot_split_vol_classification(summary: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    split = summary.get("split_vol_classification", {})
    per_class = split.get("per_class", {})
    rows = []
    for cls, vals in per_class.items():
        rows.append({
            "class": cls,
            "precision": vals.get("precision", np.nan),
            "recall": vals.get("recall", np.nan),
            "f1": vals.get("f1", np.nan),
            "support": vals.get("support", np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "split_vol_per_class_metrics.csv", index=False, encoding="utf-8-sig")

    if not df.empty:
        # Precision
        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        ax.bar(df["class"], df["precision"])
        ax.set_title("3클래스 Split-vol Precision")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        annotate_bars(ax, "{:.3f}")
        save_current_figure(out_dir / "06_split_vol_precision.png")

        # Recall
        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        ax.bar(df["class"], df["recall"])
        ax.set_title("3클래스 Split-vol Recall")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        annotate_bars(ax, "{:.3f}")
        save_current_figure(out_dir / "07_split_vol_recall.png")

        # F1
        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        ax.bar(df["class"], df["f1"])
        ax.set_title("3클래스 Split-vol F1")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        annotate_bars(ax, "{:.3f}")
        save_current_figure(out_dir / "08_split_vol_f1.png")

        # Support
        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        ax.bar(df["class"], df["support"])
        ax.set_title("3클래스 Split-vol Label Support")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        annotate_bars(ax, "{:.0f}")
        save_current_figure(out_dir / "09_split_vol_label_support.png")

    summary_rows = [
        {"metric": "Accuracy", "value": split.get("accuracy", np.nan)},
        {"metric": "Macro F1", "value": split.get("macro_f1", np.nan)},
        {"metric": "Down ROC-AUC", "value": split.get("down_high_vol_roc_auc", np.nan)},
        {"metric": "Down PR-AUC", "value": split.get("down_high_vol_pr_auc", np.nan)},
        {"metric": "Up ROC-AUC", "value": split.get("up_high_vol_roc_auc", np.nan)},
        {"metric": "Up PR-AUC", "value": split.get("up_high_vol_pr_auc", np.nan)},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "split_vol_summary_metrics.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.bar(summary_df["metric"], summary_df["value"])
    ax.set_title("3클래스 Split-vol 요약 성능")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    annotate_bars(ax, "{:.3f}")
    save_current_figure(out_dir / "10_split_vol_summary_metrics.png")

    return df


def plot_threshold_diagnostics(summary: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    diagnostics = summary.get("down_high_vol_threshold_diagnostics", [])
    df = pd.DataFrame(diagnostics)
    if df.empty:
        return df
    df.to_csv(out_dir / "down_high_vol_threshold_diagnostics.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(df["threshold"], df["down_high_vol_precision"], marker="o", label="Precision")
    ax.plot(df["threshold"], df["down_high_vol_recall"], marker="o", label="Recall")
    ax.plot(df["threshold"], df["down_high_vol_f1"], marker="o", label="F1")
    ax.set_title("하락고변동 Threshold별 Precision / Recall / F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_current_figure(out_dir / "11_down_high_vol_threshold_metrics.png")

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(df["threshold"], df["pred_down_high_vol_ratio"], marker="o")
    ax.set_title("하락고변동 Threshold별 예측 비율")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Predicted Down-high-vol Ratio")
    ax.set_ylim(0, max(1.0, float(df["pred_down_high_vol_ratio"].max()) * 1.1))
    ax.grid(True, alpha=0.3)
    save_current_figure(out_dir / "12_down_high_vol_pred_ratio.png")

    return df


# =========================
# 4. 확률/배분 시각화
# =========================

def plot_average_probabilities(summary: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    probs = summary.get("average_probabilities", {})
    rows = []
    for key, val in probs.items():
        if isinstance(val, (int, float)):
            rows.append({"probability": key, "value(%)": to_percent(val)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "average_probabilities.csv", index=False, encoding="utf-8-sig")

    if not df.empty:
        plt.figure(figsize=(11, 6))
        ax = plt.gca()
        ax.bar(df["probability"], df["value(%)"])
        ax.set_title("평균 예측 확률")
        ax.set_ylabel("Probability (%)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
        annotate_bars(ax, "{:.2f}")
        save_current_figure(out_dir / "13_average_probabilities.png")

    return df


def plot_average_weights_and_turnover(summary: Dict[str, Any], out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weights = summary.get("average_weights", {})
    weight_keys = ["avg_stock_weight", "avg_bond_weight", "avg_cash_weight"]
    weight_rows = []
    for key in weight_keys:
        if key in weights:
            weight_rows.append({"asset": key, "weight(%)": to_percent(weights[key])})
    weights_df = pd.DataFrame(weight_rows)
    weights_df.to_csv(out_dir / "average_weights.csv", index=False, encoding="utf-8-sig")

    if not weights_df.empty:
        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        ax.bar(weights_df["asset"], weights_df["weight(%)"])
        ax.set_title("평균 자산 비중")
        ax.set_ylabel("Weight (%)")
        ax.set_ylim(0, max(100, float(weights_df["weight(%)"].max()) * 1.15))
        ax.grid(axis="y", alpha=0.3)
        annotate_bars(ax, "{:.2f}")
        save_current_figure(out_dir / "14_average_weights.png")

    turnover = summary.get("turnover", {})
    turnover_rows = [
        {"metric": "avg_daily_trade_ratio(%)", "value": to_percent(turnover.get("avg_daily_trade_ratio", np.nan))},
        {"metric": "annual_turnover_estimate(%)", "value": to_percent(turnover.get("annual_turnover_estimate", np.nan))},
        {"metric": "total_transaction_cost_rate_sum(%)", "value": to_percent(turnover.get("total_transaction_cost_rate_sum", np.nan))},
    ]
    turnover_df = pd.DataFrame(turnover_rows)
    turnover_df.to_csv(out_dir / "turnover_summary.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(turnover_df["metric"], turnover_df["value"])
    ax.set_title("Turnover / 거래비용 요약")
    ax.set_ylabel("Percent (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    annotate_bars(ax, "{:.2f}")
    save_current_figure(out_dir / "15_turnover_summary.png")

    return weights_df, turnover_df


def plot_latest_signal(summary: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    latest = summary.get("latest_prediction", {})
    if not latest:
        return pd.DataFrame()

    prob_keys = ["prob_normal", "prob_high_vol", "prob_up_high_vol", "prob_down_high_vol"]
    prob_rows = []
    for key in prob_keys:
        if key in latest:
            prob_rows.append({"probability": key, "value(%)": float(latest[key])})
    prob_df = pd.DataFrame(prob_rows)
    prob_df.to_csv(out_dir / "latest_probabilities.csv", index=False, encoding="utf-8-sig")

    if not prob_df.empty:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.bar(prob_df["probability"], prob_df["value(%)"])
        ax.set_title(f"최신 예측 확률: {latest.get('date', '')}")
        ax.set_ylabel("Probability (%)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
        annotate_bars(ax, "{:.2f}")
        save_current_figure(out_dir / "16_latest_probabilities.png")

    alloc = latest.get("target_allocation", {})
    alloc_df = pd.DataFrame([
        {"asset": "stock", "weight(%)": alloc.get("stock", np.nan)},
        {"asset": "bond", "weight(%)": alloc.get("bond", np.nan)},
        {"asset": "cash", "weight(%)": alloc.get("cash", np.nan)},
    ])
    alloc_df.to_csv(out_dir / "latest_allocation.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.bar(alloc_df["asset"], alloc_df["weight(%)"])
    ax.set_title(f"최신 목표 자산 비중: {latest.get('date', '')}")
    ax.set_ylabel("Weight (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    annotate_bars(ax, "{:.2f}")
    save_current_figure(out_dir / "17_latest_allocation.png")

    return prob_df


# =========================
# 5. 피처 관련 시각화
# =========================

def infer_feature_group(feature_name: str) -> str:
    f = feature_name.lower()
    if "atr" in f or "true_range" in f or "keltner" in f:
        return "ATR/Range"
    if "parkinson" in f or "garman" in f or "rogers" in f or "yang_zhang" in f:
        return "Range-based Volatility"
    if "vol" in f or "squeeze" in f or "bb_width" in f:
        return "Volatility"
    if "downside" in f or "semi" in f or "ulcer" in f or "drawdown" in f:
        return "Downside Risk"
    if "volume" in f:
        return "Volume"
    if "ma" in f or "trend" in f or "slope" in f:
        return "Trend/MA"
    if "return" in f or "positive" in f or "large_up" in f or "large_down" in f:
        return "Return/Momentum"
    if "position" in f or "high" in f or "low" in f:
        return "Price Position"
    return "Other"


def plot_feature_group_distribution(summary: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    features = summary.get("feature_cols", [])
    rows = [{"feature": f, "group": infer_feature_group(f)} for f in features]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "feature_group_mapping.csv", index=False, encoding="utf-8-sig")

    if df.empty:
        return df

    group_counts = df["group"].value_counts().reset_index()
    group_counts.columns = ["group", "count"]
    group_counts.to_csv(out_dir / "feature_group_counts.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    ax.bar(group_counts["group"], group_counts["count"])
    ax.set_title("피처 그룹별 개수")
    ax.set_ylabel("Feature Count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    annotate_bars(ax, "{:.0f}")
    save_current_figure(out_dir / "18_feature_group_counts.png")

    return df


def extract_importance_dict(summary: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    summary JSON 안에 feature importance가 들어있는 경우만 추출합니다.
    현재 제공된 JSON에는 일반적으로 이 값이 없습니다.
    """
    candidate_paths = [
        ["feature_importance"],
        ["feature_importances"],
        ["stage1_feature_importance"],
        ["stage1_feature_importances"],
        ["stage1_risk_classification", "feature_importance"],
        ["stage1_risk_classification", "feature_importances"],
    ]

    for path in candidate_paths:
        obj = safe_get(summary, path)
        if isinstance(obj, dict) and obj:
            parsed = {}
            for k, v in obj.items():
                if isinstance(v, (int, float)):
                    parsed[str(k)] = float(v)
            if parsed:
                return parsed

    return None


def plot_feature_importance_if_available(summary: Dict[str, Any], out_dir: Path, top_n: int = 25) -> Optional[pd.DataFrame]:
    importance = extract_importance_dict(summary)
    if importance is None:
        note = (
            "현재 summary JSON에는 실제 XGBoost feature importance 값이 없습니다.\n"
            "따라서 피처 중요도 순위 그래프는 이 JSON만으로 생성할 수 없습니다.\n\n"
            "가능한 대안:\n"
            "1. 학습 코드에서 각 walk-forward 모델의 booster.feature_importances_를 저장\n"
            "2. stage1/stage2 feature importance를 평균내어 summary JSON에 추가\n"
            "3. 그 후 이 스크립트를 다시 실행\n"
        )
        (out_dir / "feature_importance_not_available.txt").write_text(note, encoding="utf-8")
        print("[INFO] feature importance가 JSON에 없어 순위 그래프를 건너뜁니다.")
        return None

    df = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    top_df = df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    ax = plt.gca()
    ax.barh(top_df["feature"], top_df["importance"])
    ax.set_title(f"XGBoost Feature Importance Top {top_n}")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    save_current_figure(out_dir / "19_feature_importance_topn.png")
    return df


def write_feature_importance_patch(out_dir: Path) -> None:
    patch_code = r'''
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
'''
    (out_dir / "feature_importance_patch_example.py").write_text(patch_code.strip() + "\n", encoding="utf-8")
    print(f"[SAVE] {out_dir / 'feature_importance_patch_example.py'}")


# =========================
# 6. 메인 실행
# =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="Pasted code.json", help="summary JSON 파일 경로")
    parser.add_argument("--out", default="figures_xgb_atr_summary", help="시각화 결과 저장 폴더")
    parser.add_argument("--top-n", type=int, default=25, help="feature importance 상위 N개")
    args = parser.parse_args()

    setup_korean_font()
    summary = load_json("qqq_xgb_two_stage_atr_split_vol_summary.json")
    out_dir = ensure_dir(args.out)

    print("[INFO] JSON loaded")
    print(f"[INFO] model_type: {summary.get('model_type')}")
    print(f"[INFO] period: {summary.get('period')}")
    print(f"[INFO] feature_count: {summary.get('feature_count')}")

    plot_strategy_cagr_mdd(summary, out_dir)
    plot_stage1_classification(summary, out_dir)
    plot_split_vol_classification(summary, out_dir)
    plot_threshold_diagnostics(summary, out_dir)
    plot_average_probabilities(summary, out_dir)
    plot_average_weights_and_turnover(summary, out_dir)
    plot_latest_signal(summary, out_dir)
    plot_feature_group_distribution(summary, out_dir)
    plot_feature_importance_if_available(summary, out_dir, top_n=args.top_n)
    write_feature_importance_patch(out_dir)

    print("\n[DONE] 모든 시각화 코드 실행 완료")
    print(f"[OUT] {out_dir.resolve()}")


if __name__ == "__main__":
    main()
