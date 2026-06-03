# -*- coding: utf-8 -*-
"""
direction_auc_validation_report.py

Direction AUC 전수 탐색 결과 검증/평가 전용 스크립트.

목적
----
사용자가 산출한 direction_auc_* 결과 파일을 읽어 다음을 자동 계산합니다.

1. Trial 전체 재평가
   - Brier baseline
   - Brier skill
   - PR-AUC gain
   - PR-AUC ratio
   - AUC edge
   - inverse signal 여부
   - candidate decision

2. Top 후보 재정렬
   - 단순 score 기준이 아니라
   - AUC, PR 개선, Brier baseline 개선, polarity, calibration 위험을 함께 반영

3. Best prediction 연도별 안정성
   - yearly ROC-AUC
   - yearly PR-AUC
   - yearly positive rate
   - yearly Brier
   - yearly ECE
   - yearly probability mean
   - yearly sample count

4. Calibration bin 분석
   - 전체 ECE
   - bin별 평균 예측확률과 실제 positive rate

5. 그룹 요약
   - horizon별
   - task별
   - label_mode별
   - feature_set별
   - model_type별

6. 최종 판정
   - stable_accept
   - holdout_candidate
   - weak_candidate
   - reject_or_diagnostic_only

입력 파일
---------
- direction_auc_trials.csv
- direction_auc_trials_top20.csv
- direction_auc_best_predictions.csv
- direction_auc_summary.json

실행 예
-------
python direction_auc_validation_report.py ^
  --trials direction_auc_trials.csv ^
  --top20 direction_auc_trials_top20.csv ^
  --predictions direction_auc_best_predictions.csv ^
  --summary direction_auc_summary.json ^
  --output-dir direction_auc_validation_output

필수 의존성
----------
- Python 3.10+
- numpy
- pandas
- scikit-learn

주의
----
이 코드는 "검증/평가 코드"입니다.
모델을 새로 학습하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


# ============================================================
# 0. 공통 유틸
# ============================================================

def safe_divide(a: float, b: float, default: float = np.nan) -> float:
    if b == 0 or pd.isna(b):
        return default
    return float(a / b)


def finite_or_nan(x) -> float:
    try:
        value = float(x)
        if math.isfinite(value):
            return value
        return float("nan")
    except Exception:
        return float("nan")


def to_jsonable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        try:
            return asdict(obj)
        except Exception:
            return str(obj)
    return str(obj)


def load_json(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=to_jsonable),
        encoding="utf-8",
    )
    return path


def save_csv(path: str | Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ============================================================
# 1. Calibration Diagnostics
# ============================================================

@dataclass
class CalibrationBin:
    bin_id: int
    left: float
    right: float
    count: int
    prob_mean: float
    actual_rate: float
    abs_gap: float


def calibration_bins(
    y_true: Sequence[float],
    prob: Sequence[float],
    n_bins: int = 10,
) -> Tuple[float, pd.DataFrame]:
    """
    Expected Calibration Error와 calibration bin table 계산.

    ECE = sum_bin (bin_count / n) * abs(avg_prob - actual_rate)
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = np.clip(p[mask], 0.0, 1.0)

    if len(y) == 0:
        return float("nan"), pd.DataFrame()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[CalibrationBin] = []
    ece = 0.0
    n = len(y)

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == n_bins - 1:
            m = (p >= left) & (p <= right)
        else:
            m = (p >= left) & (p < right)

        count = int(m.sum())
        if count == 0:
            rows.append(
                CalibrationBin(
                    bin_id=i,
                    left=float(left),
                    right=float(right),
                    count=0,
                    prob_mean=float("nan"),
                    actual_rate=float("nan"),
                    abs_gap=float("nan"),
                )
            )
            continue

        prob_mean = float(p[m].mean())
        actual_rate = float(y[m].mean())
        gap = abs(prob_mean - actual_rate)
        ece += (count / n) * gap

        rows.append(
            CalibrationBin(
                bin_id=i,
                left=float(left),
                right=float(right),
                count=count,
                prob_mean=prob_mean,
                actual_rate=actual_rate,
                abs_gap=float(gap),
            )
        )

    return float(ece), pd.DataFrame([asdict(r) for r in rows])


def binary_metrics(y_true: Sequence[float], prob: Sequence[float]) -> Dict[str, float]:
    """
    binary prediction metrics.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = np.clip(p[mask], 1e-12, 1 - 1e-12)

    if len(y) == 0:
        return {
            "sample_count": 0,
            "positive_rate": float("nan"),
            "roc_auc": float("nan"),
            "inverse_roc_auc": float("nan"),
            "best_roc_after_inversion": float("nan"),
            "pr_auc": float("nan"),
            "inverse_pr_auc": float("nan"),
            "brier": float("nan"),
            "brier_baseline": float("nan"),
            "brier_skill": float("nan"),
            "ece": float("nan"),
            "prob_mean": float("nan"),
            "prob_std": float("nan"),
        }

    positive_rate = float(y.mean())
    brier = float(brier_score_loss(y, p))
    brier_baseline = float(positive_rate * (1.0 - positive_rate))
    brier_skill = 1.0 - brier / brier_baseline if brier_baseline > 0 else float("nan")

    try:
        auc = float(roc_auc_score(y, p))
        inv_auc = float(roc_auc_score(y, 1.0 - p))
    except ValueError:
        auc = float("nan")
        inv_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y, p))
        inv_pr_auc = float(average_precision_score(y, 1.0 - p))
    except ValueError:
        pr_auc = float("nan")
        inv_pr_auc = float("nan")

    ece, _ = calibration_bins(y, p, n_bins=10)

    return {
        "sample_count": int(len(y)),
        "positive_rate": positive_rate,
        "roc_auc": auc,
        "inverse_roc_auc": inv_auc,
        "best_roc_after_inversion": max(auc, inv_auc) if np.isfinite(auc) and np.isfinite(inv_auc) else float("nan"),
        "pr_auc": pr_auc,
        "inverse_pr_auc": inv_pr_auc,
        "pr_gain": pr_auc - positive_rate if np.isfinite(pr_auc) else float("nan"),
        "pr_ratio": safe_divide(pr_auc, positive_rate),
        "brier": brier,
        "brier_baseline": brier_baseline,
        "brier_skill": brier_skill,
        "ece": ece,
        "prob_mean": float(p.mean()),
        "prob_std": float(p.std()),
    }


# ============================================================
# 2. Trial reassessment
# ============================================================

def reassess_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """
    전수 탐색 trial 결과를 baseline 대비로 재평가.
    """
    df = trials.copy()

    required = [
        "trial_id",
        "horizon",
        "task",
        "label_mode",
        "feature_set",
        "model_type",
        "base_rate",
        "roc_auc",
        "pr_auc",
        "inverse_roc_auc",
        "brier",
        "probability_polarity",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required trial columns: {missing}")

    df["brier_baseline"] = df["base_rate"] * (1.0 - df["base_rate"])
    df["brier_skill"] = 1.0 - df["brier"] / df["brier_baseline"]
    df["pr_gain"] = df["pr_auc"] - df["base_rate"]
    df["pr_ratio"] = df["pr_auc"] / df["base_rate"].replace(0, np.nan)
    df["auc_edge"] = df["roc_auc"] - 0.5
    df["inverse_auc_edge"] = df["inverse_roc_auc"] - 0.5
    df["auc_margin_vs_inverse"] = df["roc_auc"] - df["inverse_roc_auc"]

    # F1 threshold가 극단적으로 낮으면 base-rate 착시 위험 flag.
    if "best_f1_threshold" in df.columns:
        df["f1_threshold_too_low"] = df["best_f1_threshold"] < 0.05
    else:
        df["f1_threshold_too_low"] = False

    df["brier_worse_than_baseline"] = df["brier_skill"] < 0
    df["weak_auc_positive"] = df["roc_auc"] > 0.52
    df["moderate_auc_positive"] = df["roc_auc"] > 0.55
    df["normal_polarity_ok"] = df["probability_polarity"].eq("normal_better")
    df["inverse_polarity_risk"] = df["probability_polarity"].eq("inverse_better")
    df["pr_above_base"] = df["pr_gain"] > 0

    df["candidate_decision"] = df.apply(classify_trial_candidate, axis=1)

    # 보수적 후보 점수.
    # 주의: 이것은 최종 성능 점수가 아니라 정렬용 진단 점수.
    df["validation_candidate_score"] = (
        2.0 * df["auc_edge"].clip(lower=-0.2, upper=0.2)
        + 1.0 * df["pr_gain"].clip(lower=-0.2, upper=0.2)
        + 1.0 * df["brier_skill"].clip(lower=-1.0, upper=1.0)
        + 0.25 * df["normal_polarity_ok"].astype(float)
        - 0.50 * df["inverse_polarity_risk"].astype(float)
        - 0.25 * df["f1_threshold_too_low"].astype(float)
    )

    return df.sort_values("validation_candidate_score", ascending=False).reset_index(drop=True)


def classify_trial_candidate(row: pd.Series) -> str:
    """
    단일 trial 후보 등급 판정.

    stable_accept:
    - 방향성 실험만으로는 사실상 거의 나오기 어렵게 설계.
    - holdout과 after-cost 검증 전에는 최종 채택 금지.

    holdout_candidate:
    - 최소한 AUC/PR/Brier/polarity가 동시에 괜찮은 후보.

    weak_candidate:
    - 신호는 있으나 calibration이나 baseline 대비 문제가 있는 후보.

    reject_or_diagnostic_only:
    - 구조 채택 후보로 쓰기 어려운 trial.
    """
    roc = finite_or_nan(row.get("roc_auc"))
    pr_gain = finite_or_nan(row.get("pr_gain"))
    brier_skill = finite_or_nan(row.get("brier_skill"))
    normal = bool(row.get("normal_polarity_ok", False))
    inverse = bool(row.get("inverse_polarity_risk", False))
    f1_low = bool(row.get("f1_threshold_too_low", False))

    if (
        roc >= 0.56
        and pr_gain > 0
        and brier_skill > 0
        and normal
        and not f1_low
    ):
        return "holdout_candidate"

    if (
        roc >= 0.55
        and pr_gain > 0
        and normal
        and not inverse
    ):
        return "weak_candidate"

    if (
        roc >= 0.53
        and pr_gain > 0
        and normal
    ):
        return "weak_candidate"

    if inverse:
        return "diagnostic_inverse_signal"

    return "reject_or_diagnostic_only"


# ============================================================
# 3. Group summaries
# ============================================================

def summarize_by_group(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []

    for keys, part in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "trial_count": int(len(part)),
                "mean_roc_auc": float(part["roc_auc"].mean()),
                "median_roc_auc": float(part["roc_auc"].median()),
                "max_roc_auc": float(part["roc_auc"].max()),
                "mean_pr_gain": float(part["pr_gain"].mean()),
                "mean_pr_ratio": float(part["pr_ratio"].mean()),
                "mean_brier_skill": float(part["brier_skill"].mean()),
                "positive_brier_skill_rate": float((part["brier_skill"] > 0).mean()),
                "normal_polarity_rate": float((part["probability_polarity"] == "normal_better").mean()),
                "inverse_polarity_rate": float((part["probability_polarity"] == "inverse_better").mean()),
                "weak_or_better_count": int(part["candidate_decision"].isin(["holdout_candidate", "weak_candidate"]).sum()),
                "holdout_candidate_count": int((part["candidate_decision"] == "holdout_candidate").sum()),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["max_roc_auc", "mean_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)


def build_all_group_summaries(reassessed: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    groups = {
        "by_horizon": ["horizon"],
        "by_task": ["task"],
        "by_label_mode": ["label_mode"],
        "by_feature_set": ["feature_set"],
        "by_model_type": ["model_type"],
        "by_horizon_task": ["horizon", "task"],
        "by_feature_model": ["feature_set", "model_type"],
    }
    return {name: summarize_by_group(reassessed, cols) for name, cols in groups.items()}


# ============================================================
# 4. Best prediction analysis
# ============================================================

def analyze_best_predictions(pred: pd.DataFrame, n_bins: int = 10) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """
    best_predictions.csv 기반 전체/연도별/calibration bin 분석.
    """
    df = pred.copy()
    required = ["date", "y_true", "prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    # 기본적으로 prob가 있는 행만 평가.
    eval_df = df[np.isfinite(df["prob"]) & np.isfinite(df["y_true"])].copy()

    if "label_valid" in eval_df.columns:
        eval_df = eval_df[eval_df["label_valid"].astype(bool)]

    if "eval_filter" in eval_df.columns:
        eval_df = eval_df[eval_df["eval_filter"].astype(bool)]

    overall = binary_metrics(eval_df["y_true"], eval_df["prob"])
    ece, bins = calibration_bins(eval_df["y_true"], eval_df["prob"], n_bins=n_bins)

    yearly_rows = []
    for year, part in eval_df.groupby("year"):
        metrics = binary_metrics(part["y_true"], part["prob"])
        metrics["year"] = int(year)
        yearly_rows.append(metrics)

    yearly = pd.DataFrame(yearly_rows)
    if not yearly.empty:
        yearly = yearly[
            [
                "year",
                "sample_count",
                "positive_rate",
                "roc_auc",
                "inverse_roc_auc",
                "best_roc_after_inversion",
                "pr_auc",
                "pr_gain",
                "pr_ratio",
                "brier",
                "brier_baseline",
                "brier_skill",
                "ece",
                "prob_mean",
                "prob_std",
            ]
        ].sort_values("year")

    stability = compute_yearly_stability(yearly)

    report = {
        "overall": overall,
        "yearly_stability": stability,
        "evaluated_rows": int(len(eval_df)),
        "raw_rows": int(len(df)),
        "first_eval_date": str(eval_df["date"].min()) if len(eval_df) else None,
        "last_eval_date": str(eval_df["date"].max()) if len(eval_df) else None,
    }

    return report, yearly, bins


def compute_yearly_stability(yearly: pd.DataFrame) -> Dict[str, object]:
    if yearly is None or yearly.empty:
        return {}

    valid_auc = yearly["roc_auc"].dropna()
    valid_brier_skill = yearly["brier_skill"].dropna()
    valid_ece = yearly["ece"].dropna()

    return {
        "year_count": int(len(yearly)),
        "auc_mean": float(valid_auc.mean()) if len(valid_auc) else float("nan"),
        "auc_median": float(valid_auc.median()) if len(valid_auc) else float("nan"),
        "auc_std": float(valid_auc.std()) if len(valid_auc) > 1 else float("nan"),
        "auc_min": float(valid_auc.min()) if len(valid_auc) else float("nan"),
        "auc_max": float(valid_auc.max()) if len(valid_auc) else float("nan"),
        "year_auc_above_0_5_rate": float((valid_auc > 0.5).mean()) if len(valid_auc) else float("nan"),
        "year_auc_above_0_55_rate": float((valid_auc > 0.55).mean()) if len(valid_auc) else float("nan"),
        "brier_skill_positive_rate": float((valid_brier_skill > 0).mean()) if len(valid_brier_skill) else float("nan"),
        "ece_mean": float(valid_ece.mean()) if len(valid_ece) else float("nan"),
        "unstable_years_auc_below_0_5": yearly.loc[yearly["roc_auc"] < 0.5, "year"].astype(int).tolist(),
    }


# ============================================================
# 5. Final decision report
# ============================================================

def build_final_decision(
    summary: Dict,
    reassessed: pd.DataFrame,
    prediction_report: Dict[str, object],
) -> Dict[str, object]:
    best_trial = summary.get("best_trial", {})
    best_trial_id = best_trial.get("trial_id")

    best_row = None
    if best_trial_id and "trial_id" in reassessed.columns:
        match = reassessed[reassessed["trial_id"] == best_trial_id]
        if not match.empty:
            best_row = match.iloc[0].to_dict()

    polarity_counts = (
        reassessed["probability_polarity"].value_counts(dropna=False).to_dict()
        if "probability_polarity" in reassessed.columns else {}
    )

    decision = {
        "overall_judgment": "candidate_only_not_stable",
        "structure_impact": {
            "multi_head_structure": "keep",
            "direction_head_role": "auxiliary_signal",
            "riskoff_highvol_priority": "test_next",
            "allocation_direct_use": "forbidden_before_calibration_and_backtest",
        },
        "best_trial_id": best_trial_id,
        "best_trial_candidate_decision": best_row.get("candidate_decision") if best_row else None,
        "best_trial_key_metrics": {
            "horizon": best_row.get("horizon") if best_row else None,
            "task": best_row.get("task") if best_row else None,
            "label_mode": best_row.get("label_mode") if best_row else None,
            "feature_set": best_row.get("feature_set") if best_row else None,
            "model_type": best_row.get("model_type") if best_row else None,
            "roc_auc": best_row.get("roc_auc") if best_row else None,
            "pr_auc": best_row.get("pr_auc") if best_row else None,
            "base_rate": best_row.get("base_rate") if best_row else None,
            "pr_gain": best_row.get("pr_gain") if best_row else None,
            "pr_ratio": best_row.get("pr_ratio") if best_row else None,
            "brier": best_row.get("brier") if best_row else None,
            "brier_baseline": best_row.get("brier_baseline") if best_row else None,
            "brier_skill": best_row.get("brier_skill") if best_row else None,
            "probability_polarity": best_row.get("probability_polarity") if best_row else None,
        },
        "trial_counts": {
            "total_trials": int(len(reassessed)),
            "holdout_candidate_count": int((reassessed["candidate_decision"] == "holdout_candidate").sum()),
            "weak_candidate_count": int((reassessed["candidate_decision"] == "weak_candidate").sum()),
            "diagnostic_inverse_signal_count": int((reassessed["candidate_decision"] == "diagnostic_inverse_signal").sum()),
            "reject_or_diagnostic_only_count": int((reassessed["candidate_decision"] == "reject_or_diagnostic_only").sum()),
            "polarity_counts": polarity_counts,
        },
        "best_prediction_stability": prediction_report.get("yearly_stability", {}),
        "required_next_steps": [
            "Run holdout validation for top candidates.",
            "Run calibration comparison: raw vs calibrated probability.",
            "Run portfolio-level after-cost benchmark against 1/N, 60/40, buy-and-hold, constant_NORMAL.",
            "Run RiskOff and HighVol search before changing final model structure.",
            "Run multi-head ablation: Direction only, Direction+HighVol, Direction+RiskOff, Full multi-head.",
        ],
        "do_not_do": [
            "Do not promote best trial to stable model.",
            "Do not switch to Direction-centered allocation based on this result.",
            "Do not use best_f1 as core evidence when threshold is near zero.",
            "Do not use raw probability directly in AllocationService before calibration diagnostics.",
        ],
    }

    return decision


# ============================================================
# 6. Main
# ============================================================

def run_validation(
    trials_path: str | Path,
    top20_path: str | Path,
    predictions_path: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
    n_bins: int = 10,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(trials_path)
    top20 = pd.read_csv(top20_path) if top20_path else pd.DataFrame()
    predictions = pd.read_csv(predictions_path)
    summary = load_json(summary_path)

    reassessed = reassess_trials(trials)
    top20_reassessed = reassess_trials(top20) if not top20.empty else pd.DataFrame()

    group_summaries = build_all_group_summaries(reassessed)
    prediction_report, yearly, cal_bins = analyze_best_predictions(predictions, n_bins=n_bins)

    final_decision = build_final_decision(summary, reassessed, prediction_report)

    output_paths: Dict[str, Path] = {}
    output_paths["trial_reassessment"] = save_csv(
        output_dir / "direction_auc_trial_reassessment.csv",
        reassessed,
    )
    if not top20_reassessed.empty:
        output_paths["top20_reassessment"] = save_csv(
            output_dir / "direction_auc_top20_reassessment.csv",
            top20_reassessed,
        )

    for name, df in group_summaries.items():
        output_paths[name] = save_csv(output_dir / f"direction_auc_group_{name}.csv", df)

    output_paths["best_prediction_yearly"] = save_csv(
        output_dir / "direction_auc_best_prediction_yearly.csv",
        yearly,
    )
    output_paths["best_prediction_calibration_bins"] = save_csv(
        output_dir / "direction_auc_best_prediction_calibration_bins.csv",
        cal_bins,
    )
    output_paths["validation_report"] = save_json(
        output_dir / "direction_auc_validation_report.json",
        {
            "input_files": {
                "trials": str(trials_path),
                "top20": str(top20_path),
                "predictions": str(predictions_path),
                "summary": str(summary_path),
            },
            "summary_meta": {
                "created_at": summary.get("created_at"),
                "ticker": summary.get("ticker"),
                "period": summary.get("period"),
                "trial_count": summary.get("trial_count"),
                "base_script": summary.get("base_script"),
            },
            "best_prediction_report": prediction_report,
            "final_decision": final_decision,
        },
    )

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True, help="direction_auc_trials.csv")
    parser.add_argument("--top20", required=False, default="", help="direction_auc_trials_top20.csv")
    parser.add_argument("--predictions", required=True, help="direction_auc_best_predictions.csv")
    parser.add_argument("--summary", required=True, help="direction_auc_summary.json")
    parser.add_argument("--output-dir", default="direction_auc_validation_output")
    parser.add_argument("--n-bins", type=int, default=10)

    args = parser.parse_args()

    paths = run_validation(
        trials_path=args.trials,
        top20_path=args.top20,
        predictions_path=args.predictions,
        summary_path=args.summary,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
    )

    report = load_json(paths["validation_report"])

    print("[OK] Direction AUC validation completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print()
    print(json.dumps(
        {
            "overall_judgment": report["final_decision"]["overall_judgment"],
            "best_trial_id": report["final_decision"]["best_trial_id"],
            "best_trial_candidate_decision": report["final_decision"]["best_trial_candidate_decision"],
            "direction_head_role": report["final_decision"]["structure_impact"]["direction_head_role"],
            "total_trials": report["final_decision"]["trial_counts"]["total_trials"],
            "holdout_candidate_count": report["final_decision"]["trial_counts"]["holdout_candidate_count"],
            "weak_candidate_count": report["final_decision"]["trial_counts"]["weak_candidate_count"],
            "best_prediction_auc_mean_by_year": report["final_decision"]["best_prediction_stability"].get("auc_mean"),
            "best_prediction_auc_above_0_5_year_rate": report["final_decision"]["best_prediction_stability"].get("year_auc_above_0_5_rate"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
