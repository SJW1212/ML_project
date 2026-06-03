# -*- coding: utf-8 -*-
"""
rolling_highvol_diagnostics.py

Rolling HighVol 모델/전략 진단 코드.

목적
----
rolling_leakage_free_highvol_backtest.py 실행 결과를 분석해
"왜 rolling fold에서 성능이 깨지는지"를 진단합니다.

입력 파일
---------
필수:
1. rolling_fold_metrics.csv
2. rolling_oos_predictions.csv
3. rolling_strategy_daily_returns.csv
4. rolling_strategy_summary.csv

선택:
5. rolling_thresholds.csv

출력 파일
---------
output_dir/
├─ diagnostic_summary.json
├─ fold_quality_diagnostics.csv
├─ fold_success_failure_groups.csv
├─ threshold_signal_diagnostics.csv
├─ strategy_comparison_diagnostics.csv
├─ strategy_relative_performance.csv
├─ drawdown_event_attribution.csv
├─ highvol_signal_attribution.csv
└─ recommendations.json

실행 예시
--------
python rolling_highvol_diagnostics.py ^
  --fold-metrics rolling_fold_metrics.csv ^
  --predictions rolling_oos_predictions.csv ^
  --strategy-daily rolling_strategy_daily_returns.csv ^
  --strategy-summary rolling_strategy_summary.csv ^
  --thresholds rolling_thresholds.csv ^
  --output-dir rolling_highvol_diagnostics_output

간단 실행:
python rolling_highvol_diagnostics.py --input-dir rolling_highvol_results_qqq_ief --output-dir rolling_highvol_diagnostics_output

의존성
------
pip install pandas numpy

주의
----
- 이 코드는 진단용입니다.
- 모델을 Stable로 채택하는 코드는 아닙니다.
- 진단 결과는 다음 라벨/threshold/배분 구조 개선의 근거로 사용합니다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 0. 유틸
# ============================================================

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
    return str(obj)


def save_json(path: str | Path, data: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=to_jsonable), encoding="utf-8")
    return path


def save_csv(path: str | Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_csv(path: str | Path, required: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def safe_divide(a: float, b: float, default: float = np.nan) -> float:
    if b == 0 or pd.isna(b):
        return default
    return float(a / b)


def parse_quantiles_from_columns(df: pd.DataFrame) -> List[float]:
    qs = []
    for c in df.columns:
        if c.startswith("signal_q"):
            try:
                qs.append(float(c.replace("signal_q", "")))
            except Exception:
                pass
    return sorted(set(qs))


def max_drawdown_from_returns(returns: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    ret = pd.Series(returns, dtype=float).fillna(0.0)
    if ret.empty:
        return np.nan, None, None

    curve = (1.0 + ret).cumprod()
    running_max = curve.cummax()
    dd = curve / running_max - 1.0

    end_idx = dd.idxmin()
    if pd.isna(end_idx):
        return np.nan, None, None

    start_idx = curve.loc[:end_idx].idxmax()
    return float(dd.loc[end_idx]), start_idx, end_idx


def annualized_metrics(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    ret = pd.Series(returns, dtype=float).fillna(0.0)
    if len(ret) < 2:
        return {}

    curve = (1.0 + ret).cumprod()
    total_return = float(curve.iloc[-1] - 1.0)
    years = len(ret) / periods_per_year
    cagr = float(curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(ret.std() * math.sqrt(periods_per_year))
    sharpe = float(ret.mean() / ret.std() * math.sqrt(periods_per_year)) if ret.std() > 0 else np.nan

    mdd, _, _ = max_drawdown_from_returns(ret)
    calmar = safe_divide(cagr, abs(mdd))

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
    }


# ============================================================
# 1. Fold 진단
# ============================================================

def diagnose_folds(
    fold_metrics: pd.DataFrame,
    low_positive_rate: float = 0.05,
    high_positive_rate: float = 0.95,
    high_ece: float = 0.15,
) -> pd.DataFrame:
    df = fold_metrics.copy()

    numeric_cols = [
        "positive_rate", "test_positive_rate", "pr_auc", "pr_gain", "pr_ratio",
        "roc_auc", "brier_skill", "ece", "precision_at_0_5", "recall_at_0_5",
        "prob_mean", "prob_std",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # test_positive_rate가 없으면 positive_rate 사용
    if "test_positive_rate" not in df.columns and "positive_rate" in df.columns:
        df["test_positive_rate"] = df["positive_rate"]

    if "pr_gain" not in df.columns and {"pr_auc", "test_positive_rate"}.issubset(df.columns):
        df["pr_gain"] = df["pr_auc"] - df["test_positive_rate"]

    df["flag_extreme_class_imbalance"] = (
        (df["test_positive_rate"] <= low_positive_rate)
        | (df["test_positive_rate"] >= high_positive_rate)
    )

    df["flag_pr_auc_below_base_rate"] = (
        (df["pr_auc"].notna())
        & (df["test_positive_rate"].notna())
        & (df["pr_auc"] <= df["test_positive_rate"])
    )

    df["flag_negative_brier_skill"] = df["brier_skill"] < 0

    if "probability_polarity" in df.columns:
        df["flag_inverse_polarity"] = df["probability_polarity"].astype(str).str.contains("inverse", case=False, na=False)
    else:
        df["flag_inverse_polarity"] = False

    df["flag_high_ece"] = df["ece"] > high_ece if "ece" in df.columns else False

    # fold health score: 높을수록 좋음
    df["fold_health_score"] = 0.0
    df["fold_health_score"] += df["pr_gain"].fillna(0.0).clip(-1, 1) * 2.0
    df["fold_health_score"] += df["brier_skill"].fillna(0.0).clip(-1, 1) * 1.0
    df["fold_health_score"] += (df["roc_auc"].fillna(0.5) - 0.5).clip(-0.5, 0.5) * 1.0
    df["fold_health_score"] -= df["ece"].fillna(0.0).clip(0, 1) * 0.5
    df["fold_health_score"] -= df["flag_extreme_class_imbalance"].astype(float) * 0.5
    df["fold_health_score"] -= df["flag_inverse_polarity"].astype(float) * 0.5

    df["fold_group"] = "neutral_or_mixed"
    df.loc[
        (df["pr_gain"] > 0)
        & (df["brier_skill"] > 0)
        & (~df["flag_inverse_polarity"])
        & (~df["flag_extreme_class_imbalance"]),
        "fold_group",
    ] = "success"

    df.loc[
        df["flag_extreme_class_imbalance"]
        | df["flag_inverse_polarity"]
        | df["flag_negative_brier_skill"]
        | df["flag_pr_auc_below_base_rate"],
        "fold_group",
    ] = "failure_or_unstable"

    ordered_cols = [
        "fold_id", "fold_group", "fold_health_score",
        "test_start_date", "test_end_date",
        "test_positive_rate", "pr_auc", "pr_gain", "pr_ratio",
        "roc_auc", "probability_polarity", "brier_skill", "ece",
        "flag_extreme_class_imbalance", "flag_pr_auc_below_base_rate",
        "flag_negative_brier_skill", "flag_inverse_polarity", "flag_high_ece",
    ]

    existing = [c for c in ordered_cols if c in df.columns]
    rest = [c for c in df.columns if c not in existing]
    return df[existing + rest].sort_values("fold_health_score", ascending=False).reset_index(drop=True)


def summarize_fold_groups(fold_diag: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["fold_group"]
    agg_dict = {
        "fold_id": "count",
        "fold_health_score": ["mean", "median", "min", "max"],
        "test_positive_rate": ["mean", "median"],
        "pr_auc": ["mean", "median"],
        "pr_gain": ["mean", "median"],
        "roc_auc": ["mean", "median"],
        "brier_skill": ["mean", "median"],
        "ece": ["mean", "median"],
        "flag_extreme_class_imbalance": "mean",
        "flag_inverse_polarity": "mean",
        "flag_negative_brier_skill": "mean",
    }

    existing_agg = {k: v for k, v in agg_dict.items() if k in fold_diag.columns}
    out = fold_diag.groupby(group_cols).agg(existing_agg)
    out.columns = ["_".join([str(x) for x in c if str(x) != ""]) for c in out.columns]
    out = out.reset_index().rename(columns={"fold_id_count": "fold_count"})
    return out


# ============================================================
# 2. Threshold / Signal 진단
# ============================================================

def diagnose_threshold_signals(
    predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    pred = predictions.copy()
    if "date" in pred.columns:
        pred["date"] = pd.to_datetime(pred["date"], errors="coerce")

    qs = parse_quantiles_from_columns(pred)
    rows = []

    for q in qs:
        signal_col = f"signal_q{q:.2f}"
        th_col = f"threshold_q{q:.2f}"

        if signal_col not in pred.columns:
            continue

        d = pred.copy()
        d[signal_col] = pd.to_numeric(d[signal_col], errors="coerce").fillna(0).astype(int)
        d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")

        signal = d[signal_col] == 1
        event = d["y_true"] == 1

        tp = int((signal & event).sum())
        fp = int((signal & ~event).sum())
        fn = int((~signal & event).sum())
        tn = int((~signal & ~event).sum())

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        false_alarm_rate = safe_divide(fp, tp + fp)
        signal_rate = safe_divide(int(signal.sum()), len(d))
        event_rate = safe_divide(int(event.sum()), len(d))

        rows.append({
            "threshold_quantile": q,
            "rows": int(len(d)),
            "event_rate": event_rate,
            "signal_rate": signal_rate,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "signal_precision": precision,
            "signal_recall": recall,
            "false_alarm_rate": false_alarm_rate,
            "miss_rate": safe_divide(fn, tp + fn),
            "avg_threshold": float(d[th_col].mean()) if th_col in d.columns else np.nan,
            "median_threshold": float(d[th_col].median()) if th_col in d.columns else np.nan,
            "avg_prob_signal_days": float(d.loc[signal, "prob_cal"].mean()) if "prob_cal" in d.columns and signal.any() else np.nan,
            "avg_prob_non_signal_days": float(d.loc[~signal, "prob_cal"].mean()) if "prob_cal" in d.columns and (~signal).any() else np.nan,
        })

    signal_df = pd.DataFrame(rows)

    if not thresholds.empty:
        th = thresholds.copy()
        numeric_cols = ["threshold_quantile", "threshold", "cal_signal_rate", "test_signal_rate", "test_positive_rate"]
        for c in numeric_cols:
            if c in th.columns:
                th[c] = pd.to_numeric(th[c], errors="coerce")

        th_summary = th.groupby("threshold_quantile").agg(
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
            cal_signal_rate_mean=("cal_signal_rate", "mean"),
            test_signal_rate_mean=("test_signal_rate", "mean"),
            test_positive_rate_mean=("test_positive_rate", "mean"),
        ).reset_index()

        signal_df = signal_df.merge(th_summary, on="threshold_quantile", how="left")

    return signal_df.sort_values("threshold_quantile").reset_index(drop=True)


# ============================================================
# 3. 전략 비교 / 상대 성과
# ============================================================

def diagnose_strategy_summary(strategy_summary: pd.DataFrame) -> pd.DataFrame:
    df = strategy_summary.copy()

    numeric_cols = [
        "threshold_quantile", "cagr", "mdd", "calmar", "sharpe",
        "volatility", "turnover_total", "transaction_cost_total",
        "avg_equity_weight", "avg_cash_weight", "total_return",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["rank_calmar"] = df["calmar"].rank(ascending=False, method="min") if "calmar" in df.columns else np.nan
    df["rank_cagr"] = df["cagr"].rank(ascending=False, method="min") if "cagr" in df.columns else np.nan
    df["rank_mdd"] = df["mdd"].rank(ascending=False, method="min") if "mdd" in df.columns else np.nan

    if "strategy" in df.columns:
        bh = df[df["strategy"] == "buy_hold"]
        if not bh.empty:
            bh_row = bh.iloc[0]
            for metric in ["cagr", "mdd", "calmar", "sharpe", "volatility", "total_return"]:
                if metric in df.columns:
                    df[f"{metric}_diff_vs_buy_hold"] = df[metric] - bh_row[metric]

        cn = df[df["strategy"] == "constant_normal"]
        if not cn.empty:
            cn_row = cn.iloc[0]
            for metric in ["cagr", "mdd", "calmar", "sharpe", "volatility", "total_return"]:
                if metric in df.columns:
                    df[f"{metric}_diff_vs_constant_normal"] = df[metric] - cn_row[metric]

    return df.sort_values("rank_calmar", na_position="last").reset_index(drop=True)


def diagnose_relative_daily_performance(strategy_daily: pd.DataFrame) -> pd.DataFrame:
    df = strategy_daily.copy()
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["strategy_ret"] = pd.to_numeric(df["strategy_ret"], errors="coerce").fillna(0.0)

    # 전략별 일별 수익률 pivot
    key_cols = ["strategy", "threshold_quantile"]
    df["strategy_key"] = df["strategy"].astype(str)
    if "threshold_quantile" in df.columns:
        df["strategy_key"] = np.where(
            df["threshold_quantile"].notna(),
            df["strategy_key"] + "_q" + df["threshold_quantile"].astype(float).round(2).astype(str),
            df["strategy_key"],
        )

    pivot = df.pivot_table(index="date", columns="strategy_key", values="strategy_ret", aggfunc="last").fillna(0.0)

    if "buy_hold" not in pivot.columns:
        return pd.DataFrame()

    rows = []
    for col in pivot.columns:
        if col == "buy_hold":
            continue

        diff = pivot[col] - pivot["buy_hold"]
        rows.append({
            "strategy_key": col,
            "days": int(len(diff)),
            "avg_daily_excess_ret_vs_buy_hold": float(diff.mean()),
            "median_daily_excess_ret_vs_buy_hold": float(diff.median()),
            "positive_excess_day_rate": float((diff > 0).mean()),
            "cumulative_excess_return_sum": float(diff.sum()),
            "worst_daily_excess": float(diff.min()),
            "best_daily_excess": float(diff.max()),
        })

    return pd.DataFrame(rows).sort_values("avg_daily_excess_ret_vs_buy_hold", ascending=False).reset_index(drop=True)


# ============================================================
# 4. Drawdown / Signal attribution
# ============================================================

def compute_strategy_drawdowns(strategy_daily: pd.DataFrame) -> pd.DataFrame:
    df = strategy_daily.copy()
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["strategy_ret"] = pd.to_numeric(df["strategy_ret"], errors="coerce").fillna(0.0)

    rows = []
    for key, g in df.groupby(["strategy", "threshold_quantile"], dropna=False):
        strategy, q = key
        g = g.sort_values("date").copy()
        curve = (1.0 + g["strategy_ret"]).cumprod()
        running_max = curve.cummax()
        dd = curve / running_max - 1.0

        worst_idx = dd.idxmin()
        start_idx = curve.loc[:worst_idx].idxmax() if len(curve.loc[:worst_idx]) else worst_idx

        rows.append({
            "strategy": strategy,
            "threshold_quantile": q,
            "worst_drawdown": float(dd.loc[worst_idx]),
            "drawdown_start_date": str(g.loc[start_idx, "date"].date()) if pd.notna(g.loc[start_idx, "date"]) else None,
            "drawdown_end_date": str(g.loc[worst_idx, "date"].date()) if pd.notna(g.loc[worst_idx, "date"]) else None,
            "drawdown_days": int(g.index.get_loc(worst_idx) - g.index.get_loc(start_idx)) if start_idx in g.index and worst_idx in g.index else np.nan,
        })

    return pd.DataFrame(rows).sort_values("worst_drawdown").reset_index(drop=True)


def highvol_signal_attribution(
    strategy_daily: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if strategy_daily.empty or predictions.empty:
        return pd.DataFrame()

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")

    qs = parse_quantiles_from_columns(pred)
    rows = []

    # buy_hold 수익률 기준
    sd = strategy_daily.copy()
    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd["strategy_ret"] = pd.to_numeric(sd["strategy_ret"], errors="coerce").fillna(0.0)

    bh = sd[sd["strategy"] == "buy_hold"][["date", "strategy_ret"]].rename(columns={"strategy_ret": "buy_hold_ret"})
    equity_ret_map = bh.set_index("date")["buy_hold_ret"]

    pred["next_day_buy_hold_ret"] = pred["date"].map(equity_ret_map.shift(-1))
    pred["same_day_buy_hold_ret"] = pred["date"].map(equity_ret_map)

    for q in qs:
        signal_col = f"signal_q{q:.2f}"
        if signal_col not in pred.columns:
            continue

        d = pred.copy()
        d[signal_col] = pd.to_numeric(d[signal_col], errors="coerce").fillna(0).astype(int)

        signal = d[signal_col] == 1
        event = d["y_true"] == 1

        rows.append({
            "threshold_quantile": q,
            "signal_days": int(signal.sum()),
            "non_signal_days": int((~signal).sum()),
            "signal_rate": float(signal.mean()),
            "event_rate_on_signal_days": float(event[signal].mean()) if signal.any() else np.nan,
            "event_rate_on_non_signal_days": float(event[~signal].mean()) if (~signal).any() else np.nan,
            "avg_same_day_buy_hold_ret_signal": float(d.loc[signal, "same_day_buy_hold_ret"].mean()) if signal.any() else np.nan,
            "avg_same_day_buy_hold_ret_non_signal": float(d.loc[~signal, "same_day_buy_hold_ret"].mean()) if (~signal).any() else np.nan,
            "avg_next_day_buy_hold_ret_signal": float(d.loc[signal, "next_day_buy_hold_ret"].mean()) if signal.any() else np.nan,
            "avg_next_day_buy_hold_ret_non_signal": float(d.loc[~signal, "next_day_buy_hold_ret"].mean()) if (~signal).any() else np.nan,
            "worst_next_day_ret_signal": float(d.loc[signal, "next_day_buy_hold_ret"].min()) if signal.any() else np.nan,
            "worst_next_day_ret_non_signal": float(d.loc[~signal, "next_day_buy_hold_ret"].min()) if (~signal).any() else np.nan,
        })

    return pd.DataFrame(rows).sort_values("threshold_quantile").reset_index(drop=True)


def drawdown_event_attribution(
    strategy_daily: pd.DataFrame,
    predictions: pd.DataFrame,
    drawdown_threshold: float = -0.10,
) -> pd.DataFrame:
    """
    Buy & Hold의 drawdown 구간에서 HighVol signal이 얼마나 켜졌는지 확인.
    """
    if strategy_daily.empty or predictions.empty:
        return pd.DataFrame()

    sd = strategy_daily.copy()
    sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
    sd["strategy_ret"] = pd.to_numeric(sd["strategy_ret"], errors="coerce").fillna(0.0)

    bh = sd[sd["strategy"] == "buy_hold"].sort_values("date").copy()
    if bh.empty:
        return pd.DataFrame()

    curve = (1.0 + bh["strategy_ret"]).cumprod()
    dd = curve / curve.cummax() - 1.0
    bh["buy_hold_drawdown"] = dd.values
    bh["drawdown_event"] = bh["buy_hold_drawdown"] <= drawdown_threshold

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    qs = parse_quantiles_from_columns(pred)

    merged = bh[["date", "buy_hold_drawdown", "drawdown_event"]].merge(pred, on="date", how="left")

    rows = []
    event_mask = merged["drawdown_event"].fillna(False)
    for q in qs:
        signal_col = f"signal_q{q:.2f}"
        if signal_col not in merged.columns:
            continue

        signal = pd.to_numeric(merged[signal_col], errors="coerce").fillna(0).astype(int) == 1

        rows.append({
            "threshold_quantile": q,
            "drawdown_threshold": drawdown_threshold,
            "drawdown_event_days": int(event_mask.sum()),
            "signal_days_in_drawdown": int((signal & event_mask).sum()),
            "signal_coverage_in_drawdown": safe_divide(int((signal & event_mask).sum()), int(event_mask.sum())),
            "signal_days_outside_drawdown": int((signal & ~event_mask).sum()),
            "false_alarm_like_signal_rate": safe_divide(int((signal & ~event_mask).sum()), int((~event_mask).sum())),
            "avg_drawdown_signal_days": float(merged.loc[signal, "buy_hold_drawdown"].mean()) if signal.any() else np.nan,
            "avg_drawdown_non_signal_days": float(merged.loc[~signal, "buy_hold_drawdown"].mean()) if (~signal).any() else np.nan,
        })

    return pd.DataFrame(rows).sort_values("threshold_quantile").reset_index(drop=True)


# ============================================================
# 5. 추천안 생성
# ============================================================

def build_recommendations(
    fold_diag: pd.DataFrame,
    signal_diag: pd.DataFrame,
    strategy_diag: pd.DataFrame,
    drawdown_attr: pd.DataFrame,
) -> Dict[str, object]:
    recs: List[str] = []
    warnings: List[str] = []
    decisions: List[str] = []

    if not fold_diag.empty:
        positive_brier_rate = float((fold_diag["brier_skill"] > 0).mean()) if "brier_skill" in fold_diag.columns else np.nan
        normal_polarity_rate = float((~fold_diag["flag_inverse_polarity"]).mean()) if "flag_inverse_polarity" in fold_diag.columns else np.nan
        median_pr_auc = float(fold_diag["pr_auc"].median()) if "pr_auc" in fold_diag.columns else np.nan

        if positive_brier_rate < 0.4:
            warnings.append("Brier skill 양수 fold 비율이 낮습니다. 확률값을 직접 allocation 강도로 사용하지 마세요.")
            recs.append("확률값은 calibrated probability 자체보다 rolling quantile signal로만 사용하세요.")

        if normal_polarity_rate < 0.6:
            warnings.append("normal polarity fold 비율이 낮습니다. 일부 fold에서 신호 방향이 불안정합니다.")
            recs.append("fold별 성공/실패 regime을 분리하고, volatility expansion label을 추가 실험하세요.")

        if np.isfinite(median_pr_auc) and median_pr_auc < 0.10:
            warnings.append("median PR-AUC가 낮습니다. 평균 성능은 일부 fold에 의해 왜곡되었을 가능성이 큽니다.")
            recs.append("H20 단일 head 대신 H10/H20 ensemble 또는 persistence 조건을 테스트하세요.")

    best_strategy = None
    if not strategy_diag.empty and "rank_calmar" in strategy_diag.columns:
        best = strategy_diag.sort_values("rank_calmar").head(1)
        if not best.empty:
            best_strategy = best.iloc[0].to_dict()

    if best_strategy:
        if best_strategy.get("strategy") == "highvol_only":
            decisions.append("HighVol only는 후보로 유지합니다.")
            recs.append("q=0.75와 q=0.80 중 fold 안정성이 더 좋은 쪽을 별도 검증하세요.")
        else:
            warnings.append("HighVol only가 Calmar 1위가 아닙니다. 현재 구조의 실전 우위가 약합니다.")

    if not signal_diag.empty:
        # q별 signal precision/recall 확인
        if "signal_precision" in signal_diag.columns:
            max_precision = float(signal_diag["signal_precision"].max())
            if max_precision < 0.40:
                warnings.append("HighVol signal precision이 낮습니다. false alarm이 많을 수 있습니다.")
                recs.append("2~3일 persistence 조건 또는 threshold 상향을 테스트하세요.")

        if "signal_recall" in signal_diag.columns:
            max_recall = float(signal_diag["signal_recall"].max())
            if max_recall < 0.40:
                warnings.append("HighVol event recall이 낮습니다. 주요 고변동 구간을 놓칠 수 있습니다.")
                recs.append("H10/H20 ensemble 또는 high_vol label 정의 변경을 검토하세요.")

    decisions.append("Stable 채택은 아직 보류합니다.")
    recs.append("다음 개선 실험은 label 개선과 signal persistence 조건을 우선하세요.")

    return {
        "decisions": decisions,
        "warnings": warnings,
        "recommendations": recs,
    }


# ============================================================
# 6. Main runner
# ============================================================

def run_diagnostics(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    strategy_daily: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    output_dir: str | Path,
    low_positive_rate: float = 0.05,
    high_positive_rate: float = 0.95,
    high_ece: float = 0.15,
    drawdown_threshold: float = -0.10,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_diag = diagnose_folds(
        fold_metrics,
        low_positive_rate=low_positive_rate,
        high_positive_rate=high_positive_rate,
        high_ece=high_ece,
    )
    fold_groups = summarize_fold_groups(fold_diag)
    signal_diag = diagnose_threshold_signals(predictions, thresholds)
    strategy_diag = diagnose_strategy_summary(strategy_summary)
    relative_daily = diagnose_relative_daily_performance(strategy_daily)
    drawdowns = compute_strategy_drawdowns(strategy_daily)
    signal_attr = highvol_signal_attribution(strategy_daily, predictions)
    dd_attr = drawdown_event_attribution(strategy_daily, predictions, drawdown_threshold=drawdown_threshold)

    recs = build_recommendations(fold_diag, signal_diag, strategy_diag, dd_attr)

    outputs = {
        "fold_quality": save_csv(output_dir / "fold_quality_diagnostics.csv", fold_diag),
        "fold_groups": save_csv(output_dir / "fold_success_failure_groups.csv", fold_groups),
        "threshold_signal": save_csv(output_dir / "threshold_signal_diagnostics.csv", signal_diag),
        "strategy_comparison": save_csv(output_dir / "strategy_comparison_diagnostics.csv", strategy_diag),
        "strategy_relative": save_csv(output_dir / "strategy_relative_performance.csv", relative_daily),
        "drawdown_events": save_csv(output_dir / "strategy_drawdown_diagnostics.csv", drawdowns),
        "highvol_signal_attribution": save_csv(output_dir / "highvol_signal_attribution.csv", signal_attr),
        "drawdown_event_attribution": save_csv(output_dir / "drawdown_event_attribution.csv", dd_attr),
        "recommendations": save_json(output_dir / "recommendations.json", recs),
    }

    fold_summary = {
        "fold_count": int(len(fold_diag)),
        "success_fold_count": int((fold_diag["fold_group"] == "success").sum()) if "fold_group" in fold_diag.columns else None,
        "failure_or_unstable_fold_count": int((fold_diag["fold_group"] == "failure_or_unstable").sum()) if "fold_group" in fold_diag.columns else None,
        "extreme_class_imbalance_rate": float(fold_diag["flag_extreme_class_imbalance"].mean()) if "flag_extreme_class_imbalance" in fold_diag.columns else None,
        "inverse_polarity_rate": float(fold_diag["flag_inverse_polarity"].mean()) if "flag_inverse_polarity" in fold_diag.columns else None,
        "negative_brier_skill_rate": float(fold_diag["flag_negative_brier_skill"].mean()) if "flag_negative_brier_skill" in fold_diag.columns else None,
        "median_pr_auc": float(fold_diag["pr_auc"].median()) if "pr_auc" in fold_diag.columns else None,
        "mean_pr_auc": float(fold_diag["pr_auc"].mean()) if "pr_auc" in fold_diag.columns else None,
        "median_brier_skill": float(fold_diag["brier_skill"].median()) if "brier_skill" in fold_diag.columns else None,
        "mean_brier_skill": float(fold_diag["brier_skill"].mean()) if "brier_skill" in fold_diag.columns else None,
    }

    best_strategy = strategy_diag.sort_values("rank_calmar").head(1).to_dict("records")[0] if not strategy_diag.empty and "rank_calmar" in strategy_diag.columns else None

    summary = {
        "diagnostic_type": "rolling_highvol_diagnostics",
        "fold_summary": fold_summary,
        "best_strategy_by_calmar": best_strategy,
        "threshold_signal_summary": signal_diag.to_dict("records"),
        "recommendations": recs,
        "output_files": {k: str(v) for k, v in outputs.items()},
    }
    outputs["summary"] = save_json(output_dir / "diagnostic_summary.json", summary)

    return outputs


def resolve_input_paths(args) -> Dict[str, Path]:
    if args.input_dir:
        base = Path(args.input_dir)
        return {
            "fold_metrics": base / "rolling_fold_metrics.csv",
            "predictions": base / "rolling_oos_predictions.csv",
            "strategy_daily": base / "rolling_strategy_daily_returns.csv",
            "strategy_summary": base / "rolling_strategy_summary.csv",
            "thresholds": base / "rolling_thresholds.csv",
        }

    return {
        "fold_metrics": Path(args.fold_metrics),
        "predictions": Path(args.predictions),
        "strategy_daily": Path(args.strategy_daily),
        "strategy_summary": Path(args.strategy_summary),
        "thresholds": Path(args.thresholds) if args.thresholds else Path("__missing_thresholds__.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="", help="directory containing rolling output files")
    parser.add_argument("--fold-metrics", default="rolling_fold_metrics.csv")
    parser.add_argument("--predictions", default="rolling_oos_predictions.csv")
    parser.add_argument("--strategy-daily", default="rolling_strategy_daily_returns.csv")
    parser.add_argument("--strategy-summary", default="rolling_strategy_summary.csv")
    parser.add_argument("--thresholds", default="rolling_thresholds.csv")
    parser.add_argument("--output-dir", default="rolling_highvol_diagnostics_output")

    parser.add_argument("--low-positive-rate", type=float, default=0.05)
    parser.add_argument("--high-positive-rate", type=float, default=0.95)
    parser.add_argument("--high-ece", type=float, default=0.15)
    parser.add_argument("--drawdown-threshold", type=float, default=-0.10)

    args = parser.parse_args()
    paths = resolve_input_paths(args)

    fold_metrics = load_csv(paths["fold_metrics"], required=True)
    predictions = load_csv(paths["predictions"], required=True)
    strategy_daily = load_csv(paths["strategy_daily"], required=True)
    strategy_summary = load_csv(paths["strategy_summary"], required=True)
    thresholds = load_csv(paths["thresholds"], required=False)

    outputs = run_diagnostics(
        fold_metrics=fold_metrics,
        predictions=predictions,
        strategy_daily=strategy_daily,
        strategy_summary=strategy_summary,
        thresholds=thresholds,
        output_dir=args.output_dir,
        low_positive_rate=args.low_positive_rate,
        high_positive_rate=args.high_positive_rate,
        high_ece=args.high_ece,
        drawdown_threshold=args.drawdown_threshold,
    )

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    fold_summary = summary["fold_summary"]
    best = summary.get("best_strategy_by_calmar") or {}

    print("[OK] Rolling HighVol diagnostics completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print(json.dumps(
        {
            "fold_count": fold_summary.get("fold_count"),
            "success_fold_count": fold_summary.get("success_fold_count"),
            "failure_or_unstable_fold_count": fold_summary.get("failure_or_unstable_fold_count"),
            "inverse_polarity_rate": fold_summary.get("inverse_polarity_rate"),
            "negative_brier_skill_rate": fold_summary.get("negative_brier_skill_rate"),
            "median_pr_auc": fold_summary.get("median_pr_auc"),
            "mean_pr_auc": fold_summary.get("mean_pr_auc"),
            "best_strategy": best.get("strategy"),
            "best_threshold_quantile": best.get("threshold_quantile"),
            "best_calmar": best.get("calmar"),
            "decision": "stable_rejected_diagnostic_required",
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
