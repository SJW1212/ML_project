# -*- coding: utf-8 -*-
"""
dual_highvol_candidate_validator.py

Dual-HighVol Hybrid 최종 후보 검증 코드.

목적
----
dual_highvol_hybrid_sweep.py 결과에서 best candidate가 처음으로 economic gate를 통과했으므로,
바로 Stable로 승격하지 않고 아래 항목을 추가 검증합니다.

검증 항목
---------
1. 비용 민감도
   - 0, 5, 10, 20, 30 bps에서 성과 재계산

2. 연도별 안정성
   - 연도별 CAGR / MDD / Calmar / Sharpe
   - Buy & Hold 대비 차이

3. Drawdown attribution
   - Buy & Hold drawdown 구간에서 후보 전략이 얼마나 방어했는지
   - worst drawdown 구간 비교

4. Signal attribution
   - executed signal 구간과 non-signal 구간의 수익률 차이
   - 신호 발생률 / turnover / 평균 비중 확인

5. Parameter neighborhood stability
   - 같은 hybrid mode 주변 defensive equity weight / persistence / defense asset 조합이 같이 좋은지 확인

6. Head stability gate
   - highvol_h20과 highvol_expansion head 안정성 확인
   - h20은 전략성, expansion은 confirmation 안정성 담당으로 분리 판단

입력 파일
---------
--input-dir dual_highvol_hybrid_results_qqq_ief

필수 파일:
- dual_highvol_summary.json
- dual_highvol_strategy_summary.csv
- dual_highvol_strategy_daily_returns.csv
- dual_highvol_head_summary.csv
- dual_highvol_fold_metrics.csv
- dual_highvol_oos_predictions.csv

출력 파일
---------
output_dir/
├─ candidate_validation_summary.json
├─ candidate_cost_sensitivity.csv
├─ candidate_yearly_metrics.csv
├─ candidate_drawdown_attribution.csv
├─ candidate_signal_attribution.csv
├─ candidate_neighborhood.csv
├─ candidate_head_gate.csv
└─ candidate_decision.csv

실행 예시
--------
python dual_highvol_candidate_validator.py ^
  --input-dir dual_highvol_hybrid_results_qqq_ief ^
  --output-dir dual_highvol_candidate_validation

직접 파일 지정:
python dual_highvol_candidate_validator.py ^
  --summary-json dual_highvol_summary.json ^
  --strategy-summary dual_highvol_strategy_summary.csv ^
  --strategy-daily dual_highvol_strategy_daily_returns.csv ^
  --head-summary dual_highvol_head_summary.csv ^
  --fold-metrics dual_highvol_fold_metrics.csv ^
  --predictions dual_highvol_oos_predictions.csv ^
  --output-dir dual_highvol_candidate_validation
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 0. Utils
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


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_divide(a: float, b: float, default=np.nan) -> float:
    if b == 0 or pd.isna(b):
        return default
    return float(a / b)


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_json(path: str | Path) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# 1. Metrics
# ============================================================

def performance_metrics(returns: pd.Series, dates: Optional[pd.Series] = None, periods_per_year: int = 252) -> Dict[str, float]:
    ret = pd.Series(returns, dtype=float).fillna(0.0)
    if len(ret) < 2:
        return {}

    curve = (1.0 + ret).cumprod()
    total_return = float(curve.iloc[-1] - 1.0)

    if dates is not None:
        d = pd.to_datetime(dates)
        days = max((d.max() - d.min()).days, 1)
        years = days / 365.25
    else:
        years = len(ret) / periods_per_year

    cagr = float(curve.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and curve.iloc[-1] > 0 else np.nan
    vol = float(ret.std() * math.sqrt(periods_per_year))
    sharpe = float(ret.mean() / ret.std() * math.sqrt(periods_per_year)) if ret.std() > 0 else np.nan

    dd = curve / curve.cummax() - 1.0
    mdd = float(dd.min())
    calmar = safe_divide(cagr, abs(mdd))

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
    }


def drawdown_series(returns: pd.Series) -> pd.Series:
    ret = pd.Series(returns, dtype=float).fillna(0.0)
    curve = (1.0 + ret).cumprod()
    return curve / curve.cummax() - 1.0


def recompute_strategy_return(df: pd.DataFrame, cost_bps: float) -> pd.Series:
    gross = (
        df["equity_weight"].astype(float) * df["equity_ret"].astype(float)
        + df["bond_weight"].astype(float) * df["bond_ret"].astype(float)
        + df["cash_weight"].astype(float) * df["cash_ret"].astype(float)
    )
    cost = df["turnover"].astype(float) * (cost_bps / 10000.0)
    return gross - cost


# ============================================================
# 2. Candidate Selection
# ============================================================

def get_best_candidate(summary_json: Dict, strategy_summary: pd.DataFrame) -> Dict:
    best = summary_json.get("best_candidate")
    if best:
        return best

    s = strategy_summary.copy()
    if "candidate_score" in s.columns:
        s = s.sort_values("candidate_score", ascending=False)
    elif "calmar" in s.columns:
        s = s.sort_values("calmar", ascending=False)

    return s.head(1).to_dict("records")[0]


def filter_candidate_daily(strategy_daily: pd.DataFrame, candidate: Dict) -> pd.DataFrame:
    d = strategy_daily.copy()

    mask = d["strategy"].astype(str).eq(str(candidate.get("strategy", "dual_highvol_hybrid")))

    for col in ["hybrid_mode", "persistence_mode", "defense_asset", "riskoff_mode"]:
        if col in d.columns and col in candidate:
            mask &= d[col].astype(str).eq(str(candidate[col]))

    if "defensive_equity_weight" in d.columns and "defensive_equity_weight" in candidate:
        target = safe_float(candidate["defensive_equity_weight"])
        mask &= np.isclose(pd.to_numeric(d["defensive_equity_weight"], errors="coerce"), target, atol=1e-9)

    out = d[mask].sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"candidate daily rows not found for candidate={candidate}")
    return out


def filter_benchmark_daily(strategy_daily: pd.DataFrame, strategy: str = "buy_hold") -> pd.DataFrame:
    d = strategy_daily[strategy_daily["strategy"].astype(str).eq(strategy)].copy()
    d = d.sort_values("date").reset_index(drop=True)
    if d.empty:
        raise ValueError(f"benchmark daily rows not found: {strategy}")
    return d


# ============================================================
# 3. Validation tables
# ============================================================

def cost_sensitivity(candidate_daily: pd.DataFrame, buyhold_daily: pd.DataFrame, cost_bps_values: List[float]) -> pd.DataFrame:
    rows = []

    bh_ret = recompute_strategy_return(buyhold_daily, 0.0)
    bh_metrics = performance_metrics(bh_ret, buyhold_daily["date"])

    for bps in cost_bps_values:
        cand_ret = recompute_strategy_return(candidate_daily, bps)
        cand_metrics = performance_metrics(cand_ret, candidate_daily["date"])

        row = {
            "cost_bps": bps,
            **{f"candidate_{k}": v for k, v in cand_metrics.items()},
            **{f"buy_hold_{k}": v for k, v in bh_metrics.items()},
        }

        for k in ["total_return", "cagr", "mdd", "calmar", "sharpe", "volatility"]:
            row[f"{k}_diff_vs_buy_hold"] = row.get(f"candidate_{k}", np.nan) - row.get(f"buy_hold_{k}", np.nan)

        row["economic_gate"] = (
            row["calmar_diff_vs_buy_hold"] > 0.03
            and row["mdd_diff_vs_buy_hold"] > 0.03
            and row["cagr_diff_vs_buy_hold"] > -0.02
        )
        rows.append(row)

    return pd.DataFrame(rows)


def yearly_metrics(candidate_daily: pd.DataFrame, buyhold_daily: pd.DataFrame) -> pd.DataFrame:
    c = candidate_daily[["date"]].copy()
    c["candidate_ret"] = recompute_strategy_return(candidate_daily, 10.0)
    b = buyhold_daily[["date"]].copy()
    b["buy_hold_ret"] = recompute_strategy_return(buyhold_daily, 0.0)

    df = c.merge(b, on="date", how="inner")
    df["year"] = pd.to_datetime(df["date"]).dt.year

    rows = []
    for year, g in df.groupby("year"):
        cm = performance_metrics(g["candidate_ret"], g["date"])
        bm = performance_metrics(g["buy_hold_ret"], g["date"])

        row = {"year": int(year), "days": int(len(g))}
        row.update({f"candidate_{k}": v for k, v in cm.items()})
        row.update({f"buy_hold_{k}": v for k, v in bm.items()})

        for k in ["total_return", "cagr", "mdd", "calmar", "sharpe", "volatility"]:
            row[f"{k}_diff_vs_buy_hold"] = row.get(f"candidate_{k}", np.nan) - row.get(f"buy_hold_{k}", np.nan)

        row["mdd_improved"] = row["mdd_diff_vs_buy_hold"] > 0
        row["calmar_improved"] = row["calmar_diff_vs_buy_hold"] > 0
        row["cagr_not_worse_than_minus_2pct"] = row["cagr_diff_vs_buy_hold"] > -0.02
        rows.append(row)

    return pd.DataFrame(rows)


def drawdown_attribution(candidate_daily: pd.DataFrame, buyhold_daily: pd.DataFrame, drawdown_thresholds: List[float]) -> pd.DataFrame:
    c = candidate_daily[["date", "executed_signal", "equity_weight", "bond_weight", "cash_weight"]].copy()
    c["candidate_ret"] = recompute_strategy_return(candidate_daily, 10.0)
    c["candidate_dd"] = drawdown_series(c["candidate_ret"]).to_numpy()

    b = buyhold_daily[["date"]].copy()
    b["buy_hold_ret"] = recompute_strategy_return(buyhold_daily, 0.0)
    b["buy_hold_dd"] = drawdown_series(b["buy_hold_ret"]).to_numpy()

    df = c.merge(b, on="date", how="inner")

    rows = []
    for th in drawdown_thresholds:
        event = df["buy_hold_dd"] <= th
        signal = df["executed_signal"].fillna(0).astype(int) == 1

        rows.append({
            "buy_hold_drawdown_threshold": th,
            "event_days": int(event.sum()),
            "event_day_rate": float(event.mean()),
            "signal_days_in_event": int((signal & event).sum()),
            "signal_coverage_in_event": safe_divide(int((signal & event).sum()), int(event.sum())),
            "signal_days_outside_event": int((signal & ~event).sum()),
            "false_alarm_like_rate": safe_divide(int((signal & ~event).sum()), int((~event).sum())),
            "avg_candidate_ret_event_days": float(df.loc[event, "candidate_ret"].mean()) if event.any() else np.nan,
            "avg_buy_hold_ret_event_days": float(df.loc[event, "buy_hold_ret"].mean()) if event.any() else np.nan,
            "candidate_ret_minus_buy_hold_event_days": float((df.loc[event, "candidate_ret"] - df.loc[event, "buy_hold_ret"]).mean()) if event.any() else np.nan,
            "avg_candidate_dd_event_days": float(df.loc[event, "candidate_dd"].mean()) if event.any() else np.nan,
            "avg_buy_hold_dd_event_days": float(df.loc[event, "buy_hold_dd"].mean()) if event.any() else np.nan,
        })

    return pd.DataFrame(rows)


def signal_attribution(candidate_daily: pd.DataFrame) -> pd.DataFrame:
    d = candidate_daily.copy()
    d["candidate_ret"] = recompute_strategy_return(d, 10.0)
    d["gross_ret"] = (
        d["equity_weight"].astype(float) * d["equity_ret"].astype(float)
        + d["bond_weight"].astype(float) * d["bond_ret"].astype(float)
        + d["cash_weight"].astype(float) * d["cash_ret"].astype(float)
    )
    d["signal"] = d["executed_signal"].fillna(0).astype(int)

    rows = []
    for signal_value, g in d.groupby("signal"):
        rows.append({
            "executed_signal": int(signal_value),
            "days": int(len(g)),
            "day_rate": float(len(g) / len(d)),
            "avg_strategy_ret": float(g["candidate_ret"].mean()),
            "median_strategy_ret": float(g["candidate_ret"].median()),
            "avg_equity_ret": float(g["equity_ret"].mean()),
            "median_equity_ret": float(g["equity_ret"].median()),
            "avg_equity_weight": float(g["equity_weight"].mean()),
            "avg_bond_weight": float(g["bond_weight"].mean()),
            "avg_cash_weight": float(g["cash_weight"].mean()),
            "avg_turnover": float(g["turnover"].mean()),
            "total_turnover": float(g["turnover"].sum()),
            "worst_strategy_day": float(g["candidate_ret"].min()),
            "best_strategy_day": float(g["candidate_ret"].max()),
            "worst_equity_day": float(g["equity_ret"].min()),
            "best_equity_day": float(g["equity_ret"].max()),
        })

    return pd.DataFrame(rows)


def neighborhood_table(strategy_summary: pd.DataFrame, candidate: Dict) -> pd.DataFrame:
    s = strategy_summary.copy()

    # benchmarks 제외
    s = s[s["strategy"].astype(str).eq("dual_highvol_hybrid")].copy()

    # 같은 hybrid mode 중심으로 확인
    if "hybrid_mode" in candidate:
        s = s[s["hybrid_mode"].astype(str).eq(str(candidate["hybrid_mode"]))]

    # 같은 riskoff_mode 중심
    if "riskoff_mode" in candidate:
        s = s[s["riskoff_mode"].astype(str).eq(str(candidate["riskoff_mode"]))]

    target_w = safe_float(candidate.get("defensive_equity_weight"))
    if "defensive_equity_weight" in s.columns and np.isfinite(target_w):
        s["weight_distance"] = (pd.to_numeric(s["defensive_equity_weight"], errors="coerce") - target_w).abs()
    else:
        s["weight_distance"] = np.nan

    if "candidate_score" not in s.columns:
        s["candidate_score"] = (
            s.get("calmar_diff_vs_buy_hold", 0).fillna(0) * 2
            + s.get("mdd_diff_vs_buy_hold", 0).fillna(0) * 1.5
            + s.get("cagr_diff_vs_buy_hold", 0).fillna(0)
        )

    cols = [
        "strategy", "hybrid_mode", "persistence_mode", "defensive_equity_weight",
        "defense_asset", "riskoff_mode", "raw_signal_rate", "executed_signal_rate",
        "avg_equity_weight", "avg_bond_weight", "avg_cash_weight",
        "turnover_total", "transaction_cost_total",
        "cagr", "mdd", "calmar", "sharpe", "volatility",
        "cagr_diff_vs_buy_hold", "mdd_diff_vs_buy_hold", "calmar_diff_vs_buy_hold",
        "candidate_score", "stable_economic_gate", "weight_distance",
    ]
    existing = [c for c in cols if c in s.columns]

    return (
        s[existing]
        .sort_values(["stable_economic_gate", "candidate_score", "calmar"], ascending=[False, False, False])
        .head(50)
        .reset_index(drop=True)
    )


def head_gate(head_summary: pd.DataFrame) -> pd.DataFrame:
    h = head_summary.copy()

    rows = []
    for _, r in h.iterrows():
        head = str(r.get("head"))

        if head == "direction":
            pass_gate = safe_float(r.get("mean_macro_f1"), 0) >= 0.40 and safe_float(r.get("mean_balanced_accuracy"), 0) >= 0.40
            role = "auxiliary_only"
            reason = "Direction entropy and macro F1 are not strong enough for allocation trigger."
        elif head == "highvol_h20":
            pass_gate = safe_float(r.get("normal_polarity_rate"), 0) >= 0.50 and safe_float(r.get("positive_brier_skill_rate"), 0) >= 0.20
            role = "primary_strategy_signal_but_requires_confirmation"
            reason = "h20 has strong strategy utility but weak fold stability."
        elif head == "highvol_expansion":
            pass_gate = safe_float(r.get("normal_polarity_rate"), 0) >= 0.50 and safe_float(r.get("positive_brier_skill_rate"), 0) >= 0.15
            role = "confirmation_signal"
            reason = "expansion improves polarity and median PR-AUC, but probability quality is still imperfect."
        elif head == "riskoff":
            pass_gate = False
            role = "warning_only"
            reason = "RiskOff has poor Brier skill and sparse-event instability."
        else:
            pass_gate = False
            role = "unknown"
            reason = "Unknown head."

        row = r.to_dict()
        row.update({
            "head_gate_pass": bool(pass_gate),
            "recommended_role": role,
            "reason": reason,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def decision_table(
    cost_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    neighborhood_df: pd.DataFrame,
    head_gate_df: pd.DataFrame,
    candidate: Dict,
) -> pd.DataFrame:
    # Cost gate
    cost_gate = bool(cost_df["economic_gate"].all()) if not cost_df.empty else False

    # Yearly gates
    yearly_calmar_rate = float(yearly_df["calmar_improved"].mean()) if "calmar_improved" in yearly_df.columns and len(yearly_df) else 0.0
    yearly_mdd_rate = float(yearly_df["mdd_improved"].mean()) if "mdd_improved" in yearly_df.columns and len(yearly_df) else 0.0
    yearly_cagr_gate_rate = float(yearly_df["cagr_not_worse_than_minus_2pct"].mean()) if "cagr_not_worse_than_minus_2pct" in yearly_df.columns and len(yearly_df) else 0.0
    yearly_gate = yearly_calmar_rate >= 0.60 and yearly_mdd_rate >= 0.60 and yearly_cagr_gate_rate >= 0.60

    # Neighborhood: top 20 중 경제 gate 통과가 충분한지
    if "stable_economic_gate" in neighborhood_df.columns:
        neighborhood_pass_count = int(neighborhood_df.head(20)["stable_economic_gate"].astype(str).str.lower().eq("true").sum())
    else:
        neighborhood_pass_count = 0
    neighborhood_gate = neighborhood_pass_count >= 5

    # Head gate: expansion confirmation must pass, h20 may fail but should not be direct-only
    gate_map = {str(r["head"]): bool(r["head_gate_pass"]) for _, r in head_gate_df.iterrows()}
    head_gate = bool(gate_map.get("highvol_expansion", False)) and not bool(gate_map.get("riskoff", False))

    economic_best_gate = bool(candidate.get("stable_economic_gate", False))

    # final decision
    if economic_best_gate and cost_gate and neighborhood_gate and head_gate:
        decision = "provisional_stable_candidate"
        stable = False
        reason = (
            "Economic/cost/neighborhood gates passed and expansion confirmation head is usable. "
            "Still not final Stable because validation is limited to QQQ/IEF and 2019-2026."
        )
    elif economic_best_gate:
        decision = "strong_candidate_requires_more_validation"
        stable = False
        reason = (
            "Best candidate passed economic gate, but at least one robustness gate remains insufficient."
        )
    else:
        decision = "candidate_only_not_stable"
        stable = False
        reason = "Economic gate did not pass."

    rows = [
        {"gate": "best_candidate_economic_gate", "pass": economic_best_gate, "value": candidate.get("stable_economic_gate"), "required": "True"},
        {"gate": "cost_sensitivity_gate", "pass": cost_gate, "value": float(cost_df["economic_gate"].mean()) if len(cost_df) else np.nan, "required": "all tested costs pass"},
        {"gate": "yearly_gate", "pass": yearly_gate, "value": {"calmar_rate": yearly_calmar_rate, "mdd_rate": yearly_mdd_rate, "cagr_gate_rate": yearly_cagr_gate_rate}, "required": ">=0.60 each"},
        {"gate": "neighborhood_gate", "pass": neighborhood_gate, "value": neighborhood_pass_count, "required": ">=5 top20 variants pass economic gate"},
        {"gate": "head_gate", "pass": head_gate, "value": gate_map, "required": "expansion confirmation pass; riskoff remains warning"},
        {"gate": "final_decision", "pass": stable, "value": decision, "required": reason},
    ]

    return pd.DataFrame(rows)


# ============================================================
# 4. Runner
# ============================================================

def resolve_paths(args) -> Dict[str, Path]:
    if args.input_dir:
        base = Path(args.input_dir)
        return {
            "summary_json": base / "dual_highvol_summary.json",
            "strategy_summary": base / "dual_highvol_strategy_summary.csv",
            "strategy_daily": base / "dual_highvol_strategy_daily_returns.csv",
            "head_summary": base / "dual_highvol_head_summary.csv",
            "fold_metrics": base / "dual_highvol_fold_metrics.csv",
            "predictions": base / "dual_highvol_oos_predictions.csv",
        }

    return {
        "summary_json": Path(args.summary_json),
        "strategy_summary": Path(args.strategy_summary),
        "strategy_daily": Path(args.strategy_daily),
        "head_summary": Path(args.head_summary),
        "fold_metrics": Path(args.fold_metrics),
        "predictions": Path(args.predictions),
    }


def run_validation(args) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_paths(args)

    summary_json = load_json(paths["summary_json"])
    strategy_summary = load_csv(paths["strategy_summary"])
    strategy_daily = load_csv(paths["strategy_daily"])
    head_summary = load_csv(paths["head_summary"])
    fold_metrics = load_csv(paths["fold_metrics"])
    predictions = load_csv(paths["predictions"])

    candidate = get_best_candidate(summary_json, strategy_summary)
    candidate_daily = filter_candidate_daily(strategy_daily, candidate)
    buyhold_daily = filter_benchmark_daily(strategy_daily, "buy_hold")

    cost_values = [float(x) for x in str(args.cost_bps_values).split(",") if str(x).strip()]
    drawdown_thresholds = [float(x) for x in str(args.drawdown_thresholds).split(",") if str(x).strip()]

    cost_df = cost_sensitivity(candidate_daily, buyhold_daily, cost_values)
    yearly_df = yearly_metrics(candidate_daily, buyhold_daily)
    drawdown_df = drawdown_attribution(candidate_daily, buyhold_daily, drawdown_thresholds)
    signal_df = signal_attribution(candidate_daily)
    neighborhood_df = neighborhood_table(strategy_summary, candidate)
    head_gate_df = head_gate(head_summary)
    decision_df = decision_table(cost_df, yearly_df, neighborhood_df, head_gate_df, candidate)

    outputs = {
        "cost_sensitivity": save_csv(output_dir / "candidate_cost_sensitivity.csv", cost_df),
        "yearly_metrics": save_csv(output_dir / "candidate_yearly_metrics.csv", yearly_df),
        "drawdown_attribution": save_csv(output_dir / "candidate_drawdown_attribution.csv", drawdown_df),
        "signal_attribution": save_csv(output_dir / "candidate_signal_attribution.csv", signal_df),
        "neighborhood": save_csv(output_dir / "candidate_neighborhood.csv", neighborhood_df),
        "head_gate": save_csv(output_dir / "candidate_head_gate.csv", head_gate_df),
        "decision": save_csv(output_dir / "candidate_decision.csv", decision_df),
    }

    final_decision_row = decision_df[decision_df["gate"] == "final_decision"].head(1).to_dict("records")
    final_decision = final_decision_row[0]["value"] if final_decision_row else "unknown"

    summary = {
        "experiment": "dual_highvol_candidate_validation",
        "candidate": candidate,
        "validation_period": {
            "start": str(pd.to_datetime(candidate_daily["date"]).min().date()),
            "end": str(pd.to_datetime(candidate_daily["date"]).max().date()),
            "rows": int(len(candidate_daily)),
        },
        "cost_sensitivity_summary": cost_df.to_dict("records"),
        "yearly_gate_summary": {
            "year_count": int(len(yearly_df)),
            "calmar_improved_rate": float(yearly_df["calmar_improved"].mean()) if len(yearly_df) else np.nan,
            "mdd_improved_rate": float(yearly_df["mdd_improved"].mean()) if len(yearly_df) else np.nan,
            "cagr_not_worse_than_minus_2pct_rate": float(yearly_df["cagr_not_worse_than_minus_2pct"].mean()) if len(yearly_df) else np.nan,
        },
        "neighborhood_summary": {
            "top20_economic_gate_count": int(neighborhood_df.head(20)["stable_economic_gate"].astype(str).str.lower().eq("true").sum())
                if "stable_economic_gate" in neighborhood_df.columns else 0,
            "top10_best_rows": neighborhood_df.head(10).to_dict("records"),
        },
        "head_gate": head_gate_df.to_dict("records"),
        "decision_rows": decision_df.to_dict("records"),
        "final_decision": final_decision,
        "output_files": {k: str(v) for k, v in outputs.items()},
    }

    outputs["summary"] = save_json(output_dir / "candidate_validation_summary.json", summary)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--summary-json", default="dual_highvol_summary.json")
    parser.add_argument("--strategy-summary", default="dual_highvol_strategy_summary.csv")
    parser.add_argument("--strategy-daily", default="dual_highvol_strategy_daily_returns.csv")
    parser.add_argument("--head-summary", default="dual_highvol_head_summary.csv")
    parser.add_argument("--fold-metrics", default="dual_highvol_fold_metrics.csv")
    parser.add_argument("--predictions", default="dual_highvol_oos_predictions.csv")
    parser.add_argument("--output-dir", default="dual_highvol_candidate_validation")

    parser.add_argument("--cost-bps-values", default="0,5,10,20,30")
    parser.add_argument("--drawdown-thresholds", default="-0.05,-0.10,-0.15,-0.20")

    args = parser.parse_args()

    outputs = run_validation(args)

    summary = load_json(outputs["summary"])
    candidate = summary.get("candidate", {})
    print("[OK] Dual-HighVol candidate validation completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print(json.dumps(
        {
            "candidate_hybrid_mode": candidate.get("hybrid_mode"),
            "candidate_persistence_mode": candidate.get("persistence_mode"),
            "candidate_defensive_equity_weight": candidate.get("defensive_equity_weight"),
            "candidate_defense_asset": candidate.get("defense_asset"),
            "candidate_cagr": candidate.get("cagr"),
            "candidate_mdd": candidate.get("mdd"),
            "candidate_calmar": candidate.get("calmar"),
            "candidate_cagr_diff_vs_buy_hold": candidate.get("cagr_diff_vs_buy_hold"),
            "candidate_mdd_diff_vs_buy_hold": candidate.get("mdd_diff_vs_buy_hold"),
            "candidate_calmar_diff_vs_buy_hold": candidate.get("calmar_diff_vs_buy_hold"),
            "yearly_gate_summary": summary.get("yearly_gate_summary"),
            "neighborhood_top20_economic_gate_count": summary.get("neighborhood_summary", {}).get("top20_economic_gate_count"),
            "final_decision": summary.get("final_decision"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
