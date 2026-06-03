# -*- coding: utf-8 -*-
"""
signal_naive_pnl_validator_v6_5.py

6.5단계: Signal Return Validation / Naive PnL 준비 검증

목적
----
6.4 결과(v6_4_signal_predictions.csv)를 입력으로 받아 신호별 고정 보유기간
close-to-close forward return을 평가합니다.

이 스크립트는 최종 포트폴리오 백테스트가 아닙니다.
신호가 실제 종가 기준 수익률에서도 우위를 갖는지 확인하기 위한
naive event-return / daily equal-weight signal-return 진단입니다.

입력
----
- v6_4_signal_predictions.csv

필수 컬럼
---------
asset_name, date, fold_id, v6_4_signal,
future_close_return_h5, future_close_return_h10, future_close_return_h20

선택 컬럼
---------
y_up_fixed, y_down_fixed, future_max_high_return, future_min_low_return,
return_score_percentile, balanced_up_score_percentile,
balanced_down_score_percentile, defensive_down_score_percentile

출력
----
output_dir/
├─ naive_pnl_validation_summary.json
├─ naive_pnl_validation_config.json
├─ event_return_summary.csv
├─ asset_event_return_summary.csv
├─ annual_event_return_summary.csv
├─ fold_event_return_summary.csv
├─ daily_equal_weight_signal_summary.csv
├─ cost_sensitivity_summary.csv
├─ signal_pairwise_alpha.csv
├─ horizon_decision_table.csv
├─ signal_decision_report.csv
└─ event_return_distribution.csv

주의
----
- cost_bps는 round-trip cost로 해석하며, forward return에서 cost_bps / 10000을 차감합니다.
- 각 행을 독립 signal event로 취급합니다. 중복 포지션, capital allocation, turnover, MDD는 아직 엄밀히 반영하지 않습니다.
- daily_equal_weight_signal_summary는 같은 날짜의 동일 신호 자산들을 동일가중 평균한 event return을 요약합니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Utilities
# ============================================================

def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def save_csv(path: str | Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_json(path: str | Path, data: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    return path


def safe_divide(a, b, default=np.nan) -> float:
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return default
        return float(a / b)
    except Exception:
        return default


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if str(x).strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if str(x).strip()]


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}. available={list(df.columns)}")


def infer_horizons(df: pd.DataFrame, requested: List[int]) -> List[int]:
    out = []
    for h in requested:
        col = f"future_close_return_h{h}"
        if col in df.columns:
            out.append(h)
    if not out:
        available = [c for c in df.columns if c.startswith("future_close_return_h")]
        raise ValueError(f"no requested future_close_return columns found. available={available}")
    return out


def load_predictions(path: str | Path, signal_col: str, horizons: List[int]) -> Tuple[pd.DataFrame, List[int]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"predictions file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    ensure_columns(df, ["asset_name", "date", "fold_id", signal_col])
    horizons = infer_horizons(df, horizons)
    ensure_columns(df, [f"future_close_return_h{h}" for h in horizons])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year

    numeric_cols = ["fold_id", "year"] + [f"future_close_return_h{h}" for h in horizons]
    optional_numeric = [
        "y_up_fixed", "y_down_fixed", "y_up_target", "y_down_target",
        "future_max_high_return", "future_min_low_return",
        "return_score_percentile", "balanced_up_score_percentile",
        "balanced_down_score_percentile", "defensive_down_score_percentile",
    ]
    numeric_cols += [c for c in optional_numeric if c in df.columns]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "asset_name", signal_col]).copy()
    return df, horizons


# ============================================================
# Metrics
# ============================================================

def max_drawdown_from_returns(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return np.nan
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def simple_sharpe(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        return np.nan
    sd = float(r.std(ddof=1))
    if sd == 0 or pd.isna(sd):
        return np.nan
    return float(r.mean() / sd)


def event_metrics(g: pd.DataFrame, ret_col: str, baseline_mean: float | None = None) -> Dict:
    r = g[ret_col].dropna().astype(float)
    if r.empty:
        return {
            "events": 0,
            "mean_return": np.nan,
            "median_return": np.nan,
            "std_return": np.nan,
            "positive_rate": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "avg_gain": np.nan,
            "avg_loss": np.nan,
            "payoff_ratio": np.nan,
            "profit_factor": np.nan,
            "naive_compound_return": np.nan,
            "event_sharpe_like": np.nan,
            "max_event_drawdown": np.nan,
            "alpha_vs_all_mean": np.nan,
        }

    pos = r[r > 0]
    neg = r[r < 0]
    avg_gain = float(pos.mean()) if len(pos) else np.nan
    avg_loss = float(neg.mean()) if len(neg) else np.nan
    gross_gain = float(pos.sum()) if len(pos) else 0.0
    gross_loss_abs = float((-neg).sum()) if len(neg) else 0.0

    out = {
        "events": int(len(r)),
        "mean_return": float(r.mean()),
        "median_return": float(r.median()),
        "std_return": float(r.std(ddof=1)) if len(r) > 1 else np.nan,
        "positive_rate": float((r > 0).mean()),
        "p05": float(r.quantile(0.05)),
        "p10": float(r.quantile(0.10)),
        "p25": float(r.quantile(0.25)),
        "p75": float(r.quantile(0.75)),
        "p90": float(r.quantile(0.90)),
        "p95": float(r.quantile(0.95)),
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "payoff_ratio": safe_divide(avg_gain, abs(avg_loss)) if not pd.isna(avg_loss) else np.nan,
        "profit_factor": safe_divide(gross_gain, gross_loss_abs),
        "naive_compound_return": float((1.0 + r).prod() - 1.0),
        "event_sharpe_like": simple_sharpe(r),
        "max_event_drawdown": max_drawdown_from_returns(r),
        "alpha_vs_all_mean": float(r.mean() - baseline_mean) if baseline_mean is not None and not pd.isna(baseline_mean) else np.nan,
    }

    if "y_up_fixed" in g.columns:
        out["up_touch_rate"] = float(g["y_up_fixed"].dropna().mean()) if g["y_up_fixed"].notna().any() else np.nan
    if "y_down_fixed" in g.columns:
        out["down_touch_rate"] = float(g["y_down_fixed"].dropna().mean()) if g["y_down_fixed"].notna().any() else np.nan
    if "future_max_high_return" in g.columns:
        out["future_max_high_return_mean"] = float(g["future_max_high_return"].mean())
    if "future_min_low_return" in g.columns:
        out["future_min_low_return_mean"] = float(g["future_min_low_return"].mean())

    return out


def add_after_cost_returns(df: pd.DataFrame, horizons: List[int], costs_bps: List[float]) -> pd.DataFrame:
    out = df.copy()
    for h in horizons:
        base_col = f"future_close_return_h{h}"
        for cost in costs_bps:
            cost_tag = str(cost).replace(".", "p")
            out[f"net_return_h{h}_cost{cost_tag}bps"] = out[base_col].astype(float) - (float(cost) / 10000.0)
    return out


def summarize_event_returns(
    df: pd.DataFrame,
    signal_col: str,
    horizons: List[int],
    costs_bps: List[float],
    group_cols: List[str],
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        raw_col = f"future_close_return_h{h}"
        baseline_mean_by_cost = {}
        for cost in costs_bps:
            cost_tag = str(cost).replace(".", "p")
            ret_col = f"net_return_h{h}_cost{cost_tag}bps"
            baseline_mean_by_cost[cost] = float(df[ret_col].dropna().mean())

        for cost in costs_bps:
            cost_tag = str(cost).replace(".", "p")
            ret_col = f"net_return_h{h}_cost{cost_tag}bps"
            grouped = df.groupby(group_cols, dropna=False)
            for key, g in grouped:
                if not isinstance(key, tuple):
                    key = (key,)
                row = dict(zip(group_cols, key))
                row.update({
                    "horizon": h,
                    "cost_bps_roundtrip": cost,
                    "return_col": ret_col,
                    "gross_return_col": raw_col,
                })
                row.update(event_metrics(g, ret_col=ret_col, baseline_mean=baseline_mean_by_cost[cost]))
                rows.append(row)
    return pd.DataFrame(rows)


def daily_equal_weight_summary(
    df: pd.DataFrame,
    signal_col: str,
    horizons: List[int],
    costs_bps: List[float],
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        for cost in costs_bps:
            cost_tag = str(cost).replace(".", "p")
            ret_col = f"net_return_h{h}_cost{cost_tag}bps"
            daily = (
                df.groupby([signal_col, "date"], dropna=False)
                .agg(
                    daily_equal_weight_return=(ret_col, "mean"),
                    asset_count=("asset_name", "nunique"),
                    event_count=(ret_col, "count"),
                )
                .reset_index()
                .sort_values([signal_col, "date"])
            )
            for signal, g in daily.groupby(signal_col, dropna=False):
                r = g["daily_equal_weight_return"].dropna().astype(float)
                if r.empty:
                    continue
                rows.append({
                    signal_col: signal,
                    "horizon": h,
                    "cost_bps_roundtrip": cost,
                    "signal_dates": int(len(r)),
                    "mean_daily_equal_weight_return": float(r.mean()),
                    "median_daily_equal_weight_return": float(r.median()),
                    "positive_date_rate": float((r > 0).mean()),
                    "naive_compound_equal_weight_return": float((1.0 + r).prod() - 1.0),
                    "max_equal_weight_drawdown": max_drawdown_from_returns(r),
                    "equal_weight_sharpe_like": simple_sharpe(r),
                    "mean_assets_per_signal_date": float(g["asset_count"].mean()),
                    "median_assets_per_signal_date": float(g["asset_count"].median()),
                })
    return pd.DataFrame(rows)


def return_distribution(df: pd.DataFrame, signal_col: str, horizons: List[int]) -> pd.DataFrame:
    rows = []
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for h in horizons:
        col = f"future_close_return_h{h}"
        for signal, g in df.groupby(signal_col, dropna=False):
            r = g[col].dropna().astype(float)
            if r.empty:
                continue
            row = {signal_col: signal, "horizon": h, "events": int(len(r))}
            for q in quantiles:
                row[f"q{int(q*100):02d}"] = float(r.quantile(q))
            row["mean"] = float(r.mean())
            row["std"] = float(r.std(ddof=1)) if len(r) > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_alpha(event_summary: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    rows = []
    # Use cost 0 and cost 10 primarily for pairwise signal difference.
    subset = event_summary.copy()
    for (h, cost), g in subset.groupby(["horizon", "cost_bps_roundtrip"], dropna=False):
        means = g.set_index(signal_col)["mean_return"].to_dict()
        medians = g.set_index(signal_col)["median_return"].to_dict()
        win_rates = g.set_index(signal_col)["positive_rate"].to_dict()
        for a in means:
            for b in means:
                if a == b:
                    continue
                rows.append({
                    "horizon": h,
                    "cost_bps_roundtrip": cost,
                    "signal_a": a,
                    "signal_b": b,
                    "mean_alpha_a_minus_b": means[a] - means[b],
                    "median_alpha_a_minus_b": medians[a] - medians[b],
                    "positive_rate_diff_a_minus_b": win_rates[a] - win_rates[b],
                })
    return pd.DataFrame(rows)


def stability_rates(summary_df: pd.DataFrame, signal_col: str, scope_cols: List[str]) -> pd.DataFrame:
    rows = []
    # summary_df should include signal_col, horizon, cost, mean_return, median_return, positive_rate, alpha_vs_all_mean
    for (signal, h, cost), g in summary_df.groupby([signal_col, "horizon", "cost_bps_roundtrip"], dropna=False):
        if g.empty:
            continue
        rows.append({
            signal_col: signal,
            "horizon": h,
            "cost_bps_roundtrip": cost,
            "group_count": int(len(g)),
            "positive_mean_return_group_rate": float((g["mean_return"] > 0).mean()),
            "positive_median_return_group_rate": float((g["median_return"] > 0).mean()),
            "positive_alpha_vs_all_group_rate": float((g["alpha_vs_all_mean"] > 0).mean()),
            "min_group_mean_return": float(g["mean_return"].min()),
            "median_group_mean_return": float(g["mean_return"].median()),
            "mean_group_mean_return": float(g["mean_return"].mean()),
            "min_group_alpha_vs_all": float(g["alpha_vs_all_mean"].min()),
            "median_group_alpha_vs_all": float(g["alpha_vs_all_mean"].median()),
        })
    return pd.DataFrame(rows)


def build_decision_table(event_summary: pd.DataFrame, asset_summary: pd.DataFrame, annual_summary: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    rows = []
    # Primary horizon h10/h20, cost 10 bps.
    for h in sorted(event_summary["horizon"].unique()):
        for cost in sorted(event_summary["cost_bps_roundtrip"].unique()):
            if cost not in [0.0, 10.0, 20.0]:
                continue
            g = event_summary[(event_summary["horizon"] == h) & (event_summary["cost_bps_roundtrip"] == cost)]
            assets = asset_summary[(asset_summary["horizon"] == h) & (asset_summary["cost_bps_roundtrip"] == cost)]
            annual = annual_summary[(annual_summary["horizon"] == h) & (annual_summary["cost_bps_roundtrip"] == cost)]
            for _, r in g.iterrows():
                signal = r[signal_col]
                a = assets[assets[signal_col] == signal]
                y = annual[annual[signal_col] == signal]
                asset_pos_alpha = float((a["alpha_vs_all_mean"] > 0).mean()) if len(a) else np.nan
                annual_pos_alpha = float((y["alpha_vs_all_mean"] > 0).mean()) if len(y) else np.nan
                annual_pos_mean = float((y["mean_return"] > 0).mean()) if len(y) else np.nan

                decision = "UNDECIDED"
                if signal in ["HIGH_CONFIDENCE_RETURN_SEEKING", "STRONG_RETURN_SEEKING"]:
                    if r["mean_return"] > 0 and r["median_return"] > 0 and r["alpha_vs_all_mean"] > 0 and asset_pos_alpha >= 0.75 and annual_pos_alpha >= 0.60:
                        decision = "PASS_AS_NAIVE_UPSIDE_SIGNAL"
                    elif r["mean_return"] > 0 and r["median_return"] > 0:
                        decision = "PASS_BUT_WEAK_STABILITY"
                    else:
                        decision = "REJECT_OR_REPAIR"
                elif signal == "MIXED_ACTIVITY_WATCH":
                    if r["mean_return"] > 0 and r["median_return"] > 0 and asset_pos_alpha >= 0.75:
                        decision = "PROMISING_BUT_NEEDS_SAMPLE_VALIDATION"
                    else:
                        decision = "WATCH_ONLY"
                elif signal == "VOLATILITY_EXPANSION_WARNING":
                    if r.get("down_touch_rate", np.nan) > r.get("up_touch_rate", np.nan):
                        decision = "KEEP_AS_VOLATILITY_WARNING_NOT_LONG"
                    else:
                        decision = "CHECK_WARNING_ROLE"
                elif signal == "NEUTRAL_NO_EDGE":
                    decision = "BASELINE_BUCKET"

                rows.append({
                    signal_col: signal,
                    "horizon": h,
                    "cost_bps_roundtrip": cost,
                    "events": int(r["events"]),
                    "mean_return": float(r["mean_return"]),
                    "median_return": float(r["median_return"]),
                    "positive_rate": float(r["positive_rate"]),
                    "alpha_vs_all_mean": float(r["alpha_vs_all_mean"]),
                    "asset_positive_alpha_rate": asset_pos_alpha,
                    "annual_positive_alpha_rate": annual_pos_alpha,
                    "annual_positive_mean_return_rate": annual_pos_mean,
                    "decision": decision,
                })
    return pd.DataFrame(rows)


def signal_decision_report(decision_table: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    rows = []
    # prioritize h10 10bps, h20 10bps, h5 10bps
    for signal, g in decision_table.groupby(signal_col, dropna=False):
        row = {signal_col: signal}
        for h in [5, 10, 20]:
            s = g[(g["horizon"] == h) & (g["cost_bps_roundtrip"] == 10.0)]
            if len(s):
                rr = s.iloc[0]
                row[f"h{h}_mean_return_10bps"] = rr["mean_return"]
                row[f"h{h}_median_return_10bps"] = rr["median_return"]
                row[f"h{h}_positive_rate_10bps"] = rr["positive_rate"]
                row[f"h{h}_alpha_vs_all_10bps"] = rr["alpha_vs_all_mean"]
                row[f"h{h}_decision"] = rr["decision"]
        # final rule
        h10 = row.get("h10_decision", "")
        h20 = row.get("h20_decision", "")
        if signal == "HIGH_CONFIDENCE_RETURN_SEEKING":
            final = "KEEP_PRIMARY_HIGH_CONFIDENCE" if "PASS" in h10 or "PASS" in h20 else "RECHECK"
        elif signal == "STRONG_RETURN_SEEKING":
            final = "KEEP_SECONDARY_UPSIDE" if "PASS" in h10 or "PASS" in h20 else "RECHECK"
        elif signal == "MIXED_ACTIVITY_WATCH":
            final = "AUDIT_AS_POTENTIAL_UP_EDGE" if "PROMISING" in h10 or "PROMISING" in h20 else "WATCH_ONLY"
        elif signal == "VOLATILITY_EXPANSION_WARNING":
            final = "KEEP_AS_WARNING_ONLY"
        else:
            final = "BASELINE_OR_NO_EDGE"
        row["final_signal_decision"] = final
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Main run
# ============================================================

def run(args) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = parse_int_list(args.horizons)
    costs_bps = parse_float_list(args.costs_bps)

    df, horizons = load_predictions(args.predictions, signal_col=args.signal_col, horizons=horizons)
    df = add_after_cost_returns(df, horizons=horizons, costs_bps=costs_bps)

    # Main summaries
    event_summary = summarize_event_returns(df, args.signal_col, horizons, costs_bps, [args.signal_col])
    asset_summary = summarize_event_returns(df, args.signal_col, horizons, costs_bps, ["asset_name", args.signal_col])
    annual_summary = summarize_event_returns(df, args.signal_col, horizons, costs_bps, ["asset_name", "year", args.signal_col])
    fold_summary = summarize_event_returns(df, args.signal_col, horizons, costs_bps, ["asset_name", "fold_id", args.signal_col])

    daily_summary = daily_equal_weight_summary(df, args.signal_col, horizons, costs_bps)
    dist = return_distribution(df, args.signal_col, horizons)
    alpha = pairwise_alpha(event_summary, args.signal_col)
    decision_table = build_decision_table(event_summary, asset_summary, annual_summary, args.signal_col)
    decision_report = signal_decision_report(decision_table, args.signal_col)

    # Cost sensitivity is event_summary focused on key signals/horizons.
    cost_sensitivity = event_summary.copy().sort_values([args.signal_col, "horizon", "cost_bps_roundtrip"])

    config = {
        "experiment": "signal_naive_pnl_validator_v6_5",
        "input_predictions": str(args.predictions),
        "signal_col": args.signal_col,
        "horizons": horizons,
        "costs_bps_roundtrip": costs_bps,
        "cost_interpretation": "after_cost_return = future_close_return - cost_bps / 10000; cost_bps is round-trip cost.",
        "interpretation": {
            "portfolio_backtest": False,
            "overlapping_positions_resolved": False,
            "turnover_mdd_final": False,
            "purpose": "Naive event-level signal return validation before portfolio allocation/backtest.",
        },
    }

    # Compact summary
    primary = decision_report.to_dict(orient="records")
    signal_counts = df[args.signal_col].value_counts().to_dict()
    summary = {
        "experiment": "signal_naive_pnl_validator_v6_5",
        "prediction_rows": int(len(df)),
        "asset_count": int(df["asset_name"].nunique()),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "signal_counts": signal_counts,
        "horizons": horizons,
        "costs_bps_roundtrip": costs_bps,
        "primary_signal_decisions": primary,
        "decision_note": "Naive signal PnL validation only. Not final portfolio backtest.",
    }

    outputs = {
        "summary": save_json(output_dir / "naive_pnl_validation_summary.json", summary),
        "config": save_json(output_dir / "naive_pnl_validation_config.json", config),
        "event_return_summary": save_csv(output_dir / "event_return_summary.csv", event_summary),
        "asset_event_return_summary": save_csv(output_dir / "asset_event_return_summary.csv", asset_summary),
        "annual_event_return_summary": save_csv(output_dir / "annual_event_return_summary.csv", annual_summary),
        "fold_event_return_summary": save_csv(output_dir / "fold_event_return_summary.csv", fold_summary),
        "daily_equal_weight_signal_summary": save_csv(output_dir / "daily_equal_weight_signal_summary.csv", daily_summary),
        "cost_sensitivity_summary": save_csv(output_dir / "cost_sensitivity_summary.csv", cost_sensitivity),
        "signal_pairwise_alpha": save_csv(output_dir / "signal_pairwise_alpha.csv", alpha),
        "horizon_decision_table": save_csv(output_dir / "horizon_decision_table.csv", decision_table),
        "signal_decision_report": save_csv(output_dir / "signal_decision_report.csv", decision_report),
        "event_return_distribution": save_csv(output_dir / "event_return_distribution.csv", dist),
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to v6_4_signal_predictions.csv")
    parser.add_argument("--output-dir", default="signal_naive_pnl_v6_5_output")
    parser.add_argument("--signal-col", default="v6_4_signal")
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--costs-bps", default="0,5,10,20")

    args = parser.parse_args()
    outputs = run(args)
    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    print("[OK] Signal Naive PnL Validation v6.5 completed.")
    print(json.dumps({
        "prediction_rows": summary["prediction_rows"],
        "asset_count": summary["asset_count"],
        "date_start": summary["date_start"],
        "date_end": summary["date_end"],
        "signal_counts": summary["signal_counts"],
        "horizons": summary["horizons"],
        "costs_bps_roundtrip": summary["costs_bps_roundtrip"],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
