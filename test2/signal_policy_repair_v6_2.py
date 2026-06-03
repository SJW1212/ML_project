# -*- coding: utf-8 -*-
"""
signal_policy_repair_v6_2.py

6.2 Signal Policy Repair.

Reads oos_signal_predictions.csv from touch_signal_policy_builder.py and applies
repaired signal taxonomy based on 6.1 Down-side Head Decile Decomposition.

This script does not retrain models and does not evaluate portfolio returns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


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


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}. columns={list(df.columns)}")


def load_predictions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"predictions file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = [
        "date", "asset_name", "fold_id", "y_up_fixed", "y_down_fixed",
        "return_score_percentile", "balanced_up_score_percentile",
        "balanced_down_score_percentile", "defensive_down_score_percentile",
    ]
    ensure_columns(df, required)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    non_numeric_cols = {"asset_name", "date", "final_policy_signal", "return_signal", "defensive_signal", "balanced_regime"}
    for col in df.columns:
        if col in non_numeric_cols:
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    if "final_policy_signal" not in df.columns:
        df["final_policy_signal"] = "OLD_SIGNAL_MISSING"
    return df


def apply_repaired_policy(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()
    r = out["return_score_percentile"].astype(float)
    bu = out["balanced_up_score_percentile"].astype(float)
    bd = out["balanced_down_score_percentile"].astype(float)
    dd = out["defensive_down_score_percentile"].astype(float)

    strong_return = (
        (r >= args.return_strong_threshold)
        & (bu >= args.balanced_up_edge_threshold)
        & (dd < args.defensive_conflict_threshold)
    )
    volatility_warning = (
        (r >= args.return_strong_threshold)
        & (dd >= args.defensive_conflict_threshold)
    )
    standard_up_watch = (
        (r >= args.return_standard_threshold)
        & (r < args.return_strong_threshold)
        & (bu >= args.balanced_up_edge_threshold)
        & (bd < args.balanced_down_block_threshold)
        & (dd < args.defensive_conflict_threshold)
    )
    mixed_activity_watch = (
        (r < args.return_strong_threshold)
        & (bu >= args.balanced_up_edge_threshold)
        & (bd >= args.balanced_down_block_threshold)
    )

    out["repaired_policy_signal"] = "NEUTRAL_NO_EDGE"
    out.loc[mixed_activity_watch, "repaired_policy_signal"] = "MIXED_ACTIVITY_WATCH"
    out.loc[standard_up_watch, "repaired_policy_signal"] = "STANDARD_UP_WATCH"
    out.loc[strong_return, "repaired_policy_signal"] = "STRONG_RETURN_SEEKING"
    out.loc[volatility_warning, "repaired_policy_signal"] = "VOLATILITY_EXPANSION_WARNING"

    out["flag_return_strong"] = r >= args.return_strong_threshold
    out["flag_return_standard"] = r >= args.return_standard_threshold
    out["flag_balanced_up_edge"] = bu >= args.balanced_up_edge_threshold
    out["flag_balanced_down_high"] = bd >= args.balanced_down_block_threshold
    out["flag_defensive_high"] = dd >= args.defensive_conflict_threshold
    out["downside_standalone_trigger_allowed"] = False
    return out


def group_metrics(g: pd.DataFrame) -> Dict:
    n = len(g)
    if n == 0:
        return {"rows": 0}
    y_up = g["y_up_fixed"].astype(float)
    y_down = g["y_down_fixed"].astype(float)
    both = (y_up == 1) & (y_down == 1)
    no_touch = (y_up == 0) & (y_down == 0)
    return {
        "rows": int(n),
        "up_touch_rate": float(y_up.mean()),
        "down_touch_rate": float(y_down.mean()),
        "both_touch_rate": float(both.mean()),
        "no_touch_rate": float(no_touch.mean()),
        "future_max_high_return_mean": float(g["future_max_high_return"].mean()) if "future_max_high_return" in g.columns else np.nan,
        "future_max_high_return_median": float(g["future_max_high_return"].median()) if "future_max_high_return" in g.columns else np.nan,
        "future_min_low_return_mean": float(g["future_min_low_return"].mean()) if "future_min_low_return" in g.columns else np.nan,
        "future_min_low_return_median": float(g["future_min_low_return"].median()) if "future_min_low_return" in g.columns else np.nan,
        "return_score_mean": float(g["return_score_percentile"].mean()),
        "balanced_up_score_mean": float(g["balanced_up_score_percentile"].mean()),
        "balanced_down_score_mean": float(g["balanced_down_score_percentile"].mean()),
        "defensive_down_score_mean": float(g["defensive_down_score_percentile"].mean()),
    }


def summarize_policy(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    total_rows = len(df)
    global_up = float(df["y_up_fixed"].mean())
    global_down = float(df["y_down_fixed"].mean())
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(group_metrics(g))
        row["row_rate"] = safe_divide(row["rows"], total_rows)
        row["global_up_touch_rate"] = global_up
        row["global_down_touch_rate"] = global_down
        row["up_lift_vs_global"] = safe_divide(row["up_touch_rate"], global_up)
        row["down_lift_vs_global"] = safe_divide(row["down_touch_rate"], global_down)
        row["net_up_minus_down_touch"] = row["up_touch_rate"] - row["down_touch_rate"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("rows", ascending=False).reset_index(drop=True)
    return out


def transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(df["final_policy_signal"], df["repaired_policy_signal"], dropna=False).reset_index()


def threshold_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    conditions = {
        "return_score_ge_0p90": df["return_score_percentile"] >= 0.90,
        "return_score_ge_0p80": df["return_score_percentile"] >= 0.80,
        "balanced_up_ge_0p80": df["balanced_up_score_percentile"] >= 0.80,
        "balanced_down_ge_0p80": df["balanced_down_score_percentile"] >= 0.80,
        "defensive_down_ge_0p90": df["defensive_down_score_percentile"] >= 0.90,
        "return_ge_0p90_and_defensive_ge_0p90": (
            (df["return_score_percentile"] >= 0.90)
            & (df["defensive_down_score_percentile"] >= 0.90)
        ),
        "strong_return_condition": (
            (df["return_score_percentile"] >= 0.90)
            & (df["balanced_up_score_percentile"] >= 0.80)
            & (df["defensive_down_score_percentile"] < 0.90)
        ),
        "standard_up_watch_condition": (
            (df["return_score_percentile"] >= 0.80)
            & (df["return_score_percentile"] < 0.90)
            & (df["balanced_up_score_percentile"] >= 0.80)
            & (df["balanced_down_score_percentile"] < 0.80)
            & (df["defensive_down_score_percentile"] < 0.90)
        ),
    }
    global_up = float(df["y_up_fixed"].mean())
    global_down = float(df["y_down_fixed"].mean())
    rows = []
    for name, cond in conditions.items():
        g = df[cond].copy()
        m = group_metrics(g)
        m.update({
            "condition": name,
            "signal_rate": safe_divide(len(g), len(df)),
            "up_lift_vs_global": safe_divide(m.get("up_touch_rate"), global_up),
            "down_lift_vs_global": safe_divide(m.get("down_touch_rate"), global_down),
        })
        rows.append(m)
    return pd.DataFrame(rows)


def rejected_signal_audit(df: pd.DataFrame) -> pd.DataFrame:
    old_signals = [
        "DEFENSIVE_RISK", "RISK_WATCH", "WEAK_RETURN_WATCH", "STANDARD_RETURN_SEEKING",
        "CONFLICT_UP_AND_DEFENSIVE_RISK", "CONFLICT_BALANCED_BOTH_HIGH",
    ]
    rows = []
    for old_signal in old_signals:
        g = df[df["final_policy_signal"] == old_signal].copy()
        if g.empty:
            continue
        m = group_metrics(g)
        m.update({
            "old_signal": old_signal,
            "most_common_repaired_signal": g["repaired_policy_signal"].value_counts().index[0],
            "repaired_distribution_json": json.dumps(g["repaired_policy_signal"].value_counts(normalize=True).to_dict(), ensure_ascii=False),
        })
        rows.append(m)
    return pd.DataFrame(rows)


def build_decision(policy_summary: pd.DataFrame) -> Dict:
    rows = {r["repaired_policy_signal"]: r for _, r in policy_summary.iterrows()}
    decision = {}
    s = rows.get("STRONG_RETURN_SEEKING")
    if s is not None:
        decision["strong_return_seeking"] = (
            "PASS_AS_UPSIDE_OPPORTUNITY_SIGNAL"
            if s["up_lift_vs_global"] > 1.30 and s["up_touch_rate"] > s["down_touch_rate"]
            else "WEAK_OR_FAIL"
        )
    st = rows.get("STANDARD_UP_WATCH")
    if st is not None:
        decision["standard_up_watch"] = (
            "PASS_AS_WEAK_UP_WATCH"
            if st["up_lift_vs_global"] > 1.10 and st["up_touch_rate"] >= st["down_touch_rate"]
            else "WEAK_OR_REJECT"
        )
    v = rows.get("VOLATILITY_EXPANSION_WARNING")
    if v is not None:
        decision["volatility_expansion_warning"] = (
            "PASS_AS_TWO_SIDED_VOLATILITY_WARNING"
            if v["up_lift_vs_global"] > 1.20 and v["down_lift_vs_global"] > 1.20
            else "WEAK_OR_REJECT"
        )
    n = rows.get("NEUTRAL_NO_EDGE")
    if n is not None:
        decision["neutral_no_edge"] = (
            "PASS_AS_LOW_EDGE_BUCKET"
            if n["up_lift_vs_global"] < 1.0 and n["down_lift_vs_global"] <= 1.0
            else "CHECK_NEUTRAL_BUCKET"
        )
    decision["signal_rows"] = {k: {kk: json_default(vv) for kk, vv in dict(r).items()} for k, r in rows.items()}
    return decision


def run(args) -> Dict[str, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions)
    repaired = apply_repaired_policy(df, args)
    policy_summary = summarize_policy(repaired, ["repaired_policy_signal"])
    asset_summary = summarize_policy(repaired, ["asset_name", "repaired_policy_signal"])
    annual_summary = summarize_policy(repaired, ["asset_name", "year", "repaired_policy_signal"])
    fold_summary = summarize_policy(repaired, ["asset_name", "fold_id", "repaired_policy_signal"])
    transition = transition_matrix(repaired)
    threshold_diag = threshold_diagnostics(repaired)
    rejected_audit = rejected_signal_audit(repaired)
    decision = build_decision(policy_summary)

    config = {
        "experiment": "signal_policy_repair_v6_2",
        "input_file": str(args.predictions),
        "policy_thresholds": {
            "return_strong_threshold": args.return_strong_threshold,
            "return_standard_threshold": args.return_standard_threshold,
            "balanced_up_edge_threshold": args.balanced_up_edge_threshold,
            "balanced_down_block_threshold": args.balanced_down_block_threshold,
            "defensive_conflict_threshold": args.defensive_conflict_threshold,
        },
        "repaired_policy": {
            "STRONG_RETURN_SEEKING": "return_score >= 0.90 and balanced_up_score >= 0.80 and defensive_down_score < 0.90",
            "VOLATILITY_EXPANSION_WARNING": "return_score >= 0.90 and defensive_down_score >= 0.90",
            "STANDARD_UP_WATCH": "0.80 <= return_score < 0.90 and balanced_up_score >= 0.80 and balanced_down_score < 0.80 and defensive_down_score < 0.90",
            "MIXED_ACTIVITY_WATCH": "return_score < 0.90 and balanced_up_score >= 0.80 and balanced_down_score >= 0.80",
            "NEUTRAL_NO_EDGE": "all remaining cases",
        },
        "removed_or_downgraded": ["DEFENSIVE_RISK", "DEFENSIVE_WATCH_ONLY", "WEAK_RETURN_WATCH"],
        "interpretation": {
            "score_percentile": "calibration-window percentile rank, not literal probability",
            "portfolio_allocation": False,
            "risk_off_trigger": False,
        },
    }

    summary = {
        "experiment": "signal_policy_repair_v6_2",
        "prediction_rows": int(len(repaired)),
        "asset_count": int(repaired["asset_name"].nunique()),
        "date_start": str(repaired["date"].min().date()),
        "date_end": str(repaired["date"].max().date()),
        "global_up_touch_rate": float(repaired["y_up_fixed"].mean()),
        "global_down_touch_rate": float(repaired["y_down_fixed"].mean()),
        "repaired_signal_counts": repaired["repaired_policy_signal"].value_counts().to_dict(),
        "decision": decision,
        "decision_note": "Repaired taxonomy only. No portfolio return, MDD, turnover, or transaction cost evaluation.",
    }

    outputs = {
        "summary": save_json(out_dir / "signal_policy_repair_summary.json", summary),
        "config": save_json(out_dir / "signal_policy_repair_config.json", config),
        "repaired_signal_predictions": save_csv(out_dir / "repaired_signal_predictions.csv", repaired),
        "policy_repair_summary": save_csv(out_dir / "policy_repair_summary.csv", policy_summary),
        "asset_policy_repair_summary": save_csv(out_dir / "asset_policy_repair_summary.csv", asset_summary),
        "annual_policy_repair_summary": save_csv(out_dir / "annual_policy_repair_summary.csv", annual_summary),
        "fold_policy_repair_summary": save_csv(out_dir / "fold_policy_repair_summary.csv", fold_summary),
        "transition_matrix_old_to_repaired": save_csv(out_dir / "transition_matrix_old_to_repaired.csv", transition),
        "repaired_signal_threshold_diagnostics": save_csv(out_dir / "repaired_signal_threshold_diagnostics.csv", threshold_diag),
        "rejected_signal_audit": save_csv(out_dir / "rejected_signal_audit.csv", rejected_audit),
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="signal_policy_repair_v6_2_output")
    parser.add_argument("--return-strong-threshold", type=float, default=0.90)
    parser.add_argument("--return-standard-threshold", type=float, default=0.80)
    parser.add_argument("--balanced-up-edge-threshold", type=float, default=0.80)
    parser.add_argument("--balanced-down-block-threshold", type=float, default=0.80)
    parser.add_argument("--defensive-conflict-threshold", type=float, default=0.90)
    args = parser.parse_args()
    outputs = run(args)
    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    print("[OK] Signal Policy Repair v6.2 completed.")
    print(json.dumps({
        "prediction_rows": summary["prediction_rows"],
        "asset_count": summary["asset_count"],
        "global_up_touch_rate": summary["global_up_touch_rate"],
        "global_down_touch_rate": summary["global_down_touch_rate"],
        "repaired_signal_counts": summary["repaired_signal_counts"],
        "decision": summary["decision"],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
