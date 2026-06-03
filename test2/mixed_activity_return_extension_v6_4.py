# -*- coding: utf-8 -*-
"""
mixed_activity_return_extension_v6_4.py

6.4단계: Mixed Activity Audit + Return Distribution Extension

목적
----
6.3에서 의외로 좋게 나온 MIXED_ACTIVITY_WATCH를 별도 검증하고,
HIGH_CONFIDENCE_RETURN_SEEKING을 분리한 v6.4 신호 체계를 재평가합니다.

또한 원본 OHLCV가 제공되면 future close-to-close return을 추가합니다.
원본 OHLCV가 없으면 기존 prediction 파일에 포함된 future_max_high_return / future_min_low_return만으로 진단합니다.

입력
----
필수:
- repaired_signal_predictions.csv

선택:
- --ohlcv-inputs "QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv"
- --asset-names "QQQ,SPY,SOXX,XLK"

출력
----
output_dir/
├─ mixed_activity_audit_summary.json
├─ mixed_activity_audit_config.json
├─ v6_4_signal_predictions.csv
├─ v6_4_signal_profile.csv
├─ asset_v6_4_signal_profile.csv
├─ annual_v6_4_signal_profile.csv
├─ fold_v6_4_signal_profile.csv
├─ mixed_activity_asset_audit.csv
├─ mixed_activity_annual_audit.csv
├─ mixed_activity_fold_audit.csv
├─ high_confidence_vs_strong_profile.csv
├─ return_distribution_by_signal.csv
├─ return_score_threshold_sensitivity.csv
└─ score_bin_return_distribution.csv

주의
----
이 스크립트는 재학습하지 않습니다.
포트폴리오 수익률, 거래비용, turnover, MDD는 아직 평가하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
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
    for col in df.columns:
        if col not in {"asset_name", "date", "final_policy_signal", "repaired_policy_signal", "return_signal", "defensive_signal", "balanced_regime"}:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    if "repaired_policy_signal" not in df.columns:
        df["repaired_policy_signal"] = "MISSING"
    return df


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    rename = {"datetime": "date", "timestamp": "date", "adjclose": "adj_close", "adjusted_close": "adj_close"}
    out = out.rename(columns={k:v for k,v in rename.items() if k in out.columns})
    required = ["date", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV missing required columns: {missing}. columns={list(out.columns)}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "adj_close" in out.columns:
        out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
        out["_close_for_return"] = out["adj_close"].fillna(out["close"])
    else:
        out["_close_for_return"] = out["close"]
    return out


def attach_future_close_returns(pred: pd.DataFrame, ohlcv_inputs: str, asset_names: str, horizons: List[int]) -> tuple[pd.DataFrame, Dict]:
    if not ohlcv_inputs or not asset_names:
        return pred, {"close_return_available": False, "reason": "ohlcv inputs not supplied"}
    inputs = parse_list(ohlcv_inputs)
    names = parse_list(asset_names)
    if len(inputs) != len(names):
        raise ValueError(f"ohlcv input count != asset_names count: {len(inputs)} vs {len(names)}")

    out = pred.copy()
    meta = {"close_return_available": True, "assets": []}
    merged_parts = []
    for asset, path in zip(names, inputs):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"OHLCV file not found for {asset}: {p}")
        o = normalize_ohlcv_columns(pd.read_csv(p))
        keep = o[["date", "_close_for_return"]].copy()
        keep = keep.rename(columns={"_close_for_return": "close_for_return"})
        for h in horizons:
            keep[f"future_close_return_h{h}"] = keep["close_for_return"].shift(-h) / keep["close_for_return"] - 1.0
        keep["asset_name"] = asset
        keep = keep.drop(columns=["close_for_return"])
        meta["assets"].append({"asset_name": asset, "ohlcv_input": str(p), "rows": int(len(o)), "start": str(o["date"].min().date()), "end": str(o["date"].max().date())})
        merged_parts.append(keep)
    close_ret = pd.concat(merged_parts, ignore_index=True)
    out = out.merge(close_ret, on=["asset_name", "date"], how="left")
    for h in horizons:
        meta[f"future_close_return_h{h}_non_null"] = int(out[f"future_close_return_h{h}"].notna().sum())
    return out, meta


def apply_v6_4_policy(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()
    r = out["return_score_percentile"]
    bu = out["balanced_up_score_percentile"]
    bd = out["balanced_down_score_percentile"]
    dd = out["defensive_down_score_percentile"]

    high_conf = (r >= args.high_conf_threshold) & (bu >= args.balanced_up_edge) & (dd < args.defensive_conflict)
    strong = (r >= args.strong_threshold) & (r < args.high_conf_threshold) & (bu >= args.balanced_up_edge) & (dd < args.defensive_conflict)
    vol_warning = (r >= args.strong_threshold) & (dd >= args.defensive_conflict)
    mixed = (r < args.strong_threshold) & (bu >= args.balanced_up_edge) & (bd >= args.balanced_down_block)

    out["v6_4_signal"] = "NEUTRAL_NO_EDGE"
    out.loc[mixed, "v6_4_signal"] = "MIXED_ACTIVITY_WATCH"
    out.loc[strong, "v6_4_signal"] = "STRONG_RETURN_SEEKING"
    out.loc[high_conf, "v6_4_signal"] = "HIGH_CONFIDENCE_RETURN_SEEKING"
    out.loc[vol_warning, "v6_4_signal"] = "VOLATILITY_EXPANSION_WARNING"

    return out


def quantile_stats(s: pd.Series, prefix: str) -> Dict:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return {f"{prefix}_{k}": np.nan for k in ["count", "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max", "positive_rate"]}
    return {
        f"{prefix}_count": int(len(x)),
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        f"{prefix}_min": float(x.min()),
        f"{prefix}_p05": float(x.quantile(0.05)),
        f"{prefix}_p25": float(x.quantile(0.25)),
        f"{prefix}_median": float(x.median()),
        f"{prefix}_p75": float(x.quantile(0.75)),
        f"{prefix}_p95": float(x.quantile(0.95)),
        f"{prefix}_max": float(x.max()),
        f"{prefix}_positive_rate": float((x > 0).mean()),
    }


def group_profile(df: pd.DataFrame, group_cols: List[str], return_cols: List[str]) -> pd.DataFrame:
    global_up = float(df["y_up_fixed"].mean())
    global_down = float(df["y_down_fixed"].mean())
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        n = len(g)
        yup = g["y_up_fixed"].astype(float)
        ydn = g["y_down_fixed"].astype(float)
        row.update({
            "rows": int(n),
            "row_rate": safe_divide(n, len(df)),
            "up_touch_rate": float(yup.mean()),
            "down_touch_rate": float(ydn.mean()),
            "both_touch_rate": float(((yup == 1) & (ydn == 1)).mean()),
            "no_touch_rate": float(((yup == 0) & (ydn == 0)).mean()),
            "up_lift_vs_global": safe_divide(float(yup.mean()), global_up),
            "down_lift_vs_global": safe_divide(float(ydn.mean()), global_down),
            "net_up_minus_down_touch": float(yup.mean() - ydn.mean()),
            "return_score_mean": float(g["return_score_percentile"].mean()),
            "balanced_up_score_mean": float(g["balanced_up_score_percentile"].mean()),
            "balanced_down_score_mean": float(g["balanced_down_score_percentile"].mean()),
            "defensive_down_score_mean": float(g["defensive_down_score_percentile"].mean()),
        })
        for col in return_cols:
            if col in g.columns:
                row.update(quantile_stats(g[col], col))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("rows", ascending=False).reset_index(drop=True)
    return out


def threshold_sensitivity(df: pd.DataFrame, thresholds: List[float], args, return_cols: List[str], group_cols: Optional[List[str]]=None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = []
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    global_up = float(df["y_up_fixed"].mean())
    global_down = float(df["y_down_fixed"].mean())
    for thr in thresholds:
        for key, g0 in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            cond = (g0["return_score_percentile"] >= thr) & (g0["balanced_up_score_percentile"] >= args.balanced_up_edge) & (g0["defensive_down_score_percentile"] < args.defensive_conflict)
            g = g0[cond].copy()
            yup = g["y_up_fixed"].astype(float) if len(g) else pd.Series(dtype=float)
            ydn = g["y_down_fixed"].astype(float) if len(g) else pd.Series(dtype=float)
            row = dict(zip(group_cols, key))
            row.update({
                "threshold": thr,
                "rows_total_group": int(len(g0)),
                "signal_count": int(len(g)),
                "signal_rate": safe_divide(len(g), len(g0)),
                "up_touch_rate": float(yup.mean()) if len(g) else np.nan,
                "down_touch_rate": float(ydn.mean()) if len(g) else np.nan,
                "up_lift_vs_global": safe_divide(float(yup.mean()), global_up) if len(g) else np.nan,
                "down_lift_vs_global": safe_divide(float(ydn.mean()), global_down) if len(g) else np.nan,
                "net_up_minus_down_touch": float(yup.mean() - ydn.mean()) if len(g) else np.nan,
            })
            for col in return_cols:
                if col in g.columns:
                    row.update(quantile_stats(g[col], col))
            rows.append(row)
    return pd.DataFrame(rows)


def score_bin_profile(df: pd.DataFrame, return_cols: List[str]) -> pd.DataFrame:
    work = df.copy()
    valid = work["return_score_percentile"].notna()
    work.loc[valid, "return_score_decile"] = pd.qcut(work.loc[valid, "return_score_percentile"].rank(method="first"), q=10, labels=False, duplicates="drop").astype(float) + 1
    return group_profile(work.dropna(subset=["return_score_decile"]), ["return_score_decile"], return_cols)


def build_decision(signal_profile: pd.DataFrame, threshold_df: pd.DataFrame, close_meta: Dict) -> Dict:
    def get_signal(sig):
        r = signal_profile[signal_profile["v6_4_signal"] == sig]
        return None if r.empty else r.iloc[0].to_dict()
    decision = {"signals": {}, "close_return_available": bool(close_meta.get("close_return_available")), "notes": []}
    for sig in ["HIGH_CONFIDENCE_RETURN_SEEKING", "STRONG_RETURN_SEEKING", "MIXED_ACTIVITY_WATCH", "VOLATILITY_EXPANSION_WARNING", "NEUTRAL_NO_EDGE"]:
        row = get_signal(sig)
        if row is None:
            decision["signals"][sig] = "NO_SAMPLE"
            continue
        if sig in ["HIGH_CONFIDENCE_RETURN_SEEKING", "STRONG_RETURN_SEEKING"]:
            if row.get("up_lift_vs_global", np.nan) > 1.3 and row.get("up_touch_rate", 0) > row.get("down_touch_rate", 1):
                decision["signals"][sig] = "PASS_AS_UPSIDE_OPPORTUNITY"
            else:
                decision["signals"][sig] = "WEAK_OR_REJECT"
        elif sig == "MIXED_ACTIVITY_WATCH":
            if row.get("rows", 0) < 150:
                decision["signals"][sig] = "PROMISING_BUT_SMALL_SAMPLE"
            elif row.get("up_lift_vs_global", np.nan) > 1.2 and row.get("down_lift_vs_global", np.nan) < 1.0:
                decision["signals"][sig] = "PASS_AS_MIXED_UP_EDGE"
            else:
                decision["signals"][sig] = "WATCH_ONLY"
        elif sig == "VOLATILITY_EXPANSION_WARNING":
            if row.get("down_lift_vs_global", np.nan) > 1.5:
                decision["signals"][sig] = "PASS_AS_VOLATILITY_WARNING"
            else:
                decision["signals"][sig] = "WEAK_OR_REJECT"
        elif sig == "NEUTRAL_NO_EDGE":
            if row.get("up_lift_vs_global", np.nan) < 1.0 and row.get("down_lift_vs_global", np.nan) <= 1.0:
                decision["signals"][sig] = "PASS_AS_LOW_EDGE_BUCKET"
            else:
                decision["signals"][sig] = "CHECK_NEUTRAL_BUCKET"
    if not close_meta.get("close_return_available"):
        decision["notes"].append("future_close_return was not computed because OHLCV inputs were not supplied. Current return profile uses future_max_high_return and future_min_low_return only.")
    else:
        decision["notes"].append("future_close_return columns were attached from supplied OHLCV files.")
    return decision


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args.predictions)
    horizons = [int(x) for x in parse_list(args.close_return_horizons)]
    df, close_meta = attach_future_close_returns(df, args.ohlcv_inputs, args.asset_names, horizons)
    df = apply_v6_4_policy(df, args)

    return_cols = ["future_max_high_return", "future_min_low_return"]
    return_cols += [f"future_close_return_h{h}" for h in horizons if f"future_close_return_h{h}" in df.columns]

    signal_profile = group_profile(df, ["v6_4_signal"], return_cols)
    asset_profile = group_profile(df, ["asset_name", "v6_4_signal"], return_cols)
    annual_profile = group_profile(df, ["asset_name", "year", "v6_4_signal"], return_cols)
    fold_profile = group_profile(df, ["asset_name", "fold_id", "v6_4_signal"], return_cols)

    mixed = df[df["v6_4_signal"] == "MIXED_ACTIVITY_WATCH"].copy()
    mixed_asset = group_profile(mixed, ["asset_name"], return_cols) if len(mixed) else pd.DataFrame()
    mixed_annual = group_profile(mixed, ["asset_name", "year"], return_cols) if len(mixed) else pd.DataFrame()
    mixed_fold = group_profile(mixed, ["asset_name", "fold_id"], return_cols) if len(mixed) else pd.DataFrame()

    high_vs_strong = df[df["v6_4_signal"].isin(["HIGH_CONFIDENCE_RETURN_SEEKING", "STRONG_RETURN_SEEKING"])]
    high_strong_profile = group_profile(high_vs_strong, ["v6_4_signal"], return_cols) if len(high_vs_strong) else pd.DataFrame()

    thresholds = [float(x) for x in parse_list(args.thresholds)]
    threshold_global = threshold_sensitivity(df, thresholds, args, return_cols)
    threshold_asset = threshold_sensitivity(df, thresholds, args, return_cols, ["asset_name"])
    threshold_annual = threshold_sensitivity(df, thresholds, args, return_cols, ["asset_name", "year"])
    threshold_fold = threshold_sensitivity(df, thresholds, args, return_cols, ["asset_name", "fold_id"])
    bin_profile = score_bin_profile(df, return_cols)

    decision = build_decision(signal_profile, threshold_global, close_meta)

    config = {
        "experiment": "mixed_activity_return_extension_v6_4",
        "input_predictions": str(args.predictions),
        "close_return_meta": close_meta,
        "thresholds": {
            "high_conf_threshold": args.high_conf_threshold,
            "strong_threshold": args.strong_threshold,
            "balanced_up_edge": args.balanced_up_edge,
            "balanced_down_block": args.balanced_down_block,
            "defensive_conflict": args.defensive_conflict,
            "sensitivity_thresholds": thresholds,
        },
        "interpretation": {
            "score_percentile": "calibration-window percentile rank, not literal probability",
            "portfolio_allocation": False,
            "costs_turnover_mdd": False,
        },
    }
    summary = {
        "experiment": "mixed_activity_return_extension_v6_4",
        "prediction_rows": int(len(df)),
        "asset_count": int(df["asset_name"].nunique()),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "global_up_touch_rate": float(df["y_up_fixed"].mean()),
        "global_down_touch_rate": float(df["y_down_fixed"].mean()),
        "v6_4_signal_counts": df["v6_4_signal"].value_counts().to_dict(),
        "decision": decision,
        "decision_note": "This step audits mixed activity and return-side robustness. It is still not a portfolio backtest.",
    }
    outputs = {
        "summary": save_json(out_dir / "mixed_activity_audit_summary.json", summary),
        "config": save_json(out_dir / "mixed_activity_audit_config.json", config),
        "v6_4_signal_predictions": save_csv(out_dir / "v6_4_signal_predictions.csv", df),
        "v6_4_signal_profile": save_csv(out_dir / "v6_4_signal_profile.csv", signal_profile),
        "asset_v6_4_signal_profile": save_csv(out_dir / "asset_v6_4_signal_profile.csv", asset_profile),
        "annual_v6_4_signal_profile": save_csv(out_dir / "annual_v6_4_signal_profile.csv", annual_profile),
        "fold_v6_4_signal_profile": save_csv(out_dir / "fold_v6_4_signal_profile.csv", fold_profile),
        "mixed_activity_asset_audit": save_csv(out_dir / "mixed_activity_asset_audit.csv", mixed_asset),
        "mixed_activity_annual_audit": save_csv(out_dir / "mixed_activity_annual_audit.csv", mixed_annual),
        "mixed_activity_fold_audit": save_csv(out_dir / "mixed_activity_fold_audit.csv", mixed_fold),
        "high_confidence_vs_strong_profile": save_csv(out_dir / "high_confidence_vs_strong_profile.csv", high_strong_profile),
        "return_distribution_by_signal": save_csv(out_dir / "return_distribution_by_signal.csv", signal_profile),
        "return_score_threshold_sensitivity": save_csv(out_dir / "return_score_threshold_sensitivity.csv", threshold_global),
        "asset_return_score_threshold_sensitivity": save_csv(out_dir / "asset_return_score_threshold_sensitivity.csv", threshold_asset),
        "annual_return_score_threshold_sensitivity": save_csv(out_dir / "annual_return_score_threshold_sensitivity.csv", threshold_annual),
        "fold_return_score_threshold_sensitivity": save_csv(out_dir / "fold_return_score_threshold_sensitivity.csv", threshold_fold),
        "score_bin_return_distribution": save_csv(out_dir / "score_bin_return_distribution.csv", bin_profile),
    }
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to repaired_signal_predictions.csv")
    parser.add_argument("--output-dir", default="mixed_activity_return_extension_v6_4_output")
    parser.add_argument("--ohlcv-inputs", default="", help="Optional comma-separated OHLCV csv files")
    parser.add_argument("--asset-names", default="", help="Required with --ohlcv-inputs")
    parser.add_argument("--close-return-horizons", default="5,10,20")
    parser.add_argument("--high-conf-threshold", type=float, default=0.95)
    parser.add_argument("--strong-threshold", type=float, default=0.90)
    parser.add_argument("--balanced-up-edge", type=float, default=0.80)
    parser.add_argument("--balanced-down-block", type=float, default=0.80)
    parser.add_argument("--defensive-conflict", type=float, default=0.90)
    parser.add_argument("--thresholds", default="0.85,0.88,0.90,0.92,0.95")
    args = parser.parse_args()
    outputs = run(args)
    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    print("[OK] Mixed Activity Audit + Return Distribution Extension v6.4 completed.")
    print(json.dumps({
        "prediction_rows": summary["prediction_rows"],
        "asset_count": summary["asset_count"],
        "date_start": summary["date_start"],
        "date_end": summary["date_end"],
        "v6_4_signal_counts": summary["v6_4_signal_counts"],
        "decision": summary["decision"],
        "output_files": {k: str(v) for k, v in outputs.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
