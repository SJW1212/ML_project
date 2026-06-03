# -*- coding: utf-8 -*-
"""
regime_timing_attribution_validator.py

Dual-HighVol 후보의 "구간 특이 timing 의존성"을 검증하는 코드.

목적
----
cross-asset 검증 이후 제기된 핵심 의문:

1. QQQ 성과가 2020 COVID 구간에 과도하게 의존하는가?
2. 신호가 실제 drawdown 시작 전에 발화했는가, 아니면 우연히 겹쳤는가?
3. PR-AUC가 실제 positive-rate baseline 대비 유의미한가?
4. Brier skill 양수 fold 비율이 구조적으로 낮은가?
5. Economic gate 기준은 명시적으로 어떤 조건으로 통과/실패하는가?

입력 파일
---------
- dual_highvol_oos_predictions.csv
- dual_highvol_strategy_daily_returns.csv
- dual_highvol_fold_metrics.csv

선택 입력
---------
- dual_highvol_strategy_summary.csv

실행 예시
--------
python regime_timing_attribution_validator.py ^
  --predictions dual_highvol_oos_predictions.csv ^
  --strategy-daily dual_highvol_strategy_daily_returns.csv ^
  --fold-metrics dual_highvol_fold_metrics.csv ^
  --strategy-summary dual_highvol_strategy_summary.csv ^
  --output-dir regime_timing_validation

COVID 제외 분석만 빠르게:
python regime_timing_attribution_validator.py ^
  --strategy-daily dual_highvol_strategy_daily_returns.csv ^
  --fold-metrics dual_highvol_fold_metrics.csv ^
  --output-dir regime_timing_validation

출력 파일
---------
output_dir/
├─ timing_validation_summary.json
├─ period_exclusion_metrics.csv
├─ drawdown_event_map.csv
├─ drawdown_event_summary.csv
├─ fold_pr_brier_diagnostics.csv
├─ signal_timing_summary.csv
└─ economic_gate_recheck.csv

주의
----
- 이 코드는 새 파라미터 최적화가 아닙니다.
- 현재 후보의 성과 원인이 "예측력"인지 "특정 구간 timing"인지 진단합니다.
- Brier skill 양수 비율 binomial test는 엄밀한 모델 적합성 검정이 아니라 보수적 진단 지표입니다.
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

def save_csv(path: str | Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_json(path: str | Path, data: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=to_jsonable), encoding="utf-8")
    return path


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


def parse_periods(value: str) -> List[Dict[str, object]]:
    """
    Format:
    COVID:2020-02-01:2020-04-30,BEAR2022:2022-01-01:2022-12-31
    """
    if not value:
        return []
    periods = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) != 3:
            raise ValueError(f"invalid period spec: {part}")
        name, start, end = pieces
        periods.append({
            "name": name,
            "start": pd.to_datetime(start),
            "end": pd.to_datetime(end),
        })
    return periods


def exact_binom_cdf_leq(k: int, n: int, p: float = 0.5) -> float:
    """
    P[X <= k], X~Binomial(n,p)
    Used as one-sided test for "positive brier skill rate lower than 50%".
    """
    if n <= 0:
        return np.nan
    k = int(k)
    n = int(n)
    prob = 0.0
    for i in range(0, k + 1):
        prob += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return float(prob)


# ============================================================
# 1. Performance
# ============================================================

def performance_metrics(returns: pd.Series, dates: Optional[pd.Series] = None, periods_per_year: int = 252) -> Dict[str, float]:
    ret = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(ret) < 2:
        return {
            "rows": int(len(ret)),
            "total_return": np.nan,
            "cagr": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "mdd": np.nan,
            "calmar": np.nan,
        }

    curve = (1.0 + ret).cumprod()
    total_return = float(curve.iloc[-1] - 1.0)

    if dates is not None and len(dates) == len(ret):
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
        "rows": int(len(ret)),
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
    }


def drawdown_series(returns: pd.Series) -> pd.Series:
    ret = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    curve = (1.0 + ret).cumprod()
    return curve / curve.cummax() - 1.0


def recompute_strategy_return(df: pd.DataFrame, cost_bps: float = 10.0) -> pd.Series:
    required = {"equity_weight", "bond_weight", "cash_weight", "equity_ret", "bond_ret", "cash_ret", "turnover"}
    if required.issubset(set(df.columns)):
        gross = (
            pd.to_numeric(df["equity_weight"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df["equity_ret"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["bond_weight"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df["bond_ret"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["cash_weight"], errors="coerce").fillna(0.0)
            * pd.to_numeric(df["cash_ret"], errors="coerce").fillna(0.0)
        )
        cost = pd.to_numeric(df["turnover"], errors="coerce").fillna(0.0) * (cost_bps / 10000.0)
        return gross - cost

    if "strategy_ret" in df.columns:
        return pd.to_numeric(df["strategy_ret"], errors="coerce").fillna(0.0)

    raise ValueError("Cannot compute strategy return. Missing weights/returns or strategy_ret.")


# ============================================================
# 2. Filtering
# ============================================================

def filter_candidate_daily(
    strategy_daily: pd.DataFrame,
    hybrid_mode: str,
    persistence_mode: str,
    defensive_equity_weight: float,
    defense_asset: str,
    riskoff_mode: str,
) -> pd.DataFrame:
    d = strategy_daily.copy()
    mask = d["strategy"].astype(str).eq("dual_highvol_hybrid")

    if "hybrid_mode" in d.columns:
        mask &= d["hybrid_mode"].astype(str).eq(hybrid_mode)
    if "persistence_mode" in d.columns:
        mask &= d["persistence_mode"].astype(str).eq(persistence_mode)
    if "defense_asset" in d.columns:
        mask &= d["defense_asset"].astype(str).eq(defense_asset)
    if "riskoff_mode" in d.columns:
        mask &= d["riskoff_mode"].astype(str).eq(riskoff_mode)
    if "defensive_equity_weight" in d.columns:
        mask &= np.isclose(pd.to_numeric(d["defensive_equity_weight"], errors="coerce"), defensive_equity_weight, atol=1e-9)

    out = d[mask].sort_values("date").reset_index(drop=True)
    if out.empty:
        # fallback: best available dual_highvol rows
        out = d[d["strategy"].astype(str).eq("dual_highvol_hybrid")].sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("candidate daily rows not found")
    return out


def filter_benchmark(strategy_daily: pd.DataFrame, strategy: str = "buy_hold") -> pd.DataFrame:
    out = strategy_daily[strategy_daily["strategy"].astype(str).eq(strategy)].copy()
    out = out.sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"benchmark rows not found: {strategy}")
    return out


# ============================================================
# 3. Period exclusion
# ============================================================

def period_exclusion_analysis(
    candidate: pd.DataFrame,
    buyhold: pd.DataFrame,
    periods: List[Dict[str, object]],
    cost_bps: float,
) -> pd.DataFrame:
    c = candidate[["date"]].copy()
    c["candidate_ret"] = recompute_strategy_return(candidate, cost_bps)

    b = buyhold[["date"]].copy()
    b["buy_hold_ret"] = recompute_strategy_return(buyhold, 0.0)

    df = c.merge(b, on="date", how="inner").sort_values("date").reset_index(drop=True)

    rows = []

    def add_row(label: str, mask: pd.Series, note: str):
        g = df[mask].copy()
        cm = performance_metrics(g["candidate_ret"], g["date"])
        bm = performance_metrics(g["buy_hold_ret"], g["date"])

        row = {"sample": label, "note": note}
        row.update({f"candidate_{k}": v for k, v in cm.items()})
        row.update({f"buy_hold_{k}": v for k, v in bm.items()})

        for k in ["total_return", "cagr", "mdd", "calmar", "sharpe", "volatility"]:
            row[f"{k}_diff_vs_buy_hold"] = row.get(f"candidate_{k}", np.nan) - row.get(f"buy_hold_{k}", np.nan)

        row["economic_gate"] = (
            row.get("calmar_diff_vs_buy_hold", np.nan) > 0.03
            and row.get("mdd_diff_vs_buy_hold", np.nan) > 0.03
            and row.get("cagr_diff_vs_buy_hold", np.nan) > -0.02
        )
        rows.append(row)

    full_mask = pd.Series(True, index=df.index)
    add_row("full_sample", full_mask, "all OOS rows")

    for p in periods:
        name = str(p["name"])
        start = pd.to_datetime(p["start"])
        end = pd.to_datetime(p["end"])
        period_mask = (df["date"] >= start) & (df["date"] <= end)

        add_row(f"only_{name}", period_mask, f"only {name}: {start.date()}~{end.date()}")
        add_row(f"exclude_{name}", ~period_mask, f"exclude {name}: {start.date()}~{end.date()}")
        add_row(f"pre_{name}", df["date"] < start, f"before {name}")
        add_row(f"post_{name}", df["date"] > end, f"after {name}")

    return pd.DataFrame(rows)


# ============================================================
# 4. Drawdown event mapping
# ============================================================

def find_drawdown_events(df: pd.DataFrame, threshold: float) -> List[Dict[str, object]]:
    """
    Contiguous periods where buy_hold_dd <= threshold.
    """
    event = df["buy_hold_dd"] <= threshold
    events: List[Dict[str, object]] = []

    in_event = False
    start_idx = None

    for i, flag in enumerate(event.to_numpy()):
        if flag and not in_event:
            in_event = True
            start_idx = i
        if in_event and ((not flag) or i == len(event) - 1):
            end_idx = i - 1 if not flag else i
            g = df.iloc[start_idx:end_idx + 1]
            if not g.empty:
                bottom_idx = g["buy_hold_dd"].idxmin()
                events.append({
                    "event_start_idx": int(start_idx),
                    "event_end_idx": int(end_idx),
                    "event_start": g["date"].iloc[0],
                    "event_end": g["date"].iloc[-1],
                    "event_days": int(len(g)),
                    "buy_hold_bottom_date": df.loc[bottom_idx, "date"],
                    "buy_hold_min_dd": float(g["buy_hold_dd"].min()),
                    "candidate_min_dd": float(g["candidate_dd"].min()),
                })
            in_event = False
            start_idx = None

    return events


def drawdown_event_mapping(
    candidate: pd.DataFrame,
    buyhold: pd.DataFrame,
    thresholds: List[float],
    pre_windows: List[int],
    cost_bps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    c = candidate[["date"]].copy()
    c["candidate_ret"] = recompute_strategy_return(candidate, cost_bps)
    c["candidate_dd"] = drawdown_series(c["candidate_ret"]).to_numpy()

    # signal columns
    if "executed_signal" in candidate.columns:
        c["executed_signal"] = pd.to_numeric(candidate["executed_signal"], errors="coerce").fillna(0).astype(int)
    else:
        c["executed_signal"] = 0

    for col in ["raw_hybrid_signal", "persistent_signal", "h20_signal_raw", "expansion_signal_raw", "expansion_confirm_raw"]:
        if col in candidate.columns:
            c[col] = pd.to_numeric(candidate[col], errors="coerce").fillna(0).astype(int)

    b = buyhold[["date"]].copy()
    b["buy_hold_ret"] = recompute_strategy_return(buyhold, 0.0)
    b["buy_hold_dd"] = drawdown_series(b["buy_hold_ret"]).to_numpy()

    df = c.merge(b, on="date", how="inner").sort_values("date").reset_index(drop=True)

    rows = []
    summary_rows = []

    for th in thresholds:
        events = find_drawdown_events(df, th)
        for j, ev in enumerate(events):
            start_idx = ev["event_start_idx"]
            end_idx = ev["event_end_idx"]
            g = df.iloc[start_idx:end_idx + 1].copy()
            signal = g["executed_signal"] == 1

            row = {
                "threshold": th,
                "event_id": j,
                **ev,
                "signal_days_during_event": int(signal.sum()),
                "signal_coverage_during_event": safe_divide(int(signal.sum()), len(g)),
                "candidate_total_return_during_event": float((1.0 + g["candidate_ret"]).prod() - 1.0),
                "buy_hold_total_return_during_event": float((1.0 + g["buy_hold_ret"]).prod() - 1.0),
                "candidate_minus_buy_hold_return_during_event": float((1.0 + g["candidate_ret"]).prod() - (1.0 + g["buy_hold_ret"]).prod()),
                "candidate_avg_ret_during_event": float(g["candidate_ret"].mean()),
                "buy_hold_avg_ret_during_event": float(g["buy_hold_ret"].mean()),
            }

            for w in pre_windows:
                pre_start = max(0, start_idx - w)
                pre_g = df.iloc[pre_start:start_idx].copy()
                pre_signal_days = int((pre_g["executed_signal"] == 1).sum()) if not pre_g.empty else 0
                row[f"pre{w}_signal_days"] = pre_signal_days
                row[f"pre{w}_signal_rate"] = safe_divide(pre_signal_days, len(pre_g))
                if pre_signal_days > 0:
                    first_signal_date = pre_g.loc[pre_g["executed_signal"] == 1, "date"].iloc[0]
                    row[f"pre{w}_first_signal_date"] = first_signal_date
                    row[f"pre{w}_first_signal_lead_days"] = int((pd.to_datetime(ev["event_start"]) - pd.to_datetime(first_signal_date)).days)
                else:
                    row[f"pre{w}_first_signal_date"] = pd.NaT
                    row[f"pre{w}_first_signal_lead_days"] = np.nan

            rows.append(row)

        # aggregate threshold-level
        event_mask = df["buy_hold_dd"] <= th
        signal_mask = df["executed_signal"] == 1
        summary_rows.append({
            "threshold": th,
            "event_count": int(len(events)),
            "event_days": int(event_mask.sum()),
            "event_day_rate": float(event_mask.mean()),
            "signal_days_total": int(signal_mask.sum()),
            "signal_day_rate": float(signal_mask.mean()),
            "signal_days_in_event": int((event_mask & signal_mask).sum()),
            "signal_coverage_in_event": safe_divide(int((event_mask & signal_mask).sum()), int(event_mask.sum())),
            "signal_days_outside_event": int((~event_mask & signal_mask).sum()),
            "false_alarm_like_rate": safe_divide(int((~event_mask & signal_mask).sum()), int((~event_mask).sum())),
            "avg_candidate_ret_event_days": float(df.loc[event_mask, "candidate_ret"].mean()) if event_mask.any() else np.nan,
            "avg_buy_hold_ret_event_days": float(df.loc[event_mask, "buy_hold_ret"].mean()) if event_mask.any() else np.nan,
            "avg_excess_ret_event_days": float((df.loc[event_mask, "candidate_ret"] - df.loc[event_mask, "buy_hold_ret"]).mean()) if event_mask.any() else np.nan,
        })

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


# ============================================================
# 5. Fold diagnostics
# ============================================================

def fold_pr_brier_diagnostics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()

    rows = []
    for head, g0 in fold_metrics.groupby("head"):
        g = g0.copy()

        if "pr_auc" not in g.columns or "positive_rate" not in g.columns:
            continue

        g["pr_auc"] = pd.to_numeric(g["pr_auc"], errors="coerce")
        g["positive_rate"] = pd.to_numeric(g["positive_rate"], errors="coerce")
        g["brier_skill"] = pd.to_numeric(g.get("brier_skill", np.nan), errors="coerce")

        valid = g[g["pr_auc"].notna() & g["positive_rate"].notna()].copy()
        if valid.empty:
            continue

        valid["pr_lift_vs_fold_baseline"] = valid["pr_auc"] - valid["positive_rate"]
        valid["pr_ratio_vs_fold_baseline"] = valid["pr_auc"] / valid["positive_rate"].replace(0, np.nan)

        n_brier = int(g["brier_skill"].notna().sum())
        k_brier = int((g["brier_skill"] > 0).sum())
        brier_positive_rate = safe_divide(k_brier, n_brier)
        p_value_lower_than_half = exact_binom_cdf_leq(k_brier, n_brier, p=0.5) if n_brier > 0 else np.nan

        rows.append({
            "head": head,
            "fold_count": int(len(g)),
            "valid_pr_folds": int(len(valid)),
            "mean_positive_rate": float(valid["positive_rate"].mean()),
            "median_positive_rate": float(valid["positive_rate"].median()),
            "mean_pr_auc": float(valid["pr_auc"].mean()),
            "median_pr_auc": float(valid["pr_auc"].median()),
            "mean_pr_lift_vs_fold_baseline": float(valid["pr_lift_vs_fold_baseline"].mean()),
            "median_pr_lift_vs_fold_baseline": float(valid["pr_lift_vs_fold_baseline"].median()),
            "positive_pr_lift_rate": float((valid["pr_lift_vs_fold_baseline"] > 0).mean()),
            "mean_pr_ratio_vs_fold_baseline": float(valid["pr_ratio_vs_fold_baseline"].replace([np.inf, -np.inf], np.nan).mean()),
            "median_pr_ratio_vs_fold_baseline": float(valid["pr_ratio_vs_fold_baseline"].replace([np.inf, -np.inf], np.nan).median()),
            "brier_valid_folds": n_brier,
            "positive_brier_skill_folds": k_brier,
            "positive_brier_skill_rate": brier_positive_rate,
            "binom_p_value_positive_brier_skill_rate_lt_0_5": p_value_lower_than_half,
            "diagnostic_note": (
                "Binomial test is a conservative diagnostic, not a full statistical proof of model invalidity."
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. Signal timing summary
# ============================================================

def signal_timing_summary(candidate: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    d = candidate.copy()
    d["candidate_ret"] = recompute_strategy_return(d, cost_bps)
    d["signal"] = pd.to_numeric(d.get("executed_signal", 0), errors="coerce").fillna(0).astype(int)

    rows = []
    for signal_value, g in d.groupby("signal"):
        rows.append({
            "executed_signal": int(signal_value),
            "days": int(len(g)),
            "day_rate": float(len(g) / len(d)),
            "avg_candidate_ret": float(g["candidate_ret"].mean()),
            "median_candidate_ret": float(g["candidate_ret"].median()),
            "avg_equity_ret": float(pd.to_numeric(g["equity_ret"], errors="coerce").mean()) if "equity_ret" in g.columns else np.nan,
            "median_equity_ret": float(pd.to_numeric(g["equity_ret"], errors="coerce").median()) if "equity_ret" in g.columns else np.nan,
            "avg_equity_weight": float(pd.to_numeric(g["equity_weight"], errors="coerce").mean()) if "equity_weight" in g.columns else np.nan,
            "avg_cash_weight": float(pd.to_numeric(g["cash_weight"], errors="coerce").mean()) if "cash_weight" in g.columns else np.nan,
            "avg_turnover": float(pd.to_numeric(g["turnover"], errors="coerce").mean()) if "turnover" in g.columns else np.nan,
            "worst_candidate_day": float(g["candidate_ret"].min()),
            "worst_equity_day": float(pd.to_numeric(g["equity_ret"], errors="coerce").min()) if "equity_ret" in g.columns else np.nan,
            "best_candidate_day": float(g["candidate_ret"].max()),
            "best_equity_day": float(pd.to_numeric(g["equity_ret"], errors="coerce").max()) if "equity_ret" in g.columns else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. Economic gate
# ============================================================

def economic_gate_recheck(period_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in period_df.iterrows():
        rows.append({
            "sample": r.get("sample"),
            "cagr_diff_vs_buy_hold": r.get("cagr_diff_vs_buy_hold"),
            "mdd_diff_vs_buy_hold": r.get("mdd_diff_vs_buy_hold"),
            "calmar_diff_vs_buy_hold": r.get("calmar_diff_vs_buy_hold"),
            "rule_cagr": "cagr_diff > -0.02",
            "rule_mdd": "mdd_diff > 0.03",
            "rule_calmar": "calmar_diff > 0.03",
            "cagr_gate": safe_float(r.get("cagr_diff_vs_buy_hold")) > -0.02,
            "mdd_gate": safe_float(r.get("mdd_diff_vs_buy_hold")) > 0.03,
            "calmar_gate": safe_float(r.get("calmar_diff_vs_buy_hold")) > 0.03,
            "economic_gate": bool(r.get("economic_gate", False)),
        })
    return pd.DataFrame(rows)


# ============================================================
# 8. Runner
# ============================================================

def run(args) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_daily = load_csv(args.strategy_daily)
    fold_metrics = load_csv(args.fold_metrics)
    predictions = load_csv(args.predictions, required=False)
    strategy_summary = load_csv(args.strategy_summary, required=False)

    candidate = filter_candidate_daily(
        strategy_daily,
        hybrid_mode=args.hybrid_mode,
        persistence_mode=args.persistence_mode,
        defensive_equity_weight=args.defensive_equity_weight,
        defense_asset=args.defense_asset,
        riskoff_mode=args.riskoff_mode,
    )
    buyhold = filter_benchmark(strategy_daily, "buy_hold")

    periods = parse_periods(args.periods)
    thresholds = [float(x) for x in args.drawdown_thresholds.split(",") if x.strip()]
    pre_windows = [int(x) for x in args.pre_windows.split(",") if x.strip()]

    period_df = period_exclusion_analysis(candidate, buyhold, periods, args.cost_bps)
    event_map_df, event_summary_df = drawdown_event_mapping(candidate, buyhold, thresholds, pre_windows, args.cost_bps)
    fold_diag_df = fold_pr_brier_diagnostics(fold_metrics)
    signal_df = signal_timing_summary(candidate, args.cost_bps)
    gate_df = economic_gate_recheck(period_df)

    outputs = {
        "period_exclusion": save_csv(output_dir / "period_exclusion_metrics.csv", period_df),
        "drawdown_event_map": save_csv(output_dir / "drawdown_event_map.csv", event_map_df),
        "drawdown_event_summary": save_csv(output_dir / "drawdown_event_summary.csv", event_summary_df),
        "fold_pr_brier_diagnostics": save_csv(output_dir / "fold_pr_brier_diagnostics.csv", fold_diag_df),
        "signal_timing_summary": save_csv(output_dir / "signal_timing_summary.csv", signal_df),
        "economic_gate_recheck": save_csv(output_dir / "economic_gate_recheck.csv", gate_df),
    }

    # decision logic
    full_row = period_df[period_df["sample"].eq("full_sample")].head(1).to_dict("records")
    covid_excluded = period_df[period_df["sample"].str.startswith("exclude_COVID", na=False)].head(1).to_dict("records")

    covid_dependency = None
    if full_row and covid_excluded:
        full_calmar = safe_float(full_row[0].get("calmar_diff_vs_buy_hold"))
        excl_calmar = safe_float(covid_excluded[0].get("calmar_diff_vs_buy_hold"))
        full_cagr = safe_float(full_row[0].get("cagr_diff_vs_buy_hold"))
        excl_cagr = safe_float(covid_excluded[0].get("cagr_diff_vs_buy_hold"))
        covid_dependency = {
            "full_calmar_diff": full_calmar,
            "exclude_covid_calmar_diff": excl_calmar,
            "calmar_diff_drop": full_calmar - excl_calmar,
            "full_cagr_diff": full_cagr,
            "exclude_covid_cagr_diff": excl_cagr,
            "cagr_diff_drop": full_cagr - excl_cagr,
            "covid_dependency_flag": bool(excl_calmar < 0.03 or excl_cagr < -0.02),
        }

    # event timing flag
    severe = event_summary_df[event_summary_df["threshold"] <= -0.10].copy() if not event_summary_df.empty else pd.DataFrame()
    avg_coverage = float(severe["signal_coverage_in_event"].mean()) if not severe.empty else np.nan
    avg_false_alarm = float(severe["false_alarm_like_rate"].mean()) if not severe.empty else np.nan

    decision = {
        "candidate_config": {
            "hybrid_mode": args.hybrid_mode,
            "persistence_mode": args.persistence_mode,
            "defensive_equity_weight": args.defensive_equity_weight,
            "defense_asset": args.defense_asset,
            "riskoff_mode": args.riskoff_mode,
        },
        "period": {
            "start": str(pd.to_datetime(candidate["date"]).min().date()),
            "end": str(pd.to_datetime(candidate["date"]).max().date()),
            "rows": int(len(candidate)),
        },
        "covid_dependency": covid_dependency,
        "drawdown_timing": {
            "avg_signal_coverage_in_drawdown_events_threshold_le_10pct": avg_coverage,
            "avg_false_alarm_like_rate_threshold_le_10pct": avg_false_alarm,
        },
        "fold_diagnostics": fold_diag_df.to_dict("records"),
        "interpretation": {
            "if_exclude_covid_fails": "QQQ result may be regime-specific and should not be promoted.",
            "if_drawdown_coverage_low": "Signal does not reliably anticipate actual drawdown events.",
            "if_pr_lift_negative": "Head ranking quality is below prevalence baseline.",
            "if_brier_skill_rate_low": "Do not use probabilities as calibrated confidence or allocation size.",
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
    }

    outputs["summary"] = save_json(output_dir / "timing_validation_summary.json", decision)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="dual_highvol_oos_predictions.csv")
    parser.add_argument("--strategy-daily", default="dual_highvol_strategy_daily_returns.csv")
    parser.add_argument("--fold-metrics", default="dual_highvol_fold_metrics.csv")
    parser.add_argument("--strategy-summary", default="dual_highvol_strategy_summary.csv")
    parser.add_argument("--output-dir", default="regime_timing_validation")

    parser.add_argument("--hybrid-mode", default="h20_with_expansion_confirm")
    parser.add_argument("--persistence-mode", default="3of5")
    parser.add_argument("--defensive-equity-weight", type=float, default=0.60)
    parser.add_argument("--defense-asset", default="cash")
    parser.add_argument("--riskoff-mode", default="warning_only")

    parser.add_argument("--periods", default="COVID:2020-02-01:2020-04-30")
    parser.add_argument("--drawdown-thresholds", default="-0.05,-0.10,-0.15,-0.20")
    parser.add_argument("--pre-windows", default="5,10,20")
    parser.add_argument("--cost-bps", type=float, default=10.0)

    args = parser.parse_args()
    outputs = run(args)

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    print("[OK] Regime timing attribution validation completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print(json.dumps(
        {
            "period_start": summary["period"]["start"],
            "period_end": summary["period"]["end"],
            "rows": summary["period"]["rows"],
            "covid_dependency": summary.get("covid_dependency"),
            "drawdown_timing": summary.get("drawdown_timing"),
            "output_files": summary.get("outputs"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
