# -*- coding: utf-8 -*-
"""
multihead_ablation_sweep_runner.py

Rolling Multi-head Regime Experiment Ablation Sweep Runner.

목적
----
rolling_multihead_regime_experiment.py를 여러 설정으로 반복 실행해
다음 질문에 답합니다.

1. h20_current vs vol_expansion_ratio 중 어느 HighVol label이 더 나은가?
2. HighVol persistence none / 2of3 / 3of5 중 어느 쪽이 나은가?
3. equity60_cash40 / equity70_cash30 / equity80_cash20 / bond overlay 중 어느 allocation이 나은가?
4. RiskOff hard trigger는 실제로 도움이 되는가?
5. Multi-head 구조가 Buy & Hold 대비 Calmar/MDD를 안정적으로 개선하는가?

전제
----
이 파일은 이전에 제공한 아래 파일이 같은 폴더에 있거나 --base-script로 지정되어 있어야 합니다.

- rolling_multihead_regime_experiment.py

실행 예시
--------
python multihead_ablation_sweep_runner.py ^
  --base-script rolling_multihead_regime_experiment.py ^
  --equity-input QQQ_ohlcv.csv ^
  --bond-input IEF_ohlcv.csv ^
  --ticker QQQ ^
  --bond-ticker IEF ^
  --output-dir multihead_ablation_results ^
  --transaction-cost-bps 10

빠른 실행:
python multihead_ablation_sweep_runner.py ^
  --base-script rolling_multihead_regime_experiment.py ^
  --equity-input QQQ_ohlcv.csv ^
  --bond-input IEF_ohlcv.csv ^
  --output-dir multihead_ablation_results_fast ^
  --highvol-label-modes h20_current,vol_expansion_ratio ^
  --highvol-expansion-mults 1.25 ^
  --highvol-persistences 2of3 ^
  --allocation-modes equity60_cash40,equity70_cash30,equity80_cash20 ^
  --riskoff-modes warning_only

출력 파일
---------
output_dir/
├─ multihead_ablation_summary.csv
├─ multihead_ablation_top20.csv
├─ multihead_ablation_head_summary.csv
├─ multihead_ablation_manifest.csv
└─ multihead_ablation_summary.json

주의
----
- 이 스크립트는 여러 번 학습을 수행하므로 시간이 걸립니다.
- 기본값은 RiskOff hard trigger를 제외하고 warning_only만 돌립니다.
- RiskOff hard trigger를 검증하려면 --riskoff-modes warning_only,hard_cash,hard_bond_cash 로 지정하세요.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def parse_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def save_json(path: str | Path, data: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def safe_read_json(path: str | Path) -> Dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


@dataclass
class SweepConfig:
    run_id: str
    highvol_label_mode: str
    highvol_expansion_mult: float
    highvol_threshold_quantile: float
    highvol_persistence: str
    allocation_mode: str
    riskoff_mode: str


def build_grid(
    highvol_label_modes: Sequence[str],
    highvol_expansion_mults: Sequence[float],
    highvol_threshold_quantiles: Sequence[float],
    highvol_persistences: Sequence[str],
    allocation_modes: Sequence[str],
    riskoff_modes: Sequence[str],
) -> List[SweepConfig]:
    rows: List[SweepConfig] = []

    for label_mode in highvol_label_modes:
        exp_values = highvol_expansion_mults if label_mode == "vol_expansion_ratio" else [np.nan]

        for exp_mult, q, pers, alloc, ro_mode in itertools.product(
            exp_values,
            highvol_threshold_quantiles,
            highvol_persistences,
            allocation_modes,
            riskoff_modes,
        ):
            exp_tag = f"em{exp_mult}".replace(".", "p") if label_mode == "vol_expansion_ratio" else "emNA"
            run_id = (
                f"{label_mode}_{exp_tag}_q{q}_{pers}_{alloc}_{ro_mode}"
                .replace(".", "p")
                .replace("/", "_")
            )

            rows.append(
                SweepConfig(
                    run_id=run_id,
                    highvol_label_mode=label_mode,
                    highvol_expansion_mult=float(exp_mult) if not pd.isna(exp_mult) else np.nan,
                    highvol_threshold_quantile=float(q),
                    highvol_persistence=pers,
                    allocation_mode=alloc,
                    riskoff_mode=ro_mode,
                )
            )

    return rows


def build_command(
    python_exe: str,
    base_script: Path,
    cfg: SweepConfig,
    args,
    run_output_dir: Path,
) -> List[str]:
    cmd = [
        python_exe,
        str(base_script),
        "--equity-input", args.equity_input,
        "--ticker", args.ticker,
        "--output-dir", str(run_output_dir),
        "--feature-set", args.feature_set,
        "--highvol-label-mode", cfg.highvol_label_mode,
        "--direction-horizon", str(args.direction_horizon),
        "--highvol-horizon", str(args.highvol_horizon),
        "--riskoff-horizon", str(args.riskoff_horizon),
        "--vol-window", str(args.vol_window),
        "--direction-k", str(args.direction_k),
        "--high-vol-quantile", str(args.high_vol_quantile),
        "--high-vol-lookback", str(args.high_vol_lookback),
        "--riskoff-k-mdd", str(args.riskoff_k_mdd),
        "--train-window", str(args.train_window),
        "--calibration-window", str(args.calibration_window),
        "--test-window", str(args.test_window),
        "--highvol-threshold-quantile", str(cfg.highvol_threshold_quantile),
        "--riskoff-threshold-quantile", str(args.riskoff_threshold_quantile),
        "--highvol-persistence", cfg.highvol_persistence,
        "--riskoff-persistence", args.riskoff_persistence,
        "--allocation-mode", cfg.allocation_mode,
        "--riskoff-mode", cfg.riskoff_mode,
        "--calibration-method", args.calibration_method,
        "--transaction-cost-bps", str(args.transaction_cost_bps),
        "--n-estimators", str(args.n_estimators),
        "--random-state", str(args.random_state),
    ]

    if args.bond_input:
        cmd.extend(["--bond-input", args.bond_input, "--bond-ticker", args.bond_ticker])

    if cfg.highvol_label_mode == "vol_expansion_ratio":
        cmd.extend(["--highvol-expansion-mult", str(cfg.highvol_expansion_mult)])

    return cmd


def extract_head_fold_summary(summary: Dict, head: str) -> Dict[str, float]:
    rows = summary.get("head_summary", {}).get("fold_head_summary", [])
    for r in rows:
        if r.get("head") == head:
            return r
    return {}


def extract_row_from_summary(cfg: SweepConfig, summary: Dict, run_output_dir: Path, elapsed_sec: float, returncode: int) -> Dict:
    best = summary.get("best_strategy_by_calmar", {}) or {}
    strategy_rows = summary.get("strategy_rows", []) or []

    # multihead row를 우선 사용
    multi = None
    for r in strategy_rows:
        if r.get("strategy") == "multihead_allocation":
            multi = r
            break
    if multi is None:
        multi = best

    head_summary = summary.get("head_summary", {}) or {}
    direction_overall = head_summary.get("direction_overall", {}) or {}
    highvol_overall = head_summary.get("highvol_overall", {}) or {}
    riskoff_overall = head_summary.get("riskoff_overall", {}) or {}

    highvol_fold = extract_head_fold_summary(summary, "highvol")
    riskoff_fold = extract_head_fold_summary(summary, "riskoff")
    direction_fold = extract_head_fold_summary(summary, "direction")

    calmar_diff = safe_float(multi.get("calmar_diff_vs_buy_hold"))
    mdd_diff = safe_float(multi.get("mdd_diff_vs_buy_hold"))
    cagr_diff = safe_float(multi.get("cagr_diff_vs_buy_hold"))

    # 보수적 후보 점수
    # - Calmar 개선
    # - MDD 개선
    # - CAGR 훼손 최소화
    # - HighVol fold 안정성
    # - RiskOff hard trigger는 기본적으로 감점
    riskoff_penalty = -0.05 if cfg.riskoff_mode != "warning_only" else 0.0
    candidate_score = (
        2.0 * np.nan_to_num(calmar_diff, nan=0.0)
        + 1.5 * np.nan_to_num(mdd_diff, nan=0.0)
        + 1.0 * np.nan_to_num(cagr_diff, nan=0.0)
        + 0.5 * np.nan_to_num(safe_float(highvol_fold.get("normal_polarity_rate")), nan=0.0)
        + 0.5 * np.nan_to_num(safe_float(highvol_fold.get("positive_brier_skill_rate")), nan=0.0)
        + riskoff_penalty
    )

    # Stable gate는 엄격하게 둔다.
    stable_candidate = (
        np.isfinite(calmar_diff) and calmar_diff > 0.03
        and np.isfinite(mdd_diff) and mdd_diff > 0.03
        and np.isfinite(cagr_diff) and cagr_diff > -0.02
        and safe_float(highvol_fold.get("normal_polarity_rate"), 0.0) >= 0.50
        and safe_float(highvol_fold.get("positive_brier_skill_rate"), 0.0) >= 0.20
    )

    return {
        **asdict(cfg),
        "returncode": returncode,
        "elapsed_sec": elapsed_sec,
        "run_output_dir": str(run_output_dir),
        "oos_start": summary.get("oos_start"),
        "oos_end": summary.get("oos_end"),
        "rows": summary.get("rows"),

        "multi_cagr": safe_float(multi.get("cagr")),
        "multi_mdd": safe_float(multi.get("mdd")),
        "multi_calmar": safe_float(multi.get("calmar")),
        "multi_sharpe": safe_float(multi.get("sharpe")),
        "multi_volatility": safe_float(multi.get("volatility")),
        "multi_total_return": safe_float(multi.get("total_return")),
        "multi_turnover_total": safe_float(multi.get("turnover_total")),
        "multi_transaction_cost_total": safe_float(multi.get("transaction_cost_total")),
        "multi_avg_equity_weight": safe_float(multi.get("avg_equity_weight")),
        "multi_avg_bond_weight": safe_float(multi.get("avg_bond_weight")),
        "multi_avg_cash_weight": safe_float(multi.get("avg_cash_weight")),

        "cagr_diff_vs_buy_hold": cagr_diff,
        "mdd_diff_vs_buy_hold": mdd_diff,
        "calmar_diff_vs_buy_hold": calmar_diff,
        "sharpe_diff_vs_buy_hold": safe_float(multi.get("sharpe_diff_vs_buy_hold")),
        "volatility_diff_vs_buy_hold": safe_float(multi.get("volatility_diff_vs_buy_hold")),
        "total_return_diff_vs_buy_hold": safe_float(multi.get("total_return_diff_vs_buy_hold")),

        "direction_macro_f1": safe_float(direction_overall.get("macro_f1")),
        "direction_balanced_accuracy": safe_float(direction_overall.get("balanced_accuracy")),
        "direction_entropy_mean": safe_float(direction_overall.get("direction_entropy_mean")),
        "direction_fold_mean_macro_f1": safe_float(direction_fold.get("mean_macro_f1")),
        "direction_fold_median_macro_f1": safe_float(direction_fold.get("median_macro_f1")),

        "highvol_pr_auc": safe_float(highvol_overall.get("pr_auc")),
        "highvol_pr_ratio": safe_float(highvol_overall.get("pr_ratio")),
        "highvol_brier_skill": safe_float(highvol_overall.get("brier_skill")),
        "highvol_ece": safe_float(highvol_overall.get("ece")),
        "highvol_positive_rate": safe_float(highvol_overall.get("positive_rate")),
        "highvol_fold_mean_pr_auc": safe_float(highvol_fold.get("mean_pr_auc")),
        "highvol_fold_median_pr_auc": safe_float(highvol_fold.get("median_pr_auc")),
        "highvol_fold_mean_brier_skill": safe_float(highvol_fold.get("mean_brier_skill")),
        "highvol_fold_median_brier_skill": safe_float(highvol_fold.get("median_brier_skill")),
        "highvol_fold_normal_polarity_rate": safe_float(highvol_fold.get("normal_polarity_rate")),
        "highvol_fold_positive_brier_skill_rate": safe_float(highvol_fold.get("positive_brier_skill_rate")),

        "riskoff_pr_auc": safe_float(riskoff_overall.get("pr_auc")),
        "riskoff_pr_ratio": safe_float(riskoff_overall.get("pr_ratio")),
        "riskoff_brier_skill": safe_float(riskoff_overall.get("brier_skill")),
        "riskoff_ece": safe_float(riskoff_overall.get("ece")),
        "riskoff_positive_rate": safe_float(riskoff_overall.get("positive_rate")),
        "riskoff_fold_mean_pr_auc": safe_float(riskoff_fold.get("mean_pr_auc")),
        "riskoff_fold_median_pr_auc": safe_float(riskoff_fold.get("median_pr_auc")),
        "riskoff_fold_normal_polarity_rate": safe_float(riskoff_fold.get("normal_polarity_rate")),
        "riskoff_fold_positive_brier_skill_rate": safe_float(riskoff_fold.get("positive_brier_skill_rate")),

        "candidate_score": float(candidate_score),
        "stable_candidate": bool(stable_candidate),
    }


def run_sweep(args) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_script = Path(args.base_script)
    if not base_script.exists():
        raise FileNotFoundError(
            f"base script not found: {base_script}\n"
            "Place rolling_multihead_regime_experiment.py in the current folder or pass --base-script."
        )

    grid = build_grid(
        highvol_label_modes=parse_list(args.highvol_label_modes),
        highvol_expansion_mults=parse_float_list(args.highvol_expansion_mults),
        highvol_threshold_quantiles=parse_float_list(args.highvol_threshold_quantiles),
        highvol_persistences=parse_list(args.highvol_persistences),
        allocation_modes=parse_list(args.allocation_modes),
        riskoff_modes=parse_list(args.riskoff_modes),
    )

    if args.max_runs and args.max_runs > 0:
        grid = grid[: args.max_runs]

    manifest_rows = []
    result_rows = []

    python_exe = sys.executable if args.python_exe == "" else args.python_exe

    for i, cfg in enumerate(grid, start=1):
        run_output_dir = output_dir / "runs" / cfg.run_id
        summary_path = run_output_dir / "multihead_regime_summary.json"

        cmd = build_command(
            python_exe=python_exe,
            base_script=base_script,
            cfg=cfg,
            args=args,
            run_output_dir=run_output_dir,
        )

        manifest_row = {
            "run_index": i,
            "run_id": cfg.run_id,
            "cmd": " ".join(cmd),
            **asdict(cfg),
        }

        print(f"[{i}/{len(grid)}] running: {cfg.run_id}")

        start = time.time()
        returncode = None
        stdout_tail = ""
        stderr_tail = ""

        if args.dry_run:
            returncode = 0
            elapsed = 0.0
        elif args.resume and summary_path.exists():
            returncode = 0
            elapsed = 0.0
            print(f"  [SKIP] summary exists: {summary_path}")
        else:
            run_output_dir.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            returncode = proc.returncode
            elapsed = time.time() - start
            stdout_tail = proc.stdout[-2000:]
            stderr_tail = proc.stderr[-2000:]

            (run_output_dir / "runner_stdout.txt").write_text(proc.stdout, encoding="utf-8")
            (run_output_dir / "runner_stderr.txt").write_text(proc.stderr, encoding="utf-8")

            if returncode != 0:
                print(f"  [ERROR] returncode={returncode}")
                print(stderr_tail[-1000:])
            else:
                print(f"  [OK] elapsed={elapsed:.1f}s")

        manifest_row.update({
            "returncode": returncode,
            "elapsed_sec": elapsed,
            "summary_path": str(summary_path),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        })
        manifest_rows.append(manifest_row)

        summary = safe_read_json(summary_path)
        if summary:
            result_rows.append(extract_row_from_summary(cfg, summary, run_output_dir, elapsed, returncode))
        else:
            result_rows.append({
                **asdict(cfg),
                "returncode": returncode,
                "elapsed_sec": elapsed,
                "run_output_dir": str(run_output_dir),
                "candidate_score": np.nan,
                "stable_candidate": False,
            })

        # 중간 저장
        pd.DataFrame(manifest_rows).to_csv(output_dir / "multihead_ablation_manifest.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(result_rows).to_csv(output_dir / "multihead_ablation_summary.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(result_rows)

    # dry-run / failed-run에서도 정렬 컬럼이 항상 존재하도록 보강
    for col, default in [
        ("stable_candidate", False),
        ("candidate_score", np.nan),
        ("multi_calmar", np.nan),
    ]:
        if col not in summary_df.columns:
            summary_df[col] = default

    summary_df = summary_df.sort_values(
        ["stable_candidate", "candidate_score", "multi_calmar"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    top20 = summary_df.head(20).copy()

    # head summary long format
    head_rows = []
    for _, row in summary_df.iterrows():
        for head in ["direction", "highvol", "riskoff"]:
            head_rows.append({
                "run_id": row.get("run_id"),
                "highvol_label_mode": row.get("highvol_label_mode"),
                "highvol_expansion_mult": row.get("highvol_expansion_mult"),
                "highvol_threshold_quantile": row.get("highvol_threshold_quantile"),
                "highvol_persistence": row.get("highvol_persistence"),
                "allocation_mode": row.get("allocation_mode"),
                "riskoff_mode": row.get("riskoff_mode"),
                "head": head,
                "macro_f1": row.get("direction_macro_f1") if head == "direction" else np.nan,
                "balanced_accuracy": row.get("direction_balanced_accuracy") if head == "direction" else np.nan,
                "pr_auc": row.get(f"{head}_pr_auc") if head != "direction" else np.nan,
                "pr_ratio": row.get(f"{head}_pr_ratio") if head != "direction" else np.nan,
                "brier_skill": row.get(f"{head}_brier_skill") if head != "direction" else np.nan,
                "ece": row.get(f"{head}_ece") if head != "direction" else np.nan,
                "fold_mean_pr_auc": row.get(f"{head}_fold_mean_pr_auc") if head != "direction" else np.nan,
                "fold_median_pr_auc": row.get(f"{head}_fold_median_pr_auc") if head != "direction" else np.nan,
                "fold_normal_polarity_rate": row.get(f"{head}_fold_normal_polarity_rate") if head != "direction" else np.nan,
                "fold_positive_brier_skill_rate": row.get(f"{head}_fold_positive_brier_skill_rate") if head != "direction" else np.nan,
            })

    head_df = pd.DataFrame(head_rows)

    manifest_df = pd.DataFrame(manifest_rows)

    summary_path = output_dir / "multihead_ablation_summary.csv"
    top20_path = output_dir / "multihead_ablation_top20.csv"
    head_path = output_dir / "multihead_ablation_head_summary.csv"
    manifest_path = output_dir / "multihead_ablation_manifest.csv"
    json_path = output_dir / "multihead_ablation_summary.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    top20.to_csv(top20_path, index=False, encoding="utf-8-sig")
    head_df.to_csv(head_path, index=False, encoding="utf-8-sig")
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    best = top20.head(1).to_dict("records")[0] if not top20.empty else None
    stable_count = int(summary_df["stable_candidate"].sum()) if "stable_candidate" in summary_df.columns else 0

    json_summary = {
        "experiment": "multihead_ablation_sweep",
        "base_script": str(base_script),
        "grid_count": int(len(grid)),
        "completed_count": int((summary_df["returncode"] == 0).sum()) if "returncode" in summary_df.columns else None,
        "stable_candidate_count": stable_count,
        "best_candidate": best,
        "top20": top20.to_dict("records"),
        "decision_note": (
            "Use stable_candidate as a conservative gate. "
            "If no stable candidate exists, keep Multi-head as output framework and do not promote to Stable."
        ),
        "output_files": {
            "summary": str(summary_path),
            "top20": str(top20_path),
            "head_summary": str(head_path),
            "manifest": str(manifest_path),
        },
    }
    save_json(json_path, json_summary)

    return {
        "summary": summary_path,
        "top20": top20_path,
        "head_summary": head_path,
        "manifest": manifest_path,
        "json": json_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-script", default="rolling_multihead_regime_experiment.py")
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--equity-input", required=False, default="")
    parser.add_argument("--bond-input", default="")
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--bond-ticker", default="IEF")
    parser.add_argument("--output-dir", default="multihead_ablation_results")

    parser.add_argument("--feature-set", default="down_core")
    parser.add_argument("--highvol-label-modes", default="h20_current,vol_expansion_ratio")
    parser.add_argument("--highvol-expansion-mults", default="1.25,1.5")
    parser.add_argument("--highvol-threshold-quantiles", default="0.75,0.80")
    parser.add_argument("--highvol-persistences", default="none,2of3,3of5")
    parser.add_argument("--allocation-modes", default="equity60_cash40,equity70_cash30,equity80_cash20,equity80_bond20,equity70_bond20_cash10")
    parser.add_argument("--riskoff-modes", default="warning_only")

    parser.add_argument("--direction-horizon", type=int, default=20)
    parser.add_argument("--highvol-horizon", type=int, default=20)
    parser.add_argument("--riskoff-horizon", type=int, default=40)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--direction-k", type=float, default=0.25)
    parser.add_argument("--high-vol-quantile", type=float, default=0.75)
    parser.add_argument("--high-vol-lookback", type=int, default=252)
    parser.add_argument("--riskoff-k-mdd", type=float, default=2.0)

    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--riskoff-threshold-quantile", type=float, default=0.80)
    parser.add_argument("--riskoff-persistence", default="none")
    parser.add_argument("--calibration-method", default="sigmoid")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not args.dry_run and not args.equity_input:
        raise ValueError("--equity-input is required unless --dry-run is used")

    outputs = run_sweep(args)
    summary = safe_read_json(outputs["json"])
    best = summary.get("best_candidate") or {}

    print("[OK] Multi-head ablation sweep completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print(json.dumps(
        {
            "grid_count": summary.get("grid_count"),
            "completed_count": summary.get("completed_count"),
            "stable_candidate_count": summary.get("stable_candidate_count"),
            "best_run_id": best.get("run_id"),
            "best_label_mode": best.get("highvol_label_mode"),
            "best_expansion_mult": best.get("highvol_expansion_mult"),
            "best_threshold_quantile": best.get("highvol_threshold_quantile"),
            "best_persistence": best.get("highvol_persistence"),
            "best_allocation_mode": best.get("allocation_mode"),
            "best_riskoff_mode": best.get("riskoff_mode"),
            "best_cagr": best.get("multi_cagr"),
            "best_mdd": best.get("multi_mdd"),
            "best_calmar": best.get("multi_calmar"),
            "best_calmar_diff_vs_buy_hold": best.get("calmar_diff_vs_buy_hold"),
            "best_candidate_score": best.get("candidate_score"),
            "stable_candidate": best.get("stable_candidate"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
