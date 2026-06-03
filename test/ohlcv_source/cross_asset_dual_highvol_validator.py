# -*- coding: utf-8 -*-
"""
cross_asset_dual_highvol_validator.py

Dual-HighVol Provisional Candidate Cross-Asset Validator.

목적
----
dual_highvol_candidate_validator.py 결과에서 현재 후보는 다음 상태입니다.

- economic gate: pass
- cost sensitivity gate: pass
- neighborhood gate: pass
- head gate: pass
- yearly gate: fail
- final decision: provisional_stable_candidate

따라서 다음 검증은 파라미터 추가 튜닝이 아니라,
동일 후보를 여러 자산/기간에 반복 적용해 일반화 여부를 확인하는 것입니다.

검증 대상 후보 기본값
---------------------
- hybrid_mode: h20_with_expansion_confirm
- persistence_mode: 3of5
- defensive_equity_weight: 0.60
- defense_asset: cash
- riskoff_mode: warning_only
- h20_threshold_quantile: 0.75
- expansion_threshold_quantile: 0.75
- expansion_confirm_quantile: 0.50
- expansion_mult: 1.25

전제
----
이 스크립트는 이전에 제공한 아래 파일을 반복 실행합니다.

- dual_highvol_hybrid_sweep.py

실행 예시
--------
python cross_asset_dual_highvol_validator.py ^
  --base-script dual_highvol_hybrid_sweep.py ^
  --equity-files QQQ_ohlcv.csv,SPY_ohlcv.csv,SOXX_ohlcv.csv,XLK_ohlcv.csv ^
  --equity-names QQQ,SPY,SOXX,XLK ^
  --bond-input IEF_ohlcv.csv ^
  --output-dir cross_asset_dual_highvol_validation ^
  --transaction-cost-bps 10 ^
  --resume

단일 자산 장기 검증
------------------
python cross_asset_dual_highvol_validator.py ^
  --base-script dual_highvol_hybrid_sweep.py ^
  --equity-files QQQ_ohlcv_2008.csv ^
  --equity-names QQQ_2008 ^
  --bond-input IEF_ohlcv_2008.csv ^
  --output-dir qqq_2008_dual_highvol_validation ^
  --train-window 1260 ^
  --calibration-window 252 ^
  --test-window 63 ^
  --transaction-cost-bps 10 ^
  --resume

출력 파일
---------
output_dir/
├─ cross_asset_validation_summary.csv
├─ cross_asset_validation_topline.csv
├─ cross_asset_validation_decision.json
├─ cross_asset_validation_manifest.csv
└─ runs/
   └─ 각 자산별 dual_highvol_hybrid_sweep 출력 폴더

주의
----
- 이 코드는 후보 검증용입니다.
- 각 자산별로 동일 후보 설정만 실행합니다.
- 종목별 최적화/전수탐색을 하지 않으므로 overfitting 위험을 줄입니다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# 0. Utils
# ============================================================

def parse_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


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


def first_matching_row(rows: List[Dict], **criteria) -> Optional[Dict]:
    for row in rows:
        ok = True
        for k, v in criteria.items():
            if str(row.get(k)) != str(v):
                ok = False
                break
        if ok:
            return row
    return None


# ============================================================
# 1. Config
# ============================================================

@dataclass
class AssetRun:
    asset_name: str
    equity_file: str
    run_output_dir: str
    returncode: int
    elapsed_sec: float
    summary_path: str
    stdout_tail: str = ""
    stderr_tail: str = ""


# ============================================================
# 2. Build command
# ============================================================

def build_command(args, asset_name: str, equity_file: str, run_output_dir: Path) -> List[str]:
    python_exe = sys.executable if args.python_exe == "" else args.python_exe

    cmd = [
        python_exe,
        str(Path(args.base_script)),
        "--equity-input", equity_file,
        "--ticker", asset_name,
        "--output-dir", str(run_output_dir),
        "--feature-set", args.feature_set,
        "--direction-horizon", str(args.direction_horizon),
        "--highvol-horizon", str(args.highvol_horizon),
        "--riskoff-horizon", str(args.riskoff_horizon),
        "--vol-window", str(args.vol_window),
        "--direction-k", str(args.direction_k),
        "--high-vol-quantile", str(args.high_vol_quantile),
        "--high-vol-lookback", str(args.high_vol_lookback),
        "--expansion-mult", str(args.expansion_mult),
        "--riskoff-k-mdd", str(args.riskoff_k_mdd),
        "--train-window", str(args.train_window),
        "--calibration-window", str(args.calibration_window),
        "--test-window", str(args.test_window),
        "--h20-threshold-quantile", str(args.h20_threshold_quantile),
        "--expansion-threshold-quantile", str(args.expansion_threshold_quantile),
        "--expansion-confirm-quantile", str(args.expansion_confirm_quantile),
        "--riskoff-threshold-quantile", str(args.riskoff_threshold_quantile),
        "--hybrid-modes", args.hybrid_mode,
        "--persistence-modes", args.persistence_mode,
        "--defensive-equity-weights", str(args.defensive_equity_weight),
        "--defense-assets", args.defense_asset,
        "--riskoff-modes", args.riskoff_mode,
        "--calibration-method", args.calibration_method,
        "--transaction-cost-bps", str(args.transaction_cost_bps),
        "--n-estimators", str(args.n_estimators),
        "--random-state", str(args.random_state),
    ]

    if args.bond_input:
        cmd.extend(["--bond-input", args.bond_input, "--bond-ticker", args.bond_ticker])

    return cmd


# ============================================================
# 3. Extract summary
# ============================================================

def extract_asset_result(asset_name: str, equity_file: str, summary: Dict, run_output_dir: str, returncode: int, elapsed_sec: float) -> Dict:
    best = summary.get("best_candidate") or {}

    head_rows = summary.get("head_summary", []) or []
    head_map = {str(r.get("head")): r for r in head_rows}

    def head_value(head: str, key: str):
        return safe_float(head_map.get(head, {}).get(key))

    return {
        "asset_name": asset_name,
        "equity_file": equity_file,
        "returncode": returncode,
        "elapsed_sec": elapsed_sec,
        "run_output_dir": run_output_dir,

        "oos_start": summary.get("oos_start"),
        "oos_end": summary.get("oos_end"),
        "rows": summary.get("rows"),

        "strategy": best.get("strategy"),
        "hybrid_mode": best.get("hybrid_mode"),
        "persistence_mode": best.get("persistence_mode"),
        "defensive_equity_weight": safe_float(best.get("defensive_equity_weight")),
        "defense_asset": best.get("defense_asset"),
        "riskoff_mode": best.get("riskoff_mode"),
        "raw_signal_rate": safe_float(best.get("raw_signal_rate")),
        "executed_signal_rate": safe_float(best.get("executed_signal_rate")),
        "avg_equity_weight": safe_float(best.get("avg_equity_weight")),
        "avg_bond_weight": safe_float(best.get("avg_bond_weight")),
        "avg_cash_weight": safe_float(best.get("avg_cash_weight")),
        "turnover_total": safe_float(best.get("turnover_total")),
        "transaction_cost_total": safe_float(best.get("transaction_cost_total")),

        "total_return": safe_float(best.get("total_return")),
        "cagr": safe_float(best.get("cagr")),
        "mdd": safe_float(best.get("mdd")),
        "calmar": safe_float(best.get("calmar")),
        "sharpe": safe_float(best.get("sharpe")),
        "volatility": safe_float(best.get("volatility")),

        "total_return_diff_vs_buy_hold": safe_float(best.get("total_return_diff_vs_buy_hold")),
        "cagr_diff_vs_buy_hold": safe_float(best.get("cagr_diff_vs_buy_hold")),
        "mdd_diff_vs_buy_hold": safe_float(best.get("mdd_diff_vs_buy_hold")),
        "calmar_diff_vs_buy_hold": safe_float(best.get("calmar_diff_vs_buy_hold")),
        "sharpe_diff_vs_buy_hold": safe_float(best.get("sharpe_diff_vs_buy_hold")),
        "volatility_diff_vs_buy_hold": safe_float(best.get("volatility_diff_vs_buy_hold")),
        "candidate_score": safe_float(best.get("candidate_score")),
        "stable_economic_gate": bool(best.get("stable_economic_gate", False)),

        "direction_mean_macro_f1": head_value("direction", "mean_macro_f1"),
        "direction_mean_balanced_accuracy": head_value("direction", "mean_balanced_accuracy"),

        "h20_mean_pr_auc": head_value("highvol_h20", "mean_pr_auc"),
        "h20_median_pr_auc": head_value("highvol_h20", "median_pr_auc"),
        "h20_normal_polarity_rate": head_value("highvol_h20", "normal_polarity_rate"),
        "h20_positive_brier_skill_rate": head_value("highvol_h20", "positive_brier_skill_rate"),

        "expansion_mean_pr_auc": head_value("highvol_expansion", "mean_pr_auc"),
        "expansion_median_pr_auc": head_value("highvol_expansion", "median_pr_auc"),
        "expansion_normal_polarity_rate": head_value("highvol_expansion", "normal_polarity_rate"),
        "expansion_positive_brier_skill_rate": head_value("highvol_expansion", "positive_brier_skill_rate"),

        "riskoff_mean_pr_auc": head_value("riskoff", "mean_pr_auc"),
        "riskoff_median_pr_auc": head_value("riskoff", "median_pr_auc"),
        "riskoff_normal_polarity_rate": head_value("riskoff", "normal_polarity_rate"),
        "riskoff_positive_brier_skill_rate": head_value("riskoff", "positive_brier_skill_rate"),
    }


def build_decision(summary_df: pd.DataFrame) -> Dict:
    # failed-run / dry-run에서도 비교 컬럼이 항상 존재하도록 보강
    df = summary_df.copy()
    required_cols = {
        "returncode": 1,
        "stable_economic_gate": False,
        "calmar_diff_vs_buy_hold": np.nan,
        "mdd_diff_vs_buy_hold": np.nan,
        "cagr_diff_vs_buy_hold": np.nan,
        "candidate_score": np.nan,
    }
    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default

    ok = df[df["returncode"] == 0].copy()

    # summary가 없는 dry-run은 성공 실행으로 보지 않음
    if "oos_start" in ok.columns:
        ok = ok[ok["oos_start"].notna()].copy()

    if ok.empty:
        return {
            "decision": "no_successful_runs",
            "reason": "No asset run produced a valid summary. In --dry-run mode this is expected.",
            "asset_count": 0,
            "economic_pass_rate": np.nan,
            "calmar_positive_rate": np.nan,
            "mdd_positive_rate": np.nan,
            "cagr_gate_rate": np.nan,
            "cagr_positive_rate": np.nan,
            "avg_calmar_diff_vs_buy_hold": np.nan,
            "avg_mdd_diff_vs_buy_hold": np.nan,
            "avg_cagr_diff_vs_buy_hold": np.nan,
        }

    asset_count = int(len(ok))
    economic_pass_rate = float(ok["stable_economic_gate"].fillna(False).astype(bool).mean())
    calmar_positive_rate = float((ok["calmar_diff_vs_buy_hold"] > 0).mean())
    mdd_positive_rate = float((ok["mdd_diff_vs_buy_hold"] > 0).mean())
    cagr_gate_rate = float((ok["cagr_diff_vs_buy_hold"] > -0.02).mean())
    cagr_positive_rate = float((ok["cagr_diff_vs_buy_hold"] > 0).mean())

    avg_calmar_diff = float(ok["calmar_diff_vs_buy_hold"].mean())
    avg_mdd_diff = float(ok["mdd_diff_vs_buy_hold"].mean())
    avg_cagr_diff = float(ok["cagr_diff_vs_buy_hold"].mean())

    # Conservative gates
    multi_asset_gate = (
        asset_count >= 3
        and economic_pass_rate >= 0.60
        and calmar_positive_rate >= 0.60
        and mdd_positive_rate >= 0.60
        and cagr_gate_rate >= 0.60
        and avg_calmar_diff > 0.03
        and avg_mdd_diff > 0.02
    )

    if multi_asset_gate:
        decision = "multi_asset_provisional_stable"
        reason = (
            "Candidate passed broad cross-asset economic criteria. "
            "Still requires long-history and after-cost stress validation before final Stable."
        )
    elif economic_pass_rate >= 0.50 and mdd_positive_rate >= 0.50:
        decision = "candidate_remains_asset_sensitive"
        reason = (
            "Candidate works on some assets but not enough for broad Stable. "
            "Keep as QQQ-centric or asset-specific candidate."
        )
    else:
        decision = "reject_generalization"
        reason = (
            "Candidate did not generalize across tested assets."
        )

    return {
        "decision": decision,
        "reason": reason,
        "asset_count": asset_count,
        "economic_pass_rate": economic_pass_rate,
        "calmar_positive_rate": calmar_positive_rate,
        "mdd_positive_rate": mdd_positive_rate,
        "cagr_gate_rate": cagr_gate_rate,
        "cagr_positive_rate": cagr_positive_rate,
        "avg_calmar_diff_vs_buy_hold": avg_calmar_diff,
        "avg_mdd_diff_vs_buy_hold": avg_mdd_diff,
        "avg_cagr_diff_vs_buy_hold": avg_cagr_diff,
        "best_asset_by_calmar_diff": ok.sort_values("calmar_diff_vs_buy_hold", ascending=False).head(1).to_dict("records"),
        "worst_asset_by_calmar_diff": ok.sort_values("calmar_diff_vs_buy_hold", ascending=True).head(1).to_dict("records"),
    }


# ============================================================
# 4. Runner
# ============================================================

def run_cross_asset_validation(args) -> Dict[str, Path]:
    base_script = Path(args.base_script)
    if not base_script.exists():
        raise FileNotFoundError(
            f"base script not found: {base_script}\n"
            "Place dual_highvol_hybrid_sweep.py in the current folder or pass --base-script."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_files = parse_list(args.equity_files)
    equity_names = parse_list(args.equity_names)

    if not equity_files:
        raise ValueError("--equity-files is required")

    if equity_names and len(equity_names) != len(equity_files):
        raise ValueError("--equity-names length must match --equity-files length")

    if not equity_names:
        equity_names = [Path(p).stem.replace("_ohlcv", "") for p in equity_files]

    manifest_rows = []
    result_rows = []

    for i, (asset_name, equity_file) in enumerate(zip(equity_names, equity_files), start=1):
        run_output_dir = output_dir / "runs" / asset_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_output_dir / "dual_highvol_summary.json"

        cmd = build_command(args, asset_name, equity_file, run_output_dir)

        print(f"[{i}/{len(equity_files)}] running asset={asset_name}")
        print("  " + " ".join(cmd))

        start = time.time()
        stdout_tail = ""
        stderr_tail = ""

        if args.resume and summary_path.exists():
            returncode = 0
            elapsed = 0.0
            print(f"  [SKIP] existing summary: {summary_path}")
        elif args.dry_run:
            returncode = 0
            elapsed = 0.0
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            returncode = proc.returncode
            elapsed = time.time() - start
            stdout_tail = proc.stdout[-2500:]
            stderr_tail = proc.stderr[-2500:]

            (run_output_dir / "runner_stdout.txt").write_text(proc.stdout, encoding="utf-8")
            (run_output_dir / "runner_stderr.txt").write_text(proc.stderr, encoding="utf-8")

            if returncode == 0:
                print(f"  [OK] elapsed={elapsed:.1f}s")
            else:
                print(f"  [ERROR] returncode={returncode}")
                print(stderr_tail)

        run_info = AssetRun(
            asset_name=asset_name,
            equity_file=equity_file,
            run_output_dir=str(run_output_dir),
            returncode=int(returncode),
            elapsed_sec=float(elapsed),
            summary_path=str(summary_path),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
        manifest_rows.append(asdict(run_info))

        summary = safe_read_json(summary_path)
        if summary:
            result_rows.append(extract_asset_result(asset_name, equity_file, summary, str(run_output_dir), returncode, elapsed))
        else:
            result_rows.append({
                "asset_name": asset_name,
                "equity_file": equity_file,
                "returncode": returncode,
                "elapsed_sec": elapsed,
                "run_output_dir": str(run_output_dir),
                "stable_economic_gate": False,
            })

        # incremental save
        pd.DataFrame(manifest_rows).to_csv(output_dir / "cross_asset_validation_manifest.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(result_rows).to_csv(output_dir / "cross_asset_validation_summary.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(result_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    # Sort
    for col in ["stable_economic_gate", "candidate_score", "calmar_diff_vs_buy_hold"]:
        if col not in summary_df.columns:
            summary_df[col] = np.nan
    summary_df = summary_df.sort_values(
        ["stable_economic_gate", "candidate_score", "calmar_diff_vs_buy_hold"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    decision = build_decision(summary_df)

    # Topline by metric
    topline_rows = []
    ok = summary_df[summary_df["returncode"] == 0].copy()
    if not ok.empty:
        for col, default in [
            ("stable_economic_gate", False),
            ("calmar_diff_vs_buy_hold", np.nan),
            ("mdd_diff_vs_buy_hold", np.nan),
            ("cagr_diff_vs_buy_hold", np.nan),
            ("executed_signal_rate", np.nan),
            ("turnover_total", np.nan),
        ]:
            if col not in ok.columns:
                ok[col] = default

        topline_rows.extend([
            {"metric": "asset_count", "value": len(ok)},
            {"metric": "economic_pass_rate", "value": float(ok["stable_economic_gate"].fillna(False).astype(bool).mean())},
            {"metric": "calmar_positive_rate", "value": float((ok["calmar_diff_vs_buy_hold"] > 0).mean())},
            {"metric": "mdd_positive_rate", "value": float((ok["mdd_diff_vs_buy_hold"] > 0).mean())},
            {"metric": "cagr_gate_rate", "value": float((ok["cagr_diff_vs_buy_hold"] > -0.02).mean())},
            {"metric": "avg_cagr_diff_vs_buy_hold", "value": float(ok["cagr_diff_vs_buy_hold"].mean())},
            {"metric": "avg_mdd_diff_vs_buy_hold", "value": float(ok["mdd_diff_vs_buy_hold"].mean())},
            {"metric": "avg_calmar_diff_vs_buy_hold", "value": float(ok["calmar_diff_vs_buy_hold"].mean())},
            {"metric": "avg_signal_rate", "value": float(ok["executed_signal_rate"].mean())},
            {"metric": "avg_turnover_total", "value": float(ok["turnover_total"].mean())},
        ])

    summary_path = output_dir / "cross_asset_validation_summary.csv"
    manifest_path = output_dir / "cross_asset_validation_manifest.csv"
    topline_path = output_dir / "cross_asset_validation_topline.csv"
    decision_path = output_dir / "cross_asset_validation_decision.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(topline_rows).to_csv(topline_path, index=False, encoding="utf-8-sig")
    save_json(decision_path, {
        "experiment": "cross_asset_dual_highvol_validation",
        "candidate_config": {
            "hybrid_mode": args.hybrid_mode,
            "persistence_mode": args.persistence_mode,
            "defensive_equity_weight": args.defensive_equity_weight,
            "defense_asset": args.defense_asset,
            "riskoff_mode": args.riskoff_mode,
            "h20_threshold_quantile": args.h20_threshold_quantile,
            "expansion_threshold_quantile": args.expansion_threshold_quantile,
            "expansion_confirm_quantile": args.expansion_confirm_quantile,
            "expansion_mult": args.expansion_mult,
        },
        "decision": decision,
        "summary_rows": summary_df.to_dict("records"),
        "output_files": {
            "summary": str(summary_path),
            "manifest": str(manifest_path),
            "topline": str(topline_path),
            "decision": str(decision_path),
        },
    })

    return {
        "summary": summary_path,
        "manifest": manifest_path,
        "topline": topline_path,
        "decision": decision_path,
    }


# ============================================================
# 5. CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-script", default="dual_highvol_hybrid_sweep.py")
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--equity-files", default="")
    parser.add_argument("--equity-names", default="")
    parser.add_argument("--bond-input", default="")
    parser.add_argument("--bond-ticker", default="IEF")
    parser.add_argument("--output-dir", default="cross_asset_dual_highvol_validation")

    # Fixed provisional candidate config
    parser.add_argument("--hybrid-mode", default="h20_with_expansion_confirm")
    parser.add_argument("--persistence-mode", default="3of5")
    parser.add_argument("--defensive-equity-weight", type=float, default=0.60)
    parser.add_argument("--defense-asset", default="cash")
    parser.add_argument("--riskoff-mode", default="warning_only")

    # Base model config
    parser.add_argument("--feature-set", default="down_core")
    parser.add_argument("--direction-horizon", type=int, default=20)
    parser.add_argument("--highvol-horizon", type=int, default=20)
    parser.add_argument("--riskoff-horizon", type=int, default=40)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--direction-k", type=float, default=0.25)
    parser.add_argument("--high-vol-quantile", type=float, default=0.75)
    parser.add_argument("--high-vol-lookback", type=int, default=252)
    parser.add_argument("--expansion-mult", type=float, default=1.25)
    parser.add_argument("--riskoff-k-mdd", type=float, default=2.0)

    # Rolling / threshold config
    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--calibration-window", type=int, default=252)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--h20-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--expansion-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--expansion-confirm-quantile", type=float, default=0.50)
    parser.add_argument("--riskoff-threshold-quantile", type=float, default=0.80)

    parser.add_argument("--calibration-method", default="sigmoid")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    outputs = run_cross_asset_validation(args)
    decision = safe_read_json(outputs["decision"]).get("decision", {})

    print("[OK] Cross-asset Dual-HighVol validation completed.")
    print(f"[OK] Output dir: {Path(args.output_dir).resolve()}")
    print(json.dumps(
        {
            "decision": decision.get("decision"),
            "reason": decision.get("reason"),
            "asset_count": decision.get("asset_count"),
            "economic_pass_rate": decision.get("economic_pass_rate"),
            "calmar_positive_rate": decision.get("calmar_positive_rate"),
            "mdd_positive_rate": decision.get("mdd_positive_rate"),
            "cagr_gate_rate": decision.get("cagr_gate_rate"),
            "avg_cagr_diff_vs_buy_hold": decision.get("avg_cagr_diff_vs_buy_hold"),
            "avg_mdd_diff_vs_buy_hold": decision.get("avg_mdd_diff_vs_buy_hold"),
            "avg_calmar_diff_vs_buy_hold": decision.get("avg_calmar_diff_vs_buy_hold"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
