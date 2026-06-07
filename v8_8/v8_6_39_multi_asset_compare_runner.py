"""
v8.6.39 Multi-Asset Compare Runner
==================================

목적
- xgb_recency_weighted_v8_6_39.py를 여러 종목에 대해 순차 실행
- 각 종목별 summary.json / latest.json을 수집
- 비교용 CSV와 Markdown 리포트 생성

실행 예시
1) ETF 기본 비교
    python v8_6_39_multi_asset_compare_runner.py ^
      --model xgb_recency_weighted_v8_6_39.py ^
      --preset etf ^
      --speed-profile fast ^
      --h10-down-only

2) 직접 종목 지정
    python v8_6_39_multi_asset_compare_runner.py ^
      --model xgb_recency_weighted_v8_6_39.py ^
      --asset-list QQQ,SPY,IWM,DIA,XLK,SMH,SOXX,NVDA,MSFT,AAPL,TSLA ^
      --speed-profile fast ^
      --h10-down-only

3) 이미 실행된 결과만 재집계
    python v8_6_39_multi_asset_compare_runner.py ^
      --collect-only ^
      --root results_v8_6_39_multi_asset_compare
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PRESETS = {
    "etf": ["QQQ", "SPY", "IWM", "DIA", "XLK", "SMH", "SOXX", "XLY", "XLF", "XLV"],
    "mega": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO"],
    "mixed": ["QQQ", "SPY", "IWM", "DIA", "XLK", "SMH", "SOXX", "XLY", "XLF", "XLV", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO"],
    "semis": ["QQQ", "SMH", "SOXX", "NVDA", "AVGO", "AMD", "TSM", "ASML", "QCOM", "MU"],
}


def safe_ticker(ticker: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(ticker)).strip("_") or "asset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run and compare v8.6.39 model across multiple assets.")
    p.add_argument("--model", type=str, default="xgb_recency_weighted_v8_6_39.py", help="v8.6.39 원본 모델 파일 경로")
    p.add_argument("--root", type=str, default="results_v8_6_39_multi_asset_compare", help="결과 저장 루트 폴더")
    p.add_argument("--asset-list", type=str, default=None, help="콤마 구분 종목. 예: QQQ,SPY,NVDA")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="mixed", help="기본 종목 프리셋")
    p.add_argument("--speed-profile", choices=["fast", "balanced", "full"], default="fast")
    p.add_argument("--h10-down-only", action="store_true")
    p.add_argument("--allow-cash-download-fallback", action="store_true")
    p.add_argument("--collect-only", action="store_true", help="모델 실행 없이 기존 결과만 집계")
    p.add_argument("--extra-args", type=str, default="", help="원본 모델에 그대로 넘길 추가 인자 문자열")
    return p.parse_args()


def get_tickers(args: argparse.Namespace) -> List[str]:
    tickers: List[str] = []
    if args.preset:
        tickers.extend(PRESETS[args.preset])
    if args.asset_list:
        tickers.extend([x.strip().upper() for x in args.asset_list.split(",") if x.strip()])
    return list(dict.fromkeys(tickers))


def run_model_for_tickers(args: argparse.Namespace, tickers: List[str]) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        out_dir = root / safe_ticker(ticker)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(model_path),
            "--target-ticker",
            ticker,
            "--speed-profile",
            args.speed_profile,
            "--result-dir",
            str(out_dir),
        ]
        if args.h10_down_only:
            cmd.append("--h10-down-only")
        if args.allow_cash_download_fallback:
            cmd.append("--allow-cash-download-fallback")
        if args.extra_args.strip():
            cmd.extend(args.extra_args.strip().split())

        print(f"\n[RUN] {ticker}")
        print(" ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"[WARN] {ticker} failed with returncode={proc.returncode}")


def _get_nested(d: Dict, path: str, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _pct(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x) * 100.0
    except Exception:
        return None


def _raw(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def collect_results(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for summary_path in sorted(root.glob("*/*_summary.json")):
        ticker_dir = summary_path.parent
        ticker = ticker_dir.name.upper()

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                sm = json.load(f)
        except Exception as exc:
            rows.append({"ticker": ticker, "status": f"summary_read_failed: {exc}", "result_dir": str(ticker_dir)})
            continue

        latest = {}
        latest_candidates = list(ticker_dir.glob("*_latest.json"))
        if latest_candidates:
            try:
                with open(latest_candidates[0], "r", encoding="utf-8") as f:
                    latest = json.load(f)
            except Exception:
                latest = {}

        strat = _get_nested(sm, "performance.strategy_after_cost", {})
        gross = _get_nested(sm, "performance.strategy_gross", {})
        bh = _get_nested(sm, "performance.stock_buy_hold", {})
        bench6040 = _get_nested(sm, "performance.benchmark_60_40", {})
        static = _get_nested(sm, "performance.static_50_30_20", {})

        row = {
            "ticker": str(sm.get("target_ticker", ticker)).upper(),
            "status": "ok",
            "result_dir": str(ticker_dir),
            "period_start": _get_nested(sm, "period.start"),
            "period_end": _get_nested(sm, "period.end"),
            "rows": _get_nested(sm, "period.rows"),
            "feature_count": sm.get("feature_count"),

            "strategy_final_capital": _raw(strat.get("final_capital")),
            "strategy_cagr_pct": _pct(strat.get("cagr")),
            "strategy_mdd_pct": _pct(strat.get("mdd")),
            "strategy_sharpe": _raw(strat.get("sharpe")),
            "strategy_sortino": _raw(strat.get("sortino")),
            "strategy_calmar": _raw(strat.get("calmar")),

            "gross_cagr_pct": _pct(gross.get("cagr")),
            "cost_drag_cagr_pctp": None,

            "buyhold_cagr_pct": _pct(bh.get("cagr")),
            "buyhold_mdd_pct": _pct(bh.get("mdd")),
            "buyhold_sharpe": _raw(bh.get("sharpe")),
            "buyhold_calmar": _raw(bh.get("calmar")),

            "benchmark_6040_cagr_pct": _pct(bench6040.get("cagr")),
            "benchmark_6040_mdd_pct": _pct(bench6040.get("mdd")),
            "benchmark_6040_sharpe": _raw(bench6040.get("sharpe")),
            "static_50_30_20_cagr_pct": _pct(static.get("cagr")),
            "static_50_30_20_mdd_pct": _pct(static.get("mdd")),

            "avg_stock_weight_pct": _pct(_get_nested(sm, "average_weights.avg_stock_weight")),
            "min_stock_weight_pct": _pct(_get_nested(sm, "average_weights.min_stock_weight")),
            "max_stock_weight_pct": _pct(_get_nested(sm, "average_weights.max_stock_weight")),
            "annual_turnover_x": _raw(_get_nested(sm, "turnover.annual_turnover_estimate")),
            "trade_executed_ratio_pct": _pct(_get_nested(sm, "turnover.trade_executed_ratio")),
            "emergency_rebalance_ratio_pct": _pct(_get_nested(sm, "turnover.emergency_rebalance_ratio")),

            "offensive_activation_rate_pct": _pct(_get_nested(sm, "direction_strength_specialist.offensive_activation_rate")),
            "tier3_rate_pct": _pct(_get_nested(sm, "direction_strength_specialist.offensive_tier_3_rate")),
            "full_stock_rate_pct": _pct(_get_nested(sm, "direction_strength_specialist.full_stock_signal_rate")),

            "latest_date": latest.get("date"),
            "latest_pred_risk": latest.get("pred_risk"),
            "latest_prob_normal_pct": latest.get("prob_normal"),
            "latest_prob_high_vol_pct": latest.get("prob_high_vol"),
            "latest_up_strength_score_pct": latest.get("prob_up_strengthening_score"),
            "latest_down_strength_score_pct": latest.get("prob_down_strengthening_score"),
            "latest_signal_regime": latest.get("signal_regime"),
            "latest_allocation_regime": latest.get("allocation_regime"),
            "latest_stock_pct": _get_nested(latest, "executed_allocation.stock"),
            "latest_bond_pct": _get_nested(latest, "executed_allocation.bond"),
            "latest_cash_pct": _get_nested(latest, "executed_allocation.cash"),
        }

        if row["strategy_cagr_pct"] is not None and row["gross_cagr_pct"] is not None:
            row["cost_drag_cagr_pctp"] = float(row["gross_cagr_pct"]) - float(row["strategy_cagr_pct"])

        if row["strategy_cagr_pct"] is not None and row["buyhold_cagr_pct"] is not None:
            row["excess_cagr_vs_buyhold_pctp"] = float(row["strategy_cagr_pct"]) - float(row["buyhold_cagr_pct"])
        else:
            row["excess_cagr_vs_buyhold_pctp"] = None

        if row["strategy_cagr_pct"] is not None and row["benchmark_6040_cagr_pct"] is not None:
            row["excess_cagr_vs_6040_pctp"] = float(row["strategy_cagr_pct"]) - float(row["benchmark_6040_cagr_pct"])
        else:
            row["excess_cagr_vs_6040_pctp"] = None

        if row["strategy_mdd_pct"] is not None and row["buyhold_mdd_pct"] is not None:
            # 양수면 모델이 Buy&Hold보다 MDD를 덜 맞은 것.
            row["mdd_improvement_vs_buyhold_pctp"] = abs(float(row["buyhold_mdd_pct"])) - abs(float(row["strategy_mdd_pct"]))
        else:
            row["mdd_improvement_vs_buyhold_pctp"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # rank-based composite score: 종목 간 상대 비교용. 절대 성능 채택 기준이 아님.
    ok = df[df["status"] == "ok"].copy()
    if not ok.empty:
        def rank_pct(col: str, ascending: bool) -> pd.Series:
            return ok[col].astype(float).rank(pct=True, ascending=ascending)

        score = pd.Series(0.0, index=ok.index)
        score += 0.30 * rank_pct("strategy_cagr_pct", ascending=True)
        score += 0.25 * rank_pct("strategy_sharpe", ascending=True)
        score += 0.20 * rank_pct("strategy_calmar", ascending=True)
        score += 0.15 * rank_pct("strategy_mdd_pct", ascending=False)  # MDD는 덜 음수일수록 좋음
        score += 0.10 * rank_pct("excess_cagr_vs_6040_pctp", ascending=True)
        ok["model_suitability_score_0_100"] = (score * 100.0).round(2)
        df = df.merge(ok[["ticker", "model_suitability_score_0_100"]], on="ticker", how="left")
    else:
        df["model_suitability_score_0_100"] = None

    sort_cols = ["model_suitability_score_0_100", "strategy_sharpe", "strategy_cagr_pct"]
    return df.sort_values(sort_cols, ascending=[False, False, False]).reset_index(drop=True)


def build_markdown_report(df: pd.DataFrame, root: Path) -> str:
    if df.empty:
        return "# v8.6.39 Multi-Asset Comparison\n\n결과 파일을 찾지 못했습니다.\n"

    ok = df[df["status"] == "ok"].copy()
    lines: List[str] = []
    lines.append("# v8.6.39 Multi-Asset Comparison")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- 분석 종목 수: {len(ok)}")
    if not ok.empty:
        best = ok.iloc[0]
        lines.append(f"- 상대 종합 점수 1위: **{best['ticker']}**")
        lines.append(f"- 최고 CAGR: **{ok.sort_values('strategy_cagr_pct', ascending=False).iloc[0]['ticker']}**")
        lines.append(f"- 최고 Sharpe: **{ok.sort_values('strategy_sharpe', ascending=False).iloc[0]['ticker']}**")
        lines.append(f"- 최저 MDD: **{ok.sort_values('strategy_mdd_pct', ascending=False).iloc[0]['ticker']}**")
    lines.append("")
    lines.append("## Ranked Table")
    lines.append("")
    cols = [
        "ticker",
        "model_suitability_score_0_100",
        "strategy_cagr_pct",
        "strategy_mdd_pct",
        "strategy_sharpe",
        "strategy_calmar",
        "buyhold_cagr_pct",
        "buyhold_mdd_pct",
        "excess_cagr_vs_buyhold_pctp",
        "excess_cagr_vs_6040_pctp",
        "mdd_improvement_vs_buyhold_pctp",
        "avg_stock_weight_pct",
        "annual_turnover_x",
        "latest_stock_pct",
        "latest_prob_high_vol_pct",
    ]
    show = ok[[c for c in cols if c in ok.columns]].copy()
    for c in show.columns:
        if c != "ticker" and pd.api.types.is_numeric_dtype(show[c]):
            show[c] = show[c].round(2)
    try:
        lines.append(show.to_markdown(index=False))
    except Exception:
        lines.append("Markdown table generation failed. Use CSV output instead.")
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- `model_suitability_score_0_100`: 종목 간 상대 비교용 점수입니다. 절대 채택 기준이 아닙니다.")
    lines.append("- `excess_cagr_vs_buyhold_pctp`: 전략 CAGR - Buy&Hold CAGR입니다.")
    lines.append("- `mdd_improvement_vs_buyhold_pctp`: 양수면 Buy&Hold보다 낙폭이 작다는 뜻입니다.")
    lines.append("- 이 모델은 QQQ 기반으로 튜닝된 구조이므로, 개별주에서는 성과가 더 불안정할 수 있습니다.")
    lines.append("")
    lines.append(f"결과 루트: `{root}`")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    tickers = get_tickers(args)

    if not args.collect_only:
        run_model_for_tickers(args, tickers)

    df = collect_results(root)
    root.mkdir(parents=True, exist_ok=True)

    csv_path = root / "v8_6_39_multi_asset_comparison.csv"
    md_path = root / "v8_6_39_multi_asset_comparison.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_markdown_report(df, root), encoding="utf-8")

    print("\n[COMPARISON SAVED]")
    print(f"- {csv_path}")
    print(f"- {md_path}")

    if not df.empty:
        print("\n[TOP RESULTS]")
        cols = [
            "ticker",
            "model_suitability_score_0_100",
            "strategy_cagr_pct",
            "strategy_mdd_pct",
            "strategy_sharpe",
            "strategy_calmar",
            "excess_cagr_vs_buyhold_pctp",
        ]
        print(df[[c for c in cols if c in df.columns]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
