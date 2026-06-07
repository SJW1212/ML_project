"""
v8.6.40 sequential patch generator
==================================

Input : xgb_recency_weighted_v8_6_39.py
Output:
  - xgb_recency_weighted_v8_6_40a.py  # P0: mid_trend context + trace + conflict diagnostics, no hard cap
  - xgb_recency_weighted_v8_6_40b.py  # P1: + UpStrength minimum filter + no-trade override
  - xgb_recency_weighted_v8_6_40c.py  # P2: + extreme high-vol hard cap

Design rule:
  - Do not set pdn=0.0. Down-risk raw signal is ignored, but allocation gate keeps pdn=prob_high_vol.
  - Each output is self-contained; no import from patch_core is required.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HELPER_BLOCK = r'''

# ============================================================
# v8.6.40 SEQUENTIAL PATCH HELPERS
# ============================================================

V8640_MID_TREND_FEATURE_COLS = [
    "return_60d",
    "return_120d",
    "price_ma_60_gap",
    "price_ma_120_gap",
    "ma_gap_20_60",
    "trend_slope_60",
]

V8640_MID_TREND_FEATURE_COLS_EXTENDED = V8640_MID_TREND_FEATURE_COLS + [
    "drawdown_60",
    "realized_vol_60",
    "trend_slope_20",
    "ma_gap_60_120",
]


def compute_mid_trend_score(row: pd.Series) -> Tuple[int, str]:
    """
    v8.6.40 safe mid-trend score.

    v8.6.39 bug:
        raw feature context was not preserved in predictions.csv.
        _row_float(..., default=0.0) made all six checks False,
        so mid_trend_score=0 and mid_trend_state=BEAR for all rows.

    v8.6.40 behavior:
        - If enough trend context exists, score it normally.
        - If context is missing, return UNKNOWN instead of fake BEAR.
    """
    available = [c for c in V8640_MID_TREND_FEATURE_COLS if c in row.index and not pd.isna(row[c])]
    if len(available) < 3:
        return 0, "UNKNOWN"

    checks = []
    for col in V8640_MID_TREND_FEATURE_COLS:
        if col in row.index and not pd.isna(row[col]):
            try:
                checks.append(float(row[col]) > 0.0)
            except Exception:
                pass

    if not checks:
        return 0, "UNKNOWN"

    score = int(sum(bool(x) for x in checks))
    total = len(checks)
    if total >= 4:
        if score >= max(4, int(round(total * 0.67))):
            state = "BULL"
        elif score <= int(round(total * 0.33)):
            state = "BEAR"
        else:
            state = "NEUTRAL"
    else:
        if score >= 3:
            state = "BULL"
        elif score == 0:
            state = "BEAR"
        else:
            state = "NEUTRAL"
    return score, state


def v8640_patch_prediction_row_with_features(out: dict, feature_row: pd.Series, feature_cols: List[str]) -> dict:
    """Preserve raw policy-context features in walk-forward prediction rows."""
    keep = [c for c in V8640_MID_TREND_FEATURE_COLS_EXTENDED if c in feature_cols or c in feature_row.index]
    for col in keep:
        val = feature_row.get(col, np.nan)
        if not pd.isna(val):
            try:
                out[col] = float(val)
                out[f"ctx_{col}"] = float(val)
            except Exception:
                pass
    return out


def v8640_build_allocation_trace(
    base_stock: float,
    up_bonus: float,
    trend_cut: float,
    drawdown_cut: float,
    no_trade_band_hold: bool,
    final_stock: float,
    mid_trend_state: str,
    ph: float,
    pus_score: float,
    tier: int,
    regime: str,
) -> dict:
    """Human-readable trace of allocation construction."""
    return {
        "base_stock_from_highvol": round(float(base_stock), 4),
        "up_bonus": round(float(up_bonus), 4),
        "trend_cut": round(float(trend_cut), 4),
        "drawdown_cut": round(float(drawdown_cut), 4),
        "no_trade_band_hold": bool(no_trade_band_hold),
        "final_stock": round(float(final_stock), 4),
        "mid_trend_state": str(mid_trend_state),
        "prob_high_vol": round(float(ph), 4),
        "up_strength_score": round(float(pus_score), 4),
        "offensive_tier": int(tier),
        "signal_regime": str(regime),
    }


def v8640_check_allocation_conflict(
    stock: float,
    ph: float,
    mid_trend_state: str,
    drawdown_guard: float = 0.0,
) -> List[dict]:
    """Detect contradictory allocation states. v8.6.40a/b record only; v8.6.40c may cap hard rules."""
    rules = [
        (
            "midtrend_bear_high_stock",
            str(mid_trend_state) == "BEAR" and float(stock) >= 0.80,
            "WARNING",
            "mid_trend=BEAR but stock>=80%. Check trend context or policy override.",
        ),
        (
            "highvol_high_stock",
            float(ph) > 0.60 and float(stock) >= 0.80,
            "WARNING",
            "prob_high_vol>0.60 but stock>=80%.",
        ),
        (
            "extreme_highvol_full_stock",
            float(ph) > 0.80 and float(stock) >= 0.70,
            "HARD_CAP",
            "prob_high_vol>0.80 but stock>=70%.",
        ),
    ]
    out: List[dict] = []
    for rule, cond, level, msg in rules:
        if bool(cond):
            out.append({
                "rule": rule,
                "level": level,
                "message": msg,
                "stock": round(float(stock), 4),
                "ph": round(float(ph), 4),
                "mid_trend_state": str(mid_trend_state),
            })
    return out


def v8640_should_override_no_trade_band(
    ph: float,
    ph_prev: float,
    mid_trend_state: str,
    mid_trend_state_prev: str,
    drawdown_guard: float,
    drawdown_guard_prev: float,
    cfg,
) -> Tuple[bool, str]:
    """Emergency override for no-trade/scheduled hold when risk state changes sharply."""
    hv_threshold = float(getattr(cfg, "no_trade_override_hv_threshold", 0.70))
    if float(ph) >= hv_threshold and float(ph_prev) < hv_threshold:
        return True, f"prob_high_vol_cross_{hv_threshold:.2f}"

    hv_surge_delta = float(getattr(cfg, "no_trade_override_hv_surge_delta", 0.15))
    if float(ph) - float(ph_prev) >= hv_surge_delta and float(ph) >= 0.55:
        return True, "prob_high_vol_surge"

    if (
        str(mid_trend_state) != str(mid_trend_state_prev)
        and str(mid_trend_state) not in {"", "UNKNOWN"}
        and str(mid_trend_state_prev) not in {"", "UNKNOWN"}
        and {str(mid_trend_state), str(mid_trend_state_prev)} == {"BULL", "BEAR"}
    ):
        return True, f"mid_trend_flip_{mid_trend_state_prev}_to_{mid_trend_state}"

    dg_surge = float(getattr(cfg, "no_trade_override_drawdown_surge", 0.15))
    if float(drawdown_guard) - float(drawdown_guard_prev) >= dg_surge:
        return True, "drawdown_guard_surge"

    return False, ""


def v8640_build_hold_reason_diagnostics(pred_df: pd.DataFrame) -> pd.DataFrame:
    required = {"hold_reason", "strategy_return_net", "stock_next_return", "stock_weight"}
    if not required.issubset(pred_df.columns):
        return pd.DataFrame()
    df = pred_df.copy()
    df["strategy_return_net"] = pd.to_numeric(df["strategy_return_net"], errors="coerce")
    df["stock_next_return"] = pd.to_numeric(df["stock_next_return"], errors="coerce")
    rows = []
    for reason, grp in df.groupby("hold_reason", dropna=False):
        r = grp["strategy_return_net"].dropna()
        bh = grp["stock_next_return"].dropna()
        if len(r) == 0:
            continue
        ann_ret = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0)
        ann_bh = float((1.0 + bh).prod() ** (252 / max(len(bh), 1)) - 1.0) if len(bh) else np.nan
        ann_vol = float(r.std() * np.sqrt(252)) if len(r) > 1 else np.nan
        rows.append({
            "hold_reason": str(reason),
            "count": int(len(grp)),
            "pct": float(len(grp) / max(len(df), 1)),
            "ann_return_strat": ann_ret,
            "ann_return_bh": ann_bh,
            "bh_gap": ann_ret - ann_bh if not np.isnan(ann_bh) else np.nan,
            "win_rate": float((r > 0).mean()),
            "ann_vol": ann_vol,
            "sharpe": ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan,
            "avg_stock_weight": float(grp["stock_weight"].mean()),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def v8640_build_up_strength_bin_diagnostics(
    pred_df: pd.DataFrame,
    score_col: str = "prob_up_strengthening_score",
    bins: int = 10,
) -> pd.DataFrame:
    if score_col not in pred_df.columns:
        return pd.DataFrame()
    df = pred_df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df["strategy_return_net"] = pd.to_numeric(df.get("strategy_return_net", 0.0), errors="coerce")
    df["_score_bin"] = pd.cut(df[score_col], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    rows = []
    for b, grp in df.groupby("_score_bin", observed=True):
        r = grp["strategy_return_net"].dropna()
        ann_ret = float((1.0 + r).prod() ** (252 / max(len(r), 1)) - 1.0) if len(r) else np.nan
        actual_col = "actual_direction_strength_20d" if "actual_direction_strength_20d" in grp.columns else None
        actual_up_rate = float((grp[actual_col] == "UP_STRENGTHENING").mean()) if actual_col else np.nan
        rows.append({
            "score_bin": str(b),
            "count": int(len(grp)),
            "avg_score": float(grp[score_col].mean()),
            "actual_up_rate": actual_up_rate,
            "ann_return_strat": ann_ret,
            "avg_stock_weight": float(grp.get("stock_weight", pd.Series(np.nan, index=grp.index)).mean()),
        })
    return pd.DataFrame(rows)


def v8640_build_conflict_diagnostics(pred_df: pd.DataFrame) -> pd.DataFrame:
    if not {"stock_weight", "prob_high_vol", "mid_trend_state"}.issubset(pred_df.columns):
        return pd.DataFrame()
    rows = []
    for _, row in pred_df.iterrows():
        conflicts = v8640_check_allocation_conflict(
            stock=float(row.get("stock_weight", 0.0)),
            ph=float(row.get("prob_high_vol", 0.0)),
            mid_trend_state=str(row.get("mid_trend_state", "UNKNOWN")),
        )
        for c in conflicts:
            c = dict(c)
            c["Date"] = row.get("Date", None)
            c["allocation_regime"] = row.get("allocation_regime", "")
            c["hold_reason"] = row.get("hold_reason", "")
            rows.append(c)
    return pd.DataFrame(rows)


def v8640_build_auto_review(pred_df: pd.DataFrame, summary: Dict[str, object], ticker: str, version: str) -> str:
    lines = [f"# Auto Review — {ticker} {version}", ""]
    perf = summary.get("performance", {}).get("strategy_after_cost", {}) if isinstance(summary, dict) else {}
    bh = summary.get("performance", {}).get("stock_buy_hold", {}) if isinstance(summary, dict) else {}
    lines.append("## Topline")
    lines.append(f"- Strategy CAGR: {float(perf.get('cagr', np.nan)):.2%}" if perf else "- Strategy CAGR: N/A")
    lines.append(f"- Strategy MDD: {float(perf.get('mdd', np.nan)):.2%}" if perf else "- Strategy MDD: N/A")
    lines.append(f"- Strategy Sharpe: {float(perf.get('sharpe', np.nan)):.3f}" if perf else "- Strategy Sharpe: N/A")
    if bh:
        lines.append(f"- Buy&Hold CAGR: {float(bh.get('cagr', np.nan)):.2%}")
    lines.append("")

    if "mid_trend_score" in pred_df.columns:
        zero_pct = float((pd.to_numeric(pred_df["mid_trend_score"], errors="coerce") == 0).mean())
        state_dist = pred_df.get("mid_trend_state", pd.Series(index=pred_df.index, dtype=object)).value_counts(normalize=True).to_dict()
        lines.append("## mid_trend diagnostic")
        lines.append(f"- mid_trend_score==0 pct: {zero_pct:.1%}")
        lines.append(f"- mid_trend_state distribution: {state_dist}")
        if zero_pct > 0.90:
            lines.append("- WARNING: mid_trend may still be stuck. Check feature preservation.")
        lines.append("")

    hr = v8640_build_hold_reason_diagnostics(pred_df)
    if not hr.empty:
        lines.append("## hold_reason diagnostics")
        lines.append(hr.head(10).to_markdown(index=False))
        lines.append("")
    conf = v8640_build_conflict_diagnostics(pred_df)
    lines.append("## conflict diagnostics")
    if conf.empty:
        lines.append("- No conflicts detected.")
    else:
        lines.append(conf.groupby(["rule", "level"]).size().reset_index(name="count").to_markdown(index=False))
    lines.append("")
    lines.append("Truth note: This file is diagnostic, not a performance guarantee.")
    return "\n".join(lines)

'''


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Pattern not found: {label}")
    return text.replace(old, new, 1)


def replace_function(text: str, func_name: str, new_func: str) -> str:
    # Replace from def func_name(...) at top-level until next top-level def/class/comment block.
    pattern = re.compile(rf"^def {re.escape(func_name)}\(.*?\n(?=\ndef |\n# ============================================================|\n[A-Z_][A-Z0-9_]+\s*=)", re.S | re.M)
    m = pattern.search(text)
    if not m:
        raise RuntimeError(f"Function not found: {func_name}")
    return text[:m.start()] + new_func.strip() + "\n" + text[m.end():]


def versionize(text: str, version_code: str, version_dot: str) -> str:
    text = text.replace("xgb_recency_weighted_v8_6_39", f"xgb_recency_weighted_{version_code}")
    text = text.replace("results_xgb_recency_weighted_v8_6_39", f"results_xgb_recency_weighted_{version_code}")
    text = text.replace("v8.6.39", version_dot)
    text = text.replace("V8.6.39", version_dot.upper())
    return text


def patch_common(text: str, version_code: str, version_dot: str) -> str:
    text = versionize(text, version_code, version_dot)

    # Add v8.6.40 config fields after stale offensive fields.
    old = '''    enable_stale_offensive_decay: bool = True
    stale_offensive_stock_gap_threshold: float = 0.12
    stale_offensive_up_strength_reset_threshold: float = 0.20
    stale_offensive_high_vol_threshold: float = 0.72
'''
    new = old + '''
    # v8.6.40 sequential patch controls
    enable_v8640_diagnostics: bool = True
    up_strength_allocation_score_minimum: float = 0.0
    unknown_trend_max_stock: float = 0.82
    no_trade_override_hv_threshold: float = 0.70
    no_trade_override_hv_surge_delta: float = 0.15
    no_trade_override_drawdown_surge: float = 0.15
    conflict_hard_cap_extreme_highvol: bool = False
'''
    text = replace_exact(text, old, new, "Config v8640 fields")

    # Replace old compute_mid_trend_score with helper block containing safe compute function.
    old_func = re.search(r"^def compute_mid_trend_score\(.*?\n\s*return score, state\n", text, re.S | re.M)
    if not old_func:
        raise RuntimeError("old compute_mid_trend_score not found")
    text = text[:old_func.start()] + HELPER_BLOCK.strip() + "\n" + text[old_func.end():]

    # Preserve raw features at prediction-row construction.
    old = '''        })
        prediction_rows.append(out)
'''
    new = '''        })
        # [v8.6.40 P0] Preserve policy-context raw features for mid_trend_score.
        out = v8640_patch_prediction_row_with_features(out, all_df.iloc[pos], feature_cols)
        prediction_rows.append(out)
'''
    text = replace_exact(text, old, new, "prediction row feature preservation")

    # Fix down-risk allocation score to ignore raw down-risk but keep high-vol gate.
    old = '''    pdn_raw = float(np.clip(prob_down_risk, 0.0, 1.0))
    w = float(np.clip(getattr(cfg, "allocation_downrisk_weight", 0.0), 0.0, 1.0))
    return float(np.clip((1.0 - w) * ph + w * pdn_raw, 0.0, 1.0))
'''
    new = '''    # [v8.6.40 P0] Remove raw down-risk influence from allocation.
    # Important: keep high-vol as the allocation gate score. Do NOT return 0.0.
    # Otherwise RISK_OFF/EXTREME_RISK/emergency gates become too weak.
    return ph
'''
    text = replace_exact(text, old, new, "allocation_downrisk_score ph-only")

    # Add allocation trace to specialist meta.
    old = '''        "force_rebalance": bool(force_rebalance),
        "p20_up_strengthening": float(p20),
        "p20_tier": int(3 if tier3_signal else (1 if tier1_signal else 0)),
        "policy_note": "vol_base_no_weak_probs_tier2_enabled",
    }
'''
    new = '''        "force_rebalance": bool(force_rebalance),
        "p20_up_strengthening": float(p20),
        "p20_tier": int(3 if tier3_signal else (1 if tier1_signal else 0)),
        "policy_note": "vol_base_no_weak_probs_tier2_enabled",
        "allocation_trace": v8640_build_allocation_trace(
            base_stock=float(base_stock),
            up_bonus=float(max(0.0, w[0] - base_stock)),
            trend_cut=float(cut),
            drawdown_cut=0.0,
            no_trade_band_hold=False,
            final_stock=float(w[0]),
            mid_trend_state=str(trend_state),
            ph=float(ph),
            pus_score=float(pus_score),
            tier=int(offensive_tier),
            regime=str(regime),
        ),
    }
'''
    text = replace_exact(text, old, new, "allocation trace meta")

    # Add conflict diagnostics after final w is selected and before turnover.
    old = '''        turnover = 0.0 if prev_w is None else sum(abs(w[j] - prev_w[j]) for j in range(3))
'''
    new = '''        # [v8.6.40 P0/P2] Conflict diagnostics. v8.6.40a/b record only; v8.6.40c may hard-cap extreme high-vol.
        allocation_conflicts = v8640_check_allocation_conflict(
            stock=float(w[0]),
            ph=float(ph),
            mid_trend_state=str(policy_meta.get("mid_trend_state", "UNKNOWN")),
        )
        if bool(getattr(cfg, "conflict_hard_cap_extreme_highvol", False)):
            if any(c.get("rule") == "extreme_highvol_full_stock" and c.get("level") == "HARD_CAP" for c in allocation_conflicts):
                capped_stock = min(float(w[0]), 0.70)
                if capped_stock < float(w[0]) - 1e-12:
                    w = _redistribute_after_stock_change(capped_stock, w)
                    executed_regime = infer_regime_from_weights(w, current_g)
                    hold_reason = "conflict_cap_extreme_highvol"
                    trade_executed = True
                    policy_meta["allocation_trace"] = v8640_build_allocation_trace(
                        base_stock=float(base_signal_w[0]),
                        up_bonus=float(max(0.0, w[0] - base_signal_w[0])),
                        trend_cut=0.0,
                        drawdown_cut=0.0,
                        no_trade_band_hold=False,
                        final_stock=float(w[0]),
                        mid_trend_state=str(policy_meta.get("mid_trend_state", "UNKNOWN")),
                        ph=float(ph),
                        pus_score=float(row.get("prob_up_strengthening_score", row.get("prob_up_strengthening", 0.0))),
                        tier=int(policy_meta.get("offensive_tier", 0)),
                        regime=str(signal_regime),
                    )

        turnover = 0.0 if prev_w is None else sum(abs(w[j] - prev_w[j]) for j in range(3))
'''
    text = replace_exact(text, old, new, "conflict diagnostics before turnover")

    # Add outputs to row.
    old = '''            "policy_note": str(policy_meta.get("policy_note", "")),
'''
    new = '''            "policy_note": str(policy_meta.get("policy_note", "")),
            "allocation_trace": policy_meta.get("allocation_trace", {}),
            "allocation_conflict_count": int(len(allocation_conflicts)),
            "allocation_conflict_rules": ";".join([str(c.get("rule", "")) for c in allocation_conflicts]),
'''
    text = replace_exact(text, old, new, "row allocation trace/conflict")

    # Add diagnostics to diagnostics dict.
    old = '''    if not getattr(args, "no_diagnostics", False):
        diagnostics = build_optimization_diagnostics(pred_df, summary)
        summary["optimization_diagnostics_summary"] = diagnostics_summary(diagnostics)
'''
    new = '''    if not getattr(args, "no_diagnostics", False):
        diagnostics = build_optimization_diagnostics(pred_df, summary)
        if bool(getattr(cfg, "enable_v8640_diagnostics", True)):
            hold_diag = v8640_build_hold_reason_diagnostics(pred_df)
            if not hold_diag.empty:
                diagnostics["hold_reason_diagnostics"] = hold_diag
            up_bin_diag = v8640_build_up_strength_bin_diagnostics(pred_df)
            if not up_bin_diag.empty:
                diagnostics["up_strength_bin_diagnostics"] = up_bin_diag
            conflict_diag = v8640_build_conflict_diagnostics(pred_df)
            if not conflict_diag.empty:
                diagnostics["allocation_conflict_diagnostics"] = conflict_diag
        summary["optimization_diagnostics_summary"] = diagnostics_summary(diagnostics)
'''
    text = replace_exact(text, old, new, "diagnostics injection")

    # Add auto_review save after summary/latest json write.
    old = '''    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary["latest_prediction"], f, ensure_ascii=False, indent=2)

    pd.Series(summary.get("stage1_feature_importance_mean", {}), name="importance").to_csv(importance_stage1_path, encoding="utf-8-sig")
'''
    new = '''    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary["latest_prediction"], f, ensure_ascii=False, indent=2)

    auto_review_path = result_dir / f"{file_prefix}_auto_review.md"
    if bool(getattr(cfg, "enable_v8640_diagnostics", True)):
        try:
            auto_review_text = v8640_build_auto_review(pred_df, summary, ticker=str(cfg.target_ticker), version="{version_dot}")
            with open(auto_review_path, "w", encoding="utf-8") as f:
                f.write(auto_review_text)
        except Exception as exc:
            warnings.warn(f"auto_review 생성 실패: {exc}")

    pd.Series(summary.get("stage1_feature_importance_mean", {}), name="importance").to_csv(importance_stage1_path, encoding="utf-8-sig")
'''.replace("{version_dot}", version_dot)
    text = replace_exact(text, old, new, "auto review output")

    # Print auto review path.
    old = '''    print(f"- {latest_path}")
    print(f"- {importance_stage1_path}")
'''
    new = '''    print(f"- {latest_path}")
    if 'auto_review_path' in locals():
        print(f"- {auto_review_path}")
    print(f"- {importance_stage1_path}")
'''
    text = replace_exact(text, old, new, "print auto_review")

    return text


def patch_b(text: str) -> str:
    # Change default score minimum to 0.40.
    text = text.replace("up_strength_allocation_score_minimum: float = 0.0", "up_strength_allocation_score_minimum: float = 0.40")

    # Enforce score min after normal/full/tier signals are initially computed and before effects are applied.
    old = '''        # 약한 Tier 1: 기본 변동성 비중보다 너무 낮을 때만 80% 수준으로 보정한다.
        if tier1_signal:
'''
    new = '''        # [v8.6.40b P1] Ignore weak UpStrength allocation signals below validated score floor.
        up_score_min = float(getattr(cfg, "up_strength_allocation_score_minimum", 0.40))
        if pus_score < up_score_min:
            tier1_signal = False
            tier2_signal = False
            tier3_signal = False
            full_stock_signal = False
            normal_full_signal = False
            strong_all3 = False
            offensive_tier = 0
            force_rebalance = False
            short_mid_policy_action = "blocked_by_up_strength_score_min"

        # 약한 Tier 1: 기본 변동성 비중보다 너무 낮을 때만 80% 수준으로 보정한다.
        if tier1_signal:
'''
    text = replace_exact(text, old, new, "UpStrength score min filter")

    # Add no-trade override in not-rebalance branch.
    start = text.find('''        elif not rebalance_due:\n            # v8.6.21 optional: offensive target이 사라졌는데 이전 98~100% 비중이 과도하게 남는 문제를 실험적으로 제어한다.''')
    marker = '''        else:\n            total_delta_to_signal = sum(abs(signal_w[j] - prev_w[j]) for j in range(3))\n'''
    end = text.find(marker, start)
    if start == -1 or end == -1:
        raise RuntimeError("not rebalance branch not found")
    old_branch = text[start:end]
    new_branch = '''        elif not rebalance_due:
            # [v8.6.40b P1] Emergency override for stale scheduled hold when risk context changes sharply.
            ph_prev = float(pred_df.iloc[max(0, i - 1)].get("prob_high_vol", ph)) if i > 0 else ph
            mt_prev = str(pred_df.iloc[max(0, i - 1)].get("mid_trend_state", "UNKNOWN")) if i > 0 else "UNKNOWN"
            override_no_trade, override_reason = v8640_should_override_no_trade_band(
                ph=ph,
                ph_prev=ph_prev,
                mid_trend_state=str(policy_meta.get("mid_trend_state", "UNKNOWN")),
                mid_trend_state_prev=mt_prev,
                drawdown_guard=0.0,
                drawdown_guard_prev=0.0,
                cfg=cfg,
            )
            if override_no_trade:
                w = signal_w
                executed_regime = signal_regime
                hold_reason = f"no_trade_override_{override_reason[:40]}"
                trade_executed = True
            else:
                # v8.6.21 optional: offensive target이 사라졌는데 이전 98~100% 비중이 과도하게 남는 문제를 실험적으로 제어한다.
                # 기본값은 OFF다. 켜려면 --enable-stale-offensive-decay를 사용한다.
                stale_gap = float(prev_w[0] - signal_w[0])
                stale_up_prob = float(row.get("prob_up_strengthening_score", row.get("prob_up_strengthening", 0.0)))
                stale_decay = (
                    bool(getattr(cfg, "enable_stale_offensive_decay", False))
                    and stale_gap >= float(getattr(cfg, "stale_offensive_stock_gap_threshold", 0.055))
                    and (
                        stale_up_prob < float(getattr(cfg, "stale_offensive_up_strength_reset_threshold", 0.20))
                        or ph >= float(getattr(cfg, "stale_offensive_high_vol_threshold", 0.55))
                    )
                )
                if stale_decay:
                    w = signal_w
                    executed_regime = signal_regime
                    hold_reason = "stale_offensive_decay"
                    trade_executed = True
                else:
                    w = prev_w
                    executed_regime = infer_regime_from_weights(w, current_g)
                    hold_reason = "not_rebalance_day"
'''
    text = text[:start] + new_branch + text[end:]
    return text


def patch_c(text: str) -> str:
    text = text.replace("conflict_hard_cap_extreme_highvol: bool = False", "conflict_hard_cap_extreme_highvol: bool = True")
    return text


def generate(source: Path, outdir: Path) -> list[Path]:
    base = source.read_text(encoding="utf-8")
    outputs = []

    a = patch_common(base, "v8_6_40a", "v8.6.40a")
    pa = outdir / "xgb_recency_weighted_v8_6_40a.py"
    pa.write_text(a, encoding="utf-8")
    outputs.append(pa)

    b = patch_b(patch_common(base, "v8_6_40b", "v8.6.40b"))
    pb = outdir / "xgb_recency_weighted_v8_6_40b.py"
    pb.write_text(b, encoding="utf-8")
    outputs.append(pb)

    c = patch_c(patch_b(patch_common(base, "v8_6_40c", "v8.6.40c")))
    pc = outdir / "xgb_recency_weighted_v8_6_40c.py"
    pc.write_text(c, encoding="utf-8")
    outputs.append(pc)

    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/mnt/data/xgb_recency_weighted_v8_6_39.py")
    ap.add_argument("--outdir", default="/mnt/data")
    args = ap.parse_args()
    outputs = generate(Path(args.source), Path(args.outdir))
    print("Generated:")
    for p in outputs:
        print("-", p)


if __name__ == "__main__":
    main()
