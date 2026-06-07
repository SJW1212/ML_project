"""
v8.7.0 improvement patch
========================
적용 대상: xgb_trend_participation_v8_7_0.py

핵심 수정:
1) run_walk_forward() 결과 pred_raw에 policy/overlay용 과거 피처를 보존한다.
2) compute_mid_trend_score()가 피처 누락 시 BEAR로 오판하지 않게 한다.
3) trend/recovery overlay가 실제로 발동 가능한 입력을 받도록 한다.
4) policy context 진단 함수를 추가한다.

적용 방법:
- 이 파일의 함수/상수를 원본 파일에 복사한다.
- 같은 이름의 함수는 교체한다.
- run_walk_forward() 안의 out = {"Date": date} 바로 아래에 아래 1줄을 추가한다.

    append_policy_context_to_prediction_row(out, all_df.iloc[pos])

- build_summary()의 return dict에 아래 항목을 추가한다.

    "policy_context_diagnostics": build_policy_context_diagnostics(pred_df, cfg),

주의:
- 아래 컬럼은 모두 현재 시점까지의 과거 rolling feature이므로 미래 정보 누수는 아니다.
- 단, future_* 컬럼은 절대 여기에 추가하면 안 된다.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# PATCH 1. Policy/overlay context columns
# ============================================================

POLICY_CONTEXT_COLUMNS: List[str] = [
    # returns / trend participation
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "return_120d",
    "price_ma_20_gap",
    "price_ma_60_gap",
    "price_ma_120_gap",
    "price_ma_200_gap",
    "ma_gap_5_20",
    "ma_gap_20_60",
    "ma_gap_60_120",
    "ma_gap_50_200",
    "trend_slope_20",
    "trend_slope_60",
    "ma200_slope_60",
    "positive_return_ratio_20",
    "positive_return_ratio_60",
    "trend_consistency_20",
    "trend_consistency_60",
    "price_position_20",
    "price_position_60",
    "close_to_20d_high",
    "close_to_60d_high",
    # recovery / drawdown context
    "drawdown_20",
    "drawdown_60",
    "drawdown_120",
    "ulcer_index_20",
    "ulcer_index_60",
    "ulcer_rank_252",
    # volatility context for sharpe proxy and diagnostics
    "realized_vol_20",
    "realized_vol_60",
    "atr_rank_252",
    "atr_pct_20",
    "atr_pct_60",
    "bb_width_rank_252",
]


def _safe_float_or_nan(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        out = float(value)
        if not np.isfinite(out):
            return float("nan")
        return out
    except Exception:
        return float("nan")


def append_policy_context_to_prediction_row(out: Dict[str, object], source_row: pd.Series) -> None:
    """Add current-time policy context features to one walk-forward prediction row.

    이 함수는 run_walk_forward() 내부에서 호출한다.
    source_row는 all_df.iloc[pos]처럼 현재 날짜의 원본 feature row여야 한다.
    """
    for col in POLICY_CONTEXT_COLUMNS:
        if col in source_row.index:
            out[col] = _safe_float_or_nan(source_row[col])


# ============================================================
# PATCH 2. Safer row helpers and trend score
# ============================================================


def _row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    """원본 _row_float 교체 권장.

    default를 np.nan으로 넘기면 누락/NaN을 NaN으로 유지한다.
    default가 0.0이면 기존 호환 동작을 유지한다.
    """
    try:
        val = row.get(col, default)
        if pd.isna(val):
            return float(default)
        out = float(val)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _row_float_nan(row: pd.Series, col: str) -> float:
    return _row_float(row, col, float("nan"))


def compute_mid_trend_score(row: pd.Series) -> Tuple[int, str]:
    """중기 추세 필터. 원본 함수 교체 권장.

    변경 이유:
    - 원본은 필요한 피처가 pred_raw에 없을 때 전부 0으로 처리되어 mid_trend_state=BEAR가 된다.
    - 그러면 trend overlay는 0% 발동하고, 일부 공격 신호도 불필요하게 제한된다.
    - 피처가 충분하지 않으면 UNKNOWN으로 반환해 오판을 막는다.
    """
    checks = {
        "return_60d_pos": _row_float_nan(row, "return_60d") > 0.0,
        "return_120d_pos": _row_float_nan(row, "return_120d") > 0.0,
        "price_ma_60_gap_pos": _row_float_nan(row, "price_ma_60_gap") > 0.0,
        "price_ma_120_gap_pos": _row_float_nan(row, "price_ma_120_gap") > 0.0,
        "ma_gap_20_60_pos": _row_float_nan(row, "ma_gap_20_60") > 0.0,
        "trend_slope_60_pos": _row_float_nan(row, "trend_slope_60") > 0.0,
    }
    raw_values = {
        "return_60d": _row_float_nan(row, "return_60d"),
        "return_120d": _row_float_nan(row, "return_120d"),
        "price_ma_60_gap": _row_float_nan(row, "price_ma_60_gap"),
        "price_ma_120_gap": _row_float_nan(row, "price_ma_120_gap"),
        "ma_gap_20_60": _row_float_nan(row, "ma_gap_20_60"),
        "trend_slope_60": _row_float_nan(row, "trend_slope_60"),
    }
    available = sum(np.isfinite(v) for v in raw_values.values())
    if available < 4:
        return 0, "UNKNOWN"

    score = int(sum(bool(v) for v in checks.values()))
    if score >= 4:
        state = "BULL"
    elif score <= 2:
        state = "BEAR"
    else:
        state = "NEUTRAL"
    return score, state


# ============================================================
# PATCH 3. Trend participation overlay replacement
# ============================================================


def _redistribute_after_stock_change(
    new_stock: float,
    old_w: Tuple[float, float, float],
    cash_ratio: float | None = None,
) -> Tuple[float, float, float]:
    old_stock, old_bond, old_cash = old_w
    new_stock = float(np.clip(new_stock, 0.0, 1.0))
    remain = max(0.0, 1.0 - new_stock)
    defensive_total = old_bond + old_cash
    if cash_ratio is None:
        if defensive_total <= 0:
            bond_ratio = 0.65
            cash_ratio = 0.35
        else:
            cash_ratio = float(np.clip(old_cash / defensive_total, 0.0, 1.0))
            bond_ratio = 1.0 - cash_ratio
    else:
        cash_ratio = float(np.clip(cash_ratio, 0.0, 1.0))
        bond_ratio = 1.0 - cash_ratio
    vals = np.asarray([new_stock, remain * bond_ratio, remain * cash_ratio], dtype=float)
    vals = np.clip(vals, 0.0, 1.0)
    total = float(vals.sum())
    if total <= 0:
        return 1.0, 0.0, 0.0
    vals = vals / total
    return float(vals[0]), float(vals[1]), float(vals[2])


def _market_sharpe_proxy(row: pd.Series, horizon: int) -> float:
    h = int(horizon)
    ret = _row_float(row, f"return_{h}d", float("nan"))
    if not np.isfinite(ret):
        return 0.0
    vol_col = f"realized_vol_{h}" if f"realized_vol_{h}" in row.index else "realized_vol_60"
    vol = _row_float(row, vol_col, float("nan"))
    if not np.isfinite(vol) or vol <= 1e-12:
        return 0.0
    return float(ret / (vol * math.sqrt(max(h, 1))))


def _trend_floor_from_high_vol(ph: float, cfg: object) -> float:
    ph = float(np.clip(ph, 0.0, 1.0))
    if ph < 0.25:
        return float(getattr(cfg, "trend_floor_lt25", 0.98))
    if ph < 0.35:
        return float(getattr(cfg, "trend_floor_lt35", 0.96))
    if ph < 0.50:
        return float(getattr(cfg, "trend_floor_lt50", 0.92))
    if ph < 0.65:
        return float(getattr(cfg, "trend_floor_lt65", 0.86))
    return 0.0


def apply_trend_participation_overlay(
    signal_w: Tuple[float, float, float],
    row: pd.Series,
    cfg: object,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    """강한 단방향 상승 추세장의 underweight drag 완화 overlay. 원본 교체 권장."""
    ph = _row_float(row, "prob_high_vol", 0.0)
    pds_score = _row_float(row, "prob_down_strengthening_score", 0.0)
    sharpe60 = _market_sharpe_proxy(row, 60)
    sharpe120 = _market_sharpe_proxy(row, 120)
    pos_ratio60 = _row_float(row, "positive_return_ratio_60", _row_float(row, "trend_consistency_60", 0.0))
    trend_score, trend_state = compute_mid_trend_score(row)

    context_ok = bool(trend_state != "UNKNOWN")
    trend_bull = bool(
        context_ok
        and bool(getattr(cfg, "enable_trend_participation_overlay", True))
        and ph < float(getattr(cfg, "trend_max_high_vol_for_overlay", 0.65))
        and trend_score >= int(getattr(cfg, "trend_min_mid_trend_score", 4))
        and pos_ratio60 >= float(getattr(cfg, "trend_positive_ratio_60_threshold", 0.55))
        and (
            sharpe60 >= float(getattr(cfg, "trend_sharpe60_threshold", 0.50))
            or sharpe120 >= float(getattr(cfg, "trend_sharpe120_threshold", 0.45))
        )
    )

    action = "off"
    target_stock = float(signal_w[0])
    if trend_bull:
        floor = _trend_floor_from_high_vol(ph, cfg)
        if bool(getattr(cfg, "trend_full_stock_when_low_vol", False)) and ph < float(getattr(cfg, "trend_full_stock_high_vol_threshold", 0.25)):
            floor = max(floor, 1.0)
        if pds_score >= 0.50:
            floor = min(floor, max(float(signal_w[0]), 0.90))
        if floor > target_stock + 1e-12:
            target_stock = floor
            action = "trend_floor_upgrade"

    out_w = _redistribute_after_stock_change(target_stock, signal_w)
    meta = {
        "market_sharpe_60": float(sharpe60),
        "market_sharpe_120": float(sharpe120),
        "trend_positive_ratio_60": float(pos_ratio60),
        "trend_context_available": bool(context_ok),
        "trend_bull_regime": bool(trend_bull),
        "trend_participation_action": str(action),
        "trend_participation_target_stock": float(target_stock),
        "trend_participation_overlay": float(out_w[0] - signal_w[0]),
        "trend_participation_force_rebalance": bool(trend_bull and action != "off" and bool(getattr(cfg, "trend_force_rebalance", False))),
        "trend_score_for_overlay": int(trend_score),
        "trend_state_for_overlay": str(trend_state),
    }
    return out_w, meta


# ============================================================
# PATCH 4. Recovery overlay replacement
# ============================================================


def apply_recovery_rerisk_overlay(
    signal_w: Tuple[float, float, float],
    row: pd.Series,
    cfg: object,
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    """폭락 후 V자 반등 초입 재진입 지연 완화 overlay. 원본 교체 권장."""
    ph = _row_float(row, "prob_high_vol", 0.0)
    dd60 = _row_float(row, "drawdown_60", float("nan"))
    ret10 = _row_float(row, "return_10d", float("nan"))
    ret20 = _row_float(row, "return_20d", float("nan"))
    ma20_gap = _row_float(row, "price_ma_20_gap", float("nan"))
    pds_score = _row_float(row, "prob_down_strengthening_score", 0.0)
    pus_score = _row_float(row, "prob_up_strengthening_score", 0.0)

    context_ok = bool(np.isfinite(dd60) and np.isfinite(ret10) and np.isfinite(ret20) and np.isfinite(ma20_gap))
    recovery = bool(
        context_ok
        and bool(getattr(cfg, "enable_recovery_rerisk_overlay", True))
        and dd60 <= float(getattr(cfg, "recovery_dd60_threshold", -0.12))
        and (ret10 >= float(getattr(cfg, "recovery_return10_threshold", 0.06)) or ret20 >= float(getattr(cfg, "recovery_return20_threshold", 0.08)))
        and ma20_gap >= float(getattr(cfg, "recovery_price_ma20_gap_threshold", 0.0))
        and pds_score < float(getattr(cfg, "recovery_down_strength_max", 0.45))
        and ph < float(getattr(cfg, "recovery_high_vol_max", 0.90))
    )

    action = "off"
    target_stock = float(signal_w[0])
    if recovery:
        floor = float(getattr(cfg, "recovery_stock_floor", 0.78))
        if pus_score >= float(getattr(cfg, "recovery_strong_up_score_threshold", 0.30)):
            floor = max(floor, float(getattr(cfg, "recovery_strong_stock_floor", 0.90)))
        if floor > target_stock + 1e-12:
            target_stock = floor
            action = "recovery_rerisk_upgrade"

    out_w = _redistribute_after_stock_change(target_stock, signal_w)
    meta = {
        "recovery_context_available": bool(context_ok),
        "recovery_risk_on": bool(recovery),
        "recovery_rerisk_action": str(action),
        "recovery_rerisk_target_stock": float(target_stock),
        "recovery_rerisk_overlay": float(out_w[0] - signal_w[0]),
        "recovery_rerisk_force_rebalance": bool(recovery and action != "off" and bool(getattr(cfg, "recovery_force_rebalance", True))),
    }
    return out_w, meta


# ============================================================
# PATCH 5. Diagnostics
# ============================================================


def build_policy_context_diagnostics(pred_df: pd.DataFrame, cfg: object) -> Dict[str, object]:
    """summary에 추가할 policy context 진단."""
    required_for_trend = [
        "return_60d",
        "return_120d",
        "price_ma_60_gap",
        "price_ma_120_gap",
        "ma_gap_20_60",
        "trend_slope_60",
        "positive_return_ratio_60",
        "realized_vol_60",
    ]
    required_for_recovery = [
        "drawdown_60",
        "return_10d",
        "return_20d",
        "price_ma_20_gap",
    ]
    all_required = sorted(set(required_for_trend + required_for_recovery))
    missing = [c for c in all_required if c not in pred_df.columns]
    nan_rate = {
        c: float(pred_df[c].isna().mean())
        for c in all_required
        if c in pred_df.columns
    }
    trend_scores: List[int] = []
    trend_states: List[str] = []
    for _, row in pred_df.iterrows():
        s, state = compute_mid_trend_score(row)
        trend_scores.append(int(s))
        trend_states.append(str(state))
    trend_state_dist = pd.Series(trend_states).value_counts(normalize=True).mul(100).round(2).to_dict()
    return {
        "policy_context_columns_expected": all_required,
        "missing_policy_context_columns": missing,
        "nan_rate_by_context_column": nan_rate,
        "mid_trend_score_mean_recomputed": float(np.mean(trend_scores)) if trend_scores else 0.0,
        "mid_trend_state_distribution_recomputed_pct": trend_state_dist,
        "trend_overlay_enabled": bool(getattr(cfg, "enable_trend_participation_overlay", True)),
        "recovery_overlay_enabled": bool(getattr(cfg, "enable_recovery_rerisk_overlay", True)),
        "is_context_sufficient_for_trend": len([c for c in required_for_trend if c not in pred_df.columns]) == 0,
        "is_context_sufficient_for_recovery": len([c for c in required_for_recovery if c not in pred_df.columns]) == 0,
    }
