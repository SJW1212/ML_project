from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

REFERENCE_ENGINE_WARNING = (
    "현재 결과는 reference-compatible engine 기반이며, 실제 locked v8.6.41 trained model 결과가 아닙니다. "
    "백엔드/UI 흐름 검증용으로만 사용해야 합니다."
)

ACTION_LABEL_KO: Dict[str, str] = {
    "REDUCE": "비중 축소",
    "SLIGHT_REDUCE": "소폭 축소",
    "HOLD": "유지",
    "SLIGHT_INCREASE": "소폭 확대",
    "INCREASE": "비중 확대",
}

RISK_CLASS_LABEL_KO: Dict[str, str] = {
    "NORMAL": "정상",
    "WATCH": "주의",
    "HIGH_RISK": "고위험",
}

DIRECTION_CLASS_LABEL_KO: Dict[str, str] = {
    "UP_STRENGTH": "상방 강화",
    "NEUTRAL": "중립",
    "DOWN_STRENGTH": "하방 강화",
}

ALLOCATION_CLASS_LABEL_KO: Dict[str, str] = {
    "PARTICIPATION": "참여형 배분",
    "BALANCED": "균형 배분",
    "DEFENSIVE": "방어적 배분",
}

ASSET_TYPE_LABEL_KO: Dict[str, str] = {
    "stock": "주식",
    "etf": "ETF",
    "risk_asset": "위험자산",
    "bond": "채권",
    "bond_etf": "채권 ETF",
    "bond_bucket": "채권 버킷",
    "cash": "현금",
    "cash_bucket": "현금 버킷",
}

REASON_TEXT_KO: Dict[str, str] = {
    "HIGH_VOL_PROBABILITY_ELEVATED": "고변동 확률이 높아 방어적 비중 조정이 적용되었습니다.",
    "OVERALL_RISK_ELEVATED": "종합 위험 확률이 높아 위험자산 비중을 축소했습니다.",
    "DOWN_STRENGTH_ELEVATED": "하방 강화 확률이 높아 비중 축소 방향으로 판단했습니다.",
    "UP_STRENGTH_ELEVATED": "상방 강화 확률은 높지만, 위험도 조건과 함께 최종 비중을 조정했습니다.",
    "NEUTRAL_DIRECTION": "방향성 우위가 뚜렷하지 않아 중립 판단을 적용했습니다.",
    "REFERENCE_ENGINE_ONLY": "현재 결과는 실제 locked model이 아닌 reference-compatible engine 기반입니다.",
    "ACTION_REDUCE": "현재 비중보다 추천 비중이 낮아 축소 액션이 표시되었습니다.",
    "ACTION_INCREASE": "현재 비중보다 추천 비중이 높아 확대 액션이 표시되었습니다.",
    "ACTION_HOLD": "현재 비중과 추천 비중의 차이가 작아 유지로 표시되었습니다.",
}


def pct(value: Any, digits: int = 2) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value) * 100.0, digits)
    except Exception:
        return None


def engine_status(engine_mode: str, model_version: str | None = None) -> Dict[str, Any]:
    is_locked = engine_mode == "locked_v8641_trained_model"
    is_reference = engine_mode == "reference_v8641_compatible"
    status = {
        "engine_mode": engine_mode,
        "engine_mode_label_ko": "v8.6.41 호환 Reference 엔진" if is_reference else ("Locked v8.6.41 학습 모델" if is_locked else engine_mode),
        "model_version": model_version,
        "production_ready_for_investment": False if is_reference else is_locked,
        "is_reference_engine": is_reference,
        "is_locked_model": is_locked,
        "warning": REFERENCE_ENGINE_WARNING if is_reference else None,
    }
    return status


def classify_reason_codes(signal: Mapping[str, Any], action: str | None = None, engine_mode: str | None = None) -> List[str]:
    codes: List[str] = []
    prob_high_vol = float(signal.get("prob_high_vol") or 0.0)
    prob_overall_risk = float(signal.get("prob_overall_risk") or 0.0)
    prob_down = float(signal.get("prob_down_strengthening_score") or 0.0)
    prob_up = float(signal.get("prob_up_strengthening_score") or 0.0)
    direction_class = signal.get("direction_class")

    if prob_high_vol >= 0.60 or signal.get("risk_class") == "HIGH_RISK":
        codes.append("HIGH_VOL_PROBABILITY_ELEVATED")
    if prob_overall_risk >= 0.55 or signal.get("risk_class") == "HIGH_RISK":
        codes.append("OVERALL_RISK_ELEVATED")
    if prob_down >= 0.55 or direction_class == "DOWN_STRENGTH":
        codes.append("DOWN_STRENGTH_ELEVATED")
    if prob_up >= 0.55 or direction_class == "UP_STRENGTH":
        codes.append("UP_STRENGTH_ELEVATED")
    if direction_class == "NEUTRAL":
        codes.append("NEUTRAL_DIRECTION")
    if engine_mode == "reference_v8641_compatible":
        codes.append("REFERENCE_ENGINE_ONLY")
    if action in {"REDUCE", "SLIGHT_REDUCE"}:
        codes.append("ACTION_REDUCE")
    elif action in {"INCREASE", "SLIGHT_INCREASE"}:
        codes.append("ACTION_INCREASE")
    elif action == "HOLD":
        codes.append("ACTION_HOLD")

    # Preserve order while removing duplicates.
    seen = set()
    out: List[str] = []
    for c in codes:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def reason_text(reason_codes: Iterable[str]) -> str:
    parts = [REASON_TEXT_KO[c] for c in reason_codes if c in REASON_TEXT_KO]
    return " ".join(parts)


def enrich_latest_signals(latest_signals: List[Dict[str, Any]], engine_mode: str) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in latest_signals:
        item = deepcopy(row)
        item["risk_class_ko"] = RISK_CLASS_LABEL_KO.get(str(item.get("risk_class")), str(item.get("risk_class")))
        item["direction_class_ko"] = DIRECTION_CLASS_LABEL_KO.get(str(item.get("direction_class")), str(item.get("direction_class")))
        item["allocation_class_ko"] = ALLOCATION_CLASS_LABEL_KO.get(str(item.get("allocation_class")), str(item.get("allocation_class")))
        item["probability_pct"] = {
            "normal": pct(item.get("prob_normal")),
            "high_vol": pct(item.get("prob_high_vol")),
            "overall_risk": pct(item.get("prob_overall_risk")),
            "up_strengthening_score": pct(item.get("prob_up_strengthening_score")),
            "down_strengthening_score": pct(item.get("prob_down_strengthening_score")),
        }
        item["reason_codes"] = classify_reason_codes(item, engine_mode=engine_mode)
        item["reason_text"] = reason_text(item["reason_codes"])
        enriched.append(item)
    return enriched


def enrich_allocation(allocation: Dict[str, Any], latest_by_ticker: Mapping[str, Mapping[str, Any]], engine_mode: str) -> Dict[str, Any]:
    out = deepcopy(allocation)
    for row in out.get("allocation_rows", []):
        action = row.get("action")
        row["action_ko"] = ACTION_LABEL_KO.get(str(action), str(action))
        row["asset_type_ko"] = ASSET_TYPE_LABEL_KO.get(str(row.get("asset_type")), str(row.get("asset_type")))
        signal = latest_by_ticker.get(str(row.get("ticker")), {})
        row["risk_class"] = signal.get("risk_class")
        row["risk_class_ko"] = RISK_CLASS_LABEL_KO.get(str(signal.get("risk_class")), str(signal.get("risk_class"))) if signal else None
        row["direction_class"] = signal.get("direction_class")
        row["direction_class_ko"] = DIRECTION_CLASS_LABEL_KO.get(str(signal.get("direction_class")), str(signal.get("direction_class"))) if signal else None
        row["allocation_class"] = signal.get("allocation_class")
        row["allocation_class_ko"] = ALLOCATION_CLASS_LABEL_KO.get(str(signal.get("allocation_class")), str(signal.get("allocation_class"))) if signal else None
        row["current_weight_pct"] = pct(row.get("current_weight"))
        row["recommended_weight_pct"] = pct(row.get("recommended_weight"))
        codes = classify_reason_codes(signal, action=action, engine_mode=engine_mode)
        row["reason_codes"] = codes
        row["reason_text"] = reason_text(codes)
    totals = out.get("portfolio_totals", {})
    out["portfolio_totals_pct"] = {
        "stock_weight": pct(totals.get("stock_weight")),
        "bond_weight": pct(totals.get("bond_weight")),
        "cash_weight": pct(totals.get("cash_weight")),
    }
    return out


def equity_curve_tail_for_ui(perf_df: pd.DataFrame, tail: int = 20) -> List[Dict[str, Any]]:
    if perf_df is None or perf_df.empty:
        return []
    view = perf_df.tail(tail).copy()
    view["Date"] = pd.to_datetime(view["Date"]).dt.date.astype(str)
    rows = view.to_dict(orient="records")
    if rows:
        last = rows[-1]
        # stock_next_return is shift(-1), so the latest date has no realized next-day return yet.
        last["portfolio_return_raw"] = last.get("portfolio_return")
        last["portfolio_return"] = None
        last["return_status"] = "pending"
    for row in rows[:-1]:
        row["return_status"] = "realized"
    return rows


def build_ui_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    allocation = payload.get("allocation", {}) or {}
    totals = allocation.get("portfolio_totals", {}) or {}
    current_stock = 0.0
    current_bond = 0.0
    current_cash = 0.0
    for a in payload.get("portfolio_input", []) or []:
        w = float(a.get("current_weight") or 0.0)
        typ = a.get("asset_type")
        tick = a.get("ticker")
        if typ in {"cash", "cash_bucket"} or tick in {"CASH", "CASH_BUCKET"}:
            current_cash += w
        elif typ in {"bond", "bond_etf", "bond_bucket"} or tick == "BOND_BUCKET":
            current_bond += w
        else:
            current_stock += w
    recommended_stock = float(totals.get("stock_weight") or 0.0)
    recommended_bond = float(totals.get("bond_weight") or 0.0)
    recommended_cash = float(totals.get("cash_weight") or 0.0)
    return {
        "current_weights_pct": {
            "stock_weight": pct(current_stock),
            "bond_weight": pct(current_bond),
            "cash_weight": pct(current_cash),
        },
        "recommended_weights_pct": {
            "stock_weight": pct(recommended_stock),
            "bond_weight": pct(recommended_bond),
            "cash_weight": pct(recommended_cash),
        },
        "delta_pct_points": {
            "stock_weight": round((recommended_stock - current_stock) * 100.0, 3),
            "bond_weight": round((recommended_bond - current_bond) * 100.0, 3),
            "cash_weight": round((recommended_cash - current_cash) * 100.0, 3),
        },
        "primary_warning": (payload.get("engine_status") or {}).get("warning"),
    }


def build_ui_contract() -> Dict[str, Any]:
    return {
        "api": {
            "base_url_local": "http://127.0.0.1:8000",
            "evaluate_endpoint": "POST /portfolio/evaluate",
            "health_endpoint": "GET /health",
        },
        "required_response_sections": [
            "portfolio_input", "engine_status", "latest_signals", "allocation", "performance",
            "benchmarks", "validation", "data_update_status", "prediction_generation_status", "ui"
        ],
        "display_labels_ko": {
            "actions": ACTION_LABEL_KO,
            "risk_class": RISK_CLASS_LABEL_KO,
            "direction_class": DIRECTION_CLASS_LABEL_KO,
            "allocation_class": ALLOCATION_CLASS_LABEL_KO,
            "asset_type": ASSET_TYPE_LABEL_KO,
        },
        "stitch_screen_sections": [
            "포트폴리오 입력", "엔진 상태 경고", "추천 비중 요약", "현재 vs 추천 비중 비교",
            "종목별 판단 테이블", "성과 지표", "벤치마크 비교", "검증 상태"
        ],
        "safety_rules": [
            "engine_status.production_ready_for_investment가 false이면 투자 판단용 문구를 금지한다.",
            "engine_status.engine_mode는 항상 상단 배너에 노출한다.",
            "reference_v8641_compatible 결과는 백엔드/UI 검증용으로 표시한다.",
        ],
    }


def postprocess_dashboard_payload(payload: Dict[str, Any], perf_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    out = deepcopy(payload)
    mode = str(out.get("engine_mode") or "reference_v8641_compatible")
    version = out.get("model_version")
    out["engine_status"] = engine_status(mode, version)

    latest_by_ticker = {str(s.get("ticker")): s for s in out.get("latest_signals", []) or [] if s.get("ticker")}
    out["latest_signals"] = enrich_latest_signals(out.get("latest_signals", []) or [], mode)
    enriched_by_ticker = {str(s.get("ticker")): s for s in out.get("latest_signals", []) or [] if s.get("ticker")}
    out["allocation"] = enrich_allocation(out.get("allocation", {}) or {}, enriched_by_ticker or latest_by_ticker, mode)

    if perf_df is not None:
        out.setdefault("performance", {})["equity_curve_tail"] = equity_curve_tail_for_ui(perf_df)
    else:
        # Fall back to marking the last existing row as pending.
        rows = out.get("performance", {}).get("equity_curve_tail", []) or []
        if rows:
            for r in rows[:-1]:
                r.setdefault("return_status", "realized")
            rows[-1]["portfolio_return_raw"] = rows[-1].get("portfolio_return")
            rows[-1]["portfolio_return"] = None
            rows[-1]["return_status"] = "pending"

    validation = out.get("validation") or {}
    warnings = []
    if out["engine_status"].get("warning"):
        warnings.append(out["engine_status"]["warning"])
    validation.setdefault("ui_warnings", [])
    validation["ui_warnings"] = warnings + list(validation.get("ui_warnings") or [])
    out["validation"] = validation
    out["ui"] = build_ui_summary(out)
    return out
