from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from ..data.prediction_repository import PredictionRepository
from ..portfolio.allocation_policy_engine import AllocationPolicyEngine


class PredictionService:
    def __init__(self, repository: PredictionRepository, policy_engine: AllocationPolicyEngine | None = None):
        self.repository = repository
        self.policy_engine = policy_engine or AllocationPolicyEngine()

    @staticmethod
    def _col(row: pd.Series, name: str, default: float = 0.0) -> float:
        try:
            v = row.get(name, default)
            if pd.isna(v):
                return default
            return float(v)
        except Exception:
            return default

    @staticmethod
    def classify(row: pd.Series) -> tuple[str, str, str, list[str], str]:
        ph = PredictionService._col(row, "prob_high_vol")
        pu = PredictionService._col(row, "prob_up_strengthening_score")
        pdn = PredictionService._col(row, "prob_down_strengthening_score")
        risk = "NORMAL"
        warnings = []
        if ph >= 0.60:
            risk = "RISK_OFF"
            warnings.append("HIGH_VOL_RISK")
        elif ph >= 0.35 or pdn >= 0.50:
            risk = "WATCH"
            warnings.append("WATCH_SIGNAL")
        if pdn >= 0.50:
            direction = "DOWN_STRENGTH"
        elif pu >= 0.55 and pu - pdn >= 0.10:
            direction = "UP_STRENGTH"
        else:
            direction = "NEUTRAL"
        stock = PredictionService._col(row, "stock_weight", PredictionService._col(row, "executed_stock_weight", 0.0))
        allocation = "DEFENSIVE" if stock < 0.55 else "BALANCED" if stock < 0.80 else "PARTICIPATION"
        if risk == "WATCH" and direction == "DOWN_STRENGTH":
            comment = "하락 강화와 고변동 가능성이 있어 비중 확대를 보수적으로 판단해야 합니다."
        elif direction == "UP_STRENGTH":
            comment = "상승 강화 신호가 우세해 참여 비중을 유지할 수 있는 구간입니다."
        elif risk == "NORMAL":
            comment = "위험 상태는 정상이며 중립적 참여 구간입니다."
        else:
            comment = "위험 신호가 높아 방어적 확인이 필요합니다."
        return risk, direction, allocation, warnings, comment

    def latest_signals(self, tickers: Iterable[str], horizon: str = "10D") -> tuple[pd.DataFrame, list[dict]]:
        latest = self.repository.load_latest(tickers)
        load_errors = latest.attrs.get("load_errors", [])
        hnum = horizon.upper().replace("D", "")
        items = []
        for _, row in latest.iterrows():
            item = dict(row)
            # Apply the shared policy engine. Existing v8.6.41 native weights are
            # preserved; rows without native weights get the same fallback policy as live inference.
            item.update(self.policy_engine.apply_row(row, preserve_existing=True))
            risk, direction, allocation, warnings, comment = self.classify(pd.Series(item))
            item.update({
                "risk_class": risk,
                "direction_class": direction,
                "allocation_class": allocation,
                "warnings": warnings,
                "comment": comment,
                "selected_horizon": horizon.upper(),
                "selected_prob_high_vol": self._col(row, f"prob_high_vol_h{hnum}", self._col(row, "prob_high_vol")),
                "selected_prob_up_strengthening": self._col(row, f"prob_up_strengthening_{hnum}d", self._col(row, "prob_up_strengthening_score")),
                "selected_prob_down_strengthening": self._col(row, f"prob_down_strengthening_{hnum}d", self._col(row, "prob_down_strengthening_score")),
            })
            items.append(item)
        df = pd.DataFrame(items)
        df.attrs["load_errors"] = load_errors
        return df, load_errors

    @staticmethod
    def _first_not_none(*values):
        for value in values:
            try:
                if value is None or pd.isna(value):
                    continue
            except Exception:
                if value is None:
                    continue
            return value
        return None

    def performance_summary(self, tickers: Iterable[str]) -> list[dict]:
        out = []
        for ticker in tickers:
            summary = self.repository.load_summary(ticker)
            metrics = summary.get("strategy_metrics") or summary.get("metrics") or summary.get("full_period_metrics") or {}
            out.append({
                "ticker": ticker.upper(),
                "cagr": self._first_not_none(metrics.get("cagr"), metrics.get("strategy_cagr"), summary.get("cagr")),
                "mdd": self._first_not_none(metrics.get("mdd"), metrics.get("max_drawdown"), summary.get("mdd")),
                "sharpe": self._first_not_none(metrics.get("sharpe"), metrics.get("strategy_sharpe"), summary.get("sharpe")),
                "sortino": self._first_not_none(metrics.get("sortino"), summary.get("sortino")),
                "calmar": self._first_not_none(metrics.get("calmar"), summary.get("calmar")),
                "source": "summary_json",
            })
        return out
