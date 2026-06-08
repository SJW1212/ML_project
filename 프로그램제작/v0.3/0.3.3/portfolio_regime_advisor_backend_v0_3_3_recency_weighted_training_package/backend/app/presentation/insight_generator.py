from __future__ import annotations

import pandas as pd


class InsightGenerator:
    def generate(self, signals: pd.DataFrame, totals: dict) -> list[str]:
        insights = []
        insights.append(f"현재 전체 포트폴리오 비중은 주식 {totals.get('stock', 0):.1%}, 채권 {totals.get('bond', 0):.1%}, 현금 {totals.get('cash', 0):.1%}입니다.")
        if signals.empty:
            return insights
        risky = signals.sort_values(["prob_down_strengthening_score", "prob_high_vol"], ascending=False).head(1).iloc[0]
        if float(risky.get("prob_down_strengthening_score", 0)) >= 0.5 or float(risky.get("prob_high_vol", 0)) >= 0.35:
            insights.append(f"{risky['ticker']}는 하락 강화 또는 고변동 신호가 상대적으로 높아 비중 확대에 주의가 필요합니다.")
        normal_count = int((signals["risk_class"] == "NORMAL").sum()) if "risk_class" in signals else 0
        insights.append(f"분석된 {len(signals)}개 종목 중 {normal_count}개가 NORMAL 상태입니다.")
        return insights
