from __future__ import annotations

import pandas as pd


class ChartPayloadBuilder:
    @staticmethod
    def equity_curve_from_predictions(df: pd.DataFrame) -> list[dict]:
        if "strategy_equity_net" not in df.columns:
            return []
        out = df[["Date", "strategy_equity_net"]].dropna().copy()
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
        return out.rename(columns={"strategy_equity_net": "value"}).to_dict("records")
