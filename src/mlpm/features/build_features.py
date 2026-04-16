from __future__ import annotations

import pandas as pd


def build_features(discrepancies: pd.DataFrame) -> pd.DataFrame:
    if discrepancies.empty:
        return pd.DataFrame()

    frame = discrepancies.copy()
    frame["abs_gap_bps"] = frame["gap_bps"].abs()
    frame["market_prob_centered"] = frame["market_prob"] - 0.5
    frame["model_prob_centered"] = frame["model_prob"] - 0.5
    return frame
