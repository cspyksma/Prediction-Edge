from __future__ import annotations

import pandas as pd


def smoothed_win_pct(wins: int, games: int) -> float:
    return (wins + 1) / (games + 2)


def normalize_pitcher_hand(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "R"
    text = str(value).strip().upper()
    return "L" if text.startswith("L") else "R"


def current_streak(results: list[bool]) -> int:
    if not results:
        return 0
    latest = results[-1]
    streak = 0
    for result in reversed(results):
        if result != latest:
            break
        streak += 1
    return streak if latest else -streak


def coalesce_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
