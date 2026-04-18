from __future__ import annotations

import pandas as pd

from mlpm.features.utils import normalize_pitcher_hand

OFFENSE_SPLIT_PRIOR = 0.274
OFFENSE_SPLIT_PRIOR_GAMES = 10.0


def build_team_offense_split_table(batting_logs_df: pd.DataFrame, pitching_logs_df: pd.DataFrame) -> pd.DataFrame:
    if batting_logs_df.empty or pitching_logs_df.empty:
        return pd.DataFrame(columns=["team", "offense_vs_lhp", "offense_vs_rhp", "games_vs_lhp", "games_vs_rhp"])

    batting_logs = batting_logs_df.copy()
    pitching_logs = pitching_logs_df.copy()
    batting_logs["game_date"] = pd.to_datetime(batting_logs["game_date"])
    pitching_logs["game_date"] = pd.to_datetime(pitching_logs["game_date"])

    hand_lookup = pitching_logs.set_index(["game_id", "team"])["starting_pitcher_hand"].to_dict()
    batting_logs["opponent_starter_hand"] = batting_logs.apply(
        lambda row: hand_lookup.get((row["game_id"], row["opponent_team"])),
        axis=1,
    )
    batting_logs["offense_score"] = batting_logs.apply(_offense_game_score, axis=1)
    games_df = batting_logs[["team", "opponent_starter_hand", "offense_score"]].copy()
    if games_df.empty:
        return pd.DataFrame(columns=["team", "offense_vs_lhp", "offense_vs_rhp", "games_vs_lhp", "games_vs_rhp"])

    rows: list[dict[str, object]] = []
    for team, group in games_df.groupby("team", sort=True):
        vs_left = group[group["opponent_starter_hand"] == "L"]
        vs_right = group[group["opponent_starter_hand"] == "R"]
        rows.append(
            {
                "team": team,
                "offense_vs_lhp": _smoothed_offense_score(vs_left["offense_score"], len(vs_left)),
                "offense_vs_rhp": _smoothed_offense_score(vs_right["offense_score"], len(vs_right)),
                "games_vs_lhp": int(len(vs_left)),
                "games_vs_rhp": int(len(vs_right)),
            }
        )
    return pd.DataFrame(rows)


def offense_split_value(split_row: dict[str, object] | None, pitcher_hand: object) -> float:
    if not split_row:
        return OFFENSE_SPLIT_PRIOR
    hand = normalize_pitcher_hand(pitcher_hand)
    if hand == "L":
        return float(split_row.get("offense_vs_lhp", OFFENSE_SPLIT_PRIOR))
    return float(split_row.get("offense_vs_rhp", OFFENSE_SPLIT_PRIOR))


def _smoothed_offense_score(values: pd.Series, games: int) -> float:
    if games <= 0:
        return OFFENSE_SPLIT_PRIOR
    return float((values.sum() + (OFFENSE_SPLIT_PRIOR * OFFENSE_SPLIT_PRIOR_GAMES)) / (games + OFFENSE_SPLIT_PRIOR_GAMES))


def offense_score_from_stats(stats: dict[str, object]) -> float:
    at_bats = float(stats.get("at_bats", 0.0) or 0.0)
    if at_bats == 0:
        return OFFENSE_SPLIT_PRIOR
    hits = float(stats.get("hits", 0.0) or 0.0)
    walks = float(stats.get("walks", 0.0) or 0.0)
    strikeouts = float(stats.get("strikeouts", 0.0) or 0.0)
    doubles = float(stats.get("doubles", 0.0) or 0.0)
    triples = float(stats.get("triples", 0.0) or 0.0)
    home_runs = float(stats.get("home_runs", 0.0) or 0.0)

    plate_appearances = max(at_bats + walks, 1.0)
    total_bases = hits + doubles + (2.0 * triples) + (3.0 * home_runs)
    obp = (hits + walks) / plate_appearances
    slugging = total_bases / at_bats
    walk_rate = walks / plate_appearances
    strikeout_rate = strikeouts / plate_appearances
    return float((0.45 * obp) + (0.35 * slugging) + (0.15 * walk_rate) - (0.10 * strikeout_rate))


def _offense_game_score(row: pd.Series) -> float:
    return offense_score_from_stats(row.to_dict())


