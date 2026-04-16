from __future__ import annotations

import pandas as pd


def build_pitching_feature_table(
    games_df: pd.DataFrame,
    pitcher_stats_df: pd.DataFrame,
    bullpen_usage_df: pd.DataFrame,
) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    games = games_df.copy()
    if not pitcher_stats_df.empty:
        games["away_probable_pitcher_id"] = pd.to_numeric(games["away_probable_pitcher_id"], errors="coerce").astype("Int64")
        games["home_probable_pitcher_id"] = pd.to_numeric(games["home_probable_pitcher_id"], errors="coerce").astype("Int64")
        pitcher_stats_df = pitcher_stats_df.copy()
        pitcher_stats_df["pitcher_id"] = pd.to_numeric(pitcher_stats_df["pitcher_id"], errors="coerce").astype("Int64")
        away_pitchers = pitcher_stats_df.add_prefix("away_pitcher_")
        home_pitchers = pitcher_stats_df.add_prefix("home_pitcher_")
        games = games.merge(
            away_pitchers,
            left_on="away_probable_pitcher_id",
            right_on="away_pitcher_pitcher_id",
            how="left",
        )
        games = games.merge(
            home_pitchers,
            left_on="home_probable_pitcher_id",
            right_on="home_pitcher_pitcher_id",
            how="left",
        )
    if not bullpen_usage_df.empty:
        away_bullpen = bullpen_usage_df.add_prefix("away_")
        home_bullpen = bullpen_usage_df.add_prefix("home_")
        games = games.merge(away_bullpen, left_on="away_team", right_on="away_team", how="left")
        games = games.merge(home_bullpen, left_on="home_team", right_on="home_team", how="left")

    fill_zero_columns = [
        "away_bullpen_innings_3d",
        "away_bullpen_pitches_3d",
        "away_relievers_used_3d",
        "away_bullpen_innings_1d",
        "away_bullpen_pitches_1d",
        "away_relievers_used_1d",
        "home_bullpen_innings_3d",
        "home_bullpen_pitches_3d",
        "home_relievers_used_3d",
        "home_bullpen_innings_1d",
        "home_bullpen_pitches_1d",
        "home_relievers_used_1d",
    ]
    for column in fill_zero_columns:
        if column in games.columns:
            games[column] = games[column].fillna(0.0)
    return games
