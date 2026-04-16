import pandas as pd

from mlpm.features.pitching import build_pitching_feature_table
from mlpm.ingest.mlb_stats import _innings_to_float


def test_innings_to_float_handles_partial_innings() -> None:
    assert _innings_to_float("6.0") == 6.0
    assert _innings_to_float("7.1") == 7 + (1 / 3)
    assert _innings_to_float("5.2") == 5 + (2 / 3)


def test_build_pitching_feature_table_merges_pitcher_and_bullpen_features() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "1",
                "away_team": "Away",
                "home_team": "Home",
                "away_probable_pitcher_id": 10,
                "home_probable_pitcher_id": 20,
            }
        ]
    )
    pitcher_stats = pd.DataFrame(
        [
            {"pitcher_id": 10, "pitcher_name": "Away SP", "era": 3.5, "whip": 1.1, "strikeouts_per_9": 9.0, "walks_per_9": 2.0},
            {"pitcher_id": 20, "pitcher_name": "Home SP", "era": 4.0, "whip": 1.2, "strikeouts_per_9": 8.5, "walks_per_9": 2.5},
        ]
    )
    bullpen_usage = pd.DataFrame(
        [
            {"team": "Away", "bullpen_innings_3d": 8.0, "bullpen_pitches_3d": 120, "relievers_used_3d": 6, "games_counted": 3, "bullpen_innings_1d": 2.0, "bullpen_pitches_1d": 30, "relievers_used_1d": 2},
            {"team": "Home", "bullpen_innings_3d": 5.0, "bullpen_pitches_3d": 80, "relievers_used_3d": 4, "games_counted": 3, "bullpen_innings_1d": 1.0, "bullpen_pitches_1d": 15, "relievers_used_1d": 1},
        ]
    )

    merged = build_pitching_feature_table(games, pitcher_stats, bullpen_usage)
    row = merged.iloc[0]

    assert row["away_pitcher_era"] == 3.5
    assert row["home_pitcher_whip"] == 1.2
    assert row["away_bullpen_innings_3d"] == 8.0
    assert row["home_relievers_used_1d"] == 1
