import pandas as pd

from mlpm.features.team_strength import build_team_feature_table


def test_build_team_feature_table_computes_recent_form_and_run_diff() -> None:
    results = pd.DataFrame(
        [
            {
                "game_id": "1",
                "game_date": "2026-04-01",
                "away_team": "A",
                "home_team": "B",
                "winner_team": "A",
                "away_score": 5,
                "home_score": 3,
            },
            {
                "game_id": "2",
                "game_date": "2026-04-02",
                "away_team": "A",
                "home_team": "C",
                "winner_team": "C",
                "away_score": 1,
                "home_score": 4,
            },
            {
                "game_id": "3",
                "game_date": "2026-04-03",
                "away_team": "B",
                "home_team": "A",
                "winner_team": "A",
                "away_score": 2,
                "home_score": 6,
            },
        ]
    )

    features = build_team_feature_table(results)
    team_a = features[features["team"] == "A"].iloc[0]

    assert team_a["games_played"] == 3
    assert team_a["wins"] == 2
    assert round(team_a["run_diff_per_game"], 4) == round((2 - 3 + 4) / 3, 4)
    assert round(team_a["season_runs_scored_per_game"], 4) == round((5 + 1 + 6) / 3, 4)
    assert round(team_a["season_runs_allowed_per_game"], 4) == round((3 + 4 + 2) / 3, 4)
    assert team_a["streak"] == 1
    assert team_a["recent_games"] == 3


def test_build_team_feature_table_tracks_home_and_away_splits() -> None:
    results = pd.DataFrame(
        [
            {
                "game_id": "1",
                "game_date": "2026-04-01",
                "away_team": "A",
                "home_team": "B",
                "winner_team": "B",
                "away_score": 1,
                "home_score": 3,
            },
            {
                "game_id": "2",
                "game_date": "2026-04-02",
                "away_team": "C",
                "home_team": "B",
                "winner_team": "B",
                "away_score": 2,
                "home_score": 4,
            },
            {
                "game_id": "3",
                "game_date": "2026-04-03",
                "away_team": "B",
                "home_team": "D",
                "winner_team": "D",
                "away_score": 0,
                "home_score": 1,
            },
        ]
    )

    features = build_team_feature_table(results)
    team_b = features[features["team"] == "B"].iloc[0]

    assert team_b["home_games"] == 2
    assert team_b["away_games"] == 1
    assert team_b["home_win_pct"] > team_b["away_win_pct"]
