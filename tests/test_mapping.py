import pandas as pd

from mlpm.normalize.mapping import canonicalize_team_name, map_odds_to_games


def test_canonicalize_team_name() -> None:
    assert canonicalize_team_name("Los Angeles Dodgers") == "los angeles dodgers"


def test_map_odds_to_games_matches_on_team_names_and_time() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "1",
                "home_team": "Chicago Cubs",
                "away_team": "St. Louis Cardinals",
                "event_start_time": "2026-04-15T18:00:00Z",
            }
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "home_team": "Chicago Cubs",
                "away_team": "St. Louis Cardinals",
                "event_start_time": "2026-04-15T18:10:00Z",
                "outcome_team": "Chicago Cubs",
                "raw_odds": -120,
                "bookmaker": "draftkings",
                "event_id": "abc",
                "market_type": "h2h",
                "snapshot_ts": "2026-04-15T17:55:00Z",
            }
        ]
    )
    result = map_odds_to_games(odds, games)
    assert len(result) == 1
    assert result.iloc[0]["game_id"] == "1"
