from __future__ import annotations

import logging

import pandas as pd

from mlpm.features.market_priors import build_market_prior_frame
from mlpm.features.utils import coalesce_float, current_streak, smoothed_win_pct
from mlpm.models.fair_value import build_team_matchups, build_team_strengths


def test_build_team_strengths_logs_when_games_empty(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        result = build_team_strengths(pd.DataFrame())

    assert result.empty
    assert "Team strengths unavailable; games frame is empty." in caplog.text


def test_build_team_matchups_logs_when_logs_missing(monkeypatch, caplog) -> None:
    games = pd.DataFrame([{"game_id": "g1", "game_date": "2026-04-17"}])
    monkeypatch.setattr("mlpm.models.fair_value.fetch_game_batting_logs", lambda start_date, end_date: pd.DataFrame())
    monkeypatch.setattr("mlpm.models.fair_value.fetch_game_pitching_logs", lambda start_date, end_date: pd.DataFrame())

    with caplog.at_level(logging.WARNING):
        result = build_team_matchups(games)

    assert result.empty
    assert "Team matchups unavailable; batting_rows=0 pitching_rows=0 season=2026" in caplog.text


def test_build_market_prior_frame_logs_when_filters_drop_all_quotes(caplog) -> None:
    games = pd.DataFrame([{"game_id": "g1", "home_team": "Home"}])
    quotes = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "outcome_team": "Home",
                "fair_prob": 0.55,
                "source": "kalshi",
                "is_valid": False,
                "is_pregame": True,
            }
        ]
    )

    with caplog.at_level(logging.WARNING):
        result = build_market_prior_frame(games, quotes)

    assert result.empty
    assert "Market prior frame filtered to zero rows after validity/pregame checks" in caplog.text


def test_shared_feature_utils_preserve_expected_behavior() -> None:
    assert smoothed_win_pct(3, 4) == 4 / 6
    assert current_streak([True, True, False, False]) == -2
    assert coalesce_float(None) is None
    assert coalesce_float(1.25) == 1.25
