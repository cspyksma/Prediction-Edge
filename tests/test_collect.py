from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
import uuid

import pandas as pd

from mlpm.pipeline.collect import collect_snapshot


class _FakeConnection:
    def close(self) -> None:
        return None


def _workspace_dir(name: str) -> Path:
    path = Path(".tmp") / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_settings(workspace_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        mlb_lookahead_days=3,
        raw_data_dir=workspace_dir / "raw",
        duckdb_path=workspace_dir / "test.duckdb",
        freshness_window_seconds=3600,
        discrepancy_threshold_bps=25,
    )


def test_collect_snapshot_happy_path(monkeypatch) -> None:
    appended: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr("mlpm.pipeline.collect.settings", lambda: _fake_settings(_workspace_dir("collect-happy")))
    monkeypatch.setattr(
        "mlpm.pipeline.collect.fetch_upcoming_games",
        lambda lookahead_days: pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-04-17",
                    "event_start_time": "2026-04-17T18:00:00Z",
                    "snapshot_ts": "2026-04-17T10:00:00Z",
                    "away_team": "Away",
                    "home_team": "Home",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "mlpm.pipeline.collect.fetch_mlb_markets",
        lambda: (
            pd.DataFrame([{"market_ticker": "k1"}]),
            [{"source": "kalshi"}],
        ),
    )
    monkeypatch.setattr(
        "mlpm.pipeline.collect.fetch_polymarket_markets",
        lambda: (
            pd.DataFrame([{"market_id": "p1"}]),
            [{"source": "polymarket"}],
        ),
    )
    monkeypatch.setattr(
        "mlpm.pipeline.collect.map_kalshi_to_games",
        lambda markets, games: pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "event_id": "ke1",
                    "market_ticker": "k1",
                    "event_start_time": "2026-04-17T18:00:00Z",
                    "snapshot_ts": "2026-04-17T10:00:00Z",
                    "outcome_team": "Home",
                    "yes_bid": 0.52,
                    "yes_ask": 0.54,
                    "last_price": 0.53,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "mlpm.pipeline.collect.map_market_text_to_games",
        lambda markets, games, question_field: pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "event_id": "pe1",
                    "market_id": "p1",
                    "event_start_time": "2026-04-17T18:00:00Z",
                    "snapshot_ts": "2026-04-17T10:00:00Z",
                    "outcome_team": "Away",
                    "last_price": 0.47,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "mlpm.pipeline.collect.build_model_probabilities",
        lambda games_df, market_priors_df=None: pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "snapshot_ts": pd.Timestamp("2026-04-17T12:00:00Z"),
                    "team": "Home",
                    "model_name": "model",
                    "model_prob": 0.60,
                    "games_played_floor_pass": True,
                    "opponent_team": "Away",
                }
            ]
        ),
    )
    monkeypatch.setattr("mlpm.pipeline.collect.select_champion_model", lambda: "model")
    monkeypatch.setattr(
        "mlpm.pipeline.collect.build_bet_opportunities",
        lambda *args, **kwargs: pd.DataFrame([{"game_id": "g1", "team": "Home", "is_actionable": True}]),
    )
    monkeypatch.setattr("mlpm.pipeline.collect.write_raw_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlpm.pipeline.collect.connect", lambda path: _FakeConnection())
    monkeypatch.setattr("mlpm.pipeline.collect.append_dataframe", lambda conn, table, frame: appended.setdefault(table, frame.copy()))

    result = collect_snapshot()

    assert result["games"] == 1
    assert result["kalshi_quotes"] == 1
    assert result["polymarket_quotes"] == 1
    assert result["normalized_quotes"] == 2
    assert result["model_predictions"] == 1
    assert result["bet_opportunities"] == 1
    assert {"games", "raw_snapshots", "normalized_quotes", "model_predictions", "bet_opportunities", "discrepancies"} <= set(appended)


def test_collect_snapshot_continues_when_kalshi_fails(monkeypatch, caplog) -> None:
    monkeypatch.setattr("mlpm.pipeline.collect.settings", lambda: _fake_settings(_workspace_dir("collect-kalshi-fail")))
    monkeypatch.setattr(
        "mlpm.pipeline.collect.fetch_upcoming_games",
        lambda lookahead_days: pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-04-17",
                    "event_start_time": "2026-04-17T18:00:00Z",
                    "snapshot_ts": "2026-04-17T10:00:00Z",
                    "away_team": "Away",
                    "home_team": "Home",
                }
            ]
        ),
    )
    monkeypatch.setattr("mlpm.pipeline.collect.fetch_mlb_markets", lambda: (_ for _ in ()).throw(RuntimeError("kalshi down")))
    monkeypatch.setattr(
        "mlpm.pipeline.collect.fetch_polymarket_markets",
        lambda: (pd.DataFrame([{"market_id": "p1"}]), [{"source": "polymarket"}]),
    )
    monkeypatch.setattr(
        "mlpm.pipeline.collect.map_market_text_to_games",
        lambda markets, games, question_field: pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "event_id": "pe1",
                    "market_id": "p1",
                    "event_start_time": "2026-04-17T18:00:00Z",
                    "snapshot_ts": "2026-04-17T10:00:00Z",
                    "outcome_team": "Away",
                    "last_price": 0.47,
                }
            ]
        ),
    )
    monkeypatch.setattr("mlpm.pipeline.collect.build_model_probabilities", lambda games_df, market_priors_df=None: pd.DataFrame())
    monkeypatch.setattr("mlpm.pipeline.collect.select_champion_model", lambda: "model")
    monkeypatch.setattr("mlpm.pipeline.collect.build_bet_opportunities", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.pipeline.collect.write_raw_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlpm.pipeline.collect.connect", lambda path: _FakeConnection())
    monkeypatch.setattr("mlpm.pipeline.collect.append_dataframe", lambda conn, table, frame: None)

    with caplog.at_level(logging.WARNING):
        result = collect_snapshot()

    assert result["kalshi_quotes"] == 0
    assert result["polymarket_quotes"] == 1
    assert result["normalized_quotes"] == 1
    assert "Kalshi market collection failed" in caplog.text


def test_collect_snapshot_handles_no_games(monkeypatch, caplog) -> None:
    monkeypatch.setattr("mlpm.pipeline.collect.settings", lambda: _fake_settings(_workspace_dir("collect-no-games")))
    monkeypatch.setattr("mlpm.pipeline.collect.fetch_upcoming_games", lambda lookahead_days: pd.DataFrame())
    monkeypatch.setattr("mlpm.pipeline.collect.fetch_mlb_markets", lambda: (pd.DataFrame(), []))
    monkeypatch.setattr("mlpm.pipeline.collect.fetch_polymarket_markets", lambda: (pd.DataFrame(), []))
    monkeypatch.setattr("mlpm.pipeline.collect.map_kalshi_to_games", lambda markets, games: pd.DataFrame())
    monkeypatch.setattr("mlpm.pipeline.collect.map_market_text_to_games", lambda markets, games, question_field: pd.DataFrame())
    monkeypatch.setattr("mlpm.pipeline.collect.build_model_probabilities", lambda games_df, market_priors_df=None: pd.DataFrame())
    monkeypatch.setattr("mlpm.pipeline.collect.select_champion_model", lambda: "model")
    monkeypatch.setattr("mlpm.pipeline.collect.build_bet_opportunities", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.pipeline.collect.write_raw_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr("mlpm.pipeline.collect.connect", lambda path: _FakeConnection())
    monkeypatch.setattr("mlpm.pipeline.collect.append_dataframe", lambda conn, table, frame: None)

    with caplog.at_level(logging.WARNING):
        result = collect_snapshot()

    assert result == {
        "games": 0,
        "kalshi_quotes": 0,
        "polymarket_quotes": 0,
        "model_predictions": 0,
        "normalized_quotes": 0,
        "discrepancies": 0,
        "bet_opportunities": 0,
    }
    assert "Snapshot collection found no upcoming MLB games." in caplog.text
