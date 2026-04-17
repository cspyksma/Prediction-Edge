from __future__ import annotations

from pathlib import Path
import uuid

import pandas as pd

from mlpm.config.settings import settings
from mlpm.evaluation.strategy import (
    build_bet_opportunities,
    run_bet_opportunity_report,
    run_strategy_performance_report,
    select_champion_model,
)
from mlpm.storage.duckdb import append_dataframe, connect


def _workspace_db_path(name: str) -> Path:
    path = Path(".tmp") / f"{name}-{uuid.uuid4().hex}.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_build_bet_opportunities_uses_best_available_market(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_EDGE_THRESHOLD_BPS", "500")
    settings.cache_clear()
    games = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_date": "2026-04-15",
                "event_start_time": "2026-04-15T18:00:00",
                "home_team": "Home",
                "away_team": "Away",
            }
        ]
    )
    normalized_quotes = pd.DataFrame(
        [
            {"game_id": "g1", "source": "kalshi", "market_id": "k1", "outcome_team": "Home", "fair_prob": 0.67, "is_valid": True, "is_pregame": True, "snapshot_ts": "2026-04-15T12:00:00"},
            {"game_id": "g1", "source": "polymarket", "market_id": "p1", "outcome_team": "Home", "fair_prob": 0.61, "is_valid": True, "is_pregame": True, "snapshot_ts": "2026-04-15T12:00:00"},
            {"game_id": "g1", "source": "kalshi", "market_id": "k2", "outcome_team": "Away", "fair_prob": 0.39, "is_valid": True, "is_pregame": True, "snapshot_ts": "2026-04-15T12:00:00"},
        ]
    )
    model_predictions = pd.DataFrame(
        [
            {"game_id": "g1", "snapshot_ts": "2026-04-15T12:00:00", "collection_run_ts": "2026-04-15T12:00:00", "team": "Home", "opponent_team": "Away", "model_name": "champ", "model_prob": 0.70},
            {"game_id": "g1", "snapshot_ts": "2026-04-15T12:00:00", "collection_run_ts": "2026-04-15T12:00:00", "team": "Away", "opponent_team": "Home", "model_name": "champ", "model_prob": 0.30},
        ]
    )

    opportunities = build_bet_opportunities(games, normalized_quotes, model_predictions, champion_model_name="champ")

    assert len(opportunities) == 1
    row = opportunities.iloc[0]
    assert row["source"] == "polymarket"
    assert row["team"] == "Home"
    assert row["edge_bps"] == 900
    assert bool(row["is_actionable"]) is True
    assert bool(row["is_champion"]) is True


def test_strategy_reports_compute_roi_and_champion(monkeypatch) -> None:
    db_path = _workspace_db_path("strategy")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("STRATEGY_CHAMPION_MIN_BETS", "1")
    settings.cache_clear()
    conn = connect(settings().duckdb_path)
    append_dataframe(
        conn,
        "bet_opportunities",
        pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-04-14",
                    "event_start_time": "2026-04-14T18:00:00",
                    "snapshot_ts": "2026-04-14T17:00:00",
                    "collection_run_ts": "2026-04-14T17:00:00",
                    "model_name": "model_a",
                    "source": "kalshi",
                    "market_id": "m1",
                    "team": "Home",
                    "opponent_team": "Away",
                    "is_home_team": True,
                    "model_prob": 0.70,
                    "market_prob": 0.60,
                    "edge_bps": 1000,
                    "expected_value": (0.70 / 0.60) - 1.0,
                    "implied_decimal_odds": 1 / 0.60,
                    "stake_units": 1.0,
                    "is_actionable": True,
                    "is_champion": False,
                },
                {
                    "game_id": "g2",
                    "game_date": "2026-04-15",
                    "event_start_time": "2026-04-15T18:00:00",
                    "snapshot_ts": "2026-04-15T17:00:00",
                    "collection_run_ts": "2026-04-15T17:00:00",
                    "model_name": "model_b",
                    "source": "polymarket",
                    "market_id": "m2",
                    "team": "Away2",
                    "opponent_team": "Home2",
                    "is_home_team": False,
                    "model_prob": 0.58,
                    "market_prob": 0.50,
                    "edge_bps": 800,
                    "expected_value": (0.58 / 0.50) - 1.0,
                    "implied_decimal_odds": 2.0,
                    "stake_units": 1.0,
                    "is_actionable": True,
                    "is_champion": False,
                },
            ]
        ),
    )
    append_dataframe(
        conn,
        "game_results",
        pd.DataFrame(
            [
                {"game_id": "g1", "game_date": "2026-04-14", "away_team": "Away", "home_team": "Home", "winner_team": "Home", "away_score": 2, "home_score": 4},
                {"game_id": "g2", "game_date": "2026-04-15", "away_team": "Away2", "home_team": "Home2", "winner_team": "Away2", "away_score": 5, "home_score": 3},
            ]
        ),
    )
    conn.close()

    champion = select_champion_model()
    opportunities = run_bet_opportunity_report("2026-04-14", "2026-04-15")
    strategy = run_strategy_performance_report("2026-04-14", "2026-04-15")

    assert champion in {"model_a", "model_b"}
    assert opportunities["status"] == "ok"
    assert opportunities["models"]["model_a"]["actionable_bets"] == 1
    assert strategy["status"] == "ok"
    assert strategy["windows"]["model_a"]["all"]["bets"] == 1
    assert strategy["windows"]["model_a"]["all"]["units_won"] > 0
    assert strategy["champion_model"] in {"model_a", "model_b"}


def test_run_strategy_performance_report_parameterizes_model_name(monkeypatch) -> None:
    db_path = _workspace_db_path("strategy-params")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("STRATEGY_CHAMPION_MIN_BETS", "1")
    settings.cache_clear()
    conn = connect(settings().duckdb_path)
    append_dataframe(
        conn,
        "bet_opportunities",
        pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-04-14",
                    "event_start_time": "2026-04-14T18:00:00",
                    "snapshot_ts": "2026-04-14T17:00:00",
                    "collection_run_ts": "2026-04-14T17:00:00",
                    "model_name": "safe_model",
                    "source": "kalshi",
                    "market_id": "m1",
                    "team": "Home",
                    "opponent_team": "Away",
                    "is_home_team": True,
                    "model_prob": 0.7,
                    "market_prob": 0.6,
                    "edge_bps": 1000,
                    "expected_value": 0.1,
                    "implied_decimal_odds": 1.6,
                    "stake_units": 1.0,
                    "is_actionable": True,
                    "is_champion": False,
                }
            ]
        ),
    )
    append_dataframe(
        conn,
        "game_results",
        pd.DataFrame(
            [
                {"game_id": "g1", "game_date": "2026-04-14", "away_team": "Away", "home_team": "Home", "winner_team": "Home", "away_score": 2, "home_score": 4}
            ]
        ),
    )
    conn.close()

    result = run_strategy_performance_report("2026-04-14", "2026-04-14", model_name="safe_model' OR 1=1 --")

    assert result == {"status": "insufficient_data", "rows": 0}
