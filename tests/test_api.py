from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
from fastapi.testclient import TestClient

from mlpm.api.app import create_app
from mlpm.api import jobs as api_jobs
from mlpm.api.services import get_model_roster, get_research_contenders, get_research_strategies, get_training_coverage
from mlpm.storage.duckdb import append_dataframe, connect


def _seed_db(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        append_dataframe(
            conn,
            "games",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "game_date": "2026-04-23",
                        "event_start_time": "2026-04-23T18:05:00",
                        "away_team": "Chicago Cubs",
                        "home_team": "St. Louis Cardinals",
                        "away_team_id": 1,
                        "home_team_id": 2,
                        "away_probable_pitcher_id": 10,
                        "away_probable_pitcher_name": "Away Pitcher",
                        "away_probable_pitcher_hand": "R",
                        "home_probable_pitcher_id": 11,
                        "home_probable_pitcher_name": "Home Pitcher",
                        "home_probable_pitcher_hand": "L",
                        "doubleheader": "N",
                        "game_number": 1,
                        "day_night": "night",
                        "status": "scheduled",
                        "snapshot_ts": "2026-04-23T15:00:00",
                        "collection_run_ts": "2026-04-23T15:00:00",
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "discrepancies",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "snapshot_ts": "2026-04-23T15:00:00",
                        "collection_run_ts": "2026-04-23T15:00:00",
                        "source": "kalshi",
                        "market_id": "m1",
                        "team": "Chicago Cubs",
                        "market_prob": 0.44,
                        "model_prob": 0.52,
                        "gap_bps": 800,
                        "freshness_pass": True,
                        "mapping_pass": True,
                        "flagged": True,
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "bet_opportunities",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "game_date": "2026-04-23",
                        "event_start_time": "2026-04-23T18:05:00",
                        "snapshot_ts": "2026-04-23T15:00:00",
                        "collection_run_ts": "2026-04-23T15:00:00",
                        "model_name": "mlb_win_histgb_v1",
                        "source": "kalshi",
                        "market_id": "m1",
                        "team": "Chicago Cubs",
                        "opponent_team": "St. Louis Cardinals",
                        "is_home_team": False,
                        "model_prob": 0.52,
                        "market_prob": 0.44,
                        "edge_bps": 800,
                        "expected_value": 0.11,
                        "implied_decimal_odds": 2.27,
                        "stake_units": 1.0,
                        "is_actionable": True,
                        "is_champion": True,
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "model_predictions",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "snapshot_ts": "2026-04-23T15:00:00",
                        "collection_run_ts": "2026-04-23T15:00:00",
                        "team": "Chicago Cubs",
                        "model_name": "mlb_win_histgb_v1",
                        "model_prob": 0.52,
                        "games_played_floor_pass": True,
                        "opponent_team": "St. Louis Cardinals",
                        "season_win_pct": 0.6,
                        "recent_win_pct": 0.7,
                        "venue_win_pct": 0.4,
                        "run_diff_per_game": 1.4,
                        "streak": 3,
                        "elo_rating": 1540.0,
                        "season_runs_scored_per_game": 4.7,
                        "season_runs_allowed_per_game": 3.9,
                        "recent_runs_scored_per_game": 5.2,
                        "recent_runs_allowed_per_game": 3.3,
                        "rest_days": 1.0,
                        "venue_streak": 0.0,
                        "travel_switch": 1.0,
                        "is_doubleheader": False,
                        "starter_era": 3.2,
                        "starter_whip": 1.08,
                        "starter_strikeouts_per_9": 10.1,
                        "starter_walks_per_9": 2.2,
                        "bullpen_innings_3d": 8.0,
                        "bullpen_pitches_3d": 118.0,
                        "relievers_used_3d": 5.0,
                        "market_home_implied_prob": 0.56,
                        "offense_vs_starter_hand": 1.2,
                    },
                    {
                        "game_id": "g1",
                        "snapshot_ts": "2026-04-23T15:00:00",
                        "collection_run_ts": "2026-04-23T15:00:00",
                        "team": "Chicago Cubs",
                        "model_name": "mlb_win_mlp_v1",
                        "model_prob": 0.57,
                        "games_played_floor_pass": True,
                        "opponent_team": "St. Louis Cardinals",
                        "season_win_pct": 0.6,
                        "recent_win_pct": 0.7,
                        "venue_win_pct": 0.4,
                        "run_diff_per_game": 1.4,
                        "streak": 3,
                        "elo_rating": 1540.0,
                        "season_runs_scored_per_game": 4.7,
                        "season_runs_allowed_per_game": 3.9,
                        "recent_runs_scored_per_game": 5.2,
                        "recent_runs_allowed_per_game": 3.3,
                        "rest_days": 1.0,
                        "venue_streak": 0.0,
                        "travel_switch": 1.0,
                        "is_doubleheader": False,
                        "starter_era": 3.05,
                        "starter_whip": 1.02,
                        "starter_strikeouts_per_9": 10.6,
                        "starter_walks_per_9": 2.0,
                        "bullpen_innings_3d": 7.5,
                        "bullpen_pitches_3d": 112.0,
                        "relievers_used_3d": 4.0,
                        "market_home_implied_prob": 0.56,
                        "offense_vs_starter_hand": 1.2,
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "collector_runs",
            pd.DataFrame(
                [
                    {
                        "run_id": "r1",
                        "started_at": "2026-04-23T15:00:00",
                        "completed_at": "2026-04-23T15:01:00",
                        "status": "success",
                        "games": 1,
                        "normalized_quotes": 1,
                        "discrepancies": 1,
                        "game_results_synced": 1,
                        "bet_opportunities": 1,
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "historical_import_runs",
            pd.DataFrame(
                [
                    {
                        "import_run_id": "hist-1",
                        "source": "kalshi",
                        "started_at": "2026-04-22T12:00:00",
                        "completed_at": "2026-04-22T12:15:00",
                        "start_date": "2026-04-20",
                        "end_date": "2026-04-22",
                        "status": "success",
                        "request_count": 4,
                        "payload_count": 4,
                        "normalized_rows": 100,
                        "games_total": 10,
                        "games_with_markets": 9,
                        "games_with_pregame_quotes": 8,
                        "candidate_markets": 15,
                        "empty_payload_count": 0,
                        "rate_limited_count": 0,
                        "parse_error_count": 0,
                        "error_message": None,
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "game_weather",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "game_date": "2026-04-23",
                        "venue_team": "St. Louis Cardinals",
                        "az_cf_deg": 120.0,
                        "roof_type": "open",
                        "temp_f": 72.0,
                        "wind_mph": 5.0,
                        "wind_dir_deg": 190.0,
                        "wind_out_to_cf_mph": 1.0,
                        "wind_crossfield_mph": 0.4,
                        "humidity_pct": 42.0,
                        "precipitation_in": 0.0,
                        "is_dome_sealed": 0,
                        "imported_at": "2026-04-23T14:00:00",
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "feature_importances",
            pd.DataFrame(
                [
                    {
                        "trained_at": "2026-04-23T10:00:00",
                        "model_name": "mlb_win_histgb_v1",
                        "feature": "elo_rating",
                        "importance": 0.18,
                        "importance_std": 0.02,
                        "rank": 1,
                        "train_start_date": "2015-03-01",
                        "train_end_date": "2026-04-22",
                        "rows_train": 1000,
                        "rows_valid": 100,
                        "method": "permutation",
                    }
                ]
            ),
        )
    finally:
        conn.close()


def test_api_endpoints_return_dashboard_payloads(monkeypatch) -> None:
    db_path = Path(".tmp") / f"api-{uuid4().hex}.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_db(db_path)

    cfg = SimpleNamespace(
        duckdb_path=db_path,
        freshness_window_seconds=999999,
        frontend_dev_origin="http://127.0.0.1:5173",
    )
    monkeypatch.setattr("mlpm.api.services.settings", lambda: cfg)
    monkeypatch.setattr("mlpm.api.app.settings", lambda: cfg)
    monkeypatch.setattr("mlpm.api.jobs.settings", lambda: cfg)
    monkeypatch.setattr("mlpm.api.services.select_champion_model", lambda: "mlb_win_histgb_v1")

    app = create_app()
    client = TestClient(app)

    summary = client.get("/api/v1/summary")
    assert summary.status_code == 200
    assert summary.json()["champion_model"] == "mlb_win_histgb_v1"
    assert summary.json()["actionable_bets"] == 1

    opportunities = client.get("/api/v1/cockpit/opportunities")
    assert opportunities.status_code == 200
    assert opportunities.json()["total"] == 1
    assert opportunities.json()["items"][0]["team"] == "Chicago Cubs"

    game_detail = client.get("/api/v1/cockpit/games/g1")
    assert game_detail.status_code == 200
    assert game_detail.json()["game_id"] == "g1"
    assert len(game_detail.json()["quotes"]) == 1
    returned_models = {row["model_name"] for row in game_detail.json()["features"]}
    assert returned_models == {"mlb_win_histgb_v1", "mlb_win_mlp_v1"}

    contenders = client.get("/api/v1/research/contenders")
    assert contenders.status_code == 200
    assert contenders.json() == []

    feature_importance = client.get("/api/v1/research/feature-importance")
    assert feature_importance.status_code == 200
    assert feature_importance.json()[0]["feature"] == "elo_rating"

    freshness = client.get("/api/v1/ops/freshness")
    assert freshness.status_code == 200
    assert freshness.json()["stale_data"] is False

    import_status = client.get("/api/v1/ops/import-status")
    assert import_status.status_code == 200
    assert import_status.json()[0]["source"] == "kalshi"


def test_cockpit_endpoints_scope_to_latest_collection_run(monkeypatch) -> None:
    db_path = Path(".tmp") / f"api-latest-run-{uuid4().hex}.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_db(db_path)

    conn = connect(db_path)
    try:
        append_dataframe(
            conn,
            "games",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "game_date": "2026-04-24",
                        "event_start_time": "2026-04-24T18:05:00",
                        "away_team": "Chicago Cubs",
                        "home_team": "St. Louis Cardinals",
                        "away_team_id": 1,
                        "home_team_id": 2,
                        "away_probable_pitcher_id": 10,
                        "away_probable_pitcher_name": "Away Pitcher",
                        "away_probable_pitcher_hand": "R",
                        "home_probable_pitcher_id": 11,
                        "home_probable_pitcher_name": "Home Pitcher",
                        "home_probable_pitcher_hand": "L",
                        "doubleheader": "N",
                        "game_number": 1,
                        "day_night": "night",
                        "status": "scheduled",
                        "snapshot_ts": "2026-04-24T15:00:00",
                        "collection_run_ts": "2026-04-24T15:00:00",
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "discrepancies",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "snapshot_ts": "2026-04-24T15:00:00",
                        "collection_run_ts": "2026-04-24T15:00:00",
                        "source": "kalshi",
                        "market_id": "m1",
                        "team": "Chicago Cubs",
                        "market_prob": 0.49,
                        "model_prob": 0.61,
                        "gap_bps": 1200,
                        "freshness_pass": True,
                        "mapping_pass": True,
                        "flagged": True,
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "bet_opportunities",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "game_date": "2026-04-24",
                        "event_start_time": "2026-04-24T18:05:00",
                        "snapshot_ts": "2026-04-24T15:00:00",
                        "collection_run_ts": "2026-04-24T15:00:00",
                        "model_name": "mlb_win_mlp_v1",
                        "source": "kalshi",
                        "market_id": "m1",
                        "team": "Chicago Cubs",
                        "opponent_team": "St. Louis Cardinals",
                        "is_home_team": False,
                        "model_prob": 0.61,
                        "market_prob": 0.49,
                        "edge_bps": 1200,
                        "expected_value": 0.19,
                        "implied_decimal_odds": 2.04,
                        "stake_units": 1.0,
                        "is_actionable": True,
                        "is_champion": True,
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "model_predictions",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "snapshot_ts": "2026-04-24T15:00:00",
                        "collection_run_ts": "2026-04-24T15:00:00",
                        "team": "Chicago Cubs",
                        "model_name": "mlb_win_histgb_v1",
                        "model_prob": 0.54,
                        "games_played_floor_pass": True,
                        "opponent_team": "St. Louis Cardinals",
                        "season_win_pct": 0.6,
                        "recent_win_pct": 0.7,
                        "venue_win_pct": 0.4,
                        "run_diff_per_game": 1.4,
                        "streak": 3,
                        "elo_rating": 1540.0,
                        "season_runs_scored_per_game": 4.7,
                        "season_runs_allowed_per_game": 3.9,
                        "recent_runs_scored_per_game": 5.2,
                        "recent_runs_allowed_per_game": 3.3,
                        "rest_days": 1.0,
                        "venue_streak": 0.0,
                        "travel_switch": 1.0,
                        "is_doubleheader": False,
                        "starter_era": 3.2,
                        "starter_whip": 1.08,
                        "starter_strikeouts_per_9": 10.1,
                        "starter_walks_per_9": 2.2,
                        "bullpen_innings_3d": 8.0,
                        "bullpen_pitches_3d": 118.0,
                        "relievers_used_3d": 5.0,
                        "market_home_implied_prob": 0.56,
                        "offense_vs_starter_hand": 1.2,
                    },
                    {
                        "game_id": "g1",
                        "snapshot_ts": "2026-04-24T15:00:00",
                        "collection_run_ts": "2026-04-24T15:00:00",
                        "team": "Chicago Cubs",
                        "model_name": "mlb_win_mlp_v1",
                        "model_prob": 0.61,
                        "games_played_floor_pass": True,
                        "opponent_team": "St. Louis Cardinals",
                        "season_win_pct": 0.6,
                        "recent_win_pct": 0.7,
                        "venue_win_pct": 0.4,
                        "run_diff_per_game": 1.4,
                        "streak": 3,
                        "elo_rating": 1540.0,
                        "season_runs_scored_per_game": 4.7,
                        "season_runs_allowed_per_game": 3.9,
                        "recent_runs_scored_per_game": 5.2,
                        "recent_runs_allowed_per_game": 3.3,
                        "rest_days": 1.0,
                        "venue_streak": 0.0,
                        "travel_switch": 1.0,
                        "is_doubleheader": False,
                        "starter_era": 3.05,
                        "starter_whip": 1.02,
                        "starter_strikeouts_per_9": 10.6,
                        "starter_walks_per_9": 2.0,
                        "bullpen_innings_3d": 7.5,
                        "bullpen_pitches_3d": 112.0,
                        "relievers_used_3d": 4.0,
                        "market_home_implied_prob": 0.56,
                        "offense_vs_starter_hand": 1.2,
                    },
                ]
            ),
        )
    finally:
        conn.close()

    cfg = SimpleNamespace(
        duckdb_path=db_path,
        freshness_window_seconds=999999,
        frontend_dev_origin="http://127.0.0.1:5173",
    )
    monkeypatch.setattr("mlpm.api.services.settings", lambda: cfg)
    monkeypatch.setattr("mlpm.api.app.settings", lambda: cfg)
    monkeypatch.setattr("mlpm.api.jobs.settings", lambda: cfg)
    monkeypatch.setattr("mlpm.api.services.select_champion_model", lambda: "mlb_win_mlp_v1")

    app = create_app()
    client = TestClient(app)

    summary = client.get("/api/v1/summary")
    assert summary.status_code == 200
    assert summary.json()["actionable_bets"] == 1
    assert summary.json()["max_edge_bps"] == 1200
    assert summary.json()["latest_snapshot_ts"].startswith("2026-04-24T15:00:00")

    opportunities = client.get("/api/v1/cockpit/opportunities")
    assert opportunities.status_code == 200
    assert opportunities.json()["total"] == 1
    assert opportunities.json()["items"][0]["model_name"] == "mlb_win_mlp_v1"
    assert opportunities.json()["items"][0]["edge_bps"] == 1200

    game_detail = client.get("/api/v1/cockpit/games/g1")
    assert game_detail.status_code == 200
    returned_models = {row["model_name"] for row in game_detail.json()["features"]}
    assert returned_models == {"mlb_win_histgb_v1", "mlb_win_mlp_v1"}
    assert game_detail.json()["quotes"][0]["model_prob"] == 0.61


def test_job_endpoints_use_registry(monkeypatch) -> None:
    db_path = Path(".tmp") / f"jobs-{uuid4().hex}.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(
        duckdb_path=db_path,
        frontend_dev_origin="http://127.0.0.1:5173",
    )
    monkeypatch.setattr("mlpm.api.app.settings", lambda: cfg)

    started_at = datetime(2026, 4, 23, 12, 0, 0)
    job = SimpleNamespace(
        id="job-1",
        command="collect-once",
        label="Collect snapshot",
        pid=1234,
        status="running",
        started_at=started_at,
        finished_at=None,
        returncode=None,
        log_path="C:/tmp/job.log",
    )

    monkeypatch.setattr("mlpm.api.app.list_jobs", lambda: [job])
    monkeypatch.setattr("mlpm.api.app.get_job", lambda job_id: job if job_id == "job-1" else None)
    monkeypatch.setattr("mlpm.api.app.read_log_tail", lambda path: "job output")
    monkeypatch.setattr("mlpm.api.app.start_job", lambda command_key: job)

    app = create_app()
    client = TestClient(app)

    jobs = client.get("/api/v1/ops/jobs")
    assert jobs.status_code == 200
    assert jobs.json()[0]["id"] == "job-1"

    job_detail = client.get("/api/v1/ops/jobs/job-1")
    assert job_detail.status_code == 200
    assert job_detail.json()["log_tail"] == "job output"

    launch = client.post("/api/v1/jobs/collect-once")
    assert launch.status_code == 200
    assert launch.json()["job"]["label"] == "Collect snapshot"


def test_training_coverage_counts_distinct_games(monkeypatch) -> None:
    db_path = Path(".tmp") / f"coverage-{uuid4().hex}.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        append_dataframe(
            conn,
            "historical_market_priors",
            pd.DataFrame(
                [
                    {
                        "game_id": "g-overlap",
                        "game_date": "2021-08-01",
                        "source": "sbro",
                        "home_team": "A",
                        "away_team": "B",
                        "home_moneyline_close": -120,
                        "away_moneyline_close": 110,
                        "home_implied_prob_raw": 0.54,
                        "away_implied_prob_raw": 0.46,
                        "home_fair_prob": 0.53,
                        "away_fair_prob": 0.47,
                        "book": "consensus",
                        "imported_at": "2026-04-23T10:00:00",
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "historical_kalshi_quotes",
            pd.DataFrame(
                [
                    {
                        "import_run_id": "r1",
                        "source": "kalshi",
                        "collection_mode": "replay",
                        "market_id": "m1",
                        "event_id": "e1",
                        "ticker": "KXMLBGAME-1",
                        "game_id": "g-overlap",
                        "event_start_time": "2026-04-23T18:05:00",
                        "quote_ts": "2026-04-23T17:30:00",
                        "outcome_team": "A",
                        "side": "yes",
                        "home_implied_prob": 0.55,
                        "raw_prob_yes": 0.55,
                        "quote_type": "mid",
                        "volume": 10.0,
                        "open_interest": 5.0,
                        "best_price_source": "unit-test",
                        "pre_pitch_flag": True,
                        "raw_payload_path": "kalshi.json",
                        "imported_at": "2026-04-23T10:00:00",
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "normalized_quotes",
            pd.DataFrame(
                [
                    {
                        "game_id": "g-overlap",
                        "source": "kalshi",
                        "event_id": "e1",
                        "market_id": "m1",
                        "bookmaker": "kalshi",
                        "event_start_time": "2026-04-23T18:05:00",
                        "snapshot_ts": "2026-04-23T17:30:00",
                        "collection_run_ts": "2026-04-23T17:30:00",
                        "outcome_team": "A",
                        "market_type": "moneyline",
                        "raw_odds": None,
                        "raw_price": 55.0,
                        "implied_prob": 0.55,
                        "fair_prob": 0.54,
                        "quote_age_sec": 0.0,
                        "is_pregame": True,
                        "is_valid": True,
                    },
                    {
                        "game_id": "g-live-only",
                        "source": "polymarket",
                        "event_id": "e2",
                        "market_id": "m2",
                        "bookmaker": "polymarket",
                        "event_start_time": "2026-04-24T18:05:00",
                        "snapshot_ts": "2026-04-24T17:30:00",
                        "collection_run_ts": "2026-04-24T17:30:00",
                        "outcome_team": "C",
                        "market_type": "moneyline",
                        "raw_odds": None,
                        "raw_price": 58.0,
                        "implied_prob": 0.58,
                        "fair_prob": 0.57,
                        "quote_age_sec": 0.0,
                        "is_pregame": True,
                        "is_valid": True,
                    },
                ]
            ),
        )
        append_dataframe(
            conn,
            "feature_importances",
            pd.DataFrame(
                [
                    {
                        "trained_at": "2026-04-23T10:00:00",
                        "model_name": "mlb_win_histgb_v1",
                        "feature": "elo_rating",
                        "importance": 0.18,
                        "importance_std": 0.02,
                        "rank": 1,
                        "train_start_date": "2015-03-01",
                        "train_end_date": "2026-04-22",
                        "rows_train": 1000,
                        "rows_valid": 100,
                        "method": "permutation",
                    }
                ]
            ),
        )
    finally:
        conn.close()

    cfg = SimpleNamespace(
        duckdb_path=db_path,
        model_train_start_date="2015-03-01",
    )
    monkeypatch.setattr("mlpm.api.services.settings", lambda: cfg)

    payload = get_training_coverage()
    assert payload["total_games_with_prior"] == 2


def test_training_coverage_uses_latest_run_metadata(monkeypatch) -> None:
    db_path = Path(".tmp") / f"coverage-latest-{uuid4().hex}.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        append_dataframe(
            conn,
            "feature_importances",
            pd.DataFrame(
                [
                    {
                        "trained_at": "2026-04-23T10:00:00",
                        "model_name": "mlb_win_histgb_v1",
                        "feature": "elo_rating",
                        "importance": 0.18,
                        "importance_std": 0.02,
                        "rank": 1,
                        "train_start_date": "2026-03-06",
                        "train_end_date": "2026-04-18",
                        "rows_train": 100,
                        "rows_valid": 20,
                        "method": "permutation",
                    },
                    {
                        "trained_at": "2026-04-24T08:17:42",
                        "model_name": "mlb_win_histgb_v1",
                        "feature": "elo_rating",
                        "importance": 0.21,
                        "importance_std": 0.03,
                        "rank": 1,
                        "train_start_date": "2015-03-01",
                        "train_end_date": "2026-04-23",
                        "rows_train": 25000,
                        "rows_valid": 6000,
                        "method": "permutation",
                    },
                ]
            ),
        )
    finally:
        conn.close()

    cfg = SimpleNamespace(
        duckdb_path=db_path,
        model_train_start_date="2015-03-01",
    )
    monkeypatch.setattr("mlpm.api.services.settings", lambda: cfg)

    payload = get_training_coverage()
    assert payload["latest_model_train_start"] == "2015-03-01"
    assert payload["latest_model_train_end"] == "2026-04-23"
    assert payload["latest_model_trained_at"].startswith("2026-04-24T08:17:42")


def test_model_roster_falls_back_to_latest_training_run_for_bayes(monkeypatch) -> None:
    db_path = Path(".tmp") / f"model-roster-{uuid4().hex}.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        append_dataframe(
            conn,
            "games",
            pd.DataFrame(
                [
                    {
                        "game_id": "g_old",
                        "game_date": "2024-08-01",
                        "event_start_time": "2024-08-01T18:00:00",
                        "away_team": "Away",
                        "home_team": "Home",
                        "snapshot_ts": "2024-08-01T17:00:00",
                        "collection_run_ts": "2024-08-01T17:00:00",
                    },
                    {
                        "game_id": "g_new",
                        "game_date": "2026-04-20",
                        "event_start_time": "2026-04-20T18:00:00",
                        "away_team": "Away2",
                        "home_team": "Home2",
                        "snapshot_ts": "2026-04-20T17:00:00",
                        "collection_run_ts": "2026-04-20T17:00:00",
                    },
                ]
            ),
        )
        append_dataframe(
            conn,
            "game_results",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "game_date": "2026-04-20",
                        "away_team": "Away",
                        "home_team": "Home",
                        "winner_team": "Home",
                        "away_score": 2,
                        "home_score": 4,
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "model_predictions",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "snapshot_ts": "2026-04-20T17:00:00",
                        "collection_run_ts": "2026-04-20T17:00:00",
                        "team": "Home",
                        "opponent_team": "Away",
                        "model_name": "mlb_win_bayes_v1",
                        "model_prob": 0.61,
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "bet_opportunities",
            pd.DataFrame(
                [
                    {
                        "game_id": "g1",
                        "game_date": "2026-04-20",
                        "event_start_time": "2026-04-20T18:00:00",
                        "snapshot_ts": "2026-04-20T17:00:00",
                        "collection_run_ts": "2026-04-20T17:00:00",
                        "model_name": "mlb_win_bayes_v1",
                        "source": "kalshi",
                        "market_id": "m1",
                        "team": "Home",
                        "opponent_team": "Away",
                        "is_home_team": True,
                        "model_prob": 0.61,
                        "market_prob": 0.53,
                        "edge_bps": 800,
                        "expected_value": 0.15,
                        "implied_decimal_odds": 1.9,
                        "stake_units": 1.0,
                        "is_actionable": True,
                        "is_champion": True,
                    }
                ]
            ),
        )
        append_dataframe(
            conn,
            "feature_importances",
            pd.DataFrame(
                [
                    {
                        "trained_at": "2026-04-24T08:17:42",
                        "model_name": "mlb_win_histgb_v1",
                        "feature": "elo_rating",
                        "importance": 0.21,
                        "importance_std": 0.03,
                        "rank": 1,
                        "train_start_date": "2015-03-01",
                        "train_end_date": "2026-04-23",
                        "rows_train": 25000,
                        "rows_valid": 6000,
                        "method": "permutation",
                    }
                ]
            ),
        )
    finally:
        conn.close()

    cfg = SimpleNamespace(duckdb_path=db_path)
    monkeypatch.setattr("mlpm.api.services.settings", lambda: cfg)
    monkeypatch.setattr("mlpm.api.services.select_champion_model", lambda: "mlb_win_bayes_v1")

    roster = get_model_roster()
    bayes = next(row for row in roster if row["model_name"] == "mlb_win_bayes_v1")
    assert bayes["role"] == "champion"
    assert bayes["train_start_date"] == "2015-03-01"
    assert bayes["train_end_date"] == "2026-04-23"
    assert bayes["trained_at"].startswith("2026-04-24T08:17:42")


def test_research_and_roster_ignore_pre_2025_settled_rows(monkeypatch) -> None:
    db_path = Path(".tmp") / f"api-season-floor-{uuid4().hex}.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        append_dataframe(
            conn,
            "games",
            pd.DataFrame(
                [
                    {
                        "game_id": "g_old",
                        "game_date": "2024-08-01",
                        "event_start_time": "2024-08-01T18:00:00",
                        "away_team": "Away",
                        "home_team": "Home",
                        "snapshot_ts": "2024-08-01T17:00:00",
                        "collection_run_ts": "2024-08-01T17:00:00",
                    },
                    {
                        "game_id": "g_new",
                        "game_date": "2026-04-20",
                        "event_start_time": "2026-04-20T18:00:00",
                        "away_team": "Away2",
                        "home_team": "Home2",
                        "snapshot_ts": "2026-04-20T17:00:00",
                        "collection_run_ts": "2026-04-20T17:00:00",
                    },
                ]
            ),
        )
        append_dataframe(
            conn,
            "game_results",
            pd.DataFrame(
                [
                    {"game_id": "g_old", "game_date": "2024-08-01", "away_team": "Away", "home_team": "Home", "winner_team": "Home", "away_score": 1, "home_score": 2},
                    {"game_id": "g_new", "game_date": "2026-04-20", "away_team": "Away2", "home_team": "Home2", "winner_team": "Home2", "away_score": 1, "home_score": 3},
                ]
            ),
        )
        append_dataframe(
            conn,
            "model_predictions",
            pd.DataFrame(
                [
                    {
                        "game_id": "g_old",
                        "snapshot_ts": "2024-08-01T17:00:00",
                        "collection_run_ts": "2024-08-01T17:00:00",
                        "team": "Home",
                        "opponent_team": "Away",
                        "model_name": "old_model",
                        "model_prob": 0.70,
                    },
                    {
                        "game_id": "g_new",
                        "snapshot_ts": "2026-04-20T17:00:00",
                        "collection_run_ts": "2026-04-20T17:00:00",
                        "team": "Home2",
                        "opponent_team": "Away2",
                        "model_name": "new_model",
                        "model_prob": 0.60,
                    },
                ]
            ),
        )
        append_dataframe(
            conn,
            "bet_opportunities",
            pd.DataFrame(
                [
                    {
                        "game_id": "g_old",
                        "game_date": "2024-08-01",
                        "event_start_time": "2024-08-01T18:00:00",
                        "snapshot_ts": "2024-08-01T17:00:00",
                        "collection_run_ts": "2024-08-01T17:00:00",
                        "model_name": "old_model",
                        "source": "kalshi",
                        "market_id": "m_old",
                        "team": "Home",
                        "opponent_team": "Away",
                        "is_home_team": True,
                        "model_prob": 0.70,
                        "market_prob": 0.50,
                        "edge_bps": 2000,
                        "expected_value": 0.40,
                        "implied_decimal_odds": 2.0,
                        "stake_units": 1.0,
                        "is_actionable": True,
                        "is_champion": False,
                    },
                    {
                        "game_id": "g_new",
                        "game_date": "2026-04-20",
                        "event_start_time": "2026-04-20T18:00:00",
                        "snapshot_ts": "2026-04-20T17:00:00",
                        "collection_run_ts": "2026-04-20T17:00:00",
                        "model_name": "new_model",
                        "source": "kalshi",
                        "market_id": "m_new",
                        "team": "Home2",
                        "opponent_team": "Away2",
                        "is_home_team": True,
                        "model_prob": 0.60,
                        "market_prob": 0.50,
                        "edge_bps": 1000,
                        "expected_value": 0.20,
                        "implied_decimal_odds": 2.0,
                        "stake_units": 1.0,
                        "is_actionable": True,
                        "is_champion": True,
                    },
                ]
            ),
        )
        append_dataframe(
            conn,
            "feature_importances",
            pd.DataFrame(
                [
                    {
                        "trained_at": "2026-04-24T08:17:42",
                        "model_name": "new_model",
                        "feature": "elo_rating",
                        "importance": 0.21,
                        "importance_std": 0.03,
                        "rank": 1,
                        "train_start_date": "2015-03-01",
                        "train_end_date": "2026-04-23",
                        "rows_train": 25000,
                        "rows_valid": 6000,
                        "method": "permutation",
                    }
                ]
            ),
        )
    finally:
        conn.close()

    cfg = SimpleNamespace(
        duckdb_path=db_path,
        betting_stats_start_date="2025-01-01",
        model_train_start_date="2015-03-01",
    )
    monkeypatch.setattr("mlpm.api.services.settings", lambda: cfg)
    monkeypatch.setattr("mlpm.api.services.select_champion_model", lambda: "new_model")

    contenders = get_research_contenders()
    strategies = get_research_strategies()
    roster = get_model_roster()

    assert [row["model"] for row in contenders] == ["new_model"]
    assert [row["strategy_name"] for row in strategies] == ["new_model"]
    assert any(row["model_name"] == "new_model" for row in roster)
    assert not any(row["model_name"] == "old_model" for row in roster)


def test_list_jobs_includes_persisted_jobs(monkeypatch) -> None:
    tmp_path = Path(".tmp") / f"jobs-persisted-{uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "collect-once-20260423T135836-b3f2f5.log"
    log_path.write_text("[2026-04-23T13:58:36] starting job=collect-once label=Collect snapshot\n", encoding="utf-8")
    metadata_path = log_path.with_suffix(".json")
    metadata_path.write_text(
        """
        {
          "id": "20260423T135836-b3f2f5",
          "command": "collect-once",
          "label": "Collect snapshot",
          "pid": 999999,
          "log_path": "ignored.log",
          "started_at": "2026-04-23T13:58:36",
          "finished_at": "2026-04-23T14:05:22",
          "status": "success",
          "returncode": 0
        }
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_jobs, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(api_jobs, "_JOB_REGISTRY", [])

    jobs = api_jobs.list_jobs()
    assert jobs[0].id == "20260423T135836-b3f2f5"
    assert jobs[0].status == "success"
    assert jobs[0].log_path == str(log_path)


def test_list_jobs_marks_stalled_startup_job_unknown(monkeypatch) -> None:
    tmp_path = Path(".tmp") / f"jobs-stalled-{uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "collect-once-20260424T161101-b8e6bc.log"
    log_path.write_text(
        "\n".join(
            [
                    "[2026-04-24T16:11:01.827911] starting job=collect-once label=Collect snapshot",
                    "[2026-04-24T16:11:01.828083] command=python -u -m mlpm.cli collect-once",
                    "[2026-04-24T16:11:01.828146] cwd=C:\\Users\\coles\\Desktop\\MLPM",
                ]
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path = log_path.with_suffix(".json")
    metadata_path.write_text(
        """
        {
          "id": "20260424T161101-b8e6bc",
          "command": "collect-once",
          "label": "Collect snapshot",
          "pid": 2940,
          "log_path": "ignored.log",
          "started_at": "2026-04-24T16:11:01.834285",
          "finished_at": null,
          "status": "running",
          "returncode": null
        }
        """.strip(),
        encoding="utf-8",
    )
    stale_ts = datetime(2026, 4, 24, 16, 11, 1).timestamp()
    os.utime(log_path, (stale_ts, stale_ts))

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 4, 24, 16, 12, 5)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(api_jobs, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(api_jobs, "_JOB_REGISTRY", [])
    monkeypatch.setattr(api_jobs, "_is_process_alive", lambda pid: True)
    monkeypatch.setattr(api_jobs, "datetime", _FrozenDatetime)

    jobs = api_jobs.list_jobs()
    assert jobs[0].id == "20260424T161101-b8e6bc"
    assert jobs[0].status == "unknown"
