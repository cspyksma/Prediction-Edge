from __future__ import annotations

from pathlib import Path
import uuid

import pandas as pd

from mlpm.config.settings import settings
from mlpm.evaluation.settled import (
    compute_settled_calibration,
    run_settled_prediction_report,
    run_settled_window_report,
)
from mlpm.storage.duckdb import append_dataframe, connect


def _workspace_db_path(name: str) -> Path:
    path = Path(".tmp") / f"{name}-{uuid.uuid4().hex}.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_run_settled_prediction_report_uses_latest_pregame_home_prediction(monkeypatch) -> None:
    db_path = _workspace_db_path("settled")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()

    conn = connect(settings().duckdb_path)
    append_dataframe(
        conn,
        "games",
        pd.DataFrame(
            [
                {
                    "game_id": "game-1",
                    "game_date": "2026-04-14",
                    "event_start_time": "2026-04-14T18:00:00",
                    "away_team": "Away",
                    "home_team": "Home",
                    "away_team_id": 1,
                    "home_team_id": 2,
                    "away_probable_pitcher_id": None,
                    "away_probable_pitcher_name": None,
                    "away_probable_pitcher_hand": None,
                    "home_probable_pitcher_id": None,
                    "home_probable_pitcher_name": None,
                    "home_probable_pitcher_hand": None,
                    "doubleheader": "N",
                    "game_number": 1,
                    "day_night": "night",
                    "status": "Final",
                    "snapshot_ts": "2026-04-14T12:00:00",
                    "collection_run_ts": "2026-04-14T12:00:00",
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
                    "game_id": "game-1",
                    "snapshot_ts": "2026-04-14T15:00:00",
                    "collection_run_ts": "2026-04-14T15:00:00",
                    "team": "Home",
                    "model_name": "mlb_win_svm_rbf_v1",
                    "model_prob": 0.40,
                    "games_played_floor_pass": True,
                },
                {
                    "game_id": "game-1",
                    "snapshot_ts": "2026-04-14T15:00:00",
                    "collection_run_ts": "2026-04-14T15:00:00",
                    "team": "Away",
                    "model_name": "mlb_win_svm_rbf_v1",
                    "model_prob": 0.60,
                    "games_played_floor_pass": True,
                },
                {
                    "game_id": "game-1",
                    "snapshot_ts": "2026-04-14T17:45:00",
                    "collection_run_ts": "2026-04-14T17:45:00",
                    "team": "Home",
                    "model_name": "mlb_win_svm_rbf_v1",
                    "model_prob": 0.62,
                    "games_played_floor_pass": True,
                },
                {
                    "game_id": "game-1",
                    "snapshot_ts": "2026-04-14T17:45:00",
                    "collection_run_ts": "2026-04-14T17:45:00",
                    "team": "Away",
                    "model_name": "mlb_win_svm_rbf_v1",
                    "model_prob": 0.38,
                    "games_played_floor_pass": True,
                },
                {
                    "game_id": "game-1",
                    "snapshot_ts": "2026-04-14T18:05:00",
                    "collection_run_ts": "2026-04-14T18:05:00",
                    "team": "Home",
                    "model_name": "mlb_win_svm_rbf_v1",
                    "model_prob": 0.10,
                    "games_played_floor_pass": True,
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
                    "game_id": "game-1",
                    "game_date": "2026-04-14",
                    "away_team": "Away",
                    "home_team": "Home",
                    "winner_team": "Home",
                    "away_score": 2,
                    "home_score": 5,
                }
            ]
        ),
    )
    conn.close()

    result = run_settled_prediction_report("2026-04-14", "2026-04-14")

    assert result["status"] == "ok"
    assert result["rows"] == 1
    metrics = result["models"]["mlb_win_svm_rbf_v1"]
    assert metrics["games"] == 1
    assert metrics["accuracy"] == 1.0
    assert result["recent"][0]["predicted_winner"] == "Home"


def test_run_settled_window_report_builds_window_metrics(monkeypatch) -> None:
    db_path = _workspace_db_path("settled-windows")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()

    conn = connect(settings().duckdb_path)
    append_dataframe(
        conn,
        "games",
        pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-04-10",
                    "event_start_time": "2026-04-10T18:00:00",
                    "away_team": "Away1",
                    "home_team": "Home1",
                    "away_team_id": 1,
                    "home_team_id": 2,
                    "away_probable_pitcher_id": None,
                    "away_probable_pitcher_name": None,
                    "away_probable_pitcher_hand": None,
                    "home_probable_pitcher_id": None,
                    "home_probable_pitcher_name": None,
                    "home_probable_pitcher_hand": None,
                    "doubleheader": "N",
                    "game_number": 1,
                    "day_night": "night",
                    "status": "Final",
                    "snapshot_ts": "2026-04-10T12:00:00",
                    "collection_run_ts": "2026-04-10T12:00:00",
                },
                {
                    "game_id": "g2",
                    "game_date": "2026-04-14",
                    "event_start_time": "2026-04-14T18:00:00",
                    "away_team": "Away2",
                    "home_team": "Home2",
                    "away_team_id": 3,
                    "home_team_id": 4,
                    "away_probable_pitcher_id": None,
                    "away_probable_pitcher_name": None,
                    "away_probable_pitcher_hand": None,
                    "home_probable_pitcher_id": None,
                    "home_probable_pitcher_name": None,
                    "home_probable_pitcher_hand": None,
                    "doubleheader": "N",
                    "game_number": 1,
                    "day_night": "night",
                    "status": "Final",
                    "snapshot_ts": "2026-04-14T12:00:00",
                    "collection_run_ts": "2026-04-14T12:00:00",
                },
            ]
        ),
    )
    append_dataframe(
        conn,
        "model_predictions",
        pd.DataFrame(
            [
                {"game_id": "g1", "snapshot_ts": "2026-04-10T17:00:00", "collection_run_ts": "2026-04-10T17:00:00", "team": "Home1", "model_name": "mlb_win_logreg_v2", "model_prob": 0.7, "games_played_floor_pass": True},
                {"game_id": "g2", "snapshot_ts": "2026-04-14T17:00:00", "collection_run_ts": "2026-04-14T17:00:00", "team": "Home2", "model_name": "mlb_win_logreg_v2", "model_prob": 0.4, "games_played_floor_pass": True},
            ]
        ),
    )
    append_dataframe(
        conn,
        "game_results",
        pd.DataFrame(
            [
                {"game_id": "g1", "game_date": "2026-04-10", "away_team": "Away1", "home_team": "Home1", "winner_team": "Home1", "away_score": 2, "home_score": 5},
                {"game_id": "g2", "game_date": "2026-04-14", "away_team": "Away2", "home_team": "Home2", "winner_team": "Away2", "away_score": 4, "home_score": 1},
            ]
        ),
    )
    conn.close()

    result = run_settled_window_report("2026-04-01", "2026-04-15", model_name="mlb_win_logreg_v2")

    assert result["status"] == "ok"
    windows = result["windows"]["mlb_win_logreg_v2"]
    assert windows["all"]["games"] == 2
    assert windows["all"]["accuracy"] == 1.0
    assert windows["last_7d"]["games"] == 2
    assert windows["last_30d"]["games"] == 2
    assert len(result["daily"]) == 2


def test_run_settled_prediction_report_parameterizes_model_name(monkeypatch) -> None:
    db_path = _workspace_db_path("settled-params")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()

    conn = connect(settings().duckdb_path)
    append_dataframe(
        conn,
        "games",
        pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "game_date": "2026-04-14",
                    "event_start_time": "2026-04-14T18:00:00",
                    "away_team": "Away",
                    "home_team": "Home",
                    "snapshot_ts": "2026-04-14T12:00:00",
                    "collection_run_ts": "2026-04-14T12:00:00",
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
                    "snapshot_ts": "2026-04-14T17:00:00",
                    "collection_run_ts": "2026-04-14T17:00:00",
                    "team": "Home",
                    "model_name": "safe_model",
                    "model_prob": 0.7,
                    "games_played_floor_pass": True,
                }
            ]
        ),
    )
    append_dataframe(
        conn,
        "game_results",
        pd.DataFrame(
            [
                {"game_id": "g1", "game_date": "2026-04-14", "away_team": "Away", "home_team": "Home", "winner_team": "Home", "away_score": 1, "home_score": 2}
            ]
        ),
    )
    conn.close()

    result = run_settled_prediction_report("2026-04-14", "2026-04-14", model_name="safe_model' OR 1=1 --")

    assert result == {"status": "insufficient_data", "rows": 0}


def test_compute_settled_calibration_buckets_and_computes_abs_error(monkeypatch) -> None:
    db_path = _workspace_db_path("calibration")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()

    conn = connect(settings().duckdb_path)
    # Seed 4 games/home teams with two models spanning the probability range.
    # Use games + predictions + results so settled_predictions_deduped picks them up.
    append_dataframe(
        conn,
        "games",
        pd.DataFrame(
            [
                {
                    "game_id": f"g{i}",
                    "game_date": f"2026-04-{10 + i:02d}",
                    "event_start_time": f"2026-04-{10 + i:02d}T18:00:00",
                    "away_team": f"A{i}",
                    "home_team": f"H{i}",
                    "snapshot_ts": f"2026-04-{10 + i:02d}T12:00:00",
                    "collection_run_ts": f"2026-04-{10 + i:02d}T12:00:00",
                }
                for i in range(4)
            ]
        ),
    )
    preds = []
    probs = [0.15, 0.45, 0.75, 0.95]
    winners = ["A0", "A1", "H2", "H3"]  # only game 2 and 3 are home wins
    for i, p in enumerate(probs):
        preds.append(
            {
                "game_id": f"g{i}",
                "snapshot_ts": f"2026-04-{10 + i:02d}T17:00:00",
                "collection_run_ts": f"2026-04-{10 + i:02d}T17:00:00",
                "team": f"H{i}",
                "model_name": "modelX",
                "model_prob": p,
                "games_played_floor_pass": True,
            }
        )
    append_dataframe(conn, "model_predictions", pd.DataFrame(preds))
    append_dataframe(
        conn,
        "game_results",
        pd.DataFrame(
            [
                {
                    "game_id": f"g{i}",
                    "game_date": f"2026-04-{10 + i:02d}",
                    "away_team": f"A{i}",
                    "home_team": f"H{i}",
                    "winner_team": winners[i],
                    "away_score": 1 if winners[i].startswith("H") else 3,
                    "home_score": 3 if winners[i].startswith("H") else 1,
                }
                for i in range(4)
            ]
        ),
    )
    conn.close()

    frame = compute_settled_calibration(model_name="modelX", bins=4)

    assert not frame.empty
    assert set(frame.columns) >= {
        "model_name",
        "bucket",
        "bucket_lower",
        "bucket_upper",
        "games",
        "avg_predicted_prob",
        "actual_home_win_rate",
        "abs_error",
    }
    # Each of our 4 games should land in its own bucket (bins=4 splits [0,1] evenly).
    assert frame["games"].sum() == 4
    # Abs error should equal |avg_predicted_prob - actual_home_win_rate| row by row.
    for row in frame.itertuples():
        assert abs(row.abs_error - abs(row.avg_predicted_prob - row.actual_home_win_rate)) < 1e-9


def test_compute_settled_calibration_returns_empty_on_no_data(monkeypatch) -> None:
    db_path = _workspace_db_path("calibration-empty")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()
    # Create schema but no data.
    connect(settings().duckdb_path).close()

    frame = compute_settled_calibration(bins=5)
    assert frame.empty
    assert "bucket" in frame.columns
