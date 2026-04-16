from __future__ import annotations

from datetime import date
from pathlib import Path
import uuid

import pandas as pd

from mlpm.config.settings import settings
from mlpm.pipeline.runner import run_service, sync_recent_game_results
from mlpm.storage.duckdb import append_dataframe, connect, query_dataframe


def _workspace_db_path(name: str) -> Path:
    path = Path(".tmp") / f"{name}-{uuid.uuid4().hex}.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_sync_recent_game_results_replaces_existing_rows(monkeypatch) -> None:
    db_path = _workspace_db_path("runner")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()

    conn = connect(settings().duckdb_path)
    append_dataframe(
        conn,
        "game_results",
        pd.DataFrame(
            [
                {
                    "game_id": "123",
                    "game_date": "2026-04-13",
                    "winner_team": "Old Winner",
                    "away_score": 1,
                    "home_score": 0,
                }
            ]
        ),
    )
    conn.close()

    monkeypatch.setattr(
        "mlpm.pipeline.runner.fetch_final_results",
        lambda start_date, end_date: pd.DataFrame(
            [
                {
                    "game_id": "123",
                    "game_date": "2026-04-13",
                    "winner_team": "New Winner",
                    "away_score": 2,
                    "home_score": 5,
                },
                {
                    "game_id": "456",
                    "game_date": "2026-04-14",
                    "winner_team": "Another Winner",
                    "away_score": 3,
                    "home_score": 1,
                },
            ]
        ),
    )

    result = sync_recent_game_results(lookback_days=3, reference_date=date(2026, 4, 15))

    conn = connect(settings().duckdb_path)
    stored = query_dataframe(conn, "SELECT * FROM game_results_deduped ORDER BY game_id")
    conn.close()

    assert result == {"game_results_synced": 2}
    assert stored["winner_team"].tolist() == ["New Winner", "Another Winner"]


def test_run_service_records_successful_run(monkeypatch) -> None:
    db_path = _workspace_db_path("service")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("SNAPSHOT_INTERVAL_SECONDS", "0")
    settings.cache_clear()

    monkeypatch.setattr(
        "mlpm.pipeline.runner.collect_snapshot",
        lambda: {
            "games": 5,
            "kalshi_quotes": 10,
            "polymarket_quotes": 8,
            "model_predictions": 2,
            "normalized_quotes": 18,
            "discrepancies": 1,
        },
    )
    monkeypatch.setattr(
        "mlpm.pipeline.runner.sync_recent_game_results",
        lambda lookback_days=None, reference_date=None: {"game_results_synced": 4},
    )
    monkeypatch.setattr("mlpm.pipeline.runner.time.sleep", lambda seconds: None)

    run_service(iterations=1)

    conn = connect(settings().duckdb_path)
    runs = query_dataframe(conn, "SELECT status, games, discrepancies, game_results_synced FROM collector_runs")
    conn.close()

    assert len(runs) == 1
    assert runs.iloc[0].to_dict() == {
        "status": "ok",
        "games": 5,
        "discrepancies": 1,
        "game_results_synced": 4,
    }
