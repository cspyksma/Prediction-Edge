from __future__ import annotations

from pathlib import Path
import uuid

import pandas as pd

from mlpm.config.settings import settings
from mlpm.historical.common import has_successful_import_run, iter_date_windows
from mlpm.historical.normalize_kalshi_history import normalize_kalshi_candle_payload
from mlpm.historical.normalize_polymarket_history import polymarket_history_to_quote_rows
from mlpm.historical.polymarket_backfill import fetch_polymarket_batch_price_history
from mlpm.historical.status import run_historical_import_status
from mlpm.storage.duckdb import append_dataframe, connect


def _workspace_db_path(name: str) -> Path:
    path = Path(".tmp") / f"{name}-{uuid.uuid4().hex}.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_polymarket_history_to_quote_rows_normalizes_home_side() -> None:
    rows = polymarket_history_to_quote_rows(
        [{"t": 1713200400, "p": 0.62}],
        {
            "import_run_id": "run-1",
            "market_id": "market-1",
            "asset_id": "asset-1",
            "game_id": "game-1",
            "outcome_team": "Chicago Cubs",
            "home_team": "Chicago Cubs",
            "away_team": "St. Louis Cardinals",
            "event_start_time": "2024-04-15T18:00:00Z",
            "raw_payload_path": "data/raw/historical/polymarket/2024-04-15/asset-1.json",
        },
    )

    assert len(rows) == 1
    assert rows[0]["side"] == "home"
    assert rows[0]["home_implied_prob"] == 0.62
    assert rows[0]["pre_pitch_flag"] is True


def test_normalize_kalshi_candle_payload_normalizes_away_side() -> None:
    rows = normalize_kalshi_candle_payload(
        {"candlesticks": [{"end_ts": 1713206400, "close": 73, "volume": 12}]},
        "KXMLBGAME-CUBSCARDS",
        game_id="game-1",
    )

    assert len(rows) == 1
    assert rows[0]["quote_type"] == "candle_close"
    assert rows[0]["raw_prob_yes"] == 0.73


def test_polymarket_batch_price_history_rejects_batches_over_20() -> None:
    try:
        fetch_polymarket_batch_price_history([str(index) for index in range(21)], 1, 2)
    except ValueError as exc:
        assert "at most 20" in str(exc)
    else:
        raise AssertionError("Expected batch limit validation to raise.")


def test_historical_import_status_reports_aggregates(monkeypatch) -> None:
    db_path = _workspace_db_path("historical-status")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()
    conn = connect(settings().duckdb_path)
    append_dataframe(
        conn,
        "historical_import_runs",
        pd.DataFrame(
            [
                {
                    "import_run_id": "run-1",
                    "source": "polymarket",
                    "started_at": "2026-04-15T12:00:00Z",
                    "completed_at": "2026-04-15T12:01:00Z",
                    "start_date": "2024-03-01",
                    "end_date": "2024-03-31",
                    "status": "ok",
                    "request_count": 3,
                    "payload_count": 3,
                    "normalized_rows": 120,
                    "error_message": None,
                },
                {
                    "import_run_id": "run-2",
                    "source": "kalshi",
                    "started_at": "2026-04-15T12:10:00Z",
                    "completed_at": "2026-04-15T12:11:00Z",
                    "start_date": "2024-03-01",
                    "end_date": "2024-03-31",
                    "status": "ok",
                    "request_count": 5,
                    "payload_count": 5,
                    "normalized_rows": 80,
                    "error_message": None,
                },
            ]
        ),
    )
    conn.close()

    result = run_historical_import_status("2024-03-01", "2024-03-31")

    assert result["status"] == "ok"
    assert result["rows"] == 2
    assert result["sources"]["polymarket"]["normalized_rows"] == 120
    assert result["sources"]["kalshi"]["request_count"] == 5


def test_iter_date_windows_splits_ranges() -> None:
    windows = iter_date_windows("2024-03-01", "2024-03-10", chunk_days=4)

    assert windows == [
        ("2024-03-01", "2024-03-04"),
        ("2024-03-05", "2024-03-08"),
        ("2024-03-09", "2024-03-10"),
    ]


def test_has_successful_import_run_detects_completed_chunk(monkeypatch) -> None:
    db_path = _workspace_db_path("historical-resume")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()
    conn = connect(settings().duckdb_path)
    append_dataframe(
        conn,
        "historical_import_runs",
        pd.DataFrame(
            [
                {
                    "import_run_id": "run-1",
                    "source": "kalshi",
                    "started_at": "2026-04-15T12:00:00Z",
                    "completed_at": "2026-04-15T12:01:00Z",
                    "start_date": "2024-03-01",
                    "end_date": "2024-03-07",
                    "status": "ok",
                    "request_count": 3,
                    "payload_count": 3,
                    "normalized_rows": 120,
                    "error_message": None,
                }
            ]
        ),
    )
    conn.close()

    assert has_successful_import_run("kalshi", "2024-03-01", "2024-03-07") is True
    assert has_successful_import_run("kalshi", "2024-03-08", "2024-03-14") is False
