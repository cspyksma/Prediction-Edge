from __future__ import annotations

from pathlib import Path
import uuid

import pandas as pd
import pytest

from mlpm.config.settings import settings
from mlpm.historical.common import has_successful_import_run, iter_date_windows
from mlpm.historical.kalshi_backfill import (
    _extract_market_payload,
    _enrich_candidate_with_market,
    _map_kalshi_market_to_games,
    _market_query_window,
    _should_use_historical_market,
    _write_historical_kalshi_rows,
)
from mlpm.historical.normalize_kalshi_history import normalize_kalshi_candle_payload
from mlpm.historical.normalize_polymarket_history import polymarket_history_to_quote_rows
from mlpm.historical.polymarket_backfill import fetch_polymarket_batch_price_history
from mlpm.historical.replay import build_kalshi_replay_quote_rows, load_kalshi_pregame_replay
from mlpm.historical.status import run_historical_import_status
from mlpm.ingest.mlb_stats import _scheduled_first_pitch_utc
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
        {
            "candlesticks": [
                {
                    "end_period_ts": 1713206400,
                    "price": {"close_dollars": "0.73"},
                    "volume": 12,
                    "open_interest": 55,
                }
            ]
        },
        {
            "ticker": "KXMLBGAME-CUBSCARDS",
            "market_id": "KXMLBGAME-CUBSCARDS",
            "game_id": "game-1",
            "outcome_team": "St. Louis Cardinals",
            "home_team": "Chicago Cubs",
            "away_team": "St. Louis Cardinals",
            "event_start_time": "2024-04-15T19:00:00Z",
        },
    )

    assert len(rows) == 1
    assert rows[0]["quote_type"] == "candle_close"
    assert rows[0]["raw_prob_yes"] == 0.73
    assert rows[0]["side"] == "away"
    assert rows[0]["home_implied_prob"] == 0.27


def test_extract_market_payload_handles_single_market_and_list_shapes() -> None:
    market = {"ticker": "T1"}

    assert _extract_market_payload({"market": market}) == market
    assert _extract_market_payload({"markets": [market]}) == market
    assert _extract_market_payload({"markets": []}) == {}


def test_should_use_historical_market_uses_cutoff_boundary() -> None:
    assert (
        _should_use_historical_market(
            {"event_start_time": "2024-04-15T19:00:00Z"},
            {"market_settled_ts": "2024-06-01T00:00:00Z"},
        )
        is True
    )
    assert (
        _should_use_historical_market(
            {"event_start_time": "2026-04-15T19:00:00Z"},
            {"market_settled_ts": "2026-03-01T00:00:00Z"},
        )
        is False
    )


def test_map_kalshi_market_to_games_uses_aliases_and_time_window() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_date": "2026-04-01",
                "event_start_time": "2026-04-01T17:10:00Z",
                "event_start_dt": pd.Timestamp("2026-04-01T17:10:00Z"),
                "away_team": "Chicago White Sox",
                "home_team": "Miami Marlins",
            }
        ]
    )

    rows = _map_kalshi_market_to_games(
        {
            "ticker": "KXMLBGAME-26APR011310CWSMIA-CWS",
            "event_ticker": "KXMLBGAME-26APR011310CWSMIA",
            "title": "Chicago WS vs Miami Winner?",
            "yes_sub_title": "Chicago WS",
            "close_time": "2026-04-01T19:30:58Z",
        },
        games,
    )

    assert len(rows) == 1
    assert rows[0]["game_id"] == "g1"
    assert rows[0]["outcome_team"] == "Chicago White Sox"


def test_enrich_candidate_keeps_mlb_event_start_time() -> None:
    row = {
        "event_id": "evt-1",
        "market_id": "old-market",
        "ticker": "old-ticker",
        "event_start_time": "2026-04-01T17:10:00Z",
    }
    market = {
        "event_ticker": "evt-2",
        "ticker": "new-ticker",
        "close_time": "2026-04-01T20:30:00Z",
    }

    enriched = _enrich_candidate_with_market(row, market)

    assert enriched["event_id"] == "evt-2"
    assert enriched["ticker"] == "new-ticker"
    assert enriched["event_start_time"] == "2026-04-01T17:10:00Z"


def test_market_query_window_stops_at_first_pitch_not_close() -> None:
    start_ts, end_ts = _market_query_window(
        {"ticker": "T1", "event_start_time": "2026-04-01T17:10:00Z"},
        {"open_time": "2026-03-31T17:10:00Z", "close_time": "2026-04-01T20:30:00Z"},
    )

    assert start_ts == int(pd.Timestamp("2026-03-31T17:10:00Z").timestamp())
    assert end_ts == int(pd.Timestamp("2026-04-01T17:10:00Z").timestamp())


def test_scheduled_first_pitch_utc_normalizes_game_date() -> None:
    assert _scheduled_first_pitch_utc({"gameDate": "2026-04-06T22:45:00Z"}) == "2026-04-06T22:45:00Z"
    assert _scheduled_first_pitch_utc({"gameDate": "2026-04-06T17:45:00-05:00"}) == "2026-04-06T22:45:00Z"


def test_write_historical_kalshi_rows_replaces_all_rows_for_game(monkeypatch) -> None:
    db_path = _workspace_db_path("historical-kalshi-replace")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()
    conn = connect(settings().duckdb_path)
    append_dataframe(
        conn,
        "historical_kalshi_quotes",
        pd.DataFrame(
            [
                {
                    "import_run_id": "run-old",
                    "source": "kalshi",
                    "collection_mode": "historical_import",
                    "market_id": "m1",
                    "event_id": "e1",
                    "ticker": "t1",
                    "game_id": "g1",
                    "event_start_time": "2026-04-07T02:10:12Z",
                    "quote_ts": "2026-04-06T21:04:00Z",
                    "outcome_team": "Home",
                    "side": "home",
                    "home_implied_prob": 0.01,
                    "raw_prob_yes": 0.01,
                    "quote_type": "candle_close",
                    "volume": 1.0,
                    "open_interest": 1.0,
                    "best_price_source": "historical_candlesticks",
                    "pre_pitch_flag": True,
                    "raw_payload_path": "old.json",
                    "imported_at": "2026-04-16T12:00:00Z",
                }
            ]
        ),
    )
    conn.close()

    _write_historical_kalshi_rows(
        [
            {
                "import_run_id": "run-new",
                "source": "kalshi",
                "collection_mode": "historical_import",
                "market_id": "m1",
                "event_id": "e1",
                "ticker": "t1",
                "game_id": "g1",
                "event_start_time": "2026-04-06T23:07:00Z",
                "quote_ts": "2026-04-06T18:00:00Z",
                "outcome_team": "Home",
                "side": "home",
                "home_implied_prob": 0.52,
                "raw_prob_yes": 0.52,
                "quote_type": "candle_close",
                "volume": 1.0,
                "open_interest": 1.0,
                "best_price_source": "historical_candlesticks",
                "pre_pitch_flag": True,
                "raw_payload_path": "new.json",
                "imported_at": "2026-04-17T12:00:00Z",
            }
        ],
        ["g1"],
    )

    conn = connect(settings().duckdb_path)
    try:
        rows = conn.execute(
            """
            SELECT event_start_time, quote_ts, home_implied_prob
            FROM historical_kalshi_quotes
            WHERE game_id = 'g1'
            """
        ).fetchall()
    finally:
        conn.close()

    assert rows == [
        (
            pd.Timestamp("2026-04-06T23:07:00"),
            pd.Timestamp("2026-04-06T18:00:00"),
            0.52,
        )
    ]


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
                    "games_total": 31,
                    "games_with_markets": 21,
                    "games_with_pregame_quotes": 18,
                    "candidate_markets": 44,
                    "empty_payload_count": 2,
                    "rate_limited_count": 0,
                    "parse_error_count": 0,
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
                    "games_total": 31,
                    "games_with_markets": 17,
                    "games_with_pregame_quotes": 12,
                    "candidate_markets": 36,
                    "empty_payload_count": 3,
                    "rate_limited_count": 1,
                    "parse_error_count": 1,
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
    assert result["sources"]["kalshi"]["games_with_pregame_quotes"] == 12
    assert result["sources"]["kalshi"]["rate_limited_count"] == 1


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
                    "games_total": 5,
                    "games_with_markets": 4,
                    "games_with_pregame_quotes": 0,
                    "candidate_markets": 8,
                    "empty_payload_count": 2,
                    "rate_limited_count": 0,
                    "parse_error_count": 0,
                    "error_message": None,
                }
            ]
        ),
    )
    conn.close()

    assert has_successful_import_run("kalshi", "2024-03-01", "2024-03-07") is False
    assert has_successful_import_run("kalshi", "2024-03-08", "2024-03-14") is False


def test_load_kalshi_pregame_replay_selects_last_pregame_quotes(monkeypatch) -> None:
    db_path = _workspace_db_path("historical-replay")
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
                    "game_date": "2026-04-01",
                    "event_start_time": "2026-04-01T18:00:00Z",
                    "away_team": "Away",
                    "home_team": "Home",
                    "snapshot_ts": "2026-04-01T12:00:00Z",
                    "collection_run_ts": "2026-04-01T12:00:00Z",
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
                    "import_run_id": "run-1",
                    "source": "kalshi",
                    "collection_mode": "historical_import",
                    "market_id": "m-home",
                    "event_id": "e1",
                    "ticker": "t-home",
                    "game_id": "g1",
                    "event_start_time": "2026-04-01T18:00:00Z",
                    "quote_ts": "2026-04-01T16:00:00Z",
                    "outcome_team": "Home",
                    "side": "home",
                    "home_implied_prob": 0.58,
                    "raw_prob_yes": 0.58,
                    "quote_type": "candle_close",
                    "volume": 1.0,
                    "open_interest": 1.0,
                    "best_price_source": "historical_candlesticks",
                    "pre_pitch_flag": True,
                    "raw_payload_path": "x",
                    "imported_at": "2026-04-16T12:00:00Z",
                },
                {
                    "import_run_id": "run-1",
                    "source": "kalshi",
                    "collection_mode": "historical_import",
                    "market_id": "m-home",
                    "event_id": "e1",
                    "ticker": "t-home",
                    "game_id": "g1",
                    "event_start_time": "2026-04-01T18:00:00Z",
                    "quote_ts": "2026-04-01T17:30:00Z",
                    "outcome_team": "Home",
                    "side": "home",
                    "home_implied_prob": 0.62,
                    "raw_prob_yes": 0.62,
                    "quote_type": "candle_close",
                    "volume": 1.0,
                    "open_interest": 1.0,
                    "best_price_source": "historical_candlesticks",
                    "pre_pitch_flag": True,
                    "raw_payload_path": "x",
                    "imported_at": "2026-04-16T12:00:01Z",
                },
                {
                    "import_run_id": "run-1",
                    "source": "kalshi",
                    "collection_mode": "historical_import",
                    "market_id": "m-away",
                    "event_id": "e1",
                    "ticker": "t-away",
                    "game_id": "g1",
                    "event_start_time": "2026-04-01T18:00:00Z",
                    "quote_ts": "2026-04-01T17:20:00Z",
                    "outcome_team": "Away",
                    "side": "away",
                    "home_implied_prob": 0.39,
                    "raw_prob_yes": 0.61,
                    "quote_type": "candle_close",
                    "volume": 1.0,
                    "open_interest": 1.0,
                    "best_price_source": "historical_candlesticks",
                    "pre_pitch_flag": True,
                    "raw_payload_path": "x",
                    "imported_at": "2026-04-16T12:00:01Z",
                },
                {
                    "import_run_id": "run-1",
                    "source": "kalshi",
                    "collection_mode": "historical_import",
                    "market_id": "m-away",
                    "event_id": "e1",
                    "ticker": "t-away",
                    "game_id": "g1",
                    "event_start_time": "2026-04-01T18:00:00Z",
                    "quote_ts": "2026-04-01T18:05:00Z",
                    "outcome_team": "Away",
                    "side": "away",
                    "home_implied_prob": 0.35,
                    "raw_prob_yes": 0.65,
                    "quote_type": "candle_close",
                    "volume": 1.0,
                    "open_interest": 1.0,
                    "best_price_source": "historical_candlesticks",
                    "pre_pitch_flag": False,
                    "raw_payload_path": "x",
                    "imported_at": "2026-04-16T12:00:02Z",
                },
            ]
        ),
    )
    conn.close()

    replay = load_kalshi_pregame_replay("2026-04-01", "2026-04-01")
    quote_rows = build_kalshi_replay_quote_rows(replay)

    assert len(replay) == 1
    assert replay.iloc[0]["home_market_prob"] == 0.62
    assert replay.iloc[0]["away_market_prob"] == pytest.approx(0.38)
    assert len(quote_rows) == 2


def test_load_kalshi_pregame_replay_derives_both_sides_from_single_ticker(monkeypatch) -> None:
    db_path = _workspace_db_path("historical-replay-single-sided")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()
    conn = connect(settings().duckdb_path)
    append_dataframe(
        conn,
        "games",
        pd.DataFrame(
            [
                {
                    "game_id": "g2",
                    "game_date": "2026-04-02",
                    "event_start_time": "2026-04-02T18:00:00Z",
                    "away_team": "Away",
                    "home_team": "Home",
                    "snapshot_ts": "2026-04-02T12:00:00Z",
                    "collection_run_ts": "2026-04-02T12:00:00Z",
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
                    "import_run_id": "run-1",
                    "source": "kalshi",
                    "collection_mode": "historical_import",
                    "market_id": "m-away",
                    "event_id": "e2",
                    "ticker": "t-away",
                    "game_id": "g2",
                    "event_start_time": "2026-04-02T18:00:00Z",
                    "quote_ts": "2026-04-02T17:30:00Z",
                    "outcome_team": "Away",
                    "side": "away",
                    "home_implied_prob": 0.41,
                    "raw_prob_yes": 0.59,
                    "quote_type": "candle_close",
                    "volume": 1.0,
                    "open_interest": 1.0,
                    "best_price_source": "historical_candlesticks",
                    "pre_pitch_flag": True,
                    "raw_payload_path": "x",
                    "imported_at": "2026-04-16T12:00:01Z",
                }
            ]
        ),
    )
    conn.close()

    replay = load_kalshi_pregame_replay("2026-04-02", "2026-04-02")
    quote_rows = build_kalshi_replay_quote_rows(replay)

    assert len(replay) == 1
    assert replay.iloc[0]["home_market_prob"] == 0.41
    assert replay.iloc[0]["away_market_prob"] == pytest.approx(0.59)
    assert replay.iloc[0]["source_rows"] == 1
    assert len(quote_rows) == 2
