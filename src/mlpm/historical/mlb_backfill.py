"""Backfill MLB Stats API game results, pitching logs, and batting logs into DuckDB.

Runs on the user's local machine (MLB Stats API is not reachable from the Claude
sandbox egress proxy). Chunked by month so a partial run can be resumed by
re-invoking with the same date range — the per-month idempotency check skips any
month whose game_results already cover the full month.

Typical invocation:
    mlpm backfill-mlb --start-date 2015-01-01 --end-date 2026-04-18

Runtime: ~2-4 hours for the full 2015-2026 span (~26k boxscore fetches).
"""

from __future__ import annotations

import logging
import time
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from mlpm.config.settings import settings
from mlpm.ingest.mlb_stats import (
    fetch_final_results,
    fetch_game_batting_logs,
    fetch_game_pitching_logs,
)
from mlpm.storage.duckdb import append_dataframe, connect, query_dataframe, replace_dataframe

logger = logging.getLogger(__name__)

# Baseball season runs roughly March through November. We still fetch the full
# year so spring training, late-season, and postseason games are included for
# teams that played them, but months that don't have regular-season games will
# simply return zero rows and get skipped.
MLB_SEASON_MONTHS = tuple(range(1, 13))
CHUNK_SLEEP_SECONDS = 1.5  # Courtesy pause between month boundaries.


@dataclass
class MonthProgress:
    year: int
    month: int
    games: int
    pitching_rows: int
    batting_rows: int
    skipped: bool
    elapsed_seconds: float


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    first = date(year, month, 1)
    last = date(year, month, monthrange(year, month)[1])
    return first.isoformat(), last.isoformat()


def _months_in_range(start: date, end: date) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        pairs.append((cursor.year, cursor.month))
        # Advance to first of next month.
        year = cursor.year + (1 if cursor.month == 12 else 0)
        month = 1 if cursor.month == 12 else cursor.month + 1
        cursor = date(year, month, 1)
    return pairs


def _existing_game_ids(conn, table: str, start: str, end: str) -> set[str]:
    """Return game_ids already present in `table` for the date window."""
    if table == "game_results":
        sql = "SELECT DISTINCT game_id FROM game_results WHERE game_date BETWEEN ? AND ?"
    else:
        sql = f"SELECT DISTINCT game_id FROM {table} WHERE game_date BETWEEN ? AND ?"
    df = query_dataframe(conn, sql, (start, end))
    return set(df["game_id"].astype(str).tolist()) if not df.empty else set()


def _backfill_month(
    conn,
    year: int,
    month: int,
    *,
    force: bool = False,
) -> MonthProgress:
    start, end = _month_bounds(year, month)
    tic = time.perf_counter()
    logger.info("mlb_backfill month=%04d-%02d start=%s end=%s", year, month, start, end)

    existing_results = _existing_game_ids(conn, "game_results", start, end)
    existing_pitching = _existing_game_ids(conn, "mlb_pitching_logs", start, end)
    existing_batting = _existing_game_ids(conn, "mlb_batting_logs", start, end)

    # Fetch game results first. Cheap, single API call per month.
    results_df = fetch_final_results(start, end)
    if results_df.empty:
        logger.info("  no final games in %04d-%02d; skipping", year, month)
        return MonthProgress(year, month, 0, 0, 0, True, time.perf_counter() - tic)

    results_df["game_date"] = pd.to_datetime(results_df["game_date"]).dt.date
    results_df["game_id"] = results_df["game_id"].astype(str)

    if not force:
        new_results_mask = ~results_df["game_id"].isin(existing_results)
        new_results = results_df[new_results_mask]
    else:
        new_results = results_df

    if not new_results.empty:
        replace_dataframe(conn, "game_results", new_results, key_columns=["game_id"])
        logger.info("  wrote %s game_results rows", len(new_results))

    # Pitching logs — expensive (one boxscore call per game). Skip games we
    # already have unless forced.
    if not force:
        missing_pitching_ids = set(results_df["game_id"]) - existing_pitching
    else:
        missing_pitching_ids = set(results_df["game_id"])

    pitching_rows = 0
    batting_rows = 0
    if missing_pitching_ids:
        pitching_df = fetch_game_pitching_logs(start, end)
        if not pitching_df.empty:
            pitching_df["game_id"] = pitching_df["game_id"].astype(str)
            pitching_df["game_date"] = pd.to_datetime(pitching_df["game_date"]).dt.date
            pitching_df = pitching_df[pitching_df["game_id"].isin(missing_pitching_ids)].copy()
            pitching_df["imported_at"] = datetime.utcnow()
            if not pitching_df.empty:
                replace_dataframe(conn, "mlb_pitching_logs", pitching_df, key_columns=["game_id", "team"])
                pitching_rows = len(pitching_df)
                logger.info("  wrote %s pitching rows", pitching_rows)

    if not force:
        missing_batting_ids = set(results_df["game_id"]) - existing_batting
    else:
        missing_batting_ids = set(results_df["game_id"])

    if missing_batting_ids:
        batting_df = fetch_game_batting_logs(start, end)
        if not batting_df.empty:
            batting_df["game_id"] = batting_df["game_id"].astype(str)
            batting_df["game_date"] = pd.to_datetime(batting_df["game_date"]).dt.date
            batting_df = batting_df[batting_df["game_id"].isin(missing_batting_ids)].copy()
            batting_df["imported_at"] = datetime.utcnow()
            if not batting_df.empty:
                replace_dataframe(conn, "mlb_batting_logs", batting_df, key_columns=["game_id", "team"])
                batting_rows = len(batting_df)
                logger.info("  wrote %s batting rows", batting_rows)

    elapsed = time.perf_counter() - tic
    logger.info(
        "  month=%04d-%02d done games=%s pitching=%s batting=%s elapsed=%.1fs",
        year, month, len(new_results), pitching_rows, batting_rows, elapsed,
    )
    return MonthProgress(year, month, len(new_results), pitching_rows, batting_rows, False, elapsed)


def run_mlb_backfill(
    start_date: str,
    end_date: str,
    *,
    force: bool = False,
    db_path: Path | None = None,
) -> dict[str, object]:
    """Backfill MLB fundamentals for a date range. Resumable and idempotent.

    Args:
        start_date: ISO date (e.g., '2015-01-01').
        end_date: ISO date.
        force: If True, re-fetch even months already in the DB.
        db_path: Override DuckDB path; defaults to settings().duckdb_path.

    Returns:
        Summary dict with per-month progress and totals.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    db_path = db_path or settings().duckdb_path
    logger.info("mlb_backfill starting start=%s end=%s db=%s force=%s", start, end, db_path, force)
    conn = connect(db_path)
    try:
        months = _months_in_range(start, end)
        progress: list[MonthProgress] = []
        for year, month in months:
            p = _backfill_month(conn, year, month, force=force)
            progress.append(p)
            time.sleep(CHUNK_SLEEP_SECONDS)
    finally:
        conn.close()

    total_games = sum(p.games for p in progress)
    total_pitching = sum(p.pitching_rows for p in progress)
    total_batting = sum(p.batting_rows for p in progress)
    total_seconds = sum(p.elapsed_seconds for p in progress)

    logger.info(
        "mlb_backfill complete months=%s games=%s pitching=%s batting=%s total_elapsed=%.1fs",
        len(progress), total_games, total_pitching, total_batting, total_seconds,
    )

    return {
        "status": "ok",
        "months": len(progress),
        "games_added": total_games,
        "pitching_rows_added": total_pitching,
        "batting_rows_added": total_batting,
        "elapsed_seconds": total_seconds,
        "monthly": [
            {
                "year": p.year,
                "month": p.month,
                "games": p.games,
                "pitching_rows": p.pitching_rows,
                "batting_rows": p.batting_rows,
                "skipped": p.skipped,
                "elapsed_seconds": round(p.elapsed_seconds, 1),
            }
            for p in progress
        ],
    }
