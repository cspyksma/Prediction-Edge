from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from mlpm.config.settings import settings
from mlpm.ingest.mlb_stats import fetch_final_results
from mlpm.pipeline.collect import collect_snapshot
from mlpm.storage.duckdb import append_dataframe, connect, replace_dataframe

logger = logging.getLogger(__name__)


def sync_recent_game_results(
    lookback_days: int | None = None,
    reference_date: date | None = None,
) -> dict[str, int]:
    cfg = settings()
    target_date = reference_date or date.today()
    history_days = lookback_days if lookback_days is not None else cfg.results_sync_lookback_days
    start_date = target_date - timedelta(days=history_days)
    results_df = fetch_final_results(start_date.isoformat(), target_date.isoformat())
    if results_df.empty:
        return {"game_results_synced": 0}

    conn = connect(cfg.duckdb_path)
    try:
        replace_dataframe(conn, "game_results", results_df, ["game_id"])
    finally:
        conn.close()
    return {"game_results_synced": len(results_df)}


def record_collector_run(
    *,
    run_id: str,
    started_at: datetime,
    completed_at: datetime | None,
    status: str,
    error_message: str | None = None,
    counts: dict[str, int] | None = None,
) -> None:
    cfg = settings()
    payload = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "error_message": error_message,
        "games": None,
        "kalshi_quotes": None,
        "polymarket_quotes": None,
        "model_predictions": None,
        "normalized_quotes": None,
        "discrepancies": None,
        "bet_opportunities": None,
        "game_results_synced": None,
    }
    if counts:
        payload.update(counts)

    conn = connect(cfg.duckdb_path)
    try:
        append_dataframe(conn, "collector_runs", pd.DataFrame([payload]))
    finally:
        conn.close()


def run_service(iterations: int = 0) -> None:
    cfg = settings()
    count = 0
    while iterations == 0 or count < iterations:
        started_at = datetime.now(tz=UTC)
        run_id = str(uuid.uuid4())
        try:
            snapshot_counts = collect_snapshot()
            result_counts = sync_recent_game_results()
            completed_at = datetime.now(tz=UTC)
            counts = {**snapshot_counts, **result_counts}
            record_collector_run(
                run_id=run_id,
                started_at=started_at,
                completed_at=completed_at,
                status="ok",
                counts=counts,
            )
            logger.info("Collector run completed run_id=%s status=ok counts=%s", run_id, counts)
            count += 1
            sleep_seconds = max(
                0.0,
                cfg.snapshot_interval_seconds - (completed_at - started_at).total_seconds(),
            )
            if iterations == 0 or count < iterations:
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            completed_at = datetime.now(tz=UTC)
            record_collector_run(
                run_id=run_id,
                started_at=started_at,
                completed_at=completed_at,
                status="stopped",
                error_message="Interrupted by operator",
            )
            logger.info(
                "Collector run stopped run_id=%s status=stopped reason=%s",
                run_id,
                "Interrupted by operator",
            )
            return
        except Exception as exc:
            completed_at = datetime.now(tz=UTC)
            record_collector_run(
                run_id=run_id,
                started_at=started_at,
                completed_at=completed_at,
                status="error",
                error_message=str(exc),
            )
            logger.exception("Collector run failed run_id=%s status=error", run_id)
            time.sleep(cfg.runner_failure_backoff_seconds)
