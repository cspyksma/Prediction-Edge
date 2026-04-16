from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from mlpm.config.settings import settings
from mlpm.storage.duckdb import append_dataframe, connect, query_dataframe


def new_import_run_id() -> str:
    return str(uuid.uuid4())


def write_historical_payload(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def build_historical_payload_path(source: str, category: str | None, day: str, name: str) -> Path:
    root = settings().raw_data_dir / "historical" / source
    if category:
        root = root / category
    root = root / day
    return root / f"{name}.json"


def append_historical_import_run(
    *,
    import_run_id: str,
    source: str,
    started_at: datetime,
    completed_at: datetime,
    start_date: str,
    end_date: str,
    status: str,
    request_count: int,
    payload_count: int,
    normalized_rows: int,
    error_message: str | None = None,
) -> None:
    conn = connect(settings().duckdb_path)
    try:
        append_dataframe(
            conn,
            "historical_import_runs",
            pd.DataFrame(
                [
                    {
                        "import_run_id": import_run_id,
                        "source": source,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "start_date": start_date,
                        "end_date": end_date,
                        "status": status,
                        "request_count": request_count,
                        "payload_count": payload_count,
                        "normalized_rows": normalized_rows,
                        "error_message": error_message,
                    }
                ]
            ),
        )
    finally:
        conn.close()


def imported_at_utc() -> datetime:
    return datetime.now(tz=UTC)


def iter_date_windows(start_date: str, end_date: str, chunk_days: int) -> list[tuple[str, str]]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive.")
    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=chunk_days - 1), end)
        windows.append((cursor.isoformat(), window_end.isoformat()))
        cursor = window_end + timedelta(days=1)
    return windows


def has_successful_import_run(source: str, start_date: str, end_date: str) -> bool:
    conn = connect(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT 1
            FROM historical_import_runs
            WHERE source = '{source}'
              AND start_date = DATE '{start_date}'
              AND end_date = DATE '{end_date}'
              AND status = 'ok'
            LIMIT 1
            """,
        )
    finally:
        conn.close()
    return not frame.empty
