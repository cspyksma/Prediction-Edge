from __future__ import annotations

from typing import Any

from mlpm.config.settings import settings
from mlpm.storage.duckdb import connect, query_dataframe


def run_historical_import_status(start_date: str, end_date: str) -> dict[str, Any]:
    conn = connect(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT *
            FROM historical_import_status
            WHERE start_date >= DATE '{start_date}'
              AND end_date <= DATE '{end_date}'
            ORDER BY source, start_date, end_date
            """
        )
    finally:
        conn.close()
    if frame.empty:
        return {"status": "insufficient_data", "rows": 0}
    models = {
        str(row["source"]): {
            "import_runs": int(row["import_runs"]),
            "request_count": int(row["request_count"] or 0),
            "payload_count": int(row["payload_count"] or 0),
            "normalized_rows": int(row["normalized_rows"] or 0),
        }
        for row in frame.to_dict(orient="records")
    }
    return {"status": "ok", "rows": len(frame), "sources": models}
