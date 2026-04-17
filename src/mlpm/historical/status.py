from __future__ import annotations

from typing import Any

import pandas as pd

from mlpm.config.settings import settings
from mlpm.storage.duckdb import connect, query_dataframe


def _int_or_zero(value: Any) -> int:
    return 0 if value is None or pd.isna(value) else int(value)


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
            "import_runs": _int_or_zero(row["import_runs"]),
            "request_count": _int_or_zero(row["request_count"]),
            "payload_count": _int_or_zero(row["payload_count"]),
            "normalized_rows": _int_or_zero(row["normalized_rows"]),
            "games_total": _int_or_zero(row["games_total"]),
            "games_with_markets": _int_or_zero(row["games_with_markets"]),
            "games_with_pregame_quotes": _int_or_zero(row["games_with_pregame_quotes"]),
            "candidate_markets": _int_or_zero(row["candidate_markets"]),
            "empty_payload_count": _int_or_zero(row["empty_payload_count"]),
            "rate_limited_count": _int_or_zero(row["rate_limited_count"]),
            "parse_error_count": _int_or_zero(row["parse_error_count"]),
        }
        for row in frame.to_dict(orient="records")
    }
    return {"status": "ok", "rows": len(frame), "sources": models}
