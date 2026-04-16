from __future__ import annotations

from mlpm.config.settings import settings
from mlpm.features.build_features import build_features
from mlpm.models.baseline import train_baseline
from mlpm.storage.duckdb import connect_read_only, query_dataframe


def run_backtest(start_date: str, end_date: str) -> dict[str, object]:
    conn = connect_read_only(settings().duckdb_path)
    discrepancies = query_dataframe(
        conn,
        f"""
        SELECT d.*, g.game_date
        FROM discrepancies_deduped d
        JOIN games_deduped g USING (game_id)
        WHERE g.game_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
          AND d.flagged = TRUE
        ORDER BY g.game_date, d.snapshot_ts
        """,
    )
    conn.close()

    features = build_features(discrepancies)
    if features.empty or len(features) < 20:
        return {
            "status": "insufficient_data",
            "rows": len(features),
            "message": "Collect more historical snapshots before training.",
        }

    features["target"] = (features["gap_bps"].abs() > features["gap_bps"].abs().median()).astype(int)
    split_index = int(len(features) * 0.8)
    result = train_baseline(features.iloc[:split_index].copy(), features.iloc[split_index:].copy())
    return {"status": "ok", "rows": len(features), "metrics": result["report"]}
