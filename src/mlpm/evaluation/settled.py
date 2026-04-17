from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlpm.config.settings import settings
from mlpm.storage.duckdb import connect_read_only, query_dataframe


def run_settled_prediction_report(
    start_date: str,
    end_date: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    filters = ["game_date BETWEEN ? AND ?"]
    params: list[object] = [start_date, end_date]
    if model_name:
        filters.append("model_name = ?")
        params.append(model_name)

    conn = connect_read_only(settings().duckdb_path)
    try:
        settled = query_dataframe(
            conn,
            f"""
            SELECT *
            FROM settled_predictions_deduped
            WHERE {' AND '.join(filters)}
            ORDER BY game_date, event_start_time, snapshot_ts
            """,
            params=params,
        )
    finally:
        conn.close()

    if settled.empty:
        return {"status": "insufficient_data", "rows": 0}

    metrics_by_model: dict[str, dict[str, Any]] = {}
    for current_model, group in settled.groupby("model_name"):
        metrics_by_model[current_model] = _game_level_metrics(group)

    recent = settled.sort_values(["game_date", "event_start_time", "snapshot_ts"], ascending=False).head(20)
    recent_rows = [
        {
            "game_date": str(row["game_date"]),
            "model_name": row["model_name"],
            "away_team": row["away_team"],
            "home_team": row["home_team"],
            "predicted_winner": row["predicted_winner"],
            "winner_team": row["winner_team"],
            "home_win_prob": float(row["home_win_prob"]),
            "correct_prediction": bool(row["correct_prediction"]),
        }
        for row in recent.to_dict(orient="records")
    ]

    return {
        "status": "ok",
        "rows": len(settled),
        "models": metrics_by_model,
        "recent": recent_rows,
    }


def run_settled_window_report(
    start_date: str,
    end_date: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    base = run_settled_prediction_report(start_date, end_date, model_name=model_name)
    if base.get("status") != "ok":
        return base

    filters = ["game_date BETWEEN ? AND ?"]
    params: list[object] = [start_date, end_date]
    if model_name:
        filters.append("model_name = ?")
        params.append(model_name)

    conn = connect_read_only(settings().duckdb_path)
    try:
        settled = query_dataframe(
            conn,
            f"""
            SELECT *
            FROM settled_predictions_deduped
            WHERE {' AND '.join(filters)}
            ORDER BY game_date, event_start_time, snapshot_ts
            """,
            params=params,
        )
        daily = query_dataframe(
            conn,
            f"""
            SELECT *
            FROM settled_prediction_daily
            WHERE {' AND '.join(filters)}
            ORDER BY game_date DESC, model_name
            """,
            params=params,
        )
    finally:
        conn.close()

    if settled.empty:
        return {"status": "insufficient_data", "rows": 0}

    windows_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for current_model, group in settled.groupby("model_name"):
        ordered = group.sort_values(["game_date", "event_start_time", "snapshot_ts"])
        latest_date = pd.to_datetime(ordered["game_date"]).max()
        last_7d = ordered[pd.to_datetime(ordered["game_date"]) >= (latest_date - pd.Timedelta(days=6))]
        last_30d = ordered[pd.to_datetime(ordered["game_date"]) >= (latest_date - pd.Timedelta(days=29))]
        last_50_games = ordered.tail(50)
        windows_by_model[current_model] = {
            "all": _game_level_metrics(ordered),
            "last_7d": _game_level_metrics(last_7d),
            "last_30d": _game_level_metrics(last_30d),
            "last_50_games": _game_level_metrics(last_50_games),
        }

    daily_rows = [
        {
            "game_date": str(row["game_date"]),
            "model_name": row["model_name"],
            "games": int(row["games"]),
            "accuracy": float(row["accuracy"]),
            "log_loss": float(row["log_loss"]),
            "brier_score": float(row["brier_score"]),
        }
        for row in daily.head(50).to_dict(orient="records")
    ]

    return {
        "status": "ok",
        "rows": len(settled),
        "windows": windows_by_model,
        "daily": daily_rows,
    }


def _game_level_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    probabilities = np.clip(frame["home_win_prob"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    actual = frame["actual_home_win"].to_numpy(dtype=float)
    predicted_home = probabilities >= 0.5
    accuracy = float((predicted_home == actual.astype(bool)).mean())
    log_loss = float(-(actual * np.log(probabilities) + (1.0 - actual) * np.log(1.0 - probabilities)).mean())
    brier = float(np.mean((probabilities - actual) ** 2))
    return {
        "games": int(len(frame)),
        "accuracy": accuracy,
        "log_loss": log_loss,
        "brier_score": brier,
    }
