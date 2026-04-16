from __future__ import annotations

from typing import Any

import pandas as pd

from mlpm.config.settings import settings
from mlpm.storage.duckdb import connect_read_only, query_dataframe


def select_champion_model(reference_date: str | None = None) -> str | None:
    cfg = settings()
    conn = connect_read_only(cfg.duckdb_path)
    try:
        if not _table_has_rows(conn, "settled_bet_opportunities_deduped"):
            if not _table_has_rows(conn, "settled_predictions_deduped"):
                return None
            settled_predictions = query_dataframe(
                conn,
                f"""
                WITH latest_date AS (
                    SELECT COALESCE(MAX(game_date), CURRENT_DATE) AS max_game_date
                    FROM settled_predictions_deduped
                )
                SELECT
                    model_name,
                    COUNT(*) AS games,
                    AVG(
                        CASE
                            WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                            ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
                        END
                    ) AS log_loss
                FROM settled_predictions_deduped
                CROSS JOIN latest_date
                WHERE game_date >= max_game_date - {cfg.strategy_champion_window_days - 1}
                GROUP BY model_name
                HAVING COUNT(*) >= {cfg.strategy_champion_min_bets}
                ORDER BY log_loss ASC, games DESC, model_name
                LIMIT 1
                """
            )
            if settled_predictions.empty:
                settled_predictions = query_dataframe(
                    conn,
                    """
                    SELECT model_name, COUNT(*) AS games, AVG(
                        CASE
                            WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                            ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
                        END
                    ) AS log_loss
                    FROM settled_predictions_deduped
                    GROUP BY model_name
                    ORDER BY log_loss ASC, games DESC, model_name
                    LIMIT 1
                    """
                )
            return None if settled_predictions.empty else str(settled_predictions.iloc[0]["model_name"])

        performance = query_dataframe(
            conn,
            f"""
            WITH latest_date AS (
                SELECT COALESCE(MAX(game_date), CURRENT_DATE) AS max_game_date
                FROM settled_bet_opportunities_deduped
            )
            SELECT
                model_name,
                COUNT(*) AS bets,
                SUM(realized_return_units) AS units_won,
                AVG(expected_value) AS avg_expected_value,
                CASE WHEN COUNT(*) = 0 THEN NULL ELSE SUM(realized_return_units) / COUNT(*) END AS roi
            FROM settled_bet_opportunities_deduped
            CROSS JOIN latest_date
            WHERE game_date >= max_game_date - {cfg.strategy_champion_window_days - 1}
            GROUP BY model_name
            HAVING COUNT(*) >= {cfg.strategy_champion_min_bets}
            ORDER BY roi DESC, avg_expected_value DESC, bets DESC, model_name
            LIMIT 1
            """
        )
        if performance.empty:
            performance = query_dataframe(
                conn,
                """
                SELECT
                    model_name,
                    COUNT(*) AS bets,
                    SUM(realized_return_units) AS units_won,
                    AVG(expected_value) AS avg_expected_value,
                    CASE WHEN COUNT(*) = 0 THEN NULL ELSE SUM(realized_return_units) / COUNT(*) END AS roi
                FROM settled_bet_opportunities_deduped
                GROUP BY model_name
                ORDER BY roi DESC, avg_expected_value DESC, bets DESC, model_name
                LIMIT 1
                """
            )
        return None if performance.empty else str(performance.iloc[0]["model_name"])
    finally:
        conn.close()


def build_bet_opportunities(
    games_df: pd.DataFrame,
    normalized_quotes: pd.DataFrame,
    model_predictions: pd.DataFrame,
    champion_model_name: str | None = None,
) -> pd.DataFrame:
    if games_df.empty or normalized_quotes.empty or model_predictions.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "event_start_time",
                "snapshot_ts",
                "collection_run_ts",
                "model_name",
                "source",
                "market_id",
                "team",
                "opponent_team",
                "is_home_team",
                "model_prob",
                "market_prob",
                "edge_bps",
                "expected_value",
                "implied_decimal_odds",
                "stake_units",
                "is_actionable",
                "is_champion",
            ]
        )

    cfg = settings()
    threshold_bps = cfg.strategy_edge_threshold_bps
    market = (
        normalized_quotes[normalized_quotes["is_valid"] & normalized_quotes["is_pregame"]]
        .sort_values(["game_id", "outcome_team", "fair_prob", "snapshot_ts", "source"])
        .groupby(["game_id", "outcome_team"], as_index=False)
        .first()
        .rename(columns={"outcome_team": "team", "fair_prob": "market_prob"})
    )
    games_lookup = games_df.drop_duplicates(subset=["game_id"]).set_index("game_id")
    merged = model_predictions.merge(
        market[["game_id", "team", "source", "market_id", "market_prob"]],
        on=["game_id", "team"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["edge_bps"] = ((merged["model_prob"] - merged["market_prob"]) * 10_000).round().astype(int)
    merged["expected_value"] = (merged["model_prob"] / merged["market_prob"]) - 1.0
    merged["implied_decimal_odds"] = 1.0 / merged["market_prob"]
    merged["stake_units"] = 1.0
    merged["is_actionable"] = merged["edge_bps"] >= threshold_bps
    merged["is_champion"] = merged["model_name"].eq(champion_model_name)

    opportunities: list[dict[str, object]] = []
    for (game_id, model_name), group in merged.groupby(["game_id", "model_name"]):
        best = group.sort_values(
            ["expected_value", "edge_bps", "model_prob", "market_prob"],
            ascending=[False, False, False, True],
        ).iloc[0]
        game_row = games_lookup.loc[game_id]
        opportunities.append(
            {
                "game_id": game_id,
                "game_date": game_row["game_date"],
                "event_start_time": game_row["event_start_time"],
                "snapshot_ts": best["snapshot_ts"],
                "collection_run_ts": best["collection_run_ts"],
                "model_name": model_name,
                "source": best["source"],
                "market_id": best["market_id"],
                "team": best["team"],
                "opponent_team": best["opponent_team"],
                "is_home_team": bool(best["team"] == game_row["home_team"]),
                "model_prob": float(best["model_prob"]),
                "market_prob": float(best["market_prob"]),
                "edge_bps": int(best["edge_bps"]),
                "expected_value": float(best["expected_value"]),
                "implied_decimal_odds": float(best["implied_decimal_odds"]),
                "stake_units": 1.0,
                "is_actionable": bool(best["is_actionable"]),
                "is_champion": bool(model_name == champion_model_name),
            }
        )
    return pd.DataFrame(opportunities)


def run_bet_opportunity_report(start_date: str, end_date: str, model_name: str | None = None) -> dict[str, Any]:
    filters = [f"game_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'"]
    if model_name:
        escaped = model_name.replace("'", "''")
        filters.append(f"model_name = '{escaped}'")
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT *
            FROM bet_opportunities_deduped
            WHERE {' AND '.join(filters)}
            ORDER BY game_date DESC, event_start_time DESC, model_name
            """
        )
    finally:
        conn.close()
    if frame.empty:
        return {"status": "insufficient_data", "rows": 0}

    champion_model = select_champion_model()
    metrics: dict[str, dict[str, Any]] = {}
    for current_model, group in frame.groupby("model_name"):
        actionable = group[group["is_actionable"]]
        metrics[current_model] = {
            "opportunities": int(len(group)),
            "actionable_bets": int(len(actionable)),
            "avg_edge_bps": float(group["edge_bps"].mean()),
            "avg_expected_value": float(group["expected_value"].mean()),
            "champion_rows": int(len(group)) if current_model == champion_model else int(group["is_champion"].sum()),
        }
    return {"status": "ok", "rows": len(frame), "models": metrics, "champion_model": champion_model}


def run_strategy_performance_report(start_date: str, end_date: str, model_name: str | None = None) -> dict[str, Any]:
    filters = [f"game_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'"]
    if model_name:
        escaped = model_name.replace("'", "''")
        filters.append(f"model_name = '{escaped}'")
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT *
            FROM settled_bet_opportunities_deduped
            WHERE {' AND '.join(filters)}
            ORDER BY game_date, event_start_time, snapshot_ts
            """
        )
        daily = query_dataframe(
            conn,
            f"""
            SELECT *
            FROM strategy_performance_daily
            WHERE {' AND '.join(filters)}
            ORDER BY game_date DESC, model_name, source
            """
        )
    finally:
        conn.close()

    if frame.empty:
        return {"status": "insufficient_data", "rows": 0}

    windows: dict[str, dict[str, dict[str, Any]]] = {}
    for current_model, group in frame.groupby("model_name"):
        ordered = group.sort_values(["game_date", "event_start_time", "snapshot_ts"])
        latest_date = pd.to_datetime(ordered["game_date"]).max()
        last_7d = ordered[pd.to_datetime(ordered["game_date"]) >= (latest_date - pd.Timedelta(days=6))]
        last_30d = ordered[pd.to_datetime(ordered["game_date"]) >= (latest_date - pd.Timedelta(days=29))]
        last_50_bets = ordered.tail(50)
        windows[current_model] = {
            "all": _strategy_metrics(ordered),
            "last_7d": _strategy_metrics(last_7d),
            "last_30d": _strategy_metrics(last_30d),
            "last_50_bets": _strategy_metrics(last_50_bets),
        }

    return {
        "status": "ok",
        "rows": len(frame),
        "windows": windows,
        "daily": daily.head(100).to_dict(orient="records"),
        "champion_model": select_champion_model(),
    }


def _strategy_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "bets": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "units_won": 0.0,
            "avg_edge_bps": 0.0,
            "avg_expected_value": 0.0,
        }
    return {
        "bets": int(len(frame)),
        "win_rate": float(frame["won_bet"].mean()),
        "roi": float(frame["realized_return_units"].sum() / frame["stake_units"].sum()),
        "units_won": float(frame["realized_return_units"].sum()),
        "avg_edge_bps": float(frame["edge_bps"].mean()),
        "avg_expected_value": float(frame["expected_value"].mean()),
    }


def _table_has_rows(conn, view_name: str) -> bool:
    result = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()
    return bool(result and result[0] > 0)
