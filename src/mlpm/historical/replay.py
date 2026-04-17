from __future__ import annotations

from typing import Any

import pandas as pd

from mlpm.config.settings import settings
from mlpm.storage.duckdb import connect_read_only, query_dataframe


def load_kalshi_pregame_replay(start_date: str, end_date: str, games_df: pd.DataFrame | None = None):
    conn = connect_read_only(settings().duckdb_path)
    try:
        if games_df is not None:
            games = games_df[["game_id", "game_date", "event_start_time", "home_team", "away_team"]].copy()
            conn.register("replay_games", games)
            games_source = "replay_games"
            date_filter = f"CAST(g.game_date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'"
        else:
            games_source = "games_deduped"
            date_filter = f"CAST(g.game_date AS DATE) BETWEEN DATE '{start_date}' AND DATE '{end_date}'"
        return query_dataframe(
            conn,
            f"""
            WITH ranked AS (
                SELECT
                    q.game_id,
                    CAST(g.game_date AS DATE) AS game_date,
                    COALESCE(q.event_start_time, CAST(g.event_start_time AS TIMESTAMP)) AS event_start_time,
                    q.quote_ts,
                    q.ticker,
                    q.market_id,
                    q.outcome_team,
                    q.home_implied_prob,
                    q.raw_prob_yes,
                    g.home_team,
                    g.away_team,
                    ROW_NUMBER() OVER (
                        PARTITION BY q.game_id, q.outcome_team
                        ORDER BY q.quote_ts DESC, q.imported_at DESC
                    ) AS rn
                FROM historical_kalshi_quotes q
                JOIN {games_source} g
                  ON q.game_id = g.game_id
                WHERE {date_filter}
                  AND q.game_id IS NOT NULL
                  AND q.outcome_team IS NOT NULL
                  AND q.home_implied_prob IS NOT NULL
                  AND q.quote_ts <= COALESCE(q.event_start_time, CAST(g.event_start_time AS TIMESTAMP))
            ),
            selected AS (
                SELECT *
                FROM ranked
                WHERE rn = 1
            ),
            latest_game_quote AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY game_id
                        ORDER BY quote_ts DESC, home_implied_prob DESC, ticker
                    ) AS rn
                FROM selected
            ),
            aggregated AS (
                SELECT
                    s.game_id,
                    s.game_date,
                    s.event_start_time,
                    MAX(s.quote_ts) AS snapshot_ts,
                    MAX(CASE WHEN s.outcome_team = s.home_team THEN s.quote_ts END) AS home_quote_ts,
                    MAX(CASE WHEN s.outcome_team = s.away_team THEN s.quote_ts END) AS away_quote_ts,
                    s.home_team,
                    s.away_team,
                    MAX(CASE WHEN s.outcome_team = s.home_team THEN s.ticker END) AS home_ticker,
                    MAX(CASE WHEN s.outcome_team = s.away_team THEN s.ticker END) AS away_ticker,
                    MAX(CASE WHEN s.outcome_team = s.home_team THEN s.market_id END) AS home_market_id,
                    MAX(CASE WHEN s.outcome_team = s.away_team THEN s.market_id END) AS away_market_id,
                    MAX(CASE WHEN l.rn = 1 THEN l.home_implied_prob END) AS home_market_prob,
                    COUNT(*) AS source_rows
                FROM selected s
                JOIN latest_game_quote l
                  ON s.game_id = l.game_id
                 AND s.game_date = l.game_date
                 AND s.event_start_time = l.event_start_time
                 AND s.home_team = l.home_team
                 AND s.away_team = l.away_team
                 AND s.quote_ts = l.quote_ts
                 AND s.ticker = l.ticker
                GROUP BY s.game_id, s.game_date, s.event_start_time, s.home_team, s.away_team
            )
            SELECT
                game_id,
                game_date,
                event_start_time,
                snapshot_ts,
                home_quote_ts,
                away_quote_ts,
                home_team,
                away_team,
                COALESCE(home_ticker, away_ticker) AS home_ticker,
                COALESCE(away_ticker, home_ticker) AS away_ticker,
                COALESCE(home_market_id, away_market_id) AS home_market_id,
                COALESCE(away_market_id, home_market_id) AS away_market_id,
                home_market_prob,
                1.0 - home_market_prob AS away_market_prob,
                source_rows
            FROM aggregated
            WHERE home_market_prob IS NOT NULL
            ORDER BY game_date, game_id
            """
        )
    finally:
        if games_df is not None:
            conn.unregister("replay_games")
        conn.close()


def build_kalshi_replay_quote_rows(replay_df) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if replay_df.empty:
        return rows
    for row in replay_df.to_dict(orient="records"):
        snapshot_ts = row["snapshot_ts"]
        rows.append(
            {
                "game_id": row["game_id"],
                "source": "kalshi_historical_replay",
                "market_id": row["home_market_id"] or row["home_ticker"],
                "outcome_team": row["home_team"],
                "fair_prob": float(row["home_market_prob"]),
                "is_valid": True,
                "is_pregame": True,
                "snapshot_ts": snapshot_ts,
            }
        )
        rows.append(
            {
                "game_id": row["game_id"],
                "source": "kalshi_historical_replay",
                "market_id": row["away_market_id"] or row["away_ticker"],
                "outcome_team": row["away_team"],
                "fair_prob": float(row["away_market_prob"]),
                "is_valid": True,
                "is_pregame": True,
                "snapshot_ts": snapshot_ts,
            }
        )
    return rows
