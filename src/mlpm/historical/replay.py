from __future__ import annotations

from typing import Any

from mlpm.config.settings import settings
from mlpm.storage.duckdb import connect_read_only, query_dataframe


def load_kalshi_pregame_replay(start_date: str, end_date: str):
    conn = connect_read_only(settings().duckdb_path)
    try:
        return query_dataframe(
            conn,
            f"""
            WITH ranked AS (
                SELECT
                    q.game_id,
                    g.game_date,
                    COALESCE(q.event_start_time, g.event_start_time) AS event_start_time,
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
                JOIN games_deduped g
                  ON q.game_id = g.game_id
                WHERE g.game_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                  AND q.game_id IS NOT NULL
                  AND q.outcome_team IS NOT NULL
                  AND q.home_implied_prob IS NOT NULL
                  AND COALESCE(q.pre_pitch_flag, q.quote_ts < COALESCE(q.event_start_time, g.event_start_time)) = TRUE
                  AND q.quote_ts <= COALESCE(q.event_start_time, g.event_start_time)
            ),
            selected AS (
                SELECT *
                FROM ranked
                WHERE rn = 1
            )
            SELECT
                home.game_id,
                home.game_date,
                home.event_start_time,
                GREATEST(home.quote_ts, away.quote_ts) AS snapshot_ts,
                home.quote_ts AS home_quote_ts,
                away.quote_ts AS away_quote_ts,
                home.home_team,
                home.away_team,
                home.ticker AS home_ticker,
                away.ticker AS away_ticker,
                home.market_id AS home_market_id,
                away.market_id AS away_market_id,
                home.home_implied_prob AS home_market_prob,
                away.home_implied_prob AS away_home_implied_prob,
                1.0 - away.home_implied_prob AS away_market_prob
            FROM selected home
            JOIN selected away
              ON home.game_id = away.game_id
             AND home.home_team = away.home_team
             AND home.away_team = away.away_team
            WHERE home.outcome_team = home.home_team
              AND away.outcome_team = away.away_team
            ORDER BY home.game_date, home.game_id
            """
        )
    finally:
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
