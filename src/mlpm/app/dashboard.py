from __future__ import annotations

import pandas as pd
import streamlit as st

from mlpm.config.settings import settings
from mlpm.storage.duckdb import connect_read_only


@st.cache_data(ttl=30)
def load_table(sql: str) -> pd.DataFrame:
    conn = connect_read_only(settings().duckdb_path)
    try:
        return conn.execute(sql).fetchdf()
    finally:
        conn.close()


st.set_page_config(page_title="MLPM Dashboard", layout="wide")
st.title("MLB Market Inconsistency Dashboard")

summary = load_table(
    """
    SELECT
        COUNT(*) AS total_discrepancies,
        SUM(CASE WHEN flagged THEN 1 ELSE 0 END) AS flagged_discrepancies,
        AVG(ABS(gap_bps)) AS avg_abs_gap_bps
    FROM discrepancies_deduped
    """
)
settled_summary = load_table(
    """
    SELECT
        model_name,
        COUNT(*) AS games,
        AVG(CASE WHEN correct_prediction THEN 1.0 ELSE 0.0 END) AS accuracy,
        AVG(
            CASE
                WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
            END
        ) AS log_loss
    FROM settled_predictions_deduped
    GROUP BY model_name
    ORDER BY games DESC, model_name
    """
)
settled_windows = load_table(
    """
    WITH latest_date AS (
        SELECT MAX(game_date) AS max_game_date
        FROM settled_predictions_deduped
    ),
    base AS (
        SELECT s.*
        FROM settled_predictions_deduped s
        CROSS JOIN latest_date ld
    )
    SELECT
        model_name,
        COUNT(*) AS all_games,
        AVG(CASE WHEN correct_prediction THEN 1.0 ELSE 0.0 END) AS all_accuracy,
        AVG(
            CASE
                WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
            END
        ) AS all_log_loss,
        SUM(CASE WHEN game_date >= max_game_date - 6 THEN 1 ELSE 0 END) AS last_7d_games,
        AVG(CASE WHEN game_date >= max_game_date - 6 THEN CASE WHEN correct_prediction THEN 1.0 ELSE 0.0 END END) AS last_7d_accuracy,
        AVG(CASE WHEN game_date >= max_game_date - 6 THEN CASE
                WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
            END END) AS last_7d_log_loss,
        SUM(CASE WHEN game_date >= max_game_date - 29 THEN 1 ELSE 0 END) AS last_30d_games,
        AVG(CASE WHEN game_date >= max_game_date - 29 THEN CASE WHEN correct_prediction THEN 1.0 ELSE 0.0 END END) AS last_30d_accuracy,
        AVG(CASE WHEN game_date >= max_game_date - 29 THEN CASE
                WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
            END END) AS last_30d_log_loss
    FROM base
    GROUP BY model_name
    ORDER BY all_games DESC, model_name
    """
)
bet_opportunities = load_table(
    """
    SELECT
        game_date,
        event_start_time,
        model_name,
        source,
        team,
        opponent_team,
        model_prob,
        market_prob,
        edge_bps,
        expected_value,
        is_actionable,
        is_champion
    FROM bet_opportunities_deduped
    ORDER BY game_date DESC, event_start_time DESC, edge_bps DESC
    LIMIT 200
    """
)
strategy_summary = load_table(
    """
    SELECT
        model_name,
        COUNT(*) AS bets,
        SUM(realized_return_units) AS units_won,
        SUM(realized_return_units) / COUNT(*) AS roi,
        AVG(edge_bps) AS avg_edge_bps,
        AVG(expected_value) AS avg_expected_value
    FROM settled_bet_opportunities_deduped
    GROUP BY model_name
    ORDER BY roi DESC, bets DESC, model_name
    """
)
recent_settled = load_table(
    """
    SELECT
        game_date,
        model_name,
        away_team,
        home_team,
        predicted_winner,
        winner_team,
        home_win_prob,
        correct_prediction
    FROM settled_predictions_deduped
    ORDER BY game_date DESC, event_start_time DESC
    LIMIT 100
    """
)
current = load_table(
    """
    SELECT
        d.game_id,
        d.snapshot_ts,
        d.source,
        d.market_id,
        d.team,
        d.market_prob,
        d.model_prob,
        d.gap_bps,
        d.flagged
    FROM discrepancies_deduped d
    ORDER BY snapshot_ts DESC
    LIMIT 200
    """
)
model_features = load_table(
    """
    WITH latest_model AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY game_id, team
                ORDER BY
                    CASE
                        WHEN model_name = 'mlb_win_histgb_v1' THEN 4
                        WHEN model_name = 'mlb_win_logreg_v2' THEN 3
                        WHEN model_name = 'mlb_win_logreg_v1' THEN 2
                        WHEN model_name = 'team_form_logit_v2' THEN 1
                        WHEN model_name = 'team_record_logit_v1' THEN 0
                        ELSE 0
                    END DESC,
                    collection_run_ts DESC,
                    snapshot_ts DESC
            ) AS rn
        FROM model_predictions_deduped
    )
    SELECT
        game_id,
        team,
        opponent_team,
        model_name,
        model_prob,
        season_win_pct,
        recent_win_pct,
        venue_win_pct,
        run_diff_per_game,
        streak,
        elo_rating,
        season_runs_scored_per_game,
        season_runs_allowed_per_game,
        recent_runs_scored_per_game,
        recent_runs_allowed_per_game,
        rest_days,
        venue_streak,
        travel_switch,
        is_doubleheader,
        starter_era,
        starter_whip,
        starter_strikeouts_per_9,
        starter_walks_per_9,
        bullpen_innings_3d,
        bullpen_pitches_3d,
        relievers_used_3d
    FROM latest_model
    WHERE rn = 1
    """
)
current_with_features = current.merge(model_features, on=["game_id", "team", "model_prob"], how="left")

st.subheader("Summary")
st.dataframe(summary, use_container_width=True)

st.subheader("Settled Predictions")
st.dataframe(settled_summary, use_container_width=True)

st.subheader("Settled Rolling Windows")
st.dataframe(settled_windows, use_container_width=True)

st.subheader("Current Bet Opportunities")
st.dataframe(bet_opportunities, use_container_width=True)

st.subheader("Strategy Performance")
st.dataframe(strategy_summary, use_container_width=True)

st.subheader("Recent settled picks")
st.dataframe(recent_settled, use_container_width=True)

st.subheader("Recent discrepancies")
source_filter = st.multiselect(
    "Sources",
    options=sorted(current_with_features["source"].dropna().unique().tolist()),
    default=sorted(current_with_features["source"].dropna().unique().tolist()),
)
filtered = current_with_features[current_with_features["source"].isin(source_filter)].copy()
st.dataframe(filtered, use_container_width=True)

st.subheader("Model feature context")
feature_columns = [
    "snapshot_ts",
    "source",
    "team",
    "opponent_team",
    "market_prob",
    "model_prob",
    "gap_bps",
    "season_win_pct",
    "recent_win_pct",
    "venue_win_pct",
    "run_diff_per_game",
    "streak",
    "elo_rating",
    "season_runs_scored_per_game",
    "season_runs_allowed_per_game",
    "recent_runs_scored_per_game",
    "recent_runs_allowed_per_game",
    "rest_days",
    "venue_streak",
    "travel_switch",
    "is_doubleheader",
    "starter_era",
    "starter_whip",
    "starter_strikeouts_per_9",
    "starter_walks_per_9",
    "bullpen_innings_3d",
    "bullpen_pitches_3d",
    "relievers_used_3d",
]
st.dataframe(filtered[feature_columns], use_container_width=True)

if not filtered.empty:
    st.subheader("Gap history")
    chart_df = filtered[["snapshot_ts", "gap_bps", "team"]].copy()
    chart_df["snapshot_ts"] = pd.to_datetime(chart_df["snapshot_ts"])
    st.line_chart(chart_df, x="snapshot_ts", y="gap_bps", color="team")
