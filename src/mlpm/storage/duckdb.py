from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS games (
    game_id VARCHAR,
    game_date DATE,
    event_start_time TIMESTAMP,
    away_team VARCHAR,
    home_team VARCHAR,
    away_team_id BIGINT,
    home_team_id BIGINT,
    away_probable_pitcher_id BIGINT,
    away_probable_pitcher_name VARCHAR,
    away_probable_pitcher_hand VARCHAR,
    home_probable_pitcher_id BIGINT,
    home_probable_pitcher_name VARCHAR,
    home_probable_pitcher_hand VARCHAR,
    doubleheader VARCHAR,
    game_number INTEGER,
    day_night VARCHAR,
    status VARCHAR,
    snapshot_ts TIMESTAMP,
    collection_run_ts TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_snapshots (
    source VARCHAR,
    captured_at TIMESTAMP,
    file_path VARCHAR,
    collection_run_ts TIMESTAMP
);

CREATE TABLE IF NOT EXISTS historical_import_runs (
    import_run_id VARCHAR,
    source VARCHAR,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    start_date DATE,
    end_date DATE,
    status VARCHAR,
    request_count INTEGER,
    payload_count INTEGER,
    normalized_rows INTEGER,
    games_total INTEGER,
    games_with_markets INTEGER,
    games_with_pregame_quotes INTEGER,
    candidate_markets INTEGER,
    empty_payload_count INTEGER,
    rate_limited_count INTEGER,
    parse_error_count INTEGER,
    error_message VARCHAR
);

CREATE TABLE IF NOT EXISTS historical_polymarket_quotes (
    import_run_id VARCHAR,
    source VARCHAR,
    collection_mode VARCHAR,
    market_id VARCHAR,
    event_id VARCHAR,
    asset_id VARCHAR,
    game_id VARCHAR,
    event_start_time TIMESTAMP,
    quote_ts TIMESTAMP,
    outcome_team VARCHAR,
    side VARCHAR,
    home_implied_prob DOUBLE,
    raw_prob_yes DOUBLE,
    best_price_source VARCHAR,
    pre_pitch_flag BOOLEAN,
    raw_payload_path VARCHAR,
    imported_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS historical_kalshi_quotes (
    import_run_id VARCHAR,
    source VARCHAR,
    collection_mode VARCHAR,
    market_id VARCHAR,
    event_id VARCHAR,
    ticker VARCHAR,
    game_id VARCHAR,
    event_start_time TIMESTAMP,
    quote_ts TIMESTAMP,
    outcome_team VARCHAR,
    side VARCHAR,
    home_implied_prob DOUBLE,
    raw_prob_yes DOUBLE,
    quote_type VARCHAR,
    volume DOUBLE,
    open_interest DOUBLE,
    best_price_source VARCHAR,
    pre_pitch_flag BOOLEAN,
    raw_payload_path VARCHAR,
    imported_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS normalized_quotes (
    game_id VARCHAR,
    source VARCHAR,
    event_id VARCHAR,
    market_id VARCHAR,
    bookmaker VARCHAR,
    event_start_time TIMESTAMP,
    snapshot_ts TIMESTAMP,
    collection_run_ts TIMESTAMP,
    outcome_team VARCHAR,
    market_type VARCHAR,
    raw_odds DOUBLE,
    raw_price DOUBLE,
    implied_prob DOUBLE,
    fair_prob DOUBLE,
    quote_age_sec DOUBLE,
    is_pregame BOOLEAN,
    is_valid BOOLEAN
);

CREATE TABLE IF NOT EXISTS discrepancies (
    game_id VARCHAR,
    snapshot_ts TIMESTAMP,
    collection_run_ts TIMESTAMP,
    source VARCHAR,
    market_id VARCHAR,
    team VARCHAR,
    market_prob DOUBLE,
    model_prob DOUBLE,
    gap_bps INTEGER,
    freshness_pass BOOLEAN,
    mapping_pass BOOLEAN,
    flagged BOOLEAN
);

CREATE TABLE IF NOT EXISTS bet_opportunities (
    game_id VARCHAR,
    game_date DATE,
    event_start_time TIMESTAMP,
    snapshot_ts TIMESTAMP,
    collection_run_ts TIMESTAMP,
    model_name VARCHAR,
    source VARCHAR,
    market_id VARCHAR,
    team VARCHAR,
    opponent_team VARCHAR,
    is_home_team BOOLEAN,
    model_prob DOUBLE,
    market_prob DOUBLE,
    edge_bps INTEGER,
    expected_value DOUBLE,
    implied_decimal_odds DOUBLE,
    stake_units DOUBLE,
    is_actionable BOOLEAN,
    is_champion BOOLEAN
);

CREATE TABLE IF NOT EXISTS model_predictions (
    game_id VARCHAR,
    snapshot_ts TIMESTAMP,
    collection_run_ts TIMESTAMP,
    team VARCHAR,
    model_name VARCHAR,
    model_prob DOUBLE,
    games_played_floor_pass BOOLEAN,
    opponent_team VARCHAR,
    season_win_pct DOUBLE,
    recent_win_pct DOUBLE,
    venue_win_pct DOUBLE,
    run_diff_per_game DOUBLE,
    streak INTEGER,
    elo_rating DOUBLE,
    season_runs_scored_per_game DOUBLE,
    season_runs_allowed_per_game DOUBLE,
    recent_runs_scored_per_game DOUBLE,
    recent_runs_allowed_per_game DOUBLE,
    rest_days DOUBLE,
    venue_streak DOUBLE,
    travel_switch DOUBLE,
    is_doubleheader BOOLEAN,
    starter_era DOUBLE,
    starter_whip DOUBLE,
    starter_strikeouts_per_9 DOUBLE,
    starter_walks_per_9 DOUBLE,
    bullpen_innings_3d DOUBLE,
    bullpen_pitches_3d DOUBLE,
    relievers_used_3d DOUBLE,
    market_home_implied_prob DOUBLE,
    offense_vs_starter_hand DOUBLE
);

CREATE TABLE IF NOT EXISTS game_results (
    game_id VARCHAR,
    game_date DATE,
    event_start_time TIMESTAMP,
    away_team VARCHAR,
    home_team VARCHAR,
    winner_team VARCHAR,
    away_score INTEGER,
    home_score INTEGER
);

-- Persisted MLB Stats API pitching logs (boxscore-derived).
-- Backfilled once via scripts/backfill_mlb_history; training reads from here.
CREATE TABLE IF NOT EXISTS mlb_pitching_logs (
    game_id VARCHAR,
    game_date DATE,
    team VARCHAR,
    side VARCHAR,
    starting_pitcher_id BIGINT,
    starting_pitcher_name VARCHAR,
    starting_pitcher_hand VARCHAR,
    starter_innings_pitched DOUBLE,
    starter_earned_runs INTEGER,
    starter_hits INTEGER,
    starter_walks INTEGER,
    starter_strikeouts INTEGER,
    bullpen_innings DOUBLE,
    bullpen_pitches INTEGER,
    relievers_used INTEGER,
    imported_at TIMESTAMP
);

-- Persisted MLB Stats API team batting logs.
CREATE TABLE IF NOT EXISTS mlb_batting_logs (
    game_id VARCHAR,
    game_date DATE,
    team VARCHAR,
    opponent_team VARCHAR,
    at_bats INTEGER,
    hits INTEGER,
    walks INTEGER,
    strikeouts INTEGER,
    doubles INTEGER,
    triples INTEGER,
    home_runs INTEGER,
    runs_scored INTEGER,
    imported_at TIMESTAMP
);

-- Source-agnostic pre-game market probabilities (SBRO, Kalshi replay, Pinnacle close, etc.).
-- Used by the walk-forward backtest to grade model predictions against the market.
CREATE TABLE IF NOT EXISTS historical_market_priors (
    game_id VARCHAR,
    game_date DATE,
    source VARCHAR,              -- 'sbro', 'kalshi_replay', 'pinnacle', etc.
    home_team VARCHAR,
    away_team VARCHAR,
    home_moneyline_close INTEGER, -- American odds, nullable (only set for sportsbook sources)
    away_moneyline_close INTEGER,
    home_implied_prob_raw DOUBLE, -- With vig
    away_implied_prob_raw DOUBLE,
    home_fair_prob DOUBLE,       -- De-vigged
    away_fair_prob DOUBLE,
    book VARCHAR,                -- Which sportsbook the close came from (for SBRO: 'consensus')
    imported_at TIMESTAMP
);

-- Walk-forward backtest bet log. One row per flagged bet.
CREATE TABLE IF NOT EXISTS walkforward_bets (
    run_id VARCHAR,              -- Unique per backtest invocation
    model_name VARCHAR,
    game_id VARCHAR,
    game_date DATE,
    home_team VARCHAR,
    away_team VARCHAR,
    picked_team VARCHAR,
    is_home_pick BOOLEAN,
    model_prob DOUBLE,           -- Model's probability for the picked side
    market_prob DOUBLE,          -- Market fair prob for the picked side
    edge_pct DOUBLE,             -- model_prob - market_prob (positive = model sees value)
    decimal_odds DOUBLE,         -- 1 / market_prob
    stake DOUBLE,                -- Units staked (1 for flat-stake)
    won_bet BOOLEAN,
    payout DOUBLE,               -- Net profit on the bet (negative if lost)
    winner_team VARCHAR,
    train_rows INTEGER,
    train_start_date DATE,
    train_end_date DATE,
    scored_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS collector_runs (
    run_id VARCHAR,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR,
    error_message VARCHAR,
    games INTEGER,
    kalshi_quotes INTEGER,
    polymarket_quotes INTEGER,
    model_predictions INTEGER,
    normalized_quotes INTEGER,
    discrepancies INTEGER,
    bet_opportunities INTEGER,
    game_results_synced INTEGER
);

CREATE TABLE IF NOT EXISTS champion_history (
    selected_at TIMESTAMP,
    reference_date DATE,
    chosen_model VARCHAR,
    incumbent_model VARCHAR,
    challenger_model VARCHAR,
    action VARCHAR,
    reason VARCHAR,
    window_days INTEGER,
    min_bets INTEGER,
    confidence DOUBLE,
    challenger_bets INTEGER,
    challenger_roi DOUBLE,
    challenger_units DOUBLE,
    challenger_win_rate DOUBLE,
    challenger_ci_lower DOUBLE,
    challenger_ci_upper DOUBLE,
    incumbent_bets INTEGER,
    incumbent_roi DOUBLE,
    incumbent_win_rate DOUBLE
);

-- Per-game weather pulled from Open-Meteo's Historical Weather API. Populated
-- by `mlpm backfill-weather` and consumed by the training / backtest pipeline.
CREATE TABLE IF NOT EXISTS game_weather (
    game_id VARCHAR,
    game_date DATE,
    venue_team VARCHAR,
    az_cf_deg DOUBLE,
    roof_type VARCHAR,
    temp_f DOUBLE,
    wind_mph DOUBLE,
    wind_dir_deg DOUBLE,
    wind_out_to_cf_mph DOUBLE,
    wind_crossfield_mph DOUBLE,
    humidity_pct DOUBLE,
    precipitation_in DOUBLE,
    is_dome_sealed INTEGER,
    imported_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feature_importances (
    trained_at TIMESTAMP,
    model_name VARCHAR,
    train_start_date DATE,
    train_end_date DATE,
    rows_train INTEGER,
    rows_valid INTEGER,
    method VARCHAR,
    feature VARCHAR,
    importance DOUBLE,
    importance_std DOUBLE,
    rank INTEGER
);
"""

MIGRATION_SQL = [
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS away_probable_pitcher_id BIGINT",
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS away_probable_pitcher_name VARCHAR",
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS away_probable_pitcher_hand VARCHAR",
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS home_probable_pitcher_id BIGINT",
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS home_probable_pitcher_name VARCHAR",
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS home_probable_pitcher_hand VARCHAR",
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS doubleheader VARCHAR",
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS game_number INTEGER",
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS day_night VARCHAR",
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS collection_run_ts TIMESTAMP",
    "ALTER TABLE raw_snapshots ADD COLUMN IF NOT EXISTS collection_run_ts TIMESTAMP",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS request_count INTEGER",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS payload_count INTEGER",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS normalized_rows INTEGER",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS games_total INTEGER",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS games_with_markets INTEGER",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS games_with_pregame_quotes INTEGER",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS candidate_markets INTEGER",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS empty_payload_count INTEGER",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS rate_limited_count INTEGER",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS parse_error_count INTEGER",
    "ALTER TABLE historical_import_runs ADD COLUMN IF NOT EXISTS error_message VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS import_run_id VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS source VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS collection_mode VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS market_id VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS event_id VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS asset_id VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS game_id VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS event_start_time TIMESTAMP",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS quote_ts TIMESTAMP",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS outcome_team VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS side VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS home_implied_prob DOUBLE",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS raw_prob_yes DOUBLE",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS best_price_source VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS pre_pitch_flag BOOLEAN",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS raw_payload_path VARCHAR",
    "ALTER TABLE historical_polymarket_quotes ADD COLUMN IF NOT EXISTS imported_at TIMESTAMP",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS import_run_id VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS source VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS collection_mode VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS market_id VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS event_id VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS ticker VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS game_id VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS event_start_time TIMESTAMP",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS quote_ts TIMESTAMP",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS outcome_team VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS side VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS home_implied_prob DOUBLE",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS raw_prob_yes DOUBLE",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS quote_type VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS volume DOUBLE",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS open_interest DOUBLE",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS best_price_source VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS pre_pitch_flag BOOLEAN",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS raw_payload_path VARCHAR",
    "ALTER TABLE historical_kalshi_quotes ADD COLUMN IF NOT EXISTS imported_at TIMESTAMP",
    "ALTER TABLE normalized_quotes ADD COLUMN IF NOT EXISTS collection_run_ts TIMESTAMP",
    "ALTER TABLE discrepancies ADD COLUMN IF NOT EXISTS collection_run_ts TIMESTAMP",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS collection_run_ts TIMESTAMP",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS game_date DATE",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS event_start_time TIMESTAMP",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS model_name VARCHAR",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS source VARCHAR",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS market_id VARCHAR",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS team VARCHAR",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS opponent_team VARCHAR",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS is_home_team BOOLEAN",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS model_prob DOUBLE",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS market_prob DOUBLE",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS edge_bps INTEGER",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS expected_value DOUBLE",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS implied_decimal_odds DOUBLE",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS stake_units DOUBLE",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS is_actionable BOOLEAN",
    "ALTER TABLE bet_opportunities ADD COLUMN IF NOT EXISTS is_champion BOOLEAN",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS collection_run_ts TIMESTAMP",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS opponent_team VARCHAR",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS season_win_pct DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS recent_win_pct DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS venue_win_pct DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS run_diff_per_game DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS streak INTEGER",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS elo_rating DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS season_runs_scored_per_game DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS season_runs_allowed_per_game DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS recent_runs_scored_per_game DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS recent_runs_allowed_per_game DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS rest_days DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS venue_streak DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS travel_switch DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS is_doubleheader BOOLEAN",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS starter_era DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS starter_whip DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS starter_strikeouts_per_9 DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS starter_walks_per_9 DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS bullpen_innings_3d DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS bullpen_pitches_3d DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS relievers_used_3d DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS market_home_implied_prob DOUBLE",
    "ALTER TABLE model_predictions ADD COLUMN IF NOT EXISTS offense_vs_starter_hand DOUBLE",
    "ALTER TABLE game_results ADD COLUMN IF NOT EXISTS event_start_time TIMESTAMP",
    "ALTER TABLE game_results ADD COLUMN IF NOT EXISTS away_team VARCHAR",
    "ALTER TABLE game_results ADD COLUMN IF NOT EXISTS home_team VARCHAR",
    "ALTER TABLE collector_runs ADD COLUMN IF NOT EXISTS bet_opportunities INTEGER",
]

VIEW_SQL = """
CREATE OR REPLACE VIEW games_deduped AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id, COALESCE(collection_run_ts, snapshot_ts)
            ORDER BY snapshot_ts DESC
        ) AS rn
    FROM games
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;

CREATE OR REPLACE VIEW normalized_quotes_deduped AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY source, market_id, outcome_team, COALESCE(collection_run_ts, snapshot_ts)
            ORDER BY snapshot_ts DESC
        ) AS rn
    FROM normalized_quotes
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;

CREATE OR REPLACE VIEW model_predictions_deduped AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id, team, model_name, COALESCE(collection_run_ts, snapshot_ts)
            ORDER BY snapshot_ts DESC
        ) AS rn
    FROM model_predictions
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;

CREATE OR REPLACE VIEW discrepancies_deduped AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY source, market_id, team, COALESCE(collection_run_ts, snapshot_ts)
            ORDER BY snapshot_ts DESC
        ) AS rn
    FROM discrepancies
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;

CREATE OR REPLACE VIEW bet_opportunities_deduped AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id, model_name, COALESCE(collection_run_ts, snapshot_ts)
            ORDER BY snapshot_ts DESC
        ) AS rn
    FROM bet_opportunities
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;

CREATE OR REPLACE VIEW game_results_deduped AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id
            ORDER BY game_date DESC
        ) AS rn
    FROM game_results
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;

CREATE OR REPLACE VIEW mlb_pitching_logs_deduped AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id, team
            ORDER BY imported_at DESC
        ) AS rn
    FROM mlb_pitching_logs
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;

CREATE OR REPLACE VIEW mlb_batting_logs_deduped AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id, team
            ORDER BY imported_at DESC
        ) AS rn
    FROM mlb_batting_logs
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;

CREATE OR REPLACE VIEW game_weather_deduped AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id
            ORDER BY imported_at DESC
        ) AS rn
    FROM game_weather
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;

CREATE OR REPLACE VIEW historical_market_priors_deduped AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id, source
            ORDER BY imported_at DESC
        ) AS rn
    FROM historical_market_priors
)
SELECT * EXCLUDE (rn)
FROM ranked
WHERE rn = 1;

CREATE OR REPLACE VIEW settled_predictions_deduped AS
WITH latest_games AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY game_id
            ORDER BY COALESCE(collection_run_ts, snapshot_ts) DESC, snapshot_ts DESC
        ) AS rn
    FROM games
),
pregame_home_predictions AS (
    SELECT
        mp.game_id,
        g.game_date,
        g.event_start_time,
        mp.snapshot_ts,
        mp.collection_run_ts,
        g.home_team,
        g.away_team,
        mp.model_name,
        mp.model_prob AS home_win_prob,
        ROW_NUMBER() OVER (
            PARTITION BY mp.game_id, mp.model_name
            ORDER BY mp.snapshot_ts DESC, COALESCE(mp.collection_run_ts, mp.snapshot_ts) DESC
        ) AS rn
    FROM model_predictions mp
    JOIN latest_games g
      ON mp.game_id = g.game_id
     AND mp.team = g.home_team
    WHERE g.rn = 1
      AND mp.snapshot_ts <= g.event_start_time
)
SELECT
    p.game_id,
    p.game_date,
    p.event_start_time,
    p.snapshot_ts,
    p.collection_run_ts,
    p.model_name,
    p.home_team,
    p.away_team,
    p.home_win_prob,
    1.0 - p.home_win_prob AS away_win_prob,
    CASE
        WHEN p.home_win_prob >= 0.5 THEN p.home_team
        ELSE p.away_team
    END AS predicted_winner,
    r.winner_team,
    r.away_score,
    r.home_score,
    CASE
        WHEN r.winner_team = p.home_team THEN 1
        ELSE 0
    END AS actual_home_win,
    CASE
        WHEN (p.home_win_prob >= 0.5 AND r.winner_team = p.home_team)
          OR (p.home_win_prob < 0.5 AND r.winner_team = p.away_team)
        THEN TRUE
        ELSE FALSE
    END AS correct_prediction
FROM pregame_home_predictions p
JOIN game_results_deduped r
  ON p.game_id = r.game_id
WHERE p.rn = 1;

CREATE OR REPLACE VIEW settled_prediction_daily AS
SELECT
    game_date,
    model_name,
    COUNT(*) AS games,
    AVG(CASE WHEN correct_prediction THEN 1.0 ELSE 0.0 END) AS accuracy,
    AVG(
        CASE
            WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
            ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
        END
    ) AS log_loss,
    AVG(POWER(home_win_prob - actual_home_win, 2)) AS brier_score
FROM settled_predictions_deduped
GROUP BY game_date, model_name;

CREATE OR REPLACE VIEW settled_bet_opportunities_deduped AS
SELECT
    b.game_id,
    b.game_date,
    b.event_start_time,
    b.snapshot_ts,
    b.collection_run_ts,
    b.model_name,
    b.source,
    b.market_id,
    b.team,
    b.opponent_team,
    b.is_home_team,
    b.model_prob,
    b.market_prob,
    b.edge_bps,
    b.expected_value,
    b.implied_decimal_odds,
    b.stake_units,
    b.is_actionable,
    b.is_champion,
    r.winner_team,
    r.away_score,
    r.home_score,
    CASE WHEN r.winner_team = b.team THEN TRUE ELSE FALSE END AS won_bet,
    CASE
        WHEN r.winner_team = b.team THEN (b.implied_decimal_odds - 1.0) * b.stake_units
        ELSE -1.0 * b.stake_units
    END AS realized_return_units
FROM bet_opportunities_deduped b
JOIN game_results_deduped r
  ON b.game_id = r.game_id
WHERE b.is_actionable = TRUE;

CREATE OR REPLACE VIEW strategy_performance_daily AS
SELECT
    game_date,
    model_name,
    source,
    COUNT(*) AS bets,
    AVG(CASE WHEN won_bet THEN 1.0 ELSE 0.0 END) AS win_rate,
    SUM(realized_return_units) AS units_won,
    CASE WHEN COUNT(*) = 0 THEN NULL ELSE SUM(realized_return_units) / COUNT(*) END AS roi,
    AVG(edge_bps) AS avg_edge_bps,
    AVG(expected_value) AS avg_expected_value
FROM settled_bet_opportunities_deduped
GROUP BY game_date, model_name, source;

CREATE OR REPLACE VIEW latest_feature_importances AS
WITH latest AS (
    SELECT
        model_name,
        MAX(trained_at) AS trained_at
    FROM feature_importances
    GROUP BY model_name
)
SELECT
    fi.*
FROM feature_importances fi
JOIN latest l
  ON fi.model_name = l.model_name
 AND fi.trained_at = l.trained_at;

CREATE OR REPLACE VIEW latest_champion_selection AS
SELECT *
FROM champion_history
WHERE selected_at = (SELECT MAX(selected_at) FROM champion_history);

CREATE OR REPLACE VIEW historical_import_status AS
SELECT
    source,
    start_date,
    end_date,
    COUNT(*) AS import_runs,
    SUM(request_count) AS request_count,
    SUM(payload_count) AS payload_count,
    SUM(normalized_rows) AS normalized_rows,
    SUM(games_total) AS games_total,
    SUM(games_with_markets) AS games_with_markets,
    SUM(games_with_pregame_quotes) AS games_with_pregame_quotes,
    SUM(candidate_markets) AS candidate_markets,
    SUM(empty_payload_count) AS empty_payload_count,
    SUM(rate_limited_count) AS rate_limited_count,
    SUM(parse_error_count) AS parse_error_count,
    MAX(completed_at) AS last_completed_at
FROM historical_import_runs
GROUP BY source, start_date, end_date;
"""

LOCK_RETRY_ATTEMPTS = 20
LOCK_RETRY_DELAY_SECONDS = 0.25


def _is_lock_conflict(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "being used by another process" in message or "cannot open file" in message


def _open_connection(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    last_error: BaseException | None = None
    for attempt in range(LOCK_RETRY_ATTEMPTS):
        try:
            return duckdb.connect(str(path), read_only=read_only)
        except duckdb.IOException as exc:
            last_error = exc
            if not _is_lock_conflict(exc) or attempt == (LOCK_RETRY_ATTEMPTS - 1):
                break
            time.sleep(LOCK_RETRY_DELAY_SECONDS)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to open DuckDB connection for path={path}")


def ensure_database(path: Path) -> None:
    conn = connect(path)
    conn.close()


def connect(path: Path) -> duckdb.DuckDBPyConnection:
    conn = _open_connection(path, read_only=False)
    conn.execute(SCHEMA_SQL)
    for statement in MIGRATION_SQL:
        conn.execute(statement)
    conn.execute(
        """
        UPDATE games
        SET collection_run_ts = snapshot_ts
        WHERE collection_run_ts IS NULL
        """
    )
    conn.execute(
        """
        UPDATE raw_snapshots
        SET collection_run_ts = captured_at
        WHERE collection_run_ts IS NULL
        """
    )
    conn.execute(
        """
        UPDATE normalized_quotes
        SET collection_run_ts = snapshot_ts
        WHERE collection_run_ts IS NULL
        """
    )
    conn.execute(
        """
        UPDATE discrepancies
        SET collection_run_ts = snapshot_ts
        WHERE collection_run_ts IS NULL
        """
    )
    conn.execute(
        """
        UPDATE bet_opportunities
        SET collection_run_ts = snapshot_ts
        WHERE collection_run_ts IS NULL
        """
    )
    conn.execute(
        """
        UPDATE model_predictions
        SET collection_run_ts = snapshot_ts
        WHERE collection_run_ts IS NULL
        """
    )
    conn.execute(VIEW_SQL)
    return conn


def connect_read_only(path: Path) -> duckdb.DuckDBPyConnection:
    return _open_connection(path, read_only=True)


def append_dataframe(conn: duckdb.DuckDBPyConnection, table_name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    conn.register("frame_view", frame)
    columns = ", ".join(frame.columns)
    conn.execute(f"INSERT INTO {table_name} ({columns}) SELECT {columns} FROM frame_view")
    conn.unregister("frame_view")


def replace_dataframe(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> None:
    if frame.empty:
        return
    conn.register("frame_view", frame)
    join_predicate = " AND ".join(
        [f"target.{column} = source.{column}" for column in key_columns]
    )
    conn.execute(
        f"""
        DELETE FROM {table_name} AS target
        USING frame_view AS source
        WHERE {join_predicate}
        """
    )
    columns = ", ".join(frame.columns)
    conn.execute(f"INSERT INTO {table_name} ({columns}) SELECT {columns} FROM frame_view")
    conn.unregister("frame_view")


def query_dataframe(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[object] | tuple[object, ...] | None = None,
) -> pd.DataFrame:
    if params is None:
        return conn.execute(sql).fetchdf()
    return conn.execute(sql, params).fetchdf()
