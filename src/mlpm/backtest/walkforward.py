"""Walk-forward, retrain-monthly backtest for the MLB game-outcome model.

Pipeline
--------
1. Load game results + pitching/batting logs from DuckDB (backfilled via
   `mlpm backfill-mlb`). Load SBRO historical closing lines from
   `historical_market_priors` (loaded via `mlpm ingest-sbro`).
2. Build one feature row per game using `build_training_dataset` — this
   function iterates chronologically and computes every row's features from
   strictly prior games, so there is no leakage within a single build.
3. Walk forward month by month:
     for each month M in [start, end]:
         train_df  = feature rows with game_date < M_start
         score_df  = feature rows with game_date in [M_start, M_end]
         fit a classifier pipeline on train_df[FEATURE_COLUMNS]
         predict home-win prob for each row in score_df
4. Join scored rows with SBRO priors on game_id. Flag a bet whenever
   (model_prob - market_fair_prob) > edge_threshold for either side.
5. Grade: the picked team's winner_team match decides `won_bet`. Payout at
   the SBRO closing price (decimal_odds).
6. Persist one row per bet into `walkforward_bets`.

The model here intentionally has NO market feature — we want an independent
signal to measure discrepancies against.
"""

from __future__ import annotations

import logging
import uuid
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from mlpm.config.settings import settings
from mlpm.models.game_outcome import (
    FEATURE_COLUMNS,
    _build_histgb_pipeline,
    _build_knn_pipeline,
    _build_logistic_pipeline,
    _build_svm_pipeline,
    build_training_dataset,
)
from mlpm.storage.duckdb import connect, query_dataframe, replace_dataframe

logger = logging.getLogger(__name__)

DEFAULT_EDGE_THRESHOLD = 0.03  # 3pp — picked team must be ≥3pp richer than the market
DEFAULT_MIN_TRAIN_ROWS = 500   # Below this the first few months' models are too noisy to trust.
SUPPORTED_MODELS = ("logreg", "histgb", "knn", "svm")


@dataclass
class MonthlyBacktestSummary:
    model_name: str
    month_start: date
    month_end: date
    train_rows: int
    scored_games: int
    matched_to_market: int
    flagged_bets: int
    wins: int
    losses: int
    stake: float
    payout: float


def _load_feature_frame(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Pull persisted MLB history from DuckDB and run it through the
    chronological feature builder.

    `build_training_dataset` returns one row per game with FEATURE_COLUMNS
    already present. We keep `market_home_implied_prob` column unpopulated
    (NaN) — it's in the frame for compat but no longer in FEATURE_COLUMNS.
    """
    results_df = query_dataframe(
        conn,
        """
        SELECT
            game_id,
            CAST(game_date AS DATE) AS game_date,
            event_start_time,
            home_team,
            away_team,
            winner_team,
            home_score,
            away_score
        FROM game_results
        WHERE game_date BETWEEN ? AND ?
          AND winner_team IS NOT NULL
        """,
        (start_date, end_date),
    )
    if results_df.empty:
        logger.warning("No game_results rows in [%s, %s].", start_date, end_date)
        return pd.DataFrame()

    pitching_df = query_dataframe(
        conn,
        "SELECT * EXCLUDE (imported_at) FROM mlb_pitching_logs_deduped WHERE game_date BETWEEN ? AND ?",
        (start_date, end_date),
    )
    batting_df = query_dataframe(
        conn,
        "SELECT * EXCLUDE (imported_at) FROM mlb_batting_logs_deduped WHERE game_date BETWEEN ? AND ?",
        (start_date, end_date),
    )
    weather_df = _load_weather_frame(conn, start_date, end_date)

    # Market priors are NOT passed in — we're deliberately training market-free.
    feature_df = build_training_dataset(
        results_df,
        pitching_df,
        batting_logs_df=batting_df,
        market_priors_df=None,
        weather_df=weather_df,
    )
    if feature_df.empty:
        return feature_df
    feature_df["game_date"] = pd.to_datetime(feature_df["game_date"]).dt.date
    return feature_df


def _load_weather_frame(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Load per-game weather features. Returns an empty frame if the table is
    absent or empty, which causes build_training_dataset to fall back to the
    neutral defaults in _WEATHER_FEATURE_DEFAULTS."""
    empty = pd.DataFrame(
        columns=[
            "game_id",
            "weather_temp_f",
            "weather_wind_out_to_cf_mph",
            "weather_humidity_pct",
            "weather_precipitation_in",
            "weather_is_dome_sealed",
        ]
    )
    try:
        df = query_dataframe(
            conn,
            """
            SELECT
                game_id,
                temp_f AS weather_temp_f,
                wind_out_to_cf_mph AS weather_wind_out_to_cf_mph,
                humidity_pct AS weather_humidity_pct,
                precipitation_in AS weather_precipitation_in,
                CAST(is_dome_sealed AS DOUBLE) AS weather_is_dome_sealed
            FROM game_weather_deduped
            WHERE game_date BETWEEN ? AND ?
            """,
            (start_date, end_date),
        )
    except Exception as exc:
        logger.warning("game_weather lookup failed in walkforward: %s", exc)
        return empty
    return df if df is not None and not df.empty else empty


def _load_sbro_priors(conn, start_date: str, end_date: str) -> pd.DataFrame:
    df = query_dataframe(
        conn,
        """
        SELECT
            game_id,
            CAST(game_date AS DATE) AS game_date,
            home_team,
            away_team,
            home_fair_prob,
            away_fair_prob,
            home_moneyline_close,
            away_moneyline_close,
            book
        FROM historical_market_priors_deduped
        WHERE source = 'sportsbookreviewsonline'
          AND game_date BETWEEN ? AND ?
          AND home_fair_prob IS NOT NULL
        """,
        (start_date, end_date),
    )
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["game_id"] = df["game_id"].astype(str)
    return df


def _american_to_decimal(ml: float | int | None) -> float | None:
    if ml is None or pd.isna(ml):
        return None
    ml = float(ml)
    if ml > 0:
        return 1.0 + ml / 100.0
    return 1.0 + 100.0 / abs(ml)


def _month_range(start: date, end: date) -> Iterator[tuple[date, date]]:
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        month_end = date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1])
        yield cursor, min(month_end, end)
        year = cursor.year + (1 if cursor.month == 12 else 0)
        month = 1 if cursor.month == 12 else cursor.month + 1
        cursor = date(year, month, 1)


def _fit_model(train_df: pd.DataFrame, model_name: str):
    if model_name == "logreg":
        pipeline = _build_logistic_pipeline()
    elif model_name == "histgb":
        pipeline = _build_histgb_pipeline()
    elif model_name == "knn":
        pipeline = _build_knn_pipeline()
    elif model_name == "svm":
        pipeline = _build_svm_pipeline()
    else:
        raise ValueError(f"Unsupported model_name={model_name}. Must be one of {SUPPORTED_MODELS}.")
    target = train_df["target_home_win"].astype(int)
    pipeline.fit(train_df[FEATURE_COLUMNS], target)
    return pipeline


def run_walkforward_backtest(
    start_date: str,
    end_date: str,
    *,
    warmup_start: str | None = None,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    model_name: str = "all",
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
    stake: float = 1.0,
    db_path: Path | None = None,
    run_id: str | None = None,
) -> dict:
    """Run a walk-forward, retrain-monthly backtest.

    Args:
        start_date: First day of evaluation (e.g. '2017-04-01').
        end_date: Last day of evaluation (e.g. '2026-04-01').
        warmup_start: Earliest date to pull into training-only history.
            Defaults to two years before `start_date`.
        edge_threshold: Flag a bet when (model_prob - market_fair_prob) ≥ this.
        model_name: One of SUPPORTED_MODELS or 'all'.
        min_train_rows: Skip months whose training history is smaller than this.
        stake: Flat stake per bet. Payout = stake * (decimal_odds - 1) on a win,
               -stake on a loss.
        db_path: Override DuckDB path (defaults to settings().duckdb_path).
        run_id: Optional run identifier. A uuid is generated if absent.

    Returns:
        Summary dict with per-month and grand totals, plus the run_id so
        the caller can read `walkforward_bets` for detailed analysis.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    if warmup_start is None:
        warmup_start_date = date(start.year - 2, 1, 1)
    else:
        warmup_start_date = date.fromisoformat(warmup_start)

    run_id = run_id or f"wf-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    db_path = db_path or settings().duckdb_path
    logger.info(
        "walkforward run_id=%s warmup_start=%s eval=[%s,%s] model=%s edge_threshold=%.3f",
        run_id, warmup_start_date, start, end, model_name, edge_threshold,
    )
    selected_models = list(SUPPORTED_MODELS) if model_name == "all" else [model_name]
    unsupported = [name for name in selected_models if name not in SUPPORTED_MODELS]
    if unsupported:
        raise ValueError(f"Unsupported model_name={unsupported[0]}. Must be one of {SUPPORTED_MODELS} or 'all'.")

    conn = connect(db_path)
    try:
        feature_df = _load_feature_frame(conn, warmup_start_date.isoformat(), end.isoformat())
        if feature_df.empty:
            return {"status": "insufficient_data", "message": "No feature rows built. Run backfill-mlb first."}

        priors_df = _load_sbro_priors(conn, start.isoformat(), end.isoformat())
        if priors_df.empty:
            return {"status": "insufficient_data", "message": "No SBRO priors found. Run ingest-sbro first."}
        priors_by_id = priors_df.set_index("game_id").to_dict(orient="index")

        monthly_rows: list[dict] = []
        bet_rows: list[dict] = []
        feature_df = feature_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        feature_df["game_id"] = feature_df["game_id"].astype(str)

        for month_start, month_end in _month_range(start, end):
            train_mask = feature_df["game_date"] < month_start
            score_mask = (feature_df["game_date"] >= month_start) & (feature_df["game_date"] <= month_end)
            train_df = feature_df.loc[train_mask].copy()
            score_df = feature_df.loc[score_mask].copy()

            if len(train_df) < min_train_rows or train_df["target_home_win"].nunique() < 2:
                logger.info("  skip %s: train_rows=%s < min=%s", month_start.isoformat()[:7], len(train_df), min_train_rows)
                for current_model_name in selected_models:
                    monthly_rows.append(
                        MonthlyBacktestSummary(
                            model_name=current_model_name,
                            month_start=month_start,
                            month_end=month_end,
                            train_rows=len(train_df),
                            scored_games=len(score_df),
                            matched_to_market=0,
                            flagged_bets=0,
                            wins=0,
                            losses=0,
                            stake=0.0,
                            payout=0.0,
                        ).__dict__
                    )
                continue
            if score_df.empty:
                for current_model_name in selected_models:
                    monthly_rows.append(
                        MonthlyBacktestSummary(
                            model_name=current_model_name,
                            month_start=month_start,
                            month_end=month_end,
                            train_rows=len(train_df),
                            scored_games=0,
                            matched_to_market=0,
                            flagged_bets=0,
                            wins=0,
                            losses=0,
                            stake=0.0,
                            payout=0.0,
                        ).__dict__
                    )
                continue

            for current_model_name in selected_models:
                summary = MonthlyBacktestSummary(
                    model_name=current_model_name,
                    month_start=month_start,
                    month_end=month_end,
                    train_rows=len(train_df),
                    scored_games=len(score_df),
                    matched_to_market=0,
                    flagged_bets=0,
                    wins=0,
                    losses=0,
                    stake=0.0,
                    payout=0.0,
                )
                pipeline = _fit_model(train_df, current_model_name)
                model_score_df = score_df.assign(home_model_prob=pipeline.predict_proba(score_df[FEATURE_COLUMNS])[:, 1])

                for row in model_score_df.itertuples(index=False):
                    prior = priors_by_id.get(str(row.game_id))
                    if prior is None:
                        continue
                    summary.matched_to_market += 1

                    home_model = float(row.home_model_prob)
                    away_model = 1.0 - home_model
                    home_market = float(prior["home_fair_prob"])
                    away_market = 1.0 - home_market

                    home_edge = home_model - home_market
                    away_edge = away_model - away_market
                    # Pick the side with the larger positive edge.
                    if home_edge >= away_edge and home_edge >= edge_threshold:
                        picked_team = row.home_team
                        is_home_pick = True
                        model_prob = home_model
                        market_prob = home_market
                        ml_close = prior.get("home_moneyline_close")
                        edge_pct = home_edge
                    elif away_edge > home_edge and away_edge >= edge_threshold:
                        picked_team = row.away_team
                        is_home_pick = False
                        model_prob = away_model
                        market_prob = away_market
                        ml_close = prior.get("away_moneyline_close")
                        edge_pct = away_edge
                    else:
                        continue

                    decimal_odds = _american_to_decimal(ml_close)
                    if decimal_odds is None:
                        continue

                    winner_team = row.home_team if row.target_home_win == 1 else row.away_team
                    won_bet = picked_team == winner_team
                    if won_bet:
                        payout = float(stake) * (decimal_odds - 1.0)
                        summary.wins += 1
                    else:
                        payout = -float(stake)
                        summary.losses += 1
                    summary.flagged_bets += 1
                    summary.stake += float(stake)
                    summary.payout += payout

                    bet_rows.append(
                        {
                            "run_id": run_id,
                            "model_name": current_model_name,
                            "game_id": str(row.game_id),
                            "game_date": row.game_date,
                            "home_team": row.home_team,
                            "away_team": row.away_team,
                            "picked_team": picked_team,
                            "is_home_pick": bool(is_home_pick),
                            "model_prob": float(model_prob),
                            "market_prob": float(market_prob),
                            "edge_pct": float(edge_pct),
                            "decimal_odds": float(decimal_odds),
                            "stake": float(stake),
                            "won_bet": bool(won_bet),
                            "payout": float(payout),
                            "winner_team": winner_team,
                            "train_rows": int(len(train_df)),
                            "train_start_date": train_df["game_date"].min(),
                            "train_end_date": train_df["game_date"].max(),
                            "scored_at": datetime.utcnow(),
                        }
                    )

                monthly_rows.append(summary.__dict__)
                logger.info(
                    "  %s model=%s train=%s scored=%s matched=%s bets=%s wins=%s losses=%s net=%.2f",
                    month_start.isoformat()[:7],
                    current_model_name,
                    summary.train_rows,
                    summary.scored_games,
                    summary.matched_to_market,
                    summary.flagged_bets,
                    summary.wins,
                    summary.losses,
                    summary.payout,
                )

        # A reused run_id must fully replace the previous run, including the
        # case where the new execution emits fewer or zero bets.
        conn.execute("DELETE FROM walkforward_bets WHERE run_id = ?", [run_id])
        if bet_rows:
            bets_df = pd.DataFrame(bet_rows)
            # Persist idempotent per run after clearing any prior rows for this run_id.
            replace_dataframe(conn, "walkforward_bets", bets_df, key_columns=["run_id", "model_name", "game_id", "picked_team"])

        totals_by_model: dict[str, dict[str, float | int]] = {}
        for current_model_name in selected_models:
            model_months = [m for m in monthly_rows if m["model_name"] == current_model_name]
            model_bets = sum(m["flagged_bets"] for m in model_months)
            model_wins = sum(m["wins"] for m in model_months)
            model_losses = sum(m["losses"] for m in model_months)
            model_stake = sum(m["stake"] for m in model_months)
            model_payout = sum(m["payout"] for m in model_months)
            totals_by_model[current_model_name] = {
                "total_bets": int(model_bets),
                "wins": int(model_wins),
                "losses": int(model_losses),
                "hit_rate": float((model_wins / model_bets) if model_bets > 0 else 0.0),
                "stake_total": float(model_stake),
                "payout_total": float(model_payout),
                "roi": float((model_payout / model_stake) if model_stake > 0 else 0.0),
            }

        total_bets = sum(model["total_bets"] for model in totals_by_model.values())
        total_wins = sum(model["wins"] for model in totals_by_model.values())
        total_losses = sum(model["losses"] for model in totals_by_model.values())
        total_stake = sum(model["stake_total"] for model in totals_by_model.values())
        total_payout = sum(model["payout_total"] for model in totals_by_model.values())
        roi = (total_payout / total_stake) if total_stake > 0 else 0.0
        hit_rate = (total_wins / total_bets) if total_bets > 0 else 0.0

        return {
            "status": "ok",
            "run_id": run_id,
            "model_name": model_name,
            "model_names": selected_models,
            "models": totals_by_model,
            "edge_threshold": edge_threshold,
            "warmup_start": warmup_start_date.isoformat(),
            "eval_start": start.isoformat(),
            "eval_end": end.isoformat(),
            "total_bets": int(total_bets),
            "wins": int(total_wins),
            "losses": int(total_losses),
            "hit_rate": float(hit_rate),
            "stake_total": float(total_stake),
            "payout_total": float(total_payout),
            "roi": float(roi),
            "monthly": monthly_rows,
        }
    finally:
        conn.close()
