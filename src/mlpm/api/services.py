from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mlpm.config.settings import settings
from mlpm.evaluation.settled import compute_settled_calibration
from mlpm.evaluation.strategy import select_champion_model, select_champion_with_rationale
from mlpm.storage.duckdb import connect_read_only, query_dataframe


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0 and value.nanosecond == 0:
            return value.date().isoformat()
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return value.isoformat()
    if isinstance(value, datetime):
        if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0:
            return value.date().isoformat()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return float(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        rows.append({key: _normalize_value(value) for key, value in row.items()})
    return rows


def _one(sql: str, params: list[object] | None = None) -> dict[str, Any]:
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(conn, sql, params=params)
    finally:
        conn.close()
    if frame.empty:
        return {}
    return _records(frame)[0]


def _latest_collection_run_expr(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"COALESCE({prefix}collection_run_ts, {prefix}snapshot_ts)"


def _betting_stats_start_date() -> str:
    return getattr(settings(), "betting_stats_start_date", "2025-01-01")


def _latest_training_run_metadata(conn) -> dict[str, Any]:
    frame = query_dataframe(
        conn,
        """
        WITH ranked AS (
            SELECT
                trained_at,
                train_start_date,
                train_end_date,
                rows_train,
                rows_valid,
                ROW_NUMBER() OVER (
                    ORDER BY trained_at DESC, train_end_date DESC, train_start_date ASC, model_name ASC, rank ASC
                ) AS row_num
            FROM feature_importances
        )
        SELECT
            trained_at,
            train_start_date,
            train_end_date,
            rows_train,
            rows_valid
        FROM ranked
        WHERE row_num = 1
        """,
    )
    if frame.empty:
        return {}
    return _records(frame)[0]


def _latest_training_metadata_by_model(conn) -> pd.DataFrame:
    return query_dataframe(
        conn,
        """
        WITH ranked AS (
            SELECT
                model_name,
                trained_at,
                train_start_date,
                train_end_date,
                rows_train,
                rows_valid,
                ROW_NUMBER() OVER (
                    PARTITION BY model_name
                    ORDER BY trained_at DESC, train_end_date DESC, train_start_date ASC, rank ASC, feature ASC
                ) AS row_num
            FROM feature_importances
        )
        SELECT
            model_name,
            trained_at,
            train_start_date,
            train_end_date,
            rows_train,
            rows_valid
        FROM ranked
        WHERE row_num = 1
        """,
    )


def get_summary() -> dict[str, Any]:
    cfg = settings()
    base = _one(
        f"""
        WITH latest_run AS (
            SELECT MAX({_latest_collection_run_expr()}) AS run_ts
            FROM games_deduped
        )
        SELECT
            COUNT(d.game_id) AS total_discrepancies,
            COALESCE(SUM(CASE WHEN d.flagged THEN 1 ELSE 0 END), 0) AS flagged_discrepancies,
            COALESCE(AVG(ABS(d.gap_bps)), 0.0) AS avg_abs_gap_bps,
            (
                SELECT MAX(d2.snapshot_ts)
                FROM discrepancies_deduped d2
                CROSS JOIN latest_run lr
                WHERE {_latest_collection_run_expr("d2")} = lr.run_ts
            ) AS latest_snapshot_ts,
            (
                SELECT MAX(g.game_date)
                FROM games_deduped g
                CROSS JOIN latest_run lr
                WHERE {_latest_collection_run_expr("g")} = lr.run_ts
            ) AS latest_game_date,
            (
                SELECT COUNT(*)
                FROM bet_opportunities_deduped b
                CROSS JOIN latest_run lr
                WHERE {_latest_collection_run_expr("b")} = lr.run_ts
                  AND b.is_actionable = TRUE
            ) AS actionable_bets,
            (
                SELECT COALESCE(MAX(b.edge_bps), 0)
                FROM bet_opportunities_deduped b
                CROSS JOIN latest_run lr
                WHERE {_latest_collection_run_expr("b")} = lr.run_ts
            ) AS max_edge_bps
        FROM discrepancies_deduped d
        CROSS JOIN latest_run lr
        WHERE {_latest_collection_run_expr("d")} = lr.run_ts
        """
    )
    latest_snapshot = base.get("latest_snapshot_ts")
    stale = True
    if latest_snapshot:
        latest_dt = pd.Timestamp(latest_snapshot)
        now = pd.Timestamp.now(tz="UTC")
        stale = (now - latest_dt).total_seconds() > cfg.freshness_window_seconds
    return {
        **base,
        "champion_model": select_champion_model(),
        "stale_data": stale,
        "freshness_window_seconds": cfg.freshness_window_seconds,
    }


def list_opportunities(
    *,
    source: str | None = None,
    model_name: str | None = None,
    actionable_only: bool = False,
    min_edge_bps: int | None = None,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    filters: list[str] = []
    params: list[object] = []
    if source:
        filters.append("source = ?")
        params.append(source)
    if model_name:
        filters.append("model_name = ?")
        params.append(model_name)
    if actionable_only:
        filters.append("is_actionable = TRUE")
    if min_edge_bps is not None:
        filters.append("edge_bps >= ?")
        params.append(min_edge_bps)

    offset = (max(page, 1) - 1) * page_size
    conn = connect_read_only(settings().duckdb_path)
    try:
        total_frame = query_dataframe(
            conn,
            f"""
            WITH latest_run AS (
                SELECT MAX({_latest_collection_run_expr()}) AS run_ts
                FROM games_deduped
            )
            SELECT COUNT(*) AS total
            FROM bet_opportunities_deduped b
            CROSS JOIN latest_run lr
            WHERE {_latest_collection_run_expr("b")} = lr.run_ts
            {" AND " + " AND ".join(filters) if filters else ""}
            """,
            params=params or None,
        )
        rows = query_dataframe(
            conn,
            f"""
            WITH latest_run AS (
                SELECT MAX({_latest_collection_run_expr()}) AS run_ts
                FROM games_deduped
            )
            SELECT
                b.game_id,
                b.game_date,
                b.event_start_time,
                b.model_name,
                b.source,
                b.market_id,
                b.team,
                b.opponent_team,
                b.model_prob,
                b.market_prob,
                b.edge_bps,
                b.expected_value,
                b.is_actionable,
                b.is_champion
            FROM bet_opportunities_deduped b
            CROSS JOIN latest_run lr
            WHERE {_latest_collection_run_expr("b")} = lr.run_ts
            {" AND " + " AND ".join(filters) if filters else ""}
            ORDER BY b.edge_bps DESC, b.event_start_time ASC, b.model_name
            LIMIT {page_size} OFFSET {offset}
            """,
            params=params or None,
        )
    finally:
        conn.close()
    total = int(total_frame.iloc[0]["total"]) if not total_frame.empty else 0
    return {
        "items": _records(rows),
        "total": total,
        "page": page,
        "page_size": page_size,
        "champion_model": select_champion_model(),
    }


def get_game_detail(game_id: str) -> dict[str, Any]:
    conn = connect_read_only(settings().duckdb_path)
    try:
        meta = query_dataframe(
            conn,
            f"""
            WITH latest_run AS (
                SELECT MAX({_latest_collection_run_expr()}) AS run_ts
                FROM games_deduped
            )
            SELECT g.game_id, g.game_date, g.event_start_time, g.away_team, g.home_team
            FROM games_deduped g
            CROSS JOIN latest_run lr
            WHERE g.game_id = ?
              AND {_latest_collection_run_expr("g")} = lr.run_ts
            LIMIT 1
            """,
            params=[game_id],
        )
        quotes = query_dataframe(
            conn,
            f"""
            WITH latest_run AS (
                SELECT MAX({_latest_collection_run_expr()}) AS run_ts
                FROM games_deduped
            )
            SELECT d.source, d.market_id, d.team, d.market_prob, d.model_prob, d.gap_bps, d.snapshot_ts, d.flagged
            FROM discrepancies_deduped d
            CROSS JOIN latest_run lr
            WHERE d.game_id = ?
              AND {_latest_collection_run_expr("d")} = lr.run_ts
            ORDER BY snapshot_ts DESC, source, team
            LIMIT 50
            """,
            params=[game_id],
        )
        features = query_dataframe(
            conn,
            f"""
            WITH latest_run AS (
                SELECT MAX({_latest_collection_run_expr()}) AS run_ts
                FROM games_deduped
            ),
            ranked AS (
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
                    rest_days,
                    starter_era,
                    starter_whip,
                    bullpen_innings_3d,
                    ROW_NUMBER() OVER (
                        PARTITION BY game_id, team, model_name
                        ORDER BY snapshot_ts DESC, collection_run_ts DESC
                    ) AS rn
                FROM model_predictions_deduped mp
                CROSS JOIN latest_run lr
                WHERE game_id = ?
                  AND {_latest_collection_run_expr("mp")} = lr.run_ts
            )
            SELECT * EXCLUDE (rn)
            FROM ranked
            WHERE rn = 1
            ORDER BY team
            """,
            params=[game_id],
        )
        history = query_dataframe(
            conn,
            """
            SELECT snapshot_ts, team, source, gap_bps
            FROM discrepancies_deduped
            WHERE game_id = ?
            ORDER BY snapshot_ts ASC, source, team
            LIMIT 250
            """,
            params=[game_id],
        )
    finally:
        conn.close()
    meta_row = _records(meta)[0] if not meta.empty else {
        "game_id": game_id,
        "game_date": None,
        "event_start_time": None,
        "away_team": None,
        "home_team": None,
    }
    return {
        **meta_row,
        "quotes": _records(quotes),
        "features": _records(features),
        "gap_history": _records(history),
    }


def get_gap_history(*, game_id: str | None = None, team: str | None = None, source: str | None = None, limit: int = 300) -> list[dict[str, Any]]:
    filters: list[str] = []
    params: list[object] = []
    if game_id:
        filters.append("game_id = ?")
        params.append(game_id)
    if team:
        filters.append("team = ?")
        params.append(team)
    if source:
        filters.append("source = ?")
        params.append(source)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT snapshot_ts, team, source, gap_bps
            FROM discrepancies_deduped
            {where_clause}
            ORDER BY snapshot_ts DESC
            LIMIT {limit}
            """,
            params=params or None,
        )
    finally:
        conn.close()
    return _records(frame)


def get_research_contenders() -> list[dict[str, Any]]:
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT
                model_name AS model,
                COUNT(*) AS games,
                AVG(CASE WHEN correct_prediction THEN 1.0 ELSE 0.0 END) AS accuracy,
                AVG(
                    CASE
                        WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                        ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
                    END
                ) AS log_loss
            FROM settled_predictions_deduped
            WHERE game_date >= DATE '{_betting_stats_start_date()}'
            GROUP BY model_name
            ORDER BY games DESC, model_name
            """
        )
    finally:
        conn.close()
    frame["family"] = frame["model"].map(_infer_model_family)
    frame["model_family"] = frame["family"]
    frame["feature_variant"] = frame["model"].map(_infer_feature_variant)
    frame["roc_auc"] = None
    return _records(frame[["model", "family", "model_family", "feature_variant", "accuracy", "roc_auc", "log_loss"]])


def get_research_strategies() -> list[dict[str, Any]]:
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            WITH latest_date AS (
                SELECT MAX(game_date) AS max_game_date
                FROM settled_bet_opportunities_deduped
                WHERE game_date >= DATE '{_betting_stats_start_date()}'
            ),
            sliced AS (
                SELECT
                    model_name,
                    DATE_TRUNC('month', game_date) AS slice_month,
                    COUNT(*) AS slice_bets,
                    SUM(realized_return_units) AS slice_units,
                    SUM(stake_units) AS slice_stakes
                FROM settled_bet_opportunities_deduped
                WHERE game_date >= DATE '{_betting_stats_start_date()}'
                GROUP BY model_name, DATE_TRUNC('month', game_date)
            )
            SELECT
                s.model_name AS strategy_name,
                COUNT(*) AS active_slices,
                SUM(CASE WHEN slice_units > 0 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS positive_slice_rate,
                totals.bets,
                totals.roi,
                totals.units_won,
                0.0 AS roi_ci_lower,
                0.0 AS roi_ci_upper,
                0.0 AS max_drawdown,
                FALSE AS guardrails_passed
            FROM sliced s
            JOIN (
                SELECT
                    model_name,
                    COUNT(*) AS bets,
                    SUM(realized_return_units) / NULLIF(SUM(stake_units), 0) AS roi,
                    SUM(realized_return_units) AS units_won
                FROM settled_bet_opportunities_deduped
                WHERE game_date >= DATE '{_betting_stats_start_date()}'
                GROUP BY model_name
            ) totals
              ON s.model_name = totals.model_name
            GROUP BY s.model_name, totals.bets, totals.roi, totals.units_won
            ORDER BY totals.roi DESC, totals.units_won DESC
            """
        )
    finally:
        conn.close()
    frame["family"] = frame["strategy_name"].map(_infer_model_family)
    return _records(frame)


def get_champion_standings() -> dict[str, Any]:
    cfg = settings()
    decision = select_champion_with_rationale()
    conn = connect_read_only(cfg.duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT
                model_name,
                COUNT(*) AS bets,
                SUM(CASE WHEN won_bet THEN 1 ELSE 0 END) AS wins,
                AVG(CASE WHEN won_bet THEN 1.0 ELSE 0.0 END) AS win_rate,
                SUM(realized_return_units) AS units_won,
                SUM(realized_return_units) / NULLIF(SUM(stake_units), 0) AS roi,
                AVG(edge_bps) AS avg_edge_bps,
                MIN(game_date) AS first_game_date,
                MAX(game_date) AS last_game_date
            FROM settled_bet_opportunities_deduped
            WHERE game_date >= DATE '{_betting_stats_start_date()}'
            GROUP BY model_name
            ORDER BY roi DESC NULLS LAST, units_won DESC NULLS LAST, model_name
            """
        )
    finally:
        conn.close()
    if frame.empty:
        return {
            "betting_stats_start_date": _betting_stats_start_date(),
            "champion_model": decision.chosen_model,
            "decision_reason": decision.reason,
            "decision_action": decision.action,
            "rows": [],
        }

    frame["family"] = frame["model_name"].map(_infer_model_family)
    frame["is_champion"] = frame["model_name"].eq(decision.chosen_model)
    frame["ci_lower"] = frame["model_name"].map(
        lambda name: decision.challenger_ci_lower if name == decision.challenger_model else None
    )
    frame["ci_upper"] = frame["model_name"].map(
        lambda name: decision.challenger_ci_upper if name == decision.challenger_model else None
    )
    frame["incumbent_point_metric"] = frame["model_name"].map(
        lambda name: decision.incumbent_win_rate if name == decision.incumbent_model else None
    )
    return {
        "betting_stats_start_date": _betting_stats_start_date(),
        "champion_model": decision.chosen_model,
        "decision_reason": decision.reason,
        "decision_action": decision.action,
        "rows": _records(frame),
    }


def get_calibration(*, bins: int = 10) -> list[dict[str, Any]]:
    frame = compute_settled_calibration(bins=bins)
    return _records(frame)


def get_feature_importance(*, limit: int = 50) -> list[dict[str, Any]]:
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT
                model_name,
                feature,
                importance,
                importance_std,
                rank,
                trained_at,
                train_start_date,
                train_end_date,
                rows_train,
                rows_valid,
                method
            FROM latest_feature_importances
            ORDER BY model_name, rank
            LIMIT {limit}
            """
        )
    finally:
        conn.close()
    return _records(frame)


def get_freshness() -> dict[str, Any]:
    cfg = settings()
    row = _one(
        """
        SELECT
            (SELECT MAX(snapshot_ts) FROM discrepancies_deduped) AS latest_snapshot_ts,
            (SELECT MAX(game_date) FROM games_deduped) AS latest_game_date,
            (SELECT MAX(game_date) FROM game_weather_deduped) AS latest_weather_game_date,
            (SELECT MAX(completed_at) FROM historical_import_runs) AS latest_historical_import_completed_at,
            (SELECT MAX(started_at) FROM collector_runs) AS latest_results_sync
        """
    )
    latest_snapshot = row.get("latest_snapshot_ts")
    stale = True
    if latest_snapshot:
        latest_dt = pd.Timestamp(latest_snapshot)
        stale = (pd.Timestamp.now(tz="UTC") - latest_dt).total_seconds() > cfg.freshness_window_seconds
    return {**row, "stale_data": stale}


def get_import_status() -> list[dict[str, Any]]:
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            """
            SELECT *
            FROM historical_import_status
            ORDER BY last_completed_at DESC, source
            """
        )
    finally:
        conn.close()
    return _records(frame)


def get_training_coverage() -> dict[str, Any]:
    """Return the data sources that feed the training set plus the latest-run metadata."""
    conn = connect_read_only(settings().duckdb_path)
    try:
        # SBRO (2015-2021 closing-line book odds)
        sbro = query_dataframe(
            conn,
            """
            SELECT
                'sbro' AS source,
                'SBRO closing line (2015-2021)' AS label,
                MIN(game_date) AS first_date,
                MAX(game_date) AS last_date,
                COUNT(DISTINCT game_id) AS games_with_prior
            FROM historical_market_priors_deduped
            WHERE home_fair_prob IS NOT NULL
            """,
        )
        kalshi = query_dataframe(
            conn,
            """
            SELECT
                'kalshi_replay' AS source,
                'Kalshi historical replay (2025+)' AS label,
                MIN(CAST(event_start_time AS DATE)) AS first_date,
                MAX(CAST(event_start_time AS DATE)) AS last_date,
                COUNT(DISTINCT game_id) AS games_with_prior
            FROM historical_kalshi_quotes
            WHERE home_implied_prob IS NOT NULL
            """,
        )
        live = query_dataframe(
            conn,
            """
            SELECT
                'live_quotes' AS source,
                'Live Kalshi + Polymarket (current season)' AS label,
                MIN(CAST(event_start_time AS DATE)) AS first_date,
                MAX(CAST(event_start_time AS DATE)) AS last_date,
                COUNT(DISTINCT game_id) AS games_with_prior
            FROM normalized_quotes_deduped
            WHERE fair_prob IS NOT NULL
            """,
        )
        latest_model = _latest_training_run_metadata(conn)
    finally:
        conn.close()

    rows = pd.concat([sbro, kalshi, live], ignore_index=True)
    conn = connect_read_only(settings().duckdb_path)
    try:
        total_frame = query_dataframe(
            conn,
            """
            WITH all_game_ids AS (
                SELECT game_id
                FROM historical_market_priors_deduped
                WHERE home_fair_prob IS NOT NULL
                UNION
                SELECT game_id
                FROM historical_kalshi_quotes
                WHERE home_implied_prob IS NOT NULL
                UNION
                SELECT game_id
                FROM normalized_quotes_deduped
                WHERE fair_prob IS NOT NULL
            )
            SELECT COUNT(*) AS total_games_with_prior
            FROM all_game_ids
            """,
        )
    finally:
        conn.close()
    total = int(total_frame.iloc[0]["total_games_with_prior"]) if not total_frame.empty else 0
    meta = latest_model or {}
    return {
        "rows": _records(rows),
        "train_start_date": settings().model_train_start_date,
        "train_end_date": None,
        "total_games_with_prior": total,
        "latest_model_train_start": _normalize_value(meta.get("train_start_date")),
        "latest_model_train_end": _normalize_value(meta.get("train_end_date")),
        "latest_model_trained_at": _normalize_value(meta.get("trained_at")),
    }


def _bet_sizing_policies() -> list[tuple[str, str]]:
    return [
        ("flat_1u", "Flat 1u"),
        ("edge_scaled_cap_1u", "Edge-scaled (cap 1u)"),
        ("fractional_kelly_25_cap_1u", "25% Kelly (cap 1u)"),
    ]


def _apply_sizing(frame: pd.DataFrame, policy: str) -> pd.Series:
    """Return a Series of stake_units for each row under the given policy."""
    if policy == "flat_1u":
        return pd.Series(1.0, index=frame.index)
    if policy == "edge_scaled_cap_1u":
        # Map edge_bps → fraction of unit, capped at 1u.
        # 100 bps  ->  0.10u ;  500 bps -> 0.50u ;  >=1000 bps -> 1u
        scaled = frame["edge_bps"].astype(float).clip(lower=0) / 1000.0
        return scaled.clip(lower=0.0, upper=1.0)
    if policy == "fractional_kelly_25_cap_1u":
        prob = frame["model_prob"].astype(float).clip(lower=1e-6, upper=1 - 1e-6)
        dec_odds = frame["implied_decimal_odds"].astype(float).clip(lower=1.0001)
        b = dec_odds - 1.0
        kelly = (prob * dec_odds - 1.0) / b
        kelly = kelly.clip(lower=0.0)
        return (0.25 * kelly).clip(upper=1.0)
    raise ValueError(f"Unknown policy: {policy}")


def get_sizing_comparison() -> list[dict[str, Any]]:
    """For each model, compute ROI under each of the 3 bet-sizing policies and
    flag which policy has produced the best ROI historically."""
    conn = connect_read_only(settings().duckdb_path)
    try:
        bets = query_dataframe(
            conn,
            f"""
            SELECT
                model_name,
                edge_bps,
                model_prob,
                implied_decimal_odds,
                won_bet
            FROM settled_bet_opportunities_deduped
            WHERE implied_decimal_odds IS NOT NULL
              AND model_prob IS NOT NULL
              AND won_bet IS NOT NULL
              AND game_date >= DATE '{_betting_stats_start_date()}'
            """,
        )
    finally:
        conn.close()

    champion = select_champion_model()
    policies = _bet_sizing_policies()
    results: list[dict[str, Any]] = []

    if bets.empty:
        return results

    for model_name, frame in bets.groupby("model_name", sort=False):
        frame = frame.reset_index(drop=True)
        decimal = frame["implied_decimal_odds"].astype(float).clip(lower=1.0001)
        won = frame["won_bet"].astype(bool)

        policy_rows: list[dict[str, Any]] = []
        for key, label in policies:
            stake = _apply_sizing(frame, key)
            # Realized profit per bet: win -> stake*(dec_odds-1); lose -> -stake
            profit = stake * ((decimal - 1.0).where(won, -1.0))
            active = stake > 0
            total_stake = float(stake[active].sum())
            units_won = float(profit[active].sum())
            roi = units_won / total_stake if total_stake > 0 else 0.0
            policy_rows.append(
                {
                    "policy": key,
                    "label": label,
                    "bets": int(active.sum()),
                    "total_stake": total_stake,
                    "units_won": units_won,
                    "roi": roi,
                    "is_best": False,
                }
            )

        # Flag best-ROI policy (only among policies with at least one bet).
        eligible = [p for p in policy_rows if p["bets"] > 0]
        best_policy = None
        best_roi = None
        if eligible:
            winner = max(eligible, key=lambda p: p["roi"])
            winner["is_best"] = True
            best_policy = winner["policy"]
            best_roi = winner["roi"]

        family = _infer_model_family(model_name)
        role = "champion" if model_name == champion else "challenger"
        results.append(
            {
                "model_name": model_name,
                "family": family,
                "role": role,
                "policies": policy_rows,
                "best_policy": best_policy,
                "best_roi": best_roi,
            }
        )

    # Add a synthetic "ensemble" row — mean of per-game model probabilities.
    ensemble = _compute_ensemble_sizing()
    if ensemble is not None:
        results.append(ensemble)

    # Order: champion first, then by best_roi desc, ensemble last.
    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        role_rank = {"champion": 0, "challenger": 1, "ensemble": 2}.get(row["role"], 3)
        return (role_rank, -(row["best_roi"] or -999.0))

    results.sort(key=sort_key)
    return results


def _compute_ensemble_sizing() -> dict[str, Any] | None:
    """Compute bet-sizing comparison for a naive ensemble (mean of model
    probabilities per game). Ensemble probabilities are blended from
    settled_bet_opportunities_deduped per (game_id, team)."""
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT
                game_id,
                team,
                AVG(model_prob)            AS model_prob,
                AVG(implied_decimal_odds)  AS implied_decimal_odds,
                AVG(market_prob)           AS market_prob,
                MAX(won_bet::INT)          AS won_flag
            FROM settled_bet_opportunities_deduped
            WHERE implied_decimal_odds IS NOT NULL
              AND model_prob IS NOT NULL
              AND won_bet IS NOT NULL
              AND game_date >= DATE '{_betting_stats_start_date()}'
            GROUP BY game_id, team
            HAVING COUNT(*) >= 2
            """,
        )
    finally:
        conn.close()
    if frame.empty:
        return None

    frame = frame.copy()
    frame["won_bet"] = frame["won_flag"].astype(bool)
    frame["edge_bps"] = ((frame["model_prob"] - frame["market_prob"]) * 10000).round().astype(int)

    decimal = frame["implied_decimal_odds"].astype(float).clip(lower=1.0001)
    won = frame["won_bet"]

    policies = _bet_sizing_policies()
    policy_rows: list[dict[str, Any]] = []
    for key, label in policies:
        stake = _apply_sizing(frame, key)
        profit = stake * ((decimal - 1.0).where(won, -1.0))
        active = stake > 0
        total_stake = float(stake[active].sum())
        units_won = float(profit[active].sum())
        roi = units_won / total_stake if total_stake > 0 else 0.0
        policy_rows.append(
            {
                "policy": key,
                "label": label,
                "bets": int(active.sum()),
                "total_stake": total_stake,
                "units_won": units_won,
                "roi": roi,
                "is_best": False,
            }
        )

    eligible = [p for p in policy_rows if p["bets"] > 0]
    best_policy = None
    best_roi = None
    if eligible:
        winner = max(eligible, key=lambda p: p["roi"])
        winner["is_best"] = True
        best_policy = winner["policy"]
        best_roi = winner["roi"]

    return {
        "model_name": "ensemble_mean",
        "family": "ensemble",
        "role": "ensemble",
        "policies": policy_rows,
        "best_policy": best_policy,
        "best_roi": best_roi,
    }


def get_model_roster() -> list[dict[str, Any]]:
    """Return roster of every model with role (champion / challenger / ensemble)
    plus headline metrics (accuracy, log_loss, roi)."""
    champion = select_champion_model()
    conn = connect_read_only(settings().duckdb_path)
    try:
        contenders = query_dataframe(
            conn,
            f"""
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
            WHERE game_date >= DATE '{_betting_stats_start_date()}'
            GROUP BY model_name
            """,
        )
        strategies = query_dataframe(
            conn,
            f"""
            SELECT
                model_name,
                COUNT(*) AS bets,
                SUM(realized_return_units) / NULLIF(SUM(stake_units), 0) AS roi,
                SUM(realized_return_units) AS units_won
            FROM settled_bet_opportunities_deduped
            WHERE game_date >= DATE '{_betting_stats_start_date()}'
            GROUP BY model_name
            """,
        )
        training = _latest_training_metadata_by_model(conn)
        latest_training = _latest_training_run_metadata(conn)
    finally:
        conn.close()

    def merge(name: str) -> dict[str, Any]:
        contender_row = contenders[contenders["model_name"] == name]
        strategy_row = strategies[strategies["model_name"] == name]
        training_row = training[training["model_name"] == name]
        training_values = (
            {
                "trained_at": _normalize_value(training_row["trained_at"].iloc[0]),
                "train_start_date": _normalize_value(training_row["train_start_date"].iloc[0]),
                "train_end_date": _normalize_value(training_row["train_end_date"].iloc[0]),
                "rows_train": int(training_row["rows_train"].iloc[0]) if pd.notna(training_row["rows_train"].iloc[0]) else None,
                "rows_valid": int(training_row["rows_valid"].iloc[0]) if pd.notna(training_row["rows_valid"].iloc[0]) else None,
            }
            if not training_row.empty
            else {
                "trained_at": latest_training.get("trained_at"),
                "train_start_date": latest_training.get("train_start_date"),
                "train_end_date": latest_training.get("train_end_date"),
                "rows_train": latest_training.get("rows_train"),
                "rows_valid": latest_training.get("rows_valid"),
            }
        )
        return {
            "model_name": name,
            "family": _infer_model_family(name),
            "feature_variant": _infer_feature_variant(name),
            "role": "champion" if name == champion else "challenger",
            "trained_at": training_values["trained_at"],
            "train_start_date": training_values["train_start_date"],
            "train_end_date": training_values["train_end_date"],
            "rows_train": training_values["rows_train"],
            "rows_valid": training_values["rows_valid"],
            "settled_bets": int(strategy_row["bets"].iloc[0]) if not strategy_row.empty and pd.notna(strategy_row["bets"].iloc[0]) else 0,
            "accuracy": float(contender_row["accuracy"].iloc[0]) if not contender_row.empty and pd.notna(contender_row["accuracy"].iloc[0]) else None,
            "log_loss": float(contender_row["log_loss"].iloc[0]) if not contender_row.empty and pd.notna(contender_row["log_loss"].iloc[0]) else None,
            "roi": float(strategy_row["roi"].iloc[0]) if not strategy_row.empty and pd.notna(strategy_row["roi"].iloc[0]) else None,
            "units_won": float(strategy_row["units_won"].iloc[0]) if not strategy_row.empty and pd.notna(strategy_row["units_won"].iloc[0]) else None,
        }

    # Union every model that shows up anywhere in the pipeline.
    names: set[str] = set()
    for frame in (contenders, strategies, training):
        if not frame.empty:
            names.update(frame["model_name"].tolist())

    roster = [merge(n) for n in sorted(names)]
    # Add ensemble row.
    roster.append(
        {
            "model_name": "ensemble_mean",
            "family": "ensemble",
            "feature_variant": "blend",
            "role": "ensemble",
            "trained_at": None,
            "train_start_date": None,
            "train_end_date": None,
            "rows_train": None,
            "rows_valid": None,
            "settled_bets": 0,
            "accuracy": None,
            "log_loss": None,
            "roi": None,
            "units_won": None,
        }
    )

    # Champion first, then challengers by settled_bets desc, ensemble last.
    def sort_key(row: dict[str, Any]) -> tuple[int, int]:
        rank = {"champion": 0, "challenger": 1, "ensemble": 2}.get(row["role"], 3)
        return (rank, -(row["settled_bets"] or 0))

    roster.sort(key=sort_key)
    return roster


def _infer_model_family(name: str) -> str:
    lowered = name.lower()
    if "histgb" in lowered:
        return "histgb"
    if "logreg" in lowered or "logit" in lowered:
        return "logreg"
    if "knn" in lowered:
        return "knn"
    if "svm" in lowered:
        return "svm"
    if "mlp" in lowered:
        return "mlp"
    if "bayes" in lowered:
        return "bayes"
    return "model"


def _infer_feature_variant(name: str) -> str:
    lowered = name.lower()
    if "market" in lowered:
        return "market_aware"
    if "baseball" in lowered or "mlb" in lowered:
        return "baseball_only"
    return "derived"
