from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from mlpm.config.settings import settings
from mlpm.storage.duckdb import append_dataframe, connect, connect_read_only, query_dataframe


# ---------------------------------------------------------------------------
# Confidence-interval helpers
# ---------------------------------------------------------------------------

def _z_score(confidence: float) -> float:
    # Two-sided normal approximation. 0.95 -> ~1.96.
    confidence = min(max(confidence, 1e-6), 1 - 1e-6)
    alpha = 1.0 - confidence
    # Inverse of Phi(1 - alpha/2) via rational approximation (Beasley-Springer-Moro-ish light form).
    # math.erfinv exists in Python 3.11+; fall back if not.
    p = 1.0 - alpha / 2.0
    try:
        return math.sqrt(2.0) * math.erfinv(2.0 * p - 1.0)  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - older Python
        # Acklam's approximation via numpy
        from scipy.stats import norm  # lazy, optional

        return float(norm.ppf(p))


def wilson_win_rate_ci(wins: int, trials: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (point_estimate, lower, upper) Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (0.0, 0.0, 1.0)
    p_hat = wins / trials
    z = _z_score(confidence)
    denom = 1.0 + (z * z) / trials
    centre = (p_hat + (z * z) / (2.0 * trials)) / denom
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * trials)) / trials)) / denom
    return (p_hat, max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_roi_ci(
    realized_returns: np.ndarray,
    stakes: np.ndarray,
    confidence: float = 0.95,
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower, upper) percentile-bootstrap CI on ROI = sum(return)/sum(stake)."""
    n = len(realized_returns)
    if n == 0 or stakes.sum() == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    point = float(realized_returns.sum() / stakes.sum())
    indices = rng.integers(0, n, size=(n_samples, n))
    sampled_returns = realized_returns[indices].sum(axis=1)
    sampled_stakes = stakes[indices].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sampled_roi = np.where(sampled_stakes > 0, sampled_returns / sampled_stakes, 0.0)
    alpha = 1.0 - confidence
    lower = float(np.quantile(sampled_roi, alpha / 2.0))
    upper = float(np.quantile(sampled_roi, 1.0 - alpha / 2.0))
    return (point, lower, upper)


# ---------------------------------------------------------------------------
# Champion selection with guardrails
# ---------------------------------------------------------------------------

@dataclass
class ChampionDecision:
    chosen_model: str | None
    incumbent_model: str | None
    challenger_model: str | None
    action: str  # "kept_incumbent" | "switched" | "cold_start" | "no_data" | "fallback_logloss"
    reason: str
    challenger_bets: int = 0
    challenger_roi: float = 0.0
    challenger_units: float = 0.0
    challenger_win_rate: float = 0.0
    challenger_ci_lower: float = 0.0
    challenger_ci_upper: float = 0.0
    incumbent_bets: int = 0
    incumbent_roi: float = 0.0
    incumbent_win_rate: float = 0.0
    confidence: float = 0.95
    window_days: int = 30
    min_bets: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)


def _current_incumbent(conn) -> str | None:
    if not _table_has_rows(conn, "champion_history"):
        return None
    row = conn.execute(
        "SELECT chosen_model FROM champion_history ORDER BY selected_at DESC LIMIT 1"
    ).fetchone()
    return None if row is None else (str(row[0]) if row[0] is not None else None)


def _window_performance(conn, window_days: int, min_bets: int) -> pd.DataFrame:
    """Return per-model performance frame with CI bounds over the evaluation window."""
    cfg = settings()
    frame = query_dataframe(
        conn,
        f"""
        WITH latest_date AS (
            SELECT COALESCE(MAX(game_date), CURRENT_DATE) AS max_game_date
            FROM settled_bet_opportunities_deduped
        )
        SELECT
            model_name,
            CAST(won_bet AS INTEGER) AS won_bet,
            realized_return_units,
            stake_units
        FROM settled_bet_opportunities_deduped
        CROSS JOIN latest_date
        WHERE game_date >= max_game_date - {window_days - 1}
        """,
    )
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for model_name, group in frame.groupby("model_name"):
        bets = int(len(group))
        wins = int(group["won_bet"].sum())
        stakes = group["stake_units"].to_numpy(dtype=float)
        returns = group["realized_return_units"].to_numpy(dtype=float)
        roi = float(returns.sum() / stakes.sum()) if stakes.sum() else 0.0
        units_won = float(returns.sum())
        win_rate, wr_lower, wr_upper = wilson_win_rate_ci(wins, bets, confidence=cfg.strategy_champion_ci_confidence)
        if cfg.strategy_champion_ci_method == "bootstrap_roi":
            _, roi_lower, roi_upper = bootstrap_roi_ci(
                returns,
                stakes,
                confidence=cfg.strategy_champion_ci_confidence,
                n_samples=cfg.strategy_champion_bootstrap_samples,
                seed=cfg.strategy_champion_bootstrap_seed,
            )
            ci_lower, ci_upper = roi_lower, roi_upper
            primary_metric = roi
        else:
            # wilson on win rate — treat win rate as the stable guardrail metric
            ci_lower, ci_upper = wr_lower, wr_upper
            primary_metric = win_rate

        rows.append(
            {
                "model_name": model_name,
                "bets": bets,
                "wins": wins,
                "roi": roi,
                "units_won": units_won,
                "win_rate": win_rate,
                "primary_metric": primary_metric,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "meets_min_bets": bets >= min_bets,
            }
        )
    return pd.DataFrame(rows)


def _fallback_logloss_champion(conn, window_days: int, min_bets: int) -> str | None:
    """Pick the model with the lowest log-loss over settled predictions as a cold-start fallback."""
    if not _table_has_rows(conn, "settled_predictions_deduped"):
        return None
    frame = query_dataframe(
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
        WHERE game_date >= max_game_date - {window_days - 1}
        GROUP BY model_name
        HAVING COUNT(*) >= {min_bets}
        ORDER BY log_loss ASC, games DESC, model_name
        LIMIT 1
        """,
    )
    if frame.empty:
        frame = query_dataframe(
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
            """,
        )
    return None if frame.empty else str(frame.iloc[0]["model_name"])


def select_champion_with_rationale(reference_date: str | None = None) -> ChampionDecision:
    """
    Return a full ChampionDecision explaining which model should be champion and why.

    Applies guardrail logic: stick with the incumbent unless the best challenger's
    lower CI bound beats the incumbent's point estimate AND meets `min_bets`.
    """
    cfg = settings()
    decision = ChampionDecision(
        chosen_model=None,
        incumbent_model=None,
        challenger_model=None,
        action="no_data",
        reason="no settled bet opportunities and no settled predictions available",
        confidence=cfg.strategy_champion_ci_confidence,
        window_days=cfg.strategy_champion_window_days,
        min_bets=cfg.strategy_champion_min_bets,
    )

    conn = connect_read_only(cfg.duckdb_path)
    try:
        incumbent = _current_incumbent(conn)
        decision.incumbent_model = incumbent

        if not _table_has_rows(conn, "settled_bet_opportunities_deduped"):
            fallback = _fallback_logloss_champion(conn, cfg.strategy_champion_window_days, cfg.strategy_champion_min_bets)
            if fallback is None:
                return decision
            decision.chosen_model = fallback
            decision.challenger_model = fallback
            decision.action = "fallback_logloss"
            decision.reason = "no settled bets yet; picking lowest log-loss model over the window"
            return decision

        performance = _window_performance(conn, cfg.strategy_champion_window_days, cfg.strategy_champion_min_bets)
        if performance.empty:
            # widen to all-time
            performance = _window_performance(conn, window_days=10_000, min_bets=1)
            if performance.empty:
                fallback = _fallback_logloss_champion(conn, 10_000, 1)
                decision.chosen_model = fallback
                decision.challenger_model = fallback
                decision.action = "fallback_logloss" if fallback else "no_data"
                decision.reason = "no bets in window; fell back to log-loss" if fallback else decision.reason
                return decision

        eligible = performance[performance["meets_min_bets"]].sort_values(
            ["primary_metric", "ci_lower", "bets"], ascending=[False, False, False]
        )
        if eligible.empty:
            eligible = performance.sort_values(["primary_metric", "bets"], ascending=[False, False])

        challenger_row = eligible.iloc[0]
        challenger = str(challenger_row["model_name"])
        decision.challenger_model = challenger
        decision.challenger_bets = int(challenger_row["bets"])
        decision.challenger_roi = float(challenger_row["roi"])
        decision.challenger_units = float(challenger_row["units_won"])
        decision.challenger_win_rate = float(challenger_row["win_rate"])
        decision.challenger_ci_lower = float(challenger_row["ci_lower"])
        decision.challenger_ci_upper = float(challenger_row["ci_upper"])

        if incumbent is None:
            decision.chosen_model = challenger
            decision.action = "cold_start"
            decision.reason = f"no prior champion; seeding with best {cfg.strategy_champion_ci_method} metric"
            return decision

        incumbent_row = performance[performance["model_name"] == incumbent]
        if not incumbent_row.empty:
            inc = incumbent_row.iloc[0]
            decision.incumbent_bets = int(inc["bets"])
            decision.incumbent_roi = float(inc["roi"])
            decision.incumbent_win_rate = float(inc["win_rate"])
            incumbent_point = float(inc["primary_metric"])
        else:
            incumbent_point = float("-inf")  # incumbent has no data in window

        if challenger == incumbent:
            decision.chosen_model = incumbent
            decision.action = "kept_incumbent"
            decision.reason = "incumbent is already the top performer in the window"
            return decision

        if not bool(challenger_row["meets_min_bets"]):
            decision.chosen_model = incumbent
            decision.action = "kept_incumbent"
            decision.reason = (
                f"challenger {challenger} has only {int(challenger_row['bets'])} bets "
                f"(< min_bets={cfg.strategy_champion_min_bets})"
            )
            return decision

        if decision.challenger_ci_lower > incumbent_point:
            decision.chosen_model = challenger
            decision.action = "switched"
            decision.reason = (
                f"challenger lower CI {decision.challenger_ci_lower:.4f} > "
                f"incumbent point {incumbent_point:.4f} "
                f"(metric={cfg.strategy_champion_ci_method})"
            )
        else:
            decision.chosen_model = incumbent
            decision.action = "kept_incumbent"
            decision.reason = (
                f"challenger lower CI {decision.challenger_ci_lower:.4f} does not clear "
                f"incumbent point {incumbent_point:.4f}; staying with incumbent"
            )
        return decision
    finally:
        conn.close()


def record_champion_selection(
    decision: ChampionDecision,
    reference_date: str | None = None,
    conn: Any | None = None,
) -> None:
    """Append a champion_history row. Uses the provided connection if given, else opens one."""
    row = {
        "selected_at": datetime.now(),
        "reference_date": reference_date,
        "chosen_model": decision.chosen_model,
        "incumbent_model": decision.incumbent_model,
        "challenger_model": decision.challenger_model,
        "action": decision.action,
        "reason": decision.reason,
        "window_days": decision.window_days,
        "min_bets": decision.min_bets,
        "confidence": decision.confidence,
        "challenger_bets": decision.challenger_bets,
        "challenger_roi": decision.challenger_roi,
        "challenger_units": decision.challenger_units,
        "challenger_win_rate": decision.challenger_win_rate,
        "challenger_ci_lower": decision.challenger_ci_lower,
        "challenger_ci_upper": decision.challenger_ci_upper,
        "incumbent_bets": decision.incumbent_bets,
        "incumbent_roi": decision.incumbent_roi,
        "incumbent_win_rate": decision.incumbent_win_rate,
    }
    frame = pd.DataFrame([row])
    if conn is not None:
        append_dataframe(conn, "champion_history", frame)
        return
    cfg = settings()
    owned_conn = connect(cfg.duckdb_path)
    try:
        append_dataframe(owned_conn, "champion_history", frame)
    finally:
        owned_conn.close()


def select_champion_model(reference_date: str | None = None) -> str | None:
    """
    Backwards-compatible entry point: returns the chosen champion model name.

    Applies CI guardrails — an incumbent is only replaced when the challenger's
    lower confidence bound beats the incumbent's point estimate.
    """
    return select_champion_with_rationale(reference_date=reference_date).chosen_model


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
    filters = ["game_date BETWEEN ? AND ?"]
    params: list[object] = [start_date, end_date]
    if model_name:
        filters.append("model_name = ?")
        params.append(model_name)
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT *
            FROM bet_opportunities_deduped
            WHERE {' AND '.join(filters)}
            ORDER BY game_date DESC, event_start_time DESC, model_name
            """,
            params=params,
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
    filters = ["game_date BETWEEN ? AND ?"]
    params: list[object] = [start_date, end_date]
    if model_name:
        filters.append("model_name = ?")
        params.append(model_name)
    conn = connect_read_only(settings().duckdb_path)
    try:
        frame = query_dataframe(
            conn,
            f"""
            SELECT *
            FROM settled_bet_opportunities_deduped
            WHERE {' AND '.join(filters)}
            ORDER BY game_date, event_start_time, snapshot_ts
            """,
            params=params,
        )
        daily = query_dataframe(
            conn,
            f"""
            SELECT *
            FROM strategy_performance_daily
            WHERE {' AND '.join(filters)}
            ORDER BY game_date DESC, model_name, source
            """,
            params=params,
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
