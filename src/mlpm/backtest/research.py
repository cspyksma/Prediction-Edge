from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mlpm.config.settings import settings
from mlpm.evaluation.strategy import build_bet_opportunities
from mlpm.historical.replay import build_kalshi_replay_quote_rows, load_kalshi_pregame_replay
from mlpm.models.game_outcome import (
    BAYES_EVIDENCE_FEATURE_COLUMNS,
    BAYES_MODEL_NAME,
    FEATURE_COLUMNS,
    _calibration_table,
    _classification_metrics,
    _load_game_weather,
    _require_feature_columns,
    _train_bayesian_bundle,
    _bayesian_posterior_probabilities,
    build_training_dataset,
)
from mlpm.storage.duckdb import connect, connect_read_only, query_dataframe


MARKET_FEATURE_COLUMN = "market_home_implied_prob"
BASEBALL_FEATURE_COLUMNS = list(FEATURE_COLUMNS)
MARKET_AWARE_FEATURE_COLUMNS = [*FEATURE_COLUMNS, MARKET_FEATURE_COLUMN]
MARKET_ONLY_FEATURE_COLUMNS = [MARKET_FEATURE_COLUMN]

DEFAULT_EDGE_THRESHOLDS_BPS = (0, 100, 300)
DEFAULT_MIN_BETS = 20
DEFAULT_MIN_ACTIVE_SLICES = 3
DEFAULT_MAX_SLICE_BET_SHARE = 0.60


@dataclass(frozen=True)
class ContenderSpec:
    name: str
    family: str
    model_family: str
    feature_variant: str
    feature_columns: tuple[str, ...]
    builder_kind: str


@dataclass(frozen=True)
class StrategySpec:
    name: str
    edge_threshold_bps: int
    sizing_policy: str


def _build_pipeline(model_kind: str, feature_columns: list[str]) -> Pipeline:
    if model_kind == "logreg":
        return Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "numeric",
                                Pipeline(
                                    steps=[
                                        ("impute", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler()),
                                    ]
                                ),
                                feature_columns,
                            )
                        ]
                    ),
                ),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )
    if model_kind == "histgb":
        return Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "numeric",
                                Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]),
                                feature_columns,
                            )
                        ]
                    ),
                ),
                ("classifier", GradientBoostingClassifier(learning_rate=0.05, n_estimators=250, max_depth=3, random_state=42)),
            ]
        )
    if model_kind == "knn":
        return Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "numeric",
                                Pipeline(
                                    steps=[
                                        ("impute", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler()),
                                    ]
                                ),
                                feature_columns,
                            )
                        ]
                    ),
                ),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        )
    if model_kind == "svm":
        return Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "numeric",
                                Pipeline(
                                    steps=[
                                        ("impute", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler()),
                                    ]
                                ),
                                feature_columns,
                            )
                        ]
                    ),
                ),
                ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
            ]
        )
    raise ValueError(f"Unsupported model_kind={model_kind}")


def _build_research_contenders() -> list[ContenderSpec]:
    specs: list[ContenderSpec] = []
    for model_family in ("logreg", "histgb", "knn", "svm"):
        specs.append(
            ContenderSpec(
                name=f"baseball_{model_family}_v1",
                family="baseball_only",
                model_family=model_family,
                feature_variant="baseball_only",
                feature_columns=tuple(BASEBALL_FEATURE_COLUMNS),
                builder_kind=model_family,
            )
        )
        specs.append(
            ContenderSpec(
                name=f"hybrid_{model_family}_market_v1",
                family="hybrid_market_aware",
                model_family=model_family,
                feature_variant="baseball_plus_market",
                feature_columns=tuple(MARKET_AWARE_FEATURE_COLUMNS),
                builder_kind=model_family,
            )
        )
        specs.append(
            ContenderSpec(
                name=f"market_only_{model_family}_v1",
                family="market_only",
                model_family=model_family,
                feature_variant="market_only",
                feature_columns=tuple(MARKET_ONLY_FEATURE_COLUMNS),
                builder_kind=model_family,
            )
        )
    specs.append(
        ContenderSpec(
            name="market_identity_v1",
            family="market_only",
            model_family="identity",
            feature_variant="market_only",
            feature_columns=tuple(MARKET_ONLY_FEATURE_COLUMNS),
            builder_kind="identity",
        )
    )
    specs.append(
        ContenderSpec(
            name=BAYES_MODEL_NAME,
            family="hybrid_market_aware",
            model_family="bayes",
            feature_variant="bayes_posterior",
            feature_columns=tuple(BAYES_EVIDENCE_FEATURE_COLUMNS),
            builder_kind="bayes",
        )
    )
    return specs


def _build_strategy_specs() -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    for edge_threshold_bps in DEFAULT_EDGE_THRESHOLDS_BPS:
        specs.extend(
            [
                StrategySpec(
                    name=f"edge_{edge_threshold_bps}bps_flat_1u",
                    edge_threshold_bps=edge_threshold_bps,
                    sizing_policy="flat_1u",
                ),
                StrategySpec(
                    name=f"edge_{edge_threshold_bps}bps_edge_scaled",
                    edge_threshold_bps=edge_threshold_bps,
                    sizing_policy="edge_scaled_cap_1u",
                ),
                StrategySpec(
                    name=f"edge_{edge_threshold_bps}bps_kelly_25",
                    edge_threshold_bps=edge_threshold_bps,
                    sizing_policy="fractional_kelly_25_cap_1u",
                ),
            ]
        )
    return specs


def _load_sbro_market_priors(start_date: str, end_date: str) -> pd.DataFrame:
    db_path = settings().duckdb_path
    if not db_path.exists():
        return pd.DataFrame(columns=["game_id", MARKET_FEATURE_COLUMN])

    conn = connect(db_path)
    try:
        frame = query_dataframe(
            conn,
            """
            SELECT
                game_id,
                home_fair_prob AS market_home_implied_prob
            FROM historical_market_priors_deduped
            WHERE source = 'sportsbookreviewsonline'
              AND game_date BETWEEN ? AND ?
              AND home_fair_prob IS NOT NULL
            """,
            (start_date, end_date),
        )
    finally:
        conn.close()
    return frame if frame is not None and not frame.empty else pd.DataFrame(columns=["game_id", MARKET_FEATURE_COLUMN])


def _load_local_results_frame(start_date: str, end_date: str) -> pd.DataFrame:
    conn = connect_read_only(settings().duckdb_path)
    try:
        return query_dataframe(
            conn,
            """
            SELECT
                game_id,
                CAST(game_date AS DATE) AS game_date,
                event_start_time,
                home_team,
                away_team,
                winner_team,
                away_score,
                home_score
            FROM game_results_deduped
            WHERE game_date BETWEEN ? AND ?
              AND winner_team IS NOT NULL
            ORDER BY game_date, game_id
            """,
            (start_date, end_date),
        )
    finally:
        conn.close()


def _load_local_pitching_logs(start_date: str, end_date: str) -> pd.DataFrame:
    conn = connect_read_only(settings().duckdb_path)
    try:
        return query_dataframe(
            conn,
            """
            SELECT * EXCLUDE (imported_at)
            FROM mlb_pitching_logs_deduped
            WHERE game_date BETWEEN ? AND ?
            """,
            (start_date, end_date),
        )
    finally:
        conn.close()


def _load_local_batting_logs(start_date: str, end_date: str) -> pd.DataFrame:
    conn = connect_read_only(settings().duckdb_path)
    try:
        return query_dataframe(
            conn,
            """
            SELECT * EXCLUDE (imported_at)
            FROM mlb_batting_logs_deduped
            WHERE game_date BETWEEN ? AND ?
            """,
            (start_date, end_date),
        )
    finally:
        conn.close()


def _load_research_frames(
    train_start_date: str,
    train_end_date: str,
    eval_start_date: str,
    eval_end_date: str,
) -> dict[str, pd.DataFrame]:
    combined_results_df = _load_local_results_frame(train_start_date, eval_end_date)
    if combined_results_df.empty:
        return {
            "combined_results_df": combined_results_df,
            "training_df": pd.DataFrame(),
            "eval_results_df": pd.DataFrame(),
            "eval_replay_df": pd.DataFrame(),
            "valid_df": pd.DataFrame(),
        }

    eval_results_df = combined_results_df[
        pd.to_datetime(combined_results_df["game_date"]).between(pd.Timestamp(eval_start_date), pd.Timestamp(eval_end_date))
    ].copy()
    eval_replay_df = load_kalshi_pregame_replay(eval_start_date, eval_end_date, games_df=eval_results_df)

    pitching_logs_df = _load_local_pitching_logs(train_start_date, eval_end_date)
    batting_logs_df = _load_local_batting_logs(train_start_date, eval_end_date)
    sbro_market_df = _load_sbro_market_priors(train_start_date, train_end_date)
    eval_market_df = (
        eval_replay_df[["game_id", "home_market_prob"]].rename(columns={"home_market_prob": MARKET_FEATURE_COLUMN})
        if not eval_replay_df.empty
        else pd.DataFrame(columns=["game_id", MARKET_FEATURE_COLUMN])
    )
    market_priors_df = pd.concat([sbro_market_df, eval_market_df], ignore_index=True).drop_duplicates(subset=["game_id"], keep="last")
    weather_df = _load_game_weather(train_start_date, eval_end_date)

    training_df = build_training_dataset(
        combined_results_df,
        pitching_logs_df,
        batting_logs_df=batting_logs_df,
        market_priors_df=market_priors_df,
        weather_df=weather_df,
    )
    if training_df.empty:
        valid_df = pd.DataFrame()
    else:
        _require_feature_columns(training_df)
        training_dates = pd.to_datetime(training_df["game_date"])
        valid_df = training_df[training_dates.between(pd.Timestamp(eval_start_date), pd.Timestamp(eval_end_date))].copy()

    return {
        "combined_results_df": combined_results_df,
        "training_df": training_df,
        "eval_results_df": eval_results_df,
        "eval_replay_df": eval_replay_df,
        "valid_df": valid_df,
    }


def _fit_and_score_contenders(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    contender_specs = _build_research_contenders()
    probabilities_by_contender: dict[str, np.ndarray] = {}
    contender_rows: dict[str, dict[str, Any]] = {}
    calibration_rows: list[dict[str, Any]] = []

    bayes_bundle = _train_bayesian_bundle(train_df)

    for spec in contender_specs:
        if spec.builder_kind == "identity":
            probabilities = np.clip(valid_df[MARKET_FEATURE_COLUMN].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        elif spec.builder_kind == "bayes":
            probabilities = _bayesian_posterior_probabilities(
                {
                    "pipeline": bayes_bundle["pipeline"],
                    "bayes_evidence_pipeline": bayes_bundle["pipeline"],
                    "bayes_base_rate": bayes_bundle["base_rate"],
                    "bayes_prior_default": 0.5,
                    "bayes_evidence_feature_columns": BAYES_EVIDENCE_FEATURE_COLUMNS,
                },
                valid_df,
            )
        else:
            pipeline = _build_pipeline(spec.builder_kind, list(spec.feature_columns))
            pipeline.fit(train_df[list(spec.feature_columns)], train_df["target_home_win"].astype(int))
            probabilities = np.asarray(pipeline.predict_proba(valid_df[list(spec.feature_columns)]), dtype=float)[:, 1]

        metrics = _classification_metrics(valid_df["target_home_win"], probabilities)
        probabilities_by_contender[spec.name] = probabilities
        contender_rows[spec.name] = {
            "contender_name": spec.name,
            "family": spec.family,
            "model_family": spec.model_family,
            "feature_variant": spec.feature_variant,
            **metrics,
        }
        calibration_rows.extend(_calibration_table(valid_df["target_home_win"], probabilities, spec.name).to_dict(orient="records"))

    ensemble_specs = {
        "ensemble_baseball_mean_v1": [name for name in probabilities_by_contender if name.startswith("baseball_")],
        "ensemble_hybrid_mean_v1": [name for name in probabilities_by_contender if name.startswith("hybrid_") or name == BAYES_MODEL_NAME],
        "ensemble_all_mean_v1": [name for name in probabilities_by_contender if name != "market_identity_v1"],
    }
    for name, members in ensemble_specs.items():
        if not members:
            continue
        probabilities = np.mean(np.vstack([probabilities_by_contender[member] for member in members]), axis=0)
        metrics = _classification_metrics(valid_df["target_home_win"], probabilities)
        probabilities_by_contender[name] = probabilities
        contender_rows[name] = {
            "contender_name": name,
            "family": "ensemble",
            "model_family": "mean_ensemble",
            "feature_variant": "derived",
            **metrics,
        }
        calibration_rows.extend(_calibration_table(valid_df["target_home_win"], probabilities, name).to_dict(orient="records"))

    return probabilities_by_contender, contender_rows, calibration_rows


def _stake_units(sizing_policy: str, *, model_prob: float, implied_decimal_odds: float, edge_bps: int) -> float:
    if sizing_policy == "flat_1u":
        return 1.0
    if sizing_policy == "edge_scaled_cap_1u":
        return float(min(1.0, max(0.1, edge_bps / 500.0)))
    if sizing_policy == "fractional_kelly_25_cap_1u":
        b = max(implied_decimal_odds - 1.0, 1e-6)
        p = float(np.clip(model_prob, 1e-6, 1 - 1e-6))
        q = 1.0 - p
        kelly_fraction = max(0.0, ((b * p) - q) / b)
        return float(min(1.0, max(0.0, 0.25 * kelly_fraction)))
    raise ValueError(f"Unsupported sizing_policy={sizing_policy}")


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    cumulative = returns.cumsum()
    running_peak = cumulative.cummax()
    drawdowns = cumulative - running_peak
    return float(abs(drawdowns.min()))


def _slice_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    sliced = frame.copy()
    sliced["slice"] = pd.to_datetime(sliced["game_date"]).dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for slice_name, group in sliced.groupby("slice", sort=True):
        stake_total = float(group["stake_units"].sum())
        units_won = float(group["realized_return_units"].sum())
        rows.append(
            {
                "slice": slice_name,
                "bets": int(len(group)),
                "stake_total": stake_total,
                "units_won": units_won,
                "roi": float(units_won / stake_total) if stake_total > 0 else 0.0,
            }
        )
    return rows


def _evaluate_strategies(
    valid_games: pd.DataFrame,
    replay_quotes: pd.DataFrame,
    results_lookup: pd.DataFrame,
    valid_df: pd.DataFrame,
    probabilities_by_contender: dict[str, np.ndarray],
    contender_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = []
    strategy_specs = _build_strategy_specs()

    for contender_name, probabilities in probabilities_by_contender.items():
        prediction_rows: list[dict[str, Any]] = []
        for row, home_probability in zip(valid_df.to_dict(orient="records"), probabilities, strict=False):
            snapshot_ts = pd.Timestamp(
                valid_games.loc[valid_games["game_id"] == row["game_id"], "snapshot_ts"].iloc[0]
            )
            prediction_rows.append(
                {
                    "game_id": row["game_id"],
                    "snapshot_ts": snapshot_ts,
                    "collection_run_ts": snapshot_ts,
                    "team": row["home_team"],
                    "opponent_team": row["away_team"],
                    "model_name": contender_name,
                    "model_prob": float(home_probability),
                }
            )
            prediction_rows.append(
                {
                    "game_id": row["game_id"],
                    "snapshot_ts": snapshot_ts,
                    "collection_run_ts": snapshot_ts,
                    "team": row["away_team"],
                    "opponent_team": row["home_team"],
                    "model_name": contender_name,
                    "model_prob": float(1.0 - home_probability),
                }
            )

        base_opportunities = build_bet_opportunities(valid_games, replay_quotes, pd.DataFrame(prediction_rows))
        if base_opportunities.empty:
            for spec in strategy_specs:
                strategies.append(
                    {
                        "strategy_name": f"{contender_name}__{spec.name}",
                        "contender_name": contender_name,
                        "family": contender_rows[contender_name]["family"],
                        "model_family": contender_rows[contender_name]["model_family"],
                        "feature_variant": contender_rows[contender_name]["feature_variant"],
                        "entry_rule": f"edge_gte_{spec.edge_threshold_bps}bps",
                        "sizing_policy": spec.sizing_policy,
                        "bets": 0,
                        "stake_total": 0.0,
                        "units_won": 0.0,
                        "roi": 0.0,
                        "hit_rate": 0.0,
                        "avg_edge_bps": 0.0,
                        "active_slices": 0,
                        "max_slice_bet_share": 0.0,
                        "max_drawdown": 0.0,
                        "guardrails_passed": False,
                        "slice_metrics": [],
                        "log_loss": contender_rows[contender_name]["log_loss"],
                        "roc_auc": contender_rows[contender_name]["roc_auc"],
                    }
                )
            continue

        settled = base_opportunities.merge(results_lookup, on="game_id", how="left")
        settled["won_bet"] = settled["winner_team"].eq(settled["team"])

        for spec in strategy_specs:
            actionable = settled[settled["edge_bps"] >= spec.edge_threshold_bps].copy()
            if not actionable.empty:
                actionable["stake_units"] = actionable.apply(
                    lambda row: _stake_units(
                        spec.sizing_policy,
                        model_prob=float(row["model_prob"]),
                        implied_decimal_odds=float(row["implied_decimal_odds"]),
                        edge_bps=int(row["edge_bps"]),
                    ),
                    axis=1,
                )
                actionable = actionable[actionable["stake_units"] > 0].copy()
                actionable["realized_return_units"] = np.where(
                    actionable["won_bet"],
                    (actionable["implied_decimal_odds"] - 1.0) * actionable["stake_units"],
                    -1.0 * actionable["stake_units"],
                )
            stake_total = float(actionable["stake_units"].sum()) if not actionable.empty else 0.0
            units_won = float(actionable["realized_return_units"].sum()) if not actionable.empty else 0.0
            bets = int(len(actionable))
            slice_metrics = _slice_metrics(actionable)
            active_slices = len([row for row in slice_metrics if row["bets"] > 0])
            max_slice_bet_share = max((row["bets"] / bets for row in slice_metrics), default=0.0) if bets else 0.0
            guardrails_passed = (
                bets >= DEFAULT_MIN_BETS
                and active_slices >= DEFAULT_MIN_ACTIVE_SLICES
                and max_slice_bet_share <= DEFAULT_MAX_SLICE_BET_SHARE
            )
            strategies.append(
                {
                    "strategy_name": f"{contender_name}__{spec.name}",
                    "contender_name": contender_name,
                    "family": contender_rows[contender_name]["family"],
                    "model_family": contender_rows[contender_name]["model_family"],
                    "feature_variant": contender_rows[contender_name]["feature_variant"],
                    "entry_rule": f"edge_gte_{spec.edge_threshold_bps}bps",
                    "sizing_policy": spec.sizing_policy,
                    "bets": bets,
                    "stake_total": stake_total,
                    "units_won": units_won,
                    "roi": float(units_won / stake_total) if stake_total > 0 else 0.0,
                    "hit_rate": float(actionable["won_bet"].mean()) if bets else 0.0,
                    "avg_edge_bps": float(actionable["edge_bps"].mean()) if bets else 0.0,
                    "active_slices": active_slices,
                    "max_slice_bet_share": float(max_slice_bet_share),
                    "max_drawdown": _max_drawdown(actionable.sort_values(["game_date", "event_start_time"])["realized_return_units"]) if bets else 0.0,
                    "guardrails_passed": guardrails_passed,
                    "slice_metrics": slice_metrics,
                    "log_loss": contender_rows[contender_name]["log_loss"],
                    "roc_auc": contender_rows[contender_name]["roc_auc"],
                }
            )

    return strategies


def _sort_strategies(strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        strategies,
        key=lambda row: (
            bool(row["guardrails_passed"]),
            float(row["roi"]),
            float(row["units_won"]),
            -float(row["max_drawdown"]),
            -float(row["log_loss"]),
        ),
        reverse=True,
    )


def run_kalshi_edge_research_backtest(
    train_start_date: str,
    train_end_date: str,
    eval_start_date: str,
    eval_end_date: str,
) -> dict[str, Any]:
    frames = _load_research_frames(train_start_date, train_end_date, eval_start_date, eval_end_date)
    combined_results_df = frames["combined_results_df"]
    training_df = frames["training_df"]
    eval_results_df = frames["eval_results_df"]
    eval_replay_df = frames["eval_replay_df"]
    valid_df = frames["valid_df"]

    if combined_results_df.empty:
        return {
            "status": "insufficient_data",
            "rows": 0,
            "message": "No local MLB results found for the combined train/eval window. Run `mlpm backfill-mlb` first.",
        }
    if training_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "Training dataset is empty."}

    training_dates = pd.to_datetime(training_df["game_date"])
    train_df = training_df[training_dates.between(pd.Timestamp(train_start_date), pd.Timestamp(train_end_date))].copy()
    if train_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "Training dataset is empty for the requested train window."}
    if valid_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "Evaluation dataset is empty for the requested eval window."}
    if eval_replay_df.empty or eval_results_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "No replay-selected Kalshi pregame quotes found."}

    valid_replay = eval_replay_df[eval_replay_df["game_id"].isin(valid_df["game_id"])].copy()
    if valid_replay.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "Evaluation slice has no replay-selected Kalshi quotes."}
    valid_df = valid_df[valid_df["game_id"].isin(valid_replay["game_id"])].copy()
    if valid_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "Evaluation feature rows do not overlap replay-selected Kalshi quotes."}

    valid_games = valid_replay[["game_id", "game_date", "event_start_time", "snapshot_ts", "home_team", "away_team"]].copy()
    replay_quotes = pd.DataFrame(build_kalshi_replay_quote_rows(valid_replay))
    results_lookup = eval_results_df.copy()
    for column in ("away_score", "home_score"):
        if column not in results_lookup.columns:
            results_lookup[column] = None
    results_lookup = results_lookup[["game_id", "winner_team", "away_score", "home_score"]].drop_duplicates(subset=["game_id"])

    probabilities_by_contender, contender_rows, calibration_rows = _fit_and_score_contenders(train_df, valid_df)
    strategies = _evaluate_strategies(valid_games, replay_quotes, results_lookup, valid_df, probabilities_by_contender, contender_rows)
    sorted_strategies = _sort_strategies(strategies)
    eligible = [row for row in sorted_strategies if row["guardrails_passed"]]
    champion = (eligible or sorted_strategies)[0] if sorted_strategies else None

    return {
        "status": "ok",
        "rows": len(valid_df),
        "rows_train": len(train_df),
        "rows_valid": len(valid_df),
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "eval_start_date": eval_start_date,
        "eval_end_date": eval_end_date,
        "replay_rows": int(len(eval_replay_df)),
        "valid_replay_rows": int(len(valid_replay)),
        "contender_count": len(contender_rows),
        "strategy_count": len(sorted_strategies),
        "champion_strategy": champion["strategy_name"] if champion else None,
        "champion_contender": champion["contender_name"] if champion else None,
        "guardrail_champion": bool(champion["guardrails_passed"]) if champion else False,
        "contenders": list(contender_rows.values()),
        "strategies": sorted_strategies,
        "calibration": calibration_rows,
    }
