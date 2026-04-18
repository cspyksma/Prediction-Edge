from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mlpm.config.settings import settings
from mlpm.features.matchups import OFFENSE_SPLIT_PRIOR, offense_score_from_stats, offense_split_value
from mlpm.features.team_strength import ELO_BASELINE, ELO_K_FACTOR, RECENT_GAMES_WINDOW
from mlpm.features.utils import coalesce_float, current_streak, normalize_pitcher_hand, smoothed_win_pct
from mlpm.ingest.mlb_stats import fetch_final_results, fetch_game_batting_logs, fetch_game_pitching_logs
from mlpm.storage.duckdb import append_dataframe, connect, query_dataframe

logger = logging.getLogger(__name__)

LOGISTIC_MODEL_NAME = "mlb_win_logreg_v2"
HISTGB_MODEL_NAME = "mlb_win_histgb_v1"
KNN_MODEL_NAME = "mlb_win_knn_v1"
SVM_MODEL_NAME = "mlb_win_svm_rbf_v1"
BAYES_MODEL_NAME = "mlb_win_bayes_v1"
FEATURE_COLUMNS = [
    "season_win_pct_diff",
    "recent_win_pct_diff",
    "venue_win_pct_diff",
    "run_diff_per_game_diff",
    "season_runs_scored_per_game_diff",
    "season_runs_allowed_per_game_adv",
    "recent_runs_scored_per_game_diff",
    "recent_runs_allowed_per_game_adv",
    "rest_days_diff",
    "venue_streak_diff",
    "travel_switch_adv",
    "doubleheader_flag",
    "streak_diff",
    "elo_diff",
    "elo_home_win_prob",
    "market_home_implied_prob",
    "offense_vs_starter_hand_diff",
    "starter_era_adv",
    "starter_whip_adv",
    "starter_strikeouts_per_9_diff",
    "starter_walks_per_9_adv",
    "bullpen_innings_3d_adv",
    "bullpen_pitches_3d_adv",
    "relievers_used_3d_adv",
]
BAYES_EVIDENCE_FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column != "market_home_implied_prob"]
STARTER_LOOKBACK_GAMES = 5

# Heuristic benchmark weights are hand-tuned so the baseline stays interpretable and roughly aligned
# with the relative scale of the engineered tabular features. They are not learned coefficients.
HEURISTIC_WEIGHT_SEASON_WIN_PCT = 1.2
HEURISTIC_WEIGHT_RECENT_WIN_PCT = 0.8
HEURISTIC_WEIGHT_VENUE_WIN_PCT = 0.6
HEURISTIC_WEIGHT_RUN_DIFF = 0.2
HEURISTIC_WEIGHT_SEASON_RUNS_SCORED = 0.10
HEURISTIC_WEIGHT_SEASON_RUNS_ALLOWED = 0.10
HEURISTIC_WEIGHT_RECENT_RUNS_SCORED = 0.14
HEURISTIC_WEIGHT_RECENT_RUNS_ALLOWED = 0.14
HEURISTIC_WEIGHT_REST_DAYS = 0.05
HEURISTIC_WEIGHT_VENUE_STREAK = 0.02
HEURISTIC_WEIGHT_TRAVEL_SWITCH = 0.08
HEURISTIC_WEIGHT_DOUBLEHEADER = 0.05
HEURISTIC_WEIGHT_STREAK = 0.03
HEURISTIC_ELO_DIFF_DIVISOR = 150.0
HEURISTIC_ELO_PROB_CENTER_WEIGHT = 1.5
HEURISTIC_WEIGHT_STARTER_ERA = 0.12
HEURISTIC_WEIGHT_STARTER_WHIP = 0.45
HEURISTIC_WEIGHT_STARTER_K9 = 0.04
HEURISTIC_WEIGHT_STARTER_BB9 = 0.07
HEURISTIC_WEIGHT_BULLPEN_INNINGS = 0.015
HEURISTIC_BULLPEN_PITCHES_DIVISOR = 5000.0
HEURISTIC_WEIGHT_RELIEVERS_USED = 0.02


@dataclass
class TeamSnapshot:
    games_played: int
    season_win_pct: float
    recent_win_pct: float
    venue_win_pct: float
    run_diff_per_game: float
    season_runs_scored_per_game: float
    season_runs_allowed_per_game: float
    recent_runs_scored_per_game: float
    recent_runs_allowed_per_game: float
    streak: int
    elo_rating: float
    last_game_date: pd.Timestamp | None
    last_is_home: bool | None
    current_venue_streak: int


@dataclass(frozen=True)
class InformationCriteriaResult:
    features: tuple[str, ...]
    log_likelihood: float
    aic: float
    bic: float
    metrics: dict[str, float]


def artifact_path() -> Path:
    return settings().artifacts_dir / "game_outcome_model.pkl"


def load_trained_model(path: Path | None = None) -> dict[str, object] | None:
    """Load trusted-local model artifacts only. Do not load untrusted files."""
    model_path = path or artifact_path()
    if not model_path.exists():
        return None
    try:
        return joblib_load(model_path)
    except Exception:
        with model_path.open("rb") as handle:
            return pickle.load(handle)


def save_trained_model(bundle: dict[str, object], path: Path | None = None) -> Path:
    model_path = path or artifact_path()
    joblib_dump(bundle, model_path)
    return model_path


def fetch_training_inputs(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        fetch_final_results(start_date, end_date),
        fetch_game_pitching_logs(start_date, end_date),
        fetch_game_batting_logs(start_date, end_date),
        _load_historical_market_priors(start_date, end_date),
    )


def build_training_dataset(
    results_df: pd.DataFrame,
    pitching_logs_df: pd.DataFrame,
    batting_logs_df: pd.DataFrame | None = None,
    market_priors_df: pd.DataFrame | None = None,
    min_games: int | None = None,
) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    minimum_games = settings().model_min_games if min_games is None else min_games
    results = results_df.copy()
    results["game_date"] = pd.to_datetime(results["game_date"])
    skipped_min_games = 0

    pitching_logs = pitching_logs_df.copy()
    if not pitching_logs.empty:
        pitching_logs["game_date"] = pd.to_datetime(pitching_logs["game_date"])
        pitching_lookup = pitching_logs.set_index(["game_id", "team"]).to_dict(orient="index")
    else:
        pitching_lookup = {}
    batting_lookup = {}
    if batting_logs_df is not None and not batting_logs_df.empty:
        batting_logs = batting_logs_df.copy()
        batting_logs["game_date"] = pd.to_datetime(batting_logs["game_date"])
        batting_lookup = batting_logs.set_index(["game_id", "team"]).to_dict(orient="index")
    market_prior_lookup: dict[str, float] = {}
    if market_priors_df is not None and not market_priors_df.empty:
        market_prior_lookup = (
            market_priors_df.drop_duplicates(subset=["game_id"]).set_index("game_id")["market_home_implied_prob"].to_dict()
        )

    team_history: dict[str, list[dict[str, object]]] = {}
    pitcher_history: dict[int, list[dict[str, float]]] = {}
    team_pitching_history: dict[str, list[dict[str, object]]] = {}
    team_offense_hand_history: dict[str, list[dict[str, object]]] = {}
    elo_ratings: dict[str, float] = {}
    rows: list[dict[str, object]] = []

    for game in results.sort_values(["game_date", "game_id"]).to_dict(orient="records"):
        home_team = game["home_team"]
        away_team = game["away_team"]
        home_games = team_history.get(home_team, [])
        away_games = team_history.get(away_team, [])
        home_elo = elo_ratings.get(home_team, ELO_BASELINE)
        away_elo = elo_ratings.get(away_team, ELO_BASELINE)
        if min(len(home_games), len(away_games)) >= minimum_games:
            game_date = pd.to_datetime(game["game_date"])
            home_snapshot = _snapshot_team(home_games, venue_home=True, elo_rating=home_elo)
            away_snapshot = _snapshot_team(away_games, venue_home=False, elo_rating=away_elo)
            home_pitching = _pregame_pitching_context(game["game_id"], home_team, pitching_lookup, pitcher_history, team_pitching_history)
            away_pitching = _pregame_pitching_context(game["game_id"], away_team, pitching_lookup, pitcher_history, team_pitching_history)
            home_offense_vs_opp_hand = _offense_vs_hand_before_game(team_offense_hand_history.get(home_team, []), away_pitching.get("pitcher_hand"))
            away_offense_vs_opp_hand = _offense_vs_hand_before_game(team_offense_hand_history.get(away_team, []), home_pitching.get("pitcher_hand"))
            is_doubleheader = _team_games_on_date(team_history.get(home_team, []), game_date) > 0 or _team_games_on_date(team_history.get(away_team, []), game_date) > 0
            rows.append(
                {
                    "game_id": game["game_id"],
                    "game_date": game["game_date"],
                    "home_team": home_team,
                    "away_team": away_team,
                    "target_home_win": int(game["winner_team"] == home_team),
                    **_build_game_features(
                        home_snapshot,
                        away_snapshot,
                        home_pitching,
                        away_pitching,
                        game_date=game_date,
                        is_doubleheader=is_doubleheader,
                        market_home_implied_prob=_market_home_implied_prob(market_prior_lookup.get(game["game_id"])),
                        home_offense_vs_opp_hand=home_offense_vs_opp_hand,
                        away_offense_vs_opp_hand=away_offense_vs_opp_hand,
                    ),
                }
            )
        else:
            skipped_min_games += 1

        _update_histories(
            game,
            pitching_lookup,
            batting_lookup,
            team_history,
            pitcher_history,
            team_pitching_history,
            team_offense_hand_history,
        )
        _update_elo(game, elo_ratings)

    training_df = pd.DataFrame(rows)
    if skipped_min_games > 0:
        logger.warning(
            "Training dataset dropped %s games below MODEL_MIN_GAMES; source_games=%s produced_rows=%s min_games=%s",
            skipped_min_games,
            len(results),
            len(training_df),
            minimum_games,
        )
    return training_df


def train_game_outcome_model(training_df: pd.DataFrame) -> dict[str, object]:
    if training_df.empty:
        raise ValueError("Training dataset is empty.")
    if training_df["target_home_win"].nunique() < 2:
        raise ValueError("Training dataset must contain both classes.")
    _require_feature_columns(training_df)

    train_df, valid_df = _split_training_data(training_df)
    candidate_models = _fit_candidate_models(train_df)
    candidate_metrics = {
        model_name: _classification_metrics(valid_df["target_home_win"], pipeline.predict_proba(valid_df[FEATURE_COLUMNS])[:, 1])
        for model_name, pipeline in candidate_models.items()
    }
    selected_model_name = _select_model(candidate_metrics)
    selected_pipeline = candidate_models[selected_model_name]
    bayes_bundle = _train_bayesian_bundle(train_df)
    benchmarks = benchmark_game_outcome_models(train_df, valid_df, candidate_models)
    selected_metrics = dict(candidate_metrics[selected_model_name])
    selected_metrics["rows_train"] = int(len(train_df))
    selected_metrics["rows_valid"] = int(len(valid_df))

    # Permutation importance per candidate pipeline on the held-out validation set.
    feature_importances = compute_permutation_importances(candidate_models, valid_df)

    mlflow.set_tracking_uri(settings().mlflow_tracking_uri)
    with mlflow.start_run(run_name=selected_model_name):
        mlflow.log_params(
            {
                "selected_model": selected_model_name,
                "target": "home_win",
                "features": ",".join(FEATURE_COLUMNS),
            }
        )
        for model_name, metrics in candidate_metrics.items():
            prefix = model_name.replace("-", "_")
            mlflow.log_metrics({f"{prefix}_{key}": value for key, value in metrics.items()})
        if not feature_importances.empty:
            for importance_row in feature_importances.to_dict(orient="records"):
                prefix = str(importance_row["model_name"]).replace("-", "_")
                metric_name = f"feat_importance__{prefix}__{importance_row['feature']}"
                try:
                    mlflow.log_metric(metric_name, float(importance_row["importance"]))
                except Exception:  # pragma: no cover - defensive
                    pass
        try:
            mlflow.sklearn.log_model(selected_pipeline, "model")
        except PermissionError:
            pass

    return {
        "model_name": selected_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "trained_on_start_date": train_df["game_date"].min().date().isoformat(),
        "trained_on_end_date": valid_df["game_date"].max().date().isoformat(),
        "metrics": selected_metrics,
        "candidate_metrics": candidate_metrics,
        "benchmarks": benchmarks,
        "feature_importances": feature_importances.to_dict(orient="records"),
        "pipeline": selected_pipeline,
        "candidate_pipelines": candidate_models,
        "bayes_model_name": BAYES_MODEL_NAME,
        "bayes_base_rate": bayes_bundle["base_rate"],
        "bayes_evidence_pipeline": bayes_bundle["pipeline"],
        "bayes_evidence_feature_columns": BAYES_EVIDENCE_FEATURE_COLUMNS,
        "bayes_prior_default": 0.5,
        "rows_train": int(len(train_df)),
        "rows_valid": int(len(valid_df)),
    }


def train_and_save_model(start_date: str, end_date: str, path: Path | None = None) -> dict[str, object]:
    results_df, pitching_logs_df, batting_logs_df, market_priors_df = fetch_training_inputs(start_date, end_date)
    training_df = build_training_dataset(results_df, pitching_logs_df, batting_logs_df=batting_logs_df, market_priors_df=market_priors_df)
    bundle = train_game_outcome_model(training_df)
    saved_path = save_trained_model(bundle, path=path)

    importances = pd.DataFrame(bundle.get("feature_importances") or [])
    rows_written = 0
    if not importances.empty:
        try:
            rows_written = persist_feature_importances(
                importances,
                trained_at=datetime.now(),
                train_start_date=str(bundle.get("trained_on_start_date") or start_date),
                train_end_date=str(bundle.get("trained_on_end_date") or end_date),
                rows_train=int(bundle.get("rows_train") or 0),
                rows_valid=int(bundle.get("rows_valid") or 0),
                method="permutation",
            )
        except Exception:  # pragma: no cover - don't fail the training if DuckDB write fails
            logger.warning("Failed to persist feature importances", exc_info=True)

    return {
        "bundle": bundle,
        "path": saved_path,
        "rows": len(training_df),
        "feature_importance_rows": rows_written,
    }


def run_model_benchmark(start_date: str, end_date: str) -> dict[str, object]:
    results_df, pitching_logs_df, batting_logs_df, market_priors_df = fetch_training_inputs(start_date, end_date)
    training_df = build_training_dataset(results_df, pitching_logs_df, batting_logs_df=batting_logs_df, market_priors_df=market_priors_df)
    if training_df.empty:
        return {"status": "insufficient_data", "rows": 0}
    _require_feature_columns(training_df)

    train_df, valid_df = _split_training_data(training_df)
    candidates = _fit_candidate_models(train_df)
    benchmark = benchmark_game_outcome_models(train_df, valid_df, candidates)
    return {
        "status": "ok",
        "rows": len(training_df),
        "rows_train": len(train_df),
        "rows_valid": len(valid_df),
        "benchmarks": benchmark["benchmarks"],
        "calibration": benchmark["calibration"],
    }


def run_historical_kalshi_backtest(
    train_start_date: str,
    train_end_date: str,
    eval_start_date: str,
    eval_end_date: str,
) -> dict[str, object]:
    from mlpm.evaluation.strategy import build_bet_opportunities
    from mlpm.historical.replay import build_kalshi_replay_quote_rows, load_kalshi_pregame_replay

    combined_results_df = fetch_final_results(train_start_date, eval_end_date)
    if combined_results_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "No MLB results found for the combined train/eval window."}
    replay_df = load_kalshi_pregame_replay(train_start_date, eval_end_date, games_df=combined_results_df)
    eval_results_df = combined_results_df[
        pd.to_datetime(combined_results_df["game_date"]).between(pd.Timestamp(eval_start_date), pd.Timestamp(eval_end_date))
    ].copy()
    if replay_df.empty or eval_results_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "No replay-selected Kalshi pregame quotes found."}

    pitching_logs_df = fetch_game_pitching_logs(train_start_date, eval_end_date)
    batting_logs_df = fetch_game_batting_logs(train_start_date, eval_end_date)
    market_priors_df = replay_df[["game_id", "home_market_prob"]].rename(columns={"home_market_prob": "market_home_implied_prob"})
    training_df = build_training_dataset(
        combined_results_df,
        pitching_logs_df,
        batting_logs_df=batting_logs_df,
        market_priors_df=market_priors_df,
    )
    if training_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "Training dataset is empty."}
    _require_feature_columns(training_df)

    training_dates = pd.to_datetime(training_df["game_date"])
    train_df = training_df[training_dates.between(pd.Timestamp(train_start_date), pd.Timestamp(train_end_date))].copy()
    valid_df = training_df[training_dates.between(pd.Timestamp(eval_start_date), pd.Timestamp(eval_end_date))].copy()
    if train_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "Training dataset is empty for the requested train window."}
    if valid_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "Evaluation dataset is empty for the requested eval window."}

    valid_replay = replay_df[replay_df["game_id"].isin(valid_df["game_id"])].copy()
    if valid_replay.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "Evaluation slice has no replay-selected Kalshi quotes."}
    valid_df = valid_df[valid_df["game_id"].isin(valid_replay["game_id"])].copy()
    if valid_df.empty:
        return {"status": "insufficient_data", "rows": 0, "message": "Evaluation feature rows do not overlap replay-selected Kalshi quotes."}

    valid_games = valid_replay[["game_id", "game_date", "event_start_time", "home_team", "away_team"]].copy()
    replay_quotes = pd.DataFrame(build_kalshi_replay_quote_rows(valid_replay))
    results_lookup = eval_results_df.copy()
    for column in ("away_score", "home_score"):
        if column not in results_lookup.columns:
            results_lookup[column] = None
    results_lookup = results_lookup[["game_id", "winner_team", "away_score", "home_score"]].drop_duplicates(subset=["game_id"])

    candidates = _fit_candidate_models(train_df)
    bayes_bundle = _train_bayesian_bundle(train_df)
    model_probabilities: dict[str, np.ndarray] = {
        name: np.asarray(pipeline.predict_proba(valid_df[FEATURE_COLUMNS]), dtype=float)[:, 1]
        for name, pipeline in candidates.items()
    }
    model_probabilities[BAYES_MODEL_NAME] = _bayesian_posterior_probabilities(
        {
            "pipeline": bayes_bundle["pipeline"],
            "bayes_evidence_pipeline": bayes_bundle["pipeline"],
            "bayes_base_rate": bayes_bundle["base_rate"],
            "bayes_prior_default": 0.5,
            "bayes_evidence_feature_columns": BAYES_EVIDENCE_FEATURE_COLUMNS,
        },
        valid_df,
    )

    metrics_by_model: dict[str, dict[str, float]] = {}
    opportunities_by_model: dict[str, pd.DataFrame] = {}
    calibration_rows: list[dict[str, object]] = []
    for model_name, probabilities in model_probabilities.items():
        metrics = _classification_metrics(valid_df["target_home_win"], probabilities)
        calibration_rows.extend(_calibration_table(valid_df["target_home_win"], probabilities, model_name).to_dict(orient="records"))

        prediction_rows: list[dict[str, object]] = []
        for row, home_probability in zip(valid_df.to_dict(orient="records"), probabilities, strict=False):
            snapshot_ts = valid_replay.loc[valid_replay["game_id"] == row["game_id"], "snapshot_ts"].iloc[0]
            prediction_rows.append(
                {
                    "game_id": row["game_id"],
                    "snapshot_ts": snapshot_ts,
                    "collection_run_ts": snapshot_ts,
                    "team": row["home_team"],
                    "opponent_team": row["away_team"],
                    "model_name": model_name,
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
                    "model_name": model_name,
                    "model_prob": float(1.0 - home_probability),
                }
            )

        model_predictions = pd.DataFrame(prediction_rows)
        opportunities = build_bet_opportunities(valid_games, replay_quotes, model_predictions)
        settled = opportunities.merge(results_lookup, on="game_id", how="left")
        if not settled.empty:
            settled["won_bet"] = settled["winner_team"].eq(settled["team"])
            settled["realized_return_units"] = np.where(
                settled["won_bet"],
                (settled["implied_decimal_odds"] - 1.0) * settled["stake_units"],
                -1.0 * settled["stake_units"],
            )
            actionable = settled[settled["is_actionable"]].copy()
        else:
            actionable = settled

        bets = int(len(actionable))
        units_won = float(actionable["realized_return_units"].sum()) if bets else 0.0
        roi = float(units_won / bets) if bets else 0.0
        avg_edge_bps = float(actionable["edge_bps"].mean()) if bets else 0.0
        metrics_by_model[model_name] = {
            **metrics,
            "bets": bets,
            "units_won": units_won,
            "roi": roi,
            "avg_edge_bps": avg_edge_bps,
        }
        opportunities_by_model[model_name] = actionable

    champion_model = max(
        metrics_by_model,
        key=lambda name: (metrics_by_model[name]["roi"], metrics_by_model[name]["units_won"], -metrics_by_model[name]["log_loss"]),
    )
    return {
        "status": "ok",
        "rows": len(valid_df),
        "rows_train": len(train_df),
        "rows_valid": len(valid_df),
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "eval_start_date": eval_start_date,
        "eval_end_date": eval_end_date,
        "replay_rows": len(replay_df),
        "valid_replay_rows": len(valid_replay),
        "champion_model": champion_model,
        "benchmarks": metrics_by_model,
        "calibration": calibration_rows,
    }


def run_forward_feature_selection(start_date: str, end_date: str) -> dict[str, object]:
    results_df, pitching_logs_df, batting_logs_df, market_priors_df = fetch_training_inputs(start_date, end_date)
    training_df = build_training_dataset(results_df, pitching_logs_df, batting_logs_df=batting_logs_df, market_priors_df=market_priors_df)
    if training_df.empty:
        return {"status": "insufficient_data", "rows": 0}
    return analyze_forward_feature_selection(training_df)


def analyze_forward_feature_selection(training_df: pd.DataFrame) -> dict[str, object]:
    if training_df.empty:
        raise ValueError("Training dataset is empty.")
    if training_df["target_home_win"].nunique() < 2:
        raise ValueError("Training dataset must contain both classes.")
    _require_feature_columns(training_df)

    train_df, valid_df = _split_training_data(training_df)
    aic_steps, best_aic = _forward_select_features(train_df, criterion="aic")
    bic_steps, best_bic = _forward_select_features(train_df, criterion="bic")

    return {
        "status": "ok",
        "rows": len(training_df),
        "rows_train": len(train_df),
        "rows_valid": len(valid_df),
        "aic_steps": _serialize_selection_steps(aic_steps, train_df, valid_df),
        "bic_steps": _serialize_selection_steps(bic_steps, train_df, valid_df),
        "best_aic_model": _serialize_information_result(best_aic, train_df, valid_df),
        "best_bic_model": _serialize_information_result(best_bic, train_df, valid_df),
    }


def benchmark_game_outcome_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    trained_models: dict[str, Pipeline] | Pipeline | None = None,
) -> dict[str, object]:
    if train_df.empty or valid_df.empty:
        raise ValueError("Training and validation frames must be non-empty.")
    _require_feature_columns(train_df)
    _require_feature_columns(valid_df)

    if trained_models is None:
        models = _fit_candidate_models(train_df)
    elif isinstance(trained_models, Pipeline):
        models = {LOGISTIC_MODEL_NAME: trained_models}
    else:
        models = trained_models

    target = valid_df["target_home_win"].astype(int)
    benchmark_probabilities = {
        name: pipeline.predict_proba(valid_df[FEATURE_COLUMNS])[:, 1]
        for name, pipeline in models.items()
    }
    bayes_bundle = _train_bayesian_bundle(train_df)
    benchmark_probabilities[BAYES_MODEL_NAME] = _bayesian_posterior_probabilities(
        bayes_bundle,
        valid_df,
    )
    home_win_rate = float(train_df["target_home_win"].mean())
    heuristic_score = (
        (valid_df["season_win_pct_diff"] * HEURISTIC_WEIGHT_SEASON_WIN_PCT)
        + (valid_df["recent_win_pct_diff"] * HEURISTIC_WEIGHT_RECENT_WIN_PCT)
        + (valid_df["venue_win_pct_diff"] * HEURISTIC_WEIGHT_VENUE_WIN_PCT)
        + (valid_df["run_diff_per_game_diff"] * HEURISTIC_WEIGHT_RUN_DIFF)
        + (valid_df["season_runs_scored_per_game_diff"] * HEURISTIC_WEIGHT_SEASON_RUNS_SCORED)
        + (valid_df["season_runs_allowed_per_game_adv"] * HEURISTIC_WEIGHT_SEASON_RUNS_ALLOWED)
        + (valid_df["recent_runs_scored_per_game_diff"] * HEURISTIC_WEIGHT_RECENT_RUNS_SCORED)
        + (valid_df["recent_runs_allowed_per_game_adv"] * HEURISTIC_WEIGHT_RECENT_RUNS_ALLOWED)
        + (valid_df["rest_days_diff"] * HEURISTIC_WEIGHT_REST_DAYS)
        + (valid_df["venue_streak_diff"] * HEURISTIC_WEIGHT_VENUE_STREAK)
        + (valid_df["travel_switch_adv"] * HEURISTIC_WEIGHT_TRAVEL_SWITCH)
        - (valid_df["doubleheader_flag"] * HEURISTIC_WEIGHT_DOUBLEHEADER)
        + (valid_df["streak_diff"] * HEURISTIC_WEIGHT_STREAK)
        + (valid_df["elo_diff"] / HEURISTIC_ELO_DIFF_DIVISOR)
        + ((valid_df["elo_home_win_prob"] - 0.5) * HEURISTIC_ELO_PROB_CENTER_WEIGHT)
        + (valid_df["starter_era_adv"] * HEURISTIC_WEIGHT_STARTER_ERA)
        + (valid_df["starter_whip_adv"] * HEURISTIC_WEIGHT_STARTER_WHIP)
        + (valid_df["starter_strikeouts_per_9_diff"] * HEURISTIC_WEIGHT_STARTER_K9)
        + (valid_df["starter_walks_per_9_adv"] * HEURISTIC_WEIGHT_STARTER_BB9)
        + (valid_df["bullpen_innings_3d_adv"] * HEURISTIC_WEIGHT_BULLPEN_INNINGS)
        + (valid_df["bullpen_pitches_3d_adv"] / HEURISTIC_BULLPEN_PITCHES_DIVISOR)
        + (valid_df["relievers_used_3d_adv"] * HEURISTIC_WEIGHT_RELIEVERS_USED)
    )
    benchmark_probabilities.update(
        {
            "home_win_rate_baseline": np.full(len(valid_df), home_win_rate),
            "always_home": np.full(len(valid_df), 0.999),
            "heuristic_feature_score": 1.0 / (1.0 + np.exp(-heuristic_score.to_numpy())),
        }
    )

    benchmark_metrics = {
        name: _classification_metrics(target, probabilities)
        for name, probabilities in benchmark_probabilities.items()
    }
    calibration = pd.concat(
        [_calibration_table(target, probabilities, model_name) for model_name, probabilities in benchmark_probabilities.items()],
        ignore_index=True,
    )
    return {"benchmarks": benchmark_metrics, "calibration": calibration.to_dict(orient="records")}


def compute_permutation_importances(
    candidates: dict[str, Pipeline],
    valid_df: pd.DataFrame,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "neg_log_loss",
) -> pd.DataFrame:
    """
    Compute sklearn permutation importance on the validation set for each candidate pipeline.

    Returns a long-format DataFrame with columns:
      model_name, feature, importance, importance_std, rank.
    Higher `importance` means permuting that feature degrades `scoring` more,
    i.e. the feature mattered more.
    """
    _require_feature_columns(valid_df)
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df["target_home_win"].astype(int)

    rows: list[dict[str, object]] = []
    for model_name, pipeline in candidates.items():
        try:
            result = permutation_importance(
                pipeline,
                X_valid,
                y_valid,
                n_repeats=n_repeats,
                random_state=random_state,
                scoring=scoring,
                n_jobs=1,
            )
        except Exception:  # pragma: no cover - defensive; some estimators may not score cleanly
            logger.warning("Permutation importance failed for %s; skipping", model_name, exc_info=True)
            continue

        importances = np.asarray(result.importances_mean, dtype=float)
        stds = np.asarray(result.importances_std, dtype=float)
        # Rank by descending importance.
        order = np.argsort(-importances, kind="mergesort")
        for rank_position, column_index in enumerate(order, start=1):
            rows.append(
                {
                    "model_name": model_name,
                    "feature": FEATURE_COLUMNS[int(column_index)],
                    "importance": float(importances[int(column_index)]),
                    "importance_std": float(stds[int(column_index)]),
                    "rank": rank_position,
                }
            )
    return pd.DataFrame(rows)


def persist_feature_importances(
    importances: pd.DataFrame,
    *,
    trained_at: datetime,
    train_start_date: str,
    train_end_date: str,
    rows_train: int,
    rows_valid: int,
    method: str = "permutation",
    duckdb_path: Path | None = None,
) -> int:
    """Append feature importance rows to DuckDB. Returns the number of rows written."""
    if importances.empty:
        return 0
    frame = importances.copy()
    frame.insert(0, "trained_at", trained_at)
    frame.insert(1, "train_start_date", train_start_date)
    frame.insert(2, "train_end_date", train_end_date)
    frame.insert(3, "rows_train", int(rows_train))
    frame.insert(4, "rows_valid", int(rows_valid))
    frame.insert(5, "method", method)
    # Ensure column order matches the table definition.
    frame = frame[
        [
            "trained_at",
            "model_name",
            "train_start_date",
            "train_end_date",
            "rows_train",
            "rows_valid",
            "method",
            "feature",
            "importance",
            "importance_std",
            "rank",
        ]
    ]
    path = duckdb_path or settings().duckdb_path
    conn = connect(path)
    try:
        append_dataframe(conn, "feature_importances", frame)
    finally:
        conn.close()
    return int(len(frame))


def build_live_feature_frame(
    games_df: pd.DataFrame,
    team_strengths_df: pd.DataFrame,
    games_with_pitching_df: pd.DataFrame,
    team_matchups_df: pd.DataFrame | None = None,
    market_priors_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if games_df.empty or team_strengths_df.empty:
        return pd.DataFrame()

    lookup = team_strengths_df.set_index("team").to_dict(orient="index")
    matchup_lookup = (
        team_matchups_df.set_index("team").to_dict(orient="index")
        if team_matchups_df is not None and not team_matchups_df.empty
        else {}
    )
    market_prior_lookup = (
        market_priors_df.drop_duplicates(subset=["game_id"]).set_index("game_id")["market_home_implied_prob"].to_dict()
        if market_priors_df is not None and not market_priors_df.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    for game in games_with_pitching_df.to_dict(orient="records"):
        home = lookup.get(game["home_team"])
        away = lookup.get(game["away_team"])
        if not home or not away:
            continue
        if min(home["games_played"], away["games_played"]) < settings().model_min_games:
            continue
        home_snapshot = TeamSnapshot(
            games_played=int(home["games_played"]),
            season_win_pct=float(home["season_win_pct"]),
            recent_win_pct=float(home["recent_win_pct"]),
            venue_win_pct=float(home["home_win_pct"]),
            run_diff_per_game=float(home["run_diff_per_game"]),
            season_runs_scored_per_game=float(home.get("season_runs_scored_per_game", 0.0)),
            season_runs_allowed_per_game=float(home.get("season_runs_allowed_per_game", 0.0)),
            recent_runs_scored_per_game=float(home.get("recent_runs_scored_per_game", 0.0)),
            recent_runs_allowed_per_game=float(home.get("recent_runs_allowed_per_game", 0.0)),
            streak=int(home["streak"]),
            elo_rating=float(home.get("elo_rating", ELO_BASELINE)),
            last_game_date=pd.to_datetime(home.get("last_game_date")) if home.get("last_game_date") is not None else None,
            last_is_home=bool(home.get("last_is_home")) if home.get("last_is_home") is not None else None,
            current_venue_streak=int(home.get("current_venue_streak", 0)),
        )
        away_snapshot = TeamSnapshot(
            games_played=int(away["games_played"]),
            season_win_pct=float(away["season_win_pct"]),
            recent_win_pct=float(away["recent_win_pct"]),
            venue_win_pct=float(away["away_win_pct"]),
            run_diff_per_game=float(away["run_diff_per_game"]),
            season_runs_scored_per_game=float(away.get("season_runs_scored_per_game", 0.0)),
            season_runs_allowed_per_game=float(away.get("season_runs_allowed_per_game", 0.0)),
            recent_runs_scored_per_game=float(away.get("recent_runs_scored_per_game", 0.0)),
            recent_runs_allowed_per_game=float(away.get("recent_runs_allowed_per_game", 0.0)),
            streak=int(away["streak"]),
            elo_rating=float(away.get("elo_rating", ELO_BASELINE)),
            last_game_date=pd.to_datetime(away.get("last_game_date")) if away.get("last_game_date") is not None else None,
            last_is_home=bool(away.get("last_is_home")) if away.get("last_is_home") is not None else None,
            current_venue_streak=int(away.get("current_venue_streak", 0)),
        )
        home_offense_vs_opp_hand = offense_split_value(
            matchup_lookup.get(game["home_team"]),
            game.get("away_pitcher_pitcher_hand") or game.get("away_probable_pitcher_hand"),
        )
        away_offense_vs_opp_hand = offense_split_value(
            matchup_lookup.get(game["away_team"]),
            game.get("home_pitcher_pitcher_hand") or game.get("home_probable_pitcher_hand"),
        )
        game_date = pd.to_datetime(game["game_date"])
        is_doubleheader = bool(game.get("game_number") and int(game.get("game_number")) > 1) or str(game.get("doubleheader", "")).upper() not in {"", "N"}
        rows.append(
            {
                "game_id": game["game_id"],
                "snapshot_ts": game["snapshot_ts"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                **_build_game_features(
                    home_snapshot,
                    away_snapshot,
                    _live_pitching_context(game, "home"),
                    _live_pitching_context(game, "away"),
                    game_date=game_date,
                    is_doubleheader=is_doubleheader,
                    market_home_implied_prob=_market_home_implied_prob(market_prior_lookup.get(game["game_id"])),
                    home_offense_vs_opp_hand=home_offense_vs_opp_hand,
                    away_offense_vs_opp_hand=away_offense_vs_opp_hand,
                ),
                "home_season_win_pct": home_snapshot.season_win_pct,
                "away_season_win_pct": away_snapshot.season_win_pct,
                "home_recent_win_pct": home_snapshot.recent_win_pct,
                "away_recent_win_pct": away_snapshot.recent_win_pct,
                "home_venue_win_pct": home_snapshot.venue_win_pct,
                "away_venue_win_pct": away_snapshot.venue_win_pct,
                "home_run_diff_per_game": home_snapshot.run_diff_per_game,
                "away_run_diff_per_game": away_snapshot.run_diff_per_game,
                "home_season_runs_scored_per_game": home_snapshot.season_runs_scored_per_game,
                "away_season_runs_scored_per_game": away_snapshot.season_runs_scored_per_game,
                "home_season_runs_allowed_per_game": home_snapshot.season_runs_allowed_per_game,
                "away_season_runs_allowed_per_game": away_snapshot.season_runs_allowed_per_game,
                "home_recent_runs_scored_per_game": home_snapshot.recent_runs_scored_per_game,
                "away_recent_runs_scored_per_game": away_snapshot.recent_runs_scored_per_game,
                "home_recent_runs_allowed_per_game": home_snapshot.recent_runs_allowed_per_game,
                "away_recent_runs_allowed_per_game": away_snapshot.recent_runs_allowed_per_game,
                "home_rest_days": _rest_days_before_game(home_snapshot.last_game_date, game_date),
                "away_rest_days": _rest_days_before_game(away_snapshot.last_game_date, game_date),
                "home_venue_streak": _venue_streak_before_game(home_snapshot, venue_home=True),
                "away_venue_streak": _venue_streak_before_game(away_snapshot, venue_home=False),
                "home_travel_switch": _travel_switch_before_game(home_snapshot, venue_home=True),
                "away_travel_switch": _travel_switch_before_game(away_snapshot, venue_home=False),
                "is_doubleheader": is_doubleheader,
                "home_streak": home_snapshot.streak,
                "away_streak": away_snapshot.streak,
                "home_elo_rating": home_snapshot.elo_rating,
                "away_elo_rating": away_snapshot.elo_rating,
                "home_starter_era": game.get("home_pitcher_era"),
                "away_starter_era": game.get("away_pitcher_era"),
                "home_starter_whip": game.get("home_pitcher_whip"),
                "away_starter_whip": game.get("away_pitcher_whip"),
                "home_starter_strikeouts_per_9": game.get("home_pitcher_strikeouts_per_9"),
                "away_starter_strikeouts_per_9": game.get("away_pitcher_strikeouts_per_9"),
                "home_starter_walks_per_9": game.get("home_pitcher_walks_per_9"),
                "away_starter_walks_per_9": game.get("away_pitcher_walks_per_9"),
                "home_bullpen_innings_3d": game.get("home_bullpen_innings_3d"),
                "away_bullpen_innings_3d": game.get("away_bullpen_innings_3d"),
                "home_bullpen_pitches_3d": game.get("home_bullpen_pitches_3d"),
                "away_bullpen_pitches_3d": game.get("away_bullpen_pitches_3d"),
                "home_relievers_used_3d": game.get("home_relievers_used_3d"),
                "away_relievers_used_3d": game.get("away_relievers_used_3d"),
                "market_home_implied_prob": _market_home_implied_prob(market_prior_lookup.get(game["game_id"])),
                "home_offense_vs_opp_starter_hand": home_offense_vs_opp_hand,
                "away_offense_vs_opp_starter_hand": away_offense_vs_opp_hand,
            }
        )
    return pd.DataFrame(rows)


def predict_home_win_probabilities(bundle: dict[str, object], features_df: pd.DataFrame) -> pd.DataFrame:
    if features_df.empty:
        return pd.DataFrame()
    _require_feature_columns(features_df)
    results: list[pd.DataFrame] = []

    candidate_pipelines: dict[str, object] = bundle.get("candidate_pipelines") or {}
    if candidate_pipelines:
        for model_name, pipeline in candidate_pipelines.items():
            scored = features_df.copy()
            scored["home_win_prob"] = pipeline.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
            scored["model_name"] = model_name
            results.append(scored)
    else:
        scored = features_df.copy()
        scored["home_win_prob"] = bundle["pipeline"].predict_proba(scored[FEATURE_COLUMNS])[:, 1]
        scored["model_name"] = bundle["model_name"]
        results.append(scored)

    if bundle.get("bayes_evidence_pipeline") is not None and bundle.get("bayes_base_rate") is not None:
        scored = features_df.copy()
        scored["home_win_prob"] = _bayesian_posterior_probabilities(bundle, scored)
        scored["model_name"] = str(bundle.get("bayes_model_name", BAYES_MODEL_NAME))
        results.append(scored)

    return pd.concat(results, ignore_index=True)


def _fit_candidate_models(train_df: pd.DataFrame) -> dict[str, Pipeline]:
    _require_feature_columns(train_df)
    models = {
        LOGISTIC_MODEL_NAME: _build_logistic_pipeline(),
        HISTGB_MODEL_NAME: _build_histgb_pipeline(),
        KNN_MODEL_NAME: _build_knn_pipeline(),
        SVM_MODEL_NAME: _build_svm_pipeline(),
    }
    for pipeline in models.values():
        pipeline.fit(train_df[FEATURE_COLUMNS], train_df["target_home_win"])
    return models


def _train_bayesian_bundle(train_df: pd.DataFrame) -> dict[str, object]:
    _require_feature_columns(train_df)
    base_rate = float(np.clip(train_df["target_home_win"].mean(), 1e-6, 1 - 1e-6))
    pipeline = _build_logistic_pipeline(BAYES_EVIDENCE_FEATURE_COLUMNS)
    pipeline.fit(train_df[BAYES_EVIDENCE_FEATURE_COLUMNS], train_df["target_home_win"])
    return {"pipeline": pipeline, "base_rate": base_rate}


def _bayesian_posterior_probabilities(bundle: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    evidence_pipeline = bundle["bayes_evidence_pipeline"] if "bayes_evidence_pipeline" in bundle else bundle["pipeline"]
    evidence_columns = bundle.get("bayes_evidence_feature_columns", BAYES_EVIDENCE_FEATURE_COLUMNS)
    _require_columns(frame, evidence_columns, frame_name="bayesian evidence frame")
    evidence_probabilities = np.clip(
        np.asarray(evidence_pipeline.predict_proba(frame[evidence_columns]), dtype=float)[:, 1],
        1e-6,
        1 - 1e-6,
    )
    base_rate = float(np.clip(bundle.get("bayes_base_rate", 0.5), 1e-6, 1 - 1e-6))
    prior_probabilities = np.clip(_bayesian_prior_probabilities(frame, default=bundle.get("bayes_prior_default", 0.5)), 1e-6, 1 - 1e-6)

    base_rate_odds = base_rate / (1.0 - base_rate)
    evidence_odds = evidence_probabilities / (1.0 - evidence_probabilities)
    likelihood_ratio = evidence_odds / base_rate_odds
    posterior_odds = (prior_probabilities / (1.0 - prior_probabilities)) * likelihood_ratio
    posterior = posterior_odds / (1.0 + posterior_odds)
    return np.clip(posterior, 1e-6, 1 - 1e-6)


def _bayesian_prior_probabilities(frame: pd.DataFrame, default: float = 0.5) -> np.ndarray:
    _require_columns(frame, ["market_home_implied_prob"], frame_name="bayesian prior frame")
    market_prior = np.clip(frame["market_home_implied_prob"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    neutral_mask = np.isclose(market_prior, default, atol=1e-9)
    if "elo_home_win_prob" not in frame.columns:
        return market_prior
    elo_prior = np.clip(frame["elo_home_win_prob"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    return np.where(neutral_mask, elo_prior, market_prior)


def _forward_select_features(
    train_df: pd.DataFrame,
    criterion: str,
) -> tuple[list[InformationCriteriaResult], InformationCriteriaResult]:
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'.")

    selected_features: list[str] = []
    remaining_features = list(FEATURE_COLUMNS)
    steps: list[InformationCriteriaResult] = []
    best_result: InformationCriteriaResult | None = None
    best_score = np.inf

    while remaining_features:
        candidate_results = []
        for candidate in remaining_features:
            features = [*selected_features, candidate]
            result = _fit_information_criteria_model(train_df, features)
            candidate_results.append(result)

        winning_result = min(candidate_results, key=lambda item: getattr(item, criterion))
        winning_score = getattr(winning_result, criterion)
        if winning_score >= best_score:
            break

        selected_features = list(winning_result.features)
        remaining_features = [feature for feature in remaining_features if feature not in selected_features]
        steps.append(winning_result)
        best_result = winning_result
        best_score = winning_score

    if best_result is None:
        raise ValueError("Unable to fit any forward-selection models.")
    return steps, best_result


def _fit_information_criteria_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> InformationCriteriaResult:
    pipeline = _build_selection_pipeline(feature_columns)
    target = train_df["target_home_win"].astype(int)
    pipeline.fit(train_df[feature_columns], target)
    probabilities = pipeline.predict_proba(train_df[feature_columns])[:, 1]
    log_likelihood = _log_likelihood(target, probabilities)
    parameter_count = len(feature_columns) + 1
    sample_size = len(train_df)
    return InformationCriteriaResult(
        features=tuple(feature_columns),
        log_likelihood=log_likelihood,
        aic=(2.0 * parameter_count) - (2.0 * log_likelihood),
        bic=(np.log(sample_size) * parameter_count) - (2.0 * log_likelihood),
        metrics=_classification_metrics(target, probabilities),
    )


def _build_selection_pipeline(feature_columns: list[str]) -> Pipeline:
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
            ("classifier", LogisticRegression(max_iter=1000, C=np.inf)),
        ]
    )


def _serialize_selection_steps(
    steps: list[InformationCriteriaResult],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> list[dict[str, object]]:
    return [
        {
            "step": index,
            **_serialize_information_result(step_result, train_df, valid_df),
        }
        for index, step_result in enumerate(steps, start=1)
    ]


def _serialize_information_result(
    result: InformationCriteriaResult,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> dict[str, object]:
    validation_metrics = _evaluate_feature_subset(train_df, valid_df, list(result.features))
    return {
        "features": list(result.features),
        "feature_count": len(result.features),
        "log_likelihood": result.log_likelihood,
        "aic": result.aic,
        "bic": result.bic,
        "train_metrics": result.metrics,
        "valid_metrics": validation_metrics,
    }


def _evaluate_feature_subset(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    if valid_df.empty:
        return {}
    pipeline = _build_selection_pipeline(feature_columns)
    train_target = train_df["target_home_win"].astype(int)
    target = valid_df["target_home_win"].astype(int)
    pipeline.fit(train_df[feature_columns], train_target)
    probabilities = pipeline.predict_proba(valid_df[feature_columns])[:, 1]
    return _classification_metrics(target, probabilities)


def _build_logistic_pipeline(feature_columns: list[str] | None = None) -> Pipeline:
    columns = feature_columns or FEATURE_COLUMNS
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
                              columns,
                          )
                      ]
                  ),
              ),
              ("classifier", LogisticRegression(max_iter=1000)),
          ]
      )


def _build_histgb_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("numeric", Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]), FEATURE_COLUMNS)
                    ]
                ),
            ),
            ("classifier", GradientBoostingClassifier(learning_rate=0.05, n_estimators=250, max_depth=3, random_state=42)),
        ]
    )


def _build_knn_pipeline() -> Pipeline:
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
                            FEATURE_COLUMNS,
                        )
                    ]
                ),
            ),
            ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]
    )


def _build_svm_pipeline() -> Pipeline:
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
                            FEATURE_COLUMNS,
                        )
                    ]
                ),
            ),
            ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
        ]
    )


def _split_training_data(training_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = training_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    split_index = int(len(ordered) * 0.8)
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Training dataset is too small for a time-based split.")
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def _require_feature_columns(frame: pd.DataFrame) -> None:
    _require_columns(frame, FEATURE_COLUMNS, frame_name="feature frame")


def _require_columns(frame: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    duplicate_columns = sorted({column for column in columns if columns.count(column) > 1})
    if duplicate_columns:
        raise ValueError(f"Duplicate required columns in {frame_name}: {', '.join(duplicate_columns)}")
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {frame_name}: {', '.join(missing_columns)}")


def _classification_metrics(target: pd.Series, probabilities: np.ndarray | pd.Series) -> dict[str, float]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    predictions = (clipped >= 0.5).astype(int)
    target_array = target.to_numpy(dtype=int)
    if pd.Series(target_array).nunique() < 2:
        actual = float(target_array[0]) if len(target_array) else 0.0
        return {
            "accuracy": float((predictions == target_array).mean()) if len(target_array) else 0.0,
            "roc_auc": 0.5,
            "log_loss": float(log_loss(target_array, clipped, labels=[0, 1])) if len(target_array) else 0.0,
            "brier_score": float(np.mean(np.square(clipped - actual))) if len(target_array) else 0.0,
        }
    return {
        "accuracy": float((predictions == target_array).mean()),
        "roc_auc": float(roc_auc_score(target_array, clipped)),
        "log_loss": float(log_loss(target_array, clipped)),
        "brier_score": float(brier_score_loss(target_array, clipped)),
    }


def _log_likelihood(target: pd.Series, probabilities: np.ndarray | pd.Series) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    target_array = target.to_numpy(dtype=float)
    return float(np.sum((target_array * np.log(clipped)) + ((1.0 - target_array) * np.log(1.0 - clipped))))


def _select_model(candidate_metrics: dict[str, dict[str, float]]) -> str:
    metric = settings().model_selection_metric.lower().strip()
    if metric == "accuracy":
        return max(candidate_metrics, key=lambda name: (candidate_metrics[name]["accuracy"], candidate_metrics[name]["roc_auc"]))
    if metric == "roc_auc":
        return max(candidate_metrics, key=lambda name: (candidate_metrics[name]["roc_auc"], candidate_metrics[name]["accuracy"]))
    if metric == "brier_score":
        return min(candidate_metrics, key=lambda name: (candidate_metrics[name]["brier_score"], -candidate_metrics[name]["roc_auc"]))
    return min(candidate_metrics, key=lambda name: (candidate_metrics[name]["log_loss"], -candidate_metrics[name]["roc_auc"]))


def _calibration_table(target: pd.Series, probabilities: np.ndarray, model_name: str, bins: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame({"target": target.to_numpy(), "predicted_prob": np.clip(probabilities, 1e-6, 1 - 1e-6)})
    frame["bucket"] = pd.cut(frame["predicted_prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    calibration = (
        frame.groupby("bucket", observed=False)
        .agg(rows=("target", "size"), avg_predicted_prob=("predicted_prob", "mean"), actual_home_win_rate=("target", "mean"))
        .reset_index()
    )
    calibration["model_name"] = model_name
    calibration["bucket"] = calibration["bucket"].astype(str)
    return calibration


def _build_game_features(
    home_snapshot: TeamSnapshot,
    away_snapshot: TeamSnapshot,
    home_pitching: dict[str, float | None],
    away_pitching: dict[str, float | None],
    game_date: pd.Timestamp,
    is_doubleheader: bool,
    market_home_implied_prob: float,
    home_offense_vs_opp_hand: float,
    away_offense_vs_opp_hand: float,
) -> dict[str, float]:
    elo_diff = home_snapshot.elo_rating - away_snapshot.elo_rating
    elo_home_win_prob = _elo_win_prob(home_snapshot.elo_rating, away_snapshot.elo_rating)
    home_rest_days = _rest_days_before_game(home_snapshot.last_game_date, game_date)
    away_rest_days = _rest_days_before_game(away_snapshot.last_game_date, game_date)
    home_venue_streak = _venue_streak_before_game(home_snapshot, venue_home=True)
    away_venue_streak = _venue_streak_before_game(away_snapshot, venue_home=False)
    home_travel_switch = _travel_switch_before_game(home_snapshot, venue_home=True)
    away_travel_switch = _travel_switch_before_game(away_snapshot, venue_home=False)
    return {
        "season_win_pct_diff": home_snapshot.season_win_pct - away_snapshot.season_win_pct,
        "recent_win_pct_diff": home_snapshot.recent_win_pct - away_snapshot.recent_win_pct,
        "venue_win_pct_diff": home_snapshot.venue_win_pct - away_snapshot.venue_win_pct,
        "run_diff_per_game_diff": home_snapshot.run_diff_per_game - away_snapshot.run_diff_per_game,
        "season_runs_scored_per_game_diff": home_snapshot.season_runs_scored_per_game - away_snapshot.season_runs_scored_per_game,
        "season_runs_allowed_per_game_adv": away_snapshot.season_runs_allowed_per_game - home_snapshot.season_runs_allowed_per_game,
        "recent_runs_scored_per_game_diff": home_snapshot.recent_runs_scored_per_game - away_snapshot.recent_runs_scored_per_game,
        "recent_runs_allowed_per_game_adv": away_snapshot.recent_runs_allowed_per_game - home_snapshot.recent_runs_allowed_per_game,
        "rest_days_diff": home_rest_days - away_rest_days,
        "venue_streak_diff": home_venue_streak - away_venue_streak,
        "travel_switch_adv": away_travel_switch - home_travel_switch,
        "doubleheader_flag": float(is_doubleheader),
        "streak_diff": float(home_snapshot.streak - away_snapshot.streak),
        "elo_diff": elo_diff,
        "elo_home_win_prob": elo_home_win_prob,
        "market_home_implied_prob": market_home_implied_prob,
        "offense_vs_starter_hand_diff": home_offense_vs_opp_hand - away_offense_vs_opp_hand,
        "starter_era_adv": _diff_positive_for_home(away_pitching.get("era"), home_pitching.get("era")),
        "starter_whip_adv": _diff_positive_for_home(away_pitching.get("whip"), home_pitching.get("whip")),
        "starter_strikeouts_per_9_diff": _float_or_zero(home_pitching.get("strikeouts_per_9")) - _float_or_zero(away_pitching.get("strikeouts_per_9")),
        "starter_walks_per_9_adv": _diff_positive_for_home(away_pitching.get("walks_per_9"), home_pitching.get("walks_per_9")),
        "bullpen_innings_3d_adv": _diff_positive_for_home(away_pitching.get("bullpen_innings_3d"), home_pitching.get("bullpen_innings_3d")),
        "bullpen_pitches_3d_adv": _diff_positive_for_home(away_pitching.get("bullpen_pitches_3d"), home_pitching.get("bullpen_pitches_3d")),
        "relievers_used_3d_adv": _diff_positive_for_home(away_pitching.get("relievers_used_3d"), home_pitching.get("relievers_used_3d")),
    }


def _snapshot_team(history: list[dict[str, object]], venue_home: bool, elo_rating: float) -> TeamSnapshot:
    history = sorted(history, key=lambda game: (pd.Timestamp(game["game_date"]), str(game.get("game_id", ""))))
    games_played = len(history)
    wins = sum(1 for game in history if game["won"])
    recent = history[-RECENT_GAMES_WINDOW:]
    recent_wins = sum(1 for game in recent if game["won"])
    venue_games = [game for game in history if bool(game["is_home"]) is venue_home]
    venue_wins = sum(1 for game in venue_games if game["won"])
    run_diff_total = sum(int(game["run_diff"]) for game in history)
    run_diff_per_game = (run_diff_total / games_played) if games_played else 0.0
    return TeamSnapshot(
        games_played=games_played,
        season_win_pct=smoothed_win_pct(wins, games_played),
        recent_win_pct=smoothed_win_pct(recent_wins, len(recent)),
        venue_win_pct=smoothed_win_pct(venue_wins, len(venue_games)),
        run_diff_per_game=run_diff_per_game,
        season_runs_scored_per_game=(sum(int(game["runs_for"]) for game in history) / games_played) if games_played else 0.0,
        season_runs_allowed_per_game=(sum(int(game["runs_against"]) for game in history) / games_played) if games_played else 0.0,
        recent_runs_scored_per_game=(sum(int(game["runs_for"]) for game in recent) / len(recent)) if recent else 0.0,
        recent_runs_allowed_per_game=(sum(int(game["runs_against"]) for game in recent) / len(recent)) if recent else 0.0,
        streak=current_streak([bool(game["won"]) for game in history]),
        elo_rating=elo_rating,
        last_game_date=pd.to_datetime(history[-1]["game_date"]) if history else None,
        last_is_home=bool(history[-1]["is_home"]) if history else None,
        current_venue_streak=_current_venue_streak(history),
    )


def _pregame_pitching_context(
    game_id: str,
    team: str,
    pitching_lookup: dict[tuple[str, str], dict[str, object]],
    pitcher_history: dict[int, list[dict[str, float]]],
    team_pitching_history: dict[str, list[dict[str, object]]],
) -> dict[str, float | None]:
    context = {
        "era": None,
        "whip": None,
        "strikeouts_per_9": None,
        "walks_per_9": None,
        "pitcher_hand": None,
        "bullpen_innings_3d": 0.0,
        "bullpen_pitches_3d": 0.0,
        "relievers_used_3d": 0.0,
    }
    game_pitching = pitching_lookup.get((game_id, team))
    if game_pitching is not None:
        context["pitcher_hand"] = game_pitching.get("starting_pitcher_hand")
        starter_id = game_pitching.get("starting_pitcher_id")
        if starter_id is not None and not pd.isna(starter_id):
            recent_starts = pitcher_history.get(int(starter_id), [])[-STARTER_LOOKBACK_GAMES:]
            if recent_starts:
                innings = sum(float(start["innings_pitched"]) for start in recent_starts)
                earned_runs = sum(float(start["earned_runs"]) for start in recent_starts)
                hits = sum(float(start["hits"]) for start in recent_starts)
                walks = sum(float(start["walks"]) for start in recent_starts)
                strikeouts = sum(float(start["strikeouts"]) for start in recent_starts)
                if innings > 0:
                    context["era"] = earned_runs * 9.0 / innings
                    context["whip"] = (hits + walks) / innings
                    context["strikeouts_per_9"] = strikeouts * 9.0 / innings
                    context["walks_per_9"] = walks * 9.0 / innings
        recent_team_pitching = team_pitching_history.get(team, [])[-3:]
        if recent_team_pitching:
            context["bullpen_innings_3d"] = sum(float(item["bullpen_innings"]) for item in recent_team_pitching)
            context["bullpen_pitches_3d"] = sum(float(item["bullpen_pitches"]) for item in recent_team_pitching)
            context["relievers_used_3d"] = sum(float(item["relievers_used"]) for item in recent_team_pitching)
    return context


def _live_pitching_context(game: dict[str, object], side: str) -> dict[str, object]:
    raw_hand = game.get(f"{side}_pitcher_pitcher_hand") or game.get(f"{side}_probable_pitcher_hand")
    return {
        "era": _optional_float(game.get(f"{side}_pitcher_era")),
        "whip": _optional_float(game.get(f"{side}_pitcher_whip")),
        "strikeouts_per_9": _optional_float(game.get(f"{side}_pitcher_strikeouts_per_9")),
        "walks_per_9": _optional_float(game.get(f"{side}_pitcher_walks_per_9")),
        "pitcher_hand": normalize_pitcher_hand(raw_hand),
        "bullpen_innings_3d": _optional_float(game.get(f"{side}_bullpen_innings_3d")) or 0.0,
        "bullpen_pitches_3d": _optional_float(game.get(f"{side}_bullpen_pitches_3d")) or 0.0,
        "relievers_used_3d": _optional_float(game.get(f"{side}_relievers_used_3d")) or 0.0,
    }


def _update_histories(
    game: dict[str, object],
    pitching_lookup: dict[tuple[str, str], dict[str, object]],
    batting_lookup: dict[tuple[str, str], dict[str, object]],
    team_history: dict[str, list[dict[str, object]]],
    pitcher_history: dict[int, list[dict[str, float]]],
    team_pitching_history: dict[str, list[dict[str, object]]],
    team_offense_hand_history: dict[str, list[dict[str, object]]],
) -> None:
    away_won = game["away_score"] > game["home_score"]
    home_won = game["home_score"] > game["away_score"]
    team_history.setdefault(game["away_team"], []).append(
        {
            "game_id": game["game_id"],
            "game_date": game["game_date"],
            "is_home": False,
            "won": away_won,
            "run_diff": int(game["away_score"]) - int(game["home_score"]),
            "runs_for": int(game["away_score"]),
            "runs_against": int(game["home_score"]),
        }
    )
    team_history.setdefault(game["home_team"], []).append(
        {
            "game_id": game["game_id"],
            "game_date": game["game_date"],
            "is_home": True,
            "won": home_won,
            "run_diff": int(game["home_score"]) - int(game["away_score"]),
            "runs_for": int(game["home_score"]),
            "runs_against": int(game["away_score"]),
        }
    )
    for team in (game["away_team"], game["home_team"]):
        pitching = pitching_lookup.get((game["game_id"], team))
        if pitching is None:
            continue
        starter_id = pitching.get("starting_pitcher_id")
        if starter_id is not None and not pd.isna(starter_id):
            pitcher_history.setdefault(int(starter_id), []).append(
                {
                    "innings_pitched": _float_or_zero(pitching.get("starter_innings_pitched")),
                    "earned_runs": _float_or_zero(pitching.get("starter_earned_runs")),
                    "hits": _float_or_zero(pitching.get("starter_hits")),
                    "walks": _float_or_zero(pitching.get("starter_walks")),
                    "strikeouts": _float_or_zero(pitching.get("starter_strikeouts")),
                }
            )
        team_pitching_history.setdefault(team, []).append(
            {
                "game_date": game["game_date"],
                "bullpen_innings": _float_or_zero(pitching.get("bullpen_innings")),
                "bullpen_pitches": _float_or_zero(pitching.get("bullpen_pitches")),
                "relievers_used": _float_or_zero(pitching.get("relievers_used")),
            }
        )
    home_pitching = pitching_lookup.get((game["game_id"], game["home_team"]), {})
    away_pitching = pitching_lookup.get((game["game_id"], game["away_team"]), {})
    away_batting = batting_lookup.get((game["game_id"], game["away_team"]), {})
    home_batting = batting_lookup.get((game["game_id"], game["home_team"]), {})
    team_offense_hand_history.setdefault(game["away_team"], []).append(
        {
            "game_date": game["game_date"],
            "opponent_starter_hand": home_pitching.get("starting_pitcher_hand"),
            "offense_score": offense_score_from_stats(away_batting),
        }
    )
    team_offense_hand_history.setdefault(game["home_team"], []).append(
        {
            "game_date": game["game_date"],
            "opponent_starter_hand": away_pitching.get("starting_pitcher_hand"),
            "offense_score": offense_score_from_stats(home_batting),
        }
    )


def _update_elo(game: dict[str, object], elo_ratings: dict[str, float]) -> None:
    away_team = game["away_team"]
    home_team = game["home_team"]
    away_elo = elo_ratings.get(away_team, ELO_BASELINE)
    home_elo = elo_ratings.get(home_team, ELO_BASELINE)
    home_expected = _elo_win_prob(home_elo, away_elo)
    home_actual = 1.0 if game["home_score"] > game["away_score"] else 0.0
    away_actual = 1.0 - home_actual
    elo_ratings[home_team] = home_elo + ELO_K_FACTOR * (home_actual - home_expected)
    elo_ratings[away_team] = away_elo + ELO_K_FACTOR * (away_actual - (1.0 - home_expected))


def _elo_win_prob(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _rest_days_before_game(last_game_date: pd.Timestamp | None, game_date: pd.Timestamp) -> float:
    if last_game_date is None or pd.isna(last_game_date):
        return 0.0
    return float(max((game_date.normalize() - pd.Timestamp(last_game_date).normalize()).days - 1, 0))


def _current_venue_streak(history: list[dict[str, object]]) -> int:
    if not history:
        return 0
    last_is_home = bool(history[-1]["is_home"])
    streak = 0
    for game in reversed(history):
        if bool(game["is_home"]) != last_is_home:
            break
        streak += 1
    return streak


def _venue_streak_before_game(snapshot: TeamSnapshot, venue_home: bool) -> float:
    if snapshot.last_is_home is None:
        return 0.0
    return float(snapshot.current_venue_streak if snapshot.last_is_home == venue_home else 0)


def _travel_switch_before_game(snapshot: TeamSnapshot, venue_home: bool) -> float:
    if snapshot.last_is_home is None:
        return 0.0
    return float(snapshot.last_is_home != venue_home)


def _team_games_on_date(history: list[dict[str, object]], game_date: pd.Timestamp) -> int:
    normalized = game_date.normalize()
    return sum(1 for game in history if pd.Timestamp(game["game_date"]).normalize() == normalized)


def _optional_float(value: object) -> float | None:
    return coalesce_float(value)


def _float_or_zero(value: object) -> float:
    return coalesce_float(value) or 0.0


def _diff_positive_for_home(away_value: object, home_value: object) -> float:
    return _float_or_zero(away_value) - _float_or_zero(home_value)


def _offense_vs_hand_before_game(history: list[dict[str, object]], pitcher_hand: object) -> float:
    if not history:
        return OFFENSE_SPLIT_PRIOR
    hand = normalize_pitcher_hand(pitcher_hand)
    matching = [
        float(item["offense_score"])
        for item in history
        if str(item.get("opponent_starter_hand", "")).strip().upper().startswith(hand)
    ]
    if not matching:
        return OFFENSE_SPLIT_PRIOR
    return float((sum(matching) + (OFFENSE_SPLIT_PRIOR * 10.0)) / (len(matching) + 10.0))


def _market_home_implied_prob(value: object) -> float:
    if value is None or pd.isna(value):
        logger.debug("market_home_implied_prob is None/NaN; defaulting to 0.5")
        return 0.5
    clipped = float(min(max(value, 1e-6), 1 - 1e-6))
    if clipped != float(value):
        logger.debug("market_home_implied_prob clipped from %s to %s", value, clipped)
    return clipped


def _load_historical_market_priors(start_date: str, end_date: str) -> pd.DataFrame:
    db_path = settings().duckdb_path
    if not db_path.exists():
        return pd.DataFrame(columns=["game_id", "market_home_implied_prob", "market_source_count"])

    conn = connect(db_path)
    try:
        return query_dataframe(
            conn,
            f"""
            SELECT
                q.game_id,
                AVG(CASE WHEN q.outcome_team = g.home_team THEN q.fair_prob END) AS market_home_implied_prob,
                COUNT(DISTINCT q.source) AS market_source_count
            FROM normalized_quotes_deduped q
            JOIN games_deduped g ON g.game_id = q.game_id
            WHERE g.game_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
              AND q.is_valid = TRUE
              AND q.is_pregame = TRUE
            GROUP BY q.game_id
            HAVING AVG(CASE WHEN q.outcome_team = g.home_team THEN q.fair_prob END) IS NOT NULL
            """
        )
    finally:
        conn.close()
