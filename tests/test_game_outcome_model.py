import logging
import pickle
from pathlib import Path

import pandas as pd
import pytest

from mlpm.models.game_outcome import (
    BAYES_MODEL_NAME,
    FEATURE_COLUMNS,
    HISTGB_MODEL_NAME,
    KNN_MODEL_NAME,
    LOGISTIC_MODEL_NAME,
    SVM_MODEL_NAME,
    analyze_forward_feature_selection,
    benchmark_game_outcome_models,
    build_training_dataset,
    load_trained_model,
    predict_home_win_probabilities,
    run_model_benchmark,
    save_trained_model,
    train_game_outcome_model,
)


def test_build_training_dataset_creates_pregame_feature_rows() -> None:
    results = pd.DataFrame(
        [
            {"game_id": "1", "game_date": "2026-04-01", "away_team": "A", "home_team": "B", "winner_team": "A", "away_score": 5, "home_score": 3},
            {"game_id": "2", "game_date": "2026-04-01", "away_team": "C", "home_team": "D", "winner_team": "D", "away_score": 2, "home_score": 4},
            {"game_id": "3", "game_date": "2026-04-02", "away_team": "A", "home_team": "C", "winner_team": "A", "away_score": 6, "home_score": 1},
            {"game_id": "4", "game_date": "2026-04-02", "away_team": "B", "home_team": "D", "winner_team": "D", "away_score": 1, "home_score": 3},
            {"game_id": "5", "game_date": "2026-04-03", "away_team": "D", "home_team": "A", "winner_team": "A", "away_score": 2, "home_score": 5},
            {"game_id": "6", "game_date": "2026-04-03", "away_team": "C", "home_team": "B", "winner_team": "B", "away_score": 2, "home_score": 6},
        ]
    )
    pitching_logs = pd.DataFrame(
        [
            {"game_id": "1", "game_date": "2026-04-01", "team": "A", "starting_pitcher_id": 101, "starter_innings_pitched": 6.0, "starter_earned_runs": 2, "starter_hits": 4, "starter_walks": 1, "starter_strikeouts": 7, "bullpen_innings": 3.0, "bullpen_pitches": 42, "relievers_used": 3},
            {"game_id": "1", "game_date": "2026-04-01", "team": "B", "starting_pitcher_id": 201, "starting_pitcher_hand": "L", "starter_innings_pitched": 5.0, "starter_earned_runs": 4, "starter_hits": 6, "starter_walks": 2, "starter_strikeouts": 5, "bullpen_innings": 4.0, "bullpen_pitches": 58, "relievers_used": 4},
            {"game_id": "2", "game_date": "2026-04-01", "team": "C", "starting_pitcher_id": 301, "starting_pitcher_hand": "R", "starter_innings_pitched": 5.0, "starter_earned_runs": 3, "starter_hits": 5, "starter_walks": 2, "starter_strikeouts": 4, "bullpen_innings": 4.0, "bullpen_pitches": 61, "relievers_used": 4},
            {"game_id": "2", "game_date": "2026-04-01", "team": "D", "starting_pitcher_id": 401, "starting_pitcher_hand": "L", "starter_innings_pitched": 7.0, "starter_earned_runs": 2, "starter_hits": 5, "starter_walks": 1, "starter_strikeouts": 8, "bullpen_innings": 2.0, "bullpen_pitches": 28, "relievers_used": 2},
            {"game_id": "3", "game_date": "2026-04-02", "team": "A", "starting_pitcher_id": 101, "starting_pitcher_hand": "R", "starter_innings_pitched": 6.0, "starter_earned_runs": 1, "starter_hits": 4, "starter_walks": 1, "starter_strikeouts": 8, "bullpen_innings": 3.0, "bullpen_pitches": 39, "relievers_used": 3},
            {"game_id": "3", "game_date": "2026-04-02", "team": "C", "starting_pitcher_id": 301, "starting_pitcher_hand": "R", "starter_innings_pitched": 4.0, "starter_earned_runs": 5, "starter_hits": 7, "starter_walks": 3, "starter_strikeouts": 4, "bullpen_innings": 5.0, "bullpen_pitches": 73, "relievers_used": 5},
            {"game_id": "4", "game_date": "2026-04-02", "team": "B", "starting_pitcher_id": 201, "starting_pitcher_hand": "L", "starter_innings_pitched": 5.0, "starter_earned_runs": 3, "starter_hits": 6, "starter_walks": 2, "starter_strikeouts": 5, "bullpen_innings": 4.0, "bullpen_pitches": 54, "relievers_used": 4},
            {"game_id": "4", "game_date": "2026-04-02", "team": "D", "starting_pitcher_id": 401, "starting_pitcher_hand": "L", "starter_innings_pitched": 6.0, "starter_earned_runs": 1, "starter_hits": 4, "starter_walks": 1, "starter_strikeouts": 7, "bullpen_innings": 3.0, "bullpen_pitches": 40, "relievers_used": 3},
            {"game_id": "5", "game_date": "2026-04-03", "team": "D", "starting_pitcher_id": 401, "starting_pitcher_hand": "L", "starter_innings_pitched": 5.0, "starter_earned_runs": 3, "starter_hits": 5, "starter_walks": 2, "starter_strikeouts": 6, "bullpen_innings": 4.0, "bullpen_pitches": 56, "relievers_used": 4},
            {"game_id": "5", "game_date": "2026-04-03", "team": "A", "starting_pitcher_id": 101, "starting_pitcher_hand": "R", "starter_innings_pitched": 7.0, "starter_earned_runs": 2, "starter_hits": 5, "starter_walks": 1, "starter_strikeouts": 9, "bullpen_innings": 2.0, "bullpen_pitches": 27, "relievers_used": 2},
            {"game_id": "6", "game_date": "2026-04-03", "team": "C", "starting_pitcher_id": 301, "starting_pitcher_hand": "R", "starter_innings_pitched": 5.0, "starter_earned_runs": 4, "starter_hits": 7, "starter_walks": 2, "starter_strikeouts": 5, "bullpen_innings": 4.0, "bullpen_pitches": 55, "relievers_used": 4},
            {"game_id": "6", "game_date": "2026-04-03", "team": "B", "starting_pitcher_id": 201, "starting_pitcher_hand": "L", "starter_innings_pitched": 6.0, "starter_earned_runs": 2, "starter_hits": 5, "starter_walks": 1, "starter_strikeouts": 7, "bullpen_innings": 3.0, "bullpen_pitches": 37, "relievers_used": 3},
        ]
    )

    training = build_training_dataset(results, pitching_logs, min_games=1)

    assert len(training) == 4
    assert set(FEATURE_COLUMNS).issubset(training.columns)
    assert training["elo_diff"].notna().all()
    assert training["rest_days_diff"].notna().all()
    assert training["season_runs_scored_per_game_diff"].notna().all()
    assert training["market_home_implied_prob"].eq(0.5).all()
    assert training["offense_vs_starter_hand_diff"].notna().all()
    assert training["starter_era_adv"].notna().any()
    assert training["bullpen_innings_3d_adv"].notna().all()


def test_train_game_outcome_model_and_reload() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "game_id": str(index),
                "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=index),
                "target_home_win": index % 2,
                "season_win_pct_diff": 0.10 if index % 2 else -0.10,
                "recent_win_pct_diff": 0.12 if index % 2 else -0.12,
                "venue_win_pct_diff": 0.08 if index % 2 else -0.08,
                "run_diff_per_game_diff": 0.9 if index % 2 else -0.9,
                "season_runs_scored_per_game_diff": 0.6 if index % 2 else -0.6,
                "season_runs_allowed_per_game_adv": 0.5 if index % 2 else -0.5,
                "recent_runs_scored_per_game_diff": 0.7 if index % 2 else -0.7,
                "recent_runs_allowed_per_game_adv": 0.6 if index % 2 else -0.6,
                "rest_days_diff": 1.0 if index % 2 else -1.0,
                "venue_streak_diff": 2.0 if index % 2 else -2.0,
                "travel_switch_adv": 1.0 if index % 2 else -1.0,
                "doubleheader_flag": 0.0,
                "streak_diff": 2.0 if index % 2 else -2.0,
                "elo_diff": 25.0 if index % 2 else -25.0,
                "elo_home_win_prob": 0.58 if index % 2 else 0.42,
                "market_home_implied_prob": 0.57 if index % 2 else 0.43,
                "offense_vs_starter_hand_diff": 0.4 if index % 2 else -0.4,
                "starter_era_adv": 1.2 if index % 2 else -1.2,
                "starter_whip_adv": 0.3 if index % 2 else -0.3,
                "starter_strikeouts_per_9_diff": 1.5 if index % 2 else -1.5,
                "starter_walks_per_9_adv": 0.8 if index % 2 else -0.8,
                "bullpen_innings_3d_adv": 2.0 if index % 2 else -2.0,
                "bullpen_pitches_3d_adv": 30.0 if index % 2 else -30.0,
                "relievers_used_3d_adv": 2.0 if index % 2 else -2.0,
            }
        )
    training_df = pd.DataFrame(rows)

    bundle = train_game_outcome_model(training_df)
    artifact_dir = Path("data/processed/test_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    saved_path = artifact_dir / "game_outcome_model.pkl"
    try:
        saved_path.unlink(missing_ok=True)
        saved_path = save_trained_model(bundle, saved_path)
        loaded = load_trained_model(saved_path)
        scored = predict_home_win_probabilities(loaded, training_df.iloc[:3].copy())

        assert loaded is not None
        assert saved_path.exists()
        assert "home_win_prob" in scored.columns
        assert scored["home_win_prob"].between(0, 1).all()
        assert scored["model_name"].eq(BAYES_MODEL_NAME).all()
        assert "bayes_evidence_pipeline" in bundle
        assert "roc_auc" in bundle["candidate_metrics"][KNN_MODEL_NAME]
        assert "log_loss" in bundle["candidate_metrics"][KNN_MODEL_NAME]
        assert "roc_auc" in bundle["candidate_metrics"][SVM_MODEL_NAME]
        assert "log_loss" in bundle["candidate_metrics"][SVM_MODEL_NAME]
    finally:
        saved_path.unlink(missing_ok=True)


def test_load_trained_model_supports_legacy_pickle_artifact() -> None:
    artifact_dir = Path("data/processed/test_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    saved_path = artifact_dir / "legacy_game_outcome_model.pkl"
    payload = {"model_name": "legacy"}
    try:
        saved_path.unlink(missing_ok=True)
        with saved_path.open("wb") as handle:
            pickle.dump(payload, handle)
        loaded = load_trained_model(saved_path)
        assert loaded == payload
    finally:
        saved_path.unlink(missing_ok=True)


def test_build_training_dataset_logs_skipped_min_games(caplog) -> None:
    results = pd.DataFrame(
        [
            {"game_id": "1", "game_date": "2026-04-01", "away_team": "A", "home_team": "B", "winner_team": "A", "away_score": 5, "home_score": 3},
            {"game_id": "2", "game_date": "2026-04-02", "away_team": "A", "home_team": "B", "winner_team": "B", "away_score": 2, "home_score": 4},
        ]
    )
    pitching_logs = pd.DataFrame()

    with caplog.at_level(logging.WARNING):
        training = build_training_dataset(results, pitching_logs, min_games=5)

    assert training.empty
    assert "Training dataset dropped 2 games below MODEL_MIN_GAMES" in caplog.text


def test_train_game_outcome_model_raises_on_missing_feature_columns() -> None:
    rows = []
    for index in range(12):
        row = {
            "game_id": str(index),
            "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=index),
            "target_home_win": index % 2,
        }
        for feature in FEATURE_COLUMNS:
            row[feature] = 0.1 if index % 2 else -0.1
        rows.append(row)
    training_df = pd.DataFrame(rows).drop(columns=["elo_diff"])

    with pytest.raises(ValueError, match="Missing required columns in feature frame: elo_diff"):
        train_game_outcome_model(training_df)


def test_benchmark_game_outcome_models_returns_baselines() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "game_id": str(index),
                "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=index),
                "target_home_win": index % 2,
                "season_win_pct_diff": 0.10 if index % 2 else -0.10,
                "recent_win_pct_diff": 0.12 if index % 2 else -0.12,
                "venue_win_pct_diff": 0.08 if index % 2 else -0.08,
                "run_diff_per_game_diff": 0.9 if index % 2 else -0.9,
                "season_runs_scored_per_game_diff": 0.6 if index % 2 else -0.6,
                "season_runs_allowed_per_game_adv": 0.5 if index % 2 else -0.5,
                "recent_runs_scored_per_game_diff": 0.7 if index % 2 else -0.7,
                "recent_runs_allowed_per_game_adv": 0.6 if index % 2 else -0.6,
                "rest_days_diff": 1.0 if index % 2 else -1.0,
                "venue_streak_diff": 2.0 if index % 2 else -2.0,
                "travel_switch_adv": 1.0 if index % 2 else -1.0,
                "doubleheader_flag": 0.0,
                "streak_diff": 2.0 if index % 2 else -2.0,
                "elo_diff": 25.0 if index % 2 else -25.0,
                "elo_home_win_prob": 0.58 if index % 2 else 0.42,
                "market_home_implied_prob": 0.57 if index % 2 else 0.43,
                "offense_vs_starter_hand_diff": 0.4 if index % 2 else -0.4,
                "starter_era_adv": 1.2 if index % 2 else -1.2,
                "starter_whip_adv": 0.3 if index % 2 else -0.3,
                "starter_strikeouts_per_9_diff": 1.5 if index % 2 else -1.5,
                "starter_walks_per_9_adv": 0.8 if index % 2 else -0.8,
                "bullpen_innings_3d_adv": 2.0 if index % 2 else -2.0,
                "bullpen_pitches_3d_adv": 30.0 if index % 2 else -30.0,
                "relievers_used_3d_adv": 2.0 if index % 2 else -2.0,
            }
        )
    training_df = pd.DataFrame(rows)
    split_index = int(len(training_df) * 0.8)
    train_df = training_df.iloc[:split_index].copy()
    valid_df = training_df.iloc[split_index:].copy()
    bundle = train_game_outcome_model(training_df)

    benchmark = benchmark_game_outcome_models(train_df, valid_df)

    assert LOGISTIC_MODEL_NAME in benchmark["benchmarks"]
    assert HISTGB_MODEL_NAME in benchmark["benchmarks"]
    assert KNN_MODEL_NAME in benchmark["benchmarks"]
    assert SVM_MODEL_NAME in benchmark["benchmarks"]
    assert BAYES_MODEL_NAME in benchmark["benchmarks"]
    assert "always_home" in benchmark["benchmarks"]
    assert "heuristic_feature_score" in benchmark["benchmarks"]
    assert "roc_auc" in benchmark["benchmarks"][KNN_MODEL_NAME]
    assert "log_loss" in benchmark["benchmarks"][KNN_MODEL_NAME]
    assert "roc_auc" in benchmark["benchmarks"][SVM_MODEL_NAME]
    assert "log_loss" in benchmark["benchmarks"][SVM_MODEL_NAME]
    assert len(benchmark["calibration"]) > 0


def test_predict_home_win_probabilities_uses_bayesian_prior_update() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "game_id": str(index),
                "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=index),
                "target_home_win": index % 2,
                "season_win_pct_diff": 0.10 if index % 2 else -0.10,
                "recent_win_pct_diff": 0.12 if index % 2 else -0.12,
                "venue_win_pct_diff": 0.08 if index % 2 else -0.08,
                "run_diff_per_game_diff": 0.9 if index % 2 else -0.9,
                "season_runs_scored_per_game_diff": 0.6 if index % 2 else -0.6,
                "season_runs_allowed_per_game_adv": 0.5 if index % 2 else -0.5,
                "recent_runs_scored_per_game_diff": 0.7 if index % 2 else -0.7,
                "recent_runs_allowed_per_game_adv": 0.6 if index % 2 else -0.6,
                "rest_days_diff": 1.0 if index % 2 else -1.0,
                "venue_streak_diff": 2.0 if index % 2 else -2.0,
                "travel_switch_adv": 1.0 if index % 2 else -1.0,
                "doubleheader_flag": 0.0,
                "streak_diff": 2.0 if index % 2 else -2.0,
                "elo_diff": 25.0 if index % 2 else -25.0,
                "elo_home_win_prob": 0.58 if index % 2 else 0.42,
                "market_home_implied_prob": 0.57 if index % 2 else 0.43,
                "offense_vs_starter_hand_diff": 0.4 if index % 2 else -0.4,
                "starter_era_adv": 1.2 if index % 2 else -1.2,
                "starter_whip_adv": 0.3 if index % 2 else -0.3,
                "starter_strikeouts_per_9_diff": 1.5 if index % 2 else -1.5,
                "starter_walks_per_9_adv": 0.8 if index % 2 else -0.8,
                "bullpen_innings_3d_adv": 2.0 if index % 2 else -2.0,
                "bullpen_pitches_3d_adv": 30.0 if index % 2 else -30.0,
                "relievers_used_3d_adv": 2.0 if index % 2 else -2.0,
            }
        )
    bundle = train_game_outcome_model(pd.DataFrame(rows))
    feature_row = pd.DataFrame([rows[-1], {**rows[-1], "game_id": "alt", "market_home_implied_prob": 0.80}]).drop(columns=["target_home_win"])

    scored = predict_home_win_probabilities(bundle, feature_row)

    assert scored.iloc[1]["home_win_prob"] > scored.iloc[0]["home_win_prob"]


def test_analyze_forward_feature_selection_returns_aic_and_bic_models() -> None:
    rows = []
    for index in range(24):
        home_win = index % 2
        rows.append(
            {
                "game_id": str(index),
                "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=index),
                "target_home_win": home_win,
                "season_win_pct_diff": 0.18 if home_win else -0.18,
                "recent_win_pct_diff": 0.16 if home_win else -0.16,
                "venue_win_pct_diff": 0.02 if index % 3 else -0.01,
                "run_diff_per_game_diff": 1.4 if home_win else -1.4,
                "season_runs_scored_per_game_diff": 0.8 if home_win else -0.8,
                "season_runs_allowed_per_game_adv": 0.7 if home_win else -0.7,
                "recent_runs_scored_per_game_diff": 0.75 if home_win else -0.75,
                "recent_runs_allowed_per_game_adv": 0.65 if home_win else -0.65,
                "rest_days_diff": 0.5 if home_win else -0.5,
                "venue_streak_diff": 1.0 if home_win else -1.0,
                "travel_switch_adv": 0.0,
                "doubleheader_flag": float(index % 5 == 0),
                "streak_diff": 3.0 if home_win else -3.0,
                "elo_diff": 42.0 if home_win else -42.0,
                "elo_home_win_prob": 0.62 if home_win else 0.38,
                "market_home_implied_prob": 0.60 if home_win else 0.40,
                "offense_vs_starter_hand_diff": 0.6 if home_win else -0.6,
                "starter_era_adv": 1.6 if home_win else -1.6,
                "starter_whip_adv": 0.35 if home_win else -0.35,
                "starter_strikeouts_per_9_diff": 1.8 if home_win else -1.8,
                "starter_walks_per_9_adv": 0.9 if home_win else -0.9,
                "bullpen_innings_3d_adv": 1.5 if home_win else -1.5,
                "bullpen_pitches_3d_adv": 24.0 if home_win else -24.0,
                "relievers_used_3d_adv": 1.0 if home_win else -1.0,
            }
        )
    training_df = pd.DataFrame(rows)

    result = analyze_forward_feature_selection(training_df)

    assert result["status"] == "ok"
    assert result["rows"] == len(training_df)
    assert len(result["aic_steps"]) >= 1
    assert len(result["bic_steps"]) >= 1
    assert result["best_aic_model"]["feature_count"] >= 1
    assert result["best_bic_model"]["feature_count"] >= 1
    assert set(result["best_aic_model"]["features"]).issubset(FEATURE_COLUMNS)
    assert set(result["best_bic_model"]["features"]).issubset(FEATURE_COLUMNS)
    assert "log_loss" in result["best_aic_model"]["valid_metrics"]
    assert "accuracy" in result["best_bic_model"]["valid_metrics"]
