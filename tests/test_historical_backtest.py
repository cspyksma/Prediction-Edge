from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import uuid

from mlpm.backtest.walkforward import run_walkforward_backtest
from mlpm.models.game_outcome import run_historical_kalshi_backtest
from mlpm.storage.duckdb import connect_read_only


def test_run_historical_kalshi_backtest_reports_roi(monkeypatch) -> None:
    replay_df = pd.DataFrame(
        [
            {
                "game_id": "g5",
                "game_date": "2026-04-05",
                "event_start_time": "2026-04-05T18:00:00Z",
                "snapshot_ts": "2026-04-05T17:30:00Z",
                "home_team": "Home5",
                "away_team": "Away5",
                "home_market_id": "m5h",
                "away_market_id": "m5a",
                "home_ticker": "t5h",
                "away_ticker": "t5a",
                "home_market_prob": 0.45,
                "away_market_prob": 0.55,
            },
        ]
    )
    results_df = pd.DataFrame(
        [
            {"game_id": "g5", "game_date": "2026-04-05", "home_team": "Home5", "away_team": "Away5", "winner_team": "Away5"},
        ]
    )
    pitching_logs = pd.DataFrame(columns=["game_id", "team", "game_date"])
    batting_logs = pd.DataFrame(columns=["game_id", "team", "game_date"])
    training_df = pd.DataFrame(
        [
            {"game_id": "g1", "game_date": pd.Timestamp("2026-04-01"), "home_team": "Home1", "away_team": "Away1", "target_home_win": 1},
            {"game_id": "g2", "game_date": pd.Timestamp("2026-04-02"), "home_team": "Home2", "away_team": "Away2", "target_home_win": 0},
            {"game_id": "g3", "game_date": pd.Timestamp("2026-04-03"), "home_team": "Home3", "away_team": "Away3", "target_home_win": 1},
            {"game_id": "g4", "game_date": pd.Timestamp("2026-04-04"), "home_team": "Home4", "away_team": "Away4", "target_home_win": 0},
            {"game_id": "g5", "game_date": pd.Timestamp("2026-04-05"), "home_team": "Home5", "away_team": "Away5", "target_home_win": 1},
        ]
    )
    for column in [
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
    ]:
        training_df[column] = [0.1, -0.1, 0.2, -0.2, 0.15]

    class DummyPipeline:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict_proba(self, frame):
            return [[1.0 - self.value, self.value] for _ in range(len(frame))]

    monkeypatch.setattr("mlpm.historical.replay.load_kalshi_pregame_replay", lambda *_args, **_kwargs: replay_df)
    monkeypatch.setattr("mlpm.models.game_outcome.fetch_final_results", lambda *_args, **_kwargs: results_df)
    monkeypatch.setattr("mlpm.models.game_outcome.fetch_game_pitching_logs", lambda *_args, **_kwargs: pitching_logs)
    monkeypatch.setattr("mlpm.models.game_outcome.fetch_game_batting_logs", lambda *_args, **_kwargs: batting_logs)
    monkeypatch.setattr("mlpm.models.game_outcome.build_training_dataset", lambda *args, **kwargs: training_df)
    monkeypatch.setattr(
        "mlpm.models.game_outcome._fit_candidate_models",
        lambda _frame: {
            "mlb_win_logreg_v2": DummyPipeline(0.7),
            "mlb_win_histgb_v1": DummyPipeline(0.55),
        },
    )
    monkeypatch.setattr(
        "mlpm.models.game_outcome._train_bayesian_bundle",
        lambda _frame: {"pipeline": DummyPipeline(0.65), "base_rate": 0.5},
    )

    result = run_historical_kalshi_backtest(
        train_start_date="2026-04-01",
        train_end_date="2026-04-04",
        eval_start_date="2026-04-05",
        eval_end_date="2026-04-05",
    )

    assert result["status"] == "ok"
    assert result["rows_train"] == 4
    assert result["rows_valid"] == 1
    assert result["train_end_date"] == "2026-04-04"
    assert result["eval_start_date"] == "2026-04-05"
    assert "mlb_win_bayes_v1" in result["benchmarks"]
    assert "roi" in result["benchmarks"]["mlb_win_bayes_v1"]


def test_run_walkforward_backtest_clears_prior_rows_for_same_run_id(monkeypatch) -> None:
    tmp_dir = Path(".tmp") / f"walkforward-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_dir / "walkforward.duckdb"
    feature_df = pd.DataFrame(
        [
            {
                "game_id": "train-1",
                "game_date": pd.Timestamp("2026-03-15").date(),
                "home_team": "Home A",
                "away_team": "Away A",
                "target_home_win": 1,
                "season_win_pct_diff": 0.1,
                "recent_win_pct_diff": 0.1,
                "venue_win_pct_diff": 0.1,
                "run_diff_per_game_diff": 0.1,
                "season_runs_scored_per_game_diff": 0.1,
                "season_runs_allowed_per_game_adv": 0.1,
                "recent_runs_scored_per_game_diff": 0.1,
                "recent_runs_allowed_per_game_adv": 0.1,
                "rest_days_diff": 0.1,
                "venue_streak_diff": 0.1,
                "travel_switch_adv": 0.0,
                "doubleheader_flag": 0.0,
                "streak_diff": 0.1,
                "elo_diff": 10.0,
                "elo_home_win_prob": 0.55,
                "offense_vs_starter_hand_diff": 0.1,
                "starter_era_adv": 0.1,
                "starter_whip_adv": 0.1,
                "starter_strikeouts_per_9_diff": 0.1,
                "starter_walks_per_9_adv": 0.1,
                "bullpen_innings_3d_adv": 0.1,
                "bullpen_pitches_3d_adv": 1.0,
                "relievers_used_3d_adv": 0.1,
                "weather_temp_f": 72.0,
                "weather_wind_out_to_cf_mph": 0.0,
                "weather_humidity_pct": 50.0,
                "weather_precipitation_in": 0.0,
                "weather_is_dome_sealed": 0.0,
            },
            {
                "game_id": "train-2",
                "game_date": pd.Timestamp("2026-03-20").date(),
                "home_team": "Home C",
                "away_team": "Away C",
                "target_home_win": 0,
                "season_win_pct_diff": -0.1,
                "recent_win_pct_diff": -0.1,
                "venue_win_pct_diff": -0.1,
                "run_diff_per_game_diff": -0.1,
                "season_runs_scored_per_game_diff": -0.1,
                "season_runs_allowed_per_game_adv": -0.1,
                "recent_runs_scored_per_game_diff": -0.1,
                "recent_runs_allowed_per_game_adv": -0.1,
                "rest_days_diff": -0.1,
                "venue_streak_diff": -0.1,
                "travel_switch_adv": 0.0,
                "doubleheader_flag": 0.0,
                "streak_diff": -0.1,
                "elo_diff": -10.0,
                "elo_home_win_prob": 0.45,
                "offense_vs_starter_hand_diff": -0.1,
                "starter_era_adv": -0.1,
                "starter_whip_adv": -0.1,
                "starter_strikeouts_per_9_diff": -0.1,
                "starter_walks_per_9_adv": -0.1,
                "bullpen_innings_3d_adv": -0.1,
                "bullpen_pitches_3d_adv": -1.0,
                "relievers_used_3d_adv": -0.1,
                "weather_temp_f": 72.0,
                "weather_wind_out_to_cf_mph": 0.0,
                "weather_humidity_pct": 50.0,
                "weather_precipitation_in": 0.0,
                "weather_is_dome_sealed": 0.0,
            },
            {
                "game_id": "eval-1",
                "game_date": pd.Timestamp("2026-04-02").date(),
                "home_team": "Home B",
                "away_team": "Away B",
                "target_home_win": 1,
                "season_win_pct_diff": 0.2,
                "recent_win_pct_diff": 0.2,
                "venue_win_pct_diff": 0.2,
                "run_diff_per_game_diff": 0.2,
                "season_runs_scored_per_game_diff": 0.2,
                "season_runs_allowed_per_game_adv": 0.2,
                "recent_runs_scored_per_game_diff": 0.2,
                "recent_runs_allowed_per_game_adv": 0.2,
                "rest_days_diff": 0.2,
                "venue_streak_diff": 0.2,
                "travel_switch_adv": 0.0,
                "doubleheader_flag": 0.0,
                "streak_diff": 0.2,
                "elo_diff": 20.0,
                "elo_home_win_prob": 0.60,
                "offense_vs_starter_hand_diff": 0.2,
                "starter_era_adv": 0.2,
                "starter_whip_adv": 0.2,
                "starter_strikeouts_per_9_diff": 0.2,
                "starter_walks_per_9_adv": 0.2,
                "bullpen_innings_3d_adv": 0.2,
                "bullpen_pitches_3d_adv": 2.0,
                "relievers_used_3d_adv": 0.2,
                "weather_temp_f": 72.0,
                "weather_wind_out_to_cf_mph": 0.0,
                "weather_humidity_pct": 50.0,
                "weather_precipitation_in": 0.0,
                "weather_is_dome_sealed": 0.0,
            },
        ]
    )
    priors_df = pd.DataFrame(
        [
            {
                "game_id": "eval-1",
                "game_date": pd.Timestamp("2026-04-02").date(),
                "home_team": "Home B",
                "away_team": "Away B",
                "home_fair_prob": 0.50,
                "away_fair_prob": 0.50,
                "home_moneyline_close": -110,
                "away_moneyline_close": 100,
                "book": "consensus_close",
            }
        ]
    )

    class DummyPipeline:
        def __init__(self, probability: float) -> None:
            self.probability = probability

        def predict_proba(self, frame):
            return np.asarray([[1.0 - self.probability, self.probability] for _ in range(len(frame))], dtype=float)

    monkeypatch.setattr("mlpm.backtest.walkforward._load_feature_frame", lambda *_args, **_kwargs: feature_df)
    monkeypatch.setattr("mlpm.backtest.walkforward._load_sbro_priors", lambda *_args, **_kwargs: priors_df)
    monkeypatch.setattr("mlpm.backtest.walkforward._fit_model", lambda *_args, **_kwargs: DummyPipeline(0.70))

    first = run_walkforward_backtest(
        "2026-04-01",
        "2026-04-30",
        model_name="logreg",
        min_train_rows=1,
        db_path=db_path,
        run_id="wf-test",
    )
    assert first["total_bets"] == 1

    monkeypatch.setattr("mlpm.backtest.walkforward._fit_model", lambda *_args, **_kwargs: DummyPipeline(0.51))
    second = run_walkforward_backtest(
        "2026-04-01",
        "2026-04-30",
        model_name="logreg",
        min_train_rows=1,
        db_path=db_path,
        run_id="wf-test",
    )
    assert second["total_bets"] == 0

    conn = connect_read_only(db_path)
    try:
        remaining = conn.execute("SELECT COUNT(*) FROM walkforward_bets WHERE run_id = 'wf-test'").fetchone()[0]
    finally:
        conn.close()

    assert remaining == 0


def test_run_walkforward_backtest_all_models_persists_distinct_model_rows(monkeypatch) -> None:
    tmp_dir = Path(".tmp") / f"walkforward-all-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_dir / "walkforward.duckdb"
    feature_df = pd.DataFrame(
        [
            {
                "game_id": "train-1",
                "game_date": pd.Timestamp("2026-03-15").date(),
                "home_team": "Home A",
                "away_team": "Away A",
                "target_home_win": 1,
                "season_win_pct_diff": 0.1,
                "recent_win_pct_diff": 0.1,
                "venue_win_pct_diff": 0.1,
                "run_diff_per_game_diff": 0.1,
                "season_runs_scored_per_game_diff": 0.1,
                "season_runs_allowed_per_game_adv": 0.1,
                "recent_runs_scored_per_game_diff": 0.1,
                "recent_runs_allowed_per_game_adv": 0.1,
                "rest_days_diff": 0.1,
                "venue_streak_diff": 0.1,
                "travel_switch_adv": 0.0,
                "doubleheader_flag": 0.0,
                "streak_diff": 0.1,
                "elo_diff": 10.0,
                "elo_home_win_prob": 0.55,
                "offense_vs_starter_hand_diff": 0.1,
                "starter_era_adv": 0.1,
                "starter_whip_adv": 0.1,
                "starter_strikeouts_per_9_diff": 0.1,
                "starter_walks_per_9_adv": 0.1,
                "bullpen_innings_3d_adv": 0.1,
                "bullpen_pitches_3d_adv": 1.0,
                "relievers_used_3d_adv": 0.1,
                "weather_temp_f": 72.0,
                "weather_wind_out_to_cf_mph": 0.0,
                "weather_humidity_pct": 50.0,
                "weather_precipitation_in": 0.0,
                "weather_is_dome_sealed": 0.0,
            },
            {
                "game_id": "train-2",
                "game_date": pd.Timestamp("2026-03-20").date(),
                "home_team": "Home C",
                "away_team": "Away C",
                "target_home_win": 0,
                "season_win_pct_diff": -0.1,
                "recent_win_pct_diff": -0.1,
                "venue_win_pct_diff": -0.1,
                "run_diff_per_game_diff": -0.1,
                "season_runs_scored_per_game_diff": -0.1,
                "season_runs_allowed_per_game_adv": -0.1,
                "recent_runs_scored_per_game_diff": -0.1,
                "recent_runs_allowed_per_game_adv": -0.1,
                "rest_days_diff": -0.1,
                "venue_streak_diff": -0.1,
                "travel_switch_adv": 0.0,
                "doubleheader_flag": 0.0,
                "streak_diff": -0.1,
                "elo_diff": -10.0,
                "elo_home_win_prob": 0.45,
                "offense_vs_starter_hand_diff": -0.1,
                "starter_era_adv": -0.1,
                "starter_whip_adv": -0.1,
                "starter_strikeouts_per_9_diff": -0.1,
                "starter_walks_per_9_adv": -0.1,
                "bullpen_innings_3d_adv": -0.1,
                "bullpen_pitches_3d_adv": -1.0,
                "relievers_used_3d_adv": -0.1,
                "weather_temp_f": 72.0,
                "weather_wind_out_to_cf_mph": 0.0,
                "weather_humidity_pct": 50.0,
                "weather_precipitation_in": 0.0,
                "weather_is_dome_sealed": 0.0,
            },
            {
                "game_id": "eval-1",
                "game_date": pd.Timestamp("2026-04-02").date(),
                "home_team": "Home B",
                "away_team": "Away B",
                "target_home_win": 1,
                "season_win_pct_diff": 0.2,
                "recent_win_pct_diff": 0.2,
                "venue_win_pct_diff": 0.2,
                "run_diff_per_game_diff": 0.2,
                "season_runs_scored_per_game_diff": 0.2,
                "season_runs_allowed_per_game_adv": 0.2,
                "recent_runs_scored_per_game_diff": 0.2,
                "recent_runs_allowed_per_game_adv": 0.2,
                "rest_days_diff": 0.2,
                "venue_streak_diff": 0.2,
                "travel_switch_adv": 0.0,
                "doubleheader_flag": 0.0,
                "streak_diff": 0.2,
                "elo_diff": 20.0,
                "elo_home_win_prob": 0.60,
                "offense_vs_starter_hand_diff": 0.2,
                "starter_era_adv": 0.2,
                "starter_whip_adv": 0.2,
                "starter_strikeouts_per_9_diff": 0.2,
                "starter_walks_per_9_adv": 0.2,
                "bullpen_innings_3d_adv": 0.2,
                "bullpen_pitches_3d_adv": 2.0,
                "relievers_used_3d_adv": 0.2,
                "weather_temp_f": 72.0,
                "weather_wind_out_to_cf_mph": 0.0,
                "weather_humidity_pct": 50.0,
                "weather_precipitation_in": 0.0,
                "weather_is_dome_sealed": 0.0,
            },
        ]
    )
    priors_df = pd.DataFrame(
        [
            {
                "game_id": "eval-1",
                "game_date": pd.Timestamp("2026-04-02").date(),
                "home_team": "Home B",
                "away_team": "Away B",
                "home_fair_prob": 0.50,
                "away_fair_prob": 0.50,
                "home_moneyline_close": -110,
                "away_moneyline_close": 100,
                "book": "consensus_close",
            }
        ]
    )

    class DummyPipeline:
        def __init__(self, probability: float) -> None:
            self.probability = probability

        def predict_proba(self, frame):
            return np.asarray([[1.0 - self.probability, self.probability] for _ in range(len(frame))], dtype=float)

    model_probabilities = {"logreg": 0.70, "histgb": 0.68, "knn": 0.66, "svm": 0.64}

    monkeypatch.setattr("mlpm.backtest.walkforward._load_feature_frame", lambda *_args, **_kwargs: feature_df)
    monkeypatch.setattr("mlpm.backtest.walkforward._load_sbro_priors", lambda *_args, **_kwargs: priors_df)
    monkeypatch.setattr(
        "mlpm.backtest.walkforward._fit_model",
        lambda _train_df, current_model_name: DummyPipeline(model_probabilities[current_model_name]),
    )

    result = run_walkforward_backtest(
        "2026-04-01",
        "2026-04-30",
        model_name="all",
        min_train_rows=1,
        db_path=db_path,
        run_id="wf-all",
    )

    assert result["model_names"] == ["logreg", "histgb", "knn", "svm"]
    assert result["total_bets"] == 4
    assert set(result["models"]) == {"logreg", "histgb", "knn", "svm"}
    assert all(metrics["total_bets"] == 1 for metrics in result["models"].values())

    conn = connect_read_only(db_path)
    try:
        rows = conn.execute(
            "SELECT model_name, COUNT(*) AS bets FROM walkforward_bets WHERE run_id = 'wf-all' GROUP BY model_name ORDER BY model_name"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [("histgb", 1), ("knn", 1), ("logreg", 1), ("svm", 1)]
