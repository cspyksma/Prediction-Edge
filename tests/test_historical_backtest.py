from __future__ import annotations

import pandas as pd

from mlpm.models.game_outcome import run_historical_kalshi_backtest


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
