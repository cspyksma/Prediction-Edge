from __future__ import annotations

import pandas as pd

from mlpm.backtest.research import run_kalshi_edge_research_backtest
from mlpm.cli import _format_kalshi_research_output


def _make_training_df() -> pd.DataFrame:
    rows = [
        {"game_id": "train-1", "game_date": pd.Timestamp("2020-06-01"), "home_team": "A", "away_team": "B", "target_home_win": 1, "market_home_implied_prob": 0.58},
        {"game_id": "train-2", "game_date": pd.Timestamp("2020-06-02"), "home_team": "C", "away_team": "D", "target_home_win": 0, "market_home_implied_prob": 0.42},
        {"game_id": "train-3", "game_date": pd.Timestamp("2020-06-03"), "home_team": "E", "away_team": "F", "target_home_win": 1, "market_home_implied_prob": 0.56},
        {"game_id": "train-4", "game_date": pd.Timestamp("2020-06-04"), "home_team": "G", "away_team": "H", "target_home_win": 0, "market_home_implied_prob": 0.44},
        {"game_id": "eval-1", "game_date": pd.Timestamp("2025-04-01"), "home_team": "I", "away_team": "J", "target_home_win": 1, "market_home_implied_prob": 0.48},
        {"game_id": "eval-2", "game_date": pd.Timestamp("2025-05-01"), "home_team": "K", "away_team": "L", "target_home_win": 0, "market_home_implied_prob": 0.46},
    ]
    feature_values = {
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
    }
    for row in rows:
        row.update(feature_values)
    return pd.DataFrame(rows)


def test_run_kalshi_edge_research_backtest_reports_champion_and_strategy_metadata(monkeypatch) -> None:
    results_df = pd.DataFrame(
        [
            {"game_id": "train-1", "game_date": "2020-06-01", "home_team": "A", "away_team": "B", "winner_team": "A", "away_score": 1, "home_score": 3},
            {"game_id": "train-2", "game_date": "2020-06-02", "home_team": "C", "away_team": "D", "winner_team": "D", "away_score": 4, "home_score": 2},
            {"game_id": "train-3", "game_date": "2020-06-03", "home_team": "E", "away_team": "F", "winner_team": "E", "away_score": 1, "home_score": 6},
            {"game_id": "train-4", "game_date": "2020-06-04", "home_team": "G", "away_team": "H", "winner_team": "H", "away_score": 5, "home_score": 2},
            {"game_id": "eval-1", "game_date": "2025-04-01", "home_team": "I", "away_team": "J", "winner_team": "I", "away_score": 1, "home_score": 4},
            {"game_id": "eval-2", "game_date": "2025-05-01", "home_team": "K", "away_team": "L", "winner_team": "L", "away_score": 5, "home_score": 2},
        ]
    )
    replay_df = pd.DataFrame(
        [
            {
                "game_id": "eval-1",
                "game_date": "2025-04-01",
                "event_start_time": "2025-04-01T18:00:00Z",
                "snapshot_ts": "2025-04-01T17:30:00Z",
                "home_team": "I",
                "away_team": "J",
                "home_market_prob": 0.48,
                "away_market_prob": 0.52,
            },
            {
                "game_id": "eval-2",
                "game_date": "2025-05-01",
                "event_start_time": "2025-05-01T18:00:00Z",
                "snapshot_ts": "2025-05-01T17:30:00Z",
                "home_team": "K",
                "away_team": "L",
                "home_market_prob": 0.46,
                "away_market_prob": 0.54,
            },
        ]
    )
    training_df = _make_training_df()

    monkeypatch.setattr("mlpm.backtest.research.DEFAULT_MIN_BETS", 1)
    monkeypatch.setattr("mlpm.backtest.research.DEFAULT_MIN_ACTIVE_SLICES", 1)
    monkeypatch.setattr("mlpm.backtest.research.DEFAULT_MAX_SLICE_BET_SHARE", 1.0)
    monkeypatch.setattr("mlpm.backtest.research._load_local_results_frame", lambda *_args, **_kwargs: results_df)
    monkeypatch.setattr("mlpm.backtest.research._load_local_pitching_logs", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.backtest.research._load_local_batting_logs", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.backtest.research._load_game_weather", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.backtest.research._load_sbro_market_priors", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.backtest.research.load_kalshi_pregame_replay", lambda *_args, **_kwargs: replay_df)
    monkeypatch.setattr("mlpm.backtest.research.build_training_dataset", lambda *args, **kwargs: training_df)
    monkeypatch.setattr(
        "mlpm.backtest.research.build_kalshi_replay_quote_rows",
        lambda frame: [
            {
                "game_id": row["game_id"],
                "snapshot_ts": row["snapshot_ts"],
                "outcome_team": row["home_team"],
                "source": "kalshi",
                "market_id": f"{row['game_id']}-home",
                "fair_prob": row["home_market_prob"],
                "is_valid": True,
                "is_pregame": True,
            }
            for row in frame.to_dict(orient="records")
        ]
        + [
            {
                "game_id": row["game_id"],
                "snapshot_ts": row["snapshot_ts"],
                "outcome_team": row["away_team"],
                "source": "kalshi",
                "market_id": f"{row['game_id']}-away",
                "fair_prob": row["away_market_prob"],
                "is_valid": True,
                "is_pregame": True,
            }
            for row in frame.to_dict(orient="records")
        ],
    )
    monkeypatch.setattr(
        "mlpm.backtest.research._fit_and_score_contenders",
        lambda train_df, valid_df: (
            {
                "baseball_logreg_v1": pd.Series([0.62, 0.41], dtype=float).to_numpy(),
                "hybrid_logreg_market_v1": pd.Series([0.68, 0.33], dtype=float).to_numpy(),
                "market_identity_v1": valid_df["market_home_implied_prob"].to_numpy(dtype=float),
            },
            {
                "baseball_logreg_v1": {"contender_name": "baseball_logreg_v1", "family": "baseball_only", "model_family": "logreg", "feature_variant": "baseball_only", "accuracy": 1.0, "roc_auc": 1.0, "log_loss": 0.55},
                "hybrid_logreg_market_v1": {"contender_name": "hybrid_logreg_market_v1", "family": "hybrid_market_aware", "model_family": "logreg", "feature_variant": "baseball_plus_market", "accuracy": 1.0, "roc_auc": 1.0, "log_loss": 0.40},
                "market_identity_v1": {"contender_name": "market_identity_v1", "family": "market_only", "model_family": "identity", "feature_variant": "market_only", "accuracy": 0.5, "roc_auc": 0.5, "log_loss": 0.69},
            },
            [{"model_name": "hybrid_logreg_market_v1", "bucket": "(0.4, 0.6]"}],
        ),
    )

    result = run_kalshi_edge_research_backtest(
        train_start_date="2020-06-01",
        train_end_date="2020-06-04",
        eval_start_date="2025-04-01",
        eval_end_date="2025-05-31",
    )

    assert result["status"] == "ok"
    assert result["rows_train"] == 4
    assert result["rows_valid"] == 2
    assert result["snapshot_policy"] == "t_minus_30m"
    assert result["slippage_bps"] == 25
    assert result["partial_fill_rate"] == 0.9
    assert result["champion_strategy"] is not None
    assert result["champion_contender"] in {"baseball_logreg_v1", "hybrid_logreg_market_v1", "market_identity_v1"}
    assert result["contender_count"] == 3
    assert result["strategy_count"] > 0
    assert all("roi_ci_lower" in row for row in result["strategies"])
    assert all("positive_slice_rate" in row for row in result["strategies"])
    assert any(row["family"] == "hybrid_market_aware" for row in result["strategies"])
    assert any(row["family"] == "market_only" for row in result["strategies"])


def test_run_kalshi_edge_research_backtest_requires_kalshi_replay(monkeypatch) -> None:
    monkeypatch.setattr("mlpm.backtest.research._load_local_results_frame", lambda *_args, **_kwargs: pd.DataFrame([{"game_id": "g1", "game_date": "2025-04-01", "home_team": "A", "away_team": "B", "winner_team": "A"}]))
    monkeypatch.setattr("mlpm.backtest.research._load_local_pitching_logs", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.backtest.research._load_local_batting_logs", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.backtest.research._load_game_weather", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.backtest.research._load_sbro_market_priors", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.backtest.research.load_kalshi_pregame_replay", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr("mlpm.backtest.research.build_training_dataset", lambda *args, **kwargs: _make_training_df())

    result = run_kalshi_edge_research_backtest(
        train_start_date="2020-06-01",
        train_end_date="2020-06-04",
        eval_start_date="2025-04-01",
        eval_end_date="2025-05-31",
    )

    assert result["status"] == "insufficient_data"
    assert "Kalshi" in result["message"]


def test_format_kalshi_research_output_includes_champion_and_strategy_table() -> None:
    output = _format_kalshi_research_output(
        {
            "status": "ok",
            "rows": 20,
            "rows_train": 12,
            "rows_valid": 8,
            "train_start_date": "2015-01-01",
            "train_end_date": "2021-12-31",
            "eval_start_date": "2025-01-01",
            "eval_end_date": "2026-12-31",
            "replay_rows": 10,
            "valid_replay_rows": 8,
            "snapshot_policy": "t_minus_30m",
            "slippage_bps": 25,
            "partial_fill_rate": 0.9,
            "contender_count": 2,
            "strategy_count": 3,
            "champion_strategy": "hybrid_logreg_market_v1__edge_100bps_flat_1u",
            "champion_contender": "hybrid_logreg_market_v1",
            "guardrail_champion": True,
            "contenders": [
                {
                    "contender_name": "hybrid_logreg_market_v1",
                    "family": "hybrid_market_aware",
                    "model_family": "logreg",
                    "feature_variant": "baseball_plus_market",
                    "accuracy": 0.7,
                    "roc_auc": 0.71,
                    "log_loss": 0.61,
                }
            ],
            "strategies": [
                {
                    "strategy_name": "hybrid_logreg_market_v1__edge_100bps_flat_1u",
                    "family": "hybrid_market_aware",
                    "bets": 42,
                    "roi": 0.084,
                    "roi_ci_lower": 0.021,
                    "roi_ci_upper": 0.132,
                    "units_won": 3.53,
                    "max_drawdown": 1.25,
                    "positive_slice_rate": 0.67,
                    "guardrails_passed": True,
                }
            ],
            "calibration": [{"model_name": "hybrid_logreg_market_v1"}],
        }
    )

    assert "snapshot_policy: t_minus_30m" in output
    assert "champion_strategy: hybrid_logreg_market_v1__edge_100bps_flat_1u" in output
    assert "Top Strategies" in output
    assert "hybrid_logreg_market_v1" in output
