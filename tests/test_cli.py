from pathlib import Path

import pytest

from mlpm.cli import (
    _build_parser,
    app,
    _format_bet_opportunity_output,
    _format_benchmark_game_model_output,
    _format_historical_import_output,
    _format_historical_backtest_output,
    _format_historical_status_output,
    _format_settled_prediction_output,
    _format_settled_window_output,
    _format_strategy_performance_output,
    _format_train_game_model_output,
)


def test_format_train_game_model_output_includes_candidate_table() -> None:
    output = _format_train_game_model_output(
        {
            "path": Path("artifacts/game_outcome_model.pkl"),
            "rows": 120,
            "bundle": {
                "model_name": "mlb_win_knn_v1",
                "candidate_metrics": {
                    "mlb_win_logreg_v2": {"roc_auc": 0.71234, "log_loss": 0.60123},
                    "mlb_win_histgb_v1": {"roc_auc": 0.70111, "log_loss": 0.61234},
                    "mlb_win_knn_v1": {"roc_auc": 0.73333, "log_loss": 0.59001},
                },
                "benchmarks": {
                    "benchmarks": {
                        "always_home": {"roc_auc": 0.5000, "log_loss": 6.9010},
                        "heuristic_feature_score": {"roc_auc": 0.6500, "log_loss": 0.6400},
                    }
                },
            },
        }
    )

    assert "selected_model: mlb_win_knn_v1" in output
    assert "Candidate Metrics" in output
    assert "Benchmark Metrics" in output
    assert "mlb_win_knn_v1" in output
    assert "roc_auc" in output
    assert "log_loss" in output


def test_format_benchmark_game_model_output_includes_rows_and_table() -> None:
    output = _format_benchmark_game_model_output(
        {
            "status": "ok",
            "rows": 200,
            "rows_train": 160,
            "rows_valid": 40,
            "benchmarks": {
                "mlb_win_logreg_v2": {"roc_auc": 0.7100, "log_loss": 0.6100},
                "mlb_win_knn_v1": {"roc_auc": 0.7300, "log_loss": 0.5900},
            },
            "calibration": [{"model_name": "mlb_win_knn_v1", "bucket": "(0.4, 0.6]"}],
        }
    )

    assert "rows_train: 160" in output
    assert "rows_valid: 40" in output
    assert "Benchmark Metrics" in output
    assert "mlb_win_knn_v1" in output
    assert "calibration_rows: 1" in output


def test_format_settled_prediction_output_includes_metrics_table() -> None:
    output = _format_settled_prediction_output(
        {
            "status": "ok",
            "rows": 12,
            "models": {
                "mlb_win_svm_rbf_v1": {"games": 12, "accuracy": 0.6667, "log_loss": 0.6412},
            },
            "recent": [{"game_date": "2026-04-14"}],
        }
    )

    assert "Settled Metrics" in output
    assert "rows: 12" in output
    assert "mlb_win_svm_rbf_v1" in output
    assert "recent_rows: 1" in output


def test_format_settled_window_output_includes_window_metrics() -> None:
    output = _format_settled_window_output(
        {
            "status": "ok",
            "rows": 24,
            "windows": {
                "mlb_win_logreg_v2": {
                    "all": {"games": 24, "accuracy": 0.6250, "log_loss": 0.6400},
                    "last_7d": {"games": 10, "accuracy": 0.7000, "log_loss": 0.6100},
                    "last_30d": {"games": 24, "accuracy": 0.6250, "log_loss": 0.6400},
                    "last_50_games": {"games": 24, "accuracy": 0.6250, "log_loss": 0.6400},
                }
            },
            "daily": [{"game_date": "2026-04-14"}],
        }
    )

    assert "Settled Windows" in output
    assert "mlb_win_logreg_v2" in output
    assert "last_7d_accuracy" in output
    assert "daily_rows: 1" in output


def test_format_bet_opportunity_output_includes_metrics_table() -> None:
    output = _format_bet_opportunity_output(
        {
            "status": "ok",
            "rows": 10,
            "champion_model": "mlb_win_svm_rbf_v1",
            "models": {
                "mlb_win_svm_rbf_v1": {
                    "opportunities": 10,
                    "actionable_bets": 2,
                    "avg_edge_bps": 620.0,
                    "avg_expected_value": 0.08,
                    "champion_rows": 10,
                }
            },
        }
    )

    assert "Bet Opportunities" in output
    assert "rows: 10" in output
    assert "champion_model: mlb_win_svm_rbf_v1" in output
    assert "mlb_win_svm_rbf_v1" in output


def test_format_strategy_performance_output_includes_windows() -> None:
    output = _format_strategy_performance_output(
        {
            "status": "ok",
            "rows": 8,
            "champion_model": "mlb_win_svm_rbf_v1",
            "windows": {
                "mlb_win_svm_rbf_v1": {
                    "all": {"bets": 8, "roi": 0.12, "units_won": 0.96, "avg_edge_bps": 610.0},
                    "last_7d": {"bets": 4, "roi": 0.15, "units_won": 0.6, "avg_edge_bps": 640.0},
                    "last_30d": {"bets": 8, "roi": 0.12, "units_won": 0.96, "avg_edge_bps": 610.0},
                    "last_50_bets": {"bets": 8, "roi": 0.12, "units_won": 0.96, "avg_edge_bps": 610.0},
                }
            },
            "daily": [{"game_date": "2026-04-15"}],
        }
    )

    assert "Strategy Performance" in output
    assert "champion_model: mlb_win_svm_rbf_v1" in output
    assert "last_7d_roi" in output
    assert "daily_rows: 1" in output


def test_format_historical_import_output_includes_counts() -> None:
    output = _format_historical_import_output(
        {
            "status": "ok",
            "import_run_id": "run-1",
            "request_count": 4,
            "payload_count": 4,
            "normalized_rows": 100,
            "chunks_total": 3,
            "chunks_completed": 2,
            "chunks_skipped": 1,
        }
    )

    assert "import_run_id: run-1" in output
    assert "normalized_rows: 100" in output
    assert "chunks_skipped: 1" in output
    assert "games_with_pregame_quotes: 0" not in output


def test_format_historical_status_output_includes_table() -> None:
    output = _format_historical_status_output(
        {
            "status": "ok",
            "rows": 2,
            "sources": {
                "polymarket": {"import_runs": 1, "request_count": 3, "payload_count": 3, "normalized_rows": 120},
                "kalshi": {
                    "import_runs": 1,
                    "request_count": 5,
                    "payload_count": 5,
                    "normalized_rows": 80,
                    "games_total": 7,
                    "games_with_markets": 5,
                    "games_with_pregame_quotes": 4,
                    "candidate_markets": 10,
                    "empty_payload_count": 1,
                    "rate_limited_count": 1,
                    "parse_error_count": 0,
                },
            },
        }
    )

    assert "Historical Import Status" in output
    assert "rows: 2" in output
    assert "polymarket" in output


def test_format_historical_backtest_output_includes_roi_metrics() -> None:
    output = _format_historical_backtest_output(
        {
            "status": "ok",
            "rows": 20,
            "rows_train": 16,
            "rows_valid": 4,
            "train_start_date": "2026-03-01",
            "train_end_date": "2026-03-31",
            "eval_start_date": "2026-04-01",
            "eval_end_date": "2026-04-02",
            "replay_rows": 20,
            "valid_replay_rows": 4,
            "champion_model": "mlb_win_bayes_v1",
            "benchmarks": {
                "mlb_win_bayes_v1": {"roi": 0.12, "units_won": 0.48, "bets": 4, "log_loss": 0.61},
            },
            "calibration": [{"model_name": "mlb_win_bayes_v1"}],
        }
    )

    assert "Historical Kalshi Backtest" in output
    assert "train_window: 2026-03-01 to 2026-03-31" in output
    assert "eval_window: 2026-04-01 to 2026-04-02" in output
    assert "champion_model: mlb_win_bayes_v1" in output
    assert "calibration_rows: 1" in output


def test_cli_parser_validates_iso_date_arguments() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "backtest",
                "--start-date",
                "2026-99-99",
                "--end-date",
                "2026-04-07",
            ]
        )


def test_cli_parser_normalizes_valid_iso_dates() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "backtest",
            "--start-date",
            "2026-04-01",
            "--end-date",
            "2026-04-07",
        ]
    )

    assert args.start_date == "2026-04-01"
    assert args.end_date == "2026-04-07"


def test_cli_app_rejects_reversed_backtest_range(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["mlpm", "backtest", "--start-date", "2026-04-07", "--end-date", "2026-04-01"],
    )

    with pytest.raises(SystemExit, match="backtest start_date must be on or before end_date"):
        app()


def test_cli_app_rejects_reversed_historical_eval_range(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "mlpm",
            "historical-backtest-kalshi",
            "--eval-start-date",
            "2026-04-07",
            "--eval-end-date",
            "2026-04-01",
        ],
    )

    with pytest.raises(SystemExit, match="historical-backtest-kalshi eval start_date must be on or before end_date"):
        app()
