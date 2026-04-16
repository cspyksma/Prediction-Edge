from __future__ import annotations

import argparse
from datetime import date
from typing import Any

from mlpm.backtest.run_backtest import run_backtest
from mlpm.pipeline.collect import collect_snapshot
from mlpm.pipeline.runner import run_service, sync_recent_game_results
from mlpm.config.settings import settings
from mlpm.evaluation.settled import run_settled_prediction_report, run_settled_window_report
from mlpm.evaluation.strategy import run_bet_opportunity_report, run_strategy_performance_report
from mlpm.historical.kalshi_backfill import backfill_kalshi_history_for_games
from mlpm.historical.polymarket_backfill import backfill_polymarket_history_for_games
from mlpm.historical.status import run_historical_import_status
from mlpm.models.game_outcome import (
    run_forward_feature_selection,
    run_model_benchmark,
    train_and_save_model,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlpm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect-once", help="Run a single snapshot collection cycle.")

    collect_loop = subparsers.add_parser("collect-loop", help="Run the collector continuously.")
    collect_loop.add_argument("--iterations", type=int, default=0, help="Zero means run forever.")

    service = subparsers.add_parser("run-service", help="Run the collector with retries, health logging, and result sync.")
    service.add_argument("--iterations", type=int, default=0, help="Zero means run forever.")

    sync_results = subparsers.add_parser("sync-results", help="Backfill recent final MLB game results into DuckDB.")
    sync_results.add_argument("--lookback-days", type=int, default=settings().results_sync_lookback_days)

    settled = subparsers.add_parser("report-settled-predictions", help="Grade stored pregame predictions against final results.")
    settled.add_argument("--start-date", required=True)
    settled.add_argument("--end-date", required=True)
    settled.add_argument("--model-name")

    settled_windows = subparsers.add_parser("report-settled-windows", help="Summarize rolling settled prediction performance.")
    settled_windows.add_argument("--start-date", required=True)
    settled_windows.add_argument("--end-date", required=True)
    settled_windows.add_argument("--model-name")

    opportunities = subparsers.add_parser("report-bet-opportunities", help="Summarize live edge opportunities by model.")
    opportunities.add_argument("--start-date", required=True)
    opportunities.add_argument("--end-date", required=True)
    opportunities.add_argument("--model-name")

    strategy = subparsers.add_parser("report-strategy-performance", help="Summarize settled flat-stake betting performance.")
    strategy.add_argument("--start-date", required=True)
    strategy.add_argument("--end-date", required=True)
    strategy.add_argument("--model-name")

    historical_poly = subparsers.add_parser("historical-backfill-polymarket", help="Backfill historical Polymarket market prices.")
    historical_poly.add_argument("--start-date", required=True)
    historical_poly.add_argument("--end-date", required=True)
    historical_poly.add_argument("--interval", default="1m")
    historical_poly.add_argument("--chunk-days", type=int, default=7)
    historical_poly.add_argument("--no-resume", action="store_true")

    historical_kalshi = subparsers.add_parser("historical-backfill-kalshi", help="Backfill historical Kalshi market prices.")
    historical_kalshi.add_argument("--start-date", required=True)
    historical_kalshi.add_argument("--end-date", required=True)
    historical_kalshi.add_argument("--period-interval", type=int, default=1)
    historical_kalshi.add_argument("--include-trades", action="store_true")
    historical_kalshi.add_argument("--chunk-days", type=int, default=7)
    historical_kalshi.add_argument("--no-resume", action="store_true")

    historical_status = subparsers.add_parser("historical-import-status", help="Show historical import status by source.")
    historical_status.add_argument("--start-date", required=True)
    historical_status.add_argument("--end-date", required=True)

    backtest = subparsers.add_parser("backtest", help="Run a simple backtest over stored data.")
    backtest.add_argument("--start-date", required=True)
    backtest.add_argument("--end-date", required=True)

    train_model = subparsers.add_parser("train-game-model", help="Train and persist the MLB game-outcome model.")
    train_model.add_argument("--start-date", default=settings().model_train_start_date)
    train_model.add_argument("--end-date", default=date.today().isoformat())

    benchmark_model = subparsers.add_parser("benchmark-game-model", help="Benchmark the game-outcome model against simple baselines.")
    benchmark_model.add_argument("--start-date", default=settings().model_train_start_date)
    benchmark_model.add_argument("--end-date", default=date.today().isoformat())

    select_features = subparsers.add_parser(
        "forward-select-game-features",
        help="Run forward logistic feature selection scored by AIC and BIC.",
    )
    select_features.add_argument("--start-date", default=settings().model_train_start_date)
    select_features.add_argument("--end-date", default=date.today().isoformat())

    return parser


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_metrics_table(title: str, metrics_by_model: dict[str, dict[str, Any]]) -> str:
    if not metrics_by_model:
        return f"{title}\n(no metrics)"

    metric_names: list[str] = []
    for metrics in metrics_by_model.values():
        for name in metrics:
            if name not in metric_names:
                metric_names.append(name)

    columns = ["model", *metric_names]
    rows = []
    for model_name, metrics in metrics_by_model.items():
        row = {"model": model_name}
        for metric_name in metric_names:
            row[metric_name] = _format_metric_value(metrics.get(metric_name, "-"))
        rows.append(row)

    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }

    def render_row(row: dict[str, str]) -> str:
        return " | ".join(row[column].ljust(widths[column]) for column in columns)

    header = render_row({column: column for column in columns})
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = "\n".join(render_row(row) for row in rows)
    return f"{title}\n{header}\n{divider}\n{body}"


def _format_train_game_model_output(result: dict[str, Any]) -> str:
    bundle = result["bundle"]
    lines = [
        f"status: ok",
        f"path: {result['path']}",
        f"rows: {result['rows']}",
        f"selected_model: {bundle['model_name']}",
    ]
    lines.append("")
    lines.append(_format_metrics_table("Candidate Metrics", bundle.get("candidate_metrics", {})))
    benchmark_rows = bundle.get("benchmarks", {}).get("benchmarks", {})
    if benchmark_rows:
        lines.append("")
        lines.append(_format_metrics_table("Benchmark Metrics", benchmark_rows))
    return "\n".join(lines)


def _format_benchmark_game_model_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)

    lines = [
        f"status: {result['status']}",
        f"rows: {result['rows']}",
        f"rows_train: {result['rows_train']}",
        f"rows_valid: {result['rows_valid']}",
        "",
        _format_metrics_table("Benchmark Metrics", result.get("benchmarks", {})),
    ]
    calibration = result.get("calibration", [])
    if calibration:
        lines.extend(["", f"calibration_rows: {len(calibration)}"])
    return "\n".join(lines)


def _format_forward_selection_table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"{title}\n(no rows)"

    columns = ["step", "feature_count", "aic", "bic", "valid_log_loss", "valid_accuracy", "features"]
    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        valid_metrics = row.get("valid_metrics", {})
        formatted_rows.append(
            {
                "step": str(row.get("step", "-")),
                "feature_count": str(row.get("feature_count", "-")),
                "aic": _format_metric_value(row.get("aic", "-")),
                "bic": _format_metric_value(row.get("bic", "-")),
                "valid_log_loss": _format_metric_value(valid_metrics.get("log_loss", "-")),
                "valid_accuracy": _format_metric_value(valid_metrics.get("accuracy", "-")),
                "features": ", ".join(row.get("features", [])),
            }
        )

    widths = {
        column: max(len(column), *(len(row[column]) for row in formatted_rows))
        for column in columns
    }

    def render_row(row: dict[str, str]) -> str:
        return " | ".join(row[column].ljust(widths[column]) for column in columns)

    header = render_row({column: column for column in columns})
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = "\n".join(render_row(row) for row in formatted_rows)
    return f"{title}\n{header}\n{divider}\n{body}"


def _format_forward_selection_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)

    best_aic = result["best_aic_model"]
    best_bic = result["best_bic_model"]
    lines = [
        f"status: {result['status']}",
        f"rows: {result['rows']}",
        f"rows_train: {result['rows_train']}",
        f"rows_valid: {result['rows_valid']}",
        "",
        f"best_aic_features: {', '.join(best_aic['features'])}",
        f"best_aic_value: {best_aic['aic']:.4f}",
        f"best_aic_valid_log_loss: {best_aic['valid_metrics']['log_loss']:.4f}",
        f"best_bic_features: {', '.join(best_bic['features'])}",
        f"best_bic_value: {best_bic['bic']:.4f}",
        f"best_bic_valid_log_loss: {best_bic['valid_metrics']['log_loss']:.4f}",
        "",
        _format_forward_selection_table("AIC Forward Steps", result.get("aic_steps", [])),
        "",
        _format_forward_selection_table("BIC Forward Steps", result.get("bic_steps", [])),
    ]
    return "\n".join(lines)


def _format_settled_prediction_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)

    lines = [
        f"status: {result['status']}",
        f"rows: {result['rows']}",
        "",
        _format_metrics_table("Settled Metrics", result.get("models", {})),
    ]
    recent = result.get("recent", [])
    if recent:
        lines.extend(["", f"recent_rows: {len(recent)}"])
    return "\n".join(lines)


def _format_settled_window_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)

    rows: list[dict[str, Any]] = []
    for model_name, windows in result.get("windows", {}).items():
        row = {"model": model_name}
        for window_name in ("all", "last_7d", "last_30d", "last_50_games"):
            metrics = windows.get(window_name, {})
            row[f"{window_name}_games"] = metrics.get("games", 0)
            row[f"{window_name}_accuracy"] = metrics.get("accuracy", "-")
            row[f"{window_name}_log_loss"] = metrics.get("log_loss", "-")
        rows.append(row)

    columns = [
        "model",
        "all_games",
        "all_accuracy",
        "all_log_loss",
        "last_7d_games",
        "last_7d_accuracy",
        "last_7d_log_loss",
        "last_30d_games",
        "last_30d_accuracy",
        "last_30d_log_loss",
        "last_50_games_games",
        "last_50_games_accuracy",
        "last_50_games_log_loss",
    ]

    formatted_rows: list[dict[str, str]] = []
    for row in rows:
        formatted_rows.append({column: _format_metric_value(row.get(column, "-")) for column in columns})

    widths = {
        column: max(len(column), *(len(row[column]) for row in formatted_rows))
        for column in columns
    }

    def render_row(row: dict[str, str]) -> str:
        return " | ".join(row[column].ljust(widths[column]) for column in columns)

    header = render_row({column: column for column in columns})
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = "\n".join(render_row(row) for row in formatted_rows)

    lines = [
        f"status: {result['status']}",
        f"rows: {result['rows']}",
        "",
        "Settled Windows",
        header,
        divider,
        body,
    ]
    daily = result.get("daily", [])
    if daily:
        lines.extend(["", f"daily_rows: {len(daily)}"])
    return "\n".join(lines)


def _format_bet_opportunity_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)
    lines = [
        f"status: {result['status']}",
        f"rows: {result['rows']}",
        f"champion_model: {result.get('champion_model')}",
        "",
        _format_metrics_table("Bet Opportunities", result.get("models", {})),
    ]
    return "\n".join(lines)


def _format_strategy_performance_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)

    rows: list[dict[str, Any]] = []
    for model_name, windows in result.get("windows", {}).items():
        row = {"model": model_name}
        for window_name in ("all", "last_7d", "last_30d", "last_50_bets"):
            metrics = windows.get(window_name, {})
            row[f"{window_name}_bets"] = metrics.get("bets", 0)
            row[f"{window_name}_roi"] = metrics.get("roi", 0.0)
            row[f"{window_name}_units"] = metrics.get("units_won", 0.0)
            row[f"{window_name}_edge_bps"] = metrics.get("avg_edge_bps", 0.0)
        rows.append(row)

    columns = [
        "model",
        "all_bets",
        "all_roi",
        "all_units",
        "all_edge_bps",
        "last_7d_bets",
        "last_7d_roi",
        "last_7d_units",
        "last_7d_edge_bps",
        "last_30d_bets",
        "last_30d_roi",
        "last_30d_units",
        "last_30d_edge_bps",
        "last_50_bets_bets",
        "last_50_bets_roi",
        "last_50_bets_units",
        "last_50_bets_edge_bps",
    ]
    formatted_rows = [{column: _format_metric_value(row.get(column, "-")) for column in columns} for row in rows]
    widths = {column: max(len(column), *(len(row[column]) for row in formatted_rows)) for column in columns}

    def render_row(row: dict[str, str]) -> str:
        return " | ".join(row[column].ljust(widths[column]) for column in columns)

    header = render_row({column: column for column in columns})
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = "\n".join(render_row(row) for row in formatted_rows)

    lines = [
        f"status: {result['status']}",
        f"rows: {result['rows']}",
        f"champion_model: {result.get('champion_model')}",
        "",
        "Strategy Performance",
        header,
        divider,
        body,
    ]
    daily = result.get("daily", [])
    if daily:
        lines.extend(["", f"daily_rows: {len(daily)}"])
    return "\n".join(lines)


def _format_historical_import_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)
    lines = [
        f"status: {result['status']}",
        f"import_run_id: {result['import_run_id']}",
        f"request_count: {result['request_count']}",
        f"payload_count: {result['payload_count']}",
        f"normalized_rows: {result['normalized_rows']}",
    ]
    if "chunks_total" in result:
        lines.extend(
            [
                f"chunks_total: {result['chunks_total']}",
                f"chunks_completed: {result['chunks_completed']}",
                f"chunks_skipped: {result['chunks_skipped']}",
            ]
        )
    return "\n".join(lines)


def _format_historical_status_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)
    lines = [
        f"status: {result['status']}",
        f"rows: {result['rows']}",
        "",
        _format_metrics_table("Historical Import Status", result.get("sources", {})),
    ]
    return "\n".join(lines)


def app() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "collect-once":
        print(collect_snapshot())
        return

    if args.command == "collect-loop":
        run_service(iterations=args.iterations)
        return

    if args.command == "run-service":
        run_service(iterations=args.iterations)
        return

    if args.command == "sync-results":
        print(sync_recent_game_results(lookback_days=args.lookback_days))
        return

    if args.command == "report-settled-predictions":
        print(
            _format_settled_prediction_output(
                run_settled_prediction_report(args.start_date, args.end_date, model_name=args.model_name)
            )
        )
        return

    if args.command == "report-settled-windows":
        print(
            _format_settled_window_output(
                run_settled_window_report(args.start_date, args.end_date, model_name=args.model_name)
            )
        )
        return

    if args.command == "report-bet-opportunities":
        print(_format_bet_opportunity_output(run_bet_opportunity_report(args.start_date, args.end_date, model_name=args.model_name)))
        return

    if args.command == "report-strategy-performance":
        print(
            _format_strategy_performance_output(
                run_strategy_performance_report(args.start_date, args.end_date, model_name=args.model_name)
            )
        )
        return

    if args.command == "historical-backfill-polymarket":
        print(
            _format_historical_import_output(
                backfill_polymarket_history_for_games(
                    args.start_date,
                    args.end_date,
                    interval=args.interval,
                    chunk_days=args.chunk_days,
                    resume=not args.no_resume,
                )
            )
        )
        return

    if args.command == "historical-backfill-kalshi":
        print(
            _format_historical_import_output(
                backfill_kalshi_history_for_games(
                    args.start_date,
                    args.end_date,
                    period_interval=args.period_interval,
                    include_trades=args.include_trades,
                    chunk_days=args.chunk_days,
                    resume=not args.no_resume,
                )
            )
        )
        return

    if args.command == "historical-import-status":
        print(_format_historical_status_output(run_historical_import_status(args.start_date, args.end_date)))
        return

    if args.command == "backtest":
        print(run_backtest(args.start_date, args.end_date))
        return

    if args.command == "train-game-model":
        print(_format_train_game_model_output(train_and_save_model(args.start_date, args.end_date)))
        return

    if args.command == "benchmark-game-model":
        print(_format_benchmark_game_model_output(run_model_benchmark(args.start_date, args.end_date)))
        return

    if args.command == "forward-select-game-features":
        print(_format_forward_selection_output(run_forward_feature_selection(args.start_date, args.end_date)))
        return


if __name__ == "__main__":
    app()
