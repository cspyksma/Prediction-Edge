from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from typing import Any

from mlpm.backtest.run_backtest import run_backtest
from mlpm.backtest.walkforward import DEFAULT_EDGE_THRESHOLD, run_walkforward_backtest
from mlpm.backtest.research import run_kalshi_edge_research_backtest
from mlpm.pipeline.collect import collect_snapshot
from mlpm.pipeline.runner import run_service, sync_recent_game_results
from mlpm.config.settings import settings
from mlpm.evaluation.settled import run_settled_prediction_report, run_settled_window_report
from mlpm.evaluation.strategy import run_bet_opportunity_report, run_strategy_performance_report
from mlpm.historical.mlb_backfill import run_mlb_backfill
from mlpm.historical.polymarket_backfill import backfill_polymarket_history_for_games
from mlpm.historical.status import run_historical_import_status
from mlpm.ingest.sbro import ingest_sbro_directory
from mlpm.models.game_outcome import (
    run_historical_kalshi_backtest,
    run_forward_feature_selection,
    run_model_benchmark,
    train_and_save_model,
)


def _iso_date_arg(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _validate_date_range(start_date: str, end_date: str, *, label: str) -> None:
    if date.fromisoformat(start_date) > date.fromisoformat(end_date):
        raise SystemExit(f"{label} start_date must be on or before end_date.")


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
    settled.add_argument("--start-date", required=True, type=_iso_date_arg)
    settled.add_argument("--end-date", required=True, type=_iso_date_arg)
    settled.add_argument("--model-name")

    settled_windows = subparsers.add_parser("report-settled-windows", help="Summarize rolling settled prediction performance.")
    settled_windows.add_argument("--start-date", required=True, type=_iso_date_arg)
    settled_windows.add_argument("--end-date", required=True, type=_iso_date_arg)
    settled_windows.add_argument("--model-name")

    opportunities = subparsers.add_parser("report-bet-opportunities", help="Summarize live edge opportunities by model.")
    opportunities.add_argument("--start-date", required=True, type=_iso_date_arg)
    opportunities.add_argument("--end-date", required=True, type=_iso_date_arg)
    opportunities.add_argument("--model-name")

    strategy = subparsers.add_parser("report-strategy-performance", help="Summarize settled flat-stake betting performance.")
    strategy.add_argument("--start-date", required=True, type=_iso_date_arg)
    strategy.add_argument("--end-date", required=True, type=_iso_date_arg)
    strategy.add_argument("--model-name")

    historical_poly = subparsers.add_parser("historical-backfill-polymarket", help="Backfill historical Polymarket market prices.")
    historical_poly.add_argument("--start-date", required=True, type=_iso_date_arg)
    historical_poly.add_argument("--end-date", required=True, type=_iso_date_arg)
    historical_poly.add_argument("--interval", default="1m")
    historical_poly.add_argument("--chunk-days", type=int, default=7)
    historical_poly.add_argument("--no-resume", action="store_true")

    historical_kalshi = subparsers.add_parser("historical-backfill-kalshi", help="Backfill historical Kalshi market prices.")
    historical_kalshi.add_argument("--start-date", required=True, type=_iso_date_arg)
    historical_kalshi.add_argument("--end-date", required=True, type=_iso_date_arg)
    historical_kalshi.add_argument("--period-interval", type=int, default=1)
    historical_kalshi.add_argument("--include-trades", action="store_true")
    historical_kalshi.add_argument("--chunk-days", type=int, default=7)
    historical_kalshi.add_argument("--no-resume", action="store_true")

    backfill_mlb = subparsers.add_parser(
        "backfill-mlb",
        help="Backfill MLB Stats API game results + pitching + batting logs into DuckDB (runs locally, hits statsapi.mlb.com).",
    )
    backfill_mlb.add_argument("--start-date", required=True, type=_iso_date_arg)
    backfill_mlb.add_argument("--end-date", required=True, type=_iso_date_arg)
    backfill_mlb.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch months that already have data (otherwise idempotent skip).",
    )

    backfill_weather = subparsers.add_parser(
        "backfill-weather",
        help="Backfill per-game stadium weather from Open-Meteo into DuckDB (free, no API key).",
    )
    backfill_weather.add_argument("--start-date", required=True, type=_iso_date_arg)
    backfill_weather.add_argument("--end-date", required=True, type=_iso_date_arg)
    backfill_weather.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-fetch games that already have weather stored (otherwise skip them).",
    )

    ingest_sbro = subparsers.add_parser(
        "ingest-sbro",
        help="Parse every `mlb odds YYYY.xlsx` in a directory and load de-vigged closing lines into DuckDB.",
    )
    ingest_sbro.add_argument(
        "--directory",
        required=True,
        help="Path to a directory of SBRO xlsx files (download from sportsbookreviewsonline.com).",
    )

    walkforward = subparsers.add_parser(
        "walkforward-backtest",
        help="Walk-forward, retrain-monthly backtest of the market-free model against SBRO closing lines.",
    )
    walkforward.add_argument("--start-date", required=True, type=_iso_date_arg)
    walkforward.add_argument("--end-date", required=True, type=_iso_date_arg)
    walkforward.add_argument(
        "--warmup-start",
        type=_iso_date_arg,
        help="Earliest training history (defaults to 2 years before --start-date).",
    )
    walkforward.add_argument(
        "--edge-threshold",
        type=float,
        default=DEFAULT_EDGE_THRESHOLD,
        help="Flag bets where model_prob - market_fair_prob >= this (default 0.03 = 3pp).",
    )
    walkforward.add_argument(
        "--model",
        choices=("all", "logreg", "histgb", "knn", "svm"),
        default="all",
        help="Classifier to retrain each month, or 'all' to run the full candidate set.",
    )
    walkforward.add_argument(
        "--stake",
        type=float,
        default=1.0,
        help="Flat stake per bet (default 1.0 unit).",
    )
    walkforward.add_argument(
        "--min-train-rows",
        type=int,
        default=500,
        help="Skip months whose training history is smaller than this.",
    )
    walkforward.add_argument(
        "--run-id",
        help="Optional run identifier for the walkforward_bets log (auto-generated if absent).",
    )

    walkforward_report = subparsers.add_parser(
        "walkforward-report",
        help="Generate an xlsx report from walkforward_bets with summary + monthly + yearly + edge-bucket tabs.",
    )
    walkforward_report.add_argument("--output", required=True, help="Output xlsx path.")
    walkforward_report.add_argument(
        "--run-id",
        help="Run to include (defaults to the most recent run in walkforward_bets).",
    )
    walkforward_report.add_argument(
        "--no-recalc",
        action="store_true",
        help="Skip LibreOffice formula recalculation; Excel will recalc on open either way.",
    )

    historical_status = subparsers.add_parser("historical-import-status", help="Show historical import status by source.")
    historical_status.add_argument("--start-date", required=True, type=_iso_date_arg)
    historical_status.add_argument("--end-date", required=True, type=_iso_date_arg)

    historical_backtest = subparsers.add_parser("historical-backtest-kalshi", help="Backtest tabular models against replay-selected Kalshi pregame quotes.")
    historical_backtest.add_argument("--start-date", type=_iso_date_arg)
    historical_backtest.add_argument("--end-date", type=_iso_date_arg)
    historical_backtest.add_argument("--train-start-date", type=_iso_date_arg)
    historical_backtest.add_argument("--train-end-date", type=_iso_date_arg)
    historical_backtest.add_argument("--eval-start-date", type=_iso_date_arg)
    historical_backtest.add_argument("--eval-end-date", type=_iso_date_arg)

    kalshi_research = subparsers.add_parser(
        "research-kalshi-edge",
        help="Run the fixed-split Kalshi edge research backtest over contender families and strategy variants.",
    )
    kalshi_research.add_argument("--train-start-date", required=True, type=_iso_date_arg)
    kalshi_research.add_argument("--train-end-date", required=True, type=_iso_date_arg)
    kalshi_research.add_argument("--eval-start-date", required=True, type=_iso_date_arg)
    kalshi_research.add_argument("--eval-end-date", required=True, type=_iso_date_arg)

    backtest = subparsers.add_parser("backtest", help="Run a simple backtest over stored data.")
    backtest.add_argument("--start-date", required=True, type=_iso_date_arg)
    backtest.add_argument("--end-date", required=True, type=_iso_date_arg)

    train_model = subparsers.add_parser("train-game-model", help="Train and persist the MLB game-outcome model.")
    train_model.add_argument("--start-date", default=settings().model_train_start_date, type=_iso_date_arg)
    train_model.add_argument("--end-date", default=date.today().isoformat(), type=_iso_date_arg)

    benchmark_model = subparsers.add_parser("benchmark-game-model", help="Benchmark the game-outcome model against simple baselines.")
    benchmark_model.add_argument("--start-date", default=settings().model_train_start_date, type=_iso_date_arg)
    benchmark_model.add_argument("--end-date", default=date.today().isoformat(), type=_iso_date_arg)

    select_features = subparsers.add_parser(
        "forward-select-game-features",
        help="Run forward logistic feature selection scored by AIC and BIC.",
    )
    select_features.add_argument("--start-date", default=settings().model_train_start_date, type=_iso_date_arg)
    select_features.add_argument("--end-date", default=date.today().isoformat(), type=_iso_date_arg)

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
    for key in (
        "games_total",
        "games_with_markets",
        "games_with_pregame_quotes",
        "candidate_markets",
        "empty_payload_count",
        "rate_limited_count",
        "parse_error_count",
    ):
        if key in result:
            lines.append(f"{key}: {result[key]}")
    if "chunks_total" in result:
        lines.extend(
            [
                f"chunks_total: {result['chunks_total']}",
                f"chunks_completed: {result['chunks_completed']}",
                f"chunks_skipped: {result['chunks_skipped']}",
            ]
        )
    return "\n".join(lines)


def _format_walkforward_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)
    model_names = result.get("model_names") or [result["model_name"]]
    lines = [
        f"status: {result['status']}",
        f"run_id: {result['run_id']}",
        f"model_name: {result['model_name']}",
        f"model_names: {', '.join(model_names)}",
        f"edge_threshold: {result['edge_threshold']}",
        f"eval_window: {result['eval_start']} to {result['eval_end']}",
        f"warmup_start: {result['warmup_start']}",
        f"total_bets: {result['total_bets']}",
        f"wins: {result['wins']}",
        f"losses: {result['losses']}",
        f"hit_rate: {result['hit_rate']:.4f}",
        f"stake_total: {result['stake_total']:.2f}",
        f"payout_total: {result['payout_total']:.2f}",
        f"roi: {result['roi']:.4f}",
    ]
    model_totals = result.get("models", {})
    if model_totals:
        lines.append("")
        lines.append("Per-Model")
        lines.append("model | bets | wins | losses | hit_rate | stake | payout | roi")
        lines.append("---------------------------------------------------------------")
        for current_model_name, metrics in model_totals.items():
            lines.append(
                f"{current_model_name} | "
                f"{metrics['total_bets']:>4} | {metrics['wins']:>4} | {metrics['losses']:>6} | "
                f"{metrics['hit_rate']:.4f} | {metrics['stake_total']:.2f} | "
                f"{metrics['payout_total']:.2f} | {metrics['roi']:.4f}"
            )
    monthly = result.get("monthly", [])
    if monthly:
        lines.append("")
        lines.append("Per-Month (first 36 rows)")
        header = "model | month | train | scored | matched | bets | wins | losses | net"
        lines.append(header)
        lines.append("-" * len(header))
        for m in monthly[:36]:
            lines.append(
                f"{m['model_name']} | {str(m['month_start'])[:7]} | "
                f"{m['train_rows']:>5} | {m['scored_games']:>6} | "
                f"{m['matched_to_market']:>7} | {m['flagged_bets']:>4} | "
                f"{m['wins']:>4} | {m['losses']:>6} | {m['payout']:>7.2f}"
            )
        if len(monthly) > 36:
            lines.append(f"... ({len(monthly) - 36} more months in walkforward_bets for run_id={result['run_id']})")
    return "\n".join(lines)


def _format_sbro_ingest_output(result: dict[str, Any]) -> str:
    if result.get("status") not in {"ok", "no_files"}:
        return str(result)
    lines = [
        f"status: {result['status']}",
        f"files: {result.get('files', 0)}",
        f"rows_inserted: {result.get('rows_inserted', 0)}",
        f"rows_unmatched: {result.get('rows_unmatched', 0)}",
        f"rows_total: {result.get('rows_total', 0)}",
    ]
    per_year = result.get("per_year", [])
    if per_year:
        lines.append("")
        lines.append("Per-Workbook")
        header = "year | file | inserted | unmatched | total | status"
        lines.append(header)
        lines.append("-" * len(header))
        for y in per_year:
            lines.append(
                f"{y.get('year')} | {y.get('file')} | "
                f"{y.get('rows_inserted', 0)} | {y.get('rows_unmatched', 0)} | "
                f"{y.get('rows_total', 0)} | {y.get('status') or y.get('error', '-')}"
            )
    return "\n".join(lines)


def _format_mlb_backfill_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)
    lines = [
        f"status: {result['status']}",
        f"months: {result['months']}",
        f"games_added: {result['games_added']}",
        f"pitching_rows_added: {result['pitching_rows_added']}",
        f"batting_rows_added: {result['batting_rows_added']}",
        f"elapsed_seconds: {result['elapsed_seconds']:.1f}",
    ]
    monthly = result.get("monthly", [])
    if monthly:
        lines.append("")
        lines.append("Per-Month Progress")
        header = "year-month | games | pitching | batting | skipped | elapsed_s"
        lines.append(header)
        lines.append("-" * len(header))
        for m in monthly:
            lines.append(
                f"{m['year']:04d}-{m['month']:02d}    | "
                f"{m['games']:<5} | {m['pitching_rows']:<8} | {m['batting_rows']:<7} | "
                f"{str(m['skipped']):<7} | {m['elapsed_seconds']}"
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


def _format_historical_backtest_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)
    lines = [
        f"status: {result['status']}",
        f"rows: {result['rows']}",
        f"rows_train: {result['rows_train']}",
        f"rows_valid: {result['rows_valid']}",
        f"train_window: {result['train_start_date']} to {result['train_end_date']}",
        f"eval_window: {result['eval_start_date']} to {result['eval_end_date']}",
        f"replay_rows: {result['replay_rows']}",
        f"valid_replay_rows: {result['valid_replay_rows']}",
        f"champion_model: {result['champion_model']}",
        "",
        _format_metrics_table("Historical Kalshi Backtest", result.get("benchmarks", {})),
    ]
    calibration = result.get("calibration", [])
    if calibration:
        lines.extend(["", f"calibration_rows: {len(calibration)}"])
    return "\n".join(lines)


def _format_kalshi_research_output(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result)
    lines = [
        f"status: {result['status']}",
        f"rows: {result['rows']}",
        f"rows_train: {result['rows_train']}",
        f"rows_valid: {result['rows_valid']}",
        f"train_window: {result['train_start_date']} to {result['train_end_date']}",
        f"eval_window: {result['eval_start_date']} to {result['eval_end_date']}",
        f"replay_rows: {result['replay_rows']}",
        f"valid_replay_rows: {result['valid_replay_rows']}",
        f"contender_count: {result['contender_count']}",
        f"strategy_count: {result['strategy_count']}",
        f"champion_strategy: {result['champion_strategy']}",
        f"champion_contender: {result['champion_contender']}",
        f"guardrail_champion: {result['guardrail_champion']}",
    ]
    contenders = {
        row["contender_name"]: {
            "family": row["family"],
            "model_family": row["model_family"],
            "feature_variant": row["feature_variant"],
            "accuracy": row["accuracy"],
            "roc_auc": row["roc_auc"],
            "log_loss": row["log_loss"],
        }
        for row in result.get("contenders", [])
    }
    if contenders:
        lines.extend(["", _format_metrics_table("Contenders", contenders)])
    strategies = result.get("strategies", [])
    if strategies:
        lines.extend(["", "Top Strategies"])
        header = "strategy | family | bets | roi | units | drawdown | guardrails"
        lines.append(header)
        lines.append("-" * len(header))
        for row in strategies[:20]:
            lines.append(
                f"{row['strategy_name']} | {row['family']} | {row['bets']} | "
                f"{row['roi']:.4f} | {row['units_won']:.2f} | {row['max_drawdown']:.2f} | "
                f"{row['guardrails_passed']}"
            )
    calibration = result.get("calibration", [])
    if calibration:
        lines.extend(["", f"calibration_rows: {len(calibration)}"])
    return "\n".join(lines)


def app() -> None:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in {
        "report-settled-predictions",
        "report-settled-windows",
        "report-bet-opportunities",
        "report-strategy-performance",
        "historical-backfill-polymarket",
        "historical-backfill-kalshi",
        "backfill-mlb",
        "backfill-weather",
        "walkforward-backtest",
        "historical-import-status",
        "backtest",
        "train-game-model",
        "benchmark-game-model",
        "forward-select-game-features",
    }:
        _validate_date_range(args.start_date, args.end_date, label=args.command)

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
        from mlpm.historical.kalshi_backfill import backfill_kalshi_history_for_games
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

    if args.command == "backfill-mlb":
        print(
            _format_mlb_backfill_output(
                run_mlb_backfill(args.start_date, args.end_date, force=args.force)
            )
        )
        return

    if args.command == "backfill-weather":
        from mlpm.ingest.weather import backfill_weather as _backfill_weather
        result = _backfill_weather(
            args.start_date,
            args.end_date,
            resume=not args.no_resume,
        )
        lines = [f"{k}: {v}" for k, v in result.items()]
        print("\n".join(lines))
        return

    if args.command == "ingest-sbro":
        print(_format_sbro_ingest_output(ingest_sbro_directory(args.directory)))
        return

    if args.command == "walkforward-backtest":
        print(
            _format_walkforward_output(
                run_walkforward_backtest(
                    args.start_date,
                    args.end_date,
                    warmup_start=args.warmup_start,
                    edge_threshold=args.edge_threshold,
                    model_name=args.model,
                    min_train_rows=args.min_train_rows,
                    stake=args.stake,
                    run_id=args.run_id,
                )
            )
        )
        return

    if args.command == "walkforward-report":
        from mlpm.backtest.report import generate_walkforward_report

        result = generate_walkforward_report(
            args.output,
            run_id=args.run_id,
            recalc=not args.no_recalc,
        )
        lines = [f"{k}: {v}" for k, v in result.items()]
        print("\n".join(lines))
        return

    if args.command == "historical-import-status":
        print(_format_historical_status_output(run_historical_import_status(args.start_date, args.end_date)))
        return

    if args.command == "historical-backtest-kalshi":
        eval_start_date = args.eval_start_date or args.start_date
        eval_end_date = args.eval_end_date or args.end_date
        if not eval_start_date or not eval_end_date:
            raise SystemExit("historical-backtest-kalshi requires either --eval-start-date/--eval-end-date or --start-date/--end-date.")
        _validate_date_range(eval_start_date, eval_end_date, label="historical-backtest-kalshi eval")
        train_start_date = args.train_start_date or settings().model_train_start_date
        train_end_date = args.train_end_date
        if train_end_date is None:
            train_end_date = (date.fromisoformat(eval_start_date) - timedelta(days=1)).isoformat()
        _validate_date_range(train_start_date, train_end_date, label="historical-backtest-kalshi train")
        print(
            _format_historical_backtest_output(
                run_historical_kalshi_backtest(
                    train_start_date=train_start_date,
                    train_end_date=train_end_date,
                    eval_start_date=eval_start_date,
                    eval_end_date=eval_end_date,
                )
            )
        )
        return

    if args.command == "research-kalshi-edge":
        _validate_date_range(args.train_start_date, args.train_end_date, label="research-kalshi-edge train")
        _validate_date_range(args.eval_start_date, args.eval_end_date, label="research-kalshi-edge eval")
        print(
            _format_kalshi_research_output(
                run_kalshi_edge_research_backtest(
                    train_start_date=args.train_start_date,
                    train_end_date=args.train_end_date,
                    eval_start_date=args.eval_start_date,
                    eval_end_date=args.eval_end_date,
                )
            )
        )
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
