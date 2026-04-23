# MLPM

Local MLB fair-value research pipeline. The current scope is pregame MLB moneyline markets, live market-edge monitoring, settled performance grading, and historical market-data import staging.

## What It Does

- Pulls upcoming MLB games from the MLB Stats API.
- Pulls current market quotes from Kalshi and Polymarket.
- Trains and serves local MLB game-outcome models for fair win probabilities.
- Maps all quotes to canonical MLB game IDs.
- Builds model-vs-market discrepancies and edge-first bet opportunities.
- Grades stored pregame predictions against actual winners.
- Grades actionable flat-stake opportunities against actual winners and reports ROI.
- Backfills recent final MLB game results during service runs.
- Imports historical Kalshi and Polymarket market data into staging tables for later replay/backtesting work.
- Stores snapshots in DuckDB and raw payloads on disk.
- Exposes live gaps, settled summaries, and strategy performance in Streamlit.
- Ships a new local web terminal with a FastAPI backend and React frontend for cockpit, research, and ops workflows.

## Stack

- Python 3.12 target
- httpx
- pandas
- duckdb
- scikit-learn
- MLflow
- Streamlit
- FastAPI
- React + Vite

## Quick Start

1. Create a Python 3.12 virtual environment.
2. Install dependencies:

```bash
pip install -e .[dev]
```

3. Copy `.env.example` to `.env`.
4. Run one collection cycle:

```bash
python -m mlpm.cli collect-once
```

5. Run the long-lived service:

```bash
python -m mlpm.cli run-service
```

This keeps collecting on a fixed interval, retries after failures, and backfills recent final game results into DuckDB on every cycle.

6. Train the game-outcome model:

```bash
python -m mlpm.cli train-game-model
```

7. Benchmark candidate models:

```bash
python -m mlpm.cli benchmark-game-model --start-date 2026-03-01 --end-date 2026-04-15
python -m mlpm.cli forward-select-game-features --start-date 2026-03-01 --end-date 2026-04-15
```

8. Start the local dashboard:

```bash
streamlit run src/mlpm/app/dashboard.py
```

8a. Start the new local web terminal backend:

```bash
python -m mlpm.cli run-api --reload
```

The API defaults to `http://127.0.0.1:8000` and serves routes under `/api/v1`.

8b. In a second shell, install the frontend dependencies once and start Vite:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173/`.

During local development, Vite proxies `/api/*` to the FastAPI backend on port
`8000`, so the frontend does not need a hard-coded API host. For non-default
deployments, set `VITE_API_BASE` or change the proxy target in
`frontend/vite.config.ts`.

9. Review settled predictions, live opportunities, and settled strategy performance:

```bash
python -m mlpm.cli report-settled-predictions --start-date 2026-04-13 --end-date 2026-04-15
python -m mlpm.cli report-settled-windows --start-date 2026-04-13 --end-date 2026-04-15
python -m mlpm.cli report-bet-opportunities --start-date 2026-04-13 --end-date 2026-04-15
python -m mlpm.cli report-strategy-performance --start-date 2026-04-13 --end-date 2026-04-15
```

10. Historical market-data staging imports:

```bash
python -m mlpm.cli historical-backfill-polymarket --start-date 2026-04-01 --end-date 2026-04-07 --interval 1m --chunk-days 7
python -m mlpm.cli historical-backfill-kalshi --start-date 2026-04-01 --end-date 2026-04-07 --period-interval 1 --chunk-days 7
python -m mlpm.cli historical-import-status --start-date 2026-04-01 --end-date 2026-04-07
```

Historical imports are resumable by default. Re-running the same command skips chunks that already completed successfully. Use `--no-resume` to force a rerun.

## Current Features

### Live collection and scoring

- Upcoming MLB schedule ingestion with probable pitchers.
- Live Kalshi and Polymarket market ingestion.
- Quote normalization into implied probabilities.
- Current model scoring for each team/game.
- Discrepancy generation between market prices and model probabilities.
- Edge-first opportunity generation using the best available current price across sources.

### Baseball features currently in the model pipeline

- Team season and recent win rates.
- Home/away venue win rates.
- Run differential and run scoring/allowing features.
- Rest days, venue streak, travel-switch, and doubleheader features.
- Team Elo.
- Starter ERA, WHIP, strikeouts per 9, and walks per 9.
- Bullpen recent workload features.
- Offense vs opposing starter handedness.
- Stadium weather features when available.

The core baseball model families are trained market-free by default. Market prices
are used in evaluation, replay, and research flows, and some research contenders
intentionally include market inputs when the objective is Kalshi ROI rather than
pure winner prediction.

### Model families currently available

- Logistic regression.
- Histogram gradient boosting.
- K-nearest neighbors.
- SVM.
- PyTorch MLP tabular classifier.
- Heuristic legacy baselines.
- Bayesian posterior path that combines a prior with a baseball-feature evidence model.

## Evaluation and Strategy

### Settled prediction grading

- `report-settled-predictions` grades stored pregame predictions against actual winners.
- `report-settled-windows` summarizes rolling windows such as `all`, `last_7d`, `last_30d`, and `last_50_games`.

### Edge-first strategy grading

- The project is not only tracking winner accuracy.
- It also creates one best market opportunity per game/model snapshot.
- Actionable opportunities are controlled by `STRATEGY_EDGE_THRESHOLD_BPS`.
- `report-strategy-performance` settles those opportunities against actual winners using flat `1.0` unit stakes.
- Champion/challenger reporting is available for current strategy summaries.

### Historical and research backtests

- `historical-backtest-kalshi` runs a fixed train/eval replay backtest against historical Kalshi quotes.
- `walkforward-backtest` runs the monthly retrain SBRO walk-forward backtest.
- `research-kalshi-edge` runs the database-first Kalshi edge research harness with:
  - explicit train and eval windows
  - baseball-only, hybrid, market-aware, market-only, and ensemble contenders
  - multiple entry thresholds and sizing policies
  - time-sliced out-of-sample Kalshi evaluation
  - a research champion selected on Kalshi ROI with consistency guardrails

Example:

```bash
python -m mlpm.cli research-kalshi-edge --train-start-date 2015-01-01 --train-end-date 2021-12-31 --eval-start-date 2025-01-01 --eval-end-date 2026-04-19
```

This command is database-first. It reads local DuckDB tables and does not refetch
MLB fundamentals on every run. Backfill those first:

```bash
python -m mlpm.cli backfill-mlb --start-date 2015-01-01 --end-date 2026-04-19
python -m mlpm.cli backfill-weather --start-date 2015-01-01 --end-date 2026-04-19
python -m mlpm.cli ingest-sbro --directory sbro
```

## Running Unattended

- Use `python -m mlpm.cli run-service` for the durable collector process.
- Use `python -m mlpm.cli run-api` for the local FastAPI backend that powers the React dashboard.
- Set `SNAPSHOT_INTERVAL_SECONDS` for polling cadence.
- Set `RUNNER_FAILURE_BACKOFF_SECONDS` for retry delay after an API or network failure.
- Set `RESULTS_SYNC_LOOKBACK_DAYS` to control how far back the service rechecks completed MLB games.
- Set `STRATEGY_EDGE_THRESHOLD_BPS` to define the minimum actionable edge.
- The service writes run health into DuckDB table `collector_runs` and final scores into `game_results`.
- Strategy rows are written to `bet_opportunities` and settled ROI is available through `settled_bet_opportunities_deduped`.
- Dashboard job launches from the Ops page write logs under `data/dashboard_jobs/` and remain visible after API reloads/restarts.
- On Windows, run the command under Task Scheduler or NSSM so it restarts on login/reboot.

## Data Layout

Main DuckDB tables:

- `games`
- `raw_snapshots`
- `normalized_quotes`
- `model_predictions`
- `discrepancies`
- `bet_opportunities`
- `game_results`
- `mlb_pitching_logs`
- `mlb_batting_logs`
- `game_weather`
- `collector_runs`
- `historical_import_runs`
- `historical_polymarket_quotes`
- `historical_kalshi_quotes`
- `historical_market_priors`

Important derived views:

- `games_deduped`
- `normalized_quotes_deduped`
- `model_predictions_deduped`
- `discrepancies_deduped`
- `game_results_deduped`
- `mlb_pitching_logs_deduped`
- `mlb_batting_logs_deduped`
- `game_weather_deduped`
- `historical_market_priors_deduped`
- `settled_predictions_deduped`
- `settled_prediction_daily`
- `bet_opportunities_deduped`
- `settled_bet_opportunities_deduped`
- `strategy_performance_daily`
- `historical_import_status`

Raw payloads are written under `data/raw/` and historical payloads under `data/raw/historical/`.

## Historical Import Notes

- Historical imports currently stage source-native Kalshi and Polymarket data into dedicated historical tables.
- They do not yet replay that data into the live `normalized_quotes` path.
- They do not yet provide full prior-years market-edge backtesting on their own.
- Polymarket historical discovery is currently inconsistent for MLB slices we tested.
- Kalshi chunking and resume now work, but the tested one-week slice still returned `0` normalized rows, so endpoint yield and ticker matching still need more work before large-scale historical imports will be useful.

## Notes

- Kalshi and Polymarket market discovery are both public, free endpoints.
- If no trained artifact exists yet, the collector falls back to the heuristic baseline so collection still runs.
- The richer benchmark path trains multiple model families; set `MODEL_SELECTION_METRIC=accuracy` in `.env` if you want to prefer winner-pick accuracy over calibrated probabilities.
- Accuracy alone is not the main target. The project is also trying to identify model/market probability gaps that may be profitable.

## Public Release Checklist

- Keep `.env`, `data/*.duckdb`, raw payloads, artifacts, MLflow runs, and temp/cache directories out of Git.
- Keep local SBRO workbooks and generated backtest reports out of Git.
- Review `.gitignore` before pushing any new local data collection outputs.
- Treat the repo as code-only by default; do not publish collected market snapshots or databases unless you explicitly intend to.
- Re-run the test suite before pushing significant changes:

```bash
python -m pytest tests
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
