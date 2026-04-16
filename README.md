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

## Stack

- Python 3.12 target
- httpx
- pandas
- duckdb
- scikit-learn
- MLflow
- Streamlit

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
- Market-implied prior (`market_home_implied_prob`).
- Offense vs opposing starter handedness.

### Model families currently available

- Logistic regression.
- Histogram gradient boosting.
- K-nearest neighbors.
- SVM.
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

## Running Unattended

- Use `python -m mlpm.cli run-service` for the durable collector process.
- Set `SNAPSHOT_INTERVAL_SECONDS` for polling cadence.
- Set `RUNNER_FAILURE_BACKOFF_SECONDS` for retry delay after an API or network failure.
- Set `RESULTS_SYNC_LOOKBACK_DAYS` to control how far back the service rechecks completed MLB games.
- Set `STRATEGY_EDGE_THRESHOLD_BPS` to define the minimum actionable edge.
- The service writes run health into DuckDB table `collector_runs` and final scores into `game_results`.
- Strategy rows are written to `bet_opportunities` and settled ROI is available through `settled_bet_opportunities_deduped`.
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
- `collector_runs`
- `historical_import_runs`
- `historical_polymarket_quotes`
- `historical_kalshi_quotes`

Important derived views:

- `games_deduped`
- `normalized_quotes_deduped`
- `model_predictions_deduped`
- `discrepancies_deduped`
- `game_results_deduped`
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
