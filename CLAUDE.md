# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in dev mode
pip install -e .[dev]

# Run all tests
python -m pytest tests

# Run a single test file
python -m pytest tests/test_game_outcome_model.py

# Run a single test by name
python -m pytest tests/test_game_outcome_model.py::test_train_model

# Launch dashboard
streamlit run src/mlpm/app/dashboard.py

# Key CLI commands
mlpm collect-once                        # Single snapshot collection cycle
mlpm collect-loop [--iterations N]       # Continuous collection
mlpm run-service [--iterations N]        # Service with retries + health logging
mlpm train-game-model [--dates]          # Train & persist ML model
mlpm benchmark-game-model [--dates]      # Compare algorithms
mlpm report-settled-predictions          # Grade prediction accuracy
mlpm report-settled-windows              # Rolling performance windows
mlpm report-strategy-performance         # Grade betting ROI
mlpm sync-results                        # Backfill final MLB game results
mlpm historical-backtest-kalshi          # Replay-based historical backtest
```

## Architecture

MLPM is a local research pipeline that finds fair-value discrepancies between an ML model's game outcome probabilities and live prediction market prices (Kalshi, Polymarket).

### Data Flow

**`collect_snapshot()`** is the core cycle (orchestrated by `pipeline/collect.py`):
1. Fetch upcoming games from MLB Stats API → `games` table
2. Pull live quotes from Kalshi + Polymarket → raw JSON payloads in `data/raw/`
3. Map quotes to games by team name + timestamp → `normalized_quotes` table
4. Score games with trained model → `model_predictions` table
5. Diff model probability vs. market-implied probability → `discrepancies` table
6. Select champion model (best recent ROI) and flag best edge per game → `bet_opportunities` table

### Module Map

| Module | Role |
|---|---|
| `ingest/` | Connectors for MLB Stats API, Kalshi, Polymarket |
| `normalize/` | Quote-to-game mapping (`mapping.py`) and price→probability conversion (`probability.py`) |
| `features/` | Feature engineering: team strength (Elo, win%), pitching stats (ERA/WHIP), matchup features, market prior |
| `models/` | Multi-algorithm game outcome model (LogReg, HistGBM, KNN, SVM, Bayesian); MLflow tracking |
| `pipeline/` | Orchestration: `collect.py` (single cycle), `runner.py` (service loop with retry/health) |
| `evaluation/` | `settled.py` grades prediction accuracy; `strategy.py` grades flat-stake betting ROI |
| `storage/duckdb.py` | All DB operations — single DuckDB file (`data/mlb_markets.duckdb`) with 14 tables + views |
| `historical/` | Backfill and replay of historical Kalshi/Polymarket market data |
| `backtest/` | Historical strategy simulation |
| `app/dashboard.py` | Streamlit dashboard for discrepancies, model grades, and strategy ROI |
| `config/settings.py` | Pydantic settings from `.env` (collection interval, edge thresholds, paths, etc.) |
| `cli.py` | `argparse`-based entry point for all commands |

### Storage

- **DuckDB** at `data/mlb_markets.duckdb` — primary store for all structured data
- **Raw JSON payloads** in `data/raw/{kalshi,polymarket,mlb_stats}/` — timestamped files
- **ML artifacts** in `artifacts/` — persisted model files
- **MLflow runs** in `mlruns/` — experiment tracking

Key DuckDB views: `*_deduped` (latest snapshot per game), `settled_predictions_deduped` (outcomes + predictions joined), `strategy_performance_daily` (daily ROI).

### Configuration

Copy `.env.example` to `.env`. Key knobs:
- `DISCREPANCY_THRESHOLD_BPS` (default 300) — minimum gap to flag a discrepancy
- `STRATEGY_EDGE_THRESHOLD_BPS` (default 500) — minimum edge for a bet opportunity
- `STRATEGY_CHAMPION_WINDOW_DAYS` / `STRATEGY_CHAMPION_MIN_BETS` — champion model selection criteria
- `SNAPSHOT_INTERVAL_SECONDS` — service collection cadence
- `DUCKDB_PATH`, `RAW_DATA_DIR`, `ARTIFACTS_DIR` — storage locations

### Model

`models/game_outcome.py` trains on 23 engineered features (team strength, pitching matchup, market prior). The champion model (best settled ROI over the last 30 days with ≥10 bets) is selected each cycle to generate `bet_opportunities`.
