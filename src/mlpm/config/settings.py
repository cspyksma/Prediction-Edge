from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mlb_lookahead_days: int = Field(default=3, alias="MLB_LOOKAHEAD_DAYS")
    snapshot_interval_seconds: int = Field(default=60, alias="SNAPSHOT_INTERVAL_SECONDS")
    runner_failure_backoff_seconds: int = Field(default=30, alias="RUNNER_FAILURE_BACKOFF_SECONDS")
    results_sync_lookback_days: int = Field(default=7, alias="RESULTS_SYNC_LOOKBACK_DAYS")
    freshness_window_seconds: int = Field(default=180, alias="FRESHNESS_WINDOW_SECONDS")
    discrepancy_threshold_bps: int = Field(default=300, alias="DISCREPANCY_THRESHOLD_BPS")
    strategy_edge_threshold_bps: int = Field(default=500, alias="STRATEGY_EDGE_THRESHOLD_BPS")
    strategy_champion_window_days: int = Field(default=30, alias="STRATEGY_CHAMPION_WINDOW_DAYS")
    strategy_champion_min_bets: int = Field(default=10, alias="STRATEGY_CHAMPION_MIN_BETS")
    strategy_champion_ci_confidence: float = Field(default=0.95, alias="STRATEGY_CHAMPION_CI_CONFIDENCE")
    strategy_champion_ci_method: str = Field(default="wilson", alias="STRATEGY_CHAMPION_CI_METHOD")
    strategy_champion_bootstrap_samples: int = Field(default=1000, alias="STRATEGY_CHAMPION_BOOTSTRAP_SAMPLES")
    strategy_champion_bootstrap_seed: int = Field(default=42, alias="STRATEGY_CHAMPION_BOOTSTRAP_SEED")
    duckdb_path: Path = Field(default=REPO_ROOT / "data" / "mlb_markets.duckdb", alias="DUCKDB_PATH")
    raw_data_dir: Path = Field(default=REPO_ROOT / "data" / "raw", alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=REPO_ROOT / "data" / "processed", alias="PROCESSED_DATA_DIR")
    artifacts_dir: Path = Field(default=REPO_ROOT / "artifacts", alias="ARTIFACTS_DIR")
    mlflow_tracking_uri: str = Field(default=f"file:{(REPO_ROOT / 'mlruns').as_posix()}", alias="MLFLOW_TRACKING_URI")
    model_home_field_edge_bps: int = Field(default=350, alias="MODEL_HOME_FIELD_EDGE_BPS")
    model_min_games: int = Field(default=25, alias="MODEL_MIN_GAMES")
    model_selection_metric: str = Field(default="log_loss", alias="MODEL_SELECTION_METRIC")
    betting_stats_start_date: str = Field(default="2025-01-01", alias="BETTING_STATS_START_DATE")
    # Default covers the full available historical record: SBRO closing-line
    # moneylines 2015-2021, plus Kalshi ticker-based replay 2025+, plus live
    # quotes from the current season. Earlier defaults silently restricted
    # training to the current season, which is why prior runs only learned
    # from ~2025 data. Override via MODEL_TRAIN_START_DATE if needed.
    model_train_start_date: str = Field(
        default="2015-03-01",
        alias="MODEL_TRAIN_START_DATE",
    )
    kalshi_rate_limit_tier: str = Field(default="basic", alias="KALSHI_RATE_LIMIT_TIER")
    kalshi_read_limit_per_second: int | None = Field(default=None, alias="KALSHI_READ_LIMIT_PER_SECOND")
    kalshi_write_limit_per_second: int | None = Field(default=None, alias="KALSHI_WRITE_LIMIT_PER_SECOND")
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    frontend_dev_origin: str = Field(default="http://127.0.0.1:5173", alias="FRONTEND_DEV_ORIGIN")


@lru_cache(maxsize=1)
def settings() -> Settings:
    cfg = Settings()
    cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg
