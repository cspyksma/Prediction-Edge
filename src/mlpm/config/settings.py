from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    duckdb_path: Path = Field(default=Path("data/mlb_markets.duckdb"), alias="DUCKDB_PATH")
    raw_data_dir: Path = Field(default=Path("data/raw"), alias="RAW_DATA_DIR")
    processed_data_dir: Path = Field(default=Path("data/processed"), alias="PROCESSED_DATA_DIR")
    artifacts_dir: Path = Field(default=Path("artifacts"), alias="ARTIFACTS_DIR")
    mlflow_tracking_uri: str = Field(default="file:./mlruns", alias="MLFLOW_TRACKING_URI")
    model_home_field_edge_bps: int = Field(default=350, alias="MODEL_HOME_FIELD_EDGE_BPS")
    model_min_games: int = Field(default=25, alias="MODEL_MIN_GAMES")
    model_selection_metric: str = Field(default="log_loss", alias="MODEL_SELECTION_METRIC")
    model_train_start_date: str = Field(
        default_factory=lambda: f"{date.today().year}-03-01",
        alias="MODEL_TRAIN_START_DATE",
    )
    kalshi_rate_limit_tier: str = Field(default="basic", alias="KALSHI_RATE_LIMIT_TIER")
    kalshi_read_limit_per_second: int | None = Field(default=None, alias="KALSHI_READ_LIMIT_PER_SECOND")
    kalshi_write_limit_per_second: int | None = Field(default=None, alias="KALSHI_WRITE_LIMIT_PER_SECOND")


@lru_cache(maxsize=1)
def settings() -> Settings:
    cfg = Settings()
    cfg.raw_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg
