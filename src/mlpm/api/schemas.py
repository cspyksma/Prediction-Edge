from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SummaryResponse(BaseModel):
    total_discrepancies: int = 0
    flagged_discrepancies: int = 0
    avg_abs_gap_bps: float = 0.0
    champion_model: str | None = None
    latest_snapshot_ts: str | None = None
    latest_game_date: str | None = None
    actionable_bets: int = 0
    max_edge_bps: int = 0
    stale_data: bool = False
    freshness_window_seconds: int


class OpportunityRow(BaseModel):
    game_id: str
    game_date: str | None = None
    event_start_time: str | None = None
    model_name: str
    source: str | None = None
    market_id: str | None = None
    team: str
    opponent_team: str | None = None
    model_prob: float
    market_prob: float
    edge_bps: int
    expected_value: float | None = None
    is_actionable: bool = False
    is_champion: bool = False


class OpportunityListResponse(BaseModel):
    items: list[OpportunityRow]
    total: int
    page: int
    page_size: int
    champion_model: str | None = None


class GameQuoteRow(BaseModel):
    source: str | None = None
    market_id: str | None = None
    team: str
    market_prob: float
    model_prob: float
    gap_bps: int
    snapshot_ts: str | None = None
    flagged: bool = False


class GapHistoryRow(BaseModel):
    snapshot_ts: str | None = None
    team: str
    source: str | None = None
    gap_bps: int


class GameFeatureContext(BaseModel):
    team: str
    opponent_team: str | None = None
    model_name: str | None = None
    model_prob: float | None = None
    season_win_pct: float | None = None
    recent_win_pct: float | None = None
    venue_win_pct: float | None = None
    run_diff_per_game: float | None = None
    streak: int | None = None
    elo_rating: float | None = None
    rest_days: float | None = None
    starter_era: float | None = None
    starter_whip: float | None = None
    bullpen_innings_3d: float | None = None


class GameDetailResponse(BaseModel):
    game_id: str
    game_date: str | None = None
    event_start_time: str | None = None
    away_team: str | None = None
    home_team: str | None = None
    quotes: list[GameQuoteRow]
    features: list[GameFeatureContext]
    gap_history: list[GapHistoryRow]


class CalibrationRow(BaseModel):
    model_name: str
    bucket: str
    bucket_lower: float
    bucket_upper: float
    games: int
    avg_predicted_prob: float
    actual_home_win_rate: float
    abs_error: float


class FeatureImportanceRow(BaseModel):
    model_name: str
    feature: str
    importance: float
    importance_std: float | None = None
    rank: int
    trained_at: str | None = None
    train_start_date: str | None = None
    train_end_date: str | None = None
    rows_train: int | None = None
    rows_valid: int | None = None
    method: str | None = None


class ResearchContenderRow(BaseModel):
    model: str
    family: str | None = None
    model_family: str | None = None
    feature_variant: str | None = None
    accuracy: float | None = None
    roc_auc: float | None = None
    log_loss: float | None = None


class ResearchStrategyRow(BaseModel):
    strategy_name: str
    family: str | None = None
    bets: int
    roi: float
    roi_ci_lower: float | None = None
    roi_ci_upper: float | None = None
    units_won: float
    max_drawdown: float | None = None
    positive_slice_rate: float | None = None
    guardrails_passed: bool = False


class ChampionStandingRow(BaseModel):
    model_name: str
    family: str | None = None
    bets: int
    wins: int
    win_rate: float | None = None
    units_won: float | None = None
    roi: float | None = None
    avg_edge_bps: float | None = None
    first_game_date: str | None = None
    last_game_date: str | None = None
    is_champion: bool = False
    ci_lower: float | None = None
    ci_upper: float | None = None
    incumbent_point_metric: float | None = None


class ChampionStandingsResponse(BaseModel):
    betting_stats_start_date: str
    champion_model: str | None = None
    decision_reason: str
    decision_action: str
    rows: list[ChampionStandingRow]


class TrainingCoverageRow(BaseModel):
    source: str
    label: str
    first_date: str | None = None
    last_date: str | None = None
    games_with_prior: int = 0


class TrainingCoverageResponse(BaseModel):
    rows: list[TrainingCoverageRow]
    train_start_date: str | None = None
    train_end_date: str | None = None
    total_games_with_prior: int = 0
    latest_model_train_start: str | None = None
    latest_model_train_end: str | None = None
    latest_model_trained_at: str | None = None


class ModelRosterRow(BaseModel):
    model_name: str
    family: str
    feature_variant: str | None = None
    role: str = "challenger"  # champion | challenger | ensemble
    trained_at: str | None = None
    train_start_date: str | None = None
    train_end_date: str | None = None
    rows_train: int | None = None
    rows_valid: int | None = None
    settled_bets: int = 0
    accuracy: float | None = None
    log_loss: float | None = None
    roi: float | None = None
    units_won: float | None = None


class SizingPolicyRow(BaseModel):
    policy: str  # flat_1u | edge_scaled_cap_1u | fractional_kelly_25_cap_1u
    label: str
    bets: int
    total_stake: float
    units_won: float
    roi: float
    is_best: bool = False


class SizingComparisonRow(BaseModel):
    model_name: str
    family: str
    role: str = "challenger"
    policies: list[SizingPolicyRow]
    best_policy: str | None = None
    best_roi: float | None = None


class JobSummary(BaseModel):
    id: str
    command: str
    label: str
    pid: int
    status: str
    started_at: str
    finished_at: str | None = None
    returncode: int | None = None
    log_path: str
    duration_seconds: float


class JobDetail(JobSummary):
    log_tail: str = ""


class FreshnessResponse(BaseModel):
    latest_snapshot_ts: str | None = None
    latest_game_date: str | None = None
    latest_results_sync: str | None = None
    latest_weather_game_date: str | None = None
    latest_historical_import_completed_at: str | None = None
    stale_data: bool = False


class ImportStatusRow(BaseModel):
    source: str
    start_date: str | None = None
    end_date: str | None = None
    import_runs: int = 0
    request_count: int | None = None
    payload_count: int | None = None
    normalized_rows: int | None = None
    games_total: int | None = None
    games_with_markets: int | None = None
    games_with_pregame_quotes: int | None = None
    candidate_markets: int | None = None
    empty_payload_count: int | None = None
    rate_limited_count: int | None = None
    parse_error_count: int | None = None
    last_completed_at: str | None = None


class JobActionResponse(BaseModel):
    accepted: bool = True
    job: JobSummary


class HealthResponse(BaseModel):
    status: str = "ok"
    now: datetime = Field(default_factory=datetime.utcnow)
    api: str = "mlpm-local-api"


JSONValue = str | int | float | bool | None | dict[str, Any] | list[Any]
