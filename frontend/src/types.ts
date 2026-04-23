export interface Summary {
  total_discrepancies: number;
  flagged_discrepancies: number;
  avg_abs_gap_bps: number;
  champion_model: string | null;
  latest_snapshot_ts: string | null;
  latest_game_date: string | null;
  actionable_bets: number;
  max_edge_bps: number;
  stale_data: boolean;
  freshness_window_seconds: number;
}

export interface Opportunity {
  game_id: string;
  game_date: string | null;
  event_start_time: string | null;
  model_name: string;
  source: string | null;
  market_id: string | null;
  team: string;
  opponent_team: string | null;
  model_prob: number;
  market_prob: number;
  edge_bps: number;
  expected_value: number | null;
  is_actionable: boolean;
  is_champion: boolean;
}

export interface OpportunityResponse {
  items: Opportunity[];
  total: number;
  page: number;
  page_size: number;
  champion_model: string | null;
}

export interface GameQuoteRow {
  source: string | null;
  market_id: string | null;
  team: string;
  market_prob: number;
  model_prob: number;
  gap_bps: number;
  snapshot_ts: string | null;
  flagged: boolean;
}

export interface GapHistoryRow {
  snapshot_ts: string | null;
  team: string;
  source: string | null;
  gap_bps: number;
}

export interface GameFeatureContext {
  team: string;
  opponent_team: string | null;
  model_name: string | null;
  model_prob: number | null;
  season_win_pct: number | null;
  recent_win_pct: number | null;
  venue_win_pct: number | null;
  run_diff_per_game: number | null;
  streak: number | null;
  elo_rating: number | null;
  rest_days: number | null;
  starter_era: number | null;
  starter_whip: number | null;
  bullpen_innings_3d: number | null;
}

export interface GameDetail {
  game_id: string;
  game_date: string | null;
  event_start_time: string | null;
  away_team: string | null;
  home_team: string | null;
  quotes: GameQuoteRow[];
  features: GameFeatureContext[];
  gap_history: GapHistoryRow[];
}

export interface Strategy {
  strategy_name: string;
  family: string | null;
  bets: number;
  roi: number;
  roi_ci_lower: number | null;
  roi_ci_upper: number | null;
  units_won: number;
  max_drawdown: number | null;
  positive_slice_rate: number | null;
  guardrails_passed: boolean;
}

export interface Contender {
  model: string;
  family: string | null;
  model_family: string | null;
  feature_variant: string | null;
  accuracy: number | null;
  roc_auc: number | null;
  log_loss: number | null;
}

export interface CalibrationRow {
  model_name: string;
  bucket: string;
  avg_predicted_prob: number;
  actual_home_win_rate: number;
  abs_error: number;
}

export interface FeatureImportance {
  model_name: string;
  feature: string;
  importance: number;
  importance_std: number | null;
  rank: number;
  trained_at: string | null;
}

export interface JobSummary {
  id: string;
  command: string;
  label: string;
  pid: number;
  status: string;
  started_at: string;
  finished_at: string | null;
  returncode: number | null;
  log_path: string;
  duration_seconds: number;
}

export interface JobDetail extends JobSummary {
  log_tail: string;
}

export interface Freshness {
  latest_snapshot_ts: string | null;
  latest_game_date: string | null;
  latest_results_sync: string | null;
  latest_weather_game_date: string | null;
  latest_historical_import_completed_at: string | null;
  stale_data: boolean;
}

export interface ImportStatusRow {
  source: string;
  start_date: string | null;
  end_date: string | null;
  import_runs: number;
  normalized_rows: number | null;
  games_with_markets: number | null;
  games_with_pregame_quotes: number | null;
  last_completed_at: string | null;
}

export interface TrainingCoverageRow {
  source: string;
  label: string;
  first_date: string | null;
  last_date: string | null;
  games_with_prior: number;
}

export interface TrainingCoverage {
  rows: TrainingCoverageRow[];
  train_start_date: string | null;
  train_end_date: string | null;
  total_games_with_prior: number;
  latest_model_train_start: string | null;
  latest_model_train_end: string | null;
  latest_model_trained_at: string | null;
}

export interface ModelRosterRow {
  model_name: string;
  family: string;
  feature_variant: string | null;
  role: string;
  trained_at: string | null;
  train_start_date: string | null;
  train_end_date: string | null;
  rows_train: number | null;
  rows_valid: number | null;
  settled_bets: number;
  accuracy: number | null;
  log_loss: number | null;
  roi: number | null;
  units_won: number | null;
}

export interface SizingPolicyRow {
  policy: string;
  label: string;
  bets: number;
  total_stake: number;
  units_won: number;
  roi: number;
  is_best: boolean;
}

export interface SizingComparisonRow {
  model_name: string;
  family: string;
  role: string;
  policies: SizingPolicyRow[];
  best_policy: string | null;
  best_roi: number | null;
}
