/// <reference types="vite/client" />

import type {
  CalibrationRow,
  ChampionStandings,
  Contender,
  FeatureImportance,
  Freshness,
  GameDetail,
  ImportStatusRow,
  JobDetail,
  JobSummary,
  ModelRosterRow,
  OpportunityResponse,
  SizingComparisonRow,
  Strategy,
  Summary,
  TrainingCoverage,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "/api/v1").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`API ${response.status}: ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  summary: () => request<Summary>("/summary"),
  opportunities: () => request<OpportunityResponse>("/cockpit/opportunities?actionable_only=true&page_size=100"),
  gameDetail: (gameId: string) => request<GameDetail>(`/cockpit/games/${gameId}`),
  contenders: () => request<Contender[]>("/research/contenders"),
  strategies: () => request<Strategy[]>("/research/strategies"),
  championStandings: () => request<ChampionStandings>("/research/champion-standings"),
  calibration: () => request<CalibrationRow[]>("/research/calibration"),
  featureImportance: () => request<FeatureImportance[]>("/research/feature-importance?limit=100"),
  trainingCoverage: () => request<TrainingCoverage>("/research/training-coverage"),
  modelRoster: () => request<ModelRosterRow[]>("/research/model-roster"),
  sizingComparison: () => request<SizingComparisonRow[]>("/research/sizing-comparison"),
  jobs: () => request<JobSummary[]>("/ops/jobs"),
  job: (jobId: string) => request<JobDetail>(`/ops/jobs/${jobId}`),
  freshness: () => request<Freshness>("/ops/freshness"),
  importStatus: () => request<ImportStatusRow[]>("/ops/import-status"),
  launchJob: (command: string) => request<{ accepted: boolean; job: JobSummary }>(`/jobs/${command}`, { method: "POST" }),
};
