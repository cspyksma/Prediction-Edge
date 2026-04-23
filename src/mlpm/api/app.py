from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from mlpm.api.jobs import JOB_COMMANDS, get_job, list_jobs, read_log_tail, start_job
from mlpm.api.schemas import (
    CalibrationRow,
    FeatureImportanceRow,
    FreshnessResponse,
    GameDetailResponse,
    GapHistoryRow,
    HealthResponse,
    ImportStatusRow,
    JobActionResponse,
    JobDetail,
    JobSummary,
    ModelRosterRow,
    OpportunityListResponse,
    OpportunityRow,
    ResearchContenderRow,
    ResearchStrategyRow,
    SizingComparisonRow,
    SummaryResponse,
    TrainingCoverageResponse,
)
from mlpm.api.services import (
    get_calibration,
    get_feature_importance,
    get_freshness,
    get_game_detail,
    get_gap_history,
    get_import_status,
    get_model_roster,
    get_research_contenders,
    get_research_strategies,
    get_sizing_comparison,
    get_summary,
    get_training_coverage,
    list_opportunities,
)
from mlpm.config.settings import settings
from mlpm.storage.duckdb import connect


def _serialize_job(job, *, include_log: bool = False) -> dict:
    now = job.finished_at or job.started_at
    payload = {
        "id": job.id,
        "command": job.command,
        "label": job.label,
        "pid": job.pid,
        "status": job.status,
        "started_at": job.started_at.isoformat(),
        "finished_at": None if job.finished_at is None else job.finished_at.isoformat(),
        "returncode": job.returncode,
        "log_path": job.log_path,
        "duration_seconds": max((now - job.started_at).total_seconds(), 0.0),
    }
    if include_log:
        payload["log_tail"] = read_log_tail(job.log_path)
    return payload


def create_app() -> FastAPI:
    cfg = settings()
    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    connect(cfg.duckdb_path).close()
    app = FastAPI(title="MLPM Local API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[cfg.frontend_dev_origin, "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/api/v1/summary", response_model=SummaryResponse)
    def summary() -> SummaryResponse:
        return SummaryResponse(**get_summary())

    @app.get("/api/v1/cockpit/opportunities", response_model=OpportunityListResponse)
    def opportunities(
        source: str | None = None,
        model_name: str | None = None,
        actionable_only: bool = False,
        min_edge_bps: int | None = None,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
    ) -> OpportunityListResponse:
        return OpportunityListResponse(**list_opportunities(
            source=source,
            model_name=model_name,
            actionable_only=actionable_only,
            min_edge_bps=min_edge_bps,
            page=page,
            page_size=page_size,
        ))

    @app.get("/api/v1/cockpit/games/{game_id}", response_model=GameDetailResponse)
    def game_detail(game_id: str) -> GameDetailResponse:
        payload = get_game_detail(game_id)
        if not payload["quotes"] and payload["away_team"] is None:
            raise HTTPException(status_code=404, detail="Game not found")
        return GameDetailResponse(**payload)

    @app.get("/api/v1/cockpit/gap-history", response_model=list[GapHistoryRow])
    def gap_history(
        game_id: str | None = None,
        team: str | None = None,
        source: str | None = None,
        limit: int = Query(default=300, ge=1, le=1000),
    ) -> list[GapHistoryRow]:
        return [GapHistoryRow(**row) for row in get_gap_history(game_id=game_id, team=team, source=source, limit=limit)]

    @app.get("/api/v1/research/contenders", response_model=list[ResearchContenderRow])
    def research_contenders() -> list[ResearchContenderRow]:
        return [ResearchContenderRow(**row) for row in get_research_contenders()]

    @app.get("/api/v1/research/strategies", response_model=list[ResearchStrategyRow])
    def research_strategies() -> list[ResearchStrategyRow]:
        return [ResearchStrategyRow(**row) for row in get_research_strategies()]

    @app.get("/api/v1/research/calibration", response_model=list[CalibrationRow])
    def research_calibration(bins: int = Query(default=10, ge=2, le=50)) -> list[CalibrationRow]:
        return [CalibrationRow(**row) for row in get_calibration(bins=bins)]

    @app.get("/api/v1/research/feature-importance", response_model=list[FeatureImportanceRow])
    def research_feature_importance(limit: int = Query(default=50, ge=1, le=500)) -> list[FeatureImportanceRow]:
        return [FeatureImportanceRow(**row) for row in get_feature_importance(limit=limit)]

    @app.get("/api/v1/research/training-coverage", response_model=TrainingCoverageResponse)
    def research_training_coverage() -> TrainingCoverageResponse:
        return TrainingCoverageResponse(**get_training_coverage())

    @app.get("/api/v1/research/model-roster", response_model=list[ModelRosterRow])
    def research_model_roster() -> list[ModelRosterRow]:
        return [ModelRosterRow(**row) for row in get_model_roster()]

    @app.get("/api/v1/research/sizing-comparison", response_model=list[SizingComparisonRow])
    def research_sizing_comparison() -> list[SizingComparisonRow]:
        return [SizingComparisonRow(**row) for row in get_sizing_comparison()]

    @app.get("/api/v1/ops/jobs", response_model=list[JobSummary])
    def ops_jobs() -> list[JobSummary]:
        return [JobSummary(**_serialize_job(job)) for job in list_jobs()]

    @app.get("/api/v1/ops/jobs/{job_id}", response_model=JobDetail)
    def ops_job_detail(job_id: str) -> JobDetail:
        job = get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobDetail(**_serialize_job(job, include_log=True))

    @app.get("/api/v1/ops/freshness", response_model=FreshnessResponse)
    def ops_freshness() -> FreshnessResponse:
        return FreshnessResponse(**get_freshness())

    @app.get("/api/v1/ops/import-status", response_model=list[ImportStatusRow])
    def ops_import_status() -> list[ImportStatusRow]:
        return [ImportStatusRow(**row) for row in get_import_status()]

    @app.post("/api/v1/jobs/{command_key}", response_model=JobActionResponse)
    def launch_job(command_key: str) -> JobActionResponse:
        if command_key not in JOB_COMMANDS:
            raise HTTPException(status_code=404, detail="Unknown job command")
        job = start_job(command_key)
        return JobActionResponse(job=JobSummary(**_serialize_job(job)))

    return app


app = create_app()
