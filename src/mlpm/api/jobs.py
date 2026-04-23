from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mlpm.config.settings import settings


PROJECT_ROOT = Path(__file__).resolve().parents[3]
JOBS_DIR = Path(settings().duckdb_path).resolve().parent / "dashboard_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

JOB_COMMANDS: dict[str, dict[str, object]] = {
    "collect-once": {
        "label": "Collect snapshot",
        "help": "Pull fresh MLB games, Kalshi & Polymarket quotes, and recompute discrepancies.",
        "args": ["collect-once"],
    },
    "sync-results": {
        "label": "Sync game results",
        "help": "Backfill recent final MLB game results into DuckDB.",
        "args": ["sync-results"],
    },
    "train-game-model": {
        "label": "Train model",
        "help": "Retrain and persist the MLB game-outcome model.",
        "args": ["train-game-model"],
    },
}


@dataclass
class JobRecord:
    id: str
    command: str
    label: str
    pid: int
    log_path: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "running"
    returncode: int | None = None
    proc: subprocess.Popen[str] | None = None
    log_file: object | None = None


_JOB_REGISTRY: list[JobRecord] = []


def _metadata_path(log_path: Path) -> Path:
    return log_path.with_suffix(".json")


def _serialize_job(job: JobRecord) -> dict[str, object]:
    return {
        "id": job.id,
        "command": job.command,
        "label": job.label,
        "pid": job.pid,
        "log_path": job.log_path,
        "started_at": job.started_at.isoformat(),
        "finished_at": None if job.finished_at is None else job.finished_at.isoformat(),
        "status": job.status,
        "returncode": job.returncode,
    }


def _write_job_metadata(job: JobRecord) -> None:
    metadata_path = _metadata_path(Path(job.log_path))
    metadata_path.write_text(json.dumps(_serialize_job(job), indent=2), encoding="utf-8")


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _load_job_from_metadata(metadata_path: Path) -> JobRecord | None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    log_path = str(metadata_path.with_suffix(".log"))
    started_at = datetime.fromisoformat(payload["started_at"])
    finished_raw = payload.get("finished_at")
    finished_at = datetime.fromisoformat(finished_raw) if finished_raw else None
    status = str(payload.get("status") or "running")
    pid = int(payload.get("pid") or 0)
    if status == "running" and not _is_process_alive(pid):
        status = "unknown"
    return JobRecord(
        id=str(payload["id"]),
        command=str(payload.get("command") or "unknown"),
        label=str(payload.get("label") or payload.get("command") or "Unknown job"),
        pid=pid,
        log_path=log_path,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        returncode=payload.get("returncode"),
    )


def _parse_log_stem(stem: str) -> tuple[str, str]:
    for command in sorted(JOB_COMMANDS, key=len, reverse=True):
        prefix = f"{command}-"
        if stem.startswith(prefix):
            return command, stem[len(prefix):]
    command, _, job_id = stem.rpartition("-")
    return command or stem, job_id or stem


def _load_job_from_log(log_path: Path) -> JobRecord:
    stem = log_path.stem
    command, job_id = _parse_log_stem(stem)
    started_at = datetime.fromtimestamp(log_path.stat().st_mtime)
    label = command or "Unknown job"
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if " label=" in line:
                    label = line.strip().split(" label=", 1)[1]
                    break
    except OSError:
        pass
    return JobRecord(
        id=job_id or stem,
        command=command or stem,
        label=label,
        pid=0,
        log_path=str(log_path),
        started_at=started_at,
        finished_at=started_at,
        status="unknown",
        returncode=None,
    )


def _persisted_jobs() -> list[JobRecord]:
    jobs: dict[str, JobRecord] = {}
    for metadata_path in JOBS_DIR.glob("*.json"):
        job = _load_job_from_metadata(metadata_path)
        if job is not None:
            jobs[job.id] = job
    for log_path in JOBS_DIR.glob("*.log"):
        _, job_id = _parse_log_stem(log_path.stem)
        if job_id and job_id in jobs:
            continue
        fallback = _load_job_from_log(log_path)
        jobs[fallback.id] = fallback
    return sorted(jobs.values(), key=lambda job: job.started_at, reverse=True)


def poll_jobs() -> list[JobRecord]:
    for job in _JOB_REGISTRY:
        if job.status != "running" or job.proc is None:
            continue
        rc = job.proc.poll()
        if rc is None:
            continue
        job.returncode = rc
        job.finished_at = datetime.now()
        job.status = "success" if rc == 0 else "failed"
        if job.log_file is not None:
            try:
                job.log_file.close()
            except Exception:
                pass
            finally:
                job.log_file = None
        _write_job_metadata(job)
    return _JOB_REGISTRY


def start_job(command_key: str) -> JobRecord:
    if command_key not in JOB_COMMANDS:
        raise KeyError(f"Unknown job command: {command_key}")
    spec = JOB_COMMANDS[command_key]
    job_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    log_path = JOBS_DIR / f"{command_key}-{job_id}.log"
    log_file = open(log_path, "w", buffering=1)
    log_file.write(f"[{datetime.now().isoformat()}] starting job={command_key} label={spec['label']}\n")
    log_file.write(f"[{datetime.now().isoformat()}] command={sys.executable} -u -m mlpm.cli {' '.join(spec['args'])}\n")
    log_file.write(f"[{datetime.now().isoformat()}] cwd={PROJECT_ROOT}\n")
    log_file.flush()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "mlpm.cli", *spec["args"]],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        text=True,
    )
    job = JobRecord(
        id=job_id,
        command=command_key,
        label=str(spec["label"]),
        pid=proc.pid,
        log_path=str(log_path),
        started_at=datetime.now(),
        proc=proc,
        log_file=log_file,
    )
    _write_job_metadata(job)
    _JOB_REGISTRY.insert(0, job)
    return job


def list_jobs() -> list[JobRecord]:
    running = {job.id: job for job in poll_jobs()}
    merged = {job.id: job for job in _persisted_jobs()}
    merged.update(running)
    return sorted(merged.values(), key=lambda job: job.started_at, reverse=True)


def get_job(job_id: str) -> JobRecord | None:
    for job in list_jobs():
        if job.id == job_id:
            return job
    return None


def read_log_tail(path: str, n_lines: int = 80) -> str:
    try:
        with open(path, "r", errors="replace") as handle:
            return "".join(handle.readlines()[-n_lines:])
    except FileNotFoundError:
        return ""
