from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def write_raw_payload(base_dir: Path, source: str, payload: Any, captured_at: datetime | None = None) -> Path:
    captured_at = captured_at or utcnow()
    target_dir = base_dir / source
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{captured_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
