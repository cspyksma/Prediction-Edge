from __future__ import annotations

from pathlib import Path

import duckdb

from mlpm.storage import duckdb as duckdb_storage


def test_open_connection_retries_transient_lock(monkeypatch) -> None:
    attempts = {"count": 0}
    db_path = Path(".tmp") / "retry-lock.duckdb"

    class _FakeConnection:
        pass

    def fake_connect(path: str, read_only: bool = False):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise duckdb.IOException("IO Error: Cannot open file: being used by another process")
        return _FakeConnection()

    monkeypatch.setattr(duckdb_storage.duckdb, "connect", fake_connect)
    monkeypatch.setattr(duckdb_storage.time, "sleep", lambda seconds: None)

    conn = duckdb_storage._open_connection(db_path, read_only=False)
    assert isinstance(conn, _FakeConnection)
    assert attempts["count"] == 3


def test_open_connection_does_not_retry_non_lock_error(monkeypatch) -> None:
    db_path = Path(".tmp") / "retry-fail.duckdb"

    def fake_connect(path: str, read_only: bool = False):
        raise duckdb.IOException("IO Error: malformed database header")

    monkeypatch.setattr(duckdb_storage.duckdb, "connect", fake_connect)
    monkeypatch.setattr(duckdb_storage.time, "sleep", lambda seconds: None)

    try:
        duckdb_storage._open_connection(db_path, read_only=True)
    except duckdb.IOException as exc:
        assert "malformed database header" in str(exc)
    else:
        raise AssertionError("Expected duckdb.IOException")
