from __future__ import annotations

from mlpm.ingest.mlb_stats import _fetch_boxscore_payloads


def test_fetch_boxscore_payloads_batches_unique_game_ids(monkeypatch) -> None:
    calls: list[str] = []

    def fake_fetch(game_id: str) -> dict:
        calls.append(game_id)
        return {"gamePk": game_id}

    monkeypatch.setattr("mlpm.ingest.mlb_stats._fetch_boxscore_payload", fake_fetch)

    payloads = _fetch_boxscore_payloads(["1", "2", "2", "3"])

    assert set(calls) == {"1", "2", "3"}
    assert payloads == {
        "1": {"gamePk": "1"},
        "2": {"gamePk": "2"},
        "3": {"gamePk": "3"},
    }
