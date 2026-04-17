from __future__ import annotations

import httpx
import pytest

from mlpm.ingest.http import get_json_with_retries


class _FakeResponse:
    def __init__(self, status_code: int, payload: object = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> object:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)

    def get(self, url: str, params: dict[str, object] | None = None) -> _FakeResponse:
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_get_json_with_retries_retries_transient_status(monkeypatch) -> None:
    monkeypatch.setattr("mlpm.ingest.http.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlpm.ingest.http.random.uniform", lambda *_args, **_kwargs: 0.0)
    client = _FakeClient([_FakeResponse(503), _FakeResponse(200, {"ok": True})])

    payload = get_json_with_retries(client, "https://example.test")

    assert payload == {"ok": True}


def test_get_json_with_retries_raises_after_retry_budget(monkeypatch) -> None:
    monkeypatch.setattr("mlpm.ingest.http.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlpm.ingest.http.random.uniform", lambda *_args, **_kwargs: 0.0)
    client = _FakeClient([_FakeResponse(429), _FakeResponse(429)])

    with pytest.raises(httpx.HTTPStatusError):
        get_json_with_retries(client, "https://example.test", max_attempts=2)


def test_get_json_with_retries_retries_network_error(monkeypatch) -> None:
    monkeypatch.setattr("mlpm.ingest.http.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("mlpm.ingest.http.random.uniform", lambda *_args, **_kwargs: 0.0)
    client = _FakeClient([httpx.ConnectError("boom"), _FakeResponse(200, {"ok": True})])

    payload = get_json_with_retries(client, "https://example.test")

    assert payload == {"ok": True}
