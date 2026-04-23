from __future__ import annotations

from datetime import date

import httpx
import pandas as pd
import pytest

from mlpm.ingest.weather import _fetch_archive


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://archive-api.open-meteo.com/v1/archive")
            response = httpx.Response(self.status_code, request=request, headers=self.headers)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> object:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def get(
        self,
        url: str,
        params: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_fetch_archive_retries_429_then_succeeds(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("mlpm.ingest.weather.time.sleep", lambda seconds: sleeps.append(seconds))
    client = _FakeClient(
        [
            _FakeResponse(429, headers={"Retry-After": "1"}),
            _FakeResponse(429),
            _FakeResponse(
                200,
                {
                    "timezone": "America/Chicago",
                    "hourly": {
                        "time": ["2026-04-01T19:00"],
                        "temperature_2m": [70.0],
                        "windspeed_10m": [10.0],
                        "winddirection_10m": [180.0],
                        "relativehumidity_2m": [45.0],
                        "precipitation": [0.0],
                    },
                },
            ),
        ]
    )

    frame = _fetch_archive(
        client,
        lat=41.0,
        lon=-87.0,
        start=date(2026, 4, 1),
        end=date(2026, 4, 1),
    )

    assert client.calls == 3
    assert sleeps == [1.0, 4.0]
    assert not frame.empty
    assert frame["timezone"].iloc[0] == "America/Chicago"


def test_fetch_archive_raises_after_retry_budget(monkeypatch) -> None:
    monkeypatch.setattr("mlpm.ingest.weather.time.sleep", lambda *_args, **_kwargs: None)
    client = _FakeClient([_FakeResponse(429)] * 6)

    with pytest.raises(httpx.HTTPStatusError):
        _fetch_archive(
            client,
            lat=41.0,
            lon=-87.0,
            start=date(2026, 4, 1),
            end=date(2026, 4, 1),
        )

