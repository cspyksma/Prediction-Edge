from __future__ import annotations

import httpx

from mlpm.config.settings import settings
from mlpm.ingest.kalshi_rate_limits import (
    KalshiRateLimiter,
    fetch_kalshi_account_limits,
    resolve_kalshi_rate_limits,
)


class _FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requested_urls: list[str] = []

    def get(self, url: str, params: dict[str, object] | None = None) -> httpx.Response:
        self.requested_urls.append(url)
        request = httpx.Request("GET", url, params=params)
        return httpx.Response(200, request=request, json=self.payload)


def test_resolve_kalshi_rate_limits_uses_documented_tier_defaults(monkeypatch) -> None:
    monkeypatch.delenv("KALSHI_RATE_LIMIT_TIER", raising=False)
    monkeypatch.delenv("KALSHI_READ_LIMIT_PER_SECOND", raising=False)
    monkeypatch.delenv("KALSHI_WRITE_LIMIT_PER_SECOND", raising=False)
    settings.cache_clear()

    limits = resolve_kalshi_rate_limits(usage_tier="premier")

    assert limits.usage_tier == "premier"
    assert limits.read_limit == 100
    assert limits.write_limit == 100


def test_resolve_kalshi_rate_limits_honors_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("KALSHI_RATE_LIMIT_TIER", "basic")
    monkeypatch.setenv("KALSHI_READ_LIMIT_PER_SECOND", "17")
    monkeypatch.setenv("KALSHI_WRITE_LIMIT_PER_SECOND", "9")
    settings.cache_clear()

    limits = resolve_kalshi_rate_limits()

    assert limits.usage_tier == "basic"
    assert limits.read_limit == 17
    assert limits.write_limit == 9


def test_fetch_kalshi_account_limits_reads_live_payload() -> None:
    client = _FakeClient({"usage_tier": "prime", "read_limit": 222, "write_limit": 111})

    limits = fetch_kalshi_account_limits(client)

    assert limits.usage_tier == "prime"
    assert limits.read_limit == 222
    assert limits.write_limit == 111


def test_kalshi_rate_limiter_spaces_requests_after_limit() -> None:
    times = iter([0.0, 0.1, 0.2, 1.05])
    slept: list[float] = []
    limiter = KalshiRateLimiter(
        read_limit=2,
        time_fn=lambda: next(times),
        sleep_fn=lambda seconds: slept.append(seconds),
    )

    limiter.wait_for_read_slot()
    limiter.wait_for_read_slot()
    limiter.wait_for_read_slot()

    assert slept == [0.8]
