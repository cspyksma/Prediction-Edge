from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from mlpm.config.settings import settings

KALSHI_ACCOUNT_LIMITS_URL = "https://api.elections.kalshi.com/trade-api/v2/account/limits"
KALSHI_ORDER_WRITE_ENDPOINTS = frozenset(
    {
        "BatchCreateOrders",
        "BatchCancelOrders",
        "CreateOrder",
        "CancelOrder",
        "AmendOrder",
        "DecreaseOrder",
    }
)
KALSHI_DOCUMENTED_TIERS: dict[str, tuple[int, int]] = {
    "basic": (20, 10),
    "advanced": (30, 30),
    "premier": (100, 100),
    "prime": (400, 400),
}


@dataclass(frozen=True, slots=True)
class KalshiRateLimits:
    usage_tier: str
    read_limit: int
    write_limit: int


def resolve_kalshi_rate_limits(
    *,
    usage_tier: str | None = None,
    read_limit: int | None = None,
    write_limit: int | None = None,
) -> KalshiRateLimits:
    cfg = settings()
    tier_name = (usage_tier or cfg.kalshi_rate_limit_tier or "basic").strip().lower()
    default_read_limit, default_write_limit = KALSHI_DOCUMENTED_TIERS.get(
        tier_name,
        KALSHI_DOCUMENTED_TIERS["basic"],
    )
    return KalshiRateLimits(
        usage_tier=tier_name,
        read_limit=max(1, int(read_limit or cfg.kalshi_read_limit_per_second or default_read_limit)),
        write_limit=max(1, int(write_limit or cfg.kalshi_write_limit_per_second or default_write_limit)),
    )


def fetch_kalshi_account_limits(client: httpx.Client) -> KalshiRateLimits:
    response = client.get(KALSHI_ACCOUNT_LIMITS_URL)
    response.raise_for_status()
    payload = response.json()
    return resolve_kalshi_rate_limits(
        usage_tier=str(payload.get("usage_tier") or "basic"),
        read_limit=_coerce_positive_int(payload.get("read_limit")),
        write_limit=_coerce_positive_int(payload.get("write_limit")),
    )


class KalshiRateLimiter:
    """Simple sliding-window limiter for per-second Kalshi request budgets."""

    def __init__(
        self,
        *,
        read_limit: int,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._read_limit = max(1, int(read_limit))
        self._read_events: deque[float] = deque()
        self._time_fn = time_fn or time.monotonic
        self._sleep_fn = sleep_fn or time.sleep

    def wait_for_read_slot(self) -> None:
        now = self._time_fn()
        self._trim(now)
        if len(self._read_events) >= self._read_limit:
            sleep_seconds = max(0.0, 1.0 - (now - self._read_events[0]))
            if sleep_seconds > 0:
                self._sleep_fn(sleep_seconds)
            now = self._time_fn()
            self._trim(now)
        self._read_events.append(now)

    def get(self, client: httpx.Client, url: str, *, params: dict | None = None) -> httpx.Response:
        self.wait_for_read_slot()
        return client.get(url, params=params)

    def _trim(self, now: float) -> None:
        cutoff = now - 1.0
        while self._read_events and self._read_events[0] <= cutoff:
            self._read_events.popleft()


def _coerce_positive_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
