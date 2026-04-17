from __future__ import annotations

import random
import time
from typing import Any

import httpx

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def get_json_with_retries(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, object] | None = None,
    max_attempts: int = 4,
    base_delay_seconds: float = 0.5,
) -> Any:
    last_error: Exception | None = None
    last_status_code: int | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.get(url, params=params)
            if response.status_code in RETRYABLE_STATUS_CODES:
                last_status_code = response.status_code
                if attempt == max_attempts:
                    response.raise_for_status()
                _sleep_before_retry(attempt, base_delay_seconds)
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            _sleep_before_retry(attempt, base_delay_seconds)

    if last_error is not None:
        raise RuntimeError(f"HTTP GET failed for {url}: {last_error}") from last_error
    raise RuntimeError(f"HTTP GET failed for {url} with retryable status {last_status_code}")


def _sleep_before_retry(attempt: int, base_delay_seconds: float) -> None:
    jitter = random.uniform(0.0, base_delay_seconds)
    delay = (base_delay_seconds * (2 ** (attempt - 1))) + jitter
    time.sleep(delay)
