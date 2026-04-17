from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

from mlpm.ingest.http import get_json_with_retries

KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
KALSHI_MLB_GAME_SERIES = "KXMLBGAME"


def fetch_mlb_markets() -> tuple[pd.DataFrame, Any]:
    with httpx.Client(timeout=20.0) as client:
        markets_payload = get_json_with_retries(
            client,
            KALSHI_MARKETS_URL,
            params={"limit": 1000, "status": "open", "series_ticker": KALSHI_MLB_GAME_SERIES},
        )

    rows: list[dict[str, object]] = []
    for market in markets_payload.get("markets", []):
        rows.append(
            {
                "source": "kalshi",
                "event_id": market.get("event_ticker"),
                "event_title": market.get("title"),
                "event_sub_title": market.get("yes_sub_title"),
                "market_ticker": market.get("ticker"),
                "market_title": market.get("title"),
                "yes_sub_title": market.get("yes_sub_title"),
                "no_sub_title": market.get("no_sub_title"),
                "event_start_time": market.get("close_time"),
                "yes_bid": _parse_dollar_price(market.get("yes_bid_dollars")),
                "yes_ask": _parse_dollar_price(market.get("yes_ask_dollars")),
                "last_price": _parse_dollar_price(market.get("last_price_dollars")),
                "volume": market.get("volume_fp"),
                "snapshot_ts": market.get("updated_time"),
            }
        )
    return pd.DataFrame(rows), {"markets": markets_payload}


def _parse_dollar_price(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value) * 100.0
