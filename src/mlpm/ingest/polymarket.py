from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

from mlpm.ingest.http import get_json_with_retries

POLYMARKET_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
POLYMARKET_SPORTS_URL = "https://gamma-api.polymarket.com/sports"
MLB_SPORT = "mlb"

def _fetch_mlb_tag_id(client: httpx.Client) -> int:
    sports = get_json_with_retries(client, POLYMARKET_SPORTS_URL)
    for sport in sports:
        if str(sport.get("sport", "")).lower() == MLB_SPORT:
            tags = [tag.strip() for tag in str(sport.get("tags", "")).split(",") if tag.strip()]
            if not tags:
                raise ValueError("Polymarket MLB sport metadata has no tags.")
            return int(tags[-1])
    raise ValueError("Could not find Polymarket MLB sport metadata.")


def fetch_mlb_markets(limit: int = 500) -> tuple[pd.DataFrame, Any]:
    with httpx.Client(timeout=20.0) as client:
        tag_id = _fetch_mlb_tag_id(client)
        params = {"tag_id": tag_id, "limit": limit, "closed": "false", "active": "true"}
        payload = get_json_with_retries(client, POLYMARKET_MARKETS_URL, params=params)

    rows: list[dict[str, object]] = []
    for market in payload:
        if market.get("sportsMarketType") != "moneyline":
            continue

        outcomes = market.get("outcomes")
        outcome_prices = market.get("outcomePrices")
        if isinstance(outcomes, str) and isinstance(outcome_prices, str):
            # The API often returns JSON-encoded strings for these fields.
            import json

            outcomes = json.loads(outcomes)
            outcome_prices = json.loads(outcome_prices)

        if not isinstance(outcomes, list) or not isinstance(outcome_prices, list):
            continue

        for outcome, price in zip(outcomes, outcome_prices):
            rows.append(
                {
                    "source": "polymarket",
                    "event_id": str(market.get("conditionId") or market.get("id")),
                    "market_id": str(market.get("id")),
                    "question": market.get("question"),
                    "market_slug": market.get("slug"),
                    "event_start_time": market.get("gameStartTime")
                    or market.get("startTime")
                    or market.get("startDate")
                    or market.get("endDate"),
                    "outcome_team": outcome,
                    "last_price": price,
                    "volume": market.get("volume"),
                    "snapshot_ts": market.get("updatedAt") or market.get("endDate"),
                }
            )
    return pd.DataFrame(rows), payload
