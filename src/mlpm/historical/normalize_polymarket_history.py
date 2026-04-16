from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mlpm.normalize.mapping import canonicalize_team_name


def normalize_polymarket_history_payload(
    raw_payload: dict,
    market_id: str,
    game_id: str | None = None,
) -> list[dict]:
    mapping_row = {"market_id": market_id, "game_id": game_id}
    return polymarket_history_to_quote_rows(_extract_history_rows(raw_payload), mapping_row)


def polymarket_history_to_quote_rows(history_rows: list[dict], mapping_row: dict) -> list[dict]:
    rows: list[dict] = []
    imported_at = datetime.now(tz=UTC)
    outcome_team = mapping_row.get("outcome_team")
    home_team = mapping_row.get("home_team")
    away_team = mapping_row.get("away_team")
    event_start_time = mapping_row.get("event_start_time")
    event_start_dt = _parse_ts(event_start_time)

    for item in history_rows:
        quote_dt = _history_timestamp(item)
        if quote_dt is None:
            continue
        probability = _history_price(item)
        if probability is None:
            continue
        side = _infer_side(outcome_team, home_team, away_team)
        home_implied_prob = probability if side == "home" else (1.0 - probability if side == "away" else probability)
        rows.append(
            {
                "import_run_id": mapping_row.get("import_run_id"),
                "source": "polymarket",
                "collection_mode": "historical_import",
                "market_id": mapping_row.get("market_id"),
                "event_id": mapping_row.get("event_id"),
                "asset_id": mapping_row.get("asset_id"),
                "game_id": mapping_row.get("game_id"),
                "event_start_time": event_start_time,
                "quote_ts": quote_dt,
                "outcome_team": outcome_team,
                "side": side,
                "home_implied_prob": home_implied_prob,
                "raw_prob_yes": probability,
                "best_price_source": "prices-history",
                "pre_pitch_flag": (quote_dt < event_start_dt) if event_start_dt is not None else None,
                "raw_payload_path": mapping_row.get("raw_payload_path"),
                "imported_at": imported_at,
            }
        )
    return rows


def _extract_history_rows(raw_payload: dict[str, Any]) -> list[dict]:
    for key in ("history", "data", "pricesHistory", "items"):
        value = raw_payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if isinstance(raw_payload.get("history"), dict):
        nested = raw_payload["history"].get("history")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    return []


def _history_timestamp(item: dict[str, Any]) -> datetime | None:
    for key in ("t", "timestamp", "ts", "time"):
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        parsed = _parse_ts(value)
        if parsed is not None:
            return parsed
    return None


def _history_price(item: dict[str, Any]) -> float | None:
    for key in ("p", "price", "close", "value"):
        value = item.get(key)
        if value in (None, ""):
            continue
        probability = float(value)
        return probability if probability <= 1.0 else probability / 100.0
    return None


def _parse_ts(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=UTC)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _infer_side(outcome_team: str | None, home_team: str | None, away_team: str | None) -> str | None:
    outcome_key = canonicalize_team_name(outcome_team)
    if outcome_key and outcome_key == canonicalize_team_name(home_team):
        return "home"
    if outcome_key and outcome_key == canonicalize_team_name(away_team):
        return "away"
    return None
