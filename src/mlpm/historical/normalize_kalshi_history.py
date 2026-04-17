from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from mlpm.normalize.mapping import canonicalize_team_name


def normalize_kalshi_candle_payload(
    raw_payload: dict,
    mapping_row: dict[str, Any],
) -> list[dict]:
    candles = _extract_candle_rows(raw_payload)
    rows: list[dict] = []
    imported_at = datetime.now(tz=UTC)
    outcome_team = mapping_row.get("outcome_team")
    home_team = mapping_row.get("home_team")
    away_team = mapping_row.get("away_team")
    event_start_time = mapping_row.get("event_start_time")
    event_start_dt = _parse_ts(event_start_time)
    side = _infer_side(outcome_team, home_team, away_team)

    for candle in candles:
        quote_dt = _history_timestamp(candle)
        if quote_dt is None:
            continue
        probability = _candle_probability(candle)
        if probability is None:
            continue
        home_implied_prob = probability if side == "home" else (1.0 - probability if side == "away" else probability)
        rows.append(
            {
                "import_run_id": mapping_row.get("import_run_id"),
                "source": "kalshi",
                "collection_mode": "historical_import",
                "market_id": mapping_row.get("market_id"),
                "event_id": mapping_row.get("event_id"),
                "ticker": mapping_row.get("ticker"),
                "game_id": mapping_row.get("game_id"),
                "event_start_time": event_start_time,
                "quote_ts": quote_dt,
                "outcome_team": outcome_team,
                "side": side,
                "home_implied_prob": home_implied_prob,
                "raw_prob_yes": probability,
                "quote_type": "candle_close",
                "volume": _optional_float(candle.get("volume_fp") or candle.get("volume") or candle.get("volume_delta")),
                "open_interest": _optional_float(candle.get("open_interest_fp") or candle.get("open_interest")),
                "best_price_source": "historical_candlesticks",
                "pre_pitch_flag": (quote_dt < event_start_dt) if event_start_dt is not None else None,
                "raw_payload_path": mapping_row.get("raw_payload_path"),
                "imported_at": imported_at,
            }
        )
    return rows


def normalize_kalshi_trade_payload(raw_payload: dict, mapping_df) -> list[dict]:
    rows: list[dict] = []
    imported_at = datetime.now(tz=UTC)
    mappings = {str(row["ticker"]): row for row in mapping_df.to_dict(orient="records")} if mapping_df is not None else {}
    for trade in raw_payload.get("trades", []):
        ticker = str(trade.get("ticker") or trade.get("market_ticker") or "")
        mapping_row = mappings.get(ticker, {})
        quote_dt = _history_timestamp(trade)
        probability = _optional_float(trade.get("price"))
        if quote_dt is None or probability is None:
            continue
        rows.append(
            {
                "import_run_id": mapping_row.get("import_run_id"),
                "source": "kalshi",
                "collection_mode": "historical_import",
                "market_id": ticker,
                "event_id": mapping_row.get("event_id"),
                "ticker": ticker,
                "game_id": mapping_row.get("game_id"),
                "event_start_time": mapping_row.get("event_start_time"),
                "quote_ts": quote_dt,
                "outcome_team": mapping_row.get("outcome_team"),
                "side": _infer_side(mapping_row.get("outcome_team"), mapping_row.get("home_team"), mapping_row.get("away_team")),
                "home_implied_prob": probability,
                "raw_prob_yes": probability,
                "quote_type": "trade",
                "volume": _optional_float(trade.get("count")),
                "open_interest": None,
                "best_price_source": "historical_trades",
                "pre_pitch_flag": None,
                "raw_payload_path": mapping_row.get("raw_payload_path"),
                "imported_at": imported_at,
            }
        )
    return rows


def _extract_candle_rows(raw_payload: dict[str, Any]) -> list[dict]:
    for key in ("candlesticks", "candles", "history"):
        value = raw_payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _candle_probability(candle: dict[str, Any]) -> float | None:
    price_block = candle.get("price")
    if isinstance(price_block, dict):
        for key in ("close_dollars", "close", "mean_dollars", "mean", "previous_dollars", "previous"):
            value = price_block.get(key)
            probability = _optional_float(value)
            if probability is not None:
                return probability
    for key in ("close", "price", "yes_close", "last_price"):
        value = candle.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, dict):
            continue
        probability = float(value)
        return probability if probability <= 1.0 else probability / 100.0
    return None


def _history_timestamp(item: dict[str, Any]) -> datetime | None:
    for key in ("end_period_ts", "end_ts", "ts", "timestamp", "time"):
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        parsed = _parse_ts(value)
        if parsed is not None:
            return parsed
    return None


def _parse_ts(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    if text.isdigit():
        return datetime.fromtimestamp(int(text), tz=UTC)
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if not pd.isna(parsed):
        return parsed.to_pydatetime().astimezone(UTC)
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


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    probability = float(value)
    return probability if probability <= 1.0 else probability / 100.0
