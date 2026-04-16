from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from mlpm.config.settings import settings
from mlpm.historical.common import (
    append_historical_import_run,
    build_historical_payload_path,
    has_successful_import_run,
    imported_at_utc,
    iter_date_windows,
    new_import_run_id,
    write_historical_payload,
)
from mlpm.historical.normalize_polymarket_history import polymarket_history_to_quote_rows
from mlpm.ingest.mlb_stats import fetch_final_results
from mlpm.ingest.polymarket import MLB_SPORT, POLYMARKET_MARKETS_URL, _fetch_mlb_tag_id
from mlpm.normalize.mapping import canonicalize_team_name
from mlpm.storage.duckdb import connect, replace_dataframe

POLYMARKET_CLOB_BASE_URL = "https://clob.polymarket.com"
POLYMARKET_PRICES_HISTORY_URL = f"{POLYMARKET_CLOB_BASE_URL}/prices-history"
POLYMARKET_BATCH_PRICES_HISTORY_URL = f"{POLYMARKET_CLOB_BASE_URL}/batch-prices-history"


def backfill_polymarket_history_for_games(
    start_date: str,
    end_date: str,
    interval: str = "1m",
    chunk_days: int = 7,
    resume: bool = True,
) -> dict[str, int | str]:
    totals = {
        "status": "ok",
        "import_run_id": "chunked",
        "request_count": 0,
        "payload_count": 0,
        "normalized_rows": 0,
        "chunks_total": 0,
        "chunks_completed": 0,
        "chunks_skipped": 0,
    }
    for chunk_start, chunk_end in iter_date_windows(start_date, end_date, chunk_days):
        totals["chunks_total"] += 1
        if resume and has_successful_import_run("polymarket", chunk_start, chunk_end):
            totals["chunks_skipped"] += 1
            continue
        result = _backfill_polymarket_history_chunk(chunk_start, chunk_end, interval=interval)
        totals["request_count"] += int(result["request_count"])
        totals["payload_count"] += int(result["payload_count"])
        totals["normalized_rows"] += int(result["normalized_rows"])
        totals["chunks_completed"] += 1
    return totals


def _backfill_polymarket_history_chunk(start_date: str, end_date: str, interval: str = "1m") -> dict[str, int | str]:
    started_at = imported_at_utc()
    import_run_id = new_import_run_id()
    request_count = 0
    payload_count = 0
    normalized_rows: list[dict] = []
    error_message: str | None = None
    status = "ok"
    try:
        games = fetch_final_results(start_date, end_date)
        if games.empty:
            return {
                "status": status,
                "import_run_id": import_run_id,
                "request_count": request_count,
                "payload_count": payload_count,
                "normalized_rows": 0,
            }
        candidates = discover_polymarket_markets_for_games(games)
        if candidates.empty:
            return {
                "status": status,
                "import_run_id": import_run_id,
                "request_count": request_count,
                "payload_count": payload_count,
                "normalized_rows": 0,
            }
        with httpx.Client(timeout=30.0) as client:
            for _, game_rows in candidates.groupby(["game_id", "market_id"]):
                market_rows = game_rows.to_dict(orient="records")
                asset_ids = [str(row["asset_id"]) for row in market_rows if row.get("asset_id")]
                if not asset_ids:
                    continue
                history_payloads = _fetch_polymarket_market_histories(
                    client,
                    market_rows,
                    start_date,
                    end_date,
                    interval,
                    import_run_id=import_run_id,
                )
                request_count += history_payloads["request_count"]
                payload_count += history_payloads["payload_count"]
                normalized_rows.extend(history_payloads["rows"])
        _write_historical_polymarket_rows(normalized_rows)
    except Exception as exc:
        status = "error"
        error_message = str(exc)
        raise
    finally:
        completed_at = imported_at_utc()
        append_historical_import_run(
            import_run_id=import_run_id,
            source="polymarket",
            started_at=started_at,
            completed_at=completed_at,
            start_date=start_date,
            end_date=end_date,
            status=status,
            request_count=request_count,
            payload_count=payload_count,
            normalized_rows=len(normalized_rows),
            error_message=error_message,
        )
    return {
        "status": status,
        "import_run_id": import_run_id,
        "request_count": request_count,
        "payload_count": payload_count,
        "normalized_rows": len(normalized_rows),
    }


def fetch_polymarket_price_history(
    asset_id: str,
    start_ts: int,
    end_ts: int,
    interval: str = "1m",
    fidelity: int = 1,
) -> dict:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            POLYMARKET_PRICES_HISTORY_URL,
            params={"market": asset_id, "startTs": start_ts, "endTs": end_ts, "interval": interval, "fidelity": fidelity},
        )
        response.raise_for_status()
        return response.json()


def fetch_polymarket_batch_price_history(
    asset_ids: list[str],
    start_ts: int,
    end_ts: int,
    interval: str = "1m",
    fidelity: int = 1,
) -> dict:
    if len(asset_ids) > 20:
        raise ValueError("Polymarket batch history requests support at most 20 asset ids.")
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            POLYMARKET_BATCH_PRICES_HISTORY_URL,
            json={"markets": asset_ids, "startTs": start_ts, "endTs": end_ts, "interval": interval, "fidelity": fidelity},
        )
        response.raise_for_status()
        return response.json()


def discover_polymarket_markets_for_games(games_df: pd.DataFrame, limit: int = 1000) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    with httpx.Client(timeout=30.0) as client:
        tag_id = _fetch_mlb_tag_id(client)
        offset = 0
        while True:
            response = client.get(
                POLYMARKET_MARKETS_URL,
                params={"tag_id": tag_id, "limit": limit, "offset": offset, "closed": "true", "active": "false"},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break
            for market in payload:
                if market.get("sportsMarketType") != "moneyline":
                    continue
                mapped_rows = _map_polymarket_market_to_games(market, games_df)
                records.extend(mapped_rows)
            if len(payload) < limit:
                break
            offset += limit
    return pd.DataFrame(records)


def _fetch_polymarket_market_histories(
    client: httpx.Client,
    market_rows: list[dict[str, object]],
    start_date: str,
    end_date: str,
    interval: str,
    import_run_id: str,
) -> dict[str, int | list[dict]]:
    request_count = 0
    payload_count = 0
    rows: list[dict] = []
    start_ts = int(datetime.fromisoformat(f"{start_date}T00:00:00+00:00").timestamp())
    end_ts = int(datetime.fromisoformat(f"{end_date}T23:59:59+00:00").timestamp())
    asset_ids = [str(row["asset_id"]) for row in market_rows if row.get("asset_id")]
    if len(asset_ids) <= 20:
        response = client.post(
            POLYMARKET_BATCH_PRICES_HISTORY_URL,
            json={"markets": asset_ids, "startTs": start_ts, "endTs": end_ts, "interval": interval, "fidelity": 1},
        )
        request_count += 1
        if response.is_success:
            payload = response.json()
            payload_count += 1
            payload_lookup = _batch_payload_lookup(payload)
            for market_row in market_rows:
                asset_id = str(market_row["asset_id"])
                raw_payload = payload_lookup.get(asset_id, {"history": []})
                path = build_historical_payload_path("polymarket", None, market_row["day_key"], asset_id)
                write_historical_payload(path, raw_payload)
                rows.extend(
                    polymarket_history_to_quote_rows(
                        _history_rows_from_batch(raw_payload),
                        {**market_row, "import_run_id": import_run_id, "raw_payload_path": str(path), "imported_at": imported_at_utc()},
                    )
                )
            return {"request_count": request_count, "payload_count": payload_count, "rows": rows}

    for market_row in market_rows:
        asset_id = str(market_row["asset_id"])
        response = client.get(
            POLYMARKET_PRICES_HISTORY_URL,
            params={"market": asset_id, "startTs": start_ts, "endTs": end_ts, "interval": interval, "fidelity": 1},
        )
        request_count += 1
        if not response.is_success:
            continue
        payload = response.json()
        payload_count += 1
        path = build_historical_payload_path("polymarket", None, market_row["day_key"], asset_id)
        write_historical_payload(path, payload)
        rows.extend(
            polymarket_history_to_quote_rows(
                payload.get("history", payload.get("data", [])),
                {**market_row, "import_run_id": import_run_id, "raw_payload_path": str(path), "imported_at": imported_at_utc()},
            )
        )
    return {"request_count": request_count, "payload_count": payload_count, "rows": rows}


def _map_polymarket_market_to_games(market: dict[str, Any], games_df: pd.DataFrame) -> list[dict[str, object]]:
    question = canonicalize_team_name(str(market.get("question") or ""))
    market_start = _parse_ts(
        market.get("gameStartTime") or market.get("startTime") or market.get("startDate") or market.get("endDate")
    )
    if market_start is None:
        return []
    outcomes = _ensure_list(market.get("outcomes"))
    asset_ids = _extract_asset_ids(market, len(outcomes))
    records: list[dict[str, object]] = []
    for game in games_df.to_dict(orient="records"):
        away_key = canonicalize_team_name(game["away_team"])
        home_key = canonicalize_team_name(game["home_team"])
        if away_key not in question or home_key not in question:
            continue
        game_start = _parse_ts(game.get("event_start_time") or f"{game['game_date']}T00:00:00Z")
        if game_start is not None and abs((market_start - game_start).total_seconds()) > 12 * 3600:
            continue
        for index, outcome_team in enumerate(outcomes):
            asset_id = asset_ids[index] if index < len(asset_ids) else None
            records.append(
                {
                    "import_run_id": None,
                    "event_id": str(market.get("conditionId") or market.get("id")),
                    "market_id": str(market.get("id")),
                    "asset_id": asset_id,
                    "game_id": str(game["game_id"]),
                    "event_start_time": game.get("event_start_time"),
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "outcome_team": outcome_team,
                    "day_key": str(game["game_date"]),
                }
            )
    return records


def _extract_asset_ids(market: dict[str, Any], outcome_count: int) -> list[str]:
    for key in ("clobTokenIds", "tokenIds", "assetIds"):
        value = market.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = [value]
        if isinstance(value, list):
            return [str(item) for item in value[:outcome_count]]
    tokens = market.get("tokens")
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except json.JSONDecodeError:
            tokens = None
    if isinstance(tokens, list):
        ids: list[str] = []
        for token in tokens:
            if isinstance(token, dict):
                ids.append(str(token.get("token_id") or token.get("asset_id") or token.get("id")))
        return ids[:outcome_count]
    return []


def _ensure_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _batch_payload_lookup(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("histories", "data", "results"):
            value = payload.get(key)
            if isinstance(value, dict):
                return {str(k): v for k, v in value.items()}
            if isinstance(value, list):
                lookup: dict[str, Any] = {}
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    market_key = str(item.get("market") or item.get("asset_id") or item.get("id"))
                    lookup[market_key] = item
                return lookup
    return {}


def _history_rows_from_batch(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("history", "data", "pricesHistory", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _write_historical_polymarket_rows(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows).drop_duplicates(subset=["market_id", "asset_id", "quote_ts", "outcome_team"])
    conn = connect(settings().duckdb_path)
    try:
        replace_dataframe(
            conn,
            "historical_polymarket_quotes",
            frame,
            ["market_id", "asset_id", "quote_ts", "outcome_team"],
        )
    finally:
        conn.close()


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
