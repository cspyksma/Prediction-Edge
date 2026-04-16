from __future__ import annotations

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
from mlpm.historical.normalize_kalshi_history import normalize_kalshi_candle_payload
from mlpm.ingest.kalshi import KALSHI_MLB_GAME_SERIES
from mlpm.ingest.mlb_stats import fetch_final_results
from mlpm.normalize.mapping import canonicalize_team_name
from mlpm.storage.duckdb import connect, replace_dataframe

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_HISTORICAL_CUTOFF_URL = f"{KALSHI_BASE_URL}/historical/cutoff"
KALSHI_HISTORICAL_MARKETS_URL = f"{KALSHI_BASE_URL}/historical/markets"
KALSHI_HISTORICAL_CANDLES_URL = f"{KALSHI_BASE_URL}/historical/markets/{{ticker}}/candlesticks"
KALSHI_HISTORICAL_TRADES_URL = f"{KALSHI_BASE_URL}/historical/trades"
KALSHI_TEAM_CODES = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def get_kalshi_historical_cutoff() -> dict:
    with httpx.Client(timeout=30.0) as client:
        response = client.get(KALSHI_HISTORICAL_CUTOFF_URL)
        response.raise_for_status()
        return response.json()


def backfill_kalshi_history_for_games(
    start_date: str,
    end_date: str,
    period_interval: int = 1,
    include_trades: bool = False,
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
        if resume and has_successful_import_run("kalshi", chunk_start, chunk_end):
            totals["chunks_skipped"] += 1
            continue
        result = _backfill_kalshi_history_chunk(
            chunk_start,
            chunk_end,
            period_interval=period_interval,
            include_trades=include_trades,
        )
        totals["request_count"] += int(result["request_count"])
        totals["payload_count"] += int(result["payload_count"])
        totals["normalized_rows"] += int(result["normalized_rows"])
        totals["chunks_completed"] += 1
    return totals


def _backfill_kalshi_history_chunk(
    start_date: str,
    end_date: str,
    period_interval: int = 1,
    include_trades: bool = False,
) -> dict[str, int | str]:
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
        get_kalshi_historical_cutoff()
        candidates = build_kalshi_ticker_candidates(games)
        if candidates.empty:
            candidates = discover_kalshi_markets_for_games(games)
        if candidates.empty:
            return {
                "status": status,
                "import_run_id": import_run_id,
                "request_count": request_count,
                "payload_count": payload_count,
                "normalized_rows": 0,
            }
        with httpx.Client(timeout=30.0) as client:
            for row in candidates.to_dict(orient="records"):
                market_payload = fetch_kalshi_historical_market(row["ticker"], client=client)
                if not market_payload:
                    continue
                request_count += 1
                payload_count += 1
                market_path = build_historical_payload_path("kalshi", "markets", row["day_key"], row["ticker"])
                write_historical_payload(market_path, market_payload)

                candle_payload = fetch_kalshi_historical_candles(
                    row["ticker"],
                    _game_window_start_ts(start_date),
                    _game_window_end_ts(end_date),
                    period_interval=period_interval,
                    client=client,
                )
                if not candle_payload:
                    continue
                request_count += 1
                payload_count += 1
                candle_path = build_historical_payload_path("kalshi", "candles", row["day_key"], row["ticker"])
                write_historical_payload(candle_path, candle_payload)
                candle_rows = normalize_kalshi_candle_payload(candle_payload, row["ticker"], game_id=row["game_id"])
                for candle_row in candle_rows:
                    candle_row["import_run_id"] = import_run_id
                    candle_row["event_id"] = row.get("event_id")
                    candle_row["ticker"] = row.get("ticker")
                    candle_row["event_start_time"] = row.get("event_start_time")
                    candle_row["outcome_team"] = row.get("outcome_team")
                    candle_row["side"] = candle_row.get("side") or _infer_side_from_row(row)
                    candle_row["raw_payload_path"] = str(candle_path)
                normalized_rows.extend(candle_rows)
                if include_trades:
                    trades_payload = fetch_kalshi_historical_trades(
                        _game_window_start_ts(start_date),
                        _game_window_end_ts(end_date),
                        ticker=row["ticker"],
                        client=client,
                    )
                    request_count += 1
                    payload_count += 1
                    trade_path = build_historical_payload_path("kalshi", "trades", row["day_key"], row["ticker"])
                    write_historical_payload(trade_path, trades_payload)
        _write_historical_kalshi_rows(normalized_rows)
    except Exception as exc:
        status = "error"
        error_message = str(exc)
        raise
    finally:
        append_historical_import_run(
            import_run_id=import_run_id,
            source="kalshi",
            started_at=started_at,
            completed_at=imported_at_utc(),
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


def fetch_kalshi_historical_market(ticker: str, client: httpx.Client | None = None) -> dict:
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        response = client.get(f"{KALSHI_HISTORICAL_MARKETS_URL}/{ticker}")
        if response.status_code == 404:
            response = client.get(KALSHI_HISTORICAL_MARKETS_URL, params={"ticker": ticker, "limit": 1})
        if response.status_code in (400, 404):
            return {}
        if response.status_code == 429:
            return {}
        response.raise_for_status()
        return response.json()
    finally:
        if owns_client:
            client.close()


def fetch_kalshi_historical_candles(
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1,
    client: httpx.Client | None = None,
) -> dict:
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        response = client.get(
            KALSHI_HISTORICAL_CANDLES_URL.format(ticker=ticker),
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )
        if response.status_code in (400, 404, 429):
            return {}
        response.raise_for_status()
        return response.json()
    finally:
        if owns_client:
            client.close()


def fetch_kalshi_historical_trades(
    start_ts: int,
    end_ts: int,
    ticker: str | None = None,
    client: httpx.Client | None = None,
) -> dict:
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        params: dict[str, object] = {"start_ts": start_ts, "end_ts": end_ts}
        if ticker:
            params["ticker"] = ticker
        response = client.get(KALSHI_HISTORICAL_TRADES_URL, params=params)
        response.raise_for_status()
        return response.json()
    finally:
        if owns_client:
            client.close()


def discover_kalshi_markets_for_games(games_df: pd.DataFrame, limit: int = 1000) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()
    game_times = _game_start_series(games_df)
    min_start = game_times.min()
    max_start = game_times.max()
    rows: list[dict[str, object]] = []
    with httpx.Client(timeout=30.0) as client:
        cursor: str | None = None
        seen_in_range = False
        while True:
            params = {"series_ticker": KALSHI_MLB_GAME_SERIES, "limit": limit}
            if cursor:
                params["cursor"] = cursor
            response = client.get(KALSHI_HISTORICAL_MARKETS_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            markets = payload.get("markets", payload.get("data", []))
            if not markets:
                break
            page_has_in_range_market = False
            for market in markets:
                if not _is_relevant_kalshi_market(market):
                    continue
                close_time = _parse_market_time(market.get("close_time"))
                if close_time is not None and max_start is not None and close_time > max_start + pd.Timedelta(days=1):
                    continue
                if close_time is not None and min_start is not None and close_time < min_start - pd.Timedelta(days=1):
                    continue
                page_has_in_range_market = True
                rows.extend(_map_kalshi_market_to_games(market, games_df))
            if page_has_in_range_market:
                seen_in_range = True
            cursor = payload.get("cursor")
            if not cursor or len(markets) < limit:
                break
            if seen_in_range and not page_has_in_range_market:
                break
    return pd.DataFrame(rows)


def build_kalshi_ticker_candidates(games_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for game in games_df.to_dict(orient="records"):
        start_time = _parse_market_time(game.get("event_start_time"))
        away_code = KALSHI_TEAM_CODES.get(str(game.get("away_team") or ""))
        home_code = KALSHI_TEAM_CODES.get(str(game.get("home_team") or ""))
        if start_time is None or away_code is None or home_code is None:
            continue
        base = f"{KALSHI_MLB_GAME_SERIES}-{start_time.strftime('%y%b%d%H%M').upper()}{away_code}{home_code}"
        rows.append(
            {
                "event_id": base,
                "market_id": f"{base}-{home_code}",
                "ticker": f"{base}-{home_code}",
                "game_id": str(game["game_id"]),
                "event_start_time": game.get("event_start_time"),
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "outcome_team": game["home_team"],
                "day_key": str(game["game_date"]),
            }
        )
        rows.append(
            {
                "event_id": base,
                "market_id": f"{base}-{away_code}",
                "ticker": f"{base}-{away_code}",
                "game_id": str(game["game_id"]),
                "event_start_time": game.get("event_start_time"),
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "outcome_team": game["away_team"],
                "day_key": str(game["game_date"]),
            }
        )
    return pd.DataFrame(rows)


def _map_kalshi_market_to_games(market: dict[str, Any], games_df: pd.DataFrame) -> list[dict[str, object]]:
    text = canonicalize_team_name(
        " ".join(
            str(market.get(key) or "")
            for key in ("title", "yes_sub_title", "no_sub_title", "event_ticker")
        )
    )
    records: list[dict[str, object]] = []
    for game in games_df.to_dict(orient="records"):
        away_key = canonicalize_team_name(game["away_team"])
        home_key = canonicalize_team_name(game["home_team"])
        if away_key not in text or home_key not in text:
            continue
        outcome_team = _resolve_kalshi_outcome_team(market, game)
        records.append(
            {
                "event_id": market.get("event_ticker"),
                "market_id": market.get("ticker"),
                "ticker": market.get("ticker"),
                "game_id": str(game["game_id"]),
                "event_start_time": game.get("event_start_time"),
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "outcome_team": outcome_team,
                "day_key": str(game["game_date"]),
            }
        )
    return records


def _resolve_kalshi_outcome_team(market: dict[str, Any], game: dict[str, Any]) -> str | None:
    for key in ("yes_sub_title", "title"):
        value = canonicalize_team_name(str(market.get(key) or ""))
        if canonicalize_team_name(game["home_team"]) in value:
            return str(game["home_team"])
        if canonicalize_team_name(game["away_team"]) in value:
            return str(game["away_team"])
    return None


def _write_historical_kalshi_rows(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "quote_ts", "outcome_team", "quote_type"])
    conn = connect(settings().duckdb_path)
    try:
        replace_dataframe(
            conn,
            "historical_kalshi_quotes",
            frame,
            ["ticker", "quote_ts", "outcome_team", "quote_type"],
        )
    finally:
        conn.close()


def _game_window_start_ts(start_date: str) -> int:
    return int(datetime.fromisoformat(f"{start_date}T00:00:00+00:00").timestamp())


def _game_window_end_ts(end_date: str) -> int:
    return int(datetime.fromisoformat(f"{end_date}T23:59:59+00:00").timestamp())


def _infer_side_from_row(row: dict[str, object]) -> str | None:
    outcome = canonicalize_team_name(str(row.get("outcome_team") or ""))
    if outcome and outcome == canonicalize_team_name(str(row.get("home_team") or "")):
        return "home"
    if outcome and outcome == canonicalize_team_name(str(row.get("away_team") or "")):
        return "away"
    return None


def _is_relevant_kalshi_market(market: dict[str, Any]) -> bool:
    ticker = str(market.get("ticker") or "")
    event_ticker = str(market.get("event_ticker") or "")
    text = canonicalize_team_name(
        " ".join(str(market.get(key) or "") for key in ("title", "yes_sub_title", "no_sub_title"))
    )
    if not ticker.startswith(f"{KALSHI_MLB_GAME_SERIES}-"):
        return False
    if not event_ticker.startswith(f"{KALSHI_MLB_GAME_SERIES}-"):
        return False
    if "wins" not in text and "win" not in text:
        return False
    return True


def _parse_market_time(value: object) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _game_start_series(games_df: pd.DataFrame) -> pd.Series:
    if "event_start_time" in games_df.columns:
        series = pd.to_datetime(games_df["event_start_time"], utc=True, errors="coerce")
    else:
        series = pd.Series(dtype="datetime64[ns, UTC]")
    if series.isna().all() and "game_date" in games_df.columns:
        return pd.to_datetime(games_df["game_date"], utc=True, errors="coerce")
    if "game_date" in games_df.columns:
        fallback = pd.to_datetime(games_df["game_date"], utc=True, errors="coerce")
        return series.fillna(fallback)
    return series
