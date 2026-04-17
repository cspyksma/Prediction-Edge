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
from mlpm.normalize.mapping import canonicalize_team_name, team_aliases
from mlpm.storage.duckdb import append_dataframe, connect

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_MARKETS_URL = f"{KALSHI_BASE_URL}/markets"
KALSHI_CANDLES_URL = f"{KALSHI_BASE_URL}/series/{KALSHI_MLB_GAME_SERIES}/markets/{{ticker}}/candlesticks"
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
        "games_total": 0,
        "games_with_markets": 0,
        "games_with_pregame_quotes": 0,
        "candidate_markets": 0,
        "empty_payload_count": 0,
        "rate_limited_count": 0,
        "parse_error_count": 0,
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
        totals["games_total"] += int(result.get("games_total", 0))
        totals["games_with_markets"] += int(result.get("games_with_markets", 0))
        totals["games_with_pregame_quotes"] += int(result.get("games_with_pregame_quotes", 0))
        totals["candidate_markets"] += int(result.get("candidate_markets", 0))
        totals["empty_payload_count"] += int(result.get("empty_payload_count", 0))
        totals["rate_limited_count"] += int(result.get("rate_limited_count", 0))
        totals["parse_error_count"] += int(result.get("parse_error_count", 0))
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
    games_total = 0
    games_with_markets = 0
    games_with_pregame_quotes = 0
    candidate_markets = 0
    empty_payload_count = 0
    rate_limited_count = 0
    parse_error_count = 0
    error_message: str | None = None
    status = "ok"
    try:
        games = fetch_final_results(start_date, end_date)
        games_total = int(len(games))
        if games.empty:
            return {
                "status": status,
                "import_run_id": import_run_id,
                "request_count": request_count,
                "payload_count": payload_count,
                "normalized_rows": 0,
                "games_total": games_total,
                "games_with_markets": games_with_markets,
                "games_with_pregame_quotes": games_with_pregame_quotes,
                "candidate_markets": candidate_markets,
                "empty_payload_count": empty_payload_count,
                "rate_limited_count": rate_limited_count,
                "parse_error_count": parse_error_count,
            }
        historical_cutoff = get_kalshi_historical_cutoff()
        candidates = discover_kalshi_markets_for_games(games, historical_cutoff=historical_cutoff)
        candidate_markets = int(len(candidates))
        games_with_markets = int(candidates["game_id"].nunique()) if not candidates.empty else 0
        if candidates.empty:
            return {
                "status": status,
                "import_run_id": import_run_id,
                "request_count": request_count,
                "payload_count": payload_count,
                "normalized_rows": 0,
                "games_total": games_total,
                "games_with_markets": games_with_markets,
                "games_with_pregame_quotes": games_with_pregame_quotes,
                "candidate_markets": candidate_markets,
                "empty_payload_count": empty_payload_count,
                "rate_limited_count": rate_limited_count,
                "parse_error_count": parse_error_count,
            }
        with httpx.Client(timeout=30.0) as client:
            for row in candidates.to_dict(orient="records"):
                historical_mode = _should_use_historical_market(row, historical_cutoff)
                market_payload, market_status = fetch_kalshi_market(
                    row["ticker"],
                    client=client,
                    historical=historical_mode,
                )
                if market_status == "rate_limited":
                    rate_limited_count += 1
                elif market_status != "ok":
                    empty_payload_count += 1
                market = _extract_market_payload(market_payload)
                if not market:
                    continue
                request_count += 1
                payload_count += 1
                row = _enrich_candidate_with_market(row, market)
                market_path = build_historical_payload_path("kalshi", "markets", row["day_key"], row["ticker"])
                write_historical_payload(market_path, market_payload)

                start_ts, end_ts = _market_query_window(row, market)
                candle_payload, candle_status = fetch_kalshi_candles(
                    row["ticker"],
                    start_ts,
                    end_ts,
                    period_interval=period_interval,
                    client=client,
                    historical=historical_mode,
                )
                if candle_status == "rate_limited":
                    rate_limited_count += 1
                elif candle_status != "ok":
                    empty_payload_count += 1
                if not candle_payload:
                    continue
                request_count += 1
                payload_count += 1
                candle_path = build_historical_payload_path("kalshi", "candles", row["day_key"], row["ticker"])
                write_historical_payload(candle_path, candle_payload)
                try:
                    candle_rows = normalize_kalshi_candle_payload(
                        candle_payload,
                        {
                            "import_run_id": import_run_id,
                            "event_id": row.get("event_id"),
                            "ticker": row.get("ticker"),
                            "market_id": row.get("market_id") or row.get("ticker"),
                            "game_id": row.get("game_id"),
                            "event_start_time": row.get("event_start_time"),
                            "outcome_team": row.get("outcome_team"),
                            "home_team": row.get("home_team"),
                            "away_team": row.get("away_team"),
                            "raw_payload_path": str(candle_path),
                        },
                    )
                except (TypeError, ValueError):
                    parse_error_count += 1
                    continue
                for candle_row in candle_rows:
                    candle_row["side"] = candle_row.get("side") or _infer_side_from_row(row)
                normalized_rows.extend(candle_rows)
                if include_trades:
                    trades_payload = fetch_kalshi_historical_trades(
                        start_ts,
                        end_ts,
                        ticker=row["ticker"],
                        client=client,
                    )
                    request_count += 1
                    payload_count += 1
                    trade_path = build_historical_payload_path("kalshi", "trades", row["day_key"], row["ticker"])
                    write_historical_payload(trade_path, trades_payload)
        _write_historical_kalshi_rows(normalized_rows, games["game_id"].astype(str).tolist())
        games_with_pregame_quotes = len(
            {
                str(row["game_id"])
                for row in normalized_rows
                if row.get("game_id") and bool(row.get("pre_pitch_flag"))
            }
        )
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
            games_total=games_total,
            games_with_markets=games_with_markets,
            games_with_pregame_quotes=games_with_pregame_quotes,
            candidate_markets=candidate_markets,
            empty_payload_count=empty_payload_count,
            rate_limited_count=rate_limited_count,
            parse_error_count=parse_error_count,
            error_message=error_message,
        )
    return {
        "status": status,
        "import_run_id": import_run_id,
        "request_count": request_count,
        "payload_count": payload_count,
        "normalized_rows": len(normalized_rows),
        "games_total": games_total,
        "games_with_markets": games_with_markets,
        "games_with_pregame_quotes": games_with_pregame_quotes,
        "candidate_markets": candidate_markets,
        "empty_payload_count": empty_payload_count,
        "rate_limited_count": rate_limited_count,
        "parse_error_count": parse_error_count,
    }


def fetch_kalshi_market(
    ticker: str,
    client: httpx.Client | None = None,
    historical: bool = True,
) -> tuple[dict, str]:
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        if historical:
            response = client.get(f"{KALSHI_HISTORICAL_MARKETS_URL}/{ticker}")
        else:
            response = client.get(f"{KALSHI_MARKETS_URL}/{ticker}")
        if response.status_code in (400, 404):
            response = client.get(
                KALSHI_HISTORICAL_MARKETS_URL if historical else KALSHI_MARKETS_URL,
                params={"tickers": ticker, "limit": 1},
            )
        if response.status_code in (400, 404):
            return {}, "missing"
        if response.status_code == 429:
            return {}, "rate_limited"
        response.raise_for_status()
        return response.json(), "ok"
    finally:
        if owns_client:
            client.close()


def fetch_kalshi_candles(
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1,
    client: httpx.Client | None = None,
    historical: bool = True,
) -> tuple[dict, str]:
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        response = client.get(
            (KALSHI_HISTORICAL_CANDLES_URL if historical else KALSHI_CANDLES_URL).format(ticker=ticker),
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
        )
        if response.status_code in (400, 404, 429):
            return {}, "rate_limited" if response.status_code == 429 else "missing"
        response.raise_for_status()
        return response.json(), "ok"
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


def discover_kalshi_markets_for_games(
    games_df: pd.DataFrame,
    limit: int = 200,
    historical_cutoff: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()
    game_times = _game_start_series(games_df)
    min_start = game_times.min()
    max_start = game_times.max()
    games = games_df.copy()
    games["event_start_dt"] = game_times
    rows: list[dict[str, object]] = []
    seen_tickers: set[str] = set()
    historical_mode = False
    if historical_cutoff is not None:
        cutoff_dt = _parse_market_time(historical_cutoff.get("market_settled_ts"))
        historical_mode = bool(cutoff_dt is not None and max_start is not None and max_start <= cutoff_dt)
    with httpx.Client(timeout=30.0) as client:
        cursor: str | None = None
        seen_in_range = False
        while True:
            params = {"series_ticker": KALSHI_MLB_GAME_SERIES, "limit": limit}
            if not historical_mode:
                params["status"] = "settled"
            if cursor:
                params["cursor"] = cursor
            response = client.get(KALSHI_HISTORICAL_MARKETS_URL if historical_mode else KALSHI_MARKETS_URL, params=params)
            if response.status_code == 429:
                break
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
                for mapped in _map_kalshi_market_to_games(market, games):
                    ticker = str(mapped.get("ticker") or "")
                    if not ticker or ticker in seen_tickers:
                        continue
                    seen_tickers.add(ticker)
                    rows.append(mapped)
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
    market_close = _parse_market_time(market.get("close_time"))
    records: list[dict[str, object]] = []
    for game in games_df.to_dict(orient="records"):
        away_alias_set = _market_team_aliases(str(game["away_team"]))
        home_alias_set = _market_team_aliases(str(game["home_team"]))
        away_match = _text_matches_any_alias(text, away_alias_set)
        home_match = _text_matches_any_alias(text, home_alias_set)
        if not away_match or not home_match:
            continue
        event_start = _parse_market_time(game.get("event_start_dt") or game.get("event_start_time"))
        if market_close is not None and event_start is not None:
            if abs((market_close - event_start).total_seconds()) > 8 * 3600:
                continue
        outcome_team = _resolve_kalshi_outcome_team(market, game)
        if outcome_team is None:
            continue
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
        if _text_matches_any_alias(value, _market_team_aliases(str(game["home_team"]))):
            return str(game["home_team"])
        if _text_matches_any_alias(value, _market_team_aliases(str(game["away_team"]))):
            return str(game["away_team"])
    return None


def _write_historical_kalshi_rows(rows: list[dict], game_ids: list[str] | None = None) -> None:
    if not rows and not game_ids:
        return
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["ticker", "quote_ts", "outcome_team", "quote_type"])
    unique_game_ids = pd.DataFrame(
        {"game_id": sorted({str(game_id) for game_id in (game_ids or []) if game_id not in (None, "")})}
    )
    conn = connect(settings().duckdb_path)
    try:
        if not unique_game_ids.empty:
            conn.register("historical_kalshi_game_ids", unique_game_ids)
            conn.execute(
                """
                DELETE FROM historical_kalshi_quotes AS target
                USING historical_kalshi_game_ids AS source
                WHERE target.game_id = source.game_id
                """
            )
            conn.unregister("historical_kalshi_game_ids")
        if not frame.empty:
            append_dataframe(conn, "historical_kalshi_quotes", frame)
    finally:
        conn.close()


def _game_window_start_ts(start_date: str) -> int:
    return int(datetime.fromisoformat(f"{start_date}T00:00:00+00:00").timestamp())


def _game_window_end_ts(end_date: str) -> int:
    return int(datetime.fromisoformat(f"{end_date}T23:59:59+00:00").timestamp())


def _market_query_window(row: dict[str, Any], market: dict[str, Any]) -> tuple[int, int]:
    event_start = _parse_market_time(row.get("event_start_time")) or _parse_market_time(market.get("close_time"))
    open_time = _parse_market_time(market.get("open_time"))
    start_dt = open_time or (event_start - pd.Timedelta(days=1) if event_start is not None else None)
    end_dt = event_start or start_dt
    if start_dt is None or end_dt is None:
        raise ValueError(f"Unable to determine Kalshi candle window for ticker {row.get('ticker')}")
    if end_dt < start_dt:
        end_dt = start_dt
    return int(start_dt.timestamp()), int(end_dt.timestamp())


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


def _extract_market_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    market = payload.get("market")
    if isinstance(market, dict):
        return market
    markets = payload.get("markets")
    if isinstance(markets, list):
        for item in markets:
            if isinstance(item, dict):
                return item
    return {}


def _enrich_candidate_with_market(row: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["event_id"] = market.get("event_ticker") or enriched.get("event_id")
    enriched["market_id"] = market.get("ticker") or enriched.get("market_id") or enriched.get("ticker")
    enriched["ticker"] = market.get("ticker") or enriched.get("ticker")
    return enriched


def _should_use_historical_market(row: dict[str, Any], cutoff_payload: dict[str, Any]) -> bool:
    event_start = _parse_market_time(row.get("event_start_time"))
    market_cutoff = _parse_market_time(cutoff_payload.get("market_settled_ts"))
    if event_start is None or market_cutoff is None:
        return True
    return event_start <= market_cutoff


def _text_matches_any_alias(text: str, aliases: set[str]) -> bool:
    for alias in aliases:
        if alias and alias in text:
            return True
    return False


def _market_team_aliases(team_name: str) -> set[str]:
    aliases = set(team_aliases(team_name))
    custom_aliases = {
        "Athletics": {"a s", "athletics", "a's"},
        "Arizona Diamondbacks": {"arizona", "az"},
        "Kansas City Royals": {"kansas city", "kc"},
        "Chicago White Sox": {"chicago ws", "white sox", "cws"},
        "Chicago Cubs": {"chicago c", "cubs", "chc"},
        "Los Angeles Angels": {"los angeles a", "angels", "laa"},
        "Los Angeles Dodgers": {"los angeles d", "dodgers", "lad"},
        "New York Mets": {"new york m", "mets", "nym"},
        "New York Yankees": {"new york y", "yankees", "nyy"},
        "San Diego Padres": {"san diego", "sd", "padres"},
        "San Francisco Giants": {"san francisco", "sf", "giants"},
        "St. Louis Cardinals": {"st louis", "cardinals", "stl"},
        "Tampa Bay Rays": {"tampa bay", "tb", "rays"},
    }
    aliases.update(canonicalize_team_name(alias) for alias in custom_aliases.get(team_name, set()))
    return {alias for alias in aliases if alias}
