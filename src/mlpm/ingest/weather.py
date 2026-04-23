"""Backfill per-game weather for every MLB game in `game_results`.

Source
------
Open-Meteo's Historical Weather API (https://archive-api.open-meteo.com/v1/archive).
Free, no API key, very permissive rate limits. We chunk by (stadium, season) so
each team-season = one request (roughly 180 days of hourly data). Total cost
for 2015-2026 is ~30 stadiums x 12 seasons = 360 requests.

What we extract
---------------
For every game, the weather at the stadium's lat/lon in the hour nearest to
first pitch (from `game_results.event_start_time`). We derive:

    temp_f                      -- game-time temperature
    wind_mph                    -- raw wind speed
    wind_dir_deg                -- compass direction the wind is coming FROM
    wind_out_to_cf_mph          -- wind component along home->CF axis; positive
                                   means blowing out (hitter-friendly)
    wind_crossfield_mph         -- magnitude of crosswind
    humidity_pct                -- relative humidity
    precipitation_in            -- precip at/near first pitch
    is_dome_sealed              -- 1 for fixed-dome parks (Tropicana); weather
                                   is neutralized. Retractable parks use the
                                   raw outdoor conditions since we don't know
                                   roof open/closed.

All rows are idempotently replaced on (game_id) via `replace_dataframe`.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path

import httpx
import pandas as pd

from mlpm.normalize.mapping import canonicalize_team_name
from mlpm.storage.duckdb import connect, query_dataframe, replace_dataframe

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
HTTP_TIMEOUT_SECONDS = 30.0
THROTTLE_SECONDS = 0.1  # be polite; Open-Meteo doesn't require this but one req / 100ms is fine
MAX_FETCH_ATTEMPTS = 6
RETRY_BACKOFF_SECONDS = 2.0
MAX_RETRY_SLEEP_SECONDS = 60.0


@dataclass(frozen=True)
class Stadium:
    team_key: str          # canonicalized team name (matches canonicalize_team_name)
    lat: float
    lon: float
    az_cf_deg: float       # compass bearing from home plate to center field
    roof: str              # 'open', 'retractable', 'fixed_dome'
    valid_from: date
    valid_to: date         # inclusive


# Stadium registry. Azimuths are approximate (within ~5 degrees, which is close
# enough for wind-component math). Two franchises moved during our window:
# ATL (Turner Field -> Truist Park in 2017) and TEX (Globe Life Park -> Globe
# Life Field in 2020), so we list them twice with valid-date windows.
_STADIUMS: tuple[Stadium, ...] = (
    Stadium("arizona diamondbacks",    33.4453, -112.0667,  23, "retractable", date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("atlanta braves",          33.8908,  -84.4677,  52, "open",        date(2017, 1, 1), date(2100, 1, 1)),
    Stadium("atlanta braves",          33.7348,  -84.3902,  72, "open",        date(1900, 1, 1), date(2016, 12, 31)),
    Stadium("baltimore orioles",       39.2840,  -76.6217,  63, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("boston red sox",          42.3467,  -71.0972,  45, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("chicago cubs",            41.9484,  -87.6553,  35, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("chicago white sox",       41.8299,  -87.6339,  54, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("cincinnati reds",         39.0974,  -84.5064,  45, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("cleveland guardians",     41.4962,  -81.6852,   6, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("cleveland indians",       41.4962,  -81.6852,   6, "open",        date(1900, 1, 1), date(2021, 12, 31)),
    Stadium("colorado rockies",        39.7560, -104.9942,   4, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("detroit tigers",          42.3390,  -83.0485,  59, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("houston astros",          29.7572,  -95.3556, 350, "retractable", date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("kansas city royals",      39.0517,  -94.4803,  45, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("los angeles angels",      33.8003, -117.8827,  49, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("los angeles dodgers",     34.0739, -118.2400,  27, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("miami marlins",           25.7781,  -80.2197,  38, "retractable", date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("milwaukee brewers",       43.0280,  -87.9712,  25, "retractable", date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("minnesota twins",         44.9817,  -93.2776,  88, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("new york mets",           40.7571,  -73.8458,  24, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("new york yankees",        40.8296,  -73.9262,  76, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("athletics",               37.7516, -122.2005,  56, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("oakland athletics",       37.7516, -122.2005,  56, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("philadelphia phillies",   39.9061,  -75.1665,  18, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("pittsburgh pirates",      40.4469,  -80.0057, 118, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("san diego padres",        32.7073, -117.1566,  36, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("seattle mariners",        47.5914, -122.3325,  50, "retractable", date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("san francisco giants",    37.7786, -122.3893,  91, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("st louis cardinals",      38.6226,  -90.1928,  49, "open",        date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("tampa bay rays",          27.7683,  -82.6534,   0, "fixed_dome",  date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("texas rangers",           32.7473,  -97.0817,  42, "retractable", date(2020, 1, 1), date(2100, 1, 1)),
    Stadium("texas rangers",           32.7513,  -97.0829,  45, "open",        date(1900, 1, 1), date(2019, 12, 31)),
    Stadium("toronto blue jays",       43.6414,  -79.3894,   0, "retractable", date(1900, 1, 1), date(2100, 1, 1)),
    Stadium("washington nationals",    38.8730,  -77.0074,  56, "open",        date(1900, 1, 1), date(2100, 1, 1)),
)


def stadium_for(home_team: str, game_date: date) -> Stadium | None:
    """Look up the right stadium for a (team, date) pair, handling franchise moves."""
    key = canonicalize_team_name(home_team)
    for stadium in _STADIUMS:
        if stadium.team_key == key and stadium.valid_from <= game_date <= stadium.valid_to:
            return stadium
    return None


def _wind_components(wind_speed_mph: float, wind_dir_deg: float, az_cf_deg: float) -> tuple[float, float]:
    """Decompose wind into along/across the home-plate-to-center-field axis.

    ``wind_dir_deg`` is the direction the wind is coming FROM (meteorological
    convention). ``az_cf_deg`` is the compass bearing from home plate to
    center field. Positive ``out_to_cf_mph`` means wind blowing from the
    batter toward center field — i.e. hitter-friendly.
    """
    if not wind_speed_mph or math.isnan(wind_speed_mph):
        return 0.0, 0.0
    # Wind going TOWARD (wind_dir + 180) compass heading.
    wind_toward = (wind_dir_deg + 180.0) % 360.0
    delta = ((wind_toward - az_cf_deg + 540.0) % 360.0) - 180.0
    out = wind_speed_mph * math.cos(math.radians(delta))
    cross = wind_speed_mph * math.sin(math.radians(delta))
    return out, cross


def _fetch_archive(
    client: httpx.Client,
    *,
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch hourly weather for one stadium, one date range. Returns a frame
    indexed by local-timezone timestamps with columns: temp_f, wind_mph,
    wind_dir_deg, humidity_pct, precipitation_in.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m,relativehumidity_2m,precipitation",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",  # Open-Meteo infers from lat/lon; all timestamps come back in local time.
    }
    data: dict[str, object] | None = None
    last_error: Exception | None = None
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            resp = client.get(OPEN_METEO_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
            break
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code != 429 or attempt >= MAX_FETCH_ATTEMPTS:
                raise
            retry_after = _retry_after_seconds(exc.response)
            sleep_seconds = retry_after if retry_after is not None else min(
                MAX_RETRY_SLEEP_SECONDS,
                RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)),
            )
            logger.warning(
                "Open-Meteo rate limited weather fetch for lat=%s lon=%s start=%s end=%s; "
                "retrying in %.1fs (attempt %s/%s).",
                lat,
                lon,
                start,
                end,
                sleep_seconds,
                attempt,
                MAX_FETCH_ATTEMPTS,
            )
            time.sleep(sleep_seconds)
        except httpx.RequestError as exc:
            last_error = exc
            if attempt >= MAX_FETCH_ATTEMPTS:
                raise
            sleep_seconds = min(
                MAX_RETRY_SLEEP_SECONDS,
                RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)),
            )
            logger.warning(
                "Open-Meteo request error for lat=%s lon=%s start=%s end=%s; retrying in %.1fs "
                "(attempt %s/%s): %s",
                lat,
                lon,
                start,
                end,
                sleep_seconds,
                attempt,
                MAX_FETCH_ATTEMPTS,
                exc,
            )
            time.sleep(sleep_seconds)

    if data is None:
        if last_error is not None:
            raise last_error
        return pd.DataFrame()

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "local_ts": pd.to_datetime(times),
            "temp_f": hourly.get("temperature_2m", []),
            "wind_mph": hourly.get("windspeed_10m", []),
            "wind_dir_deg": hourly.get("winddirection_10m", []),
            "humidity_pct": hourly.get("relativehumidity_2m", []),
            "precipitation_in": hourly.get("precipitation", []),
        }
    )
    df["timezone"] = data.get("timezone", "UTC")
    return df


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return max(float(header), 0.0)
    except ValueError:
        return None


def _game_hour(event_start_time, local_tz: str) -> tuple[date, int] | None:
    """Pick the (local_date, local_hour) matching first pitch.

    Falls back to 19:00 local (typical 7pm night game) when timestamp is
    missing — the residual noise vs the true hour is rarely material.
    """
    if event_start_time is None or pd.isna(event_start_time):
        return None
    ts = pd.Timestamp(event_start_time)
    # Treat naive timestamps as UTC (MLB Stats API convention).
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    try:
        local = ts.tz_convert(local_tz)
    except Exception:
        local = ts.tz_convert("UTC")
    return local.date(), int(local.hour)


def _default_game_hour(game_date: date) -> tuple[date, int]:
    return game_date, 19


def _weather_row_for_game(
    stadium_weather: pd.DataFrame,
    game_date: date,
    event_start_time,
) -> pd.Series | None:
    """Pick the single hour of weather closest to first pitch."""
    if stadium_weather.empty:
        return None
    tz = stadium_weather["timezone"].iloc[0]
    picked = _game_hour(event_start_time, tz) or _default_game_hour(game_date)
    target_date, target_hour = picked
    # Build target local timestamp and find nearest hour in the frame.
    target_ts = pd.Timestamp(target_date) + pd.Timedelta(hours=target_hour)
    diffs = (stadium_weather["local_ts"] - target_ts).abs()
    idx = diffs.idxmin()
    if pd.isna(idx):
        return None
    return stadium_weather.loc[idx]


def _neutralized_dome_row() -> dict[str, float]:
    """Indoor-climate-controlled dome conditions — treated as no weather signal."""
    return {
        "temp_f": 72.0,
        "wind_mph": 0.0,
        "wind_dir_deg": 0.0,
        "wind_out_to_cf_mph": 0.0,
        "wind_crossfield_mph": 0.0,
        "humidity_pct": 50.0,
        "precipitation_in": 0.0,
        "is_dome_sealed": 1,
    }


def _derive_game_weather(
    stadium: Stadium,
    stadium_weather: pd.DataFrame,
    game_date: date,
    event_start_time,
) -> dict[str, float]:
    if stadium.roof == "fixed_dome":
        return _neutralized_dome_row()
    row = _weather_row_for_game(stadium_weather, game_date, event_start_time)
    if row is None:
        return {
            "temp_f": float("nan"),
            "wind_mph": float("nan"),
            "wind_dir_deg": float("nan"),
            "wind_out_to_cf_mph": float("nan"),
            "wind_crossfield_mph": float("nan"),
            "humidity_pct": float("nan"),
            "precipitation_in": float("nan"),
            "is_dome_sealed": 0,
        }
    wind_mph = float(row["wind_mph"]) if pd.notna(row["wind_mph"]) else 0.0
    wind_dir = float(row["wind_dir_deg"]) if pd.notna(row["wind_dir_deg"]) else 0.0
    out, cross = _wind_components(wind_mph, wind_dir, stadium.az_cf_deg)
    return {
        "temp_f": float(row["temp_f"]) if pd.notna(row["temp_f"]) else float("nan"),
        "wind_mph": wind_mph,
        "wind_dir_deg": wind_dir,
        "wind_out_to_cf_mph": out,
        "wind_crossfield_mph": abs(cross),
        "humidity_pct": float(row["humidity_pct"]) if pd.notna(row["humidity_pct"]) else float("nan"),
        "precipitation_in": float(row["precipitation_in"]) if pd.notna(row["precipitation_in"]) else 0.0,
        "is_dome_sealed": 0,
    }


def _load_games(conn, start_date: str, end_date: str) -> pd.DataFrame:
    df = query_dataframe(
        conn,
        """
        SELECT
            game_id,
            CAST(game_date AS DATE) AS game_date,
            home_team,
            away_team,
            event_start_time
        FROM game_results
        WHERE game_date BETWEEN ? AND ?
        """,
        (start_date, end_date),
    )
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def _resume_skip_set(conn) -> set[str]:
    """Any game_ids already in game_weather — skip them on resume."""
    try:
        existing = query_dataframe(conn, "SELECT DISTINCT game_id FROM game_weather")
    except Exception:
        return set()
    return set(existing["game_id"].astype(str).tolist()) if not existing.empty else set()


def backfill_weather(
    start_date: str,
    end_date: str,
    *,
    db_path: Path | None = None,
    resume: bool = True,
) -> dict:
    """Fetch + store weather for every game in [start_date, end_date].

    Chunks requests by (stadium, season) so we hit Open-Meteo at most ~360
    times for a full 2015-2026 backfill. Idempotent: re-running only processes
    game_ids that haven't been stored yet (unless `resume=False`).
    """
    from mlpm.config.settings import settings as _settings
    db_path = db_path or _settings().duckdb_path

    conn = connect(db_path)
    try:
        games = _load_games(conn, start_date, end_date)
        if games.empty:
            return {"status": "no_games", "fetched": 0, "skipped": 0}

        games["home_team"] = games["home_team"].astype(str)
        games["season"] = pd.to_datetime(games["game_date"]).dt.year

        already = _resume_skip_set(conn) if resume else set()

        total_games = len(games)
        fetched = 0
        skipped_existing = 0
        skipped_no_stadium = 0
        fetch_errors = 0

        out_rows: list[dict] = []
        # Batch one HTTP call per (home_team, season).
        with httpx.Client() as client:
            for (home_team, season), group in games.groupby(["home_team", "season"]):
                group = group.sort_values("game_date")
                min_d = pd.to_datetime(group["game_date"].min()).date()
                max_d = pd.to_datetime(group["game_date"].max()).date()

                stadium = stadium_for(home_team, min_d)
                if stadium is None:
                    skipped_no_stadium += len(group)
                    logger.warning("no stadium registered for home_team=%r; skipping %s games.", home_team, len(group))
                    continue

                # Fixed domes get a canned neutral weather row; no HTTP needed.
                if stadium.roof == "fixed_dome":
                    stadium_weather = pd.DataFrame()
                else:
                    try:
                        stadium_weather = _fetch_archive(
                            client, lat=stadium.lat, lon=stadium.lon, start=min_d, end=max_d,
                        )
                    except httpx.HTTPError as exc:
                        fetch_errors += 1
                        logger.warning("Open-Meteo fetch failed for %s %s: %s", home_team, season, exc)
                        continue
                    time.sleep(THROTTLE_SECONDS)

                for row in group.itertuples(index=False):
                    game_id = str(row.game_id)
                    if game_id in already:
                        skipped_existing += 1
                        continue
                    weather = _derive_game_weather(
                        stadium,
                        stadium_weather,
                        row.game_date,
                        row.event_start_time,
                    )
                    out_rows.append(
                        {
                            "game_id": game_id,
                            "game_date": row.game_date,
                            "venue_team": home_team,
                            "az_cf_deg": stadium.az_cf_deg,
                            "roof_type": stadium.roof,
                            **weather,
                            "imported_at": datetime.now(tz=timezone.utc).replace(tzinfo=None),
                        }
                    )
                    fetched += 1

        if not out_rows:
            return {
                "status": "ok",
                "fetched": 0,
                "skipped_existing": skipped_existing,
                "skipped_no_stadium": skipped_no_stadium,
                "fetch_errors": fetch_errors,
                "total_games": total_games,
            }

        df_out = pd.DataFrame(out_rows)
        # Column order should match the storage schema.
        column_order = [
            "game_id", "game_date", "venue_team", "az_cf_deg", "roof_type",
            "temp_f", "wind_mph", "wind_dir_deg", "wind_out_to_cf_mph", "wind_crossfield_mph",
            "humidity_pct", "precipitation_in", "is_dome_sealed", "imported_at",
        ]
        df_out = df_out[column_order]
        replace_dataframe(conn, "game_weather", df_out, key_columns=["game_id"])

        logger.info(
            "weather backfill complete: fetched=%s skipped_existing=%s skipped_no_stadium=%s fetch_errors=%s total=%s",
            fetched, skipped_existing, skipped_no_stadium, fetch_errors, total_games,
        )
        return {
            "status": "ok",
            "fetched": fetched,
            "skipped_existing": skipped_existing,
            "skipped_no_stadium": skipped_no_stadium,
            "fetch_errors": fetch_errors,
            "total_games": total_games,
        }
    finally:
        conn.close()
