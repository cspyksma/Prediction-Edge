"""Parse SportsbookReviewsOnline (SBRO) yearly MLB odds workbooks into a
tabular frame of de-vigged closing-line probabilities, then load them into
DuckDB's `historical_market_priors` table.

SBRO distributes one xlsx per season at
https://www.sportsbookreviewsonline.com/scoresoddsarchives/mlb/mlboddsarchives.htm
(the user downloads them locally — the sandbox can't reach that domain).

Sheet layout
------------
Every year's workbook has the same shape: two rows per game, with the
visiting team first and home team second. Columns differ slightly by year,
so we match on lowercased header names rather than positional indices.

Canonical columns we extract (case-insensitive contains match):
    date   → 3- or 4-digit MMDD integer (April 1 = 401, Oct 15 = 1015)
    team   → SBRO-style team abbreviation (e.g. TBR, SFG, KCR, CHW)
    final  → final runs scored (integer)
    close  → closing moneyline (American odds, signed integer)

Team abbreviations differ slightly from the MLB Stats API's abbreviations;
`SBRO_TEAM_OVERRIDES` handles the edge cases. We fall through to
`TEAM_ABBREVIATIONS` in `mlpm.normalize.mapping` for the rest.

Output
------
`parse_sbro_workbook(path, year)` returns one row per game with columns:
    game_date, away_team, home_team, away_score, home_score,
    away_ml_close, home_ml_close, away_implied_prob_raw,
    home_implied_prob_raw, away_fair_prob, home_fair_prob, source

`load_sbro_into_priors(df, conn)` joins with `game_results` on
(game_date, home_team, away_team) to attach the MLB game_id and inserts
into `historical_market_priors`.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from mlpm.normalize.mapping import TEAM_ABBREVIATIONS, canonicalize_team_name
from mlpm.normalize.probability import american_odds_to_implied_prob, devig_two_way
from mlpm.storage.duckdb import connect, query_dataframe, replace_dataframe

logger = logging.getLogger(__name__)

SBRO_SOURCE = "sportsbookreviewsonline"

# SBRO-specific abbreviation spellings that don't appear in
# mapping.TEAM_ABBREVIATIONS. Falls through to TEAM_ABBREVIATIONS if not here.
SBRO_TEAM_OVERRIDES: dict[str, str] = {
    "CHW": "Chicago White Sox",   # SBRO uses CHW; MLB feed uses CWS
    "WSN": "Washington Nationals",  # SBRO uses WSN; MLB feed uses WSH
    "TBD": "Tampa Bay Rays",       # Devil Rays historical label
    # MLB moved these franchises / rebrands — SBRO uses the contemporary
    # label; we map them to the current MLB Stats API franchise name so
    # the later (game_date, home_team, away_team) join lines up.
    "CLE": "Cleveland Guardians",
    "OAK": "Athletics",
    # SBRO's older pre-2022 workbooks use these city-style codes alongside
    # the standard 3-letter codes. Each one represents an entire team-season
    # (~162 games) so missing them drops ~5% of games per team.
    "KAN": "Kansas City Royals",
    "CUB": "Chicago Cubs",
    "LOS": "Los Angeles Dodgers",
    "WAS": "Washington Nationals",
    "SDG": "San Diego Padres",
    "SFO": "San Francisco Giants",
    "TAM": "Tampa Bay Rays",
    "BRS": "Boston Red Sox",  # seen once in 2020; likely a data-entry variant of BOS
}


def _resolve_team(abbrev: str | None) -> str | None:
    if abbrev is None:
        return None
    key = str(abbrev).strip().upper()
    if not key:
        return None
    if key in SBRO_TEAM_OVERRIDES:
        return SBRO_TEAM_OVERRIDES[key]
    return TEAM_ABBREVIATIONS.get(key)


# Franchise-rename aliases. MLB Stats API stores the contemporaneous team
# name, so 2021 Cleveland games are "Cleveland Indians" while 2025 games are
# "Cleveland Guardians". We collapse both to a single canonical token so the
# SBRO→game_results join works across renames.
FRANCHISE_ALIASES: dict[str, str] = {
    "cleveland indians": "cleveland guardians",
    "oakland athletics": "athletics",
    # Ray Vaughn — no-op renames left in for clarity; future franchise moves
    # should be added here.
}


def _franchise_key(name: str | None) -> str:
    """Canonicalize a team name to a franchise-stable key.

    Applies `canonicalize_team_name` first, then collapses any known
    historical alias to its current canonical form.
    """
    canonical = canonicalize_team_name(name)
    return FRANCHISE_ALIASES.get(canonical, canonical)


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column whose header contains any of the candidates."""
    for col in df.columns:
        for cand in candidates:
            if cand in col:
                return col
    return None


def _parse_mmdd(value, year: int) -> date | None:
    """SBRO dates are numeric MMDD (e.g. 401=Apr 1, 1015=Oct 15).

    Seasons roll: months 10-12 belong to `year`; months 1-3 belong to `year+1`
    (wildcard + World Series games in late Oct → Nov, spring training in March).
    The MLB season is roughly March-November, so a single-calendar-year
    interpretation is correct for every modern season file on SBRO.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    # Normalize to 4 digits: 401 → 0401, 1015 → 1015
    s = str(n).zfill(4)
    try:
        month = int(s[:-2])
        day = int(s[-2:])
    except ValueError:
        return None
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_american_ml(value) -> int | None:
    """Closing moneyline. SBRO stores as signed integer (-110, +145).

    Some sheets use "NL" for no line / postponed — return None there.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    if not s or s.upper() in {"NL", "N/A", "PK", "-", "OFF"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_score(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _devig_safe(home_ml: int | None, away_ml: int | None) -> tuple[float | None, float | None, float | None, float | None]:
    if home_ml is None or away_ml is None:
        return None, None, None, None
    try:
        raw_home = american_odds_to_implied_prob(home_ml)
        raw_away = american_odds_to_implied_prob(away_ml)
        fair_home, fair_away = devig_two_way(raw_home, raw_away)
    except (ValueError, ZeroDivisionError):
        return None, None, None, None
    return raw_home, raw_away, fair_home, fair_away


def parse_sbro_workbook(path: str | Path, year: int) -> pd.DataFrame:
    """Parse one SBRO yearly MLB odds workbook into one row per game.

    Args:
        path: Path to the .xlsx file.
        year: Calendar year the workbook covers (used to complete MMDD dates).

    Returns:
        DataFrame with per-game columns (see module docstring). Games where
        either team's closing moneyline is missing are dropped.
    """
    frame = pd.read_excel(path, engine="openpyxl")
    frame = _normalize_headers(frame)

    date_col = _find_column(frame, "date")
    team_col = _find_column(frame, "team")
    final_col = _find_column(frame, "final", "score")
    close_col = _find_column(frame, "close", "ml")  # close first; fall through to ml
    # SBRO's "Close" column is sometimes literally labeled "close" and
    # sometimes "close ml"; the "ml" fallback picks it up for older sheets.
    vh_col = _find_column(frame, "vh")  # optional; some sheets omit

    missing = [name for name, col in (("date", date_col), ("team", team_col), ("close", close_col)) if col is None]
    if missing:
        raise ValueError(f"SBRO workbook {path} missing required columns: {missing}. Found: {list(frame.columns)}")

    # Drop rows without a date (blank separators, header repeats, etc.).
    frame = frame[frame[date_col].notna()].reset_index(drop=True)

    if len(frame) % 2 != 0:
        logger.warning("SBRO workbook %s has an odd row count (%s); trimming last row", path, len(frame))
        frame = frame.iloc[:-1].reset_index(drop=True)

    games: list[dict] = []
    skipped_team = 0
    skipped_price = 0
    unresolved_team_codes: dict[str, int] = {}
    for i in range(0, len(frame), 2):
        row_a = frame.iloc[i]
        row_b = frame.iloc[i + 1]

        # Determine visitor/home. SBRO convention: visitor first, home second.
        # Older sheets explicitly mark V/H in the "vh" column; respect that
        # when present to guard against any stray out-of-order pairs.
        if vh_col is not None:
            va = str(row_a.get(vh_col, "")).strip().upper()[:1]
            vb = str(row_b.get(vh_col, "")).strip().upper()[:1]
            if va == "H" and vb == "V":
                row_a, row_b = row_b, row_a
            elif va == "N":  # Neutral-site game (rare — Field of Dreams, Little League Classic)
                # Treat row_a as visitor per convention; SBRO lists travelling team first.
                pass

        away_abbrev = row_a.get(team_col)
        home_abbrev = row_b.get(team_col)
        away_team = _resolve_team(away_abbrev)
        home_team = _resolve_team(home_abbrev)
        if not away_team or not home_team:
            skipped_team += 1
            for raw in (away_abbrev, home_abbrev):
                if raw is None:
                    continue
                key = str(raw).strip().upper()
                if key and _resolve_team(raw) is None:
                    unresolved_team_codes[key] = unresolved_team_codes.get(key, 0) + 1
            continue

        game_date = _parse_mmdd(row_a.get(date_col), year)
        if game_date is None:
            continue

        away_ml = _parse_american_ml(row_a.get(close_col))
        home_ml = _parse_american_ml(row_b.get(close_col))
        if away_ml is None or home_ml is None:
            skipped_price += 1
            continue

        away_score = _parse_score(row_a.get(final_col)) if final_col else None
        home_score = _parse_score(row_b.get(final_col)) if final_col else None

        raw_home, raw_away, fair_home, fair_away = _devig_safe(home_ml, away_ml)
        if fair_home is None:
            skipped_price += 1
            continue

        games.append(
            {
                "game_date": game_date,
                "away_team": away_team,
                "home_team": home_team,
                "away_score": away_score,
                "home_score": home_score,
                "away_ml_close": away_ml,
                "home_ml_close": home_ml,
                "away_implied_prob_raw": raw_away,
                "home_implied_prob_raw": raw_home,
                "away_fair_prob": fair_away,
                "home_fair_prob": fair_home,
                "source": SBRO_SOURCE,
            }
        )

    logger.info(
        "parsed SBRO workbook path=%s year=%s games=%s skipped_team=%s skipped_price=%s",
        path, year, len(games), skipped_team, skipped_price,
    )
    if unresolved_team_codes:
        top = sorted(unresolved_team_codes.items(), key=lambda kv: kv[1], reverse=True)[:15]
        logger.warning(
            "SBRO workbook %s had %s unresolved team codes. Top offenders: %s",
            path,
            len(unresolved_team_codes),
            ", ".join(f"{code!r}={count}" for code, count in top),
        )
    return pd.DataFrame(games)


def load_sbro_into_priors(
    priors_df: pd.DataFrame,
    *,
    conn=None,
    db_path: Path | None = None,
    book: str = "consensus_close",
) -> dict:
    """Insert parsed SBRO priors into `historical_market_priors`.

    Joins against `game_results` on (game_date, home_team, away_team) to
    attach the MLB game_id. Rows that don't match a known MLB game are
    logged and skipped — the backtest harness will ignore them anyway since
    they can't be joined to features.

    Returns summary dict with counts.
    """
    if priors_df.empty:
        return {"status": "ok", "rows_inserted": 0, "rows_unmatched": 0, "rows_total": 0}

    owns_conn = conn is None
    if owns_conn:
        from mlpm.config.settings import settings as _settings
        conn = connect(db_path or _settings().duckdb_path)

    try:
        # Bring in game_results so we can resolve game_id for each SBRO row.
        results = query_dataframe(
            conn,
            "SELECT game_id, CAST(game_date AS DATE) AS game_date, home_team, away_team FROM game_results",
        )
        if results.empty:
            logger.warning("game_results is empty — run `mlpm backfill-mlb` before loading SBRO priors.")
            return {"status": "empty_game_results", "rows_inserted": 0, "rows_unmatched": len(priors_df), "rows_total": len(priors_df)}

        results["home_key"] = results["home_team"].map(_franchise_key)
        results["away_key"] = results["away_team"].map(_franchise_key)
        results["game_date"] = pd.to_datetime(results["game_date"]).dt.date

        priors = priors_df.copy()
        priors["home_key"] = priors["home_team"].map(_franchise_key)
        priors["away_key"] = priors["away_team"].map(_franchise_key)
        priors["game_date"] = pd.to_datetime(priors["game_date"]).dt.date

        merged = priors.merge(
            results[["game_id", "game_date", "home_key", "away_key"]],
            on=["game_date", "home_key", "away_key"],
            how="left",
        )
        unmatched = merged[merged["game_id"].isna()]
        matched = merged[merged["game_id"].notna()].copy()

        if not unmatched.empty:
            logger.warning(
                "%s SBRO rows did not match any game_results record (date/team mismatch); dropping.",
                len(unmatched),
            )
            # Breakdown helps catch franchise-rename issues, neutral-site flips,
            # doubleheader date drift, etc. Show the top teams that show up in
            # unmatched rows so we know where to invest next.
            try:
                team_counts: dict[str, int] = {}
                for _, r in unmatched.iterrows():
                    for t in (r.get("home_team"), r.get("away_team")):
                        if isinstance(t, str) and t:
                            team_counts[t] = team_counts.get(t, 0) + 1
                top = sorted(team_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
                logger.warning(
                    "unmatched-team breakdown (top 10): %s",
                    ", ".join(f"{t}={c}" for t, c in top),
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                logger.debug("failed to log unmatched breakdown: %s", exc)

        if matched.empty:
            return {
                "status": "no_matches",
                "rows_inserted": 0,
                "rows_unmatched": len(unmatched),
                "rows_total": len(priors_df),
            }

        matched["book"] = book
        matched["imported_at"] = datetime.utcnow()
        matched["source"] = SBRO_SOURCE
        rename = {"home_ml_close": "home_moneyline_close", "away_ml_close": "away_moneyline_close"}
        matched = matched.rename(columns=rename)
        out_cols = [
            "game_id", "game_date", "source",
            "home_team", "away_team",
            "home_moneyline_close", "away_moneyline_close",
            "home_implied_prob_raw", "away_implied_prob_raw",
            "home_fair_prob", "away_fair_prob",
            "book", "imported_at",
        ]
        # Collapse any duplicate (game_id, source) pairs from doubleheader
        # cross-matches so each MLB game_id ends up with exactly one SBRO line.
        to_write = matched[out_cols].drop_duplicates(subset=["game_id", "source"], keep="first").reset_index(drop=True)
        # Idempotent: any existing (game_id, source) row for these SBRO rows
        # gets replaced. Re-parsing the same workbook is a no-op.
        replace_dataframe(
            conn,
            "historical_market_priors",
            to_write,
            key_columns=["game_id", "source"],
        )

        return {
            "status": "ok",
            "rows_inserted": int(len(to_write)),
            "rows_unmatched": int(len(unmatched)),
            "rows_total": int(len(priors_df)),
        }
    finally:
        if owns_conn:
            conn.close()


def ingest_sbro_directory(
    directory: str | Path,
    *,
    db_path: Path | None = None,
) -> dict:
    """Parse every `mlb odds YYYY.xlsx` in `directory` and load into DuckDB.

    Filename convention expected: `mlb odds 2015.xlsx`, `mlb odds 2016.xlsx`,
    etc. (case-insensitive). Year is parsed out of the filename.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"SBRO directory not found: {directory}")

    pattern = re.compile(r"(\d{4})", re.IGNORECASE)
    workbooks = sorted(p for p in directory.glob("*.xlsx") if pattern.search(p.stem))
    if not workbooks:
        return {"status": "no_files", "files": 0, "rows_inserted": 0}

    from mlpm.config.settings import settings as _settings
    conn = connect(db_path or _settings().duckdb_path)
    try:
        totals = {"files": 0, "rows_inserted": 0, "rows_unmatched": 0, "rows_total": 0}
        per_year: list[dict] = []
        for wb in workbooks:
            match = pattern.search(wb.stem)
            if not match:
                continue
            year = int(match.group(1))
            try:
                priors = parse_sbro_workbook(wb, year)
            except Exception as exc:
                logger.exception("Failed to parse SBRO workbook %s: %s", wb, exc)
                per_year.append({"year": year, "file": wb.name, "error": str(exc)})
                continue
            summary = load_sbro_into_priors(priors, conn=conn)
            per_year.append({"year": year, "file": wb.name, **summary})
            totals["files"] += 1
            totals["rows_inserted"] += int(summary.get("rows_inserted", 0))
            totals["rows_unmatched"] += int(summary.get("rows_unmatched", 0))
            totals["rows_total"] += int(summary.get("rows_total", 0))
        return {"status": "ok", **totals, "per_year": per_year}
    finally:
        conn.close()
