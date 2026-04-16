from __future__ import annotations

import re
from datetime import UTC, datetime

import pandas as pd

TEAM_ABBREVIATIONS = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "ATH": "Athletics",
    "OAK": "Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SDP": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants",
    "SFG": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}


def canonicalize_team_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def map_odds_to_games(odds_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty or games_df.empty:
        return pd.DataFrame()

    odds = odds_df.copy()
    games = games_df.copy()
    odds["home_key"] = odds["home_team"].map(canonicalize_team_name)
    odds["away_key"] = odds["away_team"].map(canonicalize_team_name)
    games["home_key"] = games["home_team"].map(canonicalize_team_name)
    games["away_key"] = games["away_team"].map(canonicalize_team_name)
    odds["event_start_dt"] = odds["event_start_time"].map(_parse_ts)
    games["event_start_dt"] = games["event_start_time"].map(_parse_ts)

    merged = odds.merge(
        games[["game_id", "home_key", "away_key", "event_start_dt"]],
        on=["home_key", "away_key"],
        suffixes=("", "_mlb"),
        how="left",
    )
    merged["start_delta_sec"] = (
        merged["event_start_dt"] - merged["event_start_dt_mlb"]
    ).dt.total_seconds().abs()
    return merged[merged["start_delta_sec"].fillna(999999) <= 3600].copy()


def infer_kalshi_team(market_text: str, games_df: pd.DataFrame) -> tuple[str | None, str | None]:
    text = canonicalize_team_name(market_text)
    for game in games_df.itertuples(index=False):
        away_aliases = team_aliases(game.away_team)
        home_aliases = team_aliases(game.home_team)
        away_match = any(alias and alias in text for alias in away_aliases)
        home_match = any(alias and alias in text for alias in home_aliases)
        if away_match and home_match:
            return str(game.game_id), game.away_team
        if away_match:
            return str(game.game_id), game.away_team
        if home_match:
            return str(game.game_id), game.home_team
    return None, None


def map_kalshi_to_games(kalshi_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    if kalshi_df.empty or games_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for market in kalshi_df.to_dict(orient="records"):
        game_id = infer_game_id_from_kalshi_event(str(market.get("event_id") or ""), games_df)
        text = " ".join(
            str(market.get(key) or "")
            for key in ["market_title", "event_title", "event_sub_title", "yes_sub_title", "no_sub_title", "event_id"]
        )
        fallback_game_id, outcome_team = infer_kalshi_team(text, games_df)
        game_id = game_id or fallback_game_id
        if not game_id or not outcome_team:
            continue
        row = dict(market)
        row["game_id"] = game_id
        team_hint = str(str(market.get("market_ticker") or "").split("-")[-1] or market.get("yes_sub_title") or outcome_team)
        row["outcome_team"] = resolve_team_name(team_hint, games_df, game_id) or outcome_team
        rows.append(row)
    return pd.DataFrame(rows)


def map_market_text_to_games(markets_df: pd.DataFrame, games_df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    if markets_df.empty or games_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for market in markets_df.to_dict(orient="records"):
        text = str(market.get(text_column) or "")
        game_id, outcome_team = infer_kalshi_team(text, games_df)
        if not game_id or not outcome_team:
            continue
        row = dict(market)
        row["game_id"] = game_id
        if not row.get("outcome_team"):
            row["outcome_team"] = outcome_team
        rows.append(row)
    return pd.DataFrame(rows)


def team_aliases(team_name: str | None) -> set[str]:
    if not team_name:
        return set()
    canonical = canonicalize_team_name(team_name)
    parts = canonical.split()
    aliases = {canonical}
    if len(parts) > 1:
        aliases.add(parts[-1])
        city = " ".join(parts[:-1])
        aliases.add(city)
    for abbreviation, full_name in TEAM_ABBREVIATIONS.items():
        if canonicalize_team_name(full_name) == canonical:
            aliases.add(canonicalize_team_name(abbreviation))
    return {alias for alias in aliases if alias}


def resolve_team_name(team_hint: str, games_df: pd.DataFrame, game_id: str) -> str | None:
    hint = canonicalize_team_name(team_hint)
    game = games_df[games_df["game_id"] == game_id]
    if game.empty:
        return None
    candidates = [game.iloc[0]["away_team"], game.iloc[0]["home_team"]]
    for team in candidates:
        if hint in team_aliases(team):
            return str(team)
    return None


def team_abbreviation_aliases(team_name: str | None) -> set[str]:
    aliases: set[str] = set()
    if not team_name:
        return aliases
    canonical = canonicalize_team_name(team_name)
    for abbreviation, full_name in TEAM_ABBREVIATIONS.items():
        if canonicalize_team_name(full_name) == canonical:
            aliases.add(abbreviation.upper())
    return aliases


def infer_game_id_from_kalshi_event(event_id: str, games_df: pd.DataFrame) -> str | None:
    tail = str(event_id).split("-")[-1].upper()
    match = re.search(r"([A-Z]+)$", tail)
    if not match:
        return None
    code = match.group(1)
    for game in games_df.to_dict(orient="records"):
        away_aliases = team_abbreviation_aliases(str(game["away_team"]))
        home_aliases = team_abbreviation_aliases(str(game["home_team"]))
        for away in away_aliases:
            for home in home_aliases:
                if away + home == code:
                    return str(game["game_id"])
    return None
