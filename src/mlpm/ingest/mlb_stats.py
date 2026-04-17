from __future__ import annotations

from datetime import date, timedelta

import httpx
import pandas as pd

from mlpm.ingest.base import utcnow

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people/{player_id}"


def _scheduled_first_pitch_utc(game: dict[str, object]) -> str | None:
    """Return the scheduled MLB first pitch as a normalized UTC timestamp."""
    for key in ("gameDate", "rescheduleDate", "resumeDate"):
        value = game.get(key)
        if value in (None, ""):
            continue
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            continue
        return parsed.isoformat().replace("+00:00", "Z")
    return None


def fetch_upcoming_games(lookahead_days: int = 3) -> pd.DataFrame:
    start = date.today()
    end = start + timedelta(days=lookahead_days)
    params = {
        "sportId": 1,
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "hydrate": "probablePitcher",
    }

    with httpx.Client(timeout=20.0) as client:
        response = client.get(MLB_SCHEDULE_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    rows: list[dict[str, object]] = []
    for schedule_date in payload.get("dates", []):
        for game in schedule_date.get("games", []):
            teams = game.get("teams", {})
            away = teams.get("away", {}).get("team", {})
            home = teams.get("home", {}).get("team", {})
            away_probable = teams.get("away", {}).get("probablePitcher", {})
            home_probable = teams.get("home", {}).get("probablePitcher", {})
            rows.append(
                {
                    "game_id": str(game["gamePk"]),
                    "game_date": schedule_date["date"],
                    "event_start_time": _scheduled_first_pitch_utc(game),
                    "away_team": away.get("name"),
                    "home_team": home.get("name"),
                    "away_team_id": away.get("id"),
                    "home_team_id": home.get("id"),
                    "away_probable_pitcher_id": away_probable.get("id"),
                    "away_probable_pitcher_name": away_probable.get("fullName"),
                    "away_probable_pitcher_hand": None,
                    "home_probable_pitcher_id": home_probable.get("id"),
                    "home_probable_pitcher_name": home_probable.get("fullName"),
                    "home_probable_pitcher_hand": None,
                    "doubleheader": game.get("doubleHeader"),
                    "game_number": game.get("gameNumber"),
                    "day_night": game.get("dayNight"),
                    "status": game.get("status", {}).get("detailedState"),
                    "snapshot_ts": utcnow().isoformat(),
                }
            )
    games_df = pd.DataFrame(rows)
    if games_df.empty:
        return games_df

    probable_pitcher_ids = pd.concat(
        [games_df["away_probable_pitcher_id"], games_df["home_probable_pitcher_id"]],
        ignore_index=True,
    ).dropna()
    if probable_pitcher_ids.empty:
        return games_df

    pitcher_hands = fetch_pitcher_handedness(probable_pitcher_ids.tolist())
    if pitcher_hands.empty:
        return games_df
    hand_lookup = pitcher_hands.set_index("pitcher_id")["pitcher_hand"].to_dict()
    games_df["away_probable_pitcher_hand"] = games_df["away_probable_pitcher_id"].map(hand_lookup)
    games_df["home_probable_pitcher_hand"] = games_df["home_probable_pitcher_id"].map(hand_lookup)
    return games_df


def fetch_final_results(start_date: str, end_date: str) -> pd.DataFrame:
    params = {"sportId": 1, "startDate": start_date, "endDate": end_date}
    with httpx.Client(timeout=20.0) as client:
        response = client.get(MLB_SCHEDULE_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    rows: list[dict[str, object]] = []
    for schedule_date in payload.get("dates", []):
        for game in schedule_date.get("games", []):
            teams = game.get("teams", {})
            away = teams.get("away", {})
            home = teams.get("home", {})
            if away.get("score") is None or home.get("score") is None:
                continue
            away_team = away.get("team", {}).get("name")
            home_team = home.get("team", {}).get("name")
            if not away_team or not home_team:
                continue
            winner = away_team if away["score"] > home["score"] else home_team
            rows.append(
                {
                    "game_id": str(game["gamePk"]),
                    "game_date": schedule_date["date"],
                    "event_start_time": _scheduled_first_pitch_utc(game),
                    "away_team": away_team,
                    "home_team": home_team,
                    "winner_team": winner,
                    "away_score": away["score"],
                    "home_score": home["score"],
                }
            )
    return pd.DataFrame(rows)


def fetch_game_pitching_logs(start_date: str, end_date: str) -> pd.DataFrame:
    results = fetch_final_results(start_date, end_date)
    if results.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    with httpx.Client(timeout=20.0) as client:
        for game in results.to_dict(orient="records"):
            response = client.get(MLB_BOXSCORE_URL.format(game_id=game["game_id"]))
            response.raise_for_status()
            payload = response.json()
            for side in ("away", "home"):
                team_box = payload.get("teams", {}).get(side, {})
                team_name = team_box.get("team", {}).get("name")
                players = team_box.get("players", {}) or {}
                pitchers = team_box.get("pitchers", []) or []

                starter_id: int | None = None
                starter_name: str | None = None
                starter_innings = 0.0
                starter_earned_runs = 0
                starter_hits = 0
                starter_walks = 0
                starter_strikeouts = 0

                bullpen_innings = 0.0
                bullpen_pitches = 0
                relievers_used = 0

                for pitcher_id in pitchers:
                    player = players.get(f"ID{pitcher_id}", {})
                    pitching = player.get("stats", {}).get("pitching", {})
                    if not pitching:
                        continue

                    games_started = int(pitching.get("gamesStarted", 0) or 0)
                    innings_pitched = _innings_to_float(pitching.get("inningsPitched"))
                    earned_runs = _to_int(pitching.get("earnedRuns"))
                    hits = _to_int(pitching.get("hits"))
                    walks = _to_int(pitching.get("baseOnBalls"))
                    strikeouts = _to_int(pitching.get("strikeOuts"))
                    pitches_thrown = _to_int(pitching.get("pitchesThrown") or pitching.get("numberOfPitches"))

                    if games_started > 0 and starter_id is None:
                        starter_id = pitcher_id
                        starter_name = player.get("person", {}).get("fullName")
                        starter_innings = innings_pitched
                        starter_earned_runs = earned_runs
                        starter_hits = hits
                        starter_walks = walks
                        starter_strikeouts = strikeouts
                        continue

                    relievers_used += 1
                    bullpen_innings += innings_pitched
                    bullpen_pitches += pitches_thrown

                rows.append(
                    {
                        "game_id": game["game_id"],
                        "game_date": game["game_date"],
                        "team": team_name,
                        "side": side,
                        "starting_pitcher_id": starter_id,
                        "starting_pitcher_name": starter_name,
                        "starting_pitcher_hand": None,
                        "starter_innings_pitched": starter_innings,
                        "starter_earned_runs": starter_earned_runs,
                        "starter_hits": starter_hits,
                        "starter_walks": starter_walks,
                        "starter_strikeouts": starter_strikeouts,
                        "bullpen_innings": bullpen_innings,
                        "bullpen_pitches": bullpen_pitches,
                        "relievers_used": relievers_used,
                    }
                )
    pitching_logs = pd.DataFrame(rows)
    if pitching_logs.empty:
        return pitching_logs

    starter_ids = pitching_logs["starting_pitcher_id"].dropna()
    if starter_ids.empty:
        return pitching_logs

    pitcher_hands = fetch_pitcher_handedness(starter_ids.tolist())
    if pitcher_hands.empty:
        return pitching_logs
    hand_lookup = pitcher_hands.set_index("pitcher_id")["pitcher_hand"].to_dict()
    pitching_logs["starting_pitcher_hand"] = pitching_logs["starting_pitcher_id"].map(hand_lookup)
    return pitching_logs


def fetch_game_batting_logs(start_date: str, end_date: str) -> pd.DataFrame:
    results = fetch_final_results(start_date, end_date)
    if results.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    with httpx.Client(timeout=20.0) as client:
        for game in results.to_dict(orient="records"):
            response = client.get(MLB_BOXSCORE_URL.format(game_id=game["game_id"]))
            response.raise_for_status()
            payload = response.json()
            for side in ("away", "home"):
                team_box = payload.get("teams", {}).get(side, {})
                team_name = team_box.get("team", {}).get("name")
                batting = team_box.get("teamStats", {}).get("batting", {}) or {}
                if not team_name or not batting:
                    continue
                is_home = side == "home"
                rows.append(
                    {
                        "game_id": game["game_id"],
                        "game_date": game["game_date"],
                        "team": team_name,
                        "opponent_team": game["away_team"] if is_home else game["home_team"],
                        "at_bats": _to_int(batting.get("atBats")),
                        "hits": _to_int(batting.get("hits")),
                        "walks": _to_int(batting.get("baseOnBalls")),
                        "strikeouts": _to_int(batting.get("strikeOuts")),
                        "doubles": _to_int(batting.get("doubles")),
                        "triples": _to_int(batting.get("triples")),
                        "home_runs": _to_int(batting.get("homeRuns")),
                        "runs_scored": int(game["home_score"] if is_home else game["away_score"]),
                    }
                )
    return pd.DataFrame(rows)


def fetch_pitcher_handedness(player_ids: list[int | str]) -> pd.DataFrame:
    unique_ids = sorted({int(player_id) for player_id in player_ids if player_id and not pd.isna(player_id)})
    if not unique_ids:
        return pd.DataFrame(columns=["pitcher_id", "pitcher_hand"])

    rows: list[dict[str, object]] = []
    with httpx.Client(timeout=20.0) as client:
        for pitcher_id in unique_ids:
            response = client.get(MLB_PEOPLE_URL.format(player_id=pitcher_id))
            response.raise_for_status()
            payload = response.json()
            people = payload.get("people", [])
            if not people:
                continue
            person = people[0]
            pitch_hand = (person.get("pitchHand") or {}).get("code")
            rows.append({"pitcher_id": pitcher_id, "pitcher_hand": pitch_hand})
    return pd.DataFrame(rows)


def fetch_recent_pitcher_form(player_ids: list[int | str], end_date: str, lookback_days: int = 21) -> pd.DataFrame:
    unique_ids = {int(player_id) for player_id in player_ids if player_id}
    if not unique_ids:
        return pd.DataFrame()

    end_dt = date.fromisoformat(end_date)
    start_dt = end_dt - timedelta(days=lookback_days)
    results = fetch_final_results(start_dt.isoformat(), end_dt.isoformat())
    if results.empty:
        return pd.DataFrame()

    appearance_rows: list[dict[str, object]] = []
    with httpx.Client(timeout=20.0) as client:
        for game in results.to_dict(orient="records"):
            response = client.get(MLB_BOXSCORE_URL.format(game_id=game["game_id"]))
            response.raise_for_status()
            payload = response.json()
            for side in ("away", "home"):
                team_box = payload.get("teams", {}).get(side, {})
                players = team_box.get("players", {}) or {}
                for pitcher_id in team_box.get("pitchers", []) or []:
                    if pitcher_id not in unique_ids:
                        continue
                    player = players.get(f"ID{pitcher_id}", {})
                    pitching = player.get("stats", {}).get("pitching", {})
                    if not pitching:
                        continue
                    appearance_rows.append(
                        {
                            "pitcher_id": pitcher_id,
                            "pitcher_name": player.get("person", {}).get("fullName"),
                            "game_id": game["game_id"],
                            "game_date": game["game_date"],
                            "games_started": _to_int(pitching.get("gamesStarted")),
                            "innings_pitched": _innings_to_float(pitching.get("inningsPitched")),
                            "earned_runs": _to_int(pitching.get("earnedRuns")),
                            "hits": _to_int(pitching.get("hits")),
                            "walks": _to_int(pitching.get("baseOnBalls")),
                            "strikeouts": _to_int(pitching.get("strikeOuts")),
                        }
                    )

    appearances = pd.DataFrame(appearance_rows)
    if appearances.empty:
        return pd.DataFrame()

    starts = appearances[appearances["games_started"] > 0].copy()
    if starts.empty:
        return pd.DataFrame()
    starts["game_date"] = pd.to_datetime(starts["game_date"])
    starts = starts.sort_values(["pitcher_id", "game_date"]).groupby("pitcher_id").tail(5)

    rows: list[dict[str, object]] = []
    for pitcher_id, group in starts.groupby("pitcher_id"):
        innings = float(group["innings_pitched"].sum())
        hits = int(group["hits"].sum())
        walks = int(group["walks"].sum())
        strikeouts = int(group["strikeouts"].sum())
        earned_runs = int(group["earned_runs"].sum())
        rows.append(
            {
                "pitcher_id": int(pitcher_id),
                "pitcher_name": group["pitcher_name"].iloc[-1],
                "pitcher_hand": None,
                "era": (earned_runs * 9.0 / innings) if innings > 0 else None,
                "whip": ((hits + walks) / innings) if innings > 0 else None,
                "strikeouts_per_9": (strikeouts * 9.0 / innings) if innings > 0 else None,
                "walks_per_9": (walks * 9.0 / innings) if innings > 0 else None,
                "hits_per_9": (hits * 9.0 / innings) if innings > 0 else None,
                "innings_pitched": innings,
                "games_started": int(group["games_started"].sum()),
                "wins": 0,
                "losses": 0,
            }
        )
    pitcher_form = pd.DataFrame(rows)
    if pitcher_form.empty:
        return pitcher_form

    handedness = fetch_pitcher_handedness(pitcher_form["pitcher_id"].tolist())
    if handedness.empty:
        return pitcher_form
    return pitcher_form.merge(handedness, on="pitcher_id", how="left")


def fetch_recent_bullpen_usage(end_date: str, lookback_days: int = 3) -> pd.DataFrame:
    end_dt = date.fromisoformat(end_date)
    start_dt = end_dt - timedelta(days=lookback_days)
    results = fetch_final_results(start_dt.isoformat(), end_dt.isoformat())
    if results.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    with httpx.Client(timeout=20.0) as client:
        for game in results.to_dict(orient="records"):
            response = client.get(MLB_BOXSCORE_URL.format(game_id=game["game_id"]))
            response.raise_for_status()
            payload = response.json()
            for side in ("away", "home"):
                team_box = payload.get("teams", {}).get(side, {})
                team_name = team_box.get("team", {}).get("name")
                pitchers = team_box.get("pitchers", []) or []
                players = team_box.get("players", {}) or {}
                bullpen_innings = 0.0
                bullpen_pitches = 0
                relievers_used = 0
                for pitcher_id in pitchers:
                    player = players.get(f"ID{pitcher_id}", {})
                    pitching = player.get("stats", {}).get("pitching", {})
                    if not pitching:
                        continue
                    if int(pitching.get("gamesStarted", 0) or 0) > 0:
                        continue
                    relievers_used += 1
                    bullpen_innings += _innings_to_float(pitching.get("inningsPitched"))
                    bullpen_pitches += _to_int(pitching.get("pitchesThrown") or pitching.get("numberOfPitches"))
                rows.append(
                    {
                        "game_id": game["game_id"],
                        "game_date": game["game_date"],
                        "team": team_name,
                        "bullpen_innings": bullpen_innings,
                        "bullpen_pitches": bullpen_pitches,
                        "relievers_used": relievers_used,
                    }
                )
    usage = pd.DataFrame(rows)
    if usage.empty:
        return usage
    usage["game_date"] = pd.to_datetime(usage["game_date"])
    yesterday = pd.Timestamp(end_date) - pd.Timedelta(days=1)
    agg = (
        usage.groupby("team", as_index=False)
        .agg(
            bullpen_innings_3d=("bullpen_innings", "sum"),
            bullpen_pitches_3d=("bullpen_pitches", "sum"),
            relievers_used_3d=("relievers_used", "sum"),
            games_counted=("game_id", "nunique"),
        )
    )
    yesterday_usage = (
        usage[usage["game_date"] == yesterday]
        .groupby("team", as_index=False)
        .agg(
            bullpen_innings_1d=("bullpen_innings", "sum"),
            bullpen_pitches_1d=("bullpen_pitches", "sum"),
            relievers_used_1d=("relievers_used", "sum"),
        )
    )
    return agg.merge(yesterday_usage, on="team", how="left").fillna(0)


def _to_float(value: object) -> float | None:
    if value in (None, "", "-", "--", "-.--", ".---"):
        return None
    return float(value)


def _to_int(value: object) -> int:
    if value in (None, "", "-", "--"):
        return 0
    return int(value)


def _innings_to_float(value: object) -> float:
    if value in (None, "", "-", "--"):
        return 0.0
    text = str(value)
    if "." not in text:
        return float(text)
    whole, outs = text.split(".", maxsplit=1)
    return int(whole) + (int(outs) / 3.0)
