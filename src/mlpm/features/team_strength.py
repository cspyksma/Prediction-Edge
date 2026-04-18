from __future__ import annotations

from math import log

import pandas as pd

from mlpm.features.utils import current_streak, smoothed_win_pct

RECENT_GAMES_WINDOW = 10
ELO_BASELINE = 1500.0
ELO_K_FACTOR = 20.0


def _safe_logit(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1 - 1e-6)
    return log(probability / (1 - probability))


def _bounded_run_diff(run_diff_per_game: float) -> float:
    return max(min(run_diff_per_game, 3.0), -3.0) / 3.0


def build_team_feature_table(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    results = results_df.copy()
    results["game_date"] = pd.to_datetime(results["game_date"])

    elo_ratings: dict[str, float] = {}
    team_games: list[dict[str, object]] = []
    for row in results.sort_values(["game_date", "game_id"]).itertuples(index=False):
        away_win = row.away_score > row.home_score
        home_win = row.home_score > row.away_score
        away_elo_pre = elo_ratings.get(row.away_team, ELO_BASELINE)
        home_elo_pre = elo_ratings.get(row.home_team, ELO_BASELINE)
        expected_away = 1.0 / (1.0 + 10 ** ((home_elo_pre - away_elo_pre) / 400.0))
        away_score_actual = 1.0 if away_win else 0.0
        home_score_actual = 1.0 if home_win else 0.0
        elo_ratings[row.away_team] = away_elo_pre + ELO_K_FACTOR * (away_score_actual - expected_away)
        elo_ratings[row.home_team] = home_elo_pre + ELO_K_FACTOR * (home_score_actual - (1.0 - expected_away))
        team_games.append(
            {
                "team": row.away_team,
                "game_id": row.game_id,
                "game_date": row.game_date,
                "is_home": False,
                "won": away_win,
                "runs_for": row.away_score,
                "runs_against": row.home_score,
                "run_diff": row.away_score - row.home_score,
                "elo_pre": away_elo_pre,
            }
        )
        team_games.append(
            {
                "team": row.home_team,
                "game_id": row.game_id,
                "game_date": row.game_date,
                "is_home": True,
                "won": home_win,
                "runs_for": row.home_score,
                "runs_against": row.away_score,
                "run_diff": row.home_score - row.away_score,
                "elo_pre": home_elo_pre,
            }
        )

    team_games_df = pd.DataFrame(team_games).sort_values(["team", "game_date", "game_id"])
    rows: list[dict[str, object]] = []
    for team, group in team_games_df.groupby("team", sort=True):
        group = group.reset_index(drop=True)
        games_played = len(group)
        wins = int(group["won"].sum())
        losses = games_played - wins
        runs_for_total = int(group["runs_for"].sum())
        runs_against_total = int(group["runs_against"].sum())
        run_diff_total = int(group["run_diff"].sum())
        run_diff_per_game = run_diff_total / games_played if games_played else 0.0

        home_group = group[group["is_home"]]
        away_group = group[~group["is_home"]]
        home_games = len(home_group)
        away_games = len(away_group)
        home_wins = int(home_group["won"].sum())
        away_wins = int(away_group["won"].sum())

        recent = group.tail(RECENT_GAMES_WINDOW)
        recent_games = len(recent)
        recent_wins = int(recent["won"].sum())
        recent_runs_for = int(recent["runs_for"].sum())
        recent_runs_against = int(recent["runs_against"].sum())
        streak = current_streak(group["won"].tolist())
        last_game_date = group["game_date"].iloc[-1]
        last_is_home = bool(group["is_home"].iloc[-1])
        venue_streak = 0
        for is_home in reversed(group["is_home"].tolist()):
            if bool(is_home) != last_is_home:
                break
            venue_streak += 1

        season_win_pct = smoothed_win_pct(wins, games_played)
        recent_win_pct = smoothed_win_pct(recent_wins, recent_games)
        home_win_pct = smoothed_win_pct(home_wins, home_games)
        away_win_pct = smoothed_win_pct(away_wins, away_games)

        rows.append(
            {
                "team": team,
                "wins": wins,
                "losses": losses,
                "games_played": games_played,
                "season_win_pct": season_win_pct,
                "recent_games": recent_games,
                "recent_win_pct": recent_win_pct,
                "home_games": home_games,
                "home_win_pct": home_win_pct,
                "away_games": away_games,
                "away_win_pct": away_win_pct,
                "run_diff_total": run_diff_total,
                "run_diff_per_game": run_diff_per_game,
                "season_runs_scored_per_game": (runs_for_total / games_played) if games_played else 0.0,
                "season_runs_allowed_per_game": (runs_against_total / games_played) if games_played else 0.0,
                "recent_runs_scored_per_game": (recent_runs_for / recent_games) if recent_games else 0.0,
                "recent_runs_allowed_per_game": (recent_runs_against / recent_games) if recent_games else 0.0,
                "streak": streak,
                "elo_rating": float(elo_ratings.get(team, ELO_BASELINE)),
                "last_game_date": last_game_date,
                "last_is_home": last_is_home,
                "current_venue_streak": venue_streak,
                "season_strength": _safe_logit(season_win_pct),
                "recent_strength": _safe_logit(recent_win_pct),
                "home_strength": _safe_logit(home_win_pct),
                "away_strength": _safe_logit(away_win_pct),
                "run_diff_strength": _bounded_run_diff(run_diff_per_game),
            }
        )

    return pd.DataFrame(rows)
