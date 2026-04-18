from __future__ import annotations

import logging
from datetime import date
from math import exp

import pandas as pd

from mlpm.config.settings import settings
from mlpm.features.market_priors import build_market_prior_frame
from mlpm.features.matchups import build_team_offense_split_table
from mlpm.features.pitching import build_pitching_feature_table
from mlpm.features.team_strength import build_team_feature_table
from mlpm.ingest.mlb_stats import (
    fetch_final_results,
    fetch_game_batting_logs,
    fetch_game_pitching_logs,
    fetch_recent_bullpen_usage,
    fetch_recent_pitcher_form,
)
from mlpm.models.game_outcome import (
    build_live_feature_frame,
    load_trained_model,
    predict_home_win_probabilities,
)
from mlpm.features.utils import coalesce_float

logger = logging.getLogger(__name__)

# Heuristic live model weights are hand-tuned to keep recent form, venue split, and run differential
# in the same rough range as the underlying logit-transformed team-strength features.
RECENT_STRENGTH_WEIGHT = 0.60
VENUE_STRENGTH_WEIGHT = 0.35
RUN_DIFF_STRENGTH_WEIGHT = 0.30
STREAK_WEIGHT = 0.04

# Pitcher adjustments are scaled to treat WHIP as the strongest single live pitching signal, with
# ERA, strikeout rate, walk rate, and recent workload acting as smaller corrections.
PITCHER_ERA_BASELINE = 4.25
PITCHER_ERA_WEIGHT = 0.10
PITCHER_WHIP_BASELINE = 1.30
PITCHER_WHIP_WEIGHT = 0.60
PITCHER_K9_BASELINE = 8.0
PITCHER_K9_WEIGHT = 0.03
PITCHER_BB9_BASELINE = 3.0
PITCHER_BB9_WEIGHT = 0.05
PITCHER_INNINGS_BASELINE = 20.0
PITCHER_INNINGS_DIVISOR = 100.0
PITCHER_INNINGS_CAP = 0.15

# Bullpen fatigue is intentionally a smaller adjustment than the starter signal and is meant to
# penalize recent usage without dominating the rating.
BULLPEN_INNINGS_WEIGHT = 0.015
BULLPEN_PITCHES_DIVISOR = 5000.0
BULLPEN_RELIEVERS_WEIGHT = 0.02


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def build_team_strengths(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        logger.warning("Team strengths unavailable; games frame is empty.")
        return pd.DataFrame()

    season = pd.to_datetime(games_df["game_date"]).dt.year.mode().iloc[0]
    results = fetch_final_results(f"{season}-03-01", date.today().isoformat())
    if results.empty:
        logger.warning("Team strengths unavailable; no final results returned for season=%s", season)
        return pd.DataFrame()
    return build_team_feature_table(results)


def build_team_matchups(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        logger.warning("Team matchups unavailable; games frame is empty.")
        return pd.DataFrame()

    season = pd.to_datetime(games_df["game_date"]).dt.year.mode().iloc[0]
    batting_logs = fetch_game_batting_logs(f"{season}-03-01", date.today().isoformat())
    pitching_logs = fetch_game_pitching_logs(f"{season}-03-01", date.today().isoformat())
    if batting_logs.empty or pitching_logs.empty:
        logger.warning(
            "Team matchups unavailable; batting_rows=%s pitching_rows=%s season=%s",
            len(batting_logs),
            len(pitching_logs),
            season,
        )
        return pd.DataFrame()
    return build_team_offense_split_table(batting_logs, pitching_logs)


def build_model_probabilities(games_df: pd.DataFrame, market_priors_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if games_df.empty:
        logger.warning("Model probabilities unavailable; games frame is empty.")
        return pd.DataFrame()

    strengths = build_team_strengths(games_df)
    if strengths.empty:
        return pd.DataFrame()
    team_matchups = build_team_matchups(games_df)
    derived_market_priors = (
        market_priors_df
        if market_priors_df is not None
        else build_market_prior_frame(games_df, pd.DataFrame())
    )

    pitcher_ids = pd.concat(
        [games_df["away_probable_pitcher_id"], games_df["home_probable_pitcher_id"]],
        ignore_index=True,
    ).dropna()
    pitcher_stats = (
        fetch_recent_pitcher_form(pitcher_ids.tolist(), date.today().isoformat(), lookback_days=21)
        if not pitcher_ids.empty
        else pd.DataFrame()
    )
    bullpen_usage = fetch_recent_bullpen_usage(date.today().isoformat(), lookback_days=3)
    games_with_pitching = build_pitching_feature_table(games_df, pitcher_stats, bullpen_usage)

    trained_model = load_trained_model()
    if trained_model is not None:
        trained_predictions = _build_trained_model_probabilities(trained_model, strengths, games_with_pitching, team_matchups, derived_market_priors)
        if not trained_predictions.empty:
            return trained_predictions

    return _build_heuristic_model_probabilities(strengths, games_with_pitching, derived_market_priors)


def _build_trained_model_probabilities(
    trained_model: dict[str, object],
    strengths_df: pd.DataFrame,
    games_with_pitching_df: pd.DataFrame,
    team_matchups_df: pd.DataFrame | None = None,
    market_priors_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    features = build_live_feature_frame(
        games_with_pitching_df,
        strengths_df,
        games_with_pitching_df,
        team_matchups_df=team_matchups_df,
        market_priors_df=market_priors_df,
    )
    if features.empty:
        logger.warning(
            "Trained model probabilities unavailable; live feature frame is empty for games_rows=%s",
            len(games_with_pitching_df),
        )
        return pd.DataFrame()

    scored = predict_home_win_probabilities(trained_model, features)
    rows: list[dict[str, object]] = []
    for game in scored.to_dict(orient="records"):
        home_prob = float(game["home_win_prob"])
        away_prob = 1.0 - home_prob
        for team, probability, is_home_team in (
            (game["home_team"], home_prob, True),
            (game["away_team"], away_prob, False),
        ):
            rows.append(
                {
                    "game_id": game["game_id"],
                    "snapshot_ts": game["snapshot_ts"],
                    "team": team,
                    "model_name": str(game["model_name"]),
                    "model_prob": probability,
                    "games_played_floor_pass": True,
                    "opponent_team": game["away_team"] if is_home_team else game["home_team"],
                    "season_win_pct": game["home_season_win_pct"] if is_home_team else game["away_season_win_pct"],
                    "recent_win_pct": game["home_recent_win_pct"] if is_home_team else game["away_recent_win_pct"],
                    "venue_win_pct": game["home_venue_win_pct"] if is_home_team else game["away_venue_win_pct"],
                    "run_diff_per_game": game["home_run_diff_per_game"] if is_home_team else game["away_run_diff_per_game"],
                    "streak": game["home_streak"] if is_home_team else game["away_streak"],
                    "elo_rating": game["home_elo_rating"] if is_home_team else game["away_elo_rating"],
                    "season_runs_scored_per_game": game["home_season_runs_scored_per_game"] if is_home_team else game["away_season_runs_scored_per_game"],
                    "season_runs_allowed_per_game": game["home_season_runs_allowed_per_game"] if is_home_team else game["away_season_runs_allowed_per_game"],
                    "recent_runs_scored_per_game": game["home_recent_runs_scored_per_game"] if is_home_team else game["away_recent_runs_scored_per_game"],
                    "recent_runs_allowed_per_game": game["home_recent_runs_allowed_per_game"] if is_home_team else game["away_recent_runs_allowed_per_game"],
                    "rest_days": game["home_rest_days"] if is_home_team else game["away_rest_days"],
                    "venue_streak": game["home_venue_streak"] if is_home_team else game["away_venue_streak"],
                    "travel_switch": game["home_travel_switch"] if is_home_team else game["away_travel_switch"],
                    "is_doubleheader": game["is_doubleheader"],
                    "starter_era": game["home_starter_era"] if is_home_team else game["away_starter_era"],
                    "starter_whip": game["home_starter_whip"] if is_home_team else game["away_starter_whip"],
                    "starter_strikeouts_per_9": game["home_starter_strikeouts_per_9"]
                    if is_home_team
                    else game["away_starter_strikeouts_per_9"],
                    "starter_walks_per_9": game["home_starter_walks_per_9"]
                    if is_home_team
                    else game["away_starter_walks_per_9"],
                    "bullpen_innings_3d": game["home_bullpen_innings_3d"] if is_home_team else game["away_bullpen_innings_3d"],
                    "bullpen_pitches_3d": game["home_bullpen_pitches_3d"] if is_home_team else game["away_bullpen_pitches_3d"],
                    "relievers_used_3d": game["home_relievers_used_3d"] if is_home_team else game["away_relievers_used_3d"],
                    "market_home_implied_prob": game.get("market_home_implied_prob"),
                    "offense_vs_starter_hand": game["home_offense_vs_opp_starter_hand"]
                    if is_home_team
                    else game["away_offense_vs_opp_starter_hand"],
                }
            )
    return pd.DataFrame(rows)


def _build_heuristic_model_probabilities(
    strengths_df: pd.DataFrame,
    games_df: pd.DataFrame,
    market_priors_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    lookup = strengths_df.set_index("team").to_dict(orient="index")
    market_prior_lookup = (
        market_priors_df.drop_duplicates(subset=["game_id"]).set_index("game_id")["market_home_implied_prob"].to_dict()
        if market_priors_df is not None and not market_priors_df.empty
        else {}
    )
    home_edge = settings().model_home_field_edge_bps / 10_000
    min_games = settings().model_min_games
    rows: list[dict[str, object]] = []
    for game in games_df.to_dict(orient="records"):
        home = lookup.get(game["home_team"])
        away = lookup.get(game["away_team"])
        if not home or not away:
            continue
        games_floor_pass = min(home["games_played"], away["games_played"]) >= min_games
        if not games_floor_pass:
            continue

        home_pitcher_edge = _pitcher_rating(game, "home")
        away_pitcher_edge = _pitcher_rating(game, "away")
        home_bullpen_penalty = _bullpen_penalty(game, "home")
        away_bullpen_penalty = _bullpen_penalty(game, "away")

        home_rating = (
            home["season_strength"]
            + (RECENT_STRENGTH_WEIGHT * home["recent_strength"])
            + (VENUE_STRENGTH_WEIGHT * home["home_strength"])
            + (RUN_DIFF_STRENGTH_WEIGHT * home["run_diff_strength"])
            + (STREAK_WEIGHT * home["streak"])
            + home_pitcher_edge
            - home_bullpen_penalty
        )
        away_rating = (
            away["season_strength"]
            + (RECENT_STRENGTH_WEIGHT * away["recent_strength"])
            + (VENUE_STRENGTH_WEIGHT * away["away_strength"])
            + (RUN_DIFF_STRENGTH_WEIGHT * away["run_diff_strength"])
            + (STREAK_WEIGHT * away["streak"])
            + away_pitcher_edge
            - away_bullpen_penalty
        )

        home_win_prob = _sigmoid(home_rating - away_rating + home_edge)
        away_win_prob = 1.0 - home_win_prob

        for team, probability, venue_split, opponent in (
            (game["home_team"], home_win_prob, home["home_win_pct"], game["away_team"]),
            (game["away_team"], away_win_prob, away["away_win_pct"], game["home_team"]),
        ):
            is_home_team = team == game["home_team"]
            rows.append(
                {
                    "game_id": game["game_id"],
                    "snapshot_ts": game["snapshot_ts"],
                    "team": team,
                    "model_name": "team_form_logit_v2",
                    "model_prob": probability,
                    "games_played_floor_pass": games_floor_pass,
                    "opponent_team": opponent,
                    "season_win_pct": home["season_win_pct"] if is_home_team else away["season_win_pct"],
                    "recent_win_pct": home["recent_win_pct"] if is_home_team else away["recent_win_pct"],
                    "venue_win_pct": venue_split,
                    "run_diff_per_game": home["run_diff_per_game"] if is_home_team else away["run_diff_per_game"],
                    "streak": home["streak"] if is_home_team else away["streak"],
                    "elo_rating": home.get("elo_rating") if is_home_team else away.get("elo_rating"),
                    "season_runs_scored_per_game": home.get("season_runs_scored_per_game") if is_home_team else away.get("season_runs_scored_per_game"),
                    "season_runs_allowed_per_game": home.get("season_runs_allowed_per_game") if is_home_team else away.get("season_runs_allowed_per_game"),
                    "recent_runs_scored_per_game": home.get("recent_runs_scored_per_game") if is_home_team else away.get("recent_runs_scored_per_game"),
                    "recent_runs_allowed_per_game": home.get("recent_runs_allowed_per_game") if is_home_team else away.get("recent_runs_allowed_per_game"),
                    "rest_days": None,
                    "venue_streak": None,
                    "travel_switch": None,
                    "is_doubleheader": None,
                    "starter_era": game.get("home_pitcher_era") if is_home_team else game.get("away_pitcher_era"),
                    "starter_whip": game.get("home_pitcher_whip") if is_home_team else game.get("away_pitcher_whip"),
                    "starter_strikeouts_per_9": game.get("home_pitcher_strikeouts_per_9")
                    if is_home_team
                    else game.get("away_pitcher_strikeouts_per_9"),
                    "starter_walks_per_9": game.get("home_pitcher_walks_per_9")
                    if is_home_team
                    else game.get("away_pitcher_walks_per_9"),
                    "bullpen_innings_3d": game.get("home_bullpen_innings_3d") if is_home_team else game.get("away_bullpen_innings_3d"),
                    "bullpen_pitches_3d": game.get("home_bullpen_pitches_3d") if is_home_team else game.get("away_bullpen_pitches_3d"),
                    "relievers_used_3d": game.get("home_relievers_used_3d") if is_home_team else game.get("away_relievers_used_3d"),
                    "market_home_implied_prob": market_prior_lookup.get(game["game_id"]),
                    "offense_vs_starter_hand": None,
                }
            )
    return pd.DataFrame(rows)


def _pitcher_rating(game: dict[str, object], side: str) -> float:
    era = coalesce_float(game.get(f"{side}_pitcher_era"))
    whip = coalesce_float(game.get(f"{side}_pitcher_whip"))
    strikeouts_per_9 = coalesce_float(game.get(f"{side}_pitcher_strikeouts_per_9"))
    walks_per_9 = coalesce_float(game.get(f"{side}_pitcher_walks_per_9"))
    innings_pitched = coalesce_float(game.get(f"{side}_pitcher_innings_pitched"))

    rating = 0.0
    if era is not None:
        rating += (PITCHER_ERA_BASELINE - era) * PITCHER_ERA_WEIGHT
    if whip is not None:
        rating += (PITCHER_WHIP_BASELINE - whip) * PITCHER_WHIP_WEIGHT
    if strikeouts_per_9 is not None:
        rating += (strikeouts_per_9 - PITCHER_K9_BASELINE) * PITCHER_K9_WEIGHT
    if walks_per_9 is not None:
        rating += (PITCHER_BB9_BASELINE - walks_per_9) * PITCHER_BB9_WEIGHT
    if innings_pitched is not None:
        rating += min(
            max((innings_pitched - PITCHER_INNINGS_BASELINE) / PITCHER_INNINGS_DIVISOR, -PITCHER_INNINGS_CAP),
            PITCHER_INNINGS_CAP,
        )
    return rating


def _bullpen_penalty(game: dict[str, object], side: str) -> float:
    innings_3d = coalesce_float(game.get(f"{side}_bullpen_innings_3d")) or 0.0
    pitches_3d = coalesce_float(game.get(f"{side}_bullpen_pitches_3d")) or 0.0
    relievers_1d = coalesce_float(game.get(f"{side}_relievers_used_1d")) or 0.0
    return (
        (innings_3d * BULLPEN_INNINGS_WEIGHT)
        + (pitches_3d / BULLPEN_PITCHES_DIVISOR)
        + (relievers_1d * BULLPEN_RELIEVERS_WEIGHT)
    )
