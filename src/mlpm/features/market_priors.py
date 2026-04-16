from __future__ import annotations

import pandas as pd


def build_market_prior_frame(games_df: pd.DataFrame, normalized_quotes_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty or normalized_quotes_df.empty:
        return pd.DataFrame(columns=["game_id", "market_home_implied_prob", "market_source_count"])

    valid_quotes = normalized_quotes_df.copy()
    if "is_valid" in valid_quotes.columns:
        valid_quotes = valid_quotes[valid_quotes["is_valid"]]
    if "is_pregame" in valid_quotes.columns:
        valid_quotes = valid_quotes[valid_quotes["is_pregame"]]
    if valid_quotes.empty:
        return pd.DataFrame(columns=["game_id", "market_home_implied_prob", "market_source_count"])

    consensus = (
        valid_quotes.groupby(["game_id", "outcome_team"], as_index=False)
        .agg(
            outcome_market_prob=("fair_prob", "mean"),
            market_source_count=("source", "nunique"),
        )
    )
    merged = consensus.merge(games_df[["game_id", "home_team"]], on="game_id", how="inner")
    home_only = merged[merged["outcome_team"] == merged["home_team"]].copy()
    if home_only.empty:
        return pd.DataFrame(columns=["game_id", "market_home_implied_prob", "market_source_count"])

    return home_only.rename(columns={"outcome_market_prob": "market_home_implied_prob"})[
        ["game_id", "market_home_implied_prob", "market_source_count"]
    ]
