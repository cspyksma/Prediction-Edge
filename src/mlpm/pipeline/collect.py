from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd

from mlpm.config.settings import settings
from mlpm.evaluation.strategy import (
    build_bet_opportunities,
    record_champion_selection,
    select_champion_with_rationale,
)
from mlpm.features.market_priors import build_market_prior_frame
from mlpm.ingest.base import write_raw_payload
from mlpm.ingest.kalshi import fetch_mlb_markets
from mlpm.ingest.mlb_stats import fetch_upcoming_games
from mlpm.ingest.polymarket import fetch_mlb_markets as fetch_polymarket_markets
from mlpm.models.fair_value import build_model_probabilities
from mlpm.normalize.mapping import map_kalshi_to_games, map_market_text_to_games
from mlpm.normalize.probability import gap_to_bps, midpoint_probability
from mlpm.storage.duckdb import append_dataframe, connect

logger = logging.getLogger(__name__)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _empty_normalized_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "game_id",
            "source",
            "event_id",
            "market_id",
            "bookmaker",
            "event_start_time",
            "snapshot_ts",
            "collection_run_ts",
            "outcome_team",
            "market_type",
            "raw_odds",
            "raw_price",
            "implied_prob",
            "fair_prob",
            "quote_age_sec",
            "is_pregame",
            "is_valid",
        ]
    )


def _normalize_market_quotes(mapped_quotes: pd.DataFrame, source: str, captured_at: datetime) -> pd.DataFrame:
    if mapped_quotes.empty:
        return _empty_normalized_quotes()

    rows: list[dict[str, object]] = []
    for quote in mapped_quotes.to_dict(orient="records"):
        event_start = _parse_ts(quote.get("event_start_time"))
        quote_updated_ts = _parse_ts(quote.get("snapshot_ts"))
        if event_start is None:
            continue
        if source == "kalshi":
            fair_prob = midpoint_probability(quote.get("yes_bid"), quote.get("yes_ask"), quote.get("last_price"))
            raw_price = quote.get("last_price")
            market_id = quote["market_ticker"]
            bookmaker = None
        else:
            fair_prob = float(quote.get("last_price"))
            raw_price = quote.get("last_price")
            market_id = quote["market_id"]
            bookmaker = None
        if fair_prob is None:
            continue
        rows.append(
            {
                "game_id": quote["game_id"],
                "source": source,
                "event_id": quote["event_id"],
                "market_id": market_id,
                "bookmaker": bookmaker,
                "event_start_time": event_start,
                "snapshot_ts": captured_at,
                "collection_run_ts": captured_at,
                "outcome_team": quote["outcome_team"],
                "market_type": "moneyline",
                "raw_odds": None,
                "raw_price": raw_price,
                "implied_prob": fair_prob,
                "fair_prob": fair_prob,
                "quote_age_sec": (captured_at - quote_updated_ts).total_seconds() if quote_updated_ts else None,
                "is_pregame": captured_at < event_start,
                "is_valid": True,
            }
        )
    return pd.DataFrame(rows)


def build_discrepancies(normalized_quotes: pd.DataFrame, model_predictions: pd.DataFrame) -> pd.DataFrame:
    if normalized_quotes.empty or model_predictions.empty:
        return pd.DataFrame()

    markets = normalized_quotes.rename(columns={"fair_prob": "market_prob"})[
        [
            "game_id",
            "snapshot_ts",
            "collection_run_ts",
            "source",
            "market_id",
            "outcome_team",
            "market_prob",
            "quote_age_sec",
            "is_valid",
        ]
    ]
    merged = markets.merge(
        model_predictions.rename(
            columns={
                "team": "outcome_team",
                "snapshot_ts": "model_snapshot_ts",
                "collection_run_ts": "model_collection_run_ts",
            }
        ),
        on=["game_id", "outcome_team"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["freshness_pass"] = merged["quote_age_sec"] <= settings().freshness_window_seconds
    merged["mapping_pass"] = merged["is_valid"] & merged["games_played_floor_pass"]
    merged["gap_bps"] = merged.apply(
        lambda row: gap_to_bps(row["market_prob"], row["model_prob"]),
        axis=1,
    )
    merged["flagged"] = (
        merged["freshness_pass"]
        & merged["mapping_pass"]
        & merged["is_valid"]
        & merged["gap_bps"].abs().ge(settings().discrepancy_threshold_bps)
    )
    merged["snapshot_ts"] = merged["snapshot_ts"]
    return merged.rename(columns={"outcome_team": "team"})[
        [
            "game_id",
            "snapshot_ts",
            "collection_run_ts",
            "source",
            "market_id",
            "team",
            "market_prob",
            "model_prob",
            "gap_bps",
            "freshness_pass",
            "mapping_pass",
            "flagged",
        ]
    ]


def collect_snapshot() -> dict[str, int]:
    cfg = settings()
    captured_at = datetime.now(tz=UTC)
    collection_run_ts = captured_at
    logger.info("collect_snapshot starting captured_at=%s lookahead_days=%s", captured_at.isoformat(), cfg.mlb_lookahead_days)

    logger.info("collect_snapshot fetching upcoming MLB games")
    games_df = fetch_upcoming_games(cfg.mlb_lookahead_days)
    logger.info("collect_snapshot fetched upcoming games count=%s", len(games_df))
    if games_df.empty:
        logger.warning("Snapshot collection found no upcoming MLB games.")

    kalshi_df = pd.DataFrame()
    kalshi_payload: object = []
    try:
        logger.info("collect_snapshot fetching Kalshi markets")
        kalshi_df, kalshi_payload = fetch_mlb_markets()
        mapped_kalshi = map_kalshi_to_games(kalshi_df, games_df)
        logger.info("collect_snapshot mapped Kalshi rows raw=%s mapped=%s", len(kalshi_df), len(mapped_kalshi))
    except Exception:
        logger.warning("Kalshi market collection failed; continuing without Kalshi quotes.", exc_info=True)
        mapped_kalshi = pd.DataFrame()

    polymarket_df = pd.DataFrame()
    polymarket_payload: object = []
    try:
        logger.info("collect_snapshot fetching Polymarket markets")
        polymarket_df, polymarket_payload = fetch_polymarket_markets()
        mapped_polymarket = map_market_text_to_games(polymarket_df, games_df, "question")
        logger.info("collect_snapshot mapped Polymarket rows raw=%s mapped=%s", len(polymarket_df), len(mapped_polymarket))
    except Exception:
        logger.warning("Polymarket collection failed; continuing without Polymarket quotes.", exc_info=True)
        mapped_polymarket = pd.DataFrame()

    logger.info("collect_snapshot normalizing market quotes")
    normalized_quotes = pd.concat(
        [
            _normalize_market_quotes(mapped_kalshi, "kalshi", captured_at),
            _normalize_market_quotes(mapped_polymarket, "polymarket", captured_at),
        ],
        ignore_index=True,
    )
    logger.info("collect_snapshot normalized quotes count=%s", len(normalized_quotes))
    logger.info("collect_snapshot building market priors and model probabilities")
    market_priors = build_market_prior_frame(games_df, normalized_quotes)
    model_predictions = build_model_probabilities(games_df, market_priors_df=market_priors)
    logger.info("collect_snapshot built model predictions count=%s", len(model_predictions))
    if not games_df.empty:
        games_df["collection_run_ts"] = collection_run_ts
    if not model_predictions.empty:
        model_predictions["collection_run_ts"] = collection_run_ts
    logger.info("collect_snapshot selecting champion model")
    champion_decision = select_champion_with_rationale()
    champion_model = champion_decision.chosen_model
    logger.info("collect_snapshot selected champion model=%s", champion_model)
    logger.info("collect_snapshot building bet opportunities")
    bet_opportunities = build_bet_opportunities(
        games_df,
        normalized_quotes,
        model_predictions,
        champion_model_name=champion_model,
    )
    logger.info("collect_snapshot built bet opportunities count=%s", len(bet_opportunities))
    if not bet_opportunities.empty:
        bet_opportunities["collection_run_ts"] = collection_run_ts

    logger.info("collect_snapshot writing raw payloads")
    write_raw_payload(cfg.raw_data_dir, "mlb_stats", games_df.to_dict(orient="records"), captured_at)
    write_raw_payload(cfg.raw_data_dir, "kalshi", kalshi_payload, captured_at)
    write_raw_payload(cfg.raw_data_dir, "polymarket", polymarket_payload, captured_at)

    logger.info("collect_snapshot computing discrepancies")
    discrepancies = build_discrepancies(normalized_quotes, model_predictions)
    logger.info("collect_snapshot computed discrepancies count=%s", len(discrepancies))

    logger.info("collect_snapshot persisting to duckdb path=%s", cfg.duckdb_path)
    conn = connect(cfg.duckdb_path)
    append_dataframe(conn, "games", games_df)
    append_dataframe(
        conn,
        "raw_snapshots",
        pd.DataFrame(
            [
                {
                    "source": "mlb_stats",
                    "captured_at": captured_at,
                    "file_path": str(cfg.raw_data_dir / "mlb_stats"),
                    "collection_run_ts": collection_run_ts,
                },
                {
                    "source": "kalshi",
                    "captured_at": captured_at,
                    "file_path": str(cfg.raw_data_dir / "kalshi"),
                    "collection_run_ts": collection_run_ts,
                },
                {
                    "source": "polymarket",
                    "captured_at": captured_at,
                    "file_path": str(cfg.raw_data_dir / "polymarket"),
                    "collection_run_ts": collection_run_ts,
                },
            ]
        ),
    )
    append_dataframe(conn, "normalized_quotes", normalized_quotes)
    append_dataframe(conn, "model_predictions", model_predictions)
    append_dataframe(conn, "bet_opportunities", bet_opportunities)
    append_dataframe(conn, "discrepancies", discrepancies)
    try:
        record_champion_selection(champion_decision, conn=conn)
    except Exception:
        logger.warning("Failed to record champion selection", exc_info=True)
    conn.close()

    counts = {
        "games": len(games_df),
        "kalshi_quotes": len(kalshi_df),
        "polymarket_quotes": len(polymarket_df),
        "model_predictions": len(model_predictions),
        "normalized_quotes": len(normalized_quotes),
        "discrepancies": len(discrepancies),
        "bet_opportunities": len(bet_opportunities),
    }
    logger.info("collect_snapshot completed counts=%s", counts)
    return counts
