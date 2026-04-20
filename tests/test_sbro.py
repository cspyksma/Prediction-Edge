"""Unit tests for the SportsbookReviewsOnline (SBRO) parser + loader.

These tests build a synthetic SBRO-shaped workbook in a tmp dir rather than
depending on real SBRO xlsx files (which live on the user's machine).
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import uuid

import pandas as pd
import pytest

from mlpm.ingest.sbro import (
    SBRO_SOURCE,
    load_sbro_into_priors,
    parse_sbro_workbook,
)
from mlpm.storage.duckdb import connect as db_connect


@pytest.fixture()
def sbro_workbook() -> Path:
    """Build a tiny SBRO-shaped xlsx with two games: one is skipped due to
    missing closing ML, the other has a known ML pair so we can assert
    probabilities exactly.

    Row 1/2  : 404 (Apr 4) ATL @ NYM, closing -110 / +100  → valid
    Row 3/4  : 405 (Apr 5) BOS @ NYY, closing NL  / NL    → skipped (no lines)
    """
    df = pd.DataFrame(
        {
            "Date": [404, 404, 405, 405],
            "VH":   ["V", "H", "V", "H"],
            "Team": ["ATL", "NYM", "BOS", "NYY"],
            "Final": [3, 5, 2, 7],
            "Close": [-110, 100, "NL", "NL"],
        }
    )
    tmp_dir = Path(".tmp") / f"sbro-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / "mlb odds 2024.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


def test_parse_sbro_workbook_extracts_devigged_probs(sbro_workbook: Path) -> None:
    df = parse_sbro_workbook(sbro_workbook, year=2024)

    # Only the first game survives — the second has no moneylines.
    assert len(df) == 1
    row = df.iloc[0]
    assert row["game_date"] == date(2024, 4, 4)
    assert row["away_team"] == "Atlanta Braves"
    assert row["home_team"] == "New York Mets"
    assert row["away_ml_close"] == -110
    assert row["home_ml_close"] == 100
    assert row["away_score"] == 3
    assert row["home_score"] == 5
    assert row["source"] == SBRO_SOURCE

    # Implied probs: -110 → 0.5238, +100 → 0.5000. Sum = 1.0238 (vig ~2.4%).
    # De-vigged away: 0.5238/1.0238 ≈ 0.5116; home: 0.5000/1.0238 ≈ 0.4884.
    assert row["away_implied_prob_raw"] == pytest.approx(110 / 210, abs=1e-4)
    assert row["home_implied_prob_raw"] == pytest.approx(100 / 200, abs=1e-4)
    assert row["away_fair_prob"] == pytest.approx((110 / 210) / (110 / 210 + 0.5), abs=1e-4)
    assert row["home_fair_prob"] == pytest.approx(0.5 / (110 / 210 + 0.5), abs=1e-4)
    # De-vigged pair must sum to exactly 1.0.
    assert row["away_fair_prob"] + row["home_fair_prob"] == pytest.approx(1.0, abs=1e-9)


def test_load_sbro_into_priors_matches_game_results_and_is_idempotent(sbro_workbook: Path) -> None:
    """End-to-end: parse, seed game_results, load priors, re-load, assert no dupes."""
    tmp_dir = Path(".tmp") / f"sbro-db-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_dir / "test.duckdb"
    conn = db_connect(db_path)
    try:
        # Seed a matching game_results row so the join resolves.
        conn.execute(
            """
            INSERT INTO game_results
                (game_id, game_date, event_start_time, away_team, home_team,
                 winner_team, away_score, home_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "747000",
                date(2024, 4, 4),
                datetime(2024, 4, 4, 23, 10),
                "Atlanta Braves",
                "New York Mets",
                "New York Mets",
                3, 5,
            ],
        )

        priors = parse_sbro_workbook(sbro_workbook, year=2024)
        result = load_sbro_into_priors(priors, conn=conn)
        assert result["status"] == "ok"
        assert result["rows_inserted"] == 1
        assert result["rows_unmatched"] == 0

        stored = conn.execute(
            "SELECT game_id, source, home_fair_prob, away_fair_prob FROM historical_market_priors"
        ).fetchdf()
        assert len(stored) == 1
        assert stored.iloc[0]["game_id"] == "747000"
        assert stored.iloc[0]["source"] == SBRO_SOURCE

        # Re-load — should replace, not duplicate.
        result2 = load_sbro_into_priors(priors, conn=conn)
        assert result2["rows_inserted"] == 1
        stored2 = conn.execute("SELECT COUNT(*) FROM historical_market_priors").fetchone()[0]
        assert stored2 == 1
    finally:
        conn.close()


def test_load_sbro_into_priors_drops_unmatched_rows(sbro_workbook: Path) -> None:
    """If game_results has no matching row, the prior is dropped cleanly."""
    tmp_dir = Path(".tmp") / f"sbro-db-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_dir / "test.duckdb"
    conn = db_connect(db_path)
    try:
        # Note: deliberately do NOT seed game_results.

        priors = parse_sbro_workbook(sbro_workbook, year=2024)
        result = load_sbro_into_priors(priors, conn=conn)
        # With no game_results rows at all, loader short-circuits.
        assert result["status"] == "empty_game_results"
        assert result["rows_inserted"] == 0
        assert result["rows_unmatched"] == 1
    finally:
        conn.close()
