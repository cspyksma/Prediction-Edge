from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from mlpm.config.settings import settings
from mlpm.evaluation.strategy import (
    ChampionDecision,
    bootstrap_roi_ci,
    record_champion_selection,
    select_champion_model,
    select_champion_with_rationale,
    wilson_win_rate_ci,
)
from mlpm.storage.duckdb import append_dataframe, connect, connect_read_only, query_dataframe


def _workspace_db_path(name: str) -> Path:
    path = Path(".tmp") / f"{name}-{uuid.uuid4().hex}.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Wilson CI invariants
# ---------------------------------------------------------------------------

def test_wilson_ci_returns_point_bounded_by_interval() -> None:
    point, lower, upper = wilson_win_rate_ci(wins=30, trials=50, confidence=0.95)
    assert 0.0 <= lower <= point <= upper <= 1.0
    assert point == 0.6


def test_wilson_ci_zero_trials_is_maximally_uncertain() -> None:
    point, lower, upper = wilson_win_rate_ci(wins=0, trials=0)
    assert point == 0.0
    assert lower == 0.0
    assert upper == 1.0


def test_wilson_ci_widens_with_fewer_trials() -> None:
    _, small_lower, small_upper = wilson_win_rate_ci(wins=6, trials=10)
    _, big_lower, big_upper = wilson_win_rate_ci(wins=600, trials=1000)
    assert (small_upper - small_lower) > (big_upper - big_lower)


# ---------------------------------------------------------------------------
# Bootstrap CI invariants
# ---------------------------------------------------------------------------

def test_bootstrap_roi_ci_is_deterministic_with_seed() -> None:
    returns = np.array([0.5, -1.0, 0.5, 1.2, -1.0, 0.9])
    stakes = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    first = bootstrap_roi_ci(returns, stakes, confidence=0.9, n_samples=200, seed=123)
    second = bootstrap_roi_ci(returns, stakes, confidence=0.9, n_samples=200, seed=123)
    assert first == second


def test_bootstrap_roi_ci_bounds_contain_point_for_positive_series() -> None:
    returns = np.array([0.5, 0.8, 0.9, 0.6, 0.7, 0.85])
    stakes = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    point, lower, upper = bootstrap_roi_ci(returns, stakes, confidence=0.95, n_samples=500, seed=42)
    assert lower <= point <= upper
    assert point > 0.0


def test_bootstrap_roi_ci_handles_empty_input() -> None:
    point, lower, upper = bootstrap_roi_ci(np.array([]), np.array([]))
    assert (point, lower, upper) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Guardrailed selection behaviour
# ---------------------------------------------------------------------------

def _seed_settled_bet_opportunities(conn, rows: list[dict]) -> None:
    """Insert rows into both `bet_opportunities` and `game_results` so the
    `settled_bet_opportunities_deduped` view surfaces them."""
    opp_rows = []
    result_rows = []
    for row in rows:
        opp_rows.append(
            {
                "game_id": row["game_id"],
                "game_date": row["game_date"],
                "event_start_time": row.get("event_start_time", f"{row['game_date']}T18:00:00"),
                "snapshot_ts": row.get("snapshot_ts", f"{row['game_date']}T17:00:00"),
                "collection_run_ts": row.get("collection_run_ts", f"{row['game_date']}T17:00:00"),
                "model_name": row["model_name"],
                "source": row.get("source", "kalshi"),
                "market_id": row.get("market_id", f"m-{row['game_id']}"),
                "team": row["team"],
                "opponent_team": row["opponent_team"],
                "is_home_team": bool(row.get("is_home_team", True)),
                "model_prob": float(row.get("model_prob", 0.6)),
                "market_prob": float(row.get("market_prob", 0.5)),
                "edge_bps": int(row.get("edge_bps", 1000)),
                "expected_value": float(row.get("expected_value", 0.1)),
                "implied_decimal_odds": float(row.get("implied_decimal_odds", 2.0)),
                "stake_units": 1.0,
                "is_actionable": True,
                "is_champion": False,
            }
        )
        result_rows.append(
            {
                "game_id": row["game_id"],
                "game_date": row["game_date"],
                "away_team": row.get("away_team", row["opponent_team"] if row.get("is_home_team", True) else row["team"]),
                "home_team": row.get("home_team", row["team"] if row.get("is_home_team", True) else row["opponent_team"]),
                "winner_team": row["winner_team"],
                "away_score": int(row.get("away_score", 2)),
                "home_score": int(row.get("home_score", 4)),
            }
        )
    append_dataframe(conn, "bet_opportunities", pd.DataFrame(opp_rows))
    append_dataframe(conn, "game_results", pd.DataFrame(result_rows))


def test_select_champion_with_rationale_returns_no_data_on_empty_db(monkeypatch) -> None:
    db_path = _workspace_db_path("champ-empty")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()
    # Initialize schema by opening/closing a RW connection
    connect(settings().duckdb_path).close()

    decision = select_champion_with_rationale()
    assert isinstance(decision, ChampionDecision)
    assert decision.action == "no_data"
    assert decision.chosen_model is None


def test_select_champion_with_rationale_cold_start_picks_top_challenger(monkeypatch) -> None:
    db_path = _workspace_db_path("champ-cold-start")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("STRATEGY_CHAMPION_MIN_BETS", "1")
    monkeypatch.setenv("STRATEGY_CHAMPION_WINDOW_DAYS", "30")
    settings.cache_clear()

    conn = connect(settings().duckdb_path)
    # model_a goes 3/3, model_b goes 0/3
    rows = []
    for idx, won in enumerate([True, True, True]):
        rows.append(
            {
                "game_id": f"a{idx}",
                "game_date": f"2026-04-{10 + idx:02d}",
                "model_name": "model_a",
                "team": "Home",
                "opponent_team": "Away",
                "is_home_team": True,
                "winner_team": "Home" if won else "Away",
            }
        )
    for idx, won in enumerate([False, False, False]):
        rows.append(
            {
                "game_id": f"b{idx}",
                "game_date": f"2026-04-{10 + idx:02d}",
                "model_name": "model_b",
                "team": "Home",
                "opponent_team": "Away",
                "is_home_team": True,
                "winner_team": "Home" if won else "Away",
            }
        )
    _seed_settled_bet_opportunities(conn, rows)
    conn.close()

    decision = select_champion_with_rationale()
    assert decision.action == "cold_start"
    assert decision.chosen_model == "model_a"
    assert decision.incumbent_model is None
    assert decision.challenger_model == "model_a"


def test_select_champion_keeps_incumbent_when_challenger_ci_does_not_clear(monkeypatch) -> None:
    db_path = _workspace_db_path("champ-keep")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("STRATEGY_CHAMPION_MIN_BETS", "3")
    monkeypatch.setenv("STRATEGY_CHAMPION_WINDOW_DAYS", "30")
    monkeypatch.setenv("STRATEGY_CHAMPION_CI_METHOD", "wilson")
    monkeypatch.setenv("STRATEGY_CHAMPION_CI_CONFIDENCE", "0.95")
    settings.cache_clear()

    conn = connect(settings().duckdb_path)

    # Incumbent: model_a with 10/15 wins (win rate ≈ 0.667)
    rows = []
    wins_a = [True] * 10 + [False] * 5
    for idx, won in enumerate(wins_a):
        rows.append(
            {
                "game_id": f"a{idx:02d}",
                "game_date": f"2026-04-{(idx % 10) + 5:02d}",
                "model_name": "model_a",
                "team": "Home",
                "opponent_team": "Away",
                "is_home_team": True,
                "winner_team": "Home" if won else "Away",
            }
        )
    # Challenger: model_b with 3/4 wins — point 0.75 looks better than A, but CI is wide (wilson
    # lower ≈ 0.30) so the guardrail should keep model_a as incumbent.
    wins_b = [True, True, True, False]
    for idx, won in enumerate(wins_b):
        rows.append(
            {
                "game_id": f"b{idx:02d}",
                "game_date": f"2026-04-{idx + 5:02d}",
                "model_name": "model_b",
                "team": "Home",
                "opponent_team": "Away",
                "is_home_team": True,
                "winner_team": "Home" if won else "Away",
            }
        )
    _seed_settled_bet_opportunities(conn, rows)

    # Seed an incumbent row into champion_history so model_a is the current champion.
    append_dataframe(
        conn,
        "champion_history",
        pd.DataFrame(
            [
                {
                    "selected_at": pd.Timestamp("2026-04-01T00:00:00"),
                    "reference_date": None,
                    "chosen_model": "model_a",
                    "incumbent_model": None,
                    "challenger_model": "model_a",
                    "action": "cold_start",
                    "reason": "seed",
                    "window_days": 30,
                    "min_bets": 3,
                    "confidence": 0.95,
                    "challenger_bets": 0,
                    "challenger_roi": 0.0,
                    "challenger_units": 0.0,
                    "challenger_win_rate": 0.0,
                    "challenger_ci_lower": 0.0,
                    "challenger_ci_upper": 0.0,
                    "incumbent_bets": 0,
                    "incumbent_roi": 0.0,
                    "incumbent_win_rate": 0.0,
                }
            ]
        ),
    )
    conn.close()

    decision = select_champion_with_rationale()
    assert decision.incumbent_model == "model_a"
    assert decision.chosen_model == "model_a"
    assert decision.action == "kept_incumbent"


def test_select_champion_switches_when_challenger_ci_clears_incumbent(monkeypatch) -> None:
    db_path = _workspace_db_path("champ-switch")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("STRATEGY_CHAMPION_MIN_BETS", "3")
    monkeypatch.setenv("STRATEGY_CHAMPION_WINDOW_DAYS", "60")
    monkeypatch.setenv("STRATEGY_CHAMPION_CI_METHOD", "wilson")
    monkeypatch.setenv("STRATEGY_CHAMPION_CI_CONFIDENCE", "0.95")
    settings.cache_clear()

    conn = connect(settings().duckdb_path)

    # Incumbent model_a: poor performer, 4/30 wins (≈0.13, upper bound very low).
    rows = []
    wins_a = [True] * 4 + [False] * 26
    for idx, won in enumerate(wins_a):
        day = (idx % 28) + 1
        rows.append(
            {
                "game_id": f"a{idx:02d}",
                "game_date": f"2026-04-{day:02d}",
                "model_name": "model_a",
                "team": "Home",
                "opponent_team": "Away",
                "is_home_team": True,
                "winner_team": "Home" if won else "Away",
            }
        )
    # Challenger model_b: strong, 30/35 wins (≈0.857, lower bound ~0.70) — well above A's point (~0.13).
    wins_b = [True] * 30 + [False] * 5
    for idx, won in enumerate(wins_b):
        day = (idx % 28) + 1
        rows.append(
            {
                "game_id": f"b{idx:02d}",
                "game_date": f"2026-04-{day:02d}",
                "model_name": "model_b",
                "team": "Home",
                "opponent_team": "Away",
                "is_home_team": True,
                "winner_team": "Home" if won else "Away",
            }
        )
    _seed_settled_bet_opportunities(conn, rows)

    # Mark model_a as incumbent.
    append_dataframe(
        conn,
        "champion_history",
        pd.DataFrame(
            [
                {
                    "selected_at": pd.Timestamp("2026-04-01T00:00:00"),
                    "reference_date": None,
                    "chosen_model": "model_a",
                    "incumbent_model": None,
                    "challenger_model": "model_a",
                    "action": "cold_start",
                    "reason": "seed",
                    "window_days": 60,
                    "min_bets": 3,
                    "confidence": 0.95,
                    "challenger_bets": 0,
                    "challenger_roi": 0.0,
                    "challenger_units": 0.0,
                    "challenger_win_rate": 0.0,
                    "challenger_ci_lower": 0.0,
                    "challenger_ci_upper": 0.0,
                    "incumbent_bets": 0,
                    "incumbent_roi": 0.0,
                    "incumbent_win_rate": 0.0,
                }
            ]
        ),
    )
    conn.close()

    decision = select_champion_with_rationale()
    assert decision.incumbent_model == "model_a"
    assert decision.challenger_model == "model_b"
    assert decision.chosen_model == "model_b"
    assert decision.action == "switched"
    assert decision.challenger_ci_lower > decision.incumbent_win_rate

    # Backwards-compatible wrapper returns the chosen name.
    assert select_champion_model() == "model_b"


def test_record_champion_selection_roundtrips_via_view(monkeypatch) -> None:
    db_path = _workspace_db_path("champ-record")
    monkeypatch.setenv("DUCKDB_PATH", str(db_path))
    settings.cache_clear()

    decision = ChampionDecision(
        chosen_model="model_z",
        incumbent_model="model_y",
        challenger_model="model_z",
        action="switched",
        reason="CI cleared",
        challenger_bets=25,
        challenger_roi=0.11,
        challenger_units=2.75,
        challenger_win_rate=0.58,
        challenger_ci_lower=0.44,
        challenger_ci_upper=0.72,
        incumbent_bets=30,
        incumbent_roi=0.01,
        incumbent_win_rate=0.52,
        confidence=0.95,
        window_days=30,
        min_bets=10,
    )
    record_champion_selection(decision)

    conn = connect_read_only(settings().duckdb_path)
    try:
        latest = query_dataframe(conn, "SELECT * FROM latest_champion_selection")
    finally:
        conn.close()

    assert not latest.empty
    assert latest.iloc[0]["chosen_model"] == "model_z"
    assert latest.iloc[0]["action"] == "switched"
    assert latest.iloc[0]["challenger_bets"] == 25
