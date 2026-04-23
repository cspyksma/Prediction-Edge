"""Smoke tests for the PyTorch MLP contender.

These tests are intentionally light: they verify the wrapper honours the
sklearn API contract the rest of ``mlpm.models.game_outcome`` depends on
(``fit``, ``predict_proba`` shape + range, joblib round-trip, deterministic
output under a fixed seed, and participation in ``_fit_candidate_models``).
Heavy/statistical quality is left to the existing benchmark/backtest harness.
"""

from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from joblib import dump as joblib_dump
from joblib import load as joblib_load

from mlpm.models.game_outcome import (
    FEATURE_COLUMNS,
    MLP_MODEL_NAME,
    _build_mlp_pipeline,
    _fit_candidate_models,
)
from mlpm.models.mlp import MLPHomeWinClassifier


def _make_synthetic_frame(n: int = 64, seed: int = 0) -> pd.DataFrame:
    """Build a frame that contains every FEATURE_COLUMN plus the target.

    The signal is a simple rule on ``elo_diff`` plus noise so all contenders,
    including the MLP, can learn something within a handful of epochs. Rows
    are dated so ``_split_training_data`` behaves as expected downstream.
    """

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index in range(n):
        direction = 1 if index % 2 == 0 else -1
        noise = rng.normal(0.0, 0.05)
        row: dict[str, object] = {
            "game_id": f"g{index}",
            "game_date": pd.Timestamp("2024-04-01") + pd.Timedelta(days=index),
            "target_home_win": int(direction > 0),
        }
        for feature in FEATURE_COLUMNS:
            row[feature] = float(direction * 0.5 + noise)
        # elo_diff and market_home_implied_prob benefit from a more realistic scale.
        row["elo_diff"] = float(direction * 30.0 + noise * 10.0)
        row["elo_home_win_prob"] = float(0.5 + direction * 0.08 + noise)
        row["market_home_implied_prob"] = float(np.clip(0.5 + direction * 0.07 + noise, 0.01, 0.99))
        rows.append(row)
    return pd.DataFrame(rows)


def test_mlp_predict_proba_shape_and_range() -> None:
    frame = _make_synthetic_frame()
    pipeline = _build_mlp_pipeline()
    pipeline.fit(frame[FEATURE_COLUMNS], frame["target_home_win"])

    probs = pipeline.predict_proba(frame[FEATURE_COLUMNS])
    assert probs.shape == (len(frame), 2)
    assert np.isfinite(probs).all()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    # The two columns must sum to 1 within float tolerance.
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(frame)), atol=1e-5)


def test_mlp_is_deterministic_under_fixed_seed() -> None:
    frame = _make_synthetic_frame()
    X = frame[FEATURE_COLUMNS]
    y = frame["target_home_win"]

    first = _build_mlp_pipeline().fit(X, y).predict_proba(X)
    second = _build_mlp_pipeline().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(first, second, atol=1e-5)


def test_mlp_joblib_roundtrip() -> None:
    frame = _make_synthetic_frame()
    pipeline = _build_mlp_pipeline()
    pipeline.fit(frame[FEATURE_COLUMNS], frame["target_home_win"])

    tmp_dir = Path(".tmp") / f"mlp-roundtrip-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    save_path = tmp_dir / "mlp_contender.joblib"
    try:
        joblib_dump(pipeline, save_path)
        restored = joblib_load(save_path)

        original_probs = pipeline.predict_proba(frame[FEATURE_COLUMNS])
        restored_probs = restored.predict_proba(frame[FEATURE_COLUMNS])
        np.testing.assert_allclose(original_probs, restored_probs, atol=1e-5)
    finally:
        save_path.unlink(missing_ok=True)
        tmp_dir.rmdir()


def test_mlp_registered_in_candidate_models() -> None:
    frame = _make_synthetic_frame()
    # _fit_candidate_models slices training_df internally via FEATURE_COLUMNS.
    fitted = _fit_candidate_models(frame)
    assert MLP_MODEL_NAME in fitted
    probs = fitted[MLP_MODEL_NAME].predict_proba(frame[FEATURE_COLUMNS])
    assert probs.shape == (len(frame), 2)


def test_mlp_rejects_unfitted_predict() -> None:
    estimator = MLPHomeWinClassifier()
    with pytest.raises(RuntimeError):
        estimator.predict_proba(np.zeros((3, len(FEATURE_COLUMNS)), dtype=np.float32))


def test_mlp_training_history_is_recorded() -> None:
    frame = _make_synthetic_frame()
    estimator = MLPHomeWinClassifier(max_epochs=5, patience=10, random_state=7)
    estimator.fit(frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32), frame["target_home_win"].to_numpy(dtype=np.float32))

    history = estimator.training_history()
    assert history, "expected at least one epoch of training history"
    assert {"epoch", "train_loss", "val_loss"}.issubset(history[0].keys())
