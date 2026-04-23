"""PyTorch tabular MLP classifier that plugs into the existing sklearn Pipeline.

The wrapper exposes the subset of the sklearn estimator API that the rest of
``mlpm.models.game_outcome`` relies on: ``fit`` and ``predict_proba``. It can
therefore be dropped straight into ``sklearn.pipeline.Pipeline`` next to
``LogisticRegression``/``GradientBoostingClassifier`` and participate in
``_fit_candidate_models`` without any special casing elsewhere.

Design choices:

* CPU-only by default. MLPM runs locally on a laptop/server and the feature
  matrix is small (~23 columns, tens of thousands of rows), so GPU would be
  pure overhead and makes joblib serialization more fragile.
* The network itself is stored in ``self._module`` but we also persist the
  weights as a plain ``state_dict`` via ``__getstate__`` / ``__setstate__``.
  This lets joblib/pickle round-trip the estimator across Python processes
  cleanly, and avoids having a half-initialized ``nn.Module`` if torch raises
  mid-load.
* Early stopping is driven off an internal 10% validation hold-out that is
  deterministically seeded so benchmark runs are reproducible. We copy the
  best weights back before returning.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

try:  # pragma: no cover - import guard; torch is a project dependency
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - handled in test setup
    raise ImportError(
        "PyTorch is required for the MLP contender. Install it with "
        "`pip install torch>=2.4.0`."
    ) from exc


@dataclass
class MLPConfig:
    """Tunables for the MLP. Defaults chosen to be a reasonable first pass for
    MLPM's ~23-dim tabular feature frame on ~15-30k rows."""

    hidden_layers: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15
    validation_fraction: float = 0.1
    random_state: int = 42
    use_batch_norm: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "hidden_layers": list(self.hidden_layers),
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "validation_fraction": self.validation_fraction,
            "random_state": self.random_state,
            "use_batch_norm": self.use_batch_norm,
        }


def _build_network(
    input_dim: int,
    hidden_layers: Iterable[int],
    dropout: float,
    use_batch_norm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden_layers:
        layers.append(nn.Linear(prev, width))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = width
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class MLPHomeWinClassifier(BaseEstimator, ClassifierMixin):
    """Binary classifier over the engineered MLB feature frame.

    Parameters mirror :class:`MLPConfig`. They are exposed at the top level so
    sklearn's ``get_params`` / ``set_params`` (used by pipelines and cloning)
    works without a custom wrapper.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.25,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 15,
        validation_fraction: float = 0.1,
        random_state: int = 42,
        use_batch_norm: bool = True,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.use_batch_norm = use_batch_norm

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    def fit(self, X, y) -> "MLPHomeWinClassifier":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X_arr.shape}.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("Feature/label length mismatch.")
        if not np.isfinite(X_arr).all():
            raise ValueError("Feature matrix contains non-finite values; upstream imputer must handle NaNs.")

        # Remember the training classes so ``predict_proba`` can emit columns
        # in a stable order, matching what sklearn classifiers do.
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = int(X_arr.shape[1])

        generator = torch.Generator()
        generator.manual_seed(int(self.random_state))
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        # Internal train/val split for early stopping. Use a random permutation
        # rather than sklearn's train_test_split to avoid another import, and
        # to keep the class proportions roughly intact via stratified sampling.
        train_idx, val_idx = _stratified_split(
            y_arr,
            validation_fraction=float(self.validation_fraction),
            random_state=int(self.random_state),
        )

        X_train = torch.from_numpy(X_arr[train_idx])
        y_train = torch.from_numpy(y_arr[train_idx])
        X_val = torch.from_numpy(X_arr[val_idx]) if len(val_idx) else None
        y_val = torch.from_numpy(y_arr[val_idx]) if len(val_idx) else None

        module = _build_network(
            input_dim=self.n_features_in_,
            hidden_layers=tuple(self.hidden_layers),
            dropout=float(self.dropout),
            use_batch_norm=bool(self.use_batch_norm),
        )
        module.train()

        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.BCEWithLogitsLoss()

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
            generator=generator,
            drop_last=False,
        )

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []

        for epoch in range(int(self.max_epochs)):
            module.train()
            batch_losses = []
            for batch_X, batch_y in loader:
                # BatchNorm needs at least 2 samples per batch in train mode.
                if batch_X.shape[0] < 2 and self.use_batch_norm:
                    continue
                optimizer.zero_grad()
                logits = module(batch_X).squeeze(-1)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))
            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

            if X_val is not None and len(X_val) > 0:
                module.eval()
                with torch.no_grad():
                    val_logits = module(X_val).squeeze(-1)
                    val_loss = float(loss_fn(val_logits, y_val).item())
            else:
                val_loss = train_loss

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss + 1e-6 < best_val_loss:
                best_val_loss = val_loss
                # Deep-copy so subsequent in-place updates don't corrupt the best snapshot.
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(self.patience):
                    break

        if best_state is not None:
            module.load_state_dict(best_state)
        module.eval()

        self._module = module
        self._state_dict = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        self._training_history = history
        self._best_val_loss = best_val_loss
        return self

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X_arr.shape}.")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: got {X_arr.shape[1]}, expected {self.n_features_in_}."
            )
        self._module.eval()
        with torch.no_grad():
            logits = self._module(torch.from_numpy(X_arr)).squeeze(-1)
            positive = torch.sigmoid(logits).detach().cpu().numpy()
        positive = np.clip(positive, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int64)

    # ------------------------------------------------------------------
    # Pickling helpers
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        # nn.Module pickles fine on CPU, but carrying both _module and
        # _state_dict is wasteful. Drop the module; rebuild it on load from
        # the state_dict.
        state.pop("_module", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        if hasattr(self, "_state_dict") and hasattr(self, "n_features_in_"):
            module = _build_network(
                input_dim=self.n_features_in_,
                hidden_layers=tuple(self.hidden_layers),
                dropout=float(self.dropout),
                use_batch_norm=bool(self.use_batch_norm),
            )
            module.load_state_dict({k: v for k, v in self._state_dict.items()})
            module.eval()
            self._module = module

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def training_history(self) -> list[dict[str, float]]:
        self._check_fitted()
        return list(getattr(self, "_training_history", []))

    def best_val_loss(self) -> float:
        self._check_fitted()
        return float(getattr(self, "_best_val_loss", float("nan")))

    def _check_fitted(self) -> None:
        if not hasattr(self, "_state_dict"):
            raise RuntimeError("MLPHomeWinClassifier is not fitted; call fit() first.")


def _stratified_split(
    y: np.ndarray,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) preserving class balance.

    Falls back to an empty validation set when the dataset is too small for
    the requested fraction (e.g. in unit tests with a dozen rows)."""

    if validation_fraction <= 0 or len(y) < 4:
        return np.arange(len(y)), np.array([], dtype=np.int64)

    rng = np.random.default_rng(random_state)
    val_target = max(1, int(round(len(y) * float(validation_fraction))))
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        # Proportional allocation, but guarantee at least 1 in train per class.
        take = max(1, int(round(len(cls_idx) * float(validation_fraction))))
        take = min(take, len(cls_idx) - 1)
        val_parts.append(cls_idx[:take])
        train_parts.append(cls_idx[take:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    # If we couldn't collect a validation set (e.g. only one class seen),
    # back off gracefully.
    if len(val_idx) == 0 or len(val_idx) > val_target * 3:
        return np.arange(len(y)), np.array([], dtype=np.int64)
    return train_idx, val_idx
