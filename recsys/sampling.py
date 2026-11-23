"""Negative sampling for the learning-to-rank / MSE models."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

NEGATIVE_SAMPLING = ("uniform", "popularity")


class NegativeSampler:
    """Draw items a user has *not* interacted with.

    Args:
        urm: training URM (used to reject items already in the user profile).
        strategy: ``uniform`` samples items uniformly, ``popularity`` samples
            proportionally to (interaction count + 1).
        seed: random seed.
    """

    def __init__(self, urm: sp.csr_matrix, strategy: str = "uniform", seed: int = 0) -> None:
        urm = sp.csr_matrix(urm)
        if not urm.has_sorted_indices:
            urm = urm.copy()
            urm.sort_indices()
        self.n_users, self.n_items = urm.shape
        rows = np.repeat(np.arange(self.n_users, dtype=np.int64), np.diff(urm.indptr))
        # Sorted keys user * n_items + item allow O(log n) membership tests.
        self._keys = rows * self.n_items + urm.indices.astype(np.int64)
        self._rng = np.random.default_rng(seed)
        self._cdf: Optional[np.ndarray] = None
        if strategy == "popularity":
            popularity = np.asarray(urm.sum(axis=0)).ravel() + 1.0
            self._cdf = np.cumsum(popularity)
            self._cdf /= self._cdf[-1]
        elif strategy != "uniform":
            raise ValueError(f"Unknown negative sampling strategy {strategy!r}")

    def _draw(self, size: int) -> np.ndarray:
        if self._cdf is None:
            return self._rng.integers(0, self.n_items, size=size)
        drawn = np.searchsorted(self._cdf, self._rng.random(size))
        return np.minimum(drawn, self.n_items - 1)

    def is_positive(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        queries = users.astype(np.int64) * self.n_items + items
        position = np.minimum(np.searchsorted(self._keys, queries), len(self._keys) - 1)
        return self._keys[position] == queries

    def sample(self, users: np.ndarray, max_rounds: int = 10) -> np.ndarray:
        """One negative item per entry of ``users`` (rejection sampling)."""
        users = np.asarray(users)
        negatives = self._draw(len(users))
        for _ in range(max_rounds):
            collisions = self.is_positive(users, negatives)
            if not collisions.any():
                break
            negatives[collisions] = self._draw(int(collisions.sum()))
        return negatives


def pointwise_samples(
    positive_users: np.ndarray,
    positive_items: np.ndarray,
    sampler: Optional[NegativeSampler],
    n_negatives: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive interactions (label 1) plus ``n_negatives`` sampled zeros each (label 0)."""
    positive_labels = np.ones(len(positive_users), dtype=np.float32)
    if sampler is None or n_negatives <= 0:
        return positive_users, positive_items, positive_labels
    negative_users = np.repeat(positive_users, n_negatives)
    negative_items = sampler.sample(negative_users)
    users = np.concatenate([positive_users, negative_users])
    items = np.concatenate([positive_items, negative_items])
    labels = np.concatenate([positive_labels, np.zeros(len(negative_users), dtype=np.float32)])
    return users, items, labels
