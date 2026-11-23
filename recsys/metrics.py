"""Offline ranking metrics (binary relevance): NDCG@N and MAP@N."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import scipy.sparse as sp


def evaluate_recommendations(
    recommendations: np.ndarray,
    target_urm: sp.csr_matrix,
    users: np.ndarray,
    cutoffs: Sequence[int],
) -> Dict[str, float]:
    """NDCG@N and MAP@N averaged over ``users``.

    Args:
        recommendations: item indices, shape (len(users), >= max(cutoffs)), best first.
        target_urm: the held-out validation interactions. Every
            user in ``users`` must have at least one interaction in it.
        users: row index of every recommendation list.
        cutoffs: list lengths N at which the metrics are computed.

    * ``ndcg@N``: DCG with a ``1/log2(position + 1)`` discount divided by the
      ideal DCG of the user (all relevant items first, at most N of them).
    * ``map@N``: average precision at N, normalised by ``min(N, #relevant)``,
      averaged over users.
    """
    users = np.asarray(users)
    max_k = recommendations.shape[1]
    n_relevant = np.diff(target_urm.indptr)[users]
    if np.any(n_relevant == 0):
        raise ValueError("Every evaluated user needs at least one held-out interaction")

    rows = np.repeat(users, max_k)
    hits = np.asarray(target_urm[rows, recommendations.ravel()]).reshape(len(users), max_k) > 0

    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    cumulative_discounts = np.cumsum(discounts)

    results: Dict[str, float] = {}
    for k in cutoffs:
        if k > max_k:
            raise ValueError(f"Cutoff {k} is larger than the recommendation length {max_k}")
        top = hits[:, :k]
        ideal_length = np.minimum(n_relevant, k)

        dcg = (top * discounts[:k]).sum(axis=1)
        idcg = cumulative_discounts[ideal_length - 1]
        precision_at_i = np.cumsum(top, axis=1) / np.arange(1, k + 1)
        average_precision = (precision_at_i * top).sum(axis=1) / ideal_length

        results[f"ndcg@{k}"] = float(np.mean(dcg / idcg))
        results[f"map@{k}"] = float(np.mean(average_precision))
    return results
