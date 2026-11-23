"""Sparse parameter updates for the mini-batch SGD models."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

OPTIMIZERS = ("sgd", "adagrad")


def make_accumulators(params: Dict[str, np.ndarray], optimizer: str) -> Dict[str, Optional[np.ndarray]]:
    """Create the Adagrad accumulators (``None`` entries for plain SGD)."""
    if optimizer == "adagrad":
        return {name: np.zeros_like(value) for name, value in params.items()}
    return {name: None for name in params}


def sparse_update(
    param: np.ndarray,
    rows: np.ndarray,
    direction: np.ndarray,
    lr: float,
    accumulator: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> None:
    """Move the given rows of ``param`` along an *ascent* direction.

    Rows that appear several times in the batch have their directions summed
    first, so the update is well defined with repeated indices.  With an
    ``accumulator`` the step is rescaled per parameter (Adagrad), otherwise it
    is plain gradient ascent.

    Args:
        param: array of shape (n,) or (n, K), updated in place.
        rows: row index of every sample in the batch.
        direction: gradient of the objective to *maximise* (minus the loss
            gradient), shape (len(rows),) or (len(rows), K).
        lr: learning rate.
        accumulator: array shaped like ``param`` holding squared gradient sums.
    """
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    summed = np.zeros((len(unique_rows),) + direction.shape[1:], dtype=np.float32)
    np.add.at(summed, inverse.ravel(), direction.astype(np.float32, copy=False))
    if accumulator is not None:
        accumulator[unique_rows] += summed * summed
        summed = summed / (np.sqrt(accumulator[unique_rows]) + eps)
    param[unique_rows] += lr * summed
