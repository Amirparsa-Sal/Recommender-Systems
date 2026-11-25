"""EASE^R - the Embarrassingly Shallow Autoencoder (Part 15; Steck, WWW 2019).

An item-item model fitted in closed form.  It minimises ``||R - R S||_F^2 + l2 ||S||_F^2``
with the constraint ``diag(S) = 0`` (so an item cannot predict itself), which has the
solution (slides, with ``P = (R'R + l2 I)^-1``)::

    S = I - P . diagMat(1 / diag(P))        i.e.   S_ij = -P_ij / P_jj   (i != j),  S_jj = 0

and the scores are ``R~ = R . S``.  There is nothing to train iteratively and a single
hyper-parameter (``--l2``) to tune.  Unlike SLIM, ``S`` is dense and can have negative
weights.

The price is memory: ``S`` has ``(items)^2`` entries, about 4.9 GB in float32 for 35k items.
The implementation never holds more than one dense matrix: ``R'R + l2 I`` is built directly
in a Fortran-ordered array, inverted in place with a Cholesky factorisation (it is positive
definite) and then rescaled in place.  Use ``--min-item-interactions`` to leave out rare
items, or ``--topk`` to keep only the strongest weights of every item after training (a
pruned model is small, can be cached with ``--cache-dir`` and used in ``hybrid_merge``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import lapack

from recsys.config import BaseConfig, param
from recsys.models.base import Recommender, register

logger = logging.getLogger(__name__)

_BLOCK = 2048  # rows / columns handled at a time by the in-place helpers


@dataclass
class EASEConfig(BaseConfig):
    l2: float = param(
        100.0,
        "L2 regularisation (lambda) of the closed-form solution. Larger = smoother, more "
        "popularity-driven weights. Typical range 10 ... 5000; try it on a log scale. "
        "On this dataset 100 beat 30 and 400.",
    )
    min_item_interactions: int = param(
        5,
        "Items with fewer training interactions are left out of the model (their score is 0). "
        "Saves memory and time: the dense matrix has (kept items)^2 entries (about 3 GB for "
        "the 27k items with >= 5 interactions, 4.9 GB for all 35k). 0 = keep every item that "
        "has at least one interaction.",
    )
    dtype: str = param(
        "float32",
        "Precision of the dense computation. float64 doubles the memory and is rarely needed.",
        choices=("float32", "float64"),
    )
    topk: int = param(
        0,
        "After training keep only the K weights with the largest absolute value for every "
        "target item (0 = keep the dense matrix). A pruned model is much smaller and faster "
        "to score, can be cached, and is required to use EASE in hybrid_merge.",
    )
    max_memory_gb: float = param(
        8.0,
        "Refuse to train if the dense matrix would need more memory than this (in GB); "
        "raise it, or raise --min-item-interactions, as appropriate.",
    )

    def validate(self) -> None:
        super().validate()
        if self.l2 <= 0:
            raise ValueError(f"EASEConfig.l2 must be positive, got {self.l2}")
        if self.min_item_interactions < 0 or self.topk < 0:
            raise ValueError("min_item_interactions and topk must be >= 0")
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")


def _symmetrize_from_upper(matrix: np.ndarray, block: int = _BLOCK) -> None:
    """Copy the upper triangle of a square matrix onto its lower triangle, in place."""
    n = matrix.shape[0]
    for start in range(0, n, block):
        stop = min(start + block, n)
        if start > 0:
            matrix[start:stop, :start] = matrix[:start, start:stop].T
        diagonal_block = matrix[start:stop, start:stop]
        diagonal_block[:] = np.triu(diagonal_block) + np.triu(diagonal_block, 1).T


def ease_weights(urm: sp.csr_matrix, l2: float, dtype: str = "float32") -> np.ndarray:
    """Dense EASE^R matrix ``S`` (n x n, C-contiguous, zero diagonal) for a binary URM.

    ``S = I - P diagMat(1 / diag P)`` with ``P = (R'R + l2 I)^-1``, computed with a single
    dense matrix kept alive (Cholesky inverse in place).
    """
    n = urm.shape[1]
    dtype = np.dtype(dtype)
    gram = (urm.T @ urm).astype(dtype).toarray(order="F")  # Fortran order: LAPACK works in place
    diagonal = np.arange(n)
    gram[diagonal, diagonal] += dtype.type(l2)

    started = time.perf_counter()
    potrf, potri = lapack.get_lapack_funcs(("potrf", "potri"), (gram,))
    factor, info = potrf(gram, lower=0, overwrite_a=1)
    if info != 0:
        raise ValueError(
            f"EASE: the Cholesky factorisation failed (LAPACK info={info}); the matrix is not "
            "numerically positive definite. Increase --l2 or use --dtype float64"
        )
    inverse, info = potri(factor, lower=0, overwrite_c=1)
    if info != 0:
        raise ValueError(f"EASE: inverting the Cholesky factor failed (LAPACK info={info})")
    del gram, factor
    _symmetrize_from_upper(inverse)
    logger.info("EASE: inverted the %d x %d matrix in %.1fs", n, n, time.perf_counter() - started)

    weights = inverse.T  # P is symmetric: the transposed Fortran buffer is P in C order
    scale = (-1.0 / np.diagonal(weights)).astype(dtype)
    weights *= scale[None, :]  # S_ij = -P_ij / P_jj
    np.fill_diagonal(weights, 0.0)
    return weights


def keep_topk_per_column(weights: np.ndarray, topk: int, block: int = 1024) -> sp.csr_matrix:
    """Keep the ``topk`` entries with the largest absolute value in every column of a dense matrix."""
    n, columns = weights.shape
    k = min(topk, n)
    rows, cols, values = [], [], []
    for start in range(0, columns, block):
        stop = min(start + block, columns)
        chunk = weights[:, start:stop]
        top = np.argpartition(-np.abs(chunk), k - 1, axis=0)[:k]  # (k, block) row indices
        values.append(np.take_along_axis(chunk, top, axis=0).ravel())
        rows.append(top.ravel())
        cols.append(np.tile(np.arange(start, stop), k))
    return sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, columns), dtype=np.float32,
    )


@register("ease")
class EASE(Recommender):
    """EASE^R: closed-form item-item autoencoder, ``S = I - P . diagMat(1 / diag P)``."""

    Config = EASEConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        cfg = self.config
        counts = np.asarray(urm.sum(axis=0)).ravel()
        kept = np.flatnonzero(counts >= max(1, cfg.min_item_interactions))
        n = len(kept)
        if n < 2:
            raise ValueError("EASE needs at least two items with interactions")
        itemsize = np.dtype(cfg.dtype).itemsize
        needed_gb = 1.3 * n * n * itemsize / 1e9  # the dense matrix plus transient blocks
        if needed_gb > cfg.max_memory_gb:
            raise ValueError(
                f"EASE with {n} items needs about {needed_gb:.1f} GB (> --max-memory-gb "
                f"{cfg.max_memory_gb}). Raise --max-memory-gb if the machine has the memory, or "
                "leave out rare items with --min-item-interactions"
            )
        logger.info(
            "EASE: %d of %d items kept, dense matrix %.1f GB (%s), l2=%g",
            n, self.n_items, n * n * itemsize / 1e9, cfg.dtype, cfg.l2,
        )
        self._kept = kept
        self._urm_kept = sp.csr_matrix(urm[:, kept], dtype=np.float32)

        weights = ease_weights(self._urm_kept, cfg.l2, cfg.dtype)
        if cfg.topk > 0:
            self._weights = keep_topk_per_column(weights, cfg.topk)
            logger.info("EASE: pruned to the top %d weights per item (%d non-zeros)", cfg.topk, self._weights.nnz)
            del weights
        else:
            self._weights = weights

    def _score(self, users: np.ndarray) -> np.ndarray:
        part = self._urm_kept[users] @ self._weights
        if sp.issparse(part):
            part = part.toarray()
        part = np.asarray(part, dtype=np.float32)
        if len(self._kept) == self.n_items:
            return part
        scores = np.zeros((len(users), self.n_items), dtype=np.float32)
        scores[:, self._kept] = part
        return scores

    def item_similarity(self) -> sp.csr_matrix:
        if not sp.issparse(self._weights):
            raise ValueError(
                "EASE keeps a dense matrix, which cannot be merged: train it with --topk K "
                "(for example 500) to use it in hybrid_merge"
            )
        if len(self._kept) == self.n_items:
            return self._weights
        selector = sp.csr_matrix(
            (np.ones(len(self._kept), dtype=np.float32), (np.arange(len(self._kept)), self._kept)),
            shape=(len(self._kept), self.n_items),
        )
        return (selector.T @ self._weights @ selector).tocsr()
