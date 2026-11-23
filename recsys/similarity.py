"""Similarity matrices (cosine, Jaccard, Dice, Tversky, dot product) with shrink and KNN."""

from __future__ import annotations

import logging
from typing import Callable, List

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

SIMILARITIES = ("cosine", "jaccard", "dice", "tversky", "dot")
_EPS = 1e-6


def blockwise_topk(
    block_fn: Callable[[int, int], np.ndarray],
    n: int,
    topk: int = 100,
    block_size: int = 0,
) -> sp.csr_matrix:
    """Build a sparse (n x n) matrix keeping the ``topk`` largest entries of every row.

    ``block_fn(start, stop)`` must return the dense rows ``start:stop`` of the full
    matrix.  The diagonal is zeroed and non-positive entries are dropped.  Working in
    blocks of rows keeps the memory usage bounded.  ``topk <= 0`` keeps everything.
    """
    keep = topk if 0 < topk < n - 1 else None
    if keep is None:
        logger.warning("topk=%d keeps every non-zero entry: the result can be very large", topk)
    if block_size <= 0:
        block_size = int(min(2000, max(1, 2e7 // n)))

    all_rows: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    all_vals: List[np.ndarray] = []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = block_fn(start, stop)

        local = np.arange(stop - start)
        block[local, start + local] = 0.0  # no self-connection

        if keep is None:
            rows, cols = np.nonzero(block)
            vals = block[rows, cols]
            rows = rows + start
        else:
            top_cols = np.argpartition(block, n - keep, axis=1)[:, n - keep :]
            vals = np.take_along_axis(block, top_cols, axis=1).ravel()
            rows = np.repeat(np.arange(start, stop), keep)
            cols = top_cols.ravel()
            positive = vals > 0
            rows, cols, vals = rows[positive], cols[positive], vals[positive]
        all_rows.append(rows)
        all_cols.append(cols)
        all_vals.append(vals)
        logger.debug("top-K: %d/%d rows done", stop, n)

    return sp.csr_matrix(
        (np.concatenate(all_vals), (np.concatenate(all_rows), np.concatenate(all_cols))),
        shape=(n, n),
        dtype=np.float32,
    )


def keep_topk_per_row(matrix: sp.spmatrix, topk: int) -> sp.csr_matrix:
    """Keep the ``topk`` largest entries of every row of a sparse matrix (``topk <= 0``: all)."""
    matrix = sp.csr_matrix(matrix)
    if topk <= 0:
        return matrix
    counts = np.diff(matrix.indptr)
    keep = np.ones(matrix.nnz, dtype=bool)
    for row in np.flatnonzero(counts > topk):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        smallest = np.argpartition(matrix.data[start:stop], stop - start - topk)[: stop - start - topk]
        keep[start + smallest] = False
    rows = np.repeat(np.arange(matrix.shape[0]), counts)[keep]
    return sp.csr_matrix(
        (matrix.data[keep], (rows, matrix.indices[keep])), shape=matrix.shape, dtype=matrix.dtype
    )


def compute_similarity(
    matrix: sp.spmatrix,
    kind: str = "cosine",
    topk: int = 100,
    shrink: float = 0.0,
    tversky_alpha: float = 1.0,
    tversky_beta: float = 1.0,
    block_size: int = 0,
) -> sp.csr_matrix:
    """Similarity between the *columns* of a binary matrix, keeping the top-K per row.

    The matrix has shape (features, entities); the result is (entities,
    entities) where row ``i`` holds the similarities of entity ``i`` with its
    ``topk`` most similar entities (diagonal excluded).  Item-based models pass
    the URM (users x items); user-based models pass the transposed URM.

    Similarities (``inter`` = co-occurrences, ``deg`` = number of interactions):

    * cosine:  ``inter / (sqrt(deg_i) * sqrt(deg_j) + shrink)``
    * tversky: ``inter / (inter + a * (deg_i - inter) + b * (deg_j - inter) + shrink)``
    * jaccard / dice: Tversky with ``a = b = 1`` / ``a = b = 0.5``
    * dot:     ``inter`` (shrink is ignored)

    The computation is done in dense blocks of rows to bound memory usage.
    """
    if kind not in SIMILARITIES:
        raise ValueError(f"Unknown similarity {kind!r}, choose one of {SIMILARITIES}")
    if kind == "jaccard":
        tversky_alpha = tversky_beta = 1.0
    elif kind == "dice":
        tversky_alpha = tversky_beta = 0.5

    x = sp.csr_matrix(matrix, dtype=np.float32)
    n = x.shape[1]
    x_t = x.T.tocsr()
    degrees = np.asarray(x.sum(axis=0)).ravel().astype(np.float32)
    norms = np.sqrt(degrees)

    def similarity_block(start: int, stop: int) -> np.ndarray:
        block = (x_t[start:stop] @ x).toarray()  # co-occurrence counts, (rows, n)
        if kind == "cosine":
            block /= norms[start:stop, None] * norms[None, :] + shrink + _EPS
        elif kind != "dot":
            denominator = (
                block
                + tversky_alpha * (degrees[start:stop, None] - block)
                + tversky_beta * (degrees[None, :] - block)
                + shrink
                + _EPS
            )
            block /= denominator
        return block

    return blockwise_topk(similarity_block, n, topk, block_size)
