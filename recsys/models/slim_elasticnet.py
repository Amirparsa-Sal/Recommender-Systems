"""SLIM trained with MSE and L1/L2 regularisation (Part 9 of the slides).

The model is ``R~ = R . S`` with ``diag(S) = 0``.  Every column ``s_i`` of S is
a separate regression problem  ``min ||r_i - R s_i||^2 + l2 ||s_i||^2 + l1 ||s_i||_1``
(``penalty`` selects ridge, lasso or elastic net), solved with coordinate
descent through scikit-learn as in the original SLIM paper.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

from recsys.config import BaseConfig, param
from recsys.models.base import Recommender, register

logger = logging.getLogger(__name__)

PENALTIES = ("elasticnet", "lasso", "ridge")


@dataclass
class SLIMElasticNetConfig(BaseConfig):
    penalty: str = param(
        "elasticnet",
        "Regularisation: elasticnet (L1+L2, uses --l1-ratio), lasso (L1 only), ridge (L2 only).",
        choices=PENALTIES,
    )
    alpha: float = param(
        1e-4, "Overall regularisation strength (scikit-learn scaling, i.e. divided by #users)."
    )
    l1_ratio: float = param(0.1, "Share of L1 in the elastic net: alpha*l1_ratio = L1, alpha*(1-l1_ratio) = L2.")
    positive: bool = param(True, "Constrain the similarity weights to be non-negative.")
    topk: int = param(100, "Keep only the K largest weights per item (0 = keep all).")
    max_iter: int = param(100, "Maximum coordinate-descent iterations per item.")
    tol: float = param(1e-4, "Coordinate-descent tolerance.")
    restrict_candidates: bool = param(
        True,
        "Only regress on items co-occurring with the target item. Exact when --positive "
        "is set and much faster; ignored otherwise.",
    )
    n_jobs: int = param(1, "Parallel workers (-1 = all cores).")


def _fit_item_chunk(
    items: np.ndarray,
    x_csc: sp.csc_matrix,
    penalty_alpha: float,
    l1_ratio: float,
    positive: bool,
    topk: int,
    max_iter: int,
    tol: float,
    restrict: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the weights of a chunk of items. Returns COO triples (row j, column i, weight)."""
    from sklearn.linear_model import ElasticNet  # imported here so workers can import it

    n_users = x_csc.shape[0]
    x_csr = x_csc.tocsr() if restrict else None
    if not restrict:
        x_csc = x_csc.copy()  # the unrestricted path edits the data in place; workers get read-only memmaps
    out_rows: List[np.ndarray] = []
    out_cols: List[np.ndarray] = []
    out_vals: List[np.ndarray] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # e.g. "coordinate descent without L1 regularisation"
        for item in items:
            start, end = x_csc.indptr[item], x_csc.indptr[item + 1]
            if end == start:
                continue
            y = np.zeros(n_users)
            y[x_csc.indices[start:end]] = 1.0

            if restrict:
                # With non-negative weights only items co-occurring with `item` can be non-zero.
                # Users with no candidate interaction contribute a constant to the loss, so they
                # are dropped and alpha is rescaled to keep the objective unchanged.
                candidates = np.unique(x_csr[x_csc.indices[start:end]].indices)
                candidates = candidates[candidates != item]
                if candidates.size == 0:
                    continue
                design = x_csc[:, candidates].tocsr()
                rows = np.flatnonzero(np.diff(design.indptr))
                design = design[rows].tocsc()
                target = y[rows]
                alpha = penalty_alpha * n_users / len(rows)
                column_ids = candidates
                work_data = None
            else:
                # Zero the column of the target item so that s_ii = 0, restore it afterwards.
                work_data = x_csc.data[start:end].copy()
                x_csc.data[start:end] = 0.0
                design, target, alpha, column_ids = x_csc, y, penalty_alpha, None

            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                positive=positive,
                fit_intercept=False,
                copy_X=False,
                max_iter=max_iter,
                tol=tol,
                selection="random",
                random_state=seed,
            )
            model.fit(design, target)
            coefficients = model.coef_
            if work_data is not None:
                x_csc.data[start:end] = work_data

            nonzero = np.flatnonzero(coefficients)
            if topk and nonzero.size > topk:
                magnitude = np.abs(coefficients[nonzero])
                nonzero = nonzero[np.argpartition(-magnitude, topk - 1)[:topk]]
            if nonzero.size == 0:
                continue
            out_rows.append(nonzero if column_ids is None else column_ids[nonzero])
            out_cols.append(np.full(nonzero.size, item))
            out_vals.append(coefficients[nonzero].astype(np.float32))

    if not out_rows:
        empty = np.empty(0)
        return empty.astype(np.int64), empty.astype(np.int64), empty.astype(np.float32)
    return np.concatenate(out_rows), np.concatenate(out_cols), np.concatenate(out_vals)


@register("slim_elasticnet")
class SLIMElasticNet(Recommender):
    """SLIM with MSE loss: ridge, lasso or elastic-net penalty, one regression per item."""

    Config = SLIMElasticNetConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        cfg = self.config
        l1_ratio = {"elasticnet": cfg.l1_ratio, "lasso": 1.0, "ridge": 0.0}[cfg.penalty]
        restrict = cfg.restrict_candidates and cfg.positive
        if cfg.restrict_candidates and not cfg.positive:
            logger.warning("--restrict-candidates is only exact with --positive; disabled")

        x_csc = urm.tocsc().astype(np.float64)
        items_with_data = np.flatnonzero(np.diff(x_csc.indptr))
        n_workers = cfg.n_jobs if cfg.n_jobs > 0 else max(1, os.cpu_count() or 1)
        chunks = np.array_split(items_with_data, max(1, min(len(items_with_data), n_workers * 20)))
        logger.info(
            "SLIM %s: fitting %d items in %d chunks (alpha=%g, l1_ratio=%g, restrict=%s)",
            cfg.penalty, len(items_with_data), len(chunks), cfg.alpha, l1_ratio, restrict,
        )

        results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(
            delayed(_fit_item_chunk)(
                chunk, x_csc, cfg.alpha, l1_ratio, cfg.positive, cfg.topk,
                cfg.max_iter, cfg.tol, restrict, cfg.seed,
            )
            for chunk in chunks
        )
        rows = np.concatenate([r[0] for r in results])
        cols = np.concatenate([r[1] for r in results])
        vals = np.concatenate([r[2] for r in results])
        self._similarity = sp.csr_matrix((vals, (rows, cols)), shape=(self.n_items, self.n_items))
        logger.info("SLIM: learned %d non-zero weights", self._similarity.nnz)

    def _score(self, users: np.ndarray) -> np.ndarray:
        return (self.urm[users] @ self._similarity).toarray()

    def item_similarity(self) -> sp.csr_matrix:
        return self._similarity
