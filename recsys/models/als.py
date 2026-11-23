"""Matrix factorization trained with Alternating Least Squares (Part 10).

Minimises ``||R - X Y||^2 + l2 (||X||^2 + ||Y||^2)`` over *all* entries of the
matrix, i.e. the missing-as-negative assumption: unobserved interactions are
zeros.  Because every entry has the same weight, each alternating step has the
closed form ``X = R Y (Y'Y + l2 I)^-1`` (and symmetrically for Y), so a whole
factor matrix is updated with one sparse-dense product and a small K x K solve.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from recsys.config import BaseConfig, param
from recsys.models.base import Recommender, register

logger = logging.getLogger(__name__)


@dataclass
class MFALSConfig(BaseConfig):
    factors: int = param(64, "Number of latent factors K.")
    regularization: float = param(10.0, "L2 regularisation on both factor matrices.")
    iterations: int = param(15, "Alternating iterations (each updates X then Y).")
    init_std: float = param(0.1, "Standard deviation of the random initialisation of Y.")


def _solve(rhs: np.ndarray, fixed: np.ndarray, regularization: float) -> np.ndarray:
    """Return ``rhs (F'F + reg I)^-1`` for the fixed factor matrix F."""
    gram = fixed.T @ fixed + regularization * np.eye(fixed.shape[1])
    return np.linalg.solve(gram, rhs.T).T


@register("mf_als")
class MFALS(Recommender):
    """Matrix factorization with ALS, missing interactions treated as zeros."""

    Config = MFALSConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        urm_t = urm.T.tocsr()
        item_factors = rng.normal(0.0, cfg.init_std, (self.n_items, cfg.factors))
        user_factors = np.zeros((self.n_users, cfg.factors))

        for iteration in range(1, cfg.iterations + 1):
            started = time.perf_counter()
            user_factors = _solve(urm @ item_factors, item_factors, cfg.regularization)
            item_factors = _solve(urm_t @ user_factors, user_factors, cfg.regularization)
            logger.info(
                "ALS iteration %d/%d: squared error=%.1f (%.1fs)",
                iteration, cfg.iterations,
                self._squared_error(urm, user_factors, item_factors),
                time.perf_counter() - started,
            )
        self.user_factors = user_factors.astype(np.float32)
        self.item_factors = item_factors.astype(np.float32)

    @staticmethod
    def _squared_error(urm: sp.csr_matrix, x: np.ndarray, y: np.ndarray) -> float:
        """``||R - XY'||^2`` without forming the dense matrix (R is binary)."""
        coo = urm.tocoo()
        observed = np.einsum("nk,nk->n", x[coo.row], y[coo.col])
        return float(coo.nnz - 2.0 * observed.sum() + np.sum((x.T @ x) * (y.T @ y)))

    def _score(self, users: np.ndarray) -> np.ndarray:
        return self.user_factors[users] @ self.item_factors.T
