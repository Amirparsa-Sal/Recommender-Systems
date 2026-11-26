"""GF-CF - Graph-Filter Collaborative Filtering (Part 16; Shen et al., CIKM 2021).

An item-based model that needs no training loop.  The slides start from graph
convolution (LightGCN): ``E^h = G^h . E^0`` raises the singular values of the normalised
adjacency matrix to the power ``h``, so large singular values (popular, "easy" signal) are
amplified and small ones disappear.  GF-CF embraces this and computes the result
analytically:

* normalise the URM, ``R^ = D_U^-a . R . D_I^-b``  (``a = b = 1/2`` in the paper, 0.4 here),
* take its truncated SVD, ``R^ ~ U_K . Sigma_K . V_K'``,
* build the item-item similarity (slides, with ``a = b = 1/2``)::

      S = linear_weight . R^'R^  +  alpha . D_I^-b . V_K . V_K' . D_I^b

  The first term is a (normalised) co-occurrence filter, the second one an *ideal low-pass
  filter*: the projection on the ``K`` strongest singular directions, written back into
  the original item space (note the opposite signs of the exponents of ``D_I``).

and the scores are ``R . S``.  ``--filter-power p`` generalises the ideal filter: direction
``k`` is weighted by ``(sigma_k / sigma_1)^p`` (``p = 0``: all ``K`` directions count
equally, ``p > 0``: a soft low-pass that favours the strongest ones, ``p < 0``: the weak
directions count more, which pushes away from popular items).

``S`` is dense (``items x items``, 4.9 GB in float32 for 35k items) and slow to use, so by
default it is never built: the scores of a batch of users are computed as
``(r . R^') . R^`` (two sparse products) plus the low-rank part ``((r . D_I^-b) . V_K) . V_K'``.
``--topk K`` builds ``S`` blockwise instead and keeps the ``K`` strongest entries of every
column; the pruned model is a plain item-item similarity that can be used by
``hybrid_merge``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from recsys.config import BaseConfig, param
from recsys.models.base import Recommender, register

logger = logging.getLogger(__name__)

SVD_SOLVERS = ("randomized", "arpack")


def _power(degrees: np.ndarray, exponent: float) -> np.ndarray:
    """``degrees ** exponent`` with 0 where the degree is 0 (empty users / items)."""
    result = np.zeros(len(degrees), dtype=np.float32)
    positive = degrees > 0
    result[positive] = np.power(degrees[positive].astype(np.float64), exponent)
    return result


@dataclass
class GFCFConfig(BaseConfig):
    n_factors: int = param(
        512,
        "K: number of singular directions kept by the low-pass filter (truncated SVD rank). "
        "Larger = more personalised / slower; on this dataset 256 / 512 / 768 / 1024 gave "
        "0.1325 / 0.1342 / 0.1342 / 0.1333.",
    )
    alpha: float = param(
        1.0,
        "Weight of the ideal low-pass (SVD) term relative to the co-occurrence term "
        "(the paper uses 0.3). 0 = only the co-occurrence filter.",
    )
    linear_weight: float = param(
        1.0,
        "Weight of the normalised co-occurrence term R^'R^ (0 = only the low-pass term).",
    )
    user_exponent: float = param(
        0.4,
        "Exponent a of the user-degree normalisation, R^ = D_U^-a . R . D_I^-b "
        "(the paper uses 0.5, 0 = none). On this dataset 0.4 beat 0.5.",
    )
    item_exponent: float = param(
        0.4,
        "Exponent b of the item-degree normalisation (the paper uses 0.5). Larger values "
        "penalise popular items more (0 = none, 1 = full degree normalisation). "
        "On this dataset 0.4 beat 0.5; 0.7 was much worse.",
    )
    filter_power: float = param(
        0.0,
        "Weight (sigma_k / sigma_1)^p of the singular directions in the low-pass term: "
        "0 = ideal filter (all K equal), > 0 = soft low-pass, < 0 = favour the weak directions.",
    )
    svd_solver: str = param(
        "randomized",
        "Truncated SVD: 'randomized' (fast, approximate) or 'arpack' (exact Lanczos, slower).",
        choices=SVD_SOLVERS,
    )
    svd_iterations: int = param(
        10, "Power iterations of the randomized solver; more = more accurate, slower. "
            "Ignored by arpack."
    )
    topk: int = param(
        0,
        "0 = score with the factored form (no item-item matrix is built). K > 0: build S "
        "blockwise and keep the K largest entries of every column; needed for hybrid_merge.",
    )

    def validate(self) -> None:
        super().validate()
        self.n_factors = int(self.n_factors)
        self.svd_iterations = int(self.svd_iterations)
        self.topk = int(self.topk)
        if self.n_factors < 1:
            raise ValueError("n_factors must be >= 1")
        if self.alpha < 0 or self.linear_weight < 0:
            raise ValueError("alpha and linear_weight must be >= 0")
        if self.alpha == 0 and self.linear_weight == 0:
            raise ValueError("alpha and linear_weight cannot both be 0")
        if self.user_exponent < 0 or self.item_exponent < 0:
            raise ValueError("user_exponent and item_exponent must be >= 0")
        if self.svd_iterations < 0 or self.topk < 0:
            raise ValueError("svd_iterations and topk must be >= 0")


def truncated_svd(matrix: sp.csr_matrix, rank: int, solver: str, iterations: int, seed: int):
    """Top-``rank`` singular triplets ``(U, sigma, Vt)`` sorted by decreasing sigma."""
    if solver == "arpack":
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(min(matrix.shape))
        u, sigma, vt = svds(matrix.astype(np.float64), k=rank, v0=v0)
        order = np.argsort(-sigma)
        return u[:, order], sigma[order], vt[order]
    from sklearn.utils.extmath import randomized_svd

    return randomized_svd(matrix, n_components=rank, n_oversamples=20, n_iter=iterations,
                          random_state=seed)


@register("gfcf")
class GFCF(Recommender):
    """GF-CF: normalised co-occurrence filter + ideal low-pass filter (truncated SVD)."""

    Config = GFCFConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        cfg = self.config
        binary = sp.csr_matrix(urm, dtype=np.float32)
        binary.data[:] = 1.0
        user_degree = np.asarray(binary.sum(axis=1)).ravel()
        item_degree = np.asarray(binary.sum(axis=0)).ravel()

        self._binary = binary
        self._item_down = _power(item_degree, -cfg.item_exponent)  # D_I^-b
        self._item_up = _power(item_degree, cfg.item_exponent)  # D_I^b
        normalised = (
            sp.diags(_power(user_degree, -cfg.user_exponent)) @ binary @ sp.diags(self._item_down)
        ).tocsr().astype(np.float32)
        self._norm = normalised
        self._norm_t = normalised.T.tocsr()

        rank = max(1, min(cfg.n_factors, min(normalised.shape) - 1))
        started = time.perf_counter()
        _, sigma, vt = truncated_svd(normalised, rank, cfg.svd_solver, cfg.svd_iterations, cfg.seed)
        logger.info(
            "GF-CF: truncated SVD (%s) with K=%d in %.1fs, sigma_1=%.3f, sigma_K=%.3f",
            cfg.svd_solver, rank, time.perf_counter() - started, sigma[0], sigma[-1],
        )
        weights = np.power(np.maximum(sigma / sigma[0], 1e-12), cfg.filter_power)
        self._v = np.ascontiguousarray(vt.T, dtype=np.float32)  # (items, K)
        self._v_weighted = np.ascontiguousarray(self._v * weights[None, :].astype(np.float32))
        self._sigma = sigma.astype(np.float32)

        self._similarity = None
        if cfg.topk > 0:
            self._similarity = self._pruned_similarity(cfg.topk)

    # ---------------------------------------------------------------- similarity
    def _pruned_similarity(self, topk: int, block: int = 1024) -> sp.csr_matrix:
        """Build ``S`` column block by column block and keep the ``topk`` largest entries per column."""
        cfg = self.config
        n = self.n_items
        k = min(topk, n - 1)
        left = (self._item_down[:, None] * self._v_weighted).astype(np.float32)  # D_I^-b V W
        right = (self._item_up[:, None] * self._v).astype(np.float32)  # D_I^b V
        rows, cols, values = [], [], []
        started = time.perf_counter()
        for start in range(0, n, block):
            stop = min(start + block, n)
            dense = cfg.alpha * (left @ right[start:stop].T)
            if cfg.linear_weight > 0:
                dense += cfg.linear_weight * (self._norm_t @ self._norm[:, start:stop]).toarray()
            dense[np.arange(start, stop), np.arange(stop - start)] = 0.0  # no self-similarity
            top = np.argpartition(-dense, k - 1, axis=0)[:k]
            chunk = np.take_along_axis(dense, top, axis=0)
            keep = chunk > 0
            rows.append(top[keep])
            values.append(chunk[keep])
            cols.append(np.broadcast_to(np.arange(start, stop), top.shape)[keep])
        logger.info("GF-CF: built the pruned similarity (top %d per item) in %.1fs",
                    topk, time.perf_counter() - started)
        return sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n), dtype=np.float32,
        )

    def item_similarity(self) -> sp.csr_matrix:
        if self._similarity is None:
            raise ValueError(
                "GF-CF without --topk never builds the dense item-item matrix, which therefore "
                "cannot be merged: train it with --topk K (for example 500) to use it in hybrid_merge"
            )
        return self._similarity

    # ------------------------------------------------------------------- scoring
    def _score(self, users: np.ndarray) -> np.ndarray:
        cfg = self.config
        profiles = self._binary[users]
        if self._similarity is not None:
            return np.asarray((profiles @ self._similarity).toarray(), dtype=np.float32)

        scores = np.zeros((len(users), self.n_items), dtype=np.float32)
        if cfg.linear_weight > 0:  # r . R^' . R^  (user-based two-step walk, no item matrix)
            two_step = (profiles @ self._norm_t) @ self._norm
            scores += cfg.linear_weight * two_step.toarray()
        if cfg.alpha > 0:  # ((r . D_I^-b) . V_K . W) . V_K' . D_I^b
            projected = (profiles @ sp.diags(self._item_down)) @ self._v_weighted
            scores += cfg.alpha * (np.asarray(projected) @ self._v.T) * self._item_up[None, :]
        return scores
