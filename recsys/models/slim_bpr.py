"""SLIM trained with the BPR (or WARP-style) pairwise ranking loss (Parts 9 and 11).

``R~ = R . S`` with ``diag(S) = 0``.  For a sampled triple (u, i, j) with ``i``
a positive and ``j`` a negative item of user ``u``::

    x_uij = r_u . s_i - r_u . s_j        loss = -log sigmoid(x_uij)

Only the weights between ``i`` / ``j`` and the items of the profile of ``u``
receive a gradient, so each step touches ``|profile|`` entries of S per item.
S is stored densely (n_items x n_items, float32) while training and pruned to
the top-K weights per item at the end.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from recsys.config import param
from recsys.models.base import Recommender, register
from recsys.models.pairwise import PAIRWISE_LOSSES, draw_triples
from recsys.models.validation import EarlyStoppingConfig, ValidationMonitor
from recsys.sampling import NEGATIVE_SAMPLING, NegativeSampler
from recsys.utils import gather_profiles

logger = logging.getLogger(__name__)


@dataclass
class SLIMBPRConfig(EarlyStoppingConfig):
    epochs: int = param(20, "Maximum number of passes over the positive interactions.")
    lr: float = param(0.01, "Learning rate (plain SGD).")
    lambda_pos: float = param(1e-4, "L2 regularisation on the weights of the positive item.")
    lambda_neg: float = param(1e-4, "L2 regularisation on the weights of the negative item.")
    batch_size: int = param(1000, "Triples per mini-batch.")
    loss: str = param("bpr", "bpr: random negatives; warp: keep only negatives ranked wrongly.", choices=PAIRWISE_LOSSES)
    warp_trials: int = param(10, "Negatives tried per positive with --loss warp.")
    warp_margin: float = param(0.0, "A negative violates the ranking if score_j + margin >= score_i.")
    negative_sampling: str = param("uniform", "How negative items are drawn.", choices=NEGATIVE_SAMPLING)
    non_negative: bool = param(True, "Clip the similarity weights at zero after every update.")
    topk: int = param(100, "Keep only the K largest weights per item at the end (0 = keep all).")
    max_memory_gb: float = param(8.0, "Refuse to train if the dense similarity matrix needs more memory.")


@register("slim_bpr")
class SLIMBPR(Recommender):
    """SLIM optimised with BPR / WARP-style pairwise ranking."""

    Config = SLIMBPRConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        cfg = self.config
        n = self.n_items
        needed_gb = n * n * 4 / 1e9
        if needed_gb > cfg.max_memory_gb:
            raise MemoryError(
                f"slim_bpr needs a dense {n}x{n} float32 matrix ({needed_gb:.1f} GB) but "
                f"--max-memory-gb is {cfg.max_memory_gb}. Raise the limit or use slim_elasticnet."
            )
        rng = np.random.default_rng(cfg.seed)
        sampler = NegativeSampler(urm, cfg.negative_sampling, cfg.seed)
        # _weights[i, j] = S[j, i]: row i holds the weights s_i used to score item i.
        self._weights = np.zeros((n, n), dtype=np.float32)
        coo = urm.tocoo()
        positive_users, positive_items = coo.row, coo.col
        monitor = ValidationMonitor(self, cfg)
        self._similarity = None
        best_similarity = None  # pruned snapshot of the best epoch (the dense matrix is too big to copy)

        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            order = rng.permutation(len(positive_users))
            total_loss, total_pairs = 0.0, 0
            for start in range(0, len(order), cfg.batch_size):
                batch = order[start : start + cfg.batch_size]
                users, positives, negatives = draw_triples(
                    positive_users[batch], positive_items[batch], sampler, cfg.loss,
                    self._pair_scores, cfg.warp_trials, cfg.warp_margin,
                )
                if len(users) == 0:
                    continue
                total_loss += self._step(users, positives, negatives) * len(users)
                total_pairs += len(users)
            logger.info(
                "SLIM-BPR epoch %d/%d: loss=%.4f pairs=%d (%.1fs)",
                epoch, cfg.epochs, total_loss / max(total_pairs, 1), total_pairs,
                time.perf_counter() - started,
            )
            if monitor.due(epoch, cfg.epochs):
                # Validate the pruned model, i.e. exactly what will be used for recommendations.
                self._similarity = self._prune(self._weights, cfg.topk).T.tocsr()
                if monitor.step(epoch):
                    best_similarity = self._similarity
                if monitor.should_stop:
                    logger.info("SLIM-BPR: early stopping at epoch %d", epoch)
                    break

        if best_similarity is not None:
            self._similarity = best_similarity
            logger.info("SLIM-BPR: restored the best epoch (%d)", monitor.best_epoch)
        else:
            self._similarity = self._prune(self._weights, cfg.topk).T.tocsr()
        self._weights = None
        logger.info("SLIM-BPR: kept %d non-zero weights", self._similarity.nnz)

    def _pair_scores(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Score x_ui = r_u . s_i for aligned (user, item) pairs."""
        segment, profile = gather_profiles(self.urm, users)
        values = self._weights[items[segment], profile]
        return np.bincount(segment, weights=values, minlength=len(users))

    def _step(self, users: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> float:
        """One mini-batch of gradient ascent on sum(log sigmoid(x_uij)); returns the mean loss."""
        cfg = self.config
        n = self.n_items
        flat = self._weights.reshape(-1)  # view on the dense matrix

        segment, profile = gather_profiles(self.urm, users)
        positive_rows = positives[segment].astype(np.int64)
        negative_rows = negatives[segment].astype(np.int64)
        positive_index = positive_rows * n + profile
        negative_index = negative_rows * n + profile
        positive_weights = flat[positive_index]
        negative_weights = flat[negative_index]

        x_uij = np.bincount(segment, weights=positive_weights - negative_weights, minlength=len(users))
        coefficient = expit(-x_uij)[segment].astype(np.float32)  # d(-loss)/dx = sigmoid(-x)

        # The weight s_ii is fixed to zero: profile entries equal to the item itself get no update.
        positive_step = np.where(
            profile != positives[segment], coefficient - cfg.lambda_pos * positive_weights, 0.0
        )
        negative_step = np.where(
            profile != negatives[segment], -coefficient - cfg.lambda_neg * negative_weights, 0.0
        )
        np.add.at(flat, positive_index, (cfg.lr * positive_step).astype(np.float32))
        np.add.at(flat, negative_index, (cfg.lr * negative_step).astype(np.float32))
        if cfg.non_negative:
            flat[positive_index] = np.maximum(flat[positive_index], 0.0)
            flat[negative_index] = np.maximum(flat[negative_index], 0.0)
        return float(np.logaddexp(0.0, -x_uij).mean())

    @staticmethod
    def _prune(weights: np.ndarray, topk: int, block: int = 1000) -> sp.csr_matrix:
        """Sparse copy of the dense weights keeping the ``topk`` largest entries per row."""
        n = weights.shape[0]
        rows, cols, vals = [], [], []
        for start in range(0, n, block):
            chunk = weights[start : start + block]
            if topk and topk < n:
                top = np.argpartition(-np.abs(chunk), topk - 1, axis=1)[:, :topk]
                top_vals = np.take_along_axis(chunk, top, axis=1)
                r = np.repeat(np.arange(start, start + chunk.shape[0]), topk)
                c, v = top.ravel(), top_vals.ravel()
            else:
                r, c = np.nonzero(chunk)
                v = chunk[r, c]
                r = r + start
            nonzero = v != 0
            rows.append(r[nonzero]); cols.append(c[nonzero]); vals.append(v[nonzero])
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        )

    def _score(self, users: np.ndarray) -> np.ndarray:
        return (self.urm[users] @ self._similarity).toarray()

    def item_similarity(self) -> sp.csr_matrix:
        return self._similarity
