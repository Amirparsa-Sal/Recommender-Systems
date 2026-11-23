"""Matrix factorization trained with the BPR (or WARP-style) pairwise loss (Part 11).

``x_uij = x_u . (y_i - y_j)`` for a positive item ``i`` and a negative item
``j`` of user ``u``; the model maximises ``sum log sigmoid(x_uij)`` with
L2 regularisation using mini-batch gradient ascent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from recsys.config import param
from recsys.models.base import Recommender, register
from recsys.models.pairwise import PAIRWISE_LOSSES, draw_triples
from recsys.models.validation import EarlyStoppingConfig, ValidationMonitor
from recsys.optim import OPTIMIZERS, make_accumulators, sparse_update
from recsys.sampling import NEGATIVE_SAMPLING, NegativeSampler

logger = logging.getLogger(__name__)


@dataclass
class BPRMFConfig(EarlyStoppingConfig):
    factors: int = param(64, "Number of latent factors K.")
    epochs: int = param(30, "Maximum number of passes over the positive interactions.")
    lr: float = param(0.05, "Learning rate.")
    optimizer: str = param("adagrad", "Update rule.", choices=OPTIMIZERS)
    lambda_user: float = param(1e-4, "L2 regularisation on the user factors.")
    lambda_pos: float = param(1e-4, "L2 regularisation on the positive item factors.")
    lambda_neg: float = param(1e-4, "L2 regularisation on the negative item factors.")
    batch_size: int = param(1024, "Triples per mini-batch.")
    loss: str = param("bpr", "bpr: random negatives; warp: keep only negatives ranked wrongly.", choices=PAIRWISE_LOSSES)
    warp_trials: int = param(10, "Negatives tried per positive with --loss warp.")
    warp_margin: float = param(0.0, "A negative violates the ranking if score_j + margin >= score_i.")
    negative_sampling: str = param("uniform", "How negative items are drawn.", choices=NEGATIVE_SAMPLING)
    init_std: float = param(0.1, "Standard deviation of the factor initialisation.")


@register("bpr_mf")
class BPRMF(Recommender):
    """Matrix factorization optimised with BPR / WARP-style pairwise ranking."""

    Config = BPRMFConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        self.user_factors = rng.normal(0.0, cfg.init_std, (self.n_users, cfg.factors)).astype(np.float32)
        self.item_factors = rng.normal(0.0, cfg.init_std, (self.n_items, cfg.factors)).astype(np.float32)
        self._accumulators: Dict[str, Optional[np.ndarray]] = make_accumulators(
            {"user_factors": self.user_factors, "item_factors": self.item_factors}, cfg.optimizer
        )
        sampler = NegativeSampler(urm, cfg.negative_sampling, cfg.seed)
        coo = urm.tocoo()
        positive_users, positive_items = coo.row, coo.col
        monitor = ValidationMonitor(self, cfg)
        best_factors = None

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
                "BPR-MF epoch %d/%d: loss=%.4f pairs=%d (%.1fs)",
                epoch, cfg.epochs, total_loss / max(total_pairs, 1), total_pairs,
                time.perf_counter() - started,
            )
            if monitor.due(epoch, cfg.epochs):
                if monitor.step(epoch):
                    best_factors = (self.user_factors.copy(), self.item_factors.copy())
                if monitor.should_stop:
                    logger.info("BPR-MF: early stopping at epoch %d", epoch)
                    break

        if best_factors is not None:
            self.user_factors, self.item_factors = best_factors
            logger.info("BPR-MF: restored the best epoch (%d)", monitor.best_epoch)

    def _pair_scores(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return np.einsum("nk,nk->n", self.user_factors[users], self.item_factors[items])

    def _step(self, users: np.ndarray, positives: np.ndarray, negatives: np.ndarray) -> float:
        """One mini-batch of gradient ascent on sum(log sigmoid(x_uij)); returns the mean loss."""
        cfg = self.config
        x_u = self.user_factors[users]
        y_i = self.item_factors[positives]
        y_j = self.item_factors[negatives]
        x_uij = np.einsum("nk,nk->n", x_u, y_i - y_j)
        coefficient = expit(-x_uij).astype(np.float32)[:, None]  # d(-loss)/dx = sigmoid(-x)

        grad_user = coefficient * (y_i - y_j) - cfg.lambda_user * x_u
        grad_positive = coefficient * x_u - cfg.lambda_pos * y_i
        grad_negative = -coefficient * x_u - cfg.lambda_neg * y_j

        acc = self._accumulators
        sparse_update(self.user_factors, users, grad_user, cfg.lr, acc["user_factors"])
        sparse_update(
            self.item_factors,
            np.concatenate([positives, negatives]),
            np.concatenate([grad_positive, grad_negative]),
            cfg.lr,
            acc["item_factors"],
        )
        return float(np.logaddexp(0.0, -x_uij).mean())

    def _score(self, users: np.ndarray) -> np.ndarray:
        return self.user_factors[users] @ self.item_factors.T
