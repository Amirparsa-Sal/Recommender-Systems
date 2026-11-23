"""Asymmetric SVD (Part 10): user factors are a function of the items the user consumed.

``r~_ui = (sum_{j in N(u)} z_j) . y_i / sqrt(|N(u)|)`` -- equivalently
``R~ = R . Z . Y`` -- so the model has no per-user parameters and can score
users that were not seen during training.  The product ``Z.Y`` is a (non
symmetric) low-rank item-item similarity matrix.

Training minimises the squared error on the positive interactions plus
sampled zeros.  For a positive pair (u, i) the item i itself is removed from
the profile used to predict it; otherwise the model could trivially learn
``Z.Y = I``.

With a validation set (``evaluate``) the model is validated every ``--eval-every`` epochs, stops
after ``--patience`` evaluations without improvement and restores the best epoch.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

from recsys.config import param
from recsys.models.base import Recommender, register
from recsys.optim import OPTIMIZERS, make_accumulators, sparse_update
from recsys.sampling import NEGATIVE_SAMPLING, NegativeSampler, pointwise_samples
from recsys.models.validation import EarlyStoppingConfig, ValidationMonitor

logger = logging.getLogger(__name__)


@dataclass
class AsymmetricSVDConfig(EarlyStoppingConfig):
    factors: int = param(32, "Number of latent factors K.")
    epochs: int = param(20, "Maximum number of training epochs.")
    lr: float = param(0.05, "Learning rate.")
    optimizer: str = param("adagrad", "Update rule.", choices=OPTIMIZERS)
    reg_z: float = param(1e-3, "L2 regularisation on the profile (Z) factors.")
    reg_y: float = param(1e-3, "L2 regularisation on the item (Y) factors.")
    reg_bias: float = param(1e-3, "L2 regularisation on the biases (--global-effects).")
    n_negatives: int = param(3, "Sampled zeros per positive interaction.")
    negative_sampling: str = param("uniform", "How the zeros are drawn.", choices=NEGATIVE_SAMPLING)
    normalize: bool = param(True, "Divide the profile sum by sqrt(profile size).")
    global_effects: bool = param(False, "Learn global mean, user and item biases.")
    batch_size: int = param(1024, "Samples per mini-batch.")
    init_std: float = param(0.05, "Standard deviation of the factor initialisation.")


@register("asymmetric_svd")
class AsymmetricSVD(Recommender):
    """Asymmetric SVD: ``R~ = R . Z . Y`` trained with MSE and negative sampling."""

    Config = AsymmetricSVDConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        k = cfg.factors
        self.profile_factors = rng.normal(0.0, cfg.init_std, (self.n_items, k)).astype(np.float32)  # Z
        self.item_factors = rng.normal(0.0, cfg.init_std, (self.n_items, k)).astype(np.float32)  # Y
        self.user_bias = np.zeros(self.n_users, dtype=np.float32)
        self.item_bias = np.zeros(self.n_items, dtype=np.float32)
        self._accumulators: Dict[str, Optional[np.ndarray]] = make_accumulators(
            {
                "profile_factors": self.profile_factors, "item_factors": self.item_factors,
                "user_bias": self.user_bias, "item_bias": self.item_bias,
            },
            cfg.optimizer,
        )
        sampler = NegativeSampler(urm, cfg.negative_sampling, cfg.seed)
        self._global_mean = 1.0 / (1 + cfg.n_negatives)
        coo = urm.tocoo()
        monitor = ValidationMonitor(self, cfg)
        best_state = None

        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            users, items, labels = pointwise_samples(coo.row, coo.col, sampler, cfg.n_negatives)
            order = rng.permutation(len(users))
            total = 0.0
            for start in range(0, len(order), cfg.batch_size):
                batch = order[start : start + cfg.batch_size]
                total += self._step(users[batch], items[batch], labels[batch]) * len(batch)
            logger.info(
                "AsymmetricSVD epoch %d/%d: mse=%.5f (%.1fs)",
                epoch, cfg.epochs, total / len(order), time.perf_counter() - started,
            )
            if monitor.due(epoch, cfg.epochs):
                if monitor.step(epoch):
                    best_state = (self.profile_factors.copy(), self.item_factors.copy(),
                                  self.user_bias.copy(), self.item_bias.copy())
                if monitor.should_stop:
                    logger.info("AsymmetricSVD: early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            (self.profile_factors, self.item_factors, self.user_bias, self.item_bias) = best_state
            logger.info("AsymmetricSVD: restored the best epoch (%d)", monitor.best_epoch)

    def _step(self, users: np.ndarray, items: np.ndarray, labels: np.ndarray) -> float:
        """One SGD step on a mini-batch; returns the mean squared error before the update."""
        cfg = self.config
        profiles = self.urm[users]
        # Compact column space: only the items that appear in the profiles of this batch.
        touched, inverse = np.unique(profiles.indices, return_inverse=True)
        compact = sp.csr_matrix(
            (profiles.data, inverse.ravel(), profiles.indptr), shape=(len(users), len(touched))
        )
        z_touched = self.profile_factors[touched]
        is_positive = labels > 0

        # Profile sum without the target item (it is part of the profile for positives).
        profile_sum = compact @ z_touched
        profile_sum[is_positive] -= self.profile_factors[items[is_positive]]
        effective_size = np.diff(profiles.indptr) - is_positive
        scale = (
            1.0 / np.sqrt(np.maximum(effective_size, 1)) if cfg.normalize
            else np.ones(len(users))
        ).astype(np.float32)
        x_u = profile_sum * scale[:, None]

        y_i = self.item_factors[items]
        prediction = (x_u * y_i).sum(axis=1)
        if cfg.global_effects:
            prediction += self._global_mean + self.user_bias[users] + self.item_bias[items]
        error = labels - prediction

        grad_x = error[:, None] * y_i * scale[:, None]  # direction for every profile member
        grad_z = compact.T @ grad_x
        # Remove the contribution that would flow to the (excluded) target item itself.
        np.subtract.at(
            grad_z, np.searchsorted(touched, items[is_positive]), grad_x[is_positive]
        )
        grad_z -= cfg.reg_z * z_touched
        grad_y = error[:, None] * x_u - cfg.reg_y * y_i

        acc = self._accumulators
        sparse_update(self.profile_factors, touched, grad_z, cfg.lr, acc["profile_factors"])
        sparse_update(self.item_factors, items, grad_y, cfg.lr, acc["item_factors"])
        if cfg.global_effects:
            sparse_update(self.user_bias, users, error - cfg.reg_bias * self.user_bias[users], cfg.lr, acc["user_bias"])
            sparse_update(self.item_bias, items, error - cfg.reg_bias * self.item_bias[items], cfg.lr, acc["item_bias"])
        return float(np.mean(error ** 2))

    def _score(self, users: np.ndarray) -> np.ndarray:
        profile_sum = self.urm[users] @ self.profile_factors
        if self.config.normalize:
            sizes = np.maximum(self._profile_sizes[users], 1)
            profile_sum = profile_sum / np.sqrt(sizes)[:, None]
        scores = np.asarray(profile_sum) @ self.item_factors.T
        if self.config.global_effects:
            scores += self.item_bias[None, :]
        return scores
