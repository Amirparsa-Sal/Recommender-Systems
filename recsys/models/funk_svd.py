"""Funk SVD / SVD++-style matrix factorization trained with mini-batch SGD (Part 10).

``r~_ui = x_u . y_i`` (plus ``mu + b_u + b_i`` with ``global_effects``) minimising
the squared error.  Variants controlled by flags:

* ``missing=random``: the error is minimised on the observed interactions only
  (missing-as-random).  With binary data every target is 1, so this variant is
  nearly degenerate and is mostly useful as a baseline to compare against.
* ``missing=negative_sampling``: sampled unobserved pairs are added as zeros
  (approximation of missing-as-negative, the "sample zero ratings" fix).
* ``incremental``: factors are learned one at a time, as in the original Funk SVD.
* ``global_effects``: learn global mean, user bias and item bias (the slides' "SVD++").

With a validation set (``evaluate``) the model stops early, see :mod:`recsys.models.validation`.
Without ``--incremental`` the validation runs every ``--eval-every`` epochs and the best epoch is
restored.  With ``--incremental`` there are two levels: *inside* every factor the usual
``--eval-every`` / ``--patience`` rule picks the best epoch of that factor (a factor is frozen at
its best epoch, not at ``--epochs``), and *across* factors the model stops adding factors after
``--factor-patience`` factors in a row without improving the best validation score, keeping only
the factors up to the best one.
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

MISSING_ASSUMPTIONS = ("random", "negative_sampling")


@dataclass
class FunkSVDConfig(EarlyStoppingConfig):
    factors: int = param(32, "Number of latent factors K.")
    epochs: int = param(20, "Maximum epochs (per factor when --incremental).")
    lr: float = param(0.05, "Learning rate.")
    optimizer: str = param("adagrad", "Update rule.", choices=OPTIMIZERS)
    reg_user: float = param(1e-3, "L2 regularisation on user factors.")
    reg_item: float = param(1e-3, "L2 regularisation on item factors.")
    reg_bias: float = param(1e-3, "L2 regularisation on the biases (--global-effects).")
    missing: str = param(
        "negative_sampling", "Treatment of missing interactions.", choices=MISSING_ASSUMPTIONS
    )
    n_negatives: int = param(3, "Sampled zeros per positive with --missing negative_sampling.")
    negative_sampling: str = param("uniform", "How the zeros are drawn.", choices=NEGATIVE_SAMPLING)
    incremental: bool = param(False, "Learn the factors one at a time (original Funk SVD).")
    factor_patience: int = param(
        3,
        "Only with --incremental: stop adding factors after this many factors in a row that do "
        "not improve the best validation NDCG (0 = never stop adding factors, but still keep the "
        "best number of factors). --eval-every / --patience then act on the epochs of each factor.",
    )
    global_effects: bool = param(False, "Learn global mean, user and item biases (SVD++ in the slides).")
    batch_size: int = param(2048, "Samples per mini-batch.")
    init_std: float = param(0.1, "Standard deviation of the factor initialisation.")


@register("funk_svd")
class FunkSVD(Recommender):
    """Funk SVD (MSE, SGD) with optional negative sampling, incremental factors and biases."""

    Config = FunkSVDConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        k = cfg.factors
        self.user_factors = rng.normal(0.0, cfg.init_std, (self.n_users, k)).astype(np.float32)
        self.item_factors = rng.normal(0.0, cfg.init_std, (self.n_items, k)).astype(np.float32)
        self.user_bias = np.zeros(self.n_users, dtype=np.float32)
        self.item_bias = np.zeros(self.n_items, dtype=np.float32)
        self._accumulators: Dict[str, Optional[np.ndarray]] = make_accumulators(
            {
                "user_factors": self.user_factors, "item_factors": self.item_factors,
                "user_bias": self.user_bias, "item_bias": self.item_bias,
            },
            cfg.optimizer,
        )
        sampler = None
        if cfg.missing == "negative_sampling":
            sampler = NegativeSampler(urm, cfg.negative_sampling, cfg.seed)
        n_labels = 1 + (cfg.n_negatives if sampler is not None else 0)
        self._global_mean = 1.0 / n_labels  # mean of the training labels

        coo = urm.tocoo()
        monitor = ValidationMonitor(self, cfg)
        best_state = None  # best epoch (plain mode) or best number of factors (incremental)
        best_score, best_n_factors, factors_without_gain = -np.inf, 0, 0  # incremental bookkeeping
        self._score_mask = None  # incremental mode: factors that are already trained
        phases = range(k) if cfg.incremental else [None]
        for phase in phases:
            factor_state = None  # best epoch of the factor that is being trained
            if phase is None:
                predict_mask = update_mask = np.ones(k, dtype=np.float32)
            else:  # factors < phase are frozen, factor `phase` is trained
                predict_mask = (np.arange(k) <= phase).astype(np.float32)
                update_mask = (np.arange(k) == phase).astype(np.float32)
                self._score_mask = predict_mask  # untrained factors must not take part in validation
                monitor.reset()  # every factor gets its own best epoch and patience
            for epoch in range(1, cfg.epochs + 1):
                started = time.perf_counter()
                users, items, labels = pointwise_samples(coo.row, coo.col, sampler, cfg.n_negatives)
                order = rng.permutation(len(users))
                total = 0.0
                for start in range(0, len(order), cfg.batch_size):
                    batch = order[start : start + cfg.batch_size]
                    total += self._step(users[batch], items[batch], labels[batch], predict_mask, update_mask) * len(batch)
                logger.info(
                    "FunkSVD%s epoch %d/%d: mse=%.5f (%.1fs)",
                    "" if phase is None else f" factor {phase + 1}/{k}",
                    epoch, cfg.epochs, total / len(order), time.perf_counter() - started,
                )
                if monitor.due(epoch, cfg.epochs):
                    if monitor.step(epoch):
                        if phase is None:
                            best_state = self._snapshot(None)
                        else:
                            factor_state = self._snapshot_factor(phase)
                    if monitor.should_stop:
                        logger.info("FunkSVD: early stopping%s at epoch %d",
                                    "" if phase is None else f" of factor {phase + 1}", epoch)
                        break

            if phase is not None and monitor.enabled:
                # Freeze the factor at its best epoch, then decide whether more factors are worth it.
                self._restore_factor(factor_state, phase)
                improved = monitor.best_score > best_score
                if improved:
                    best_score, best_n_factors, factors_without_gain = monitor.best_score, phase + 1, 0
                    best_state = self._snapshot(best_n_factors)
                else:
                    factors_without_gain += 1
                logger.info(
                    "FunkSVD factor %d/%d: best validation ndcg=%.4f at epoch %d (%s)",
                    phase + 1, k, monitor.best_score, monitor.best_epoch,
                    "new overall best" if improved else f"overall best {best_score:.4f} with {best_n_factors} factors",
                )
                if cfg.factor_patience > 0 and factors_without_gain >= cfg.factor_patience:
                    logger.info("FunkSVD: no improvement for %d factors, stopping after factor %d/%d",
                                factors_without_gain, phase + 1, k)
                    break

        self._score_mask = None
        if best_state is not None:
            self._restore(best_state)
            logger.info("FunkSVD: restored the best %s (%d)",
                        "number of factors" if cfg.incremental else "epoch",
                        best_n_factors if cfg.incremental else monitor.best_epoch)

    def _snapshot(self, n_factors: Optional[int]) -> Dict[str, object]:
        """Copy of the parameters; ``n_factors`` = number of trained factors (None = all)."""
        return {
            "user_factors": self.user_factors.copy(), "item_factors": self.item_factors.copy(),
            "user_bias": self.user_bias.copy(), "item_bias": self.item_bias.copy(),
            "n_factors": n_factors,
        }

    def _snapshot_factor(self, phase: int) -> Dict[str, np.ndarray]:
        """Copy of what the training of factor ``phase`` changes (its columns and the biases)."""
        return {
            "user_column": self.user_factors[:, phase].copy(), "item_column": self.item_factors[:, phase].copy(),
            "user_bias": self.user_bias.copy(), "item_bias": self.item_bias.copy(),
        }

    def _restore_factor(self, state: Dict[str, np.ndarray], phase: int) -> None:
        self.user_factors[:, phase] = state["user_column"]
        self.item_factors[:, phase] = state["item_column"]
        self.user_bias, self.item_bias = state["user_bias"], state["item_bias"]

    def _restore(self, state: Dict[str, object]) -> None:
        self.user_factors, self.item_factors = state["user_factors"], state["item_factors"]
        self.user_bias, self.item_bias = state["user_bias"], state["item_bias"]
        if state["n_factors"] is not None:  # factors that were not trained yet stay at zero
            self.user_factors[:, state["n_factors"]:] = 0.0
            self.item_factors[:, state["n_factors"]:] = 0.0

    def _step(
        self,
        users: np.ndarray,
        items: np.ndarray,
        labels: np.ndarray,
        predict_mask: np.ndarray,
        update_mask: np.ndarray,
    ) -> float:
        """One SGD step on a mini-batch; returns the mean squared error before the update."""
        cfg = self.config
        x_u = self.user_factors[users]
        y_i = self.item_factors[items]
        prediction = (x_u * y_i * predict_mask).sum(axis=1)
        if cfg.global_effects:
            prediction += self._global_mean + self.user_bias[users] + self.item_bias[items]
        error = labels - prediction

        grad_x = (error[:, None] * y_i - cfg.reg_user * x_u) * update_mask
        grad_y = (error[:, None] * x_u - cfg.reg_item * y_i) * update_mask
        acc = self._accumulators
        sparse_update(self.user_factors, users, grad_x, cfg.lr, acc["user_factors"])
        sparse_update(self.item_factors, items, grad_y, cfg.lr, acc["item_factors"])
        if cfg.global_effects:
            sparse_update(self.user_bias, users, error - cfg.reg_bias * self.user_bias[users], cfg.lr, acc["user_bias"])
            sparse_update(self.item_bias, items, error - cfg.reg_bias * self.item_bias[items], cfg.lr, acc["item_bias"])
        return float(np.mean(error ** 2))

    def _score(self, users: np.ndarray) -> np.ndarray:
        user_factors = self.user_factors[users]
        if self._score_mask is not None:
            user_factors = user_factors * self._score_mask
        scores = user_factors @ self.item_factors.T
        if self.config.global_effects:
            scores += self.item_bias[None, :]  # mu and b_u do not change the ranking of a user
        return scores
