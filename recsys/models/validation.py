"""Validation-based early stopping for the iterative models (BPR-MF, SLIM-BPR, Funk SVD, Asymmetric SVD).

The models receive a validation set from :meth:`Recommender.fit` (it is carved
out of ``data_train.csv`` by the evaluation split).  A :class:`ValidationMonitor`
scores the current model on it every ``eval_every`` epochs; the model keeps a
snapshot of the best epoch, stops after ``patience`` evaluations without
improvement and restores that snapshot.

Without a validation set (for example in ``main.py submit``) the monitor is
disabled and the model simply trains for ``--epochs`` epochs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from recsys.config import BaseConfig, param
from recsys.metrics import evaluate_recommendations

if TYPE_CHECKING:  # pragma: no cover
    from recsys.models.base import Recommender

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig(BaseConfig):
    """Parameters shared by the models that support validation-based early stopping."""

    eval_every: int = param(
        5, "Evaluate on the validation set every N epochs (0 disables validation and early stopping)."
    )
    patience: int = param(
        3,
        "Stop after this many evaluations without improvement "
        "(0 = never stop early, but still restore the best epoch).",
    )
    validation_k: int = param(20, "List length of the NDCG used for validation.")
    validation_users: int = param(
        5000, "Validation users sampled for each evaluation to save time (0 = all)."
    )


class ValidationMonitor:
    """Tracks the validation NDCG of a model that is being trained.

    Usage inside ``_fit``::

        monitor = ValidationMonitor(self, cfg)
        for epoch in range(1, cfg.epochs + 1):
            ...train one epoch...
            if monitor.due(epoch, cfg.epochs):
                if monitor.step(epoch):
                    snapshot = ...copy of the current parameters...
                if monitor.should_stop:
                    break
    """

    def __init__(self, model: "Recommender", config: EarlyStoppingConfig) -> None:
        self.model = model
        self.config = config
        self.best_score = -np.inf
        self.best_epoch = 0
        self._evaluations_without_improvement = 0
        self.users = np.empty(0, dtype=np.int64)

        validation = model.validation
        if config.eval_every <= 0:
            self.enabled = False
            logger.info("Validation disabled (--eval-every 0): training for the full number of epochs")
        elif validation is None or validation.nnz == 0:
            self.enabled = False
            logger.info("No validation set given: early stopping disabled, training for the full number of epochs")
        else:
            self.enabled = True
            users = np.flatnonzero(np.diff(validation.indptr) > 0)
            if 0 < config.validation_users < len(users):
                rng = np.random.default_rng(config.seed)
                users = np.sort(rng.choice(users, size=config.validation_users, replace=False))
            self.users = users
            logger.info(
                "Early stopping on validation NDCG@%d (%d users, every %d epochs, patience %d)",
                config.validation_k, len(users), config.eval_every, config.patience,
            )

    def reset(self) -> None:
        """Forget the best score and the patience counter (e.g. when a new factor starts)."""
        self.best_score = -np.inf
        self.best_epoch = 0
        self._evaluations_without_improvement = 0

    def due(self, epoch: int, last_epoch: int) -> bool:
        """True when the model must be evaluated after ``epoch`` (the last epoch always is)."""
        return self.enabled and (epoch % self.config.eval_every == 0 or epoch == last_epoch)

    def step(self, epoch: int) -> bool:
        """Evaluate the current model; returns True if it is the best one so far."""
        k = self.config.validation_k
        recommendations = self.model.recommend(self.users, k=k)
        score = evaluate_recommendations(
            recommendations, self.model.validation, self.users, [k]
        )[f"ndcg@{k}"]
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch = score, epoch
            self._evaluations_without_improvement = 0
        else:
            self._evaluations_without_improvement += 1
        logger.info(
            "epoch %d: validation ndcg@%d=%.4f (best %.4f at epoch %d)",
            epoch, k, score, self.best_score, self.best_epoch,
        )
        return improved

    @property
    def should_stop(self) -> bool:
        patience = self.config.patience
        return patience > 0 and self._evaluations_without_improvement >= patience
