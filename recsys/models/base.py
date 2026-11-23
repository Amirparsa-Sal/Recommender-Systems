"""Base class and registry shared by every recommender."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Optional, Tuple, Type

import numpy as np
import scipy.sparse as sp

from recsys.config import BaseConfig, config_from_dict

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type["Recommender"]] = {}


def register(name: str):
    """Class decorator adding a model to :data:`MODEL_REGISTRY` under ``name``."""

    def decorator(cls: Type["Recommender"]) -> Type["Recommender"]:
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Type["Recommender"]:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}; available models: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def create_model(name: str, params: Optional[dict] = None) -> "Recommender":
    """Instantiate a registered model from a dict of hyper-parameters."""
    return get_model_class(name)(**(params or {}))


class Recommender(ABC):
    """A Top-N recommender trained on a binary URM.

    Subclasses define a ``Config`` dataclass, implement :meth:`_fit` and
    :meth:`_score`, and are registered with :func:`register`.  The public
    :meth:`recommend` takes care of masking seen items, breaking ties by
    popularity, handling users without history and selecting the top-N.
    """

    name: ClassVar[str] = ""
    Config: ClassVar[Type[BaseConfig]] = BaseConfig

    def __init__(self, config: Optional[BaseConfig] = None, **params) -> None:
        if config is not None and params:
            raise ValueError("Pass either a config object or keyword parameters, not both")
        self.config = config if config is not None else config_from_dict(self.Config, params)
        self.urm: Optional[sp.csr_matrix] = None
        self.seen: Optional[sp.csr_matrix] = None
        self.validation: Optional[sp.csr_matrix] = None

    # ------------------------------------------------------------------ training
    def fit(
        self,
        urm: sp.spmatrix,
        validation: Optional[sp.spmatrix] = None,
        seen: Optional[sp.spmatrix] = None,
    ) -> "Recommender":
        """Train the model on a (users x items) binary matrix.

        Args:
            urm: training interactions.
            validation: optional held-out interactions of the same shape. Only the
                iterative models with early stopping (``bpr_mf``, ``slim_bpr``) use
                it, to pick the number of epochs; the other models ignore it.
            seen: interactions that :meth:`recommend` must not suggest again. Defaults to
                ``urm``; the pipelining hybrid passes the original interactions when it
                trains a model on an enlarged URM.
        """
        urm = sp.csr_matrix(urm, dtype=np.float32)
        urm.sort_indices()
        self.urm = urm
        self.seen = urm if seen is None else sp.csr_matrix(seen, dtype=np.float32)
        self.validation = None if validation is None else sp.csr_matrix(validation, dtype=np.float32)
        self.n_users, self.n_items = urm.shape
        self._profile_sizes = np.diff(urm.indptr)
        popularity = np.asarray(urm.sum(axis=0)).ravel().astype(np.float32)
        self.item_popularity = popularity
        self._popularity_norm = popularity / max(float(popularity.max()), 1.0)
        self._fit(urm)
        return self

    @abstractmethod
    def _fit(self, urm: sp.csr_matrix) -> None:
        """Learn the model parameters from ``urm``."""

    # ----------------------------------------------------------------- inference
    @abstractmethod
    def _score(self, users: np.ndarray) -> np.ndarray:
        """Return a dense (len(users), n_items) array of item scores."""

    def item_similarity(self) -> sp.csr_matrix:
        """The (n_items x n_items) matrix ``S`` such that the scores are ``R . S``.

        Only available for item-item models (``item_knn``, ``slim_*``, ``p3alpha``,
        ``rp3beta`` and the ``hybrid_merge`` of those); it is what the merging
        hybrid combines.
        """
        raise NotImplementedError(
            f"'{self.name}' is not an item-item similarity model (scores are not R . S), "
            "so it cannot be merged"
        )

    def recommend(
        self,
        users: np.ndarray,
        k: int = 20,
        batch_size: int = 1000,
        exclude_seen: bool = True,
        popularity_tiebreak: bool = True,
    ) -> np.ndarray:
        """Top-``k`` item indices for every user, best first.

        Args:
            users: row indices into the training URM.
            exclude_seen: never recommend items already in the user's training profile.
            popularity_tiebreak: among items with (almost) equal score prefer the
                most popular one; users without history get plain popularity.
        """
        items, _ = self.recommend_with_scores(users, k, batch_size, exclude_seen, popularity_tiebreak)
        return items

    def recommend_with_scores(
        self,
        users: np.ndarray,
        k: int = 20,
        batch_size: int = 1000,
        exclude_seen: bool = True,
        popularity_tiebreak: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Like :meth:`recommend` but also returns the scores of the recommended items.

        The scores include the (negligible) popularity tie-break and are ``-inf`` for
        the rare case of fewer than ``k`` unseen items.
        """
        if self.urm is None:
            raise RuntimeError("Call fit() before recommend()")
        if k > self.n_items:
            raise ValueError(f"k={k} is larger than the number of items ({self.n_items})")
        users = np.asarray(users)
        recommendations = np.empty((len(users), k), dtype=np.int32)
        top_scores = np.empty((len(users), k), dtype=np.float32)

        for start in range(0, len(users), batch_size):
            batch = users[start : start + batch_size]
            scores = np.array(self._score(batch), dtype=np.float32)  # private writable copy

            cold = self._profile_sizes[batch] == 0
            if cold.any():
                scores[cold] = self.item_popularity
            if popularity_tiebreak:
                spread = scores.max(axis=1) - scores.min(axis=1)
                spread[spread == 0] = 1.0
                scores += (1e-5 * spread)[:, None] * self._popularity_norm[None, :]
            if exclude_seen:
                seen = self.seen[batch].tocoo()
                scores[seen.row, seen.col] = -np.inf

            candidates = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            candidate_scores = np.take_along_axis(scores, candidates, axis=1)
            order = np.argsort(-candidate_scores, axis=1)
            stop = start + len(batch)
            recommendations[start:stop] = np.take_along_axis(candidates, order, axis=1)
            top_scores[start:stop] = np.take_along_axis(candidate_scores, order, axis=1)
        return recommendations, top_scores
