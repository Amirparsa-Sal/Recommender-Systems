"""Neighbourhood collaborative filtering: item-based and user-based KNN (Part 8)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from recsys.config import BaseConfig, param
from recsys.models.base import Recommender, register
from recsys.similarity import SIMILARITIES, compute_similarity

logger = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass
class KNNConfig(BaseConfig):
    similarity: str = param("cosine", "Similarity measure.", choices=SIMILARITIES)
    topk: int = param(100, "Neighbours kept per row of the similarity matrix (0 = keep all).")
    shrink: float = param(10.0, "Shrink term penalising similarities with low support.")
    tversky_alpha: float = param(1.0, "Tversky alpha (only for --similarity tversky).")
    tversky_beta: float = param(1.0, "Tversky beta (only for --similarity tversky).")
    normalize: bool = param(
        False,
        "Divide the score by the sum of the neighbours' similarities, as in the slides' "
        "weighted-average formula.",
    )
    block_size: int = param(0, "Rows per dense block when computing similarities (0 = automatic).")


def _similarity_kwargs(config: KNNConfig) -> dict:
    return dict(
        kind=config.similarity,
        topk=config.topk,
        shrink=config.shrink,
        tversky_alpha=config.tversky_alpha,
        tversky_beta=config.tversky_beta,
        block_size=config.block_size,
    )


@register("item_knn")
class ItemKNN(Recommender):
    """Item-based CF: ``R~ = R . S`` with S the (K-nearest-neighbour) item-item similarity."""

    Config = KNNConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        # Row i of W holds the K items most similar to i.
        weights = compute_similarity(urm, **_similarity_kwargs(self.config))
        self._similarity = weights.T.tocsr()  # R @ S: column i collects the neighbours of i
        self._column_scale = None
        if self.config.normalize:
            neighbour_sums = np.asarray(weights.sum(axis=1)).ravel()
            self._column_scale = (1.0 / (neighbour_sums + _EPS)).astype(np.float32)

    def _score(self, users: np.ndarray) -> np.ndarray:
        scores = (self.urm[users] @ self._similarity).toarray()
        if self._column_scale is not None:
            scores *= self._column_scale[None, :]
        return scores

    def item_similarity(self) -> sp.csr_matrix:
        if self._column_scale is None:
            return self._similarity
        return (self._similarity @ sp.diags(self._column_scale)).tocsr()


@register("user_knn")
class UserKNN(Recommender):
    """User-based CF: ``R~ = S . R`` with S the (K-nearest-neighbour) user-user similarity."""

    Config = KNNConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        self._similarity = compute_similarity(urm.T.tocsr(), **_similarity_kwargs(self.config))
        self._row_scale = None
        if self.config.normalize:
            neighbour_sums = np.asarray(self._similarity.sum(axis=1)).ravel()
            self._row_scale = (1.0 / (neighbour_sums + _EPS)).astype(np.float32)

    def _score(self, users: np.ndarray) -> np.ndarray:
        scores = (self._similarity[users] @ self.urm).toarray()
        if self._row_scale is not None:
            scores *= self._row_scale[users][:, None]
        return scores
