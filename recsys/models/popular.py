"""Non-personalized Top Popular recommender (Part 4 of the slides)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from recsys.config import BaseConfig
from recsys.models.base import Recommender, register


@dataclass
class TopPopularConfig(BaseConfig):
    """Top Popular has no hyper-parameters."""


@register("top_popular")
class TopPopular(Recommender):
    """Recommend the items with the largest number of interactions to everybody."""

    Config = TopPopularConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        # The popularity vector is computed in Recommender.fit.
        pass

    def _score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.item_popularity, (len(users), 1))
