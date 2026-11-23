"""Graph-based recommenders on the bipartite user-item graph (Part 14 of the slides).

The graph has the users and the items as nodes and one undirected edge per
interaction.  With ``R`` the URM, ``D_U`` / ``D_I`` the user / item degrees and
the block structure ``P = [[0, P_UI], [P_IU, 0]]`` of the transition matrix:

* ``P_UI = diag(1 / D_U) . R``      (user -> item transition probabilities)
* ``P_IU = diag(1 / D_I) . R^T``    (item -> user transition probabilities)

Models:

* ``random_walk``: state probabilities after ``h`` hops, ``Pi^(h) = Pi^(h-1) . P``,
  starting from the user (``U -> I -> U -> I`` for h = 3).
* ``rwr``: random walk with restart (personalised PageRank),
  ``Pi = gamma Pi P + (1 - gamma) Pi^(0)``, solved by power iteration.
* ``p3alpha``: the 3-hop walk collapsed into an item-item similarity
  ``S = P_IU . P_UI`` whose transition probabilities are raised to the power
  ``alpha`` (plain P3 for ``alpha = 1``).
* ``rp3beta``: P3-alpha with the similarity of every destination item ``j``
  divided by ``d_j ** beta`` to reduce the popularity bias.

The attribute (tripartite) graphs of the slides need side information and are
not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from recsys.config import BaseConfig, param
from recsys.models.base import Recommender, register
from recsys.similarity import blockwise_topk


# ------------------------------------------------------------------------- helpers
def _inverse_power(degrees: np.ndarray, power: float = 1.0) -> np.ndarray:
    """``degrees ** -power`` with 0 for nodes without edges."""
    result = np.zeros(len(degrees), dtype=np.float64)
    positive = degrees > 0
    result[positive] = np.power(degrees[positive], -power)
    return result


def transition_matrices(urm: sp.csr_matrix) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return ``(P_UI, P_IU)``: row-normalised user->item and item->user transitions."""
    user_degree = np.asarray(urm.sum(axis=1)).ravel()
    item_degree = np.asarray(urm.sum(axis=0)).ravel()
    p_ui = (sp.diags(_inverse_power(user_degree)) @ urm).tocsr().astype(np.float32)
    p_iu = (sp.diags(_inverse_power(item_degree)) @ urm.T).tocsr().astype(np.float32)
    return p_ui, p_iu


def _dense_times_sparse(dense: np.ndarray, sparse_transposed: sp.csr_matrix) -> np.ndarray:
    """Compute ``dense @ M`` given ``M^T`` (sparse @ dense is the fast scipy product)."""
    return np.ascontiguousarray((sparse_transposed @ dense.T).T)


# ------------------------------------------------------------------ h-hop random walk
@dataclass
class RandomWalkConfig(BaseConfig):
    hops: int = param(
        3, "Number of hops of the walk, odd and at least 3 (the slides: 3, sometimes 5, rarely 7)."
    )

    def validate(self) -> None:
        super().validate()
        if self.hops < 3 or self.hops % 2 == 0:
            raise ValueError(
                f"hops must be odd and >= 3 so that the walk ends on an item node, got {self.hops}"
            )


@register("random_walk")
class RandomWalk(Recommender):
    """Random walk of h hops from the user on the user-item graph (state probabilities)."""

    Config = RandomWalkConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        self._p_ui, self._p_iu = transition_matrices(urm)
        self._p_ui_t = self._p_ui.T.tocsr()
        self._p_iu_t = self._p_iu.T.tocsr()

    def _score(self, users: np.ndarray) -> np.ndarray:
        # Pi^(1) on the items: the one-hot user state times P_UI is the user's row of P_UI.
        item_state = self._p_ui[users].toarray()
        for _ in range((self.config.hops - 1) // 2):
            user_state = _dense_times_sparse(item_state, self._p_iu_t)   # I -> U
            item_state = _dense_times_sparse(user_state, self._p_ui_t)   # U -> I
        return item_state


# ------------------------------------------------------- random walk with restart
@dataclass
class RWRConfig(BaseConfig):
    damping: float = param(
        0.7,
        "Probability gamma of following an edge; 1 - gamma is the restart probability "
        "(smaller = more personalised).",
    )
    iterations: int = param(10, "Maximum number of power iterations.")
    tolerance: float = param(
        1e-4, "Stop early when the state changes by less than this (largest L1 change over users)."
    )

    def validate(self) -> None:
        super().validate()
        if not 0.0 < self.damping < 1.0:
            raise ValueError(f"damping must be in (0, 1), got {self.damping}")


@register("rwr")
class RandomWalkWithRestart(Recommender):
    """Random walk with restart (personalised PageRank) from the user, solved by power iteration."""

    Config = RWRConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        p_ui, p_iu = transition_matrices(urm)
        self._p_ui_t = p_ui.T.tocsr()
        self._p_iu_t = p_iu.T.tocsr()

    def _score(self, users: np.ndarray) -> np.ndarray:
        cfg = self.config
        gamma = np.float32(cfg.damping)
        restart = np.zeros((len(users), self.n_users), dtype=np.float32)
        restart[np.arange(len(users)), users] = 1.0

        user_state = restart.copy()  # Pi^(0): all the probability on the starting user
        item_state = np.zeros((len(users), self.n_items), dtype=np.float32)
        for _ in range(cfg.iterations):
            # Pi <- gamma * Pi . P + (1 - gamma) * Pi^(0); P only connects users and items.
            new_item_state = gamma * _dense_times_sparse(user_state, self._p_ui_t)
            new_user_state = (
                gamma * _dense_times_sparse(item_state, self._p_iu_t) + (1.0 - gamma) * restart
            )
            change = max(
                np.abs(new_item_state - item_state).sum(axis=1).max(),
                np.abs(new_user_state - user_state).sum(axis=1).max(),
            )
            item_state, user_state = new_item_state, new_user_state
            if change < cfg.tolerance:
                break
        return item_state


# ------------------------------------------------------------------ P3-alpha, RP3-beta
@dataclass
class P3AlphaConfig(BaseConfig):
    alpha: float = param(
        1.0, "Power applied to the transition probabilities (1 = plain P3; larger = more popularity bias)."
    )
    topk: int = param(100, "Destinations kept per source item in the similarity matrix (0 = keep all).")
    block_size: int = param(0, "Rows per dense block when computing the similarity (0 = automatic).")

    def validate(self) -> None:
        super().validate()
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")


@dataclass
class RP3BetaConfig(P3AlphaConfig):
    beta: float = param(
        0.5, "Popularity penalty: similarities are divided by (destination item degree) ** beta."
    )

    def validate(self) -> None:
        super().validate()
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")


def p3_similarity(
    urm: sp.csr_matrix, alpha: float, beta: float, topk: int, block_size: int = 0
) -> sp.csr_matrix:
    """Item-item similarity ``S_ij = d_j^-beta * sum_u (P_IU[i,u] * P_UI[u,j]) ** alpha``.

    Row ``i`` is the source item and column ``j`` the destination; every row is pruned
    to its ``topk`` largest entries.
    """
    p_ui, p_iu = transition_matrices(urm)
    if alpha != 1.0:
        p_ui.data **= alpha  # elementwise power of the transition probabilities
        p_iu.data **= alpha
    if beta > 0:
        item_degree = np.asarray(urm.sum(axis=0)).ravel()
        penalty = sp.diags(_inverse_power(item_degree, beta))
        p_ui = (p_ui @ penalty).tocsr().astype(np.float32)  # scales column j by d_j ** -beta

    def similarity_block(start: int, stop: int) -> np.ndarray:
        return (p_iu[start:stop] @ p_ui).toarray()

    return blockwise_topk(similarity_block, urm.shape[1], topk, block_size)


class _ItemItemWalk(Recommender):
    """Scores items with ``R . S`` where S comes from the collapsed 3-hop walk.

    For a user ``u`` the walk gives ``Pi_I^(3) = (e_u . P_UI) . S``.  For binary data
    ``e_u . P_UI`` is the profile ``r_u`` times a constant, which does not change the
    ranking, so the scores are computed as ``r_u . S``.
    """

    def _beta(self) -> float:
        return 0.0

    def _fit(self, urm: sp.csr_matrix) -> None:
        cfg = self.config
        self._similarity = p3_similarity(urm, cfg.alpha, self._beta(), cfg.topk, cfg.block_size)

    def _score(self, users: np.ndarray) -> np.ndarray:
        return (self.urm[users] @ self._similarity).toarray()

    def item_similarity(self) -> sp.csr_matrix:
        return self._similarity


@register("p3alpha")
class P3Alpha(_ItemItemWalk):
    """P3-alpha: 3-hop random walk as item-item similarity, transitions raised to alpha."""

    Config = P3AlphaConfig


@register("rp3beta")
class RP3Beta(_ItemItemWalk):
    """RP3-beta: P3-alpha with the popularity of the destination item penalised by beta."""

    Config = RP3BetaConfig

    def _beta(self) -> float:
        return self.config.beta
