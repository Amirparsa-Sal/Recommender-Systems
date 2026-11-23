"""Triple sampling shared by the pairwise ranking models (BPR and WARP-style)."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from recsys.sampling import NegativeSampler

PAIRWISE_LOSSES = ("bpr", "warp")

ScoreFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def draw_triples(
    users: np.ndarray,
    positives: np.ndarray,
    sampler: NegativeSampler,
    loss: str,
    score_fn: ScoreFunction,
    warp_trials: int = 10,
    warp_margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair every positive (u, i) with a negative item j.

    * ``bpr``: j is drawn by the negative sampler.
    * ``warp``: up to ``warp_trials`` negatives are drawn and the first one the
      model currently ranks wrongly (``score_j + margin >= score_i``) is kept.
      Pairs for which no violating negative is found are skipped, so training
      concentrates on the pairs the model still gets wrong.

    Returns:
        ``(users, positives, negatives)``, possibly shorter than the inputs for ``warp``.
    """
    if loss == "bpr":
        return users, positives, sampler.sample(users)

    positive_scores = score_fn(users, positives)
    chosen = np.full(len(users), -1, dtype=np.int64)
    for _ in range(warp_trials):
        pending = np.flatnonzero(chosen < 0)
        if pending.size == 0:
            break
        candidates = sampler.sample(users[pending])
        violating = score_fn(users[pending], candidates) + warp_margin >= positive_scores[pending]
        chosen[pending[violating]] = candidates[violating]
    keep = chosen >= 0
    return users[keep], positives[keep], chosen[keep]
