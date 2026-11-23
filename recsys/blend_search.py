"""Fast search for the weights of a linear blend (``hybrid_linear``).

Evaluating a weight grid through ``evaluate.py`` rebuilds the full score matrix of every
component for every weight setting.  Here the components are trained (or loaded from the
cache) once and scored once; every weight setting is then evaluated on a small pool of
*candidate items* per user:

* for every validation user, the top ``candidates`` unseen items of every component are
  collected (their union is the candidate set);
* the normalised score of every component is stored for those candidates only;
* a weight setting is a weighted sum of those stored scores followed by a top-N, which
  takes a fraction of a second for all users.

The blend's top-N is almost always made of items that are in the top few hundred of at
least one component, so the numbers match the real ``hybrid_linear`` evaluation (the
search script re-evaluates the best setting the exact way to confirm it).  Differences can
only come from items that no component ranks in its top ``candidates`` and from the
popularity tie-break of the real recommender.

Only linear blends can be searched this way: weights of zero drop a component, so the grid
also finds the best *subset* of components.  Scores are scale invariant (multiplying all
the weights by a constant does not change the ranking), so the weights are reported
normalised to sum to 1 and equivalent settings are evaluated once.
"""

from __future__ import annotations

import itertools
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.models.base import Recommender
from recsys.models.hybrid import NORMALIZATIONS, normalize_scores

logger = logging.getLogger(__name__)

MAX_RAW_COMBINATIONS = 2_000_000


class CandidatePool:
    """Pre-computed candidate items and component scores for a set of validation users."""

    def __init__(
        self,
        models: Sequence[Recommender],
        urm: sp.csr_matrix,
        validation: sp.csr_matrix,
        users: np.ndarray,
        cutoff: int = 20,
        candidates: int = 100,
        normalizations: Sequence[str] = ("max",),
        batch_size: int = 500,
    ) -> None:
        if len(models) < 2:
            raise ValueError("A blend needs at least two components")
        unknown = [name for name in normalizations if name not in NORMALIZATIONS]
        if unknown:
            raise ValueError(f"Unknown normalization(s) {unknown}, choose from {NORMALIZATIONS}")
        n_items = urm.shape[1]
        per_model = min(max(candidates, cutoff), n_items)
        if len(models) * per_model < cutoff:
            raise ValueError("Too few candidates for the requested cut-off")

        users = np.asarray(users)
        self.normalizations = list(dict.fromkeys(normalizations))
        self.cutoff = cutoff
        self.n_models = len(models)
        n_users, width = len(users), len(models) * per_model
        self.scores = np.zeros((len(self.normalizations), len(models), n_users, width), np.float32)
        self.valid = np.zeros((n_users, width), dtype=bool)
        self.relevant = np.zeros((n_users, width), dtype=bool)
        logger.info(
            "Candidate pool: %d users x %d candidates x %d components x %d normalisation(s) = %.0f MB",
            n_users, width, len(models), len(self.normalizations), self.scores.nbytes / 1e6,
        )

        popularity = np.asarray(urm.sum(axis=0)).ravel().astype(np.float32)
        profile_sizes = np.diff(urm.indptr)
        for start in range(0, n_users, batch_size):
            batch = users[start : start + batch_size]
            stop = start + len(batch)
            seen = (urm[batch] > 0).toarray()
            cold = profile_sizes[batch] == 0
            raw = []
            for model in models:
                scores = np.array(model._score(batch), dtype=np.float32)
                if cold.any():  # the recommender falls back to popularity for empty profiles
                    scores[cold] = popularity
                raw.append(scores)

            tops = [
                np.argpartition(-np.where(seen, -np.inf, scores), per_model - 1, axis=1)[:, :per_model]
                for scores in raw
            ]
            candidate_items = np.sort(np.concatenate(tops, axis=1), axis=1)
            valid = np.ones(candidate_items.shape, dtype=bool)
            valid[:, 1:] = candidate_items[:, 1:] != candidate_items[:, :-1]  # drop duplicates
            valid &= ~np.take_along_axis(seen, candidate_items, axis=1)
            held_out = validation[batch].toarray() > 0

            self.valid[start:stop] = valid
            self.relevant[start:stop] = np.take_along_axis(held_out, candidate_items, axis=1)
            for m, method in enumerate(self.normalizations):
                for j, scores in enumerate(raw):
                    normalised = normalize_scores(scores, seen, method)
                    self.scores[m, j, start:stop] = np.take_along_axis(normalised, candidate_items, axis=1)

        n_relevant = np.diff(validation.indptr)[users]
        if np.any(n_relevant == 0):
            raise ValueError("Every evaluated user needs at least one held-out interaction")
        self._ideal_length = np.minimum(n_relevant, cutoff)
        self._discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
        self._ideal_dcg = np.cumsum(self._discounts)[self._ideal_length - 1]

    def evaluate(self, weights: Sequence[float], normalization: str) -> Tuple[float, float]:
        """``(ndcg@cutoff, map@cutoff)`` of the blend with these weights."""
        if len(weights) != self.n_models:
            raise ValueError(f"Expected {self.n_models} weights, got {len(weights)}")
        m = self.normalizations.index(normalization)
        combined = None
        for j, weight in enumerate(weights):
            if weight == 0:
                continue
            term = self.scores[m, j] * np.float32(weight)
            combined = term if combined is None else combined + term
        if combined is None:
            raise ValueError("At least one weight must be positive")
        combined[~self.valid] = -np.inf

        k = self.cutoff
        top = np.argpartition(-combined, k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(combined, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        finite = np.isfinite(np.take_along_axis(top_scores, order, axis=1))
        hits = np.take_along_axis(self.relevant, top, axis=1) & finite

        dcg = (hits * self._discounts).sum(axis=1)
        precision_at_i = np.cumsum(hits, axis=1) / np.arange(1, k + 1)
        average_precision = (precision_at_i * hits).sum(axis=1) / self._ideal_length
        return float(np.mean(dcg / self._ideal_dcg)), float(np.mean(average_precision))


# ----------------------------------------------------------------------------- searches
def _scaled_key(weights: Sequence[float]) -> tuple:
    top = max(weights)
    return tuple(round(float(w) / top, 6) for w in weights)


def _log_progress(done: int, total: int, started: float, best: float, every: int) -> None:
    if done % every == 0 or done == total:
        elapsed = time.perf_counter() - started
        eta = elapsed / done * (total - done)
        logger.info("%d/%d settings evaluated, best ndcg so far %.4f (%.0fs left)", done, total, best, eta)


def grid_search(
    pool: CandidatePool,
    value_lists: Sequence[Sequence[float]],
    normalizations: Sequence[str],
    max_combinations: int = 20000,
) -> List[Dict]:
    """Evaluate every combination of the candidate weights of the components."""
    raw_size = int(np.prod([len(values) for values in value_lists], dtype=np.float64))
    if raw_size > MAX_RAW_COMBINATIONS:
        raise ValueError(
            f"The weight grid has {raw_size} combinations; use fewer values per component, "
            "fix the strongest component to a single value, or use the coordinate strategy"
        )
    settings: Dict[tuple, Tuple[float, ...]] = {}
    for values in itertools.product(*value_lists):
        if max(values) > 0:
            settings.setdefault(_scaled_key(values), tuple(values))
    total = len(settings) * len(normalizations)
    if total > max_combinations:
        raise ValueError(
            f"The grid has {total} distinct settings (> max_combinations={max_combinations}); "
            "use fewer values per component, fix the strongest component to [1], "
            "raise max_combinations or use the coordinate strategy"
        )
    logger.info("Grid search: %d distinct weight settings x %d normalisation(s)", len(settings), len(normalizations))

    rows: List[Dict] = []
    started, best = time.perf_counter(), 0.0
    for method in normalizations:
        for weights in settings.values():
            ndcg, average_precision = pool.evaluate(weights, method)
            rows.append({"weights": list(weights), "normalize": method, "ndcg": ndcg, "map": average_precision})
            best = max(best, ndcg)
            _log_progress(len(rows), total, started, best, max(total // 10, 1))
    return rows


def coordinate_search(
    pool: CandidatePool,
    value_lists: Sequence[Sequence[float]],
    normalizations: Sequence[str],
    max_rounds: int = 5,
) -> List[Dict]:
    """Coordinate ascent: start from the best single component, then repeatedly try every
    candidate value of one component at a time (the others fixed) and keep improvements."""
    rows: List[Dict] = []
    n_models = len(value_lists)
    for method in normalizations:
        seen: Dict[tuple, Tuple[float, float]] = {}

        def score(weights: Sequence[float]) -> float:
            key = _scaled_key(weights)
            if key not in seen:
                seen[key] = pool.evaluate(weights, method)
                rows.append({"weights": list(weights), "normalize": method,
                             "ndcg": seen[key][0], "map": seen[key][1]})
            return seen[key][0]

        best_weights, best = None, -1.0
        for j in range(n_models):  # best single component
            weights = [0.0] * n_models
            weights[j] = 1.0
            value = score(weights)
            if value > best:
                best, best_weights = value, weights
        for round_number in range(1, max_rounds + 1):
            improved = False
            for j in range(n_models):
                for value in value_lists[j]:
                    weights = list(best_weights)
                    weights[j] = float(value)
                    if max(weights) <= 0:
                        continue
                    result = score(weights)
                    if result > best + 1e-9:
                        best, best_weights, improved = result, weights, True
            logger.info("[%s] round %d: best ndcg %.4f with weights %s", method, round_number, best,
                        [round(w, 3) for w in best_weights])
            if not improved:
                break
    return rows


def results_frame(rows: List[Dict], labels: Sequence[str], cutoff: int) -> pd.DataFrame:
    """One row per evaluated setting, best first; weights are shown normalised to sum 1."""
    records = []
    for row in rows:
        weights = np.asarray(row["weights"], dtype=float)
        record = dict(zip(labels, weights / weights.sum()))
        record.update({"normalize": row["normalize"], f"ndcg@{cutoff}": row["ndcg"], f"map@{cutoff}": row["map"]})
        records.append(record)
    frame = pd.DataFrame(records)
    return frame.sort_values(f"ndcg@{cutoff}", ascending=False).reset_index(drop=True)


def resolve_value_lists(
    labels: Sequence[str],
    entry_values: Sequence[Optional[Sequence[float]]],
    per_label: Dict[str, Sequence[float]],
    default_values: Sequence[float],
) -> List[List[float]]:
    """Candidate weights per component: its own list, else ``per_label``, else the default."""
    unknown = set(per_label) - set(labels)
    if unknown:
        raise ValueError(f"component_weights names unknown component(s) {sorted(unknown)}; labels are {list(labels)}")
    lists = []
    for label, own in zip(labels, entry_values):
        values = own if own is not None else per_label.get(label, default_values)
        values = [float(v) for v in values]
        if not values or any(v < 0 for v in values):
            raise ValueError(f"Invalid weight values {values} for component {label!r}")
        lists.append(sorted(set(values)))
    if all(max(values) <= 0 for values in lists):
        raise ValueError("At least one component needs a positive candidate weight")
    return lists
