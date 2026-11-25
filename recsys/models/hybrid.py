"""Hybrid recommenders that combine other recommenders (Part 12 of the slides).

Every hybrid takes a list of ``components`` (any registered model with its own
hyper-parameters, including other hybrids, which gives the *hierarchical* hybrids
of the slides) and optional ``weights``:

* ``hybrid_linear``   - linear combination of the predicted scores,
                        ``r = a . r_A + b . r_B``, after an optional per-user
                        normalisation so that the scores share a scale.
* ``hybrid_list``     - list combination: the top-N lists of the components are
                        merged by round-robin (or Borda / reciprocal-rank fusion).
* ``hybrid_pipeline`` - pipelining: model A predicts new interactions that are added
                        to the URM on which model B is trained, and so on.
* ``hybrid_merge``    - merging models: the item-item matrices ``S = a S_A + b S_B``
                        of item-based models are combined into a single model.

Co-training (S-SLIM) needs an item content matrix and is not implemented.

Components trained by the linear / list / merge hybrids and the first stage of a
pipeline are cached in memory, so evaluating many weight settings in one run
(``evaluate.py`` with a ``grid`` over ``weights``) trains every component only once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from recsys.config import BaseConfig, config_from_dict, param
from recsys.models.base import Recommender, create_model, get_model_class, register
from recsys.similarity import keep_topk_per_row

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------ components
@dataclass(frozen=True)
class ComponentSpec:
    """A model name with its hyper-parameters."""

    model: str
    params: dict


def parse_components(raw: Any) -> List[ComponentSpec]:
    """Validate the ``components`` parameter.

    A list whose items are either a model name (default hyper-parameters) or
    ``{"model": name, "params": {...}}``.  Hyper-parameters are validated, so a typo
    fails before anything is trained.  Components may be hybrids themselves.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "components must be a non-empty list, e.g. "
            '\'[{"model": "item_knn", "params": {"topk": 200}}, {"model": "rp3beta"}]\' '
            "or simply item_knn,rp3beta"
        )
    specs = []
    for entry in raw:
        if isinstance(entry, str):
            entry = {"model": entry}
        if not isinstance(entry, dict) or "model" not in entry or set(entry) - {"model", "params"}:
            raise ValueError(
                f"Invalid component {entry!r}: expected a model name or "
                '{"model": name, "params": {...}}'
            )
        params = dict(entry.get("params") or {})
        config_from_dict(get_model_class(entry["model"]).Config, params)  # validates
        specs.append(ComponentSpec(entry["model"], params))
    return specs


# Parameters that only say *where / how fast* a model is trained and do not change the result:
# they are left out of the cache key, so e.g. a network trained with --device mps is found
# again by a run that says --device cpu.
_RUNTIME_ONLY_PARAMS = ("device", "num_threads")

_FIT_CACHE: "OrderedDict[tuple, Recommender]" = OrderedDict()
FIT_CACHE_SIZE = 8


def clear_fit_cache() -> None:
    """Forget every cached component (frees memory)."""
    _FIT_CACHE.clear()


def _fingerprint(matrix: Optional[sp.spmatrix]) -> Optional[str]:
    if matrix is None:
        return None
    matrix = sp.csr_matrix(matrix)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(matrix.shape).tobytes())
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def fit_component(
    spec: ComponentSpec,
    urm: sp.csr_matrix,
    validation: Optional[sp.csr_matrix],
    seen: Optional[sp.csr_matrix] = None,
    use_cache: bool = True,
    cache_dir: str = "",
    memory: bool = True,
) -> Recommender:
    """Train one component, reusing an identical already trained model if cached.

    ``memory=False`` skips the in-memory cache and only uses the on-disk one (for a
    single big model nobody will ask for again within the same run).

    Trained models are cached in memory and, when ``cache_dir`` is given, also on disk
    (pickle files named after the model and a hash of its hyper-parameters and data), so
    they survive between runs.  A model is only reused when the model, its
    hyper-parameters, the training matrix and the validation matrix are all identical.
    """
    model = create_model(spec.model, spec.params)
    key = path = None
    if use_cache:
        cache_params = {k: v for k, v in model.config.to_dict().items() if k not in _RUNTIME_ONLY_PARAMS}
        key = (
            spec.model,
            json.dumps(cache_params, sort_keys=True),
            _fingerprint(urm),
            _fingerprint(validation),
            _fingerprint(seen),
        )
        if memory and key in _FIT_CACHE:
            _FIT_CACHE.move_to_end(key)
            logger.info("Component %s: reusing the cached trained model", spec.model)
            return _FIT_CACHE[key]
        if cache_dir:
            digest = hashlib.blake2b(repr(key).encode(), digest_size=12).hexdigest()
            path = Path(cache_dir) / f"{spec.model}-{digest}.pkl"
            loaded = _load_model(path)
            if loaded is not None:
                logger.info("Component %s: loaded the trained model from %s", spec.model, path)
                if memory:
                    _remember(key, loaded)
                return loaded
    logger.info("Training component %s %s", spec.model, spec.params or "")
    model.fit(urm, validation=validation, seen=seen)
    if key is not None and memory:
        _remember(key, model)
    if path is not None:
        _save_model(model, path)
    return model


def fit_cached(
    name: str,
    params: dict,
    urm: sp.csr_matrix,
    validation: Optional[sp.csr_matrix] = None,
    cache_dir: str = "",
) -> Recommender:
    """Train any model, loading it from ``cache_dir`` when an identical one was saved before.

    Used by ``main.py evaluate / submit`` and ``evaluate.py`` for their ``--cache-dir`` flag.
    The cache key is the model, its full hyper-parameters, and the training and validation
    matrices, so changing any of them trains again.  A hybrid is not pickled as a whole (that
    would store every component a second time): the folder is handed to its own
    ``cache_dir`` parameter instead, so its components are cached one by one.
    """
    model = create_model(name, params)
    if not cache_dir:
        return model.fit(urm, validation=validation)
    if hasattr(model.config, "cache_dir"):  # a hybrid
        if not model.config.cache_dir:
            model = create_model(name, {**params, "cache_dir": cache_dir})
        return model.fit(urm, validation=validation)
    return fit_component(ComponentSpec(name, dict(params)), urm, validation,
                         cache_dir=cache_dir, memory=False)


def _remember(key: tuple, model: Recommender) -> None:
    _FIT_CACHE[key] = model
    while len(_FIT_CACHE) > FIT_CACHE_SIZE:
        _FIT_CACHE.popitem(last=False)


def _load_model(path: Path) -> Optional[Recommender]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception as error:  # corrupt or written by an incompatible version: train again
        logger.warning("Could not read the cached model %s (%s); training it again", path, error)
        return None


def _save_model(model: Recommender, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with open(temporary, "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)  # never leave a half-written file behind
        logger.info("Saved the trained model to %s", path)
    except OSError as error:
        logger.warning("Could not save the model to %s: %s", path, error)


# ----------------------------------------------------------------------- shared config
@dataclass
class HybridConfig(BaseConfig):
    components: list = param(
        [],
        "Models to combine: JSON list of model names or {\"model\": name, \"params\": {...}} "
        "objects, @file.json, or a comma separated list of names. Components can be hybrids.",
        json=True,
    )
    weights: list = param(
        [],
        "One non-negative weight per component, JSON or comma separated (empty = all 1). "
        "For two components 0.7,0.3 is the slides' alpha, 1 - alpha.",
        json=True,
    )
    cache: bool = param(
        True,
        "Keep trained components in memory and reuse them when the same component, data "
        "and hyper-parameters come up again (e.g. when grid-searching the weights).",
    )
    cache_dir: str = param(
        "",
        "Also save the trained components in this folder (e.g. cache) and load them from "
        "there in later runs, so new weights need no retraining. Empty = memory only.",
    )

    def validate(self) -> None:
        super().validate()
        specs = parse_components(self.components)
        if len(specs) < 2:
            raise ValueError("A hybrid needs at least two components")
        if self.weights:
            if len(self.weights) != len(specs):
                raise ValueError(
                    f"{len(self.weights)} weights given for {len(specs)} components"
                )
            if any(not isinstance(w, (int, float)) or w < 0 for w in self.weights):
                raise ValueError(f"weights must be non-negative numbers, got {self.weights}")
            if sum(self.weights) <= 0:
                raise ValueError("at least one weight must be positive")


class _Hybrid(Recommender):
    """Common machinery: the parsed components and their weights."""

    Config = HybridConfig

    def __init__(self, config: Optional[BaseConfig] = None, **params) -> None:
        super().__init__(config, **params)
        self.specs = parse_components(self.config.components)
        weights = self.config.weights or [1.0] * len(self.specs)
        self.weights = [float(w) for w in weights]
        self._models: List[Recommender] = []

    def _fit_components(self, urm: sp.csr_matrix) -> None:
        self._models = [
            fit_component(spec, urm, self.validation, use_cache=self.config.cache,
                          cache_dir=self.config.cache_dir)
            for spec in self.specs
        ]


# ---------------------------------------------------------------------- linear combination
NORMALIZATIONS = ("none", "max", "minmax", "zscore", "l1", "l2")


def normalize_scores(scores: np.ndarray, seen: np.ndarray, method: str) -> np.ndarray:
    """Normalise every user's scores (row) so that components share a scale.

    Statistics are computed on the items the user has not seen only, since the
    seen items are never recommended.  ``seen`` is a boolean mask like ``scores``.
    """
    if method == "none":
        return scores
    unseen = ~seen
    count = np.maximum(unseen.sum(axis=1, keepdims=True), 1)
    if method == "max":  # best unseen item gets 1
        top = np.where(seen, -np.inf, scores).max(axis=1, keepdims=True)
        top = np.where(np.isfinite(top) & (top > 0), top, 1.0)
        return scores / top
    if method == "minmax":
        low = np.where(seen, np.inf, scores).min(axis=1, keepdims=True)
        high = np.where(seen, -np.inf, scores).max(axis=1, keepdims=True)
        low = np.where(np.isfinite(low), low, 0.0)
        span = np.where(np.isfinite(high) & (high - low > 0), high - low, 1.0)
        return (scores - low) / span
    if method == "zscore":
        mean = np.where(seen, 0.0, scores).sum(axis=1, keepdims=True) / count
        var = (np.where(seen, 0.0, scores - mean) ** 2).sum(axis=1, keepdims=True) / count
        return (scores - mean) / np.where(var > 0, np.sqrt(var), 1.0)
    if method == "l1":
        total = np.where(seen, 0.0, np.abs(scores)).sum(axis=1, keepdims=True)
        return scores / np.where(total > 0, total, 1.0)
    if method == "l2":
        norm = np.sqrt((np.where(seen, 0.0, scores) ** 2).sum(axis=1, keepdims=True))
        return scores / np.where(norm > 0, norm, 1.0)
    raise ValueError(f"Unknown normalization {method!r}, choose one of {NORMALIZATIONS}")


@dataclass
class LinearConfig(HybridConfig):
    normalize: str = param(
        "max",
        "Per-user score normalisation applied to every component before the weighted sum "
        "(the slides: works better when scores share a scale). max = divide by the best "
        "unseen score; minmax / zscore also centre the scores (use them for models with "
        "negative scores, e.g. matrix factorisation); l1 / l2 divide by the norm; none.",
        choices=NORMALIZATIONS,
    )


@register("hybrid_linear")
class HybridLinear(_Hybrid):
    """Linear combination of the scores of several models: r = sum_j w_j . normalise(r_j)."""

    Config = LinearConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        self._fit_components(urm)

    def _score(self, users: np.ndarray) -> np.ndarray:
        seen = (self.urm[users] > 0).toarray()
        total = np.zeros(seen.shape, dtype=np.float32)
        for weight, model in zip(self.weights, self._models):
            if weight == 0:
                continue
            scores = np.asarray(model._score(users), dtype=np.float32)
            total += np.float32(weight) * normalize_scores(scores, seen, self.config.normalize)
        return total


# ----------------------------------------------------------------------- list combination
LIST_METHODS = ("round_robin", "borda", "rrf")


@dataclass
class ListConfig(HybridConfig):
    method: str = param(
        "round_robin",
        "round_robin = take one item from each list in turn, skipping items already taken "
        "(weights = relative share of the positions); borda = sum of weight * (list_size - rank); "
        "rrf = sum of weight / (rrf_k + rank), the reciprocal rank fusion.",
        choices=LIST_METHODS,
    )
    list_size: int = param(
        0, "Length of the list taken from every component (0 = the requested list length)."
    )
    rrf_k: float = param(60.0, "Constant of the reciprocal rank fusion (only for --method rrf).")


@register("hybrid_list")
class HybridList(_Hybrid):
    """List combination: merge the top-N lists of several models (round-robin, Borda, RRF)."""

    Config = ListConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        self._fit_components(urm)

    def _score(self, users: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "hybrid_list only produces ranked lists, not scores: it cannot be a component of "
            "hybrid_linear or hybrid_merge (it can be one of hybrid_list or hybrid_pipeline)"
        )

    def recommend_with_scores(
        self,
        users: np.ndarray,
        k: int = 20,
        batch_size: int = 1000,
        exclude_seen: bool = True,
        popularity_tiebreak: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fused lists; the returned scores are the fusion scores (rank based for round-robin)."""
        if self.urm is None:
            raise RuntimeError("Call fit() before recommend()")
        users = np.asarray(users)
        size = max(k, self.config.list_size)
        lists = [
            model.recommend(users, k=size, batch_size=batch_size, exclude_seen=exclude_seen,
                            popularity_tiebreak=popularity_tiebreak)
            for model in self._models
        ]
        if self.config.method == "round_robin":
            items = self._round_robin(lists, k)
            scores = np.tile(np.linspace(1.0, 1.0 / k, k, dtype=np.float32), (len(users), 1))
            return items, scores
        return self._score_fusion(lists, k, size)

    def _round_robin(self, lists: List[np.ndarray], k: int) -> np.ndarray:
        """Each component takes its turns in proportion to its weight; duplicates burn the turn.

        With equal weights this is the alternation of the slides (A1, B1, A2, B2, ...): when
        the item at a component's turn is already in the result, the turn is lost.
        """
        n_users, size = lists[0].shape
        step = [1.0 / w if w > 0 else np.inf for w in self.weights]
        python_lists = [array.tolist() for array in lists]
        result = np.empty((n_users, k), dtype=np.int32)
        for user in range(n_users):
            pointer = [0] * len(lists)
            taken_turns = [0] * len(lists)
            used, row = set(), []
            while len(row) < k:
                best, best_time = -1, np.inf
                for j in range(len(lists)):
                    due = (taken_turns[j] + 1) * step[j]
                    if pointer[j] < size and due < best_time:
                        best, best_time = j, due
                if best < 0:
                    break
                item = python_lists[best][user][pointer[best]]
                pointer[best] += 1
                taken_turns[best] += 1
                if item not in used:
                    used.add(item)
                    row.append(item)
            if len(row) < k:  # cannot happen when every list holds k distinct items
                row += [item for item in python_lists[0][user] if item not in used][: k - len(row)]
            result[user] = row
        return result

    def _score_fusion(self, lists: List[np.ndarray], k: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
        n_users = lists[0].shape[0]
        ranks = np.arange(size, dtype=np.float32)
        if self.config.method == "borda":
            points = size - ranks
        else:  # rrf
            points = 1.0 / (self.config.rrf_k + ranks + 1.0)
        rows = np.arange(n_users)[:, None]
        fused = np.zeros((n_users, self.n_items), dtype=np.float32)
        for weight, items in zip(self.weights, lists):
            fused[rows, items] += np.float32(weight) * points[None, :]
        # items with the same fused score: prefer the more popular one
        fused += np.float32(1e-3 * points.min()) * self._popularity_norm[None, :]
        candidates = np.argpartition(-fused, k - 1, axis=1)[:, :k]
        candidate_scores = np.take_along_axis(fused, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        return (
            np.take_along_axis(candidates, order, axis=1).astype(np.int32),
            np.take_along_axis(candidate_scores, order, axis=1),
        )

    def recommend(
        self,
        users: np.ndarray,
        k: int = 20,
        batch_size: int = 1000,
        exclude_seen: bool = True,
        popularity_tiebreak: bool = True,
    ) -> np.ndarray:
        users = np.asarray(users)
        out = np.empty((len(users), k), dtype=np.int32)
        for start in range(0, len(users), batch_size):
            batch = users[start : start + batch_size]
            out[start : start + len(batch)], _ = self.recommend_with_scores(
                batch, k, batch_size, exclude_seen, popularity_tiebreak
            )
        return out


# --------------------------------------------------------------------------- pipelining
@dataclass
class PipelineConfig(HybridConfig):
    top_n: int = param(10, "Items predicted by a stage for every user and added to the URM.")
    added_value: float = param(1.0, "Value written in the URM for an added interaction.")
    use_scores: bool = param(
        False,
        "Scale the value of an added interaction by the prediction confidence: the score "
        "relative to the user's best predicted item, times --added-value.",
    )
    max_profile_size: int = param(
        0, "Only add predictions to users with at most this many interactions (0 = all users)."
    )
    batch_size: int = param(1000, "Users scored at once when generating the added interactions.")


@register("hybrid_pipeline")
class HybridPipeline(_Hybrid):
    """Pipelining: each model's top predictions are added to the URM used to train the next one."""

    Config = PipelineConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        cfg = self.config
        current = urm
        self._models = []
        for index, spec in enumerate(self.specs):
            first = index == 0
            model = fit_component(
                spec, current, self.validation,
                seen=None if first else urm,  # never recommend what the user really has seen
                use_cache=cfg.cache and first,
                cache_dir=cfg.cache_dir if first else "",
            )
            self._models.append(model)
            if index < len(self.specs) - 1:
                before = current.nnz
                current = self._augment(model, current)
                logger.info("Pipeline stage %d (%s): URM grew from %d to %d interactions",
                            index + 1, spec.model, before, current.nnz)

    def _augment(self, model: Recommender, current: sp.csr_matrix) -> sp.csr_matrix:
        cfg = self.config
        sizes = np.diff(self.urm.indptr)
        eligible = np.flatnonzero(sizes > 0)  # no predictions for users without history
        if cfg.max_profile_size > 0:
            eligible = eligible[sizes[eligible] <= cfg.max_profile_size]

        rows, cols, values = [], [], []
        for start in range(0, len(eligible), cfg.batch_size):
            batch = eligible[start : start + cfg.batch_size]
            items, scores = model.recommend_with_scores(batch, k=cfg.top_n, batch_size=len(batch))
            if cfg.use_scores:
                best = scores[:, :1]
                value = cfg.added_value * np.clip(scores / np.where(best > 0, best, 1.0), 0.0, 1.0)
            else:
                value = np.full(scores.shape, cfg.added_value, dtype=np.float32)
            valid = np.isfinite(scores) & (value > 0)
            rows.append(np.repeat(batch, cfg.top_n)[valid.ravel()])
            cols.append(items.ravel()[valid.ravel()])
            values.append(value.ravel()[valid.ravel()])

        shape = current.shape
        added = sp.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=shape, dtype=np.float32,
        )
        already = (current > 0).astype(np.float32)
        added = added - added.multiply(already)  # keep only interactions that are new
        added.eliminate_zeros()
        return sp.csr_matrix(current + added, dtype=np.float32)

    def _score(self, users: np.ndarray) -> np.ndarray:
        return self._models[-1]._score(users)


# ---------------------------------------------------------------------- merging models
MERGE_NORMALIZATIONS = ("none", "max", "mean")


@dataclass
class MergeConfig(HybridConfig):
    normalize: str = param(
        "max",
        "Scale of every similarity matrix before the weighted sum: max = divide by the largest "
        "entry, mean = divide by the mean non-zero entry, none.",
        choices=MERGE_NORMALIZATIONS,
    )
    topk: int = param(
        0,
        "Neighbours kept per target item in the merged matrix, i.e. a KNN on the merged "
        "similarity (0 = keep the union of the components' neighbours; the result is then "
        "equivalent to a linear combination of the scores).",
    )


@register("hybrid_merge")
class HybridMerge(_Hybrid):
    """Merging models: S = sum_j w_j . S_j for item-item models (KNN, SLIM, P3-alpha, RP3-beta)."""

    Config = MergeConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        self._fit_components(urm)
        merged = None
        for weight, model, spec in zip(self.weights, self._models, self.specs):
            try:
                similarity = sp.csr_matrix(model.item_similarity(), dtype=np.float32)
            except NotImplementedError as error:
                raise ValueError(f"Cannot merge component '{spec.model}': {error}") from error
            similarity = similarity * np.float32(weight / self._scale(similarity))
            merged = similarity if merged is None else merged + similarity
        merged = sp.csr_matrix(merged, dtype=np.float32)
        if self.config.topk > 0:  # keep the strongest sources of every target item (column)
            merged = keep_topk_per_row(merged.T.tocsr(), self.config.topk).T.tocsr()
        self._similarity = merged
        logger.info("Merged similarity matrix: %d non-zero entries", merged.nnz)

    def _scale(self, similarity: sp.csr_matrix) -> float:
        magnitude = np.abs(similarity.data)
        if self.config.normalize == "none" or magnitude.size == 0:
            return 1.0
        scale = float(magnitude.max() if self.config.normalize == "max" else magnitude.mean())
        return scale if scale > 0 else 1.0

    def _score(self, users: np.ndarray) -> np.ndarray:
        return (self.urm[users] @ self._similarity).toarray()

    def item_similarity(self) -> sp.csr_matrix:
        return self._similarity
