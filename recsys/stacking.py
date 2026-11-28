"""Stacking: a learned ranker (XGBoost) on top of the candidates of several recommenders.

A linear or gated blend applies one fixed weight vector to the scores of the components.  The
ranker instead *learns* from examples how to combine them, and can use interactions such as
"trust model A unless the item is very popular and model B disagrees".

How it works
------------
1. **Candidates.**  For every user the top ``candidates`` unseen items of every component are
   collected (their union is the candidate set of the user).
2. **Features.**  Every (user, candidate) pair is described by the score of every component
   (raw, divided by the user's best score, z-score), the rank of the candidate in every
   component, agreement features (in how many lists is it, reciprocal-rank fusion, a plain
   blend...), item features (popularity, how active its users are) and user features (profile
   length, how mainstream the profile is).
3. **Labels.**  A pair is positive when the item is one of the user's held-out interactions.
   The components are trained on the training matrix only, so they never saw these labels.
4. **Learning.**  XGBoost learns to order the candidates of each user: it classifies every pair
   as relevant or not (``binary:logistic``), or optimises the order directly (``rank:ndcg`` is
   LambdaMART: its gradients are weighted by the change of NDCG, so errors near the top of
   the list matter most).
5. **Evaluation.**  Users are split in folds; the ranker is trained on some users and scored on
   the others, so every reported number is out of sample (see :func:`cross_validate`).
6. **Submission.**  The ranker is trained on all the users, the components are retrained on
   all the data and the ranker re-orders their candidates.  Count features (profile length,
   item popularity) are multiplied by ``1 - validation_fraction`` so that they stay on the
   scale the ranker was trained on.

XGBoost runs in a separate Python process (see :func:`_run_worker`): on macOS it can crash with a
segmentation fault when PyTorch (needed by the neural components) is loaded in the same process,
because both bring their own OpenMP runtime.

The ranker can only re-order the candidates: a relevant item that no component lists in its top
``candidates`` is lost (:func:`candidate_recall` measures that ceiling).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from recsys.config import BaseConfig, param
from recsys.models.base import Recommender
from recsys.models.hybrid import normalize_scores

logger = logging.getLogger(__name__)

OBJECTIVES = ("rank:ndcg", "rank:pairwise", "rank:map", "binary:logistic")
RRF_K = 60.0
MAX_RANK_DEPTH = 30000  # ranks are stored as int16


def _import_xgboost():
    try:
        import xgboost
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The stacking ranker needs XGBoost: pip install xgboost"
        ) from error
    return xgboost


# --------------------------------------------------------------------------------- config
@dataclass
class RankerConfig(BaseConfig):
    """Candidate generation, features and XGBoost hyper-parameters of the ranker."""

    candidates: int = param(
        50,
        "Top items of every component added to the candidate set of a user (the union over the "
        "components is re-ranked). More candidates = higher recall ceiling but more rows to "
        "train on. Must be at least the evaluation cut-off.",
    )
    rank_depth: int = param(
        500,
        "Rank features look this deep into every component's list; an item below is given rank "
        "rank_depth.",
    )
    raw_scores: bool = param(
        False,
        "Also use the raw score of every component, and the best raw score of the user, as "
        "features. They depend on the amount of training data, so by default the ranker only sees "
        "scale-free information (normalised scores, ranks, agreement), which stays valid when the "
        "components of the submission are trained on more data. In our test the raw scores added "
        "about 0.0003 NDCG@20 (noise) and made the submission features drift.",
    )
    zscores: bool = param(True, "Add the per-user z-score of every component as a feature.")
    blend_weights: list = param(
        [],
        "Weights of the plain linear blend (max-normalised scores) given to the ranker as one "
        "feature, one per component, in order (JSON or comma separated). Empty = equal weights. "
        "Put the weights found by tune_blend.py / tune_gated_blend.py here so that the ranker "
        "starts from the best blend.",
        json=True,
    )
    objective: str = param(
        "binary:logistic",
        "binary:logistic = classify every pair as relevant or not (the default: it was the best "
        "in our tests); rank:ndcg = LambdaMART optimising NDCG, worth comparing; rank:pairwise; "
        "rank:map.",
        choices=OBJECTIVES,
    )
    pair_method: str = param(
        "topk",
        "How the ranking objectives build the pairs: topk focuses on the top positions of the "
        "list (the ones NDCG@N rewards), mean samples pairs for every item.",
        choices=("topk", "mean"),
    )
    pairs_per_sample: int = param(
        0,
        "Truncation level for pair_method=topk (0 = the evaluation cut-off), pairs per item for "
        "pair_method=mean (0 = 8).",
    )
    n_estimators: int = param(
        2000, "Maximum number of trees (with early stopping the best number is found automatically)."
    )
    early_stopping_rounds: int = param(
        100,
        "Stop when the NDCG of the inner validation users does not improve for this many trees "
        "(0 = no early stopping, train exactly n_estimators trees).",
    )
    learning_rate: float = param(0.05, "Shrinkage of every tree (eta). Lower is better but needs more trees.")
    max_depth: int = param(6, "Depth of the trees = the order of the feature interactions the model can use.")
    min_child_weight: float = param(
        1.0, "Minimum sum of hessian in a leaf; raise it to block small leaves (ranking hessians are small)."
    )
    subsample: float = param(0.8, "Fraction of the rows used by every tree.")
    colsample_bytree: float = param(0.8, "Fraction of the features used by every tree.")
    reg_lambda: float = param(5.0, "L2 penalty on the leaf weights.")
    reg_alpha: float = param(0.0, "L1 penalty on the leaf weights.")
    gamma: float = param(0.0, "Minimum loss reduction to make a split.")
    max_bin: int = param(256, "Histogram bins per feature.")
    n_jobs: int = param(0, "Threads used by XGBoost (0 = all cores).")
    device: str = param("cpu", "cpu or cuda (XGBoost has no Apple GPU backend).", choices=("cpu", "cuda"))

    def validate(self) -> None:
        super().validate()
        if self.candidates < 1:
            raise ValueError("candidates must be at least 1")
        if not self.candidates <= self.rank_depth <= MAX_RANK_DEPTH:
            raise ValueError(f"rank_depth must be between candidates ({self.candidates}) and {MAX_RANK_DEPTH}")
        if self.n_estimators < 1 or self.early_stopping_rounds < 0:
            raise ValueError("n_estimators must be positive and early_stopping_rounds non-negative")
        if self.blend_weights:
            if any(not isinstance(w, (int, float)) or w < 0 for w in self.blend_weights) or sum(self.blend_weights) <= 0:
                raise ValueError(f"blend_weights must be non-negative with a positive sum, got {self.blend_weights}")

    @property
    def is_ranking(self) -> bool:
        return self.objective.startswith("rank:")


# --------------------------------------------------------------------------- the data
@dataclass
class RankerData:
    """Candidates of a set of users, grouped by user, with their features.

    ``indptr[i]:indptr[i+1]`` are the rows of the user ``users[i]`` (a row of the training
    matrix).  ``labels`` and ``n_relevant`` are only meaningful when the data was built with
    held-out interactions.
    """

    users: np.ndarray
    indptr: np.ndarray
    items: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    feature_names: List[str]
    n_relevant: Optional[np.ndarray] = None

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_rows(self) -> int:
        return len(self.items)

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.indptr)

    def group_ids(self) -> np.ndarray:
        """Group (user position) of every row, non-decreasing as XGBoost requires."""
        return np.repeat(np.arange(self.n_users), self.sizes)

    def subset(self, positions: Sequence[int]) -> "RankerData":
        """The data of some users (positions in ``users``), in increasing position order."""
        positions = np.sort(np.asarray(positions, dtype=np.int64))
        sizes = self.sizes[positions]
        indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        rows = np.arange(indptr[-1]) + np.repeat(self.indptr[positions] - indptr[:-1], sizes)
        return RankerData(
            users=self.users[positions], indptr=indptr, items=self.items[rows],
            features=self.features[rows], labels=self.labels[rows], feature_names=self.feature_names,
            n_relevant=None if self.n_relevant is None else self.n_relevant[positions],
        )

    def positives_per_user(self) -> np.ndarray:
        total = np.concatenate([[0], np.cumsum(self.labels, dtype=np.int64)])
        return total[self.indptr[1:]] - total[self.indptr[:-1]]


def feature_names(labels: Sequence[str], config: RankerConfig, cutoff: int) -> List[str]:
    """Names of the feature columns, in order."""
    names: List[str] = []
    for label in labels:
        if config.raw_scores:
            names.append(f"raw:{label}")
        names.append(f"max:{label}")
        if config.zscores:
            names.append(f"z:{label}")
        names.append(f"rank:{label}")
        if config.raw_scores:
            names.append(f"top:{label}")
    names += ["blend_max"]
    if config.zscores:
        names += ["blend_z"]
    names += ["rrf", f"n_top{cutoff}", "n_cand", "n_listed", "rank_best", "rank_worst", "rank_mean_log",
              "spread_max", "item_pop", "item_pop_pct", "item_user_len", "user_len", "user_mean_pop", "pop_ratio"]
    return names


def build_features(
    models: Sequence[Recommender],
    labels: Sequence[str],
    urm: sp.csr_matrix,
    users: np.ndarray,
    config: RankerConfig,
    cutoff: int = 20,
    validation: Optional[sp.csr_matrix] = None,
    count_scale: float = 1.0,
    batch_size: int = 250,
) -> RankerData:
    """Candidates and features of ``users`` (users with an empty profile are skipped).

    Args:
        models: trained components; ``urm`` must be the matrix they were trained on (its
            interactions are never recommended).
        validation: held-out interactions giving the labels; omit when predicting.
        count_scale: factor applied to the count features (profile length, item popularity),
            e.g. ``1 - validation_fraction`` when ``urm`` holds more data than the matrix the
            ranker was trained on.
    """
    n_models = len(models)
    if n_models < 1 or len(labels) != n_models:
        raise ValueError("Need one label per component")
    if config.candidates < cutoff:
        raise ValueError(f"candidates ({config.candidates}) must be at least the cut-off ({cutoff})")
    if config.blend_weights and len(config.blend_weights) != n_models:
        raise ValueError(f"blend_weights has {len(config.blend_weights)} values for {n_models} components")
    blend = np.asarray(config.blend_weights or [1.0] * n_models, dtype=np.float32)
    blend = blend / blend.sum()

    urm = sp.csr_matrix(urm)
    n_items = urm.shape[1]
    per_model = min(config.candidates, n_items)
    depth = min(config.rank_depth, n_items)
    profile = np.diff(urm.indptr)
    users = np.asarray(users)
    users = users[profile[users] > 0]
    names = feature_names(labels, config, cutoff)
    column = {name: index for index, name in enumerate(names)}
    n_features = len(names)

    popularity = np.asarray(urm.sum(axis=0)).ravel().astype(np.float32)
    item_user_len = np.asarray(urm.T @ profile.astype(np.float32)).ravel() / np.maximum(popularity, 1.0)
    pop_pct = (rankdata(popularity) / n_items).astype(np.float32)
    user_mean_pop = np.asarray(urm @ popularity).ravel() / np.maximum(profile, 1)
    scale = np.float32(count_scale)

    blocks, items_out, label_out, counts = [], [], [], []
    started = time.perf_counter()
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        size = len(batch)
        seen = (urm[batch] > 0).toarray()
        raws, tops, top_scores = [], [], []
        for model in models:
            raw = np.array(model._score(batch), dtype=np.float32)
            masked = np.where(seen, -np.inf, raw)
            top = np.argpartition(-masked, depth - 1, axis=1)[:, :depth]
            scores = np.take_along_axis(masked, top, axis=1)
            order = np.argsort(-scores, axis=1, kind="stable")
            raws.append(raw)
            tops.append(np.take_along_axis(top, order, axis=1))
            top_scores.append(np.take_along_axis(scores, order, axis=1))

        candidates = np.sort(np.concatenate([top[:, :per_model] for top in tops], axis=1), axis=1)
        valid = np.ones(candidates.shape, dtype=bool)
        valid[:, 1:] = candidates[:, 1:] != candidates[:, :-1]  # the same item from several components
        valid &= ~np.take_along_axis(seen, candidates, axis=1)
        width = candidates.shape[1]

        block = np.zeros((size, width, n_features), dtype=np.float32)
        rows = np.arange(size)[:, None]
        max_all, z_all, rank_all = [], [], []
        for j, label in enumerate(labels):
            raw_c = np.take_along_axis(raws[j], candidates, axis=1)
            best = top_scores[j][:, 0]
            denominator = np.where(np.isfinite(best) & (best > 0), best, 1.0)[:, None]
            max_c = raw_c / denominator
            position = np.full((size, n_items), depth, dtype=np.int16)
            position[rows, tops[j]] = np.arange(depth, dtype=np.int16)
            rank_c = np.take_along_axis(position, candidates, axis=1).astype(np.float32)
            if config.raw_scores:
                block[:, :, column[f"raw:{label}"]] = raw_c
            block[:, :, column[f"max:{label}"]] = max_c
            block[:, :, column[f"rank:{label}"]] = rank_c
            if config.raw_scores:
                block[:, :, column[f"top:{label}"]] = np.where(np.isfinite(best), best, 0.0)[:, None]
            max_all.append(max_c)
            rank_all.append(rank_c)
            if config.zscores:
                z_c = np.take_along_axis(normalize_scores(raws[j], seen, "zscore"), candidates, axis=1)
                block[:, :, column[f"z:{label}"]] = z_c
                z_all.append(z_c)

        max_all, rank_all = np.stack(max_all), np.stack(rank_all)
        block[:, :, column["blend_max"]] = np.tensordot(blend, max_all, axes=1)
        if config.zscores:
            block[:, :, column["blend_z"]] = np.tensordot(blend, np.stack(z_all), axes=1)
        block[:, :, column["rrf"]] = (1.0 / (RRF_K + rank_all + 1.0)).sum(axis=0)
        block[:, :, column[f"n_top{cutoff}"]] = (rank_all < cutoff).sum(axis=0)
        block[:, :, column["n_cand"]] = (rank_all < per_model).sum(axis=0)
        block[:, :, column["n_listed"]] = (rank_all < depth).sum(axis=0)
        block[:, :, column["rank_best"]] = rank_all.min(axis=0)
        block[:, :, column["rank_worst"]] = rank_all.max(axis=0)
        block[:, :, column["rank_mean_log"]] = np.log1p(rank_all).mean(axis=0)
        block[:, :, column["spread_max"]] = max_all.std(axis=0)

        item_pop = popularity[candidates]
        block[:, :, column["item_pop"]] = item_pop * scale
        block[:, :, column["item_pop_pct"]] = pop_pct[candidates]
        block[:, :, column["item_user_len"]] = item_user_len[candidates] * scale
        block[:, :, column["user_len"]] = (profile[batch] * scale)[:, None]
        block[:, :, column["user_mean_pop"]] = (user_mean_pop[batch] * scale)[:, None]
        block[:, :, column["pop_ratio"]] = item_pop / np.maximum(user_mean_pop[batch], 1e-6)[:, None]

        blocks.append(block[valid])
        items_out.append(candidates[valid].astype(np.int32))
        counts.append(valid.sum(axis=1))
        if validation is not None:
            held_out = validation[batch].toarray() > 0
            label_out.append(np.take_along_axis(held_out, candidates, axis=1)[valid].astype(np.int8))
        else:
            label_out.append(np.zeros(int(valid.sum()), dtype=np.int8))
        done = start + size
        if (start // batch_size) % 10 == 9 or done == len(users):
            elapsed = time.perf_counter() - started
            logger.info("Features: %d/%d users (%.0fs left)", done, len(users), elapsed / done * (len(users) - done))

    counts = np.concatenate(counts) if counts else np.zeros(0, dtype=np.int64)
    n_relevant = None
    if validation is not None:
        n_relevant = np.diff(sp.csr_matrix(validation).indptr)[users]
    data = RankerData(
        users=users,
        indptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        items=np.concatenate(items_out) if items_out else np.zeros(0, np.int32),
        features=np.concatenate(blocks) if blocks else np.zeros((0, n_features), np.float32),
        labels=np.concatenate(label_out) if label_out else np.zeros(0, np.int8),
        feature_names=names,
        n_relevant=n_relevant,
    )
    logger.info("Ranker data: %d users, %d candidate rows (%.0f per user), %d features, %.0f MB",
                data.n_users, data.n_rows, data.n_rows / max(data.n_users, 1), n_features, data.features.nbytes / 1e6)
    return data


# --------------------------------------------------------------------------- evaluation
def _padded(data: RankerData, values: np.ndarray, fill: Any, dtype) -> np.ndarray:
    sizes = data.sizes
    width = int(sizes.max()) if len(sizes) else 0
    out = np.full((data.n_users, width), fill, dtype=dtype)
    rows = np.repeat(np.arange(data.n_users), sizes)
    cols = np.arange(data.n_rows) - np.repeat(data.indptr[:-1], sizes)
    out[rows, cols] = values
    return out


def top_k_items(data: RankerData, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``k`` best candidates of every user by ``scores``: ``(items, hits, best_first)``.

    ``items`` and ``hits`` (is the item a held-out interaction) have shape (users, k).
    """
    if data.n_users and data.sizes.min() < k:
        raise ValueError(f"A user has fewer than {k} candidates; raise --candidates")
    padded = _padded(data, np.asarray(scores, dtype=np.float32), -np.inf, np.float32)
    top = np.argpartition(-padded, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(padded, top, axis=1), axis=1, kind="stable")
    top = np.take_along_axis(top, order, axis=1)
    items = np.take_along_axis(_padded(data, data.items, 0, np.int32), top, axis=1)
    hits = np.take_along_axis(_padded(data, data.labels > 0, False, bool), top, axis=1)
    return items, hits, top


def ranking_metrics(
    data: RankerData, scores: np.ndarray, cutoff: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-user ``(ndcg, ap, top_items)`` of ordering every user's candidates by ``scores``.

    ``top_items`` has shape (users, cutoff).  The metrics use the user's total number of
    held-out interactions, so relevant items that are not candidates count as misses (exactly
    like the real evaluation).
    """
    if data.n_relevant is None:
        raise ValueError("The data has no labels")
    if np.any(data.n_relevant == 0):
        raise ValueError("Every evaluated user needs at least one held-out interaction")
    items, hits, _ = top_k_items(data, scores, cutoff)
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
    ideal_length = np.minimum(data.n_relevant, cutoff)
    ndcg = (hits * discounts).sum(axis=1) / np.cumsum(discounts)[ideal_length - 1]
    precision_at_i = np.cumsum(hits, axis=1) / np.arange(1, cutoff + 1)
    average_precision = (precision_at_i * hits).sum(axis=1) / ideal_length
    return ndcg, average_precision, items


def candidate_recall(data: RankerData, cutoff: int = 20) -> Dict[str, float]:
    """The ceiling of any re-ranking: held-out items that are candidates, and the NDCG of a perfect order."""
    perfect, _, _ = ranking_metrics(data, data.labels.astype(np.float32), cutoff)
    return {
        "candidate_recall": float(data.labels.sum() / max(data.n_relevant.sum(), 1)),
        "ndcg_ceiling": float(perfect.mean()),
        "candidates_per_user": float(data.n_rows / max(data.n_users, 1)),
    }


# ------------------------------------------------------------------------------ training
def _xgb_params(config: RankerConfig, cutoff: int) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "objective": config.objective,
        "eta": config.learning_rate,
        "max_depth": config.max_depth,
        "min_child_weight": config.min_child_weight,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "lambda": config.reg_lambda,
        "alpha": config.reg_alpha,
        "gamma": config.gamma,
        "max_bin": config.max_bin,
        "tree_method": "hist",
        "device": config.device,
        "seed": config.seed,
        "eval_metric": f"ndcg@{cutoff}-",  # a group without a positive counts as 0
        "verbosity": 0,
    }
    if config.n_jobs > 0:
        params["nthread"] = config.n_jobs
    if config.is_ranking:
        params["lambdarank_pair_method"] = config.pair_method
        default_pairs = cutoff if config.pair_method == "topk" else 8
        params["lambdarank_num_pair_per_sample"] = config.pairs_per_sample or default_pairs
    return params


def _usable(data: RankerData, config: RankerConfig) -> RankerData:
    """Ranking objectives learn nothing from a user without any relevant candidate."""
    if not config.is_ranking:
        return data
    keep = np.flatnonzero(data.positives_per_user() > 0)
    return data if len(keep) == data.n_users else data.subset(keep)


def train_booster(
    train: RankerData,
    valid: Optional[RankerData],
    config: RankerConfig,
    cutoff: int = 20,
    rounds: Optional[int] = None,
):
    """Train one XGBoost model; returns ``(booster, number_of_trees_to_use)``.

    With ``valid`` and ``early_stopping_rounds`` the number of trees is chosen on ``valid``;
    otherwise exactly ``rounds`` (default ``n_estimators``) trees are trained.
    """
    xgb = _import_xgboost()
    train = _usable(train, config)
    if train.n_rows == 0 or train.labels.sum() == 0:
        raise ValueError("The training users have no relevant candidate")
    dtrain = xgb.QuantileDMatrix(train.features, label=train.labels, qid=train.group_ids(), max_bin=config.max_bin)
    params = _xgb_params(config, cutoff)
    use_validation = valid is not None and config.early_stopping_rounds > 0
    if use_validation:
        valid = _usable(valid, config)
        dvalid = xgb.QuantileDMatrix(valid.features, label=valid.labels, qid=valid.group_ids(),
                                     ref=dtrain, max_bin=config.max_bin)
        booster = xgb.train(params, dtrain, config.n_estimators, evals=[(dvalid, "valid")],
                            early_stopping_rounds=config.early_stopping_rounds, verbose_eval=False)
        return booster, int(booster.best_iteration) + 1
    rounds = rounds or config.n_estimators
    return xgb.train(params, dtrain, rounds, verbose_eval=False), rounds


def predict_scores(booster, features: np.ndarray, rounds: int) -> np.ndarray:
    """Margins of the first ``rounds`` trees (only their order matters)."""
    xgb = _import_xgboost()
    return booster.predict(xgb.DMatrix(features), iteration_range=(0, rounds), output_margin=True).astype(np.float32)


def importance_share(boosters: Sequence[Any], names: Sequence[str]) -> pd.Series:
    """Share of the total gain of every feature, averaged over models, largest first."""
    total = np.zeros(len(names))
    for booster in boosters:
        gain = booster.get_score(importance_type="total_gain")
        values = np.array([gain.get(f"f{i}", 0.0) for i in range(len(names))])
        total += values / max(values.sum(), 1e-12)
    return pd.Series(total / max(len(boosters), 1), index=list(names)).sort_values(ascending=False)


@dataclass
class CrossValidation:
    """Out-of-fold scores of every candidate row and what was learned on the way."""

    scores: np.ndarray
    rounds: List[int]
    fold_ndcg: List[float]
    importance: pd.Series
    seconds: float


def make_folds(n_users: int, folds: int, seed: int) -> np.ndarray:
    fold_of = np.empty(n_users, dtype=np.int64)
    fold_of[np.random.default_rng(seed).permutation(n_users)] = np.arange(n_users) % folds
    return fold_of


def cross_validate(
    data: RankerData,
    config: RankerConfig,
    cutoff: int = 20,
    folds: int = 4,
    inner_fraction: float = 0.2,
) -> CrossValidation:
    """User-level cross-validation: every user is scored by a model that never saw them.

    The number of trees of each fold is chosen on ``inner_fraction`` of its *training* users
    (early stopping), never on the users being scored, so the out-of-fold scores are honest.
    """
    if folds < 2:
        raise ValueError("Cross-validation needs at least 2 folds")
    fold_of = make_folds(data.n_users, folds, config.seed)
    scores = np.zeros(data.n_rows, dtype=np.float32)
    rounds, fold_ndcg, boosters = [], [], []
    rng = np.random.default_rng(config.seed + 1)
    started = time.perf_counter()
    for fold in range(folds):
        train_positions = np.flatnonzero(fold_of != fold)
        test_positions = np.flatnonzero(fold_of == fold)
        valid_positions = None
        if config.early_stopping_rounds > 0:
            shuffled = rng.permutation(train_positions)
            n_valid = max(int(len(shuffled) * inner_fraction), 1)
            valid_positions, train_positions = shuffled[:n_valid], shuffled[n_valid:]
        train = data.subset(train_positions)
        valid = None if valid_positions is None else data.subset(valid_positions)
        booster, used = train_booster(train, valid, config, cutoff)
        del train, valid
        test = data.subset(test_positions)
        test_scores = predict_scores(booster, test.features, used)
        ndcg = float(ranking_metrics(test, test_scores, cutoff)[0].mean())
        rows = np.concatenate([np.arange(data.indptr[p], data.indptr[p + 1]) for p in test_positions])
        scores[rows] = test_scores
        rounds.append(used)
        fold_ndcg.append(ndcg)
        boosters.append(booster)
        logger.info("Fold %d/%d: %d trees, held-out NDCG@%d %.4f (%.0fs)",
                    fold + 1, folds, used, cutoff, ndcg, time.perf_counter() - started)
    return CrossValidation(scores, rounds, fold_ndcg, importance_share(boosters, data.feature_names),
                           time.perf_counter() - started)


# ------------------------------------------------------------------- isolated XGBoost worker
def save_data(data: RankerData, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("users", "indptr", "items", "features", "labels"):
        np.save(directory / f"{name}.npy", getattr(data, name))
    if data.n_relevant is not None:
        np.save(directory / "n_relevant.npy", data.n_relevant)
    (directory / "feature_names.json").write_text(json.dumps(data.feature_names))


def load_data(directory: Path) -> RankerData:
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r" if name == "features" else None)
              for name in ("users", "indptr", "items", "features", "labels")}
    relevant = directory / "n_relevant.npy"
    return RankerData(feature_names=json.loads((directory / "feature_names.json").read_text()),
                      n_relevant=np.load(relevant) if relevant.exists() else None, **arrays)


def _worker_main(job_path: str) -> None:
    """Entry point of the XGBoost worker process: run one job described by ``job.pkl``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", force=True)
    work = Path(job_path).parent
    job = pickle.loads(Path(job_path).read_bytes())
    operation, config, cutoff = job["op"], job["config"], job.get("cutoff", 20)
    result: Dict[str, Any] = {}
    if operation == "cv":
        cv = cross_validate(load_data(work / "data"), config, cutoff, job["folds"], job["inner_fraction"])
        np.save(work / "scores.npy", cv.scores)
        result = {"rounds": cv.rounds, "fold_ndcg": cv.fold_ndcg, "importance": cv.importance,
                  "seconds": cv.seconds}
    elif operation == "fit":
        booster, used = train_booster(load_data(work / "data"), None, config, cutoff, rounds=job["rounds"])
        booster.save_model(str(work / "ranker.json"))
        result = {"rounds": used}
    elif operation == "predict":
        xgb = _import_xgboost()
        booster = xgb.Booster()
        booster.load_model(str(work / "ranker.json"))
        np.save(work / "scores.npy", predict_scores(booster, np.load(work / "features.npy"), job["rounds"]))
    else:
        raise ValueError(f"Unknown job {operation!r}")
    (work / "result.pkl").write_bytes(pickle.dumps(result))


def _run_worker(job: Dict[str, Any], work: Path) -> Dict[str, Any]:
    """Run ``job`` in a fresh Python process and return its result.

    A fresh process never has PyTorch loaded, so XGBoost cannot clash with it; if XGBoost still
    crashes, this process survives and says so.
    """
    job_path = work / "job.pkl"
    job_path.write_bytes(pickle.dumps(job))
    root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH", "")])))
    code = "import sys; from recsys.stacking import _worker_main; _worker_main(sys.argv[1])"
    completed = subprocess.run([sys.executable, "-c", code, str(job_path)], env=env)
    if completed.returncode != 0 or not (work / "result.pkl").exists():
        raise RuntimeError(
            f"The XGBoost worker process failed (exit code {completed.returncode}; a negative value is a "
            "crash signal, e.g. -11 = segmentation fault). Try --n-jobs 1, update xgboost, and run "
            "`python3 -X faulthandler -c \"import xgboost\"` to check that it imports on its own."
        )
    return pickle.loads((work / "result.pkl").read_bytes())


def cross_validate_isolated(
    data: RankerData, config: RankerConfig, cutoff: int = 20, folds: int = 4, inner_fraction: float = 0.2
) -> CrossValidation:
    """:func:`cross_validate` in a separate process (see the module docstring)."""
    with tempfile.TemporaryDirectory(prefix="ranker_") as directory:
        work = Path(directory)
        save_data(data, work / "data")
        result = _run_worker({"op": "cv", "config": config, "cutoff": cutoff, "folds": folds,
                              "inner_fraction": inner_fraction}, work)
        scores = np.load(work / "scores.npy")
    return CrossValidation(scores, result["rounds"], result["fold_ndcg"], result["importance"], result["seconds"])


# ----------------------------------------------------------------------------- the model
@dataclass
class StackedRanker:
    """A trained ranker with everything needed to re-create its features.

    ``model`` is the XGBoost model in its JSON form; training and prediction run in a worker
    process, so no XGBoost object lives in this one.
    """

    model: bytes
    rounds: int
    config: RankerConfig
    cutoff: int
    feature_names: List[str]
    component_labels: List[str]
    count_scale: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(cls, data: RankerData, config: RankerConfig, cutoff: int, rounds: int,
            component_labels: Sequence[str], count_scale: float, **extra: Any) -> "StackedRanker":
        with tempfile.TemporaryDirectory(prefix="ranker_") as directory:
            work = Path(directory)
            save_data(data, work / "data")
            result = _run_worker({"op": "fit", "config": config, "cutoff": cutoff, "rounds": rounds}, work)
            model = (work / "ranker.json").read_bytes()
        return cls(model, result["rounds"], config, cutoff, list(data.feature_names), list(component_labels),
                   count_scale, dict(extra))

    def predict(self, data: RankerData) -> np.ndarray:
        if list(data.feature_names) != self.feature_names:
            raise ValueError("The features of the data differ from the ones the ranker was trained on "
                             "(different components or ranker options)")
        with tempfile.TemporaryDirectory(prefix="ranker_") as directory:
            work = Path(directory)
            (work / "ranker.json").write_bytes(self.model)
            np.save(work / "features.npy", data.features)
            _run_worker({"op": "predict", "config": self.config, "rounds": self.rounds}, work)
            return np.load(work / "scores.npy")

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "ranker.json").write_bytes(self.model)
        (directory / "ranker_meta.json").write_text(json.dumps({
            "rounds": self.rounds, "config": self.config.to_dict(), "cutoff": self.cutoff,
            "feature_names": self.feature_names, "component_labels": self.component_labels,
            "count_scale": self.count_scale, "extra": self.extra,
        }, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "StackedRanker":
        from recsys.config import config_from_dict
        directory = Path(directory)
        meta_path, model_path = directory / "ranker_meta.json", directory / "ranker.json"
        if not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"No trained ranker in {directory}; run `stack_ranker.py cv` first")
        meta = json.loads(meta_path.read_text())
        return cls(model_path.read_bytes(), meta["rounds"], config_from_dict(RankerConfig, meta["config"]),
                   meta["cutoff"], meta["feature_names"], meta["component_labels"], meta["count_scale"],
                   meta.get("extra", {}))


def feature_statistics(data: RankerData) -> Dict[str, List[float]]:
    """Mean and standard deviation of every feature (stored with the ranker to detect a shift later)."""
    mean, std = data.features.mean(axis=0), data.features.std(axis=0)
    return {name: [float(m), float(s)] for name, m, s in zip(data.feature_names, mean, std)}


def feature_shift(reference: Dict[str, List[float]], data: RankerData) -> pd.Series:
    """Shift of every feature's mean in standard deviations of the training data, largest first.

    The submission is built from components trained on more data than the ranker saw, so scores
    and counts can drift; a feature far from 0 here (above about 0.3) deserves a look.
    """
    mean = data.features.mean(axis=0)
    shift = {name: (float(m) - reference[name][0]) / max(reference[name][1], 1e-9)
             for name, m in zip(data.feature_names, mean)}
    return pd.Series(shift).sort_values(key=np.abs, ascending=False)


def popularity_top(urm: sp.csr_matrix, k: int) -> np.ndarray:
    """The ``k`` most popular items (the answer for users without any interaction)."""
    popularity = np.asarray(urm.sum(axis=0)).ravel()
    return np.argsort(-popularity, kind="stable")[:k].astype(np.int32)


def assemble_recommendations(
    all_users: np.ndarray, data: RankerData, top_items: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """Lists for ``all_users``: the ranker's for the users in ``data``, ``fallback`` for the others."""
    all_users = np.asarray(all_users)
    recommendations = np.tile(fallback[None, :top_items.shape[1]], (len(all_users), 1)).astype(np.int32)
    order = np.argsort(all_users, kind="stable")
    position = order[np.searchsorted(all_users[order], data.users)]
    if not np.array_equal(all_users[position], data.users):
        raise ValueError("Some users of the ranker data are not in the requested users")
    recommendations[position] = top_items
    return recommendations
