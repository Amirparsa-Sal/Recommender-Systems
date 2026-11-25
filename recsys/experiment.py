"""Offline evaluation: building a split, running models, comparing results."""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.config import config_from_dict
from recsys.data import Dataset, SPLIT_MODES, random_holdout, restrict_to_users
from recsys.metrics import evaluate_recommendations
from recsys.models import get_model_class
from recsys.models.hybrid import fit_cached
from recsys.utils import Timer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationOptions:
    """Everything that defines an offline evaluation protocol."""

    data_dir: str = "data"
    validation_fraction: float = 0.2
    split_mode: str = "global"
    split_seed: int = 42
    cutoffs: Sequence[int] = (20,)
    target_users_only: bool = False
    subsample_users: int = 0
    rec_batch_size: int = 1000
    output: Optional[str] = None


@dataclass
class Split:
    """Train / validation partition shared by every model of a comparison.

    The competition test set has no labels, so the only labelled hold-out is the
    ``validation`` set carved out of ``data_train.csv``.  Models are trained on
    ``train``; those with early stopping also use ``validation`` to pick their
    number of epochs, and the reported metrics are computed on it for the users
    in ``users``.
    """

    train: sp.csr_matrix
    validation: sp.csr_matrix
    users: np.ndarray

    def describe(self) -> str:
        return (
            f"train={self.train.nnz}, validation={self.validation.nnz} interactions; "
            f"metrics on {len(self.users)} validation users"
        )


def make_split(dataset: Dataset, options: EvaluationOptions) -> Split:
    """Random holdout of the validation set from the training interactions."""
    if options.split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {SPLIT_MODES}")

    urm = dataset.urm
    if options.subsample_users > 0:
        rng = np.random.default_rng(options.split_seed)
        active = np.flatnonzero(np.diff(urm.indptr))
        chosen = rng.choice(active, size=min(options.subsample_users, len(active)), replace=False)
        urm = restrict_to_users(urm, chosen)
        logger.info("Subsampled the data to %d users", len(chosen))

    train, validation = random_holdout(
        urm, options.validation_fraction, options.split_seed, options.split_mode
    )
    users = np.flatnonzero(np.diff(validation.indptr) > 0)
    if options.target_users_only:
        users = np.intersect1d(users, dataset.target_users)
    if len(users) == 0:
        raise ValueError("No user has validation interactions; check the split options")

    split = Split(train, validation, users)
    logger.info("Split (%s, seed=%d): %s", options.split_mode, options.split_seed, split.describe())
    return split


@dataclass
class ModelSpec:
    """A model name with hyper-parameters (validated on creation) and a display label."""

    model: str
    params: Dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        config_from_dict(get_model_class(self.model).Config, self.params)  # fail early on typos
        if not self.label:
            self.label = self.model


def expand_specs(entries: Sequence[Mapping[str, Any]]) -> List[ModelSpec]:
    """Turn experiment entries into model specs, expanding ``grid`` entries.

    Each entry is ``{"model": name, "params": {...}, "label": "...", "grid": {param: [values]}}``.
    A grid creates one spec per combination of values.
    """
    specs: List[ModelSpec] = []
    for entry in entries:
        unknown = set(entry) - {"model", "params", "label", "grid"}
        if unknown:
            raise ValueError(f"Unknown key(s) {sorted(unknown)} in experiment entry {dict(entry)}")
        if "model" not in entry:
            raise ValueError(f"Experiment entry without 'model': {dict(entry)}")
        model = entry["model"]
        base_params = dict(entry.get("params", {}))
        base_label = entry.get("label") or model
        grid = entry.get("grid")
        if not grid:
            specs.append(ModelSpec(model, base_params, base_label))
            continue
        keys = list(grid)
        for values in itertools.product(*(grid[key] for key in keys)):
            params = {**base_params, **dict(zip(keys, values))}
            suffix = ",".join(f"{key}={value}" for key, value in zip(keys, values))
            specs.append(ModelSpec(model, params, f"{base_label}[{suffix}]"))
    return specs


def run_comparison(
    specs: Sequence[ModelSpec],
    split: Split,
    options: EvaluationOptions,
    continue_on_error: bool = True,
    sort_by: Optional[str] = None,
    cache_dir: str = "",
) -> pd.DataFrame:
    """Train and evaluate every spec on the same split; one result row per spec.

    With ``cache_dir`` a model identical to one trained in an earlier run (same model,
    hyper-parameters and split) is loaded from that folder instead of being trained again;
    ``fit_s`` is then the loading time.
    """
    cutoffs = sorted(options.cutoffs)
    sort_by = sort_by or f"ndcg@{cutoffs[-1]}"
    rows: List[Dict[str, Any]] = []

    for number, spec in enumerate(specs, start=1):
        logger.info("[%d/%d] %s: %s", number, len(specs), spec.label, spec.params or "default parameters")
        try:
            with Timer() as fit_timer:
                model = fit_cached(spec.model, spec.params, split.train, split.validation, cache_dir)
            with Timer() as recommend_timer:
                recommendations = model.recommend(
                    split.users, k=cutoffs[-1], batch_size=options.rec_batch_size
                )
            metrics = evaluate_recommendations(recommendations, split.validation, split.users, cutoffs)
            row = {
                "label": spec.label,
                "model": spec.model,
                **metrics,
                "fit_s": round(fit_timer.seconds, 1),
                "recommend_s": round(recommend_timer.seconds, 1),
                "params": json.dumps(model.config.to_dict()),
            }
            logger.info("[%d/%d] %s: %s=%.4f (fit %.1fs)", number, len(specs), spec.label,
                        sort_by, metrics.get(sort_by, float("nan")), fit_timer.seconds)
        except Exception as exc:  # keep comparing the other models
            if not continue_on_error:
                raise
            logger.exception("[%d/%d] %s failed", number, len(specs), spec.label)
            row = {"label": spec.label, "model": spec.model, "error": repr(exc)}
        rows.append(row)

    results = pd.DataFrame(rows)
    if sort_by in results.columns:
        results = results.sort_values(sort_by, ascending=False, na_position="last")
    return results.reset_index(drop=True)


def format_results(results: pd.DataFrame) -> str:
    """Compact table: model label, NDCG and MAP (plus an error column only if a run failed).

    Timings, model names and hyper-parameters stay in the DataFrame / CSV output.
    """
    columns = ["label"]
    columns += [c for c in results.columns if c.startswith(("ndcg@", "map@"))]
    if "error" in results.columns and results["error"].notna().any():
        columns.append("error")
    return results[columns].to_string(
        index=False, float_format=lambda value: f"{value:.4f}", na_rep="-"
    )
