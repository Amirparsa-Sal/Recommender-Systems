"""Learned ranker (XGBoost) on top of several recommenders: evaluate it, then write a submission.

The components (the same ``components`` list as tune_blend.py / tune_gated_blend.py, e.g.
experiments/analyze_bests.json) propose candidates; XGBoost learns to re-order them from features of
every (user, candidate) pair.  See ``recsys/stacking.py`` for the method.

Two steps:

``cv``      Trains the components on the training split (or loads them from --cache-dir, so
            nothing is trained again if analyze_models.py / tune_blend.py ran before), builds the
            candidates and features of the validation users, and cross-validates the ranker over
            *users* (a user is only scored by a ranker that never saw them).  Prints the single
            components, a plain blend and the ranker side by side, with paired standard errors,
            feature importances and an exact check through the real evaluation metric.  Then trains
            the final ranker on all the users and saves it to --out-dir.
``submit``  Retrains the components on all the data (again through --cache-dir), builds the
            candidates of the target users, re-orders them with the saved ranker and writes the
            submission file.

Examples:
    python3 stack_ranker.py cv --config experiments/analyze_bests.json
    python3 stack_ranker.py cv --config experiments/analyze_bests.json --candidates 100 --max-depth 5
    python3 stack_ranker.py cv --config experiments/analyze_bests.json --subsample-users 3000   # smoke test
    python3 stack_ranker.py submit --config experiments/analyze_bests.json --output submissions/ranker.csv

Ranker options (flags, or the "ranker" object of the JSON file; flags win) are listed by ``--help``.
Everything else (``components``, ``cache_dir``, the split options) is read from the same JSON files
as the tuning scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from recsys.cli import add_evaluation_arguments, resolve_evaluation_options
from recsys.config import add_config_arguments, config_from_dict
from recsys.data import load_dataset, write_submission
from recsys.experiment import make_split
from recsys.logging_utils import setup_logging
from recsys.metrics import evaluate_recommendations
from recsys.models import hybrid
from recsys.models.hybrid import fit_component
from recsys.stacking import (
    RankerConfig,
    StackedRanker,
    assemble_recommendations,
    build_features,
    candidate_recall,
    cross_validate_isolated,
    feature_shift,
    feature_statistics,
    popularity_top,
    ranking_metrics,
    top_k_items,
)
from tune_blend import FILE_KEYS as TUNE_FILE_KEYS
from tune_blend import load_components, pick

logger = logging.getLogger("stack_ranker")

# "weight_steps", "n_buckets", ... belong to the other scripts; they are accepted so that the same file works.
FILE_KEYS = set(TUNE_FILE_KEYS) | {
    "out_dir", "n_buckets", "bucket_edges", "min_bucket_users", "folds", "weight_steps", "ranker",
    "inner_fraction", "feature_batch_size", "model_dir",
}


# ----------------------------------------------------------------------------- arguments
def add_ranker_arguments(parser: argparse.ArgumentParser) -> None:
    """One flag per ranker option; a flag that is not given stays out of the namespace."""
    group = parser.add_argument_group(
        "ranker options", "Also settable in the \"ranker\" object of the JSON file (flags win)."
    )
    add_config_arguments(group, RankerConfig)
    for action in group._group_actions:
        action.help = action.help.replace("%(default)s", str(action.default))
        action.default = argparse.SUPPRESS


def common_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--config", help="JSON file with the components (e.g. experiments/analyze_bests.json)")
    parent.add_argument("--components", help="Components: JSON list or @file.json")
    parent.add_argument("--cache-dir", default=None,
                        help="Trained-component cache shared with evaluate.py / tune_blend.py (default: cache)")
    parent.add_argument("--out-dir", default=None,
                        help="Folder of the trained ranker and the tables (default: results/ranker)")
    parent.add_argument("-v", "--verbose", action="store_true", help="Show debug logging")
    return parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XGBoost ranker over the candidates of several models. "
                    "`cv` evaluates it and trains the final ranker, `submit` writes the submission.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    cv = commands.add_parser("cv", parents=[common_parent()], help="Cross-validate the ranker and train the final one",
                             description="Cross-validate the ranker over users and save the final ranker.")
    cv.add_argument("--folds", type=int, default=None, help="Cross-validation folds of the users (default: 4)")
    cv.add_argument("--inner-fraction", type=float, default=None,
                    help="Share of the training users of every fold used to pick the number of trees (default: 0.2)")
    cv.add_argument("--feature-batch-size", type=int, default=None,
                    help="Users scored at once while building the features (default: 250)")
    add_evaluation_arguments(cv, with_defaults=False)
    add_ranker_arguments(cv)
    cv.set_defaults(handler=run_cv)

    submit = commands.add_parser("submit", parents=[common_parent()], help="Write a submission with the saved ranker",
                                 description="Train the components on all the data and re-rank their candidates.")
    submit.add_argument("--data-dir", default=None, help="Folder with the competition CSV files (default: data)")
    submit.add_argument("--output", default=None, help="Submission file (default: submissions/ranker.csv)")
    submit.add_argument("--k", type=int, default=None, help="Items recommended per user (default: 20)")
    submit.add_argument("--feature-batch-size", type=int, default=None,
                        help="Users scored at once while building the features (default: 250)")
    submit.set_defaults(handler=run_submit)
    return parser


def load_file_config(args: argparse.Namespace) -> Dict[str, Any]:
    return json.loads(Path(args.config).read_text()) if args.config else {}


def read_components(args: argparse.Namespace, file_config: Dict[str, Any], fallback: Optional[list] = None):
    raw = args.components if args.components is not None else file_config.get("components", fallback)
    if raw is None:
        raise SystemExit("no components: pass --components or put a 'components' list in --config")
    if isinstance(raw, str) and not raw.startswith("@"):
        raw = json.loads(raw)
    return load_components(raw)


def ranker_config(args: argparse.Namespace, file_config: Dict[str, Any]) -> RankerConfig:
    values = dict(file_config.get("ranker", {}))
    values.update({f.name: getattr(args, f.name) for f in fields(RankerConfig) if hasattr(args, f.name)})
    return config_from_dict(RankerConfig, values)


# ----------------------------------------------------------------------------- reporting
def user_buckets(lengths: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Quartile buckets of the profile length (for the breakdown table)."""
    edges = sorted({int(round(v)) for v in np.quantile(lengths, [0.25, 0.5, 0.75])})
    bucket = np.searchsorted(np.asarray(edges), lengths, side="right")
    names = [f"<{edges[0]}"] + [f"{lo}-{hi - 1}" for lo, hi in zip(edges, edges[1:])] + [f"{edges[-1]}+"]
    return bucket, names


def paired(a: np.ndarray, b: np.ndarray) -> str:
    difference = a - b
    gain, se = float(difference.mean()), float(difference.std(ddof=1) / np.sqrt(len(difference)))
    return f"{gain:+.4f} +- {se:.4f} ({'above noise' if abs(gain) > 2 * se else 'within noise'})"


# ----------------------------------------------------------------------------- cv
def run_cv(args: argparse.Namespace) -> None:
    file_config = load_file_config(args)
    options = resolve_evaluation_options(args, file_config, extra_keys=FILE_KEYS)
    specs, labels, _, _ = read_components(args, file_config)
    config = ranker_config(args, file_config)
    cache_dir = pick(args, file_config, "cache_dir", "cache")
    out_dir = Path(pick(args, file_config, "out_dir", "results/ranker"))
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = int(pick(args, file_config, "folds", 4))
    inner_fraction = float(pick(args, file_config, "inner_fraction", 0.2))
    cutoff = max(options.cutoffs)
    logger.info("Ranker configuration: %s", json.dumps(config.to_dict()))

    dataset = load_dataset(options.data_dir)
    split = make_split(dataset, options)
    hybrid.FIT_CACHE_SIZE = max(hybrid.FIT_CACHE_SIZE, len(specs) + 2)
    models = [fit_component(spec, split.train, split.validation, cache_dir=cache_dir) for spec in specs]

    data = build_features(
        models, labels, split.train, split.users, config, cutoff, validation=split.validation,
        batch_size=int(pick(args, file_config, "feature_batch_size", 250)),
    )
    del models
    ceiling = candidate_recall(data, cutoff)
    print(f"\nCandidates: {data.n_users} users with a training profile, {ceiling['candidates_per_user']:.0f} candidates "
          f"per user, {data.features.shape[1]} features.\nThe candidates contain {ceiling['candidate_recall']:.1%} of the "
          f"held-out interactions; a perfect re-ordering would reach NDCG@{cutoff} {ceiling['ndcg_ceiling']:.4f}.")

    cv = cross_validate_isolated(data, config, cutoff, folds, inner_fraction)

    # ---- the methods, all on the same candidates and the same users
    column = {name: i for i, name in enumerate(data.feature_names)}
    methods: Dict[str, np.ndarray] = {}
    for label in labels:
        methods[label] = data.features[:, column[f"max:{label}"]]
    single_names = list(methods)
    weights_note = "blend_weights" if config.blend_weights else "equal weights"
    methods[f"blend of the components ({weights_note})"] = data.features[:, column["blend_max"]]
    methods["reciprocal rank fusion"] = data.features[:, column["rrf"]]
    methods[f"XGBoost ranker ({folds}-fold, out of sample)"] = cv.scores
    metrics = {name: ranking_metrics(data, scores, cutoff) for name, scores in methods.items()}
    summary = pd.DataFrame({name: {f"ndcg@{cutoff}": m[0].mean(), f"map@{cutoff}": m[1].mean()}
                            for name, m in metrics.items()}).T
    ranker_name, blend_name = list(methods)[-1], list(methods)[-3]
    best_single = max(single_names, key=lambda name: metrics[name][0].mean())

    print(f"\nNDCG@{cutoff} on the {data.n_users} validation users with a training profile "
          f"(same candidates; the ranker's numbers are out of sample):")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    ranker_ndcg = metrics[ranker_name][0]
    print(f"\nPaired differences over users (mean +- standard error):")
    print(f"  ranker - best single ({best_single}): {paired(ranker_ndcg, metrics[best_single][0])}")
    print(f"  ranker - blend: {paired(ranker_ndcg, metrics[blend_name][0])}")
    print(f"  trees per fold: {cv.rounds}; held-out NDCG per fold: {[round(v, 4) for v in cv.fold_ndcg]}")

    lengths = np.diff(split.train.indptr)[data.users]
    bucket, bucket_names = user_buckets(lengths)
    rows = {}
    for name in (best_single, blend_name, ranker_name):
        rows[name] = [metrics[name][0][bucket == b].mean() for b in range(len(bucket_names))]
    print("\nNDCG by profile length (quartiles of the training interactions per user):")
    print(pd.DataFrame(rows, index=[f"{n} ({int((bucket == b).sum())} users)" for b, n in enumerate(bucket_names)]).T
          .to_string(float_format=lambda v: f"{v:.4f}"))

    top = cv.importance.head(20)
    print("\nMost useful features (share of the total gain, averaged over the folds):")
    print("  " + "\n  ".join(f"{name:<40s} {value:6.1%}" for name, value in top.items()))

    # ---- exact check with the real metric, cold users (empty profile) get the popularity list
    fallback = popularity_top(split.train, cutoff)
    exact = {}
    for name in (best_single, blend_name, ranker_name):
        items = top_k_items(data, methods[name], cutoff)[0]
        recommendations = assemble_recommendations(split.users, data, items, fallback)
        exact[name] = evaluate_recommendations(recommendations, split.validation, split.users, [cutoff])[f"ndcg@{cutoff}"]
    print(f"\nExact NDCG@{cutoff} on all {len(split.users)} validation users (users without a training profile get "
          "the popularity list):")
    for name, value in exact.items():
        print(f"  {name}: {value:.4f}")

    # ---- final ranker on every user
    rounds = max(int(np.ceil(np.mean(cv.rounds) * 1.1)), 1)  # a bit more trees: it now trains on more users
    components = [{"model": spec.model, "params": spec.params} for spec in specs]
    ranker = StackedRanker.fit(
        data, config, cutoff, rounds, labels, count_scale=1.0 - options.validation_fraction,
        components=components, cv_ndcg=float(ranker_ndcg.mean()),
        feature_stats=feature_statistics(data),
    )
    ranker.save(out_dir)
    summary.to_csv(out_dir / "cv_summary.csv")
    cv.importance.rename("gain_share").to_csv(out_dir / "feature_importance.csv")
    logger.info("Final ranker (%d trees) and tables written to %s", rounds, out_dir)

    config_arg = f"--config {args.config}" if args.config else f"--components {args.components}"
    print(f"\nFinal ranker trained on all {data.n_users} users with {rounds} trees, saved to {out_dir}/.")
    print("To write the submission (components retrained on all the data, through the same cache):")
    print(f"  python3 stack_ranker.py submit {config_arg} --out-dir {out_dir} --output submissions/ranker.csv")


# ----------------------------------------------------------------------------- submit
def run_submit(args: argparse.Namespace) -> None:
    file_config = load_file_config(args)
    out_dir = Path(pick(args, file_config, "out_dir", "results/ranker"))
    ranker = StackedRanker.load(out_dir)
    specs, labels, _, _ = read_components(args, file_config, fallback=ranker.extra.get("components"))
    if labels != ranker.component_labels:
        raise SystemExit(f"The components {labels} differ from the ones the ranker was trained with "
                         f"{ranker.component_labels}: run `cv` again")
    cache_dir = pick(args, file_config, "cache_dir", "cache")
    k = int(pick(args, file_config, "k", ranker.cutoff))
    data_dir = pick(args, file_config, "data_dir", "data")

    dataset = load_dataset(data_dir)
    hybrid.FIT_CACHE_SIZE = max(hybrid.FIT_CACHE_SIZE, len(specs) + 2)
    models = [fit_component(spec, dataset.urm, None, cache_dir=cache_dir) for spec in specs]
    data = build_features(
        models, labels, dataset.urm, dataset.target_users, ranker.config, ranker.cutoff,
        count_scale=ranker.count_scale, batch_size=int(pick(args, file_config, "feature_batch_size", 250)),
    )
    if "feature_stats" in ranker.extra:
        shift = feature_shift(ranker.extra["feature_stats"], data)
        print("Features whose mean moved most between the ranker's training data and now "
              "(in training standard deviations):")
        print("  " + "\n  ".join(f"{name:<40s} {value:+.2f}" for name, value in shift.head(5).items()))
        if abs(shift.iloc[0]) > 0.3:
            print("  WARNING: a large shift. Scale-dependent features (raw scores, counts) drift when the components "
                  "see more data; keep --no-raw-scores (the default) and compare the submission with the blend's.")
    items, _, _ = top_k_items(data, ranker.predict(data), k)
    recommendations = assemble_recommendations(dataset.target_users, data, items, popularity_top(dataset.urm, k))
    n_cold = len(dataset.target_users) - data.n_users
    logger.info("%d target users re-ranked, %d without any interaction got the popularity list", data.n_users, n_cold)
    if any(len(set(row)) != k for row in recommendations.tolist()):
        raise RuntimeError("A recommendation list contains duplicates")
    output = args.output or "submissions/ranker.csv"
    write_submission(output, dataset, dataset.target_users, recommendations)
    print(f"Submission written to {output}")


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    args.handler(args)


if __name__ == "__main__":
    main()
