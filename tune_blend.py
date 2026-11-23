"""Search the weights (and the subset of components) of a linear blend.

Every component is trained once (or loaded from --cache-dir), scored once, and then all
weight settings are evaluated quickly on the same train/validation split; see
``recsys/blend_search.py``.  The best setting is finally re-evaluated with the real
``hybrid_linear`` model to confirm the number.

Examples:
    python3 tune_blend.py --config experiments/tune_blend.json
    python3 tune_blend.py --config experiments/tune_blend.json --strategy coordinate
    python3 tune_blend.py --config experiments/tune_blend.json --normalize max zscore l2 \\
        --output results/tune_blend.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from recsys.blend_search import (
    CandidatePool,
    coordinate_search,
    grid_search,
    resolve_value_lists,
    results_frame,
)
from recsys.cli import add_evaluation_arguments, resolve_evaluation_options
from recsys.data import load_dataset
from recsys.experiment import EvaluationOptions, Split, make_split
from recsys.logging_utils import setup_logging
from recsys.metrics import evaluate_recommendations
from recsys.models import create_model
from recsys.models import hybrid
from recsys.models.hybrid import NORMALIZATIONS, ComponentSpec, fit_component, parse_components

logger = logging.getLogger("tune_blend")

FILE_KEYS = {
    "components", "weight_values", "component_weights", "normalize", "strategy", "candidates",
    "batch_size", "cache_dir", "top", "verify", "max_combinations", "max_rounds",
}
DEFAULT_WEIGHT_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0]


# --------------------------------------------------------------------------- configuration
def load_components(raw: Any) -> Tuple[List[ComponentSpec], List[str], List[Optional[list]], Optional[str]]:
    """Parse the components: specs, unique labels, per-component weight lists and the file used.

    ``raw`` is a list of model names / ``{"model", "params", "label", "weights"}`` objects, or
    ``"@file.json"`` holding such a list (the same file that ``--components`` accepts).
    """
    source = None
    if isinstance(raw, str):
        if not raw.startswith("@"):
            raise ValueError("components must be a JSON list or '@file.json'")
        source = raw[1:]
        raw = json.loads(Path(source).read_text())
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("components must be a list of at least two models")

    specs, labels, weight_lists = [], [], []
    for entry in raw:
        entry = {"model": entry} if isinstance(entry, str) else dict(entry)
        label = entry.pop("label", None)
        weights = entry.pop("weights", None)
        spec = parse_components([entry])[0]
        label = label or spec.model
        labels.append(label)
        specs.append(spec)
        weight_lists.append(weights)
    for index, label in enumerate(labels):  # make the labels unique
        if labels.count(label) > 1:
            labels[index] = f"{label}#{labels[:index].count(label) + 1}"
    return specs, labels, weight_lists, source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search the weights of a linear blend of models on the validation split. "
                    "Command line flags override the JSON file.",
    )
    parser.add_argument("--config", help="JSON file (see experiments/tune_blend.json)")
    parser.add_argument("--components",
                        help="Components: JSON list or @file.json (the file given to hybrid_linear)")
    parser.add_argument("--weight-values", type=float, nargs="+", default=None,
                        help="Candidate weights tried for every component "
                             f"(default: {DEFAULT_WEIGHT_VALUES}). Only ratios matter")
    parser.add_argument("--normalize", nargs="+", choices=NORMALIZATIONS, default=None,
                        help="Per-user score normalisations to compare (default: max)")
    parser.add_argument("--strategy", choices=("grid", "coordinate"), default=None,
                        help="grid = every combination; coordinate = one component at a time "
                             "(far fewer evaluations for many components). Default: grid")
    parser.add_argument("--candidates", type=int, default=None,
                        help="Top items of every component kept as candidates per user (default: 100)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Users scored at once while building the pool (default: 500)")
    parser.add_argument("--cache-dir", default=None,
                        help="Folder of the trained-component cache shared with hybrid_linear "
                             "(default: cache; '' = memory only)")
    parser.add_argument("--top", type=int, default=None, help="Rows of the table to print (default: 15)")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=None,
                        help="Re-evaluate the best setting with the real hybrid_linear (default: on)")
    parser.add_argument("--max-combinations", type=int, default=None,
                        help="Refuse grids larger than this (default: 20000)")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Passes of the coordinate strategy (default: 5)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logging")
    add_evaluation_arguments(parser, with_defaults=False)
    return parser


def pick(args: argparse.Namespace, file_config: Dict[str, Any], name: str, default: Any) -> Any:
    """Command line flag, else the JSON file, else the default."""
    value = getattr(args, name, None)
    if value is not None:
        return value
    return file_config.get(name, default)


# ------------------------------------------------------------------------------ workflow
def verify_best(
    specs: List[ComponentSpec], weights: List[float], normalization: str,
    split: Split, options: EvaluationOptions, cache_dir: str,
) -> Dict[str, float]:
    """Exact metrics of the blend through the real ``hybrid_linear`` (components come from the cache)."""
    model = create_model("hybrid_linear", {
        "components": [{"model": spec.model, "params": spec.params} for spec in specs],
        "weights": weights,
        "normalize": normalization,
        "cache": True,
        "cache_dir": cache_dir,
    })
    model.fit(split.train, validation=split.validation)
    cutoff = max(options.cutoffs)
    recommendations = model.recommend(split.users, k=cutoff, batch_size=options.rec_batch_size)
    return evaluate_recommendations(recommendations, split.validation, split.users, [cutoff])


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    file_config: Dict[str, Any] = {}
    if args.config:
        file_config = json.loads(Path(args.config).read_text())
    options = resolve_evaluation_options(args, file_config, extra_keys=FILE_KEYS)

    raw_components = args.components if args.components is not None else file_config.get("components")
    if raw_components is None:
        parser.error("no components: pass --components or put a 'components' list in --config")
    if isinstance(raw_components, str) and not raw_components.startswith("@"):
        raw_components = json.loads(raw_components)
    specs, labels, own_values, components_file = load_components(raw_components)

    default_values = pick(args, file_config, "weight_values", DEFAULT_WEIGHT_VALUES)
    value_lists = resolve_value_lists(
        labels, own_values, file_config.get("component_weights", {}), default_values
    )
    normalizations = pick(args, file_config, "normalize", ["max"])
    normalizations = [normalizations] if isinstance(normalizations, str) else list(normalizations)
    strategy = pick(args, file_config, "strategy", "grid")
    cache_dir = pick(args, file_config, "cache_dir", "cache")
    top = int(pick(args, file_config, "top", 15))
    cutoff = max(options.cutoffs)

    dataset = load_dataset(options.data_dir)
    split = make_split(dataset, options)

    hybrid.FIT_CACHE_SIZE = max(hybrid.FIT_CACHE_SIZE, len(specs) + 2)  # keep every component in memory
    models = [fit_component(spec, split.train, split.validation, cache_dir=cache_dir) for spec in specs]

    pool = CandidatePool(
        models, split.train, split.validation, split.users, cutoff=cutoff,
        candidates=int(pick(args, file_config, "candidates", 100)),
        normalizations=normalizations,
        batch_size=int(pick(args, file_config, "batch_size", 500)),
    )
    singles = {label: pool.evaluate([1.0 if i == j else 0.0 for j in range(len(specs))], normalizations[0])[0]
               for i, label in enumerate(labels)}
    logger.info("Single components (%s normalisation): %s", normalizations[0],
                {label: round(value, 4) for label, value in singles.items()})

    if strategy == "grid":
        rows = grid_search(pool, value_lists, normalizations,
                           max_combinations=int(pick(args, file_config, "max_combinations", 20000)))
    else:
        rows = coordinate_search(pool, value_lists, normalizations,
                                 max_rounds=int(pick(args, file_config, "max_rounds", 5)))
    results = results_frame(rows, labels, cutoff)

    print("\nSingle components:")
    print("  " + "\n  ".join(f"{label}: ndcg@{cutoff}={value:.4f}" for label, value in singles.items()))
    print(f"\nBest {min(top, len(results))} of {len(results)} evaluated settings "
          "(weights normalised to sum 1):")
    print(results.head(top).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    best = results.iloc[0]
    best_weights = [round(float(best[label]), 4) for label in labels]
    best_norm = str(best["normalize"])
    verified = None
    if pick(args, file_config, "verify", True):
        verified = verify_best(specs, best_weights, best_norm, split, options, cache_dir)
        print(f"\nBest setting: search ndcg@{cutoff}={best[f'ndcg@{cutoff}']:.4f}, "
              f"exact hybrid_linear ndcg@{cutoff}={verified[f'ndcg@{cutoff}']:.4f} "
              f"map@{cutoff}={verified[f'map@{cutoff}']:.4f}")
    else:
        print(f"\nBest setting: search ndcg@{cutoff}={best[f'ndcg@{cutoff}']:.4f} (not verified)")

    if components_file is None:
        components_file = "blend_components.json"
        Path(components_file).write_text(
            json.dumps([{"model": s.model, "params": s.params} for s in specs], indent=2)
        )
        logger.info("Components written to %s", components_file)
    command = (
        f"python3 main.py submit hybrid_linear --components @{components_file} "
        f"--weights {','.join(str(w) for w in best_weights)} --normalize {best_norm}"
    )
    if cache_dir:
        command += f" --cache-dir {cache_dir}"
    print("\nTo write the submission with these weights (trained on all the data):")
    print(f"  {command} --output submissions/blend.csv")
    print("Component order: " + ", ".join(labels))

    if options.output:
        Path(options.output).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(options.output, index=False)
        logger.info("All %d evaluated settings written to %s", len(results), options.output)


if __name__ == "__main__":
    main()
