"""Command line interface: ``python main.py {list-models,evaluate,submit}``.

One sub-command per model is generated automatically from the model's config
dataclass, so every hyper-parameter is available as a flag:

    python main.py evaluate item_knn --topk 200 --shrink 20 --similarity jaccard
    python main.py submit slim_bpr --epochs 30 --lr 0.005 --output submissions/slim_bpr.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from recsys.config import add_config_arguments, config_from_namespace
from recsys.data import SPLIT_MODES, load_dataset, write_submission
from recsys.experiment import (
    EvaluationOptions,
    ModelSpec,
    format_results,
    make_split,
    run_comparison,
)
from recsys.logging_utils import setup_logging
from recsys.models import MODEL_REGISTRY
from recsys.models.hybrid import fit_cached

logger = logging.getLogger(__name__)

_DEFAULTS = EvaluationOptions()


# --------------------------------------------------------------------- shared arguments
def _first_line(text: Optional[str]) -> str:
    return (text or "").strip().splitlines()[0] if text else ""


def add_evaluation_arguments(parser: argparse.ArgumentParser, with_defaults: bool = True) -> None:
    """Flags that define the offline evaluation protocol.

    With ``with_defaults=False`` every default is ``None`` so that callers can
    tell which flags were given explicitly (used to merge with a JSON file).
    """

    def default(value: Any) -> Any:
        return value if with_defaults else None

    def describe(text: str) -> str:
        return text + (" (default: %(default)s)" if with_defaults else "")

    group = parser.add_argument_group("evaluation protocol")
    group.add_argument("--data-dir", default=default(_DEFAULTS.data_dir),
                       help=describe("Folder with data_train.csv and data_target_users_test.csv"))
    group.add_argument("--validation-fraction", type=float, default=default(_DEFAULTS.validation_fraction),
                       help=describe("Fraction of the interactions held out as validation set (the "
                                     "competition test set is unlabelled). Shared by all models; used "
                                     "for early stopping and for the reported metrics"))
    group.add_argument("--split-mode", choices=SPLIT_MODES, default=default(_DEFAULTS.split_mode),
                       help=describe("global = random holdout over all interactions like the "
                                     "competition; per_user = hold out the same fraction of every user"))
    group.add_argument("--split-seed", type=int, default=default(_DEFAULTS.split_seed),
                       help=describe("Seed of the train/validation split"))
    group.add_argument("--cutoffs", type=int, nargs="+", default=default(list(_DEFAULTS.cutoffs)),
                       help=describe("Recommendation list lengths at which metrics are computed"))
    group.add_argument("--target-users-only", action=argparse.BooleanOptionalAction,
                       default=default(_DEFAULTS.target_users_only),
                       help="Evaluate only users listed in data_target_users_test.csv")
    group.add_argument("--subsample-users", type=int, default=default(_DEFAULTS.subsample_users),
                       help=describe("Keep only N random users for a quick smoke test, 0 = all"))
    group.add_argument("--rec-batch-size", type=int, default=default(_DEFAULTS.rec_batch_size),
                       help=describe("Users scored at once when generating recommendations"))
    group.add_argument("--output", default=default(_DEFAULTS.output),
                       help="Write the results table to this CSV file")


CACHE_DIR_HELP = ("Save the trained model in this folder and load it from there when an identical "
                  "model (same hyper-parameters and data) was trained before. Empty = no caching")


def resolve_evaluation_options(
    args: argparse.Namespace,
    file_options: Optional[Mapping[str, Any]] = None,
    extra_keys: Iterable[str] = ("models", "sort_by", "cache_dir"),
) -> EvaluationOptions:
    """Merge command line flags (highest priority), a JSON mapping and the defaults.

    ``extra_keys`` are the other keys the calling script accepts in the JSON file.
    """
    file_options = file_options or {}
    values = {}
    for name in EvaluationOptions.__dataclass_fields__:
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            values[name] = cli_value
        elif name in file_options:
            values[name] = file_options[name]
    unknown = set(file_options) - set(EvaluationOptions.__dataclass_fields__) - set(extra_keys)
    if unknown:
        raise ValueError(f"Unknown option(s) in experiment file: {sorted(unknown)}")
    return EvaluationOptions(**values)


def _verbosity_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("-v", "--verbose", action="store_true", help="Show debug logging")
    return parent


# ------------------------------------------------------------------------------ commands
def _run_list_models(args: argparse.Namespace) -> None:
    width = max(len(name) for name in MODEL_REGISTRY)
    for name, cls in MODEL_REGISTRY.items():
        print(f"{name:<{width}}  {_first_line(cls.__doc__)}")
    print("\nShow the hyper-parameters of a model with:  python main.py evaluate <model> --help")


def _run_evaluate(args: argparse.Namespace) -> None:
    options = resolve_evaluation_options(args)
    config = config_from_namespace(MODEL_REGISTRY[args.model].Config, args)
    dataset = load_dataset(options.data_dir)
    split = make_split(dataset, options)
    spec = ModelSpec(args.model, config.to_dict(), args.model)
    results = run_comparison(
        [spec], split, options, continue_on_error=False, cache_dir=args.cache_dir
    )
    print("\n" + format_results(results))
    if options.output:
        Path(options.output).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(options.output, index=False)
        logger.info("Results written to %s", options.output)


def _run_submit(args: argparse.Namespace) -> None:
    cls = MODEL_REGISTRY[args.model]
    config = config_from_namespace(cls.Config, args)
    dataset = load_dataset(args.data_dir)
    model = fit_cached(args.model, config.to_dict(), dataset.urm, None, args.cache_dir)
    recommendations = model.recommend(
        dataset.target_users, k=args.k, batch_size=args.rec_batch_size
    )
    output = args.output or f"submissions/{args.model}.csv"
    write_submission(output, dataset, dataset.target_users, recommendations)
    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_model, "wb") as handle:
            pickle.dump(model, handle)
        logger.info("Model saved to %s", args.save_model)
    logger.info("Hyper-parameters used: %s", json.dumps(config.to_dict()))


# ------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Recommender systems for the location-recommendation challenge.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    lister = commands.add_parser("list-models", help="List the available models")
    lister.set_defaults(handler=_run_list_models, verbose=False)

    evaluate = commands.add_parser(
        "evaluate", help="Train on a train/validation split and report NDCG and MAP on the validation set"
    )
    evaluate_models = evaluate.add_subparsers(dest="model", required=True, metavar="model")
    submit = commands.add_parser(
        "submit", help="Train on all the data and write a submission CSV"
    )
    submit_models = submit.add_subparsers(dest="model", required=True, metavar="model")

    for name, cls in MODEL_REGISTRY.items():
        description = _first_line(cls.__doc__)

        eval_parser = evaluate_models.add_parser(
            name, help=description, description=description, parents=[_verbosity_parent()]
        )
        add_evaluation_arguments(eval_parser)
        has_own_cache_flag = "cache_dir" in cls.Config.__dataclass_fields__  # hybrids
        if not has_own_cache_flag:
            eval_parser.add_argument_group("model cache").add_argument(
                "--cache-dir", default="", help=CACHE_DIR_HELP + " (default: off)")
        add_config_arguments(eval_parser.add_argument_group(f"{name} hyper-parameters"), cls.Config)
        eval_parser.set_defaults(handler=_run_evaluate)

        submit_parser = submit_models.add_parser(
            name, help=description, description=description, parents=[_verbosity_parent()]
        )
        submit_group = submit_parser.add_argument_group("submission")
        submit_group.add_argument("--data-dir", default="data",
                                  help="Folder with the competition CSV files (default: %(default)s)")
        submit_group.add_argument("--output", default=None,
                                  help="Submission file (default: submissions/<model>.csv)")
        submit_group.add_argument("--k", type=int, default=20,
                                  help="Items recommended per user (default: %(default)s)")
        submit_group.add_argument("--rec-batch-size", type=int, default=1000,
                                  help="Users scored at once (default: %(default)s)")
        submit_group.add_argument("--save-model", default=None,
                                  help="Also pickle the trained model to this path")
        if not has_own_cache_flag:
            submit_group.add_argument("--cache-dir", default="", help=CACHE_DIR_HELP + " (default: off)")
        add_config_arguments(submit_parser.add_argument_group(f"{name} hyper-parameters"), cls.Config)
        submit_parser.set_defaults(handler=_run_submit)
    return parser


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    args.handler(args)
