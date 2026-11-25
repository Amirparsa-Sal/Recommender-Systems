"""Compare several models (or hyper-parameter configurations) on the same split.

Examples:
    python evaluate.py --models top_popular item_knn user_knn
    python evaluate.py --config experiments/example.json --output results/run1.csv
    python evaluate.py --config experiments/example.json --subsample-users 2000   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from recsys.cli import CACHE_DIR_HELP, add_evaluation_arguments, resolve_evaluation_options
from recsys.data import load_dataset
from recsys.experiment import expand_specs, format_results, make_split, run_comparison
from recsys.logging_utils import setup_logging
from recsys.models import MODEL_REGISTRY

logger = logging.getLogger("evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate several models on the same train/validation split and "
                    "print a comparison table. Command line flags override the JSON file.",
    )
    parser.add_argument("--config", help="JSON experiment file (see experiments/example.json)")
    parser.add_argument("--models", nargs="*", default=[], choices=sorted(MODEL_REGISTRY),
                        help="Models to evaluate with their default hyper-parameters "
                             "(added to the ones in --config)")
    parser.add_argument("--sort-by", default=None,
                        help="Metric used to sort the table (default: ndcg@<largest cutoff>)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop at the first model that raises an error (default: record it and continue)")
    parser.add_argument("--cache-dir", default=None,
                        help=CACHE_DIR_HELP + " (also the \"cache_dir\" key of the JSON file; default: off)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logging")
    add_evaluation_arguments(parser, with_defaults=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    file_config = {}
    if args.config:
        file_config = json.loads(Path(args.config).read_text())
    options = resolve_evaluation_options(args, file_config)

    entries = list(file_config.get("models", [])) + [{"model": name} for name in args.models]
    if not entries:
        parser.error("nothing to evaluate: pass --models and/or --config with a 'models' list")
    specs = expand_specs(entries)
    logger.info("Comparing %d model configuration(s)", len(specs))

    dataset = load_dataset(options.data_dir)
    split = make_split(dataset, options)
    results = run_comparison(
        specs, split, options,
        continue_on_error=not args.fail_fast,
        sort_by=args.sort_by or file_config.get("sort_by"),
        cache_dir=args.cache_dir if args.cache_dir is not None else file_config.get("cache_dir", ""),
    )

    print("\n" + format_results(results))
    if options.output:
        Path(options.output).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(options.output, index=False)
        logger.info("Results (including hyper-parameters) written to %s", options.output)


if __name__ == "__main__":
    main()
