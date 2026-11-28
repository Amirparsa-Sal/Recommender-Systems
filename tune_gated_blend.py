"""Do different models deserve different weights for users with short and long profiles?

A user with 5 known interactions and one with 80 pose different problems: with little to go on a
popularity-leaning or very smooth model may win, with a long profile a sharp item-item or
user-based model may.  This script tests whether weights that depend on the *profile length*
(``hybrid_gated``) beat one global linear blend, without fooling itself:

1. **Profile-length buckets** - the validation users (with at least one training interaction)
   are cut into buckets (``--n-buckets`` quantile buckets, or ``--bucket-edges``).
2. **Per-bucket report** - NDCG@K of every component in every bucket, the best component of each
   bucket, and what a *gating among single models* (the best component per bucket) would give.
3. **Cross-validated comparison** (``--folds``, default 2) - the users are split in folds; the
   weights are tuned on the other folds and evaluated on the held-out one, so every number is
   out-of-sample:  best single model  /  best single model per bucket  /  one global blend  /
   one blend per bucket.  The gain of the per-bucket blend over the global blend is reported with
   its paired standard error over users (below about two standard errors it is noise).
4. **Final weights** tuned on all users, verified exactly through the real ``hybrid_gated`` and
   ``hybrid_linear`` models, plus the ``main.py`` commands (``evaluate`` and ``submit``).

Weights are searched exactly like ``tune_blend.py`` does (candidate pool, ``--strategy
coordinate`` by default because it needs far fewer evaluations; ``grid`` tries every combination).
Everything uses one normalisation (``--normalize``, default ``max``), the one the gated model
will use.  Components come from ``--cache-dir`` (the cache shared with ``evaluate.py`` and
``tune_blend.py``) or are trained once.

Examples:
    python3 tune_gated_blend.py --config experiments/analyze_bests.json
    python3 tune_gated_blend.py --config experiments/analyze_bests.json --bucket-edges 5 10 20 50
    python3 tune_gated_blend.py --config experiments/analyze_bests.json --subsample-users 3000   # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from recsys.blend_search import CandidatePool, coordinate_search, grid_search, resolve_value_lists  # noqa: E402
from recsys.cli import add_evaluation_arguments, resolve_evaluation_options  # noqa: E402
from recsys.data import load_dataset  # noqa: E402
from recsys.experiment import make_split  # noqa: E402
from recsys.logging_utils import setup_logging  # noqa: E402
from recsys.metrics import evaluate_recommendations  # noqa: E402
from recsys.models import create_model, hybrid  # noqa: E402
from recsys.models.hybrid import NORMALIZATIONS, fit_component  # noqa: E402
from tune_blend import DEFAULT_WEIGHT_VALUES, load_components, pick  # noqa: E402
from tune_blend import FILE_KEYS as TUNE_FILE_KEYS  # noqa: E402

logger = logging.getLogger("tune_gated_blend")

# "weight_steps" belongs to analyze_models.py; it is accepted so that the same file can be used.
FILE_KEYS = set(TUNE_FILE_KEYS) | {"out_dir", "n_buckets", "bucket_edges", "min_bucket_users", "folds", "weight_steps"}

SURFACE, INK, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("sequential_blue", BLUE_RAMP)


# ----------------------------------------------------------------------------- arguments
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test whether blend weights that depend on the profile length of the user "
                    "(hybrid_gated) beat one global blend. Command line flags override the JSON file.",
    )
    parser.add_argument("--config", help="JSON file with the components (e.g. experiments/analyze_bests.json)")
    parser.add_argument("--components", help="Components: JSON list or @file.json")
    parser.add_argument("--bucket-edges", type=int, nargs="+", default=None,
                        help="Profile-length thresholds, e.g. 5 10 20 50 = buckets <5, 5-9, 10-19, 20-49, "
                             ">=50 (overrides --n-buckets)")
    parser.add_argument("--n-buckets", type=int, default=None,
                        help="Number of buckets with (about) the same number of users (default: 4)")
    parser.add_argument("--min-bucket-users", type=int, default=500,
                        help="Buckets with fewer users are merged with their smaller neighbour (default: 500)")
    parser.add_argument("--folds", type=int, default=None,
                        help="Cross-validation folds of the users (default: 2; 0 = skip the check)")
    parser.add_argument("--normalize", choices=NORMALIZATIONS, default=None,
                        help="Per-user score normalisation of the blend (default: max)")
    parser.add_argument("--strategy", choices=("grid", "coordinate"), default=None,
                        help="Weight search: coordinate (default) or grid")
    parser.add_argument("--weight-values", type=float, nargs="+", default=None,
                        help=f"Candidate weights of every component (default: {DEFAULT_WEIGHT_VALUES})")
    parser.add_argument("--candidates", type=int, default=None,
                        help="Top items of every component kept as candidates per user (default: 100)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Users scored at once while building the pool (default: 500)")
    parser.add_argument("--cache-dir", default=None,
                        help="Trained-component cache shared with evaluate.py / tune_blend.py (default: cache)")
    parser.add_argument("--max-combinations", type=int, default=None, help="Grid size limit (default: 20000)")
    parser.add_argument("--max-rounds", type=int, default=None, help="Passes of the coordinate search (default: 5)")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=None,
                        help="Re-evaluate the final weights with the real models (default: on)")
    parser.add_argument("--out-dir", default=None, help="Folder for the tables and the chart (default: results/gated_blend)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logging")
    add_evaluation_arguments(parser, with_defaults=False)
    return parser


# ----------------------------------------------------------------------------- buckets
def bucket_of(lengths: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    """Bucket index of every profile length (``edges`` are the lower bounds of the next bucket)."""
    return np.searchsorted(np.asarray(edges, dtype=np.int64), lengths, side="right")


def quantile_edges(lengths: np.ndarray, n_buckets: int) -> List[int]:
    cuts = np.quantile(lengths, np.linspace(0, 1, n_buckets + 1)[1:-1])
    return sorted({int(max(1, round(c))) for c in cuts})


def merge_small_buckets(lengths: np.ndarray, edges: List[int], minimum: int) -> List[int]:
    """Remove edges until every bucket has at least ``minimum`` users."""
    edges = list(edges)
    while edges:
        counts = np.bincount(bucket_of(lengths, edges), minlength=len(edges) + 1)
        smallest = int(np.argmin(counts))
        if counts[smallest] >= minimum:
            break
        left = counts[smallest - 1] if smallest > 0 else np.inf  # merge with the smaller neighbour
        right = counts[smallest + 1] if smallest < len(edges) else np.inf
        del edges[smallest - 1 if left <= right else smallest]
    return edges


def bucket_labels(edges: Sequence[int]) -> List[str]:
    if not edges:
        return ["all"]
    labels = [f"<{edges[0]}"]
    labels += [f"{lo}-{hi - 1}" if hi - 1 > lo else f"{lo}" for lo, hi in zip(edges, edges[1:])]
    return labels + [f"{edges[-1]}+"]


# ----------------------------------------------------------------------------- searching
class Searcher:
    """Best linear weights of a set of pool rows."""

    def __init__(self, pool: CandidatePool, value_lists, normalization: str, strategy: str,
                 max_combinations: int, max_rounds: int) -> None:
        self.pool, self.value_lists, self.normalization = pool, value_lists, normalization
        self.strategy, self.max_combinations, self.max_rounds = strategy, max_combinations, max_rounds

    def best_weights(self, rows: np.ndarray) -> Tuple[List[float], float]:
        sub = self.pool.subset(rows, [self.normalization])
        if self.strategy == "grid":
            found = grid_search(sub, self.value_lists, [self.normalization], self.max_combinations)
        else:
            found = coordinate_search(sub, self.value_lists, [self.normalization], self.max_rounds)
        best = max(found, key=lambda row: row["ndcg"])
        weights = np.asarray(best["weights"], dtype=float)
        return list(weights / weights.sum()), best["ndcg"]

    def user_ndcg(self, rows: np.ndarray, weights: Sequence[float]) -> np.ndarray:
        return self.pool.subset(rows, [self.normalization]).per_user(weights, self.normalization)[0]

    def user_ndcg_gated(self, rows: np.ndarray, buckets: np.ndarray, weights_by_bucket: Dict[int, Sequence[float]]) -> np.ndarray:
        out = np.zeros(len(rows))
        for b, weights in weights_by_bucket.items():
            mask = buckets[rows] == b
            if mask.any():
                out[mask] = self.user_ndcg(rows[mask], weights)
        return out


def tune_buckets(searcher: Searcher, rows: np.ndarray, buckets: np.ndarray, n_buckets: int,
                 global_weights: List[float], minimum: int) -> Dict[int, List[float]]:
    """Weights of every bucket tuned on ``rows``; small buckets fall back to the global weights."""
    result: Dict[int, List[float]] = {}
    for b in range(n_buckets):
        in_bucket = rows[buckets[rows] == b]
        result[b] = searcher.best_weights(in_bucket)[0] if len(in_bucket) >= minimum else global_weights
    return result


# ----------------------------------------------------------------------------- reporting
def paired_gain(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Mean of ``a - b`` and its standard error over users."""
    difference = a - b
    return float(difference.mean()), float(difference.std(ddof=1) / np.sqrt(len(difference)))


def save_chart(table: pd.DataFrame, counts: Sequence[int], labels: Sequence[str], cutoff: int, path: Path,
               highlight_rows: Sequence[str]) -> None:
    """Heat map: rows = models and blends, columns = buckets; colour = NDCG relative to the bucket's best single."""
    values = table.to_numpy(dtype=float)
    singles = [r for r in table.index if r not in highlight_rows]
    best = table.loc[singles].max(axis=0).to_numpy()
    relative = values / best[None, :]
    low, high = float(np.nanmin(relative)), float(np.nanmax(relative))
    fig, ax = plt.subplots(figsize=(1.6 + 1.25 * len(labels), 1.4 + 0.42 * len(table)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.pcolormesh(relative, cmap=CMAP, vmin=low, vmax=high, edgecolors=SURFACE, linewidth=2)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            dark = (relative[i, j] - low) / max(high - low, 1e-9) > 0.55
            ax.text(j + 0.5, i + 0.5, f"{values[i, j]:.4f}", ha="center", va="center", fontsize=9,
                    color="white" if dark else INK, fontweight="bold" if table.index[i] in highlight_rows else "normal")
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels([f"{label}\n{count} users" for label, count in zip(labels, counts)], color=INK, fontsize=9)
    ax.set_yticks(np.arange(len(table)) + 0.5)
    ax.set_yticklabels(table.index, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"NDCG@{cutoff} by profile length (number of training interactions)", color=INK, loc="left",
                 fontsize=11, fontweight="bold", pad=34)
    ax.text(0, 1.03, "colour: closer to the best single model of the bucket = darker\n"
                     "bold rows are blends tuned on these same users (optimistic)",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=8.5, linespacing=1.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    if not args.verbose:
        logging.getLogger("recsys.blend_search").setLevel(logging.WARNING)  # one line per search is too many

    file_config: Dict[str, Any] = json.loads(Path(args.config).read_text()) if args.config else {}
    options = resolve_evaluation_options(args, file_config, extra_keys=FILE_KEYS)
    raw = args.components if args.components is not None else file_config.get("components")
    if raw is None:
        parser.error("no components: pass --components or put a 'components' list in --config")
    if isinstance(raw, str) and not raw.startswith("@"):
        raw = json.loads(raw)
    specs, labels, own_values, components_file = load_components(raw)

    normalization = pick(args, file_config, "normalize", "max")
    normalization = normalization[0] if isinstance(normalization, list) else normalization
    strategy = pick(args, file_config, "strategy", "coordinate")
    cache_dir = pick(args, file_config, "cache_dir", "cache")
    folds = int(pick(args, file_config, "folds", 2))
    min_users = int(pick(args, file_config, "min_bucket_users", 500))
    out_dir = Path(pick(args, file_config, "out_dir", "results/gated_blend"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = max(options.cutoffs)
    value_lists = resolve_value_lists(
        labels, own_values, file_config.get("component_weights", {}),
        pick(args, file_config, "weight_values", DEFAULT_WEIGHT_VALUES),
    )

    dataset = load_dataset(options.data_dir)
    split = make_split(dataset, options)
    hybrid.FIT_CACHE_SIZE = max(hybrid.FIT_CACHE_SIZE, len(specs) + 2)
    models = [fit_component(spec, split.train, split.validation, cache_dir=cache_dir) for spec in specs]

    profile = np.diff(split.train.indptr)
    users = split.users[profile[split.users] > 0]  # empty profiles are popularity for every model
    pool = CandidatePool(
        models, split.train, split.validation, users, cutoff=cutoff,
        candidates=int(pick(args, file_config, "candidates", 100)),
        normalizations=[normalization], batch_size=int(pick(args, file_config, "batch_size", 500)),
    )
    n_users = len(users)
    lengths = profile[pool.users]

    explicit = pick(args, file_config, "bucket_edges", None)
    edges = sorted(set(int(e) for e in explicit)) if explicit else quantile_edges(lengths, int(pick(args, file_config, "n_buckets", 4)))
    if not explicit:
        edges = merge_small_buckets(lengths, edges, min_users)
    buckets = bucket_of(lengths, edges)
    n_buckets = len(edges) + 1
    names = bucket_labels(edges)
    counts = np.bincount(buckets, minlength=n_buckets)
    logger.info("%d users in %d buckets %s: %s", n_users, n_buckets, names, counts.tolist())
    if n_buckets == 1:
        parser.error("only one bucket: give --bucket-edges or lower --min-bucket-users")

    searcher = Searcher(pool, value_lists, normalization, strategy,
                        int(pick(args, file_config, "max_combinations", 20000)),
                        int(pick(args, file_config, "max_rounds", 5)))
    everyone = np.arange(n_users)
    singles = np.array([pool.per_user([1.0 if i == j else 0.0 for i in range(len(specs))], normalization)[0]
                        for j in range(len(specs))])  # (components, users)

    # ---- 1. per-bucket report of the single components
    per_bucket = pd.DataFrame({name: [singles[j][buckets == b].mean() for j in range(len(specs))]
                               for b, name in enumerate(names)}, index=labels)
    per_bucket["all users"] = singles.mean(axis=1)
    best_single_per_bucket = per_bucket.drop(columns="all users").idxmax(axis=0)
    gating_of_singles = np.mean([singles[labels.index(best_single_per_bucket[names[b]])][i]
                                 for i, b in enumerate(buckets)])
    print(f"\nNDCG@{cutoff} of the components by profile length (profile = training interactions):")
    print(pd.concat([pd.DataFrame([counts.tolist() + [n_users]], index=["users"], columns=per_bucket.columns), per_bucket]).to_string(
        float_format=lambda v: f"{v:.4f}"))
    print("\nBest single component per bucket: " + ", ".join(f"{names[b]}: {best_single_per_bucket[names[b]]}" for b in range(n_buckets)))
    print(f"Best single overall: {per_bucket['all users'].idxmax()} {per_bucket['all users'].max():.4f}; "
          f"choosing the best single per bucket (in-sample): {gating_of_singles:.4f}")

    # ---- 2. cross-validation (every number is out-of-sample)
    cv_rows = []
    if folds >= 2:
        rng = np.random.default_rng(options.split_seed)
        fold_of = np.empty(n_users, dtype=int)
        fold_of[rng.permutation(n_users)] = np.arange(n_users) % folds
        out = {name: np.zeros(n_users) for name in ("best single", "best single per bucket", "global blend", "blend per bucket")}
        for fold in range(folds):
            train, test = np.where(fold_of != fold)[0], np.where(fold_of == fold)[0]
            global_weights, _ = searcher.best_weights(train)
            bucket_weights = tune_buckets(searcher, train, buckets, n_buckets, global_weights, min_users // 2)
            best_overall = int(np.argmax(singles[:, train].mean(axis=1)))
            best_in_bucket = {b: int(np.argmax(singles[:, train[buckets[train] == b]].mean(axis=1))) for b in range(n_buckets)}
            out["best single"][test] = singles[best_overall][test]
            out["best single per bucket"][test] = [singles[best_in_bucket[buckets[i]]][i] for i in test]
            out["global blend"][test] = searcher.user_ndcg(test, global_weights)
            out["blend per bucket"][test] = searcher.user_ndcg_gated(test, buckets, bucket_weights)
            logger.info("fold %d/%d: global blend %.4f, blend per bucket %.4f (held out)", fold + 1, folds,
                        out["global blend"][test].mean(), out["blend per bucket"][test].mean())
        print(f"\nCross-validated NDCG@{cutoff} ({folds} folds of users; weights tuned on the other folds):")
        for name, values in out.items():
            cv_rows.append({"method": name, f"ndcg@{cutoff}": values.mean(),
                            **{names[b]: values[buckets == b].mean() for b in range(n_buckets)}})
        cv = pd.DataFrame(cv_rows).set_index("method")
        print(cv.to_string(float_format=lambda v: f"{v:.4f}"))
        for a, b in (("blend per bucket", "global blend"), ("blend per bucket", "best single per bucket"), ("global blend", "best single")):
            gain, se = paired_gain(out[a], out[b])
            verdict = "above noise" if abs(gain) > 2 * se else "within noise"
            print(f"  {a} - {b}: {gain:+.4f} +- {se:.4f} ({verdict})")
        cv.to_csv(out_dir / "cv_summary.csv")

    # ---- 3. final weights on all users
    global_weights, global_ndcg = searcher.best_weights(everyone)
    bucket_weights = tune_buckets(searcher, everyone, buckets, n_buckets, global_weights, min_users // 2)
    gated_user = searcher.user_ndcg_gated(everyone, buckets, bucket_weights)
    global_user = searcher.user_ndcg(everyone, global_weights)
    weights_table = pd.DataFrame([global_weights] + [bucket_weights[b] for b in range(n_buckets)],
                                 index=["global"] + names, columns=labels)
    print(f"\nFinal weights (tuned on all {n_users} users, normalised to sum 1; in-sample, so optimistic):")
    print(weights_table.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"In-sample NDCG@{cutoff}: global blend {global_user.mean():.4f}, blend per bucket {gated_user.mean():.4f}")
    weights_table.to_csv(out_dir / "weights.csv")

    chart_table = per_bucket.drop(columns="all users").copy()
    chart_table.loc["blend (global weights)"] = [global_user[buckets == b].mean() for b in range(n_buckets)]
    chart_table.loc["blend (weights per bucket)"] = [gated_user[buckets == b].mean() for b in range(n_buckets)]
    chart_table.assign(**{"all users": list(per_bucket["all users"]) + [global_user.mean(), gated_user.mean()]}) \
        .to_csv(out_dir / "by_profile_length.csv")
    save_chart(chart_table, counts.tolist(), names, cutoff, out_dir / "by_profile_length.png",
               ["blend (global weights)", "blend (weights per bucket)"])

    # ---- 4. exact check and commands
    rounded_global = [round(w, 4) for w in global_weights]
    rounded_buckets = [[round(w, 4) for w in bucket_weights[b]] for b in range(n_buckets)]
    component_params = [{"model": s.model, "params": s.params} for s in specs]
    if pick(args, file_config, "verify", True):
        exact = {}
        for name, params in (
            ("hybrid_linear", {"weights": rounded_global}),
            ("hybrid_gated", {"bucket_edges": edges, "bucket_weights": rounded_buckets}),
        ):
            model = create_model(name, {"components": component_params, "normalize": normalization,
                                        "cache": True, "cache_dir": cache_dir, **params})
            model.fit(split.train, validation=split.validation)
            recommendations = model.recommend(split.users, k=cutoff, batch_size=options.rec_batch_size)
            exact[name] = evaluate_recommendations(recommendations, split.validation, split.users, [cutoff])[f"ndcg@{cutoff}"]
        print(f"\nExact check on all {len(split.users)} validation users (cold users included): "
              f"hybrid_linear {exact['hybrid_linear']:.4f}, hybrid_gated {exact['hybrid_gated']:.4f}")

    if components_file is None:
        components_file = str(out_dir / "components.json")
        Path(components_file).write_text(json.dumps(component_params, indent=2))
    scale = 1.0 / (1.0 - options.validation_fraction)
    submit_edges = [max(1, int(round(e * scale))) for e in edges]
    common = (f"--components @{components_file} --normalize {normalization}"
              + (f" --cache-dir {cache_dir}" if cache_dir else ""))
    bucket_json = shlex.quote(json.dumps(rounded_buckets))
    print("\nTo evaluate the gated blend with the real model:")
    print(f"  python3 main.py evaluate hybrid_gated {common} --bucket-edges {','.join(map(str, edges))} --bucket-weights {bucket_json}")
    print(f"To write the submission (trained on all the data, so profiles are about {scale:.2f}x longer: edges scaled):")
    print(f"  python3 main.py submit hybrid_gated {common} --bucket-edges {','.join(map(str, submit_edges))} "
          f"--bucket-weights {bucket_json} --output submissions/gated.csv")
    print("Component order: " + ", ".join(labels))
    (out_dir / "gated_config.json").write_text(json.dumps({
        "components": component_params, "normalize": normalization, "bucket_edges": edges,
        "bucket_edges_for_submit": submit_edges, "bucket_weights": rounded_buckets, "global_weights": rounded_global,
    }, indent=2))
    logger.info("Tables, chart and gated_config.json written to %s", out_dir)


if __name__ == "__main__":
    main()
