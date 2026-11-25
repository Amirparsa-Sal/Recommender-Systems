"""Which components of a hybrid think alike, and which pairs actually blend well.

Every component is trained once (or loaded from ``--cache-dir``, the same cache as
``evaluate.py`` / ``tune_blend.py``), scored once on the validation users and reduced to a
pool of candidate items (the union of the components' top ``--candidates``).  Everything
below is computed from that pool, so the whole analysis takes about as long as one run of
``tune_blend.py`` with a small grid.

Core set of analyses (``K`` = the largest cut-off, 20 by default):

1. **Score correlation** - per-user Spearman correlation of two components' scores on the
   items that at least one of them ranks in its top ``--candidates`` (so a pair's value does
   not depend on the other components), averaged over users.  Scores are ranked per user, so
   this does not depend on how the scores are scaled; the components' very different score
   ranges (similarities, SLIM weights, walk probabilities, ...) are no problem.  This is the
   region a blend works on, but choosing the items by "high for at least one of the two"
   pushes the value down (it can even be negative): read it as a ranking of the pairs, not as
   a full-catalogue correlation.  A negative value means one model's favourites are the other's
   non-favourites.  Also shown: the average overlap of the top-K lists (``pairs.csv`` also has
   the overlap of the top-``--candidates`` lists, which has no selection effect).
2. **Unique hits and error correlation** - which validation items are found (inside the
   top-K) by one component but not by the other, and whether the components do well and
   badly on the same *users* (correlation of the per-user NDCG@K).
3. **Pairwise blend gain** - for every pair of components, the best linear blend (weights
   ``1-w`` / ``w``, all normalisations in ``--normalize``) against the better of the two
   components.  This is the ground truth the other numbers should be read against.
4. **Oracle headroom** - NDCG@K if every user were served by the component that is best for
   *that user*.  The gap to the best single component is the most any per-user combination
   can gain; for a pair it measures how complementary the two components are.  It contains
   luck (picking the better of two noisy per-user scores with the labels in hand is above
   average even for identical models), so compare pairs with each other, not with zero, and
   do not expect a blend to reach it.

Normalisation: rankings, overlaps and hit counts only depend on the order of the scores
within a user, so they need none.  The blend gain is the one place where scores of different
models are added, so there each component is normalised per user exactly as
``hybrid_linear`` does it (``max`` and ``zscore`` are compared by default; the better one is
reported per pair).

Choices worth knowing:

* Users without training interactions are left out: every model falls back to popularity
  for them, so they only add fake agreement.
* As in ``tune_blend.py`` the candidate pool has no popularity tie-break; numbers can differ
  from ``evaluate.py`` in the third decimal, but all components are treated alike.
* The best weight of a pair is picked on the same validation users, so the gains are
  slightly optimistic.  ``blend_gain_se`` is the paired standard error over users: a gain
  below about two standard errors is not distinguishable from noise.

Examples:
    python3 analyze_models.py --config experiments/tune_blend.json
    python3 analyze_models.py --components @experiments/blend_components.json --out-dir results/analysis
    python3 analyze_models.py --config experiments/tune_blend.json --subsample-users 3000   # smoke test
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from recsys.blend_search import CandidatePool  # noqa: E402
from recsys.cli import add_evaluation_arguments, resolve_evaluation_options  # noqa: E402
from recsys.data import load_dataset  # noqa: E402
from recsys.experiment import make_split  # noqa: E402
from recsys.logging_utils import setup_logging  # noqa: E402
from recsys.models import hybrid  # noqa: E402
from recsys.models.hybrid import NORMALIZATIONS, fit_component  # noqa: E402
from tune_blend import FILE_KEYS as TUNE_FILE_KEYS  # noqa: E402
from tune_blend import load_components, pick  # noqa: E402

logger = logging.getLogger("analyze_models")

FILE_KEYS = set(TUNE_FILE_KEYS) | {"out_dir", "max_users", "weight_steps"}

# Chart styling: one sequential blue ramp (light = low, dark = high), recessive ink.
SURFACE, INK, INK_SECONDARY, DIAGONAL = "#fcfcfb", "#0b0b0b", "#52514e", "#ecebe8"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("sequential_blue", BLUE_RAMP)


# ----------------------------------------------------------------------------- arguments
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agreement / complementarity analysis of the components of a linear blend. "
                    "Takes the same --config / --components as tune_blend.py; command line flags "
                    "override the JSON file.",
    )
    parser.add_argument("--config", help="JSON file (e.g. experiments/tune_blend.json)")
    parser.add_argument("--components",
                        help="Components: JSON list or @file.json (the file given to hybrid_linear)")
    parser.add_argument("--normalize", nargs="+", choices=NORMALIZATIONS, default=None,
                        help="Per-user score normalisations tried for the blend gain "
                             "(default: max zscore)")
    parser.add_argument("--candidates", type=int, default=None,
                        help="Top items of every component kept as candidates per user (default: 100)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Users scored at once while building the pool (default: 500)")
    parser.add_argument("--cache-dir", default=None,
                        help="Trained-component cache shared with evaluate.py / tune_blend.py "
                             "(default: cache; '' = memory only)")
    parser.add_argument("--max-users", type=int, default=None,
                        help="Analyse a random subset of the validation users (0 = all). The models "
                             "are still trained on the full split, so the cache stays valid")
    parser.add_argument("--weight-steps", type=int, default=None,
                        help="Interior weights tried per pair and normalisation: w = 1/(n+1) ... "
                             "n/(n+1) (default: 9, i.e. 0.1 ... 0.9)")
    parser.add_argument("--out-dir", default=None,
                        help="Folder for the PNG and CSV files (default: results/analysis)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logging")
    add_evaluation_arguments(parser, with_defaults=False)
    return parser


# ------------------------------------------------------------------------------ analyses
def one_hot(n: int, index: int) -> List[float]:
    weights = [0.0] * n
    weights[index] = 1.0
    return weights


def top_k_masks(pool: CandidatePool, k: int) -> np.ndarray:
    """``(models, users, candidates)`` booleans: the candidates in every component's top ``k``.

    Normalisations are monotone per user, so the first one gives every component's own ranking.
    """
    masks = np.zeros((pool.n_models, *pool.valid.shape), dtype=bool)
    for j in range(pool.n_models):
        scores = np.where(pool.valid, pool.scores[0, j], -np.inf)
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        np.put_along_axis(masks[j], top, True, axis=1)
    return masks


def pair_rank_correlation(pool: CandidatePool, candidate_masks: np.ndarray, a: int, b: int,
                          chunk: int = 2000) -> float:
    """Mean over users of the Spearman correlation between components ``a`` and ``b``.

    Computed on the items that at least one of the two ranks in its top ``--candidates`` (so the
    value of a pair does not depend on the other components).  Ties share their average rank.
    Users for whom either component gives constant scores on that set are skipped.
    """
    total, count = 0.0, 0
    for start in range(0, pool.valid.shape[0], chunk):
        rows = slice(start, start + chunk)
        used = pool.valid[rows] & (candidate_masks[a, rows] | candidate_masks[b, rows])
        weight = used.astype(np.float32)
        n_used = np.maximum(weight.sum(axis=1, keepdims=True), 1.0)
        centred = []
        for j in (a, b):
            scores = np.where(used, pool.scores[0, j, rows], -np.inf)
            ranks = rankdata(scores, axis=1, method="average").astype(np.float32)
            mean = (ranks * weight).sum(axis=1, keepdims=True) / n_used
            centred.append((ranks - mean) * weight)  # unused columns count 0
        covariance = (centred[0] * centred[1]).sum(axis=1)
        spread = np.sqrt((centred[0] ** 2).sum(axis=1) * (centred[1] ** 2).sum(axis=1))
        defined = spread > 0
        total += float((covariance[defined] / spread[defined]).sum())
        count += int(defined.sum())
    return total / count if count else float("nan")


def pairwise(function, n_models: int) -> np.ndarray:
    """Fill an (n, n) matrix with ``function(a, b)`` for every ordered pair (diagonal NaN)."""
    matrix = np.full((n_models, n_models), np.nan)
    for a, b in itertools.permutations(range(n_models), 2):
        matrix[a, b] = function(a, b)
    return matrix


def pair_blend(
    pool: CandidatePool, a: int, b: int, normalizations: Sequence[str],
    weights_b: Sequence[float], per_user_single: np.ndarray,
) -> Dict[str, Any]:
    """Best linear blend of components ``a`` and ``b`` (weights ``1-w`` / ``w``) against the better single."""
    n_models = pool.n_models
    singles = per_user_single.mean(axis=1)
    base = a if singles[a] >= singles[b] else b
    best = {"ndcg": singles[base], "weight_b": 0.0 if base == a else 1.0,
            "normalize": normalizations[0], "per_user": per_user_single[base]}
    for method in normalizations:
        for w in weights_b:
            weights = [0.0] * n_models
            weights[a], weights[b] = 1.0 - w, w
            ndcg = pool.evaluate(weights, method)[0]
            if ndcg > best["ndcg"]:
                best.update(ndcg=ndcg, weight_b=float(w), normalize=method, weights=weights, per_user=None)
    if best["per_user"] is None:  # per-user values of the winning blend, for the standard error
        best["per_user"] = pool.per_user(best["weights"], best["normalize"])[0]
    diff = best["per_user"] - per_user_single[base]
    best["gain"] = float(best["ndcg"] - singles[base])
    best["gain_se"] = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return best


# ----------------------------------------------------------------------------------- plots
def draw_heatmap(ax, matrix: np.ndarray, labels: Sequence[str], title: str, subtitle: str,
                 vmin: float, vmax: float, fmt: str = "{:.2f}") -> None:
    n = len(labels)
    ax.set_facecolor(SURFACE)
    for i in range(n):  # diagonal: a model against itself carries no information
        ax.add_patch(Rectangle((i, i), 1, 1, facecolor=DIAGONAL, edgecolor=SURFACE, linewidth=2))
    ax.pcolormesh(np.ma.masked_invalid(matrix), cmap=CMAP, vmin=vmin, vmax=vmax,
                  edgecolors=SURFACE, linewidth=2)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=35, ha="right", color=INK_SECONDARY, fontsize=9)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    span = (vmax - vmin) or 1.0
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            r, g, b, _ = CMAP(float(np.clip((value - vmin) / span, 0, 1)))
            dark_cell = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.42
            ax.text(j + 0.5, i + 0.5, fmt.format(value), ha="center", va="center", fontsize=9,
                    color="white" if dark_cell else INK)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=36)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY,
            va="bottom", ha="left")


def save_heatmaps(panels: List[Dict[str, Any]], labels: Sequence[str], out_dir: Path) -> None:
    def style(fig):
        fig.patch.set_facecolor(SURFACE)

    for panel in panels:  # one file per analysis
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        style(fig)
        draw_heatmap(ax, panel["matrix"], labels, panel["title"], panel["subtitle"],
                     panel["vmin"], panel["vmax"], panel.get("fmt", "{:.2f}"))
        fig.tight_layout()
        fig.savefig(out_dir / f"{panel['name']}.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))  # overview: top row = alike?, bottom = useful?
    style(fig)
    for ax, panel in zip(axes.ravel(), panels):
        draw_heatmap(ax, panel["matrix"], labels, panel["title"], panel["subtitle"],
                     panel["vmin"], panel["vmax"], panel.get("fmt", "{:.2f}"))
    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    fig.savefig(out_dir / "overview.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------ workflow
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
    specs, labels, _, _ = load_components(raw_components)

    normalizations = pick(args, file_config, "normalize", ["max", "zscore"])
    normalizations = [normalizations] if isinstance(normalizations, str) else list(normalizations)
    cache_dir = pick(args, file_config, "cache_dir", "cache")
    max_users = int(pick(args, file_config, "max_users", 0))
    weight_steps = int(pick(args, file_config, "weight_steps", 9))
    out_dir = Path(pick(args, file_config, "out_dir", "results/analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cutoff = max(options.cutoffs)
    n_models = len(specs)

    dataset = load_dataset(options.data_dir)
    split = make_split(dataset, options)
    hybrid.FIT_CACHE_SIZE = max(hybrid.FIT_CACHE_SIZE, n_models + 2)
    models = [fit_component(spec, split.train, split.validation, cache_dir=cache_dir) for spec in specs]

    users = split.users
    warm = np.diff(split.train.indptr)[users] > 0
    if not warm.all():
        logger.info("Leaving out %d validation users without training interactions", int((~warm).sum()))
    users = users[warm]
    if max_users and len(users) > max_users:
        users = np.sort(np.random.default_rng(options.split_seed).choice(users, max_users, replace=False))
    logger.info("Analysing %d users and %d components", len(users), n_models)

    pool = CandidatePool(
        models, split.train, split.validation, users, cutoff=cutoff,
        candidates=int(pick(args, file_config, "candidates", 100)),
        normalizations=normalizations,
        batch_size=int(pick(args, file_config, "batch_size", 500)),
    )
    n_users = len(users)

    # --- per-model results: NDCG per user, top-K lists, hits -----------------------------
    per_user = np.stack([pool.per_user(one_hot(n_models, j), normalizations[0])[0] for j in range(n_models)])
    single_ndcg = per_user.mean(axis=1)
    top = top_k_masks(pool, cutoff)
    hits = top & pool.relevant[None]
    hit_counts = hits.sum(axis=(1, 2))

    # --- 1. agreement -----------------------------------------------------------------
    logger.info("Score correlations ...")
    candidate_masks = top_k_masks(pool, pool.scores.shape[-1] // n_models)
    spearman = np.full((n_models, n_models), np.nan)
    for a, b in itertools.combinations(range(n_models), 2):
        spearman[a, b] = spearman[b, a] = pair_rank_correlation(pool, candidate_masks, a, b)
    depth = pool.scores.shape[-1] // n_models
    overlap_deep = pairwise(
        lambda a, b: float((candidate_masks[a] & candidate_masks[b]).sum(axis=1).mean() / depth), n_models)
    del candidate_masks
    overlap = pairwise(lambda a, b: float((top[a] & top[b]).sum(axis=1).mean() / cutoff), n_models)

    # --- 2. unique hits and error correlation ---------------------------------------------
    only_hits = pairwise(lambda a, b: float((hits[a] & ~hits[b]).sum() / max(hit_counts[a], 1)), n_models)
    with np.errstate(divide="ignore", invalid="ignore"):
        error_corr = np.corrcoef(per_user)
    np.fill_diagonal(error_corr, np.nan)

    # --- 4. oracle (cheap, from the per-user NDCG) -----------------------------------------
    best_single = float(single_ndcg.max())
    oracle_all = float(per_user.max(axis=0).mean())
    oracle_gain = pairwise(
        lambda a, b: float(np.maximum(per_user[a], per_user[b]).mean() - max(single_ndcg[a], single_ndcg[b])),
        n_models)

    # --- 3. pairwise blend gain -------------------------------------------------------------
    weights_b = [round(i / (weight_steps + 1), 4) for i in range(1, weight_steps + 1)]
    pairs = list(itertools.combinations(range(n_models), 2))
    blend_gain = np.full((n_models, n_models), np.nan)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for number, (a, b) in enumerate(pairs, start=1):
        result = pair_blend(pool, a, b, normalizations, weights_b, per_user)
        blend_gain[a, b] = blend_gain[b, a] = result["gain"]
        rows.append({
            "model_a": labels[a], "model_b": labels[b],
            "spearman": spearman[a, b],
            f"overlap@{cutoff}": overlap[a, b],
            f"overlap@{depth}": overlap_deep[a, b],
            "ndcg_corr": error_corr[a, b],
            "a_hits_missed_by_b": only_hits[a, b], "b_hits_missed_by_a": only_hits[b, a],
            "union_hit_uplift": float((hits[a] | hits[b]).sum() / max(hit_counts[a], hit_counts[b], 1) - 1),
            f"oracle_gain": oracle_gain[a, b],
            "blend_gain": result["gain"], "blend_gain_se": result["gain_se"],
            "best_weight_b": result["weight_b"], "best_normalize": result["normalize"],
            f"blend_ndcg@{cutoff}": result["ndcg"],
        })
        elapsed = time.perf_counter() - started
        logger.info("Blend search %d/%d pairs done (%.0fs left)", number, len(pairs),
                    elapsed / number * (len(pairs) - number))
    pairs_table = pd.DataFrame(rows).sort_values("blend_gain", ascending=False).reset_index(drop=True)
    pairs_table["gain_vs_noise"] = pairs_table["blend_gain"] / pairs_table["blend_gain_se"]

    equal = {m: pool.evaluate([1.0] * n_models, m)[0] for m in normalizations}

    # --- outputs ----------------------------------------------------------------------------
    singles_table = pd.DataFrame({
        "component": labels, f"ndcg@{cutoff}": single_ndcg,
        f"hits@{cutoff}": hit_counts, f"users_with_a_hit@{cutoff}": hits.any(axis=2).mean(axis=1),
    })
    matrices = {
        "score_correlation": spearman, f"overlap_at_{cutoff}": overlap, "ndcg_correlation": error_corr,
        "unique_hits": only_hits, "oracle_gain": oracle_gain, "blend_gain": blend_gain,
    }
    for name, matrix in matrices.items():
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(out_dir / f"{name}.csv", float_format="%.5f")
    pairs_table.to_csv(out_dir / "pairs.csv", index=False, float_format="%.5f")
    singles_table.to_csv(out_dir / "components.csv", index=False, float_format="%.5f")

    gain_top = max(float(np.nanmax(blend_gain)), float(np.nanmax(oracle_gain)), 1e-4)
    panels = [
        {"name": "score_correlation", "matrix": spearman, "vmin": 0, "vmax": 1,
         "title": "Score correlation", "subtitle": "per-user Spearman on the items either model ranks in its top-100\nhigher = more alike"},
        {"name": f"overlap_at_{cutoff}", "matrix": overlap, "vmin": 0, "vmax": 1,
         "title": f"Top-{cutoff} overlap", "subtitle": f"share of the {cutoff} recommended items in common\nhigher = more alike"},
        {"name": "ndcg_correlation", "matrix": error_corr, "vmin": 0, "vmax": 1,
         "title": "Same good / bad users?", "subtitle": f"correlation of per-user NDCG@{cutoff}\nhigher = fail on the same users"},
        {"name": "unique_hits", "matrix": only_hits, "vmin": 0, "vmax": 1,
         "title": "Unique hits", "subtitle": "share of the row model's hits that the column model misses\nhigher = more complementary"},
        {"name": "oracle_gain", "matrix": oracle_gain, "vmin": 0, "vmax": gain_top, "fmt": "{:.4f}",
         "title": "Oracle gain", "subtitle": f"NDCG@{cutoff} if each user got the better of the two\nupper bound, includes luck"},
        {"name": "blend_gain", "matrix": blend_gain, "vmin": 0, "vmax": gain_top, "fmt": "{:.4f}",
         "title": "Blend gain", "subtitle": f"best linear blend minus the better single, NDCG@{cutoff}\nhigher = better pair"},
    ]
    save_heatmaps(panels, labels, out_dir)

    show = ["model_a", "model_b", "spearman", f"overlap@{cutoff}", "ndcg_corr", "a_hits_missed_by_b",
            "b_hits_missed_by_a", "oracle_gain", "best_weight_b", "best_normalize", "blend_gain",
            "blend_gain_se"]
    lines = [
        f"Users analysed: {n_users}   components: {n_models}   candidates per component: "
        f"{pool.scores.shape[-1] // n_models}   cut-off: {cutoff}",
        "",
        "Single components:",
        singles_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        f"Oracle (best component chosen per user, labels known): ndcg@{cutoff}={oracle_all:.4f}  "
        f"(best single {best_single:.4f}, headroom +{oracle_all - best_single:.4f}, "
        f"{(oracle_all / best_single - 1) * 100:+.1f}%; an upper bound that includes luck, "
        "not reachable by a blend)",
        "Equal-weight blend of all components: "
        + ", ".join(f"{m}: {v:.4f}" for m, v in equal.items()),
        "",
        "Pairs, best blend gain first (blend_gain = best linear blend minus the better of the two; "
        "best_weight_b = weight of model_b, model_a gets 1 - w):",
        pairs_table[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        "Read: low spearman / overlap + high hits missed + low ndcg_corr = the models disagree; "
        "blend_gain says whether that disagreement turned into accuracy "
        f"(gain_vs_noise in pairs.csv: below ~2 it is within noise with {n_users} users).",
        f"Files in {out_dir}/: overview.png, one PNG and CSV per analysis, pairs.csv, components.csv",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "summary.txt").write_text(report + "\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
