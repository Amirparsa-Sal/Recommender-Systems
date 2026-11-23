# RecSys challenge – location recommendation

Collaborative-filtering recommenders for the Kaggle challenge
`recommender-systems-2026-competition-polimi` (implicit feedback, top-20 list per
user, NDCG@20). Everything uses only `(UserID, ItemID)` interactions – no item
features – and implements the methods of the course slides.

> **Status:** the code has been written and statically checked but **never
> executed** (it was not run on the real data). Start with the quick checks in
> [First steps](#first-steps) and expect to fix small things.

## Setup

```bash
pip install -r requirements.txt        # numpy, scipy, pandas, scikit-learn, joblib
```

Python 3.9+ is required. Put the three competition CSV files in `data/`
(already done): `data_train.csv`, `data_target_users_test.csv`,
`alg_sample_submission.csv`.

## Project layout

```
main.py                  CLI: list-models / evaluate / submit
evaluate.py              compare many models or configs on the same split
tune_blend.py            search the weights / subset of components of a hybrid_linear blend
experiments/example.json example comparison file (every model and variant)
experiments/tune_blend.json example weight search for a blend
data/                    competition CSV files
submissions/             default output folder of `main.py submit`
recsys/
  config.py              hyper-parameter dataclasses -> CLI flags / JSON validation
  data.py                loading, id mapping, train/validation split, submission writer
  metrics.py             NDCG@N and MAP@N
  similarity.py          cosine / jaccard / dice / tversky / dot + shrink + top-K
  sampling.py            negative sampling (uniform / popularity)
  optim.py               sparse SGD / Adagrad updates
  experiment.py          split building, running and comparing models
  blend_search.py        fast weight search for linear blends (used by tune_blend.py)
  cli.py                 argparse CLI (one sub-command per model, generated automatically)
  models/                one file per model family (see below; graph.py = graph-based models, hybrid.py = hybrids)
```

## First steps

Run these in order; each is much cheaper than the next.

```bash
python main.py list-models
python main.py evaluate top_popular                          # seconds
python main.py evaluate item_knn --subsample-users 3000      # smoke test on 3000 users
python main.py evaluate item_knn                             # full data
```

`--subsample-users N` keeps the interactions of N random users only. Use it to
check that a model runs before launching it on the full data. Scores on a
subsample are **not** comparable with full-data scores.

## Commands

### `python main.py list-models`
Prints the available model names.

### `python main.py evaluate <model> [flags]`
Splits `data_train.csv` into **train / validation**, trains on the train part,
recommends for the users that have validation interactions and prints NDCG and MAP
on the validation set (see [Data split](#data-split-and-early-stopping)).

```bash
python main.py evaluate item_knn --similarity cosine --topk 200 --shrink 20
python main.py evaluate slim_bpr --epochs 30 --lr 0.005 --loss warp --output results/slim_bpr.csv
python main.py evaluate bpr_mf --factors 128 --epochs 100 --eval-every 5 --patience 3
```

Evaluation-protocol flags (same for every model):

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `data` | folder with the CSV files |
| `--validation-fraction` | `0.2` | fraction of the interactions held out as validation set (same 80/20 as the competition); shared by all models |
| `--split-mode` | `global` | `global` = random holdout over all interactions, like the competition; `per_user` = same fraction held out from every user (at least 1, never all) |
| `--split-seed` | `42` | seed of the split (keep it fixed to compare models) |
| `--cutoffs` | `20` | list lengths for the metrics (one `ndcg@N` / `map@N` column per cutoff) |
| `--target-users-only` / `--no-target-users-only` | off | evaluate only users of `data_target_users_test.csv` |
| `--subsample-users` | `0` | smoke test on N random users (0 = all) |
| `--rec-batch-size` | `1000` | users scored at once (lower it if memory is tight) |
| `--output` | – | save the results table (with hyper-parameters) as CSV |
| `-v` | off | debug logging |

Model hyper-parameters are flags too: `python main.py evaluate <model> --help`
lists them with their defaults. Booleans come as pairs, e.g. `--positive` /
`--no-positive`. Every model also accepts `--seed`.

### `python main.py submit <model> [flags]`
Trains on **all** the data and writes the submission
(`UserID,item_list`, 20 raw ItemIDs separated by spaces, users in the order of
`data_target_users_test.csv`).

```bash
python main.py submit item_knn --topk 200 --shrink 20 --output submissions/item_knn.csv
```

| Flag | Default | Meaning |
|---|---|---|
| `--output` | `submissions/<model>.csv` | submission file |
| `--k` | `20` | items per user |
| `--data-dir` | `data` | folder with the CSV files |
| `--rec-batch-size` | `1000` | users scored at once |
| `--save-model` | – | also pickle the trained model |

Items already in a user's training history are never recommended; users without
history receive the most popular items.

## Comparing models: `evaluate.py`

Trains every configuration on the **same** train/validation split and prints
one row per model (`label`, `ndcg@20`, `map@20`), sorted by `ndcg@<largest cutoff>`.

```bash
python evaluate.py --models top_popular item_knn user_knn            # default hyper-parameters
python evaluate.py --config experiments/example.json --output results/run1.csv
python evaluate.py --config experiments/example.json --subsample-users 3000   # smoke test
```

Flags: `--config`, `--models`, `--sort-by`, `--fail-fast`, `-v`, plus all the
evaluation-protocol flags above. A model that raises an error is recorded in the
`error` column and the run continues (unless `--fail-fast`). Command line flags
override the values in the JSON file.

### Experiment file format

```json
{
  "data_dir": "data",
  "validation_fraction": 0.2,
  "split_mode": "global",
  "split_seed": 42,
  "cutoffs": [20],
  "sort_by": "ndcg@20",
  "models": [
    {"model": "top_popular"},
    {"model": "item_knn", "label": "knn cosine", "params": {"topk": 100, "shrink": 10}},
    {"model": "item_knn", "label": "knn", "params": {"shrink": 10},
     "grid": {"similarity": ["cosine", "jaccard"], "topk": [50, 200]}}
  ]
}
```

* `params` use the same names as the CLI flags with underscores
  (`l1_ratio`, `n_jobs`, ...). Unknown names or invalid values fail before any
  training starts.
* `grid` expands to one run per combination (`itertools.product`); the label gets
  a `[param=value,...]` suffix.
* `experiments/example.json` contains every model and variant.

### Metrics and output

The printed table only has `label`, `ndcg@N` and `map@N` (plus an `error` column if
a model crashed). `ndcg@N` uses binary relevance and a log2 discount, normalised by
the ideal ranking of each user; `map@N` is average precision at N normalised by
`min(N, #relevant)`. Both are averaged over the users with at least one held-out
interaction in the evaluated set.

With `--output file.csv` the CSV additionally contains the model name, `fit_s` /
`recommend_s` (seconds) and the full hyper-parameters of every run.

## Data split and early stopping

The competition test set has no labels, so the only labelled hold-out available is a
validation set carved out of `data_train.csv`. The interactions are split once, with a
fixed seed, and **every model sees the same partition**:

```
data_train.csv ── 80% ──> train        (what every model is fitted on)
               └─ 20% ──> validation   (--validation-fraction; early stopping + reported metrics)
```

* The reported NDCG@20 / MAP@20 are computed on the validation set for the users that
  have at least one validation interaction.
* Non-iterative models (KNN, SLIM ElasticNet, ALS, ...) ignore the validation set
  during training, but they are still trained on the train part only, so all models
  are compared on equal terms. `main.py submit` trains on 100% of the data, so its
  models see more data than in `evaluate`.
* `bpr_mf`, `slim_bpr`, `funk_svd` and `asymmetric_svd` use the validation set for **early stopping**. Every
  `--eval-every` epochs (default 5) the model is scored with NDCG@`--validation-k`
  (default 20) on up to `--validation-users` validation users (default 5000). The model
  keeps the best epoch, stops after `--patience` evaluations without improvement
  (default 3; `0` = never stop but still restore the best epoch) and restores the best
  epoch at the end. `--epochs` is therefore a maximum: raise it (e.g. 100 for `bpr_mf`).
  `--eval-every 0` turns validation off. For `slim_bpr` the pruned top-K model is the
  one that gets validated, and each validation needs a pass over the dense matrix, so
  keep `--eval-every` at 5 or more. With `funk_svd --incremental` early stopping has two
  levels: inside every factor `--eval-every` / `--patience` pick the best *epoch* of that
  factor (the factor is frozen there, not after `--epochs`), and across factors
  `--factor-patience` (default 3) stops adding factors after that many in a row that do not
  improve the best validation NDCG, keeping only the factors up to the best one
  (`--factor-patience 0` = try all `--factors`). Use `--eval-every 2` or more with many
  factors: every check scores the validation users. `mf_als` has no
  early stopping (it converges in a few iterations).
* **`submit` has no early stopping**: there is no validation set there, so the model
  trains for exactly `--epochs`. Find the best epoch with `evaluate` (the log prints
  `restored the best epoch (N)`) and pass it as `--epochs N` to `submit`.
* Because the early-stopping models choose their epoch on the same validation set that
  scores them, their validation numbers are slightly optimistic compared with the
  other models. Use the Kaggle leaderboard as the final, unbiased check.

## Models and variants

| Model | Slides | What it is | Variants / key flags |
|---|---|---|---|
| `top_popular` | Part 4 | most-interacted items for everyone | – |
| `item_knn` | Part 8 | `R~ = R·S`, item-item similarity pruned to K neighbours | `--similarity {cosine,jaccard,dice,tversky,dot}`, `--topk`, `--shrink`, `--tversky-alpha/--tversky-beta`, `--normalize` |
| `user_knn` | Part 8 | `R~ = S·R`, user-user similarity | same flags as `item_knn` |
| `slim_elasticnet` | Part 9 | SLIM, MSE loss, `diag(S)=0`, one regression per item (scikit-learn coordinate descent) | `--penalty {elasticnet,lasso,ridge}`, `--alpha`, `--l1-ratio`, `--positive`, `--topk`, `--restrict-candidates`, `--n-jobs` |
| `slim_bpr` | Parts 9+11 | SLIM trained with the pairwise BPR loss (SGD) | `--loss {bpr,warp}`, `--lambda-pos/--lambda-neg`, `--negative-sampling`, `--non-negative`, `--topk` |
| `mf_als` | Part 10 | MF with ALS on the full matrix (missing = 0), closed-form updates | `--factors`, `--regularization`, `--iterations` |
| `funk_svd` | Part 10 | MF trained with MSE and SGD | `--missing {random,negative_sampling}`, `--n-negatives`, `--incremental`, `--global-effects`, `--optimizer {sgd,adagrad}` |
| `asymmetric_svd` | Part 10 | `R~ = R·Z·Y`, user factors derived from the profile (no per-user parameters) | `--normalize`, `--global-effects`, `--n-negatives` |
| `bpr_mf` | Part 11 | MF trained with BPR | `--loss {bpr,warp}`, `--lambda-user/--lambda-pos/--lambda-neg`, `--negative-sampling`, `--optimizer` |
| `random_walk` | Part 14 | state probabilities of an h-hop random walk from the user on the user-item graph | `--hops {3,5,7,...}` (odd) |
| `rwr` | Part 14 | random walk with restart (personalised PageRank), power iteration | `--damping`, `--iterations`, `--tolerance` |
| `hybrid_linear` | Part 12 | weighted sum of the (normalised) scores of several models | `--components`, `--weights`, `--normalize` |
| `hybrid_list` | Part 12 | merges the top-N lists of several models (round-robin, Borda, RRF) | `--components`, `--weights`, `--method`, `--list-size` |
| `hybrid_pipeline` | Part 12 | each model's predictions are added to the URM that trains the next model | `--components`, `--top-n`, `--use-scores`, `--max-profile-size` |
| `hybrid_merge` | Part 12 | merges the item-item matrices of item-based models into one model | `--components`, `--weights`, `--normalize`, `--topk` |
| `p3alpha` | Part 14 | 3-hop walk as item-item similarity `S = P_IU·P_UI`, transitions raised to α (plain P3 at α=1) | `--alpha`, `--topk` |
| `rp3beta` | Part 14 | P3-α with the destination item's popularity penalised by β | `--alpha`, `--beta`, `--topk` |

### Notes per model

* **KNN** – similarity formulas (`inter` = co-occurrences, `deg` = interactions):
  cosine `inter / (√deg_i·√deg_j + shrink)`; Tversky
  `inter / (inter + α(deg_i−inter) + β(deg_j−inter) + shrink)`, with Jaccard
  (α=β=1) and Dice (α=β=0.5) as special cases. Pearson / adjusted cosine from the
  slides are for explicit ratings and are not included. `--normalize` divides the
  score by the sum of the neighbours' similarities (the slides' weighted
  average); it is off by default because for top-N it is usually not better.
* **SLIM ElasticNet** – `alpha·l1_ratio` is the L1 weight and
  `alpha·(1−l1_ratio)` the L2 weight; `lasso` forces `l1_ratio=1`, `ridge` forces
  `l1_ratio=0`. The slides present gradient descent for this problem; here
  coordinate descent (scikit-learn) is used as in the original SLIM paper, since it
  handles the L1 term properly. `alpha` uses scikit-learn scaling (divided by the
  number of users), so good values are small – try `1e-6 … 1e-3` on a log scale.
  With `--positive` (non-negative weights) only items that co-occur with the
  target item can get a non-zero weight, so `--restrict-candidates` (default)
  regresses on those items only. It gives the same solution and is far faster.
  Use `--n-jobs -1` to use all cores.
* **SLIM BPR** – trains a dense `n_items × n_items` float32 matrix (about 5 GB for
  35k items), pruned to the top-K per item at the end. `--max-memory-gb` (default 8)
  stops the run if the matrix would be larger. `--loss warp` keeps, for each
  positive, the first sampled negative that the model currently ranks wrongly
  (up to `--warp-trials` draws) and skips the pair if none is found.
  Regularisation is applied lazily, only to the weights touched in a batch.
* **mf_als** – full-matrix MSE, i.e. missing-as-negative, solved exactly per
  alternating step (no SGD needed), so it is fast.
* **funk_svd** – `--missing random` fits only the observed 1s: with binary data this
  is nearly degenerate and is included as the baseline the slides discuss; the
  default `negative_sampling` adds sampled zeros. `--incremental` learns one factor
  at a time (`--epochs` per factor). `--global-effects` adds learned `mu`, `b_u`,
  `b_i` (the slides' SVD++ naming); only `b_i` affects the ranking. For top-N
  ranking `bpr_mf` or `mf_als` are the stronger choices.
* **asymmetric_svd** – the target item is removed from the profile when
  predicting a positive pair, otherwise the model could learn `Z·Y = I`.
* **bpr_mf / SGD models** – early stopping is implemented for `bpr_mf`, `slim_bpr`,
  `funk_svd` and `asymmetric_svd` (not for `mf_als`); `--epochs` is then a maximum.
  `--optimizer adagrad` (default) or plain `sgd`. Updates
  of repeated rows inside a mini-batch are summed first. All SGD learning rates and
  regularisation values are untuned starting points.

### Graph-based models (Part 14)

The graph has users and items as nodes and one undirected edge per interaction. With
`D_U` / `D_I` the user / item degrees, the transitions are `P_UI = diag(1/D_U)·R`
(user→item) and `P_IU = diag(1/D_I)·R^T` (item→user).

* **`random_walk`** – `Π^(h) = Π^(h-1)·P` starting from a one-hot user; the item part
  of `Π^(h)` is the score. `h` must be odd so that the walk ends on an item (the slides:
  3 is the usual optimum, sometimes 5, rarely 7; a large `h` drifts towards popularity).
  It is computed exactly, one dense batch of users at a time, with no pruning. For
  `--hops 3` this equals `p3alpha --alpha 1` without top-K.
* **`rwr`** – `Π = γ·Π·P + (1−γ)·Π⁰`, solved by power iteration instead of an exact
  linear solve (the slides call the exact solve expensive). `--damping` is γ: smaller
  means more restarts and a more personalised ranking. It stops after `--iterations`
  or when the state changes by less than `--tolerance`. Each iteration is two sparse
  products over a dense batch of users, so it is slower than the similarity-based models.
* **`p3alpha` / `rp3beta`** – the 3-hop walk collapsed into
  `S_ij = d_j^(-β) · Σ_u (P_IU[i,u]·P_UI[u,j])^α` (row i = source item, column j =
  destination, each row pruned to `--topk`; `β = 0` for `p3alpha`). Scores are
  `r_u·S`: for binary data the user's row of `P_UI` is `r_u` times a constant, which does
  not change the ranking. `--alpha > 1` strengthens the popularity bias, `--beta > 0`
  fights it. The slides write `S_ij` with `d_i·d_j` in the denominator; this implementation
  follows the matrix definition `S = P_IU·P_UI` (source-item degree `d_i` and user
  degree `d_u`), with `d_j` only in the RP3-β penalty.
* Not implemented: the tripartite graphs with item or user attributes (they need side
  information), and the steady-state of the walk (it is the same for every starting
  user, so it is not personalised).

### Hybrid models (Part 12)

A hybrid combines any registered models, including other hybrids (the slides'
hierarchical hybrids). The models are given with `--components` (or `"components"` in
an experiment file) and the mixing weights with `--weights`:

```bash
# components: names (default hyper-parameters) ...
python main.py evaluate hybrid_linear --components item_knn,rp3beta --weights 1,0.5

# ... or JSON with hyper-parameters, inline or from a file (@file.json)
python main.py evaluate hybrid_linear --weights 0.7,0.3 --components \
  '[{"model": "item_knn", "params": {"topk": 200, "shrink": 20}},
    {"model": "rp3beta",  "params": {"alpha": 0.7, "beta": 0.3}}]'
python main.py submit hybrid_linear --components @experiments/best_components.json --weights 0.7,0.3
```

`--weights` has one non-negative number per component (default: all 1). For two
components `0.7,0.3` is the slides' `α, 1-α`; only the ratio matters.

| Model | Slides | What it does |
| --- | --- | --- |
| `hybrid_linear` | 1. Linear combination | `r = Σ w_j · normalise(r_j)`. `--normalize` rescales every user's scores first, since the slides note that the combination works better when the scores share a scale: `max` (default, best unseen item = 1), `minmax`, `zscore` (also centre; use for models with negative scores such as the MF models), `l1`, `l2`, `none`. Statistics ignore the items the user has already seen. Works with every model except `hybrid_list`. |
| `hybrid_list` | 2. List combination | `--method round_robin` (default) takes the components' items in turns, A1, B1, A2, B2, ...; when an item is already in the list the turn is lost, exactly as in the slides' examples. With unequal weights the turns are shared in proportion (2,1 gives A, A, B, A, A, B, ...). `borda` and `rrf` (reciprocal rank fusion, `--rrf-k`) are rank-based alternatives that sum `weight × points(rank)`. It needs no common score scale, but cannot be a component of `hybrid_linear` or `hybrid_merge`. |
| `hybrid_pipeline` | 3. Pipelining | Stage 1 is trained on the URM, its `--top-n` best unseen items per user are added to the URM (value `--added-value`, or scaled by the prediction confidence with `--use-scores`) and stage 2 is trained on the enlarged URM, and so on; the last stage produces the scores. `--max-profile-size N` only enlarges the profiles of users with at most N interactions, users without history are never enlarged. Items added by a stage are never recommended as "new": the final list excludes the user's real training interactions only. The slides warn about memory and about wrong predictions being added; try a small `--top-n`. |
| `hybrid_merge` | 4. Merging models | `S = Σ w_j · S_j` for item-item models (`item_knn`, `slim_elasticnet`, `slim_bpr`, `p3alpha`, `rp3beta`, other merges); the scores are `R · S`. `--normalize` (`max`, `mean`, `none`) puts the matrices on a common scale first. With `--topk 0` the result equals a linear combination of the scores with `--normalize none`; `--topk N` additionally applies a KNN to the merged matrix (N strongest sources per target item), which is what makes the merge different from a linear combination. |

Notes:

* **Nesting:** a component can itself be a hybrid, e.g.
  `{"model": "hybrid_merge", "params": {"components": [...]}}` inside a `hybrid_linear`.
* **Tuning the weights is cheap:** trained components (and the first stage of a pipeline)
  are cached in memory, so a grid over `weights` in an experiment file trains every
  component once. Disable with `--no-cache` if memory is tight.
* **Keep trained components between runs:** add `--cache-dir cache` (or `"cache_dir": "cache"`
  in the hybrid's `params`). Every trained component is then also saved to that folder and
  loaded from it in later runs, so you can change weights, normalisation or the hybrid type
  and start again without retraining. A file is only reused when the model, its
  hyper-parameters, the training data and the validation split are identical, so `evaluate`
  (validation split) and `submit` (all data) keep separate files. Delete the folder after
  changing a model's code. The cache is ignored by git.
* **Early stopping** (BPR, Funk SVD and Asymmetric SVD models) works inside hybrids: the components receive the same
  validation set. `submit` trains every component on all the data without validation.
* Fit time of a hybrid is the sum of its components; the other costs are one scoring pass
  per component. Round-robin / Borda / RRF run a Python loop per user (a few seconds for 20k users).
* In an experiment file `components` and `weights` are native JSON values:

```json
{"model": "hybrid_linear", "label": "knn+rp3beta",
 "params": {"components": [{"model": "item_knn", "params": {"topk": 200}},
                           {"model": "rp3beta",  "params": {"alpha": 0.7, "beta": 0.3}}],
            "normalize": "max"},
 "grid": {"weights": [[1, 0.25], [1, 0.5], [1, 1], [0.5, 1]]}}
```

#### Searching the weights of a blend: `tune_blend.py`

A grid over `weights` in `evaluate.py` is slow even with cached components, because every
setting scores all the components again. `tune_blend.py` loads each component once (from the
same `--cache-dir` cache), scores it once, and then evaluates every weight setting on a small
pool of candidate items per user (the union of the components' top `--candidates`, 100 by
default), which takes a fraction of a second per setting:

```bash
python3 tune_blend.py --config experiments/tune_blend.json --output results/tune_blend.csv
python3 tune_blend.py --config experiments/tune_blend.json --strategy coordinate
python3 tune_blend.py --config experiments/tune_blend.json --normalize max zscore l2
```

The JSON file has the evaluation options (as in `evaluate.py`), the `components` (model,
`params`, an optional `label`, an optional list of candidate `weights`; or `"@components.json"`,
the file you give to `hybrid_linear`) and the search settings:

| Key | Meaning |
| --- | --- |
| `weight_values` | Candidate weights tried for every component (default `[0, 0.1, 0.25, 0.5, 1]`). A weight of 0 removes the component, so the grid also finds the best subset. |
| `component_weights` / `weights` in a component | Own candidates for one component (by label). Only ratios matter, so fix your strongest component to `[1]` to cut the grid. |
| `strategy` | `grid` (every combination, `max_combinations` guards the size) or `coordinate` (starts from the best single component and improves one weight at a time, `max_rounds` passes; far fewer evaluations when there are many components). |
| `normalize` | One or several score normalisations to compare (`max`, `minmax`, `zscore`, `l1`, `l2`, `none`). |
| `candidates` | Top items per component kept as candidates (default 100). The blend's top 20 almost always lies inside; the final check below confirms it. |
| `cache_dir` | Trained-component cache, shared with `evaluate.py` / `main.py` (default `cache`, `""` = memory only). |
| `verify` | Re-evaluate the best setting with the real `hybrid_linear` (default on). |

The output lists the single components, the best settings (weights normalised to sum 1,
NDCG@20 and MAP@20), the exact score of the best one and the `main.py submit` command with
those weights. Settings that differ by a constant factor are evaluated once.

Notes: the search only covers `hybrid_linear` (list, pipeline and merge hybrids need the
real models). With a validation set of a few thousand users differences below about 0.001 are
within noise, so prefer round weights and a simple subset over the single best row, and use the
same `--split-mode` and `--split-seed` as in your other evaluations. A setting tuned on one
validation split can be slightly optimistic: repeat the best candidates with another
`--split-seed` before submitting.

Not implemented: **co-training / S-SLIM** (slides 5), which learns one similarity from the URM
and the item content matrix; it needs item features.

### What is not included, and why

* Content-based filtering, hybrids and feature-aware models – no item features.
* Best Rated, Global Effects, Pearson/adjusted cosine, SVD++ as a rating
  predictor, MAE/MSE evaluation – they need explicit ratings.
* Sequence-aware models and time-based splits – no timestamps.
* Autoencoders and two-tower models – left out on request (taxonomy boxes only in the
  slides). Graph neural networks are not covered by the graph models above.
* Tripartite user-item-attribute graphs (Part 14) and co-training / S-SLIM (Part 12) – they need
  item or user attributes.

## Practical tips

* Keep `--split-seed` and `--validation-fraction` fixed while comparing models, so all
  runs are evaluated on the same validation set. Check the best candidates on the Kaggle
  leaderboard rather than trusting small validation differences.
* Start with `item_knn` (minutes), then `slim_elasticnet` (use `--n-jobs -1`),
  `mf_als` and `bpr_mf`; `slim_bpr` and `funk_svd` take the longest.
* `evaluate --split-mode global` mirrors the competition's random split. `--target-users-only`
  restricts the evaluation to the users you must submit.
* Memory: dense score blocks are `rec_batch_size × n_items` float32 (140 MB for
  1000 × 35k). Lower `--rec-batch-size` on small machines.
