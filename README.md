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
pip install torch                      # only for autoencoder / denoising_ae / mult_vae / lightgcn
pip install xgboost                    # only for stack_ranker.py (the learned ranker)
```

PyTorch is imported lazily: the rest of the project (including `ease`) works without it.
Check the competition rules on allowed frameworks before submitting a neural model.

Python 3.9+ is required. Put the three competition CSV files in `data/`
(already done): `data_train.csv`, `data_target_users_test.csv`,
`alg_sample_submission.csv`.

## Project layout

```
main.py                  CLI: list-models / evaluate / submit
evaluate.py              compare many models or configs on the same split
tune_blend.py            search the weights / subset of components of a hybrid_linear blend
tune_gated_blend.py      do weights that depend on the profile length of the user beat one global blend? (hybrid_gated)
stack_ranker.py          XGBoost ranker that re-orders the candidates of several models (stacking): cv, then submit
analyze_models.py        which components agree / disagree usefully, and which pairs blend well (plots + CSV)
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
  blend_search.py        fast weight search for linear blends (used by tune_blend.py, tune_gated_blend.py)
  stacking.py            candidates, features and XGBoost ranker (used by stack_ranker.py)
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
| `--cache-dir` | off | save the trained model in this folder and load it in later runs (see [Model cache](#model-cache---cache-dir)) |
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
| `--cache-dir` | off | reuse a model trained earlier with identical hyper-parameters (see [Model cache](#model-cache---cache-dir)) |

### Model cache: `--cache-dir`

`main.py evaluate`, `main.py submit` and `evaluate.py` accept `--cache-dir <folder>`
(in the experiment file: `"cache_dir": "<folder>"`; the command line wins). The trained
model is pickled there, and a later run with the same **model, hyper-parameters and
data** (train and validation matrices; for `submit` the full data) loads it instead of
training, which is useful to re-run with another `--cutoffs`, `--target-users-only`,
`--rec-batch-size`, or to re-evaluate a long neural or EASE run. Changing any hyper-parameter,
`--split-seed`, `--validation-fraction`, `--subsample-users` or the data trains a new
model (the old file stays). `fit_s` in the results then shows the loading time.

* Off by default. `evaluate` and `submit` do not share files, because they train on
  different data.
* Hybrids are not pickled as a whole; their own `--cache-dir` flag caches the components
  (see [Hybrid models](#hybrid-models-part-12)), and `evaluate.py --cache-dir` hands the
  folder to every hybrid that has none of its own.
* `--device` and `--num-threads` are not part of the key, so a network
  trained with `--device mps` is found again by a run (e.g. `analyze_models.py`) that
  says `cpu`. Every other hyper-parameter, including `--seed`, `--epochs`, `--patience`,
  must match exactly, so copy them into the experiment / analysis file.
* Delete the folder after changing a model's code: the key does not include the code.
* Dense `ease` is several GB on disk (use `--topk`), and cached models are not purged.

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
| `hybrid_gated` | Part 12 (extension) | linear blend whose weights depend on the profile length (number of training interactions) of the user | `--components`, `--bucket-edges`, `--bucket-weights`, `--normalize` |
| `hybrid_merge` | Part 12 | merges the item-item matrices of item-based models into one model | `--components`, `--weights`, `--normalize`, `--topk` |
| `ease` | Part 15 | EASE^R: closed-form item-item linear autoencoder, `S = I − P·diagMat(1⊘diag P)`, `P = (RᵀR+λI)⁻¹` | `--l2`, `--min-item-interactions`, `--topk`, `--dtype`, `--max-memory-gb` |
| `autoencoder` | Part 15 | user-profile autoencoder (shallow / symmetric / deep, optional tied weights), BCE or MSE | `--hidden-dims`, `--activation`, `--tied-weights`, `--loss`, `--positive-weight`, `--device`, `--num-threads`, `--epochs`, `--batch-size`, `--lr`, `--optimizer`, `--dropout`, ... |
| `denoising_ae` | Part 15 | the same autoencoder trained to rebuild the clean profile from a corrupted one | all `autoencoder` flags + `--noise-dropout`, `--noise-add` |
| `mult_vae` | Part 15 | variational autoencoder with multinomial likelihood and KL annealing | `--hidden-dims`, `--latent-dim`, `--beta`, `--anneal-steps`, `--noise-dropout`, `--input-normalization`, `--device`, ... |
| `gfcf` | Part 16 | GF-CF: normalised co-occurrence filter + ideal low-pass filter from a truncated SVD; no training loop | `--n-factors`, `--alpha`, `--linear-weight`, `--user-exponent`, `--item-exponent`, `--filter-power`, `--svd-solver`, `--topk` |
| `lightgcn` | Part 16 | LightGCN: user / item embeddings smoothed over the user-item graph (no weights, no non-linearity), trained with BPR | `--embedding-dim`, `--n-layers`, `--layer-combination`, `--user-exponent`, `--item-exponent`, `--l2`, `--batch-size`, `--lr`, `--device`, `--num-threads`, `--propagation`, ... |
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

### EASE and neural models (Part 15)

Two-tower models are not implemented (they need side information); the graph
convolution models are in the Part 16 sections below (GF-CF and LightGCN).

* **`ease`** – closed form, no epochs and no early stopping. The dense `n×n` weight
  matrix is built with one Cholesky inversion (LAPACK `potrf`/`potri`, in place), so the
  memory is about `n²·4` bytes in `float32`: **~3 GB for the 27k items with at least 5
  training interactions, ~4.9 GB for all 35k**, plus working space. `--min-item-interactions`
  (default 5) drops rare items (they score 0); the guard `--max-memory-gb` (default 8)
  refuses to start when the matrix would not fit. Fitting takes about 2.5 minutes on 2 CPU
  threads. On the 80/20 split (min items 5, warm users) λ=30 / 100 / 400 gave NDCG@20
  0.1265 / 0.1307 / 0.1274, so `--l2` defaults to 100; it is a strong single model next
  to SLIM (0.132) in the same comparison. The dense model is several GB, so it is
  not cached on disk by `hybrid_*` with `--cache-dir` unless you set `--topk K`: this
  keeps the K largest-magnitude weights per target item, which makes the model small and
  fast to score, picklable, and usable inside `hybrid_merge` (which needs an item-item
  matrix). Without `--topk`, `ease` works in `hybrid_linear` / `hybrid_list` /
  `hybrid_pipeline` but not in `hybrid_merge`.
* **`autoencoder`** – input = the user's whole profile (a 0/1 vector over all items), output
  = a score for every item; the loss is computed over all items, with `--positive-weight`
  to up-weight the observed ones. `--hidden-dims 600` is a shallow autoencoder,
  `--hidden-dims 600,200` a deep symmetric one (the decoder mirrors the encoder),
  `--tied-weights` makes the decoder use the transposed encoder weights (about half the
  parameters), and `--activation identity` with a single layer gives the linear model
  of the slides. `--loss bce` (default) uses sigmoid outputs, `--loss mse` raw outputs.
* **`denoising_ae`** – the same network, trained on a corrupted profile: every observed
  interaction is removed with probability `--noise-dropout` (default 0.5) and every
  missing one is switched on with probability `--noise-add` (default 1e-4, about 3 false
  interactions per user); the target is always the clean profile. Scoring uses the clean profile.
* **`mult_vae`** – encoder → mean/log-variance of a `--latent-dim` Gaussian → decoder with a
  softmax over all items; loss = multinomial log-likelihood + β·KL. β grows linearly from
  0 to `--beta` (default 0.2) over `--anneal-steps` updates (0 = constant β). Defaults
  follow Liang et al. 2018: input dropout 0.5, l2-normalised input, tanh. `--hidden-dims`
  may be empty (`[]`, linear encoder).
* **Hardware flags** – `--device {auto,cpu,cuda,mps}` (default `cpu`; `auto` takes cuda,
  then mps, then cpu; an unavailable device stops with a clear error) and
  `--num-threads N` (0 = leave PyTorch's default). Pass JSON or comma lists for dimensions:
  `--hidden-dims 600,200`, `--hidden-dims '[600]'`.
* **Training flags** (shared by the three neural models) – `--epochs`, `--batch-size`,
  `--lr`, `--optimizer {adam,adamw,sgd,adagrad}`, `--momentum` (sgd only),
  `--weight-decay`, `--grad-clip` (0 = off), `--dropout` (hidden layers),
  `--input-normalization {none,l2}`. Early stopping works like the other iterative
  models: `--eval-every` (default 1), `--patience` (default 5), `--validation-users`;
  the best epoch's weights are restored. `submit` trains for exactly `--epochs`
  (see [Data split and early stopping](#data-split-and-early-stopping)).
* **Scores** – the autoencoders output probabilities (sigmoid), `mult_vae` a softmax
  distribution, `--loss mse` raw values. They are scaled very differently from
  KNN/SLIM scores, which is what the `--normalize` option of `hybrid_linear` is for.
* **Cost and caching** – a `--hidden-dims 600` network has about 2·35k·600 ≈ 42M
  weights; expect tens of seconds to minutes per epoch on a laptop CPU, so start with
  `--subsample-users` and a small `--hidden-dims` for smoke tests. Trained networks are
  stored as CPU weights, so `--cache-dir` works for all three neural models.
* Nothing is tuned on the real data: the defaults come from the slides and the VAE paper.
  Tune `--lr`, `--hidden-dims`, the noise level and (VAE) `--beta` first.

### GF-CF (Part 16)

`gfcf` is the graph-filter model of the graph-convolution slides: no embeddings, no
epochs, no early stopping, and no PyTorch. With the normalised URM `R^ = D_U^-a · R · D_I^-b`
and its truncated SVD `R^ ≈ U_K Σ_K V_Kᵀ`, the scores are `r · S` with

    S = linear_weight · R^ᵀR^  +  alpha · D_I^-b · V_K · V_Kᵀ · D_I^b

(the first term is a co-occurrence filter, the second an ideal low-pass filter on the `K`
strongest singular directions; the exponents of `D_I` have opposite signs).

* **Speed and memory** – the SVD takes about 10 s for `--n-factors 512` (the randomized
  solver; `--svd-solver arpack` is exact but about 5× slower) and scoring all validation
  users about 25 s. The dense `items × items` matrix `S` is never built: scores come from
  two sparse products plus the low-rank part, so the model is only `V_K` (about 150 MB) and
  caches fine with `--cache-dir`.
* **`--topk K`** builds `S` anyway, column block by column block, keeps the `K` largest
  entries per item and scores with that sparse matrix. It takes about 75 s to build; it is
  needed to use `gfcf` in `hybrid_merge` (which wants an item-item matrix). NDCG@20 on a
  3000-user sample: 0.1338 exact, 0.1333 with `--topk 1000`, 0.1277 with `--topk 200`.
* **Defaults differ from the paper** (`alpha 0.3`, exponents 0.5, `K 256`) because they were
  tuned on the 80/20 split (warm users, NDCG@20, SLIM = 0.132 in the same comparison):

  | setting | NDCG@20 |
  |---|---|
  | paper defaults (α 0.3, exponents 0.5/0.5, K 256) | 0.1237 |
  | α 1 | 0.1253 |
  | α 1, exponents 0.4/0.4 | 0.1325 |
  | α 1, exponents 0.4/0.4, K 512 or 768 (default: 512) | 0.1342 |
  | α 1, exponents 0.4/0.4, K 1024 | 0.1333 |

  Exponents of 0.7 are much worse (about 0.10 on the 3000-user sample): they penalise
  popular items too much. Differences below about 0.002 are noise.
* **Tuning tips** – `--item-exponent` and `--user-exponent` matter most (try 0.3–0.5), then
  `--alpha` (0.6–2 is flat) and `--n-factors`. `--filter-power` generalises the ideal filter
  by weighting direction `k` with `(σ_k/σ_1)^p`; values from −1 to 1 made no clear
  difference here. `--linear-weight 0` keeps only the low-pass term (lower: 0.1228 vs
  0.1269 at α 1 on the sample), `--alpha 0` only the co-occurrence term (0.1083).

### LightGCN (Part 16)

`lightgcn` is the graph convolution network of the slides, in its light version. The only
trainable parameters are one embedding per user and per item. A convolution layer replaces
every embedding by the weighted mean of its neighbours' embeddings (weights `1/√(d_u·d_i)`,
no feature transformation, no non-linearity, no self-connection); in matrix form
`E^(k) = G^k · E^(0)`. The final embedding is the mean of the layers `0 … h`
(`--layer-combination mean`, the paper) or the last layer only (`last`, the slides' `E^(h)`),
the score is the dot product of user and item, and the loss is BPR with an L2 penalty on the
embeddings in the batch. After training the model is two dense matrices: scoring needs no
PyTorch, so it caches (`--cache-dir`) and loads on any device.

* **Flags** – model: `--embedding-dim` (64), `--n-layers` (4; 0 = plain matrix
  factorisation), `--layer-combination` (`mean`), `--user-exponent` / `--item-exponent`
  (0.4 / 0.4; the edge weights are `D_U^-a · R · D_I^-b`, 0.5/0.5 is the paper),
  `--init-std`. Loss: `--l2` (1e-4), `--negative-sampling {uniform,popularity}`.
  Optimisation: `--epochs` (100), `--batch-size` (16384 triples), `--lr` (0.01),
  `--optimizer`, `--momentum`, `--weight-decay`, `--grad-clip`. Hardware: `--device
  {auto,cpu,cuda,mps}` (`cpu`), `--num-threads`, `--propagation {auto,sparse,scatter}`.
  Early stopping as in the neural models: `--eval-every` 1, `--patience` 5, the best epoch is
  restored; `submit` trains for exactly `--epochs`, so pass the best epoch of `evaluate`.
* **Cost** – every training step propagates through the whole graph (about 400k edges), so
  an epoch is 7–13 s on 2 CPU threads with `--batch-size 16384` (30 s with 4096, which is
  why the default is large). The default run stops after about 35 epochs, 6–7 minutes.
  `--propagation sparse` (cpu / cuda) uses sparse matrix products; `scatter` (gather +
  scatter-add, chosen automatically on `mps`, which has no sparse tensors) works anywhere but is
  about 5× slower and needs more memory. The `mps` path is tested only against the sparse path
  on cpu, not on Apple hardware: if it fails, use `--device cpu`.
* **Results** (80/20 split, NDCG@20) – full validation users with the defaults: **0.1240**
  (GF-CF 0.1342, EASE 0.131, SLIM 0.132, BPR-MF 0.110 in the same comparison). Tuning on a
  3000-user sample (differences below about 0.002 are noise):

  | setting | NDCG@20 |
  |---|---|
  | 3 layers, exponents 0.5/0.5, lr 0.01, batch 16384 | 0.1200 |
  | + lr 0.03 (peaks at epoch 25, then overfits) | 0.1146 |
  | + exponents 0.4/0.4 | 0.1211 |
  | + exponents 0.3/0.3 (overfits after 3 epochs) | 0.1071 |
  | exponents 0.4/0.4, `--l2 0.001` | 0.1213 |
  | exponents 0.4/0.4, 128 dims | no gain (peak 0.1186 vs 0.1187) |
  | exponents 0.4/0.4, **4 layers (default)** | **0.1251** |
  | exponents 0.4/0.4, 6 layers | worse (peak at epoch 23, 0.1178 on the validation sample vs 0.121 for 4) |

* **Reading the numbers** – LightGCN trails the item-based models here (GF-CF, EASE, SLIM), as
  expected for embedding models on a sparse dataset with a median profile of 13 items, but it
  is the best of the embedding models. It is embedding-based, so its mistakes should differ from
  theirs: check it with `analyze_models.py` and `hybrid_linear` (`--normalize` is needed, the scores
  are raw dot products). Tune `--n-layers`, the two exponents, `--lr` and `--l2` first. If the
  validation curve peaks early and then drops, the model is overfitting: lower `--lr` or raise
  `--l2` / `--weight-decay`.

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

#### Weights that depend on the profile length: `hybrid_gated` and `tune_gated_blend.py`

With 5 known interactions a user gives little to go on, with 80 a lot, and the best model
often differs (in a test with four components the user-based model won for profiles shorter
than 6, GF-CF for longer ones). `hybrid_gated` is a `hybrid_linear` with one list of weights
per profile-length bucket:

```bash
python main.py evaluate hybrid_gated --components @blend_components.json --normalize max \
  --bucket-edges 5,10,20 --bucket-weights '[[0,1,0.5,0],[0.2,0,0,1],[0.1,0,0,1],[0.1,0,0,1]]'
```

`--bucket-edges 5,10,20` makes the buckets `<5`, `5-9`, `10-19` and `>=20` (the edges are the
lower bounds of the next bucket), and `--bucket-weights` has one list of component weights per
bucket. The profile length is the number of interactions in the data the model is trained on:
`submit` trains on all the data (profiles about 1.25× longer than in `evaluate`), so multiply the
edges by `1 / (1 - validation fraction)` there; `tune_gated_blend.py` prints both commands.

`tune_gated_blend.py` tests whether the gating is worth it, without fooling itself. It takes the
same `--config` / `--components` as `tune_blend.py` and the same `--cache-dir`, so no retraining:

```bash
python3 tune_gated_blend.py --config experiments/analyze_bests.json
python3 tune_gated_blend.py --config experiments/analyze_bests.json --bucket-edges 5 10 20 50
python3 tune_gated_blend.py --config experiments/analyze_bests.json --folds 5 --strategy grid
```

1. **Buckets** – validation users with at least one training interaction are cut into
   `--n-buckets` (default 4) buckets of about equal size, or at your `--bucket-edges`; a bucket
   with fewer than `--min-bucket-users` (500) users is merged with its smaller neighbour.
2. **Per-bucket report** – NDCG@20 of every component in every bucket, the best component per
   bucket, and what picking it per bucket would give.
3. **Cross-validation** (`--folds`, default 2; 0 = skip) – the weights are tuned on the other
   folds of users and evaluated on the held-out one, so every number is out-of-sample: best
   single model, best single model per bucket, one global blend, one blend per bucket. The gain
   of the per-bucket blend over the global one is printed with its paired standard error; below
   about two standard errors it is noise. Small buckets (fewer than half of `--min-bucket-users`
   users in a training fold) fall back to the global weights.
4. **Final weights** tuned on all users, checked through the real `hybrid_linear` and
   `hybrid_gated` models, and the `main.py evaluate` / `submit` commands to reproduce them.

It writes `by_profile_length.png` (heat map: models and blends × buckets), `by_profile_length.csv`,
`cv_summary.csv`, `weights.csv` and `gated_config.json` to `--out-dir` (default
`results/gated_blend`). The weights are searched as in `tune_blend.py` (`--strategy coordinate`
by default, which needs far fewer evaluations than `grid`; `--weight-values`, `--candidates`), with
one `--normalize` (default `max`; the first entry if the file lists several).

How to read it: in a test with item-KNN, user-KNN, RP3β and GF-CF (80/20 split, 4 buckets), the
per-bucket blend beat the global blend by +0.0015 NDCG@20 out of sample (about five standard
errors), almost all of it in the shortest bucket, and was only +0.0003 (noise) above simply
choosing the best single model per bucket. Expect a small gain, not a jump: the per-bucket weights
are tuned on fewer users, so with many components and few users per bucket they overfit; use fewer
buckets, fewer components or a coarser `--weight-values`. Run time grows with the number of
components, buckets and folds (about 4 minutes for that test, from a cold cache).

#### Which components should be blended? `analyze_models.py`

Takes the same `--config` / `--components` as `tune_blend.py` and the same `--cache-dir`, so no
retraining. It writes heatmaps and CSV files to `--out-dir` (default `results/analysis`):

```bash
python3 analyze_models.py --config experiments/tune_blend.json
python3 analyze_models.py --components @experiments/blend_components.json --max-users 5000
```

| Output | Question it answers |
| --- | --- |
| `score_correlation`, `overlap_at_20` | Do the models rank alike? Per-user Spearman on the items either model puts in its top-100, and the share of the top-20 in common. |
| `ndcg_correlation` | Do they do well / badly on the same users? |
| `unique_hits` | How many of model A's validation hits does model B miss (and vice versa)? |
| `oracle_gain` | NDCG@20 gain if every user got the better of the two models; an upper bound that includes luck. |
| `blend_gain` | Ground truth: best linear blend of the pair minus the better single model (weights and normalisation searched). |
| `pairs.csv`, `summary.txt` | All of the above per pair, sorted by blend gain, with the standard error of the gain over users. |

Rankings, overlaps and hit counts only depend on the order of the scores within a user, so they
need no normalisation; the blend gain normalises per user like `hybrid_linear` (`--normalize`,
default `max zscore`). Users without training interactions are left out. A gain below about two
standard errors (`gain_vs_noise` in `pairs.csv`) is not distinguishable from noise, and the
weights are picked on the same validation users, so confirm a promising pair with another
`--split-seed` and `tune_blend.py`.

#### A learned ranker on top of the components: `stack_ranker.py`

A blend applies one fixed weight vector to the scores of the components. A *ranker* learns from
examples how to combine them, per user and per item. The components propose candidates (the top
`--candidates` items of each, 50 by default), every (user, candidate) pair is described by features,
and XGBoost learns to order each user's candidates so that the held-out interactions come first.
By default it classifies every pair as relevant or not (`binary:logistic`); `--objective rank:ndcg` is
LambdaMART instead, whose gradients are weighted by how much NDCG changes when two items swap, so
mistakes near the top of the list matter most. In our test the two were within about 0.001 NDCG@20
of each other (the classifier slightly ahead), so compare both on your own components.

Features of a pair: for every component its score divided by the user's best score, z-score and
rank (with `--raw-scores` also the raw score and the user's best raw score, see below); agreement
features (in how many top-20 lists, reciprocal-rank fusion, a plain blend, best / worst / mean
rank, spread); item features (popularity, popularity percentile, how active its users are); user
features (profile length, how popular the profile's items are, item popularity relative to that).

It takes the same `--config` / `--components` as the tuning scripts (so
`experiments/analyze_bests.json` works unchanged) and the same `--cache-dir`: the components that
`analyze_models.py` / `tune_blend.py` trained on the validation split are loaded, not trained again.

```bash
# 1. evaluate (out of sample) and train the final ranker; writes results/ranker/
python3 stack_ranker.py cv --config experiments/analyze_bests.json
# 2. components retrained on all the data (through the cache) + the saved ranker -> submission
python3 stack_ranker.py submit --config experiments/analyze_bests.json --output submissions/ranker.csv
```

`cv` is honest: the components never saw the validation interactions that give the labels, and the
validation users are split in folds so a user is only scored by a ranker that was trained on other
users. The number of trees of a fold is chosen on a slice (`--inner-fraction`, 20%) of that fold's
*training* users, never on the users being scored. It prints the single components, the plain blend, reciprocal-rank fusion
and the ranker on the same candidates, the paired differences with standard errors, a breakdown by
profile length, the most useful features, the candidate recall (the ceiling: the ranker can only
re-order candidates) and an exact check through the real metric. Settings are the ranker flags (see
`--help`) or the `"ranker"` object of the JSON file:

```json
"ranker": {"candidates": 100, "max_depth": 5, "blend_weights": [0.1, 0.3, 0.2, 0.4]}
```

Put the weights found by `tune_blend.py` in `blend_weights`: the ranker then gets the best blend as
a feature and starts from it. Other useful knobs: `--candidates` (more = higher ceiling, more rows),
`--objective rank:ndcg` (alternative to the default `binary:logistic`), `--max-depth`, `--learning-rate`,
`--min-child-weight`, `--raw-scores`, `--n-jobs`.

Things to know:

- The components of `submit` are trained on all the data, so they are a little stronger than the
  ones the ranker was trained with. The count features (profile length, item popularity) are
  scaled by `1 - validation_fraction` to stay on the same scale, and raw scores are left out by
  default (`--raw-scores` adds them): they were worth about 0.0003 NDCG@20 in our test (noise) but
  shift when the components see more data. `submit` prints which feature means moved most (in
  training standard deviations) and warns above 0.3.
- XGBoost always runs in its own Python process (training and prediction). On macOS it can crash with
  a segmentation fault when PyTorch, which the neural components load while scoring, is in the same
  process (the two bring different OpenMP runtimes). The worker takes the data through temporary
  files, so expect a few seconds of disk I/O; if it still crashes you get an error message instead
  of a dead run (try `--n-jobs 1` and update `xgboost`).
- Changing the components or the feature options requires running `cv` again (the saved ranker
  checks that the feature names match).
- Components that use the validation set for early stopping (`bpr_mf`, `slim_bpr`, the neural
  models) pick only their number of epochs on it; the effect on the labels is negligible but it is
  there.
- The gain is relative to the components: expect it to be largest when the components disagree
  usefully (use `analyze_models.py`), and smaller when most candidates come from one dominant model.
- Memory is about 4 bytes x rows x features, with rows = users x candidates per user (about 120
  per user with 4 components and `--candidates 50`).

### What is not included, and why

* Content-based filtering, hybrids and feature-aware models – no item features.
* Best Rated, Global Effects, Pearson/adjusted cosine, SVD++ as a rating
  predictor, MAE/MSE evaluation – they need explicit ratings.
* Sequence-aware models and time-based splits – no timestamps.
* Two-tower models – left out on request.
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
