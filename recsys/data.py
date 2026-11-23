"""Dataset loading, train/test splitting and submission writing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

TRAIN_FILE = "data_train.csv"
TARGET_FILE = "data_target_users_test.csv"

SPLIT_MODES = ("global", "per_user")


@dataclass
class Dataset:
    """The implicit-feedback dataset in matrix form.

    Attributes:
        urm: binary User-Rating-Matrix, shape (n_users, n_items), CSR, float32.
        user_ids: raw UserID of every row of the URM.
        item_ids: raw ItemID of every column of the URM.
        target_users: row indices of the users that must appear in the submission,
            in the order of ``data_target_users_test.csv``.
    """

    urm: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    target_users: np.ndarray

    @property
    def n_users(self) -> int:
        return self.urm.shape[0]

    @property
    def n_items(self) -> int:
        return self.urm.shape[1]

    def summary(self) -> str:
        density = self.urm.nnz / (self.n_users * self.n_items)
        return (
            f"{self.n_users} users, {self.n_items} items, {self.urm.nnz} interactions "
            f"(density {density:.4%}), {len(self.target_users)} target users"
        )


def load_dataset(data_dir: str | Path) -> Dataset:
    """Load ``data_train.csv`` (and the target users when present)."""
    data_dir = Path(data_dir)
    train_path = data_dir / TRAIN_FILE
    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find {train_path}; use --data-dir to set the data folder")
    train = pd.read_csv(train_path, dtype={"UserID": np.int64, "ItemID": np.int64})

    target_path = data_dir / TARGET_FILE
    if target_path.exists():
        target_raw = pd.read_csv(target_path, dtype={"UserID": np.int64})["UserID"].to_numpy()
    else:
        logger.warning("%s not found: all users with interactions are treated as targets", target_path)
        target_raw = np.empty(0, dtype=np.int64)

    # Target users that never appear in the training data still get a (empty) row.
    user_ids = np.unique(np.concatenate([train["UserID"].to_numpy(), target_raw]))
    item_ids = np.unique(train["ItemID"].to_numpy())
    rows = np.searchsorted(user_ids, train["UserID"].to_numpy())
    cols = np.searchsorted(item_ids, train["ItemID"].to_numpy())

    urm = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )
    urm.data[:] = 1.0  # duplicated interactions were summed during construction
    urm.sort_indices()

    if target_raw.size:
        target_users = np.searchsorted(user_ids, target_raw)
    else:
        target_users = np.flatnonzero(np.diff(urm.indptr))
    dataset = Dataset(urm, user_ids, item_ids, target_users)
    logger.info("Loaded dataset: %s", dataset.summary())
    return dataset


def restrict_to_users(urm: sp.csr_matrix, keep: np.ndarray) -> sp.csr_matrix:
    """Return a copy of the URM where only the rows in ``keep`` have interactions."""
    mask = np.zeros(urm.shape[0], dtype=np.float32)
    mask[keep] = 1.0
    restricted = sp.diags(mask).dot(urm).tocsr()
    restricted.eliminate_zeros()
    return restricted


def random_holdout(
    urm: sp.csr_matrix,
    test_fraction: float,
    seed: int,
    mode: str = "global",
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Randomly split the interactions into a train and a test URM.

    Args:
        test_fraction: fraction of interactions moved to the test set.
        mode: ``global`` mirrors the competition split (every interaction is
            held out independently at random); ``per_user`` holds out the same
            fraction of every user's interactions (at least one, never all).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if mode not in SPLIT_MODES:
        raise ValueError(f"mode must be one of {SPLIT_MODES}")

    rng = np.random.default_rng(seed)
    coo = urm.tocoo()
    n = coo.nnz
    is_test = np.zeros(n, dtype=bool)

    if mode == "global":
        n_test = int(round(n * test_fraction))
        is_test[rng.permutation(n)[:n_test]] = True
    else:
        order = np.lexsort((rng.random(n), coo.row))  # sort by user, random inside user
        rows_sorted = coo.row[order]
        counts = np.bincount(coo.row, minlength=urm.shape[0])
        starts = np.cumsum(counts) - counts
        rank_in_user = np.arange(n) - starts[rows_sorted]
        n_test_user = np.rint(counts * test_fraction).astype(np.int64)
        n_test_user = np.minimum(np.maximum(n_test_user, 1), counts - 1)
        n_test_user[counts < 2] = 0
        is_test[order] = rank_in_user < n_test_user[rows_sorted]

    def build(mask: np.ndarray) -> sp.csr_matrix:
        matrix = sp.csr_matrix(
            (np.ones(mask.sum(), dtype=np.float32), (coo.row[mask], coo.col[mask])),
            shape=urm.shape,
        )
        matrix.sort_indices()
        return matrix

    return build(~is_test), build(is_test)


def write_submission(
    path: str | Path, dataset: Dataset, users: np.ndarray, recommendations: np.ndarray
) -> None:
    """Write ``UserID,item_list`` using the raw ids of the competition files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_users = dataset.user_ids[users]
    raw_items = dataset.item_ids[recommendations]
    with path.open("w") as handle:
        handle.write("UserID,item_list\n")
        for user, items in zip(raw_users, raw_items):
            handle.write(f"{user}," + " ".join(map(str, items)) + "\n")
    logger.info("Wrote submission for %d users to %s", len(raw_users), path)
