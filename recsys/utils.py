"""Small shared helpers."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import scipy.sparse as sp


def gather_profiles(urm: sp.csr_matrix, users: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten the profiles of a batch of users into two aligned arrays.

    Returns:
        ``segment``: position (0..len(users)-1) of the user each entry belongs to.
        ``items``: the item index of every interaction of those users.
    """
    users = np.asarray(users)
    starts = urm.indptr[users]
    lengths = urm.indptr[users + 1] - starts
    total = int(lengths.sum())
    segment = np.repeat(np.arange(len(users)), lengths)
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    items = urm.indices[np.repeat(starts, lengths) + offsets]
    return segment, items


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc_info) -> None:
        self.seconds = time.perf_counter() - self._start
