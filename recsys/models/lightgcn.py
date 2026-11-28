"""LightGCN - light graph convolution network for collaborative filtering (Part 16; He et al., SIGIR 2020).

Users and items are the nodes of the bipartite interaction graph.  Every node has a free
embedding ``e^0`` (the only trainable parameters).  A graph convolution replaces the embedding of
a node by the weighted mean of the embeddings of its neighbours, with the symmetric weights
of the slides ``1 / sqrt(d_u d_i)``; LightGCN removes everything else (no feature transformation,
no non-linearity, no self-connection).  With the normalised interaction matrix
``B = D_U^-a . R . D_I^-b`` (``a = b = 1/2``), one hop is::

    e_u^(k+1) = B  . e_i^(k)          e_i^(k+1) = B' . e_u^(k)

which in the slides' block notation is ``E^(k) = G^k . E^(0)`` with ``G = [[0, B], [B', 0]]``.
The final embedding is the mean of the layers ``0 ... h`` (``--layer-combination mean``, as in the
paper) or the last one only (``last``, the slides' ``E^(h)``).  The score of an item is the dot product
of the final user and item embeddings, and training minimises the BPR loss
``-log sigmoid(x_u . (x_i - x_j))`` with an L2 penalty ``--l2`` on the (initial) embeddings of the
triples in the batch.

Every training step propagates through the whole graph (the sparse products are cheap, about 1M
non-zeros), and the embeddings are propagated once more with no gradient to evaluate the model, so
the trained model is just two dense matrices: scoring needs no PyTorch and the model can be pickled
for ``--cache-dir`` whatever device it was trained on.

PyTorch is imported lazily (``pip install torch``).  Sparse products run on ``cpu`` and ``cuda``;
Apple's ``mps`` has no sparse support, so there the propagation uses gather / scatter-add
(``--propagation scatter``, chosen automatically), which needs more memory.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from recsys.config import param
from recsys.models.autoencoder import DEVICES, NN_OPTIMIZERS, _import_torch, _make_optimizer, resolve_device
from recsys.models.base import Recommender, register
from recsys.models.validation import EarlyStoppingConfig, ValidationMonitor
from recsys.sampling import NEGATIVE_SAMPLING, NegativeSampler

logger = logging.getLogger(__name__)

LAYER_COMBINATIONS = ("mean", "last")
PROPAGATIONS = ("auto", "sparse", "scatter")


@dataclass
class LightGCNConfig(EarlyStoppingConfig):
    # model
    embedding_dim: int = param(64, "Size of the user / item embeddings.")
    n_layers: int = param(
        4, "Number of graph convolution layers (hops) h; 0 = plain matrix factorisation. "
           "Paper: 1 ... 4. On this dataset 4 beat 3 (+0.004 NDCG) and 6.",
    )
    layer_combination: str = param(
        "mean", "Final embedding: mean of the layers 0 ... h (the paper) or only the last layer h "
                "(the slides' E^h).", choices=LAYER_COMBINATIONS,
    )
    user_exponent: float = param(
        0.4, "Exponent a of the user degree in the edge weights D_U^-a . R . D_I^-b (the paper uses "
             "0.5 = the symmetric weights 1/sqrt(d_u d_i)). On this dataset 0.4 beat 0.5 and 0.3.",
    )
    item_exponent: float = param(
        0.4, "Exponent b of the item degree in the edge weights (the paper uses 0.5). Both exponents "
             "set how strongly popular nodes are down-weighted when they are averaged.",
    )
    init_std: float = param(0.1, "Standard deviation of the normal initialisation of the embeddings.")
    # loss and sampling
    l2: float = param(
        1e-4, "L2 regularisation of the initial embeddings of the users and items in each batch "
              "(0.5 * squared norm / batch size, as in the paper).",
    )
    negative_sampling: str = param("uniform", "How negative items are drawn.", choices=NEGATIVE_SAMPLING)
    # optimisation
    epochs: int = param(100, "Maximum number of passes over the training interactions.")
    batch_size: int = param(
        16384, "Training triples (user, positive, negative) per step. Every step propagates "
               "through the whole graph, so larger batches are much faster per epoch (the paper "
               "uses 2048; 4096 took 30 s per epoch here, 16384 about 7-13 s).",
    )
    lr: float = param(1e-2, "Learning rate (0.03 overfits after about 25 epochs, 0.01 is slower but better).")
    optimizer: str = param("adam", "Optimiser.", choices=NN_OPTIMIZERS)
    momentum: float = param(0.9, "Momentum (only for --optimizer sgd).")
    weight_decay: float = param(0.0, "Optimiser weight decay on all embeddings, on top of --l2.")
    grad_clip: float = param(0.0, "Clip the gradient norm to this value (0 = no clipping).")
    # hardware
    device: str = param(
        "cpu", "Where the training runs: cpu, cuda (NVIDIA), mps (Apple silicon) or auto "
               "(cuda, else mps, else cpu).", choices=DEVICES,
    )
    num_threads: int = param(0, "CPU threads used by PyTorch (0 = PyTorch default, usually all cores).")
    propagation: str = param(
        "auto", "How the graph is propagated: sparse (sparse matrix products), scatter (gather + "
                "scatter-add, works on every device but needs more memory) or auto (scatter on mps, "
                "sparse otherwise).", choices=PROPAGATIONS,
    )
    # early stopping (one epoch is expensive and the curve is noisy)
    eval_every: int = param(
        1, "Evaluate on the validation set every N epochs (0 disables validation and early stopping)."
    )
    patience: int = param(
        5, "Stop after this many evaluations without improvement "
           "(0 = never stop early, but still restore the best epoch).",
    )

    def validate(self) -> None:
        super().validate()
        self.embedding_dim = int(self.embedding_dim)
        self.n_layers = int(self.n_layers)
        if self.embedding_dim < 1 or self.n_layers < 0:
            raise ValueError("embedding_dim must be >= 1 and n_layers >= 0")
        if self.epochs <= 0 or self.batch_size <= 0 or self.lr <= 0:
            raise ValueError("epochs, batch_size and lr must be positive")
        if self.user_exponent < 0 or self.item_exponent < 0:
            raise ValueError("user_exponent and item_exponent must be >= 0")
        if self.init_std <= 0:
            raise ValueError("init_std must be positive")
        if self.l2 < 0 or self.weight_decay < 0 or self.grad_clip < 0 or self.num_threads < 0:
            raise ValueError("l2, weight_decay, grad_clip and num_threads must be >= 0")


def normalized_interactions(urm: sp.csr_matrix, user_exponent: float, item_exponent: float) -> sp.csr_matrix:
    """``B = D_U^-a . R . D_I^-b`` of the binary URM (0 for users / items without interactions)."""
    binary = sp.csr_matrix(urm, dtype=np.float32)
    binary.data[:] = 1.0

    def inverse_power(degree: np.ndarray, exponent: float) -> np.ndarray:
        result = np.zeros(len(degree), dtype=np.float32)
        positive = degree > 0
        result[positive] = np.power(degree[positive].astype(np.float64), -exponent)
        return result

    users = inverse_power(np.asarray(binary.sum(axis=1)).ravel(), user_exponent)
    items = inverse_power(np.asarray(binary.sum(axis=0)).ravel(), item_exponent)
    return (sp.diags(users) @ binary @ sp.diags(items)).tocsr().astype(np.float32)


class _Graph:
    """The propagation ``(e_u, e_i) -> (B e_i, B' e_u)`` as a differentiable torch operation."""

    def __init__(self, torch, matrix: sp.csr_matrix, device, mode: str) -> None:
        self.torch = torch
        self.mode = mode
        self.use_csr = True
        self._matrix = matrix
        coo = matrix.tocoo()
        rows = torch.from_numpy(coo.row.astype(np.int64)).to(device)
        cols = torch.from_numpy(coo.col.astype(np.int64)).to(device)
        values = torch.from_numpy(coo.data.astype(np.float32)).to(device)
        self.shape = matrix.shape
        if mode == "scatter":
            self.rows, self.cols, self.values = rows, cols, values
        else:
            with warnings.catch_warnings():  # "sparse CSR support is in beta"
                warnings.simplefilter("ignore")
                self.forward = self._sparse(matrix.tocsr(), device)  # users x items
                self.backward = self._sparse(matrix.T.tocsr(), device)  # items x users
                self._check_backward(device)

    def _sparse(self, m: sp.csr_matrix, device):
        torch = self.torch
        if self.use_csr:
            return torch.sparse_csr_tensor(
                torch.from_numpy(m.indptr.astype(np.int64)), torch.from_numpy(m.indices.astype(np.int64)),
                torch.from_numpy(m.data.astype(np.float32)), size=m.shape,
            ).to(device)
        coo = m.tocoo()
        return torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([coo.row, coo.col]).astype(np.int64)),
            torch.from_numpy(coo.data.astype(np.float32)), size=m.shape,
        ).coalesce().to(device)

    def _check_backward(self, device) -> None:
        """CSR products are in beta: make sure they can be differentiated, else use COO."""
        torch = self.torch
        try:
            probe = torch.ones(self.shape[1], 2, device=device, requires_grad=True)
            (self.forward @ probe).sum().backward()
        except Exception:  # pragma: no cover - depends on the torch build
            self.use_csr = False
            self.forward = self._sparse(self._matrix.tocsr(), device)
            self.backward = self._sparse(self._matrix.T.tocsr(), device)

    def users_from_items(self, item_embeddings):
        torch = self.torch
        if self.mode == "scatter":
            out = torch.zeros(self.shape[0], item_embeddings.shape[1], dtype=item_embeddings.dtype,
                              device=item_embeddings.device)
            return out.index_add(0, self.rows, item_embeddings[self.cols] * self.values[:, None])
        return self.forward @ item_embeddings

    def items_from_users(self, user_embeddings):
        torch = self.torch
        if self.mode == "scatter":
            out = torch.zeros(self.shape[1], user_embeddings.shape[1], dtype=user_embeddings.dtype,
                              device=user_embeddings.device)
            return out.index_add(0, self.cols, user_embeddings[self.rows] * self.values[:, None])
        return self.backward @ user_embeddings

    def final_embeddings(self, user0, item0, n_layers: int, combination: str):
        users, items = user0, item0
        user_sum, item_sum = user0, item0
        for _ in range(n_layers):
            users, items = self.users_from_items(items), self.items_from_users(users)  # old values on the right
            user_sum, item_sum = user_sum + users, item_sum + items
        if combination == "last":
            return users, items
        scale = 1.0 / (n_layers + 1)
        return user_sum * scale, item_sum * scale


@register("lightgcn")
class LightGCN(Recommender):
    """LightGCN: embeddings smoothed over the user-item graph, trained with the BPR loss."""

    Config = LightGCNConfig

    def _fit(self, urm: sp.csr_matrix) -> None:
        torch = _import_torch()
        cfg = self.config
        if cfg.num_threads > 0:
            torch.set_num_threads(cfg.num_threads)
        torch.manual_seed(cfg.seed)
        device = resolve_device(torch, cfg.device)
        mode = cfg.propagation if cfg.propagation != "auto" else ("scatter" if device.type == "mps" else "sparse")
        graph = _Graph(torch, normalized_interactions(urm, cfg.user_exponent, cfg.item_exponent), device, mode)

        user0 = torch.nn.Parameter(torch.randn(self.n_users, cfg.embedding_dim, device=device) * cfg.init_std)
        item0 = torch.nn.Parameter(torch.randn(self.n_items, cfg.embedding_dim, device=device) * cfg.init_std)
        optimizer = _make_optimizer(torch, cfg, [user0, item0])
        logger.info(
            "LightGCN: %d users + %d items, dim %d, %d layers (%s), %d parameters, device %s, "
            "propagation %s, %d threads",
            self.n_users, self.n_items, cfg.embedding_dim, cfg.n_layers, cfg.layer_combination,
            user0.numel() + item0.numel(), device, mode, torch.get_num_threads(),
        )

        def refresh() -> None:
            """Propagate without gradient and store the final embeddings used for scoring."""
            with torch.no_grad():
                users, items = graph.final_embeddings(user0, item0, cfg.n_layers, cfg.layer_combination)
            self._user_embeddings = users.cpu().numpy().astype(np.float32)
            self._item_embeddings = items.cpu().numpy().astype(np.float32)

        coo = urm.tocoo()
        positive_users, positive_items = coo.row.astype(np.int64), coo.col.astype(np.int64)
        sampler = NegativeSampler(urm, cfg.negative_sampling, cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        monitor = ValidationMonitor(self, cfg)
        best = None

        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            order = rng.permutation(len(positive_users))
            negatives_all = sampler.sample(positive_users[order])
            total_loss, total_bpr = 0.0, 0.0
            for start in range(0, len(order), cfg.batch_size):
                chosen = order[start:start + cfg.batch_size]
                users = torch.from_numpy(positive_users[chosen]).to(device)
                positives = torch.from_numpy(positive_items[chosen]).to(device)
                negatives = torch.from_numpy(negatives_all[start:start + len(chosen)]).to(device)

                final_users, final_items = graph.final_embeddings(user0, item0, cfg.n_layers, cfg.layer_combination)
                x_u, x_i, x_j = final_users[users], final_items[positives], final_items[negatives]
                bpr = torch.nn.functional.softplus((x_u * x_j).sum(1) - (x_u * x_i).sum(1)).mean()
                regularisation = 0.5 * (
                    user0[users].pow(2).sum() + item0[positives].pow(2).sum() + item0[negatives].pow(2).sum()
                ) / len(chosen)
                loss = bpr + cfg.l2 * regularisation

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([user0, item0], cfg.grad_clip)
                optimizer.step()
                total_loss += float(loss.detach()) * len(chosen)
                total_bpr += float(bpr.detach()) * len(chosen)
            logger.info(
                "LightGCN epoch %d/%d: loss=%.4f bpr=%.4f (%.1fs)", epoch, cfg.epochs,
                total_loss / len(order), total_bpr / len(order), time.perf_counter() - started,
            )
            if monitor.due(epoch, cfg.epochs) or epoch == cfg.epochs:
                refresh()
                if monitor.due(epoch, cfg.epochs):
                    if monitor.step(epoch):
                        best = (self._user_embeddings.copy(), self._item_embeddings.copy())
                    if monitor.should_stop:
                        logger.info("LightGCN: early stopping at epoch %d", epoch)
                        break

        if best is not None:
            self._user_embeddings, self._item_embeddings = best
            logger.info("LightGCN: restored the best epoch (%d)", monitor.best_epoch)

    def _score(self, users: np.ndarray) -> np.ndarray:
        return self._user_embeddings[users] @ self._item_embeddings.T
