"""Neural autoencoders for collaborative filtering (Part 15).

The input of every model is the *full profile* of a user (a row of the URM, 0/1 over all items)
and the output is a score for every item, so the models are model-based: a new profile can be
scored as it is.  Three registered models share one PyTorch training loop:

* ``autoencoder``    - fully-connected encoder / decoder with a bottleneck.  With
                       ``--hidden-dims 600 --activation identity --no-tied-weights`` it is the slides'
                       shallow (linear) autoencoder ``R~ = R . W_e . W_d`` (a factorised asymmetric
                       item-item model); ``--tied-weights`` makes it the symmetric ``R . W_e . W_e'``;
                       several sizes (``--hidden-dims 1000,200``) give a deep autoencoder.
* ``denoising_ae``   - the same network trained to rebuild the *clean* profile from a corrupted one
                       (salt & pepper noise: ``--noise-dropout`` removes positives,
                       ``--noise-add`` adds a few spurious ones).
* ``mult_vae``       - Variational autoencoder with a multinomial likelihood (Liang et al., WWW 2018).
                       The encoder outputs the mean and log-variance of a Gaussian, the decoder
                       reconstructs the profile from a sample (reparametrisation trick); the loss is
                       ``-sum_i x_i log softmax(logits)_i + beta . KL``, with ``beta`` annealed from 0.

Scores returned to the recommender are positive and comparable across users: probabilities for
the BCE autoencoders (sigmoid) and for the VAE (softmax); raw outputs for ``--loss mse``.

PyTorch is imported lazily, so the rest of the project works without it
(``pip install torch``).  ``--device`` selects where the network runs (``cpu``, ``cuda``, ``mps`` for
Apple silicon, or ``auto``), ``--num-threads`` the CPU threads PyTorch uses.  Early stopping on the
validation NDCG works like in the other iterative models (``--eval-every``, ``--patience``); the best
epoch is restored at the end.  Trained models are pickled as CPU weights, so the ``--cache-dir`` of the
hybrids works and a model trained on a GPU can be loaded anywhere.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

from recsys.config import param
from recsys.models.base import Recommender, register
from recsys.models.validation import EarlyStoppingConfig, ValidationMonitor

logger = logging.getLogger(__name__)

DEVICES = ("auto", "cpu", "cuda", "mps")
ACTIVATIONS = ("relu", "tanh", "sigmoid", "leaky_relu", "identity")
NN_OPTIMIZERS = ("adam", "adamw", "sgd", "adagrad")
INPUT_NORMALIZATIONS = ("none", "l2")
AE_LOSSES = ("bce", "mse")
_SCORING_CHUNK = 512  # users scored at once at inference time


# ------------------------------------------------------------------------------ helpers
def _import_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The neural models (autoencoder, denoising_ae, mult_vae, lightgcn) need PyTorch: pip install torch"
        ) from error
    return torch


def resolve_device(torch, name: str):
    """``auto`` picks cuda, then mps, then cpu; an explicit device must be available."""
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    elif name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested but CUDA is not available; use cpu, mps or auto")
    elif name == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        raise ValueError("--device mps was requested but MPS is not available; use cpu, cuda or auto")
    return torch.device(name)


def _activation(torch, name: str):
    functional = torch.nn.functional
    return {
        "relu": functional.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "leaky_relu": functional.leaky_relu,
        "identity": lambda x: x,
    }[name]


def _clean_dims(value: Any, name: str, allow_empty: bool) -> List[int]:
    """Layer sizes given as a number, a list of numbers or a comma separated string -> list of ints."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of positive integers, got {value!r}")
    dims = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)) or entry <= 0 or entry != int(entry):
            raise ValueError(f"{name} must contain positive integers, got {value!r}")
        dims.append(int(entry))
    if not dims and not allow_empty:
        raise ValueError(f"{name} must contain at least one layer size")
    return dims


def _make_optimizer(torch, config: "NeuralConfig", parameters):
    common = dict(lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "adam":
        return torch.optim.Adam(parameters, **common)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(parameters, **common)
    if config.optimizer == "adagrad":
        return torch.optim.Adagrad(parameters, **common)
    return torch.optim.SGD(parameters, momentum=config.momentum, **common)


# ------------------------------------------------------------------------------- configs
@dataclass
class NeuralConfig(EarlyStoppingConfig):
    """Parameters shared by the three neural models."""

    # hardware
    device: str = param(
        "cpu", "Where the network runs: cpu, cuda (NVIDIA), mps (Apple silicon) or auto "
               "(cuda, else mps, else cpu).", choices=DEVICES,
    )
    num_threads: int = param(0, "CPU threads used by PyTorch (0 = PyTorch default, usually all cores).")
    # optimisation
    epochs: int = param(100, "Maximum number of passes over the users (early stopping usually ends sooner).")
    batch_size: int = param(256, "Users per mini-batch.")
    lr: float = param(1e-3, "Learning rate.")
    optimizer: str = param("adam", "Optimiser.", choices=NN_OPTIMIZERS)
    momentum: float = param(0.9, "Momentum (only for --optimizer sgd).")
    weight_decay: float = param(0.0, "L2 penalty applied by the optimiser to all weights.")
    grad_clip: float = param(0.0, "Clip the gradient norm to this value (0 = no clipping).")
    dropout: float = param(0.0, "Dropout probability on the hidden layers (0 = off).")
    # corruption / scaling of the input profile
    noise_dropout: float = param(
        0.0, "Denoising: probability of removing each positive interaction from the input profile. "
             "The loss still reconstructs the full, clean profile.",
    )
    noise_add: float = param(
        0.0, "Denoising: probability of adding each missing interaction to the input as a spurious "
             "positive (tiny values such as 1e-4 add a few items per user).",
    )
    input_normalization: str = param(
        "none", "Scaling of the (corrupted) input profile: none, or l2 (unit Euclidean norm, as in Mult-VAE).",
        choices=INPUT_NORMALIZATIONS,
    )
    # early stopping (defaults differ from the SGD models: one epoch is expensive, the curve is noisy)
    eval_every: int = param(
        1, "Evaluate on the validation set every N epochs (0 disables validation and early stopping)."
    )
    patience: int = param(
        5, "Stop after this many evaluations without improvement "
           "(0 = never stop early, but still restore the best epoch).",
    )

    def validate(self) -> None:
        super().validate()
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        for name in ("dropout", "noise_dropout", "noise_add"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {getattr(self, name)}")
        if self.weight_decay < 0 or self.grad_clip < 0 or self.num_threads < 0:
            raise ValueError("weight_decay, grad_clip and num_threads must be >= 0")


@dataclass
class AutoencoderConfig(NeuralConfig):
    hidden_dims: list = param(
        [600],
        "Layer sizes of the encoder from the input to the embedding; the last one is the bottleneck "
        "and the decoder mirrors them. E.g. 600 (one hidden layer / embedding of 600), "
        "600,200 or [1000, 200] (deeper).",
        json=True,
    )
    activation: str = param(
        "relu", "Non-linearity of every layer except the output (identity = linear autoencoder).",
        choices=ACTIVATIONS,
    )
    tied_weights: bool = param(
        False, "The decoder reuses the transposed encoder weights (symmetric autoencoder, "
               "slides: R . We . We'); the decoder biases stay separate.",
    )
    loss: str = param(
        "bce", "Reconstruction loss over all items: bce (binary cross-entropy on logits) or mse.",
        choices=AE_LOSSES,
    )
    positive_weight: float = param(
        1.0, "Weight of the positive entries in the loss (>1 counters the huge number of zeros).",
    )

    def validate(self) -> None:
        super().validate()
        self.hidden_dims = _clean_dims(self.hidden_dims, "hidden_dims", allow_empty=False)
        if self.positive_weight <= 0:
            raise ValueError("positive_weight must be positive")


@dataclass
class DenoisingAEConfig(AutoencoderConfig):
    noise_dropout: float = param(
        0.5, "Denoising: probability of removing each positive interaction from the input profile. "
             "The loss still reconstructs the full, clean profile.",
    )
    noise_add: float = param(
        1e-4, "Denoising: probability of adding each missing interaction to the input as a spurious "
              "positive (tiny values such as 1e-4 add a few items per user).",
    )


@dataclass
class MultVAEConfig(NeuralConfig):
    hidden_dims: list = param(
        [600],
        "Hidden layer sizes between the input and the latent layer (mirrored in the decoder); "
        "may be empty ([]) to connect the input directly to the latent layer.",
        json=True,
    )
    latent_dim: int = param(200, "Size of the latent Gaussian.")
    activation: str = param(
        "tanh", "Non-linearity of the hidden layers (identity = none).", choices=ACTIVATIONS,
    )
    beta: float = param(0.2, "Final weight of the KL term (Mult-VAE^PR uses 0.2; 1 = the plain ELBO).")
    anneal_steps: int = param(
        2000, "Optimiser steps over which the KL weight grows linearly from 0 to --beta (0 = constant beta).",
    )
    noise_dropout: float = param(
        0.5, "Probability of removing each positive interaction from the input profile (input dropout).",
    )
    input_normalization: str = param(
        "l2", "Scaling of the (corrupted) input profile: none, or l2 (unit Euclidean norm, as in Mult-VAE).",
        choices=INPUT_NORMALIZATIONS,
    )

    def validate(self) -> None:
        super().validate()
        self.hidden_dims = _clean_dims(self.hidden_dims, "hidden_dims", allow_empty=True)
        if self.latent_dim <= 0 or self.beta < 0 or self.anneal_steps < 0:
            raise ValueError("latent_dim must be positive; beta and anneal_steps must be >= 0")


# ------------------------------------------------------------------------------ networks
def _init_linear(torch, layer) -> None:
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.normal_(layer.bias, mean=0.0, std=0.001)


def build_autoencoder(torch, n_items: int, config: AutoencoderConfig):
    """Encoder ``n_items -> dims`` and mirrored decoder; activations everywhere but the output."""
    nn, functional = torch.nn, torch.nn.functional
    act = _activation(torch, config.activation)
    dims, tied = list(config.hidden_dims), config.tied_weights
    sizes = [n_items] + dims
    depth = len(dims)

    class Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(depth)])
            if tied:
                self.decoder_bias = nn.ParameterList(
                    [nn.Parameter(torch.zeros(sizes[i])) for i in reversed(range(depth))]
                )
            else:
                self.decoder = nn.ModuleList(
                    [nn.Linear(sizes[i + 1], sizes[i]) for i in reversed(range(depth))]
                )
            self.drop = nn.Dropout(config.dropout)
            for layer in list(self.encoder) + ([] if tied else list(self.decoder)):
                _init_linear(torch, layer)

        def forward(self, x):
            h = x
            for layer in self.encoder:
                h = self.drop(act(layer(h)))
            for k in range(depth):
                if tied:
                    h = functional.linear(h, self.encoder[depth - 1 - k].weight.t(), self.decoder_bias[k])
                else:
                    h = self.decoder[k](h)
                if k < depth - 1:
                    h = self.drop(act(h))
            return h  # logits (or the raw reconstruction for the MSE loss)

    return Autoencoder()


def build_vae(torch, n_items: int, config: MultVAEConfig):
    """Gaussian encoder ``n_items -> hidden -> (mu, logvar)`` and decoder ``latent -> hidden -> n_items``."""
    nn = torch.nn
    act = _activation(torch, config.activation)
    hidden, latent = list(config.hidden_dims), config.latent_dim
    encoder_sizes = [n_items] + hidden
    decoder_sizes = [latent] + hidden[::-1] + [n_items]

    class MultVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.ModuleList(
                [nn.Linear(encoder_sizes[i], encoder_sizes[i + 1]) for i in range(len(hidden))]
            )
            self.to_stats = nn.Linear(encoder_sizes[-1], 2 * latent)
            self.decoder = nn.ModuleList(
                [nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]) for i in range(len(decoder_sizes) - 1)]
            )
            self.drop = nn.Dropout(config.dropout)
            for layer in list(self.encoder) + [self.to_stats] + list(self.decoder):
                _init_linear(torch, layer)

        def forward(self, x):
            h = x
            for layer in self.encoder:
                h = self.drop(act(layer(h)))
            mu, logvar = self.to_stats(h).chunk(2, dim=1)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
            h = z
            for k, layer in enumerate(self.decoder):
                h = layer(h)
                if k < len(self.decoder) - 1:
                    h = self.drop(act(h))
            return h, mu, logvar

    return MultVAE()


# --------------------------------------------------------------------- shared recommender
class _TorchRecommender(Recommender):
    """Training loop, early stopping and scoring shared by the neural models."""

    def __init__(self, config=None, **params) -> None:
        super().__init__(config, **params)
        self._net = None  # live network (on the training device); never pickled
        self._state = None  # CPU copy of the weights, this is what gets pickled

    # ---- to be provided by the models
    def _build_network(self, torch):  # pragma: no cover - interface
        raise NotImplementedError

    def _training_loss(self, torch, clean, step: int) -> Tuple[Any, Dict[str, float]]:  # pragma: no cover
        raise NotImplementedError

    def _predict(self, torch, net, profile):  # pragma: no cover - interface
        """Scores (batch x items) for the (already normalised) input profiles."""
        raise NotImplementedError

    # ---- input handling
    def _normalise(self, torch, profile):
        if self.config.input_normalization == "l2":
            return torch.nn.functional.normalize(profile, p=2.0, dim=1)
        return profile

    def _corrupt(self, torch, clean):
        """Masking / salt & pepper noise on the 0-1 profile, then the input scaling."""
        cfg = self.config
        profile = clean
        if cfg.noise_dropout > 0:
            profile = profile * (torch.rand_like(clean) >= cfg.noise_dropout).to(clean.dtype)
        if cfg.noise_add > 0:
            profile = torch.maximum(profile, (torch.rand_like(clean) < cfg.noise_add).to(clean.dtype))
        return self._normalise(torch, profile)

    # ---- training
    def _fit(self, urm: sp.csr_matrix) -> None:
        torch = _import_torch()
        cfg = self.config
        if cfg.num_threads > 0:
            torch.set_num_threads(cfg.num_threads)
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        device = resolve_device(torch, cfg.device)
        net = self._build_network(torch).to(device)
        self._net = net
        n_parameters = sum(p.numel() for p in net.parameters())
        optimizer = _make_optimizer(torch, cfg, net.parameters())
        logger.info("%s: %d parameters, device %s, %d threads",
                    self.name, n_parameters, device, torch.get_num_threads())

        active = np.flatnonzero(self._profile_sizes > 0)
        monitor = ValidationMonitor(self, cfg)
        best_state = None
        step = 0
        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            net.train()
            totals: Dict[str, float] = {}
            order = rng.permutation(active)
            for start in range(0, len(order), cfg.batch_size):
                batch = order[start:start + cfg.batch_size]
                clean = torch.from_numpy(urm[batch].toarray()).to(device)
                loss, parts = self._training_loss(torch, clean, step)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                optimizer.step()
                step += 1
                totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * len(batch)
                for name, value in parts.items():
                    totals[name] = totals.get(name, 0.0) + float(value) * len(batch)
            logger.info(
                "%s epoch %d/%d: %s (%.1fs)", self.name, epoch, cfg.epochs,
                " ".join(f"{k}={v / len(order):.4g}" for k, v in totals.items()),
                time.perf_counter() - started,
            )
            if monitor.due(epoch, cfg.epochs):
                if monitor.step(epoch):
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                if monitor.should_stop:
                    logger.info("%s: early stopping at epoch %d", self.name, epoch)
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
            logger.info("%s: restored the best epoch (%d)", self.name, monitor.best_epoch)
        net.eval()
        self._state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    # ---- inference
    def _network(self, torch):
        """The live network, rebuilt from the saved CPU weights after unpickling."""
        if self._net is None:
            net = self._build_network(torch)
            net.load_state_dict(self._state)
            self._net = net.to(resolve_device(torch, self.config.device))
        return self._net

    def _score(self, users: np.ndarray) -> np.ndarray:
        torch = _import_torch()
        net = self._network(torch)
        device = next(net.parameters()).device
        scores = np.empty((len(users), self.n_items), dtype=np.float32)
        net.eval()
        with torch.no_grad():
            for start in range(0, len(users), _SCORING_CHUNK):
                chunk = users[start:start + _SCORING_CHUNK]
                profile = torch.from_numpy(self.urm[chunk].toarray()).to(device)
                scores[start:start + len(chunk)] = (
                    self._predict(torch, net, self._normalise(torch, profile)).float().cpu().numpy()
                )
        return scores

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_net"] = None  # device tensors are not portable; the CPU weights are kept in _state
        return state


# ---------------------------------------------------------------------------------- models
@register("autoencoder")
class Autoencoder(_TorchRecommender):
    """Fully-connected autoencoder on the user profile (shallow/linear, symmetric or deep), BCE or MSE."""

    Config = AutoencoderConfig

    def _build_network(self, torch):
        return build_autoencoder(torch, self.n_items, self.config)

    def _training_loss(self, torch, clean, step: int):
        cfg = self.config
        output = self._net(self._corrupt(torch, clean))
        if cfg.loss == "bce":
            pos_weight = torch.tensor(cfg.positive_weight, dtype=clean.dtype, device=clean.device)
            per_entry = torch.nn.functional.binary_cross_entropy_with_logits(
                output, clean, pos_weight=pos_weight, reduction="none"
            )
        else:
            per_entry = (1.0 + (cfg.positive_weight - 1.0) * clean) * (output - clean) ** 2
        return per_entry.sum(dim=1).mean(), {}

    def _predict(self, torch, net, profile):
        output = net(profile)
        return torch.sigmoid(output) if self.config.loss == "bce" else output


@register("denoising_ae")
class DenoisingAutoencoder(Autoencoder):
    """Denoising autoencoder: rebuilds the clean profile from one with removed / added interactions."""

    Config = DenoisingAEConfig


@register("mult_vae")
class MultVAE(_TorchRecommender):
    """Variational autoencoder with multinomial likelihood (Mult-VAE, Liang et al. 2018)."""

    Config = MultVAEConfig

    def _build_network(self, torch):
        return build_vae(torch, self.n_items, self.config)

    def _training_loss(self, torch, clean, step: int):
        cfg = self.config
        logits, mu, logvar = self._net(self._corrupt(torch, clean))
        reconstruction = -(torch.nn.functional.log_softmax(logits, dim=1) * clean).sum(dim=1).mean()
        kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
        beta = cfg.beta * min(1.0, step / cfg.anneal_steps) if cfg.anneal_steps > 0 else cfg.beta
        loss = reconstruction + beta * kl
        return loss, {"recon": reconstruction.detach(), "kl": kl.detach()}

    def _predict(self, torch, net, profile):
        logits, _, _ = net(profile)  # eval mode: the latent code is the mean, no sampling
        return torch.softmax(logits, dim=1)
