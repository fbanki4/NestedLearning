"""
Surprise signals: token perplexity and concept perplexity
==========================================================

PLAIN ENGLISH
-------------
"Perplexity" here just means *surprise* — how much a new observation violates
what the model already expects. Nested Learning's memories should write hard
when surprised and barely move when bored (this is the "surprise gating" idea
from the Titans/Nested-Learning line of work).

We measure surprise at two granularities, because the user asked for both:

1. **Token perplexity** — surprise about *one position* in the sequence
   (one EEG patch, one token). It is computed inside each block's memory as the
   normalized error of "predict my value from my key". Big local error = this
   particular token is unexpected here. This module supplies the *standardizer*
   that turns that raw error into a comparable 0..1 number.

2. **Concept perplexity** — surprise about the *whole input's concept*: is this
   sample like things we've seen, or is it a new kind of thing? We keep a small
   running "concept codebook" (a handful of prototype vectors, updated slowly,
   which is itself a slow Nested-Learning memory). If a pooled representation is
   far from every known concept, concept perplexity is high.

A block updates more when either signal is high — so the *frequency* of updates
genuinely depends on token and concept perplexity, exactly as requested, on top
of the depth schedule in ``frequency_schedule.py``.

THE Q2 FIX (carried over from the JAX repo)
-------------------------------------------
Raw error magnitude is fooled by noise: random inputs have large norm and thus
large raw gradients, so an unfixed surprise signal ranks *noise* above genuinely
novel structure. We divide by the input norm before standardizing, which was the
fix the repo found for its "is surprise fooled by noise?" experiment (Q2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningStandardizer(nn.Module):
    """Turn an arbitrary positive 'surprise' scalar into a 0..1 gate.

    Keeps an exponential moving average (EMA) of the mean and variance of the
    values it has seen, converts a new value to a z-score, and squashes it with
    a sigmoid. Output near 1 = "much more surprising than usual", near 0 =
    "less surprising than usual". State lives in buffers (no autograd).
    """

    def __init__(self, momentum: float = 0.02, scale: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.momentum = float(momentum)
        self.scale = float(scale)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        if not bool(self.initialized):
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var + self.eps)
            self.initialized.fill_(True)
        else:
            m = self.momentum
            self.mean.mul_(1 - m).add_(m * batch_mean)
            self.var.mul_(1 - m).add_(m * batch_var)

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.sigmoid(self.scale * z)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        if update:
            self.update(x)
        return self.standardize(x)


def token_surprise(error: torch.Tensor, key: torch.Tensor,
                   standardizer: RunningStandardizer, normalize: bool = True,
                   update: bool = True) -> torch.Tensor:
    """Per-position token perplexity from a memory's prediction error.

    Parameters
    ----------
    error : (B, S, H, Dh) prediction error ``M(k) - v`` from a block memory.
    key   : (B, S, H, Dh) the keys, used only for norm-normalization.
    standardizer : converts raw magnitude -> 0..1.
    normalize : apply the Q2 norm fix (divide by key norm).

    Returns
    -------
    (B, S) surprise in [0, 1].
    """
    mag = error.pow(2).sum(dim=-1)  # (B, S, H)
    if normalize:
        knorm = key.norm(dim=-1).clamp_min(1e-6)  # (B, S, H)
        mag = mag / (knorm ** 2)
    raw = mag.mean(dim=-1)  # (B, S) average over heads
    return standardizer(raw, update=update)


class ConceptCodebook(nn.Module):
    """A small set of prototype vectors = the model's current 'known concepts'.

    Concept perplexity of a sample is how far its pooled representation sits from
    the nearest prototype (standardized to 0..1). Prototypes are nudged toward
    the samples they explain via a slow EMA, so the codebook itself is a slow
    memory in the Nested-Learning sense.
    """

    def __init__(self, dim: int, num_concepts: int = 16, ema: float = 0.01,
                 temperature: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_concepts = num_concepts
        self.ema = float(ema)
        self.temperature = float(temperature)
        # Prototypes are non-learnable state (updated by EMA, not backprop).
        self.register_buffer("prototypes", torch.randn(num_concepts, dim) * 0.02)
        self.register_buffer("counts", torch.zeros(num_concepts))
        self.dist_standardizer = RunningStandardizer(momentum=0.02)

    @torch.no_grad()
    def _update(self, z: torch.Tensor, nearest: torch.Tensor) -> None:
        # Move each selected prototype a little toward the mean of its samples.
        for c in torch.unique(nearest):
            sel = z[nearest == c]
            if sel.numel() == 0:
                continue
            target = sel.mean(dim=0)
            self.prototypes[c].mul_(1 - self.ema).add_(self.ema * target)
            self.counts[c] += sel.shape[0]

    @torch.no_grad()
    def forward(self, z: torch.Tensor, update: bool = True) -> dict:
        """
        z : (B, dim) pooled representation (one 'concept query' per sample).
        Returns dict with ``novelty`` (B,) in [0,1], ``entropy`` (B,),
        and ``assign`` (B,).
        """
        dist = torch.cdist(z, self.prototypes)  # (B, K)
        min_dist, nearest = dist.min(dim=1)  # (B,), (B,)
        novelty = self.dist_standardizer(min_dist, update=update)  # (B,)

        soft = F.softmax(-dist / self.temperature, dim=1)  # (B, K)
        entropy = -(soft * (soft.clamp_min(1e-9)).log()).sum(dim=1)  # (B,)

        if update:
            self._update(z, nearest)
        return {"novelty": novelty, "entropy": entropy, "assign": nearest}


def combine_surprise(tok: torch.Tensor, concept: torch.Tensor,
                     w_token: float = 0.5, w_concept: float = 0.5) -> torch.Tensor:
    """Blend token surprise (B, S) with concept surprise (B,) into a gate (B, S).

    Returns values in [0, 1]; used to scale how hard a block writes to memory.
    """
    gate = w_token * tok + w_concept * concept.unsqueeze(1)
    return gate.clamp(0.0, 1.0)
