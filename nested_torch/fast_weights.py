"""
Per-block fast-weight associative memory
=========================================

PLAIN ENGLISH
-------------
This is the little "notebook" we bolt onto each frozen transformer block. It is
a direct PyTorch port of the JAX ``AssociativeMemory`` in this repo, with one
deliberate change for continual learning (see below).

How it works, in one breath: we project the block's hidden state into a key,
a query and a value. The memory is a linear map ``M`` (one small matrix per
head). *Reading* is ``y = M(query)`` — a plain matrix multiply, no learning.
*Writing* nudges ``M`` so that ``M(key) ≈ value`` using the delta rule with a
momentum term (which the Nested-Learning paper shows *is* SGD-with-momentum,
i.e. "momentum is memory"). Writes are gated by surprise and can carry a
per-position weight, so boring inputs barely move the memory.

The output we hand back is a **correction** added to the frozen block's output:

    hidden' = FrozenBlock(hidden) + out_proj(M(query))

Because ``M`` starts at zero, the correction starts at zero — so on step 0 the
retrofitted model is byte-for-byte the original frozen model. It only departs
from the frozen model as it adapts.

DELIBERATE CHANGE vs THE JAX VERSION
------------------------------------
The JAX version keeps a separate memory *per batch element* (good for many
independent sequences). Here the memory has **no batch dimension**: it is one
persistent state that accumulates across the whole task stream, updated with the
batch-averaged, gate-weighted delta. That is what we want for continual
learning — the memory is the thing that should remember across batches/tasks.

The state (``W`` and the surprise momentum ``S``) lives in **buffers**, not
Parameters: it is updated by the local delta rule at adaptation time, never by
autograd. The projections are Parameters so an optional outer loop *could* learn
them, but pure online adaptation leaves them fixed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .surprise import RunningStandardizer, token_surprise, combine_surprise


class BlockMemory(nn.Module):
    """Surprise-gated delta-rule associative memory for one transformer block."""

    def __init__(self, dim: int, num_heads: int = 4, lr: float = 0.1,
                 momentum: float = 0.9, forget: float = 0.05,
                 normalize_surprise: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.forget = float(forget)
        self.normalize_surprise = bool(normalize_surprise)

        # The "shape" of the memory (fixed during pure online adaptation).
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # The memory content and its surprise momentum — persistent state.
        self.register_buffer("W", torch.zeros(num_heads, self.head_dim, self.head_dim))
        self.register_buffer("S", torch.zeros(num_heads, self.head_dim, self.head_dim))

        # Standardizes this block's raw token-error into a 0..1 surprise.
        self.tok_standardizer = RunningStandardizer(momentum=0.02)

    # -- helpers ---------------------------------------------------------
    def _heads(self, x: torch.Tensor, lin: nn.Linear) -> torch.Tensor:
        b, s, _ = x.shape
        return lin(x).view(b, s, self.num_heads, self.head_dim)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        """Correction to add to the frozen block output: ``out_proj(M(query))``."""
        q = self._heads(x, self.query_proj)                 # (B,S,H,Dh)
        y = torch.einsum("bshd,hde->bshe", q, self.W)       # (B,S,H,Dh)
        y = y.reshape(x.shape[0], x.shape[1], self.dim)
        return self.out_proj(y)

    def _error_and_key(self, x: torch.Tensor):
        k = self._heads(x, self.key_proj)                   # (B,S,H,Dh)
        v = self._heads(x, self.value_proj)                 # (B,S,H,Dh)
        pred = torch.einsum("bshd,hde->bshe", k, self.W)    # (B,S,H,Dh)
        return pred - v, k                                  # error, key

    @torch.no_grad()
    def _apply_write(self, key: torch.Tensor, error: torch.Tensor, gate: torch.Tensor) -> None:
        # grad of ||M(k)-v||^2 wrt W is outer(k, error); average over B,S with the gate.
        kf = key
        if self.normalize_surprise:
            knorm = key.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # (B,S,H,1)
            kf = key / knorm
        kf = kf * gate.unsqueeze(-1).unsqueeze(-1)          # weight by surprise gate
        n = key.shape[0] * key.shape[1]
        grad = torch.einsum("bshd,bshe->hde", kf, error) / max(n, 1)
        self.S.mul_(self.momentum).add_(grad, alpha=-self.lr)
        self.W.mul_(1.0 - self.forget).add_(self.S)

    # -- reusable pieces (used by the default path AND the RL controller) --
    @torch.no_grad()
    def read_and_assess(self, x: torch.Tensor, concept_surprise: torch.Tensor,
                        w_token: float = 0.5, w_concept: float = 0.5):
        """Compute everything needed to decide/perform an update, WITHOUT writing.

        Returns ``(correction, key, error, tok, gate)`` where ``correction`` is
        added to the frozen output, ``tok`` (B,S) is token perplexity, and
        ``gate`` (B,S) is the surprise-blended write weight. Nothing is mutated
        except the token-surprise running stats.
        """
        correction = self.read(x)                            # uses current memory
        error, key = self._error_and_key(x)
        tok = token_surprise(error, key, self.tok_standardizer,
                             normalize=self.normalize_surprise, update=True)  # (B,S)
        gate = combine_surprise(tok, concept_surprise, w_token, w_concept)    # (B,S)
        return correction, key, error, tok, gate

    @torch.no_grad()
    def write(self, key: torch.Tensor, error: torch.Tensor, gate: torch.Tensor) -> None:
        """Apply a gated delta-rule write (public wrapper around the update)."""
        self._apply_write(key, error, gate)

    # -- the default (non-RL) path the retrofit uses --------------------
    def step(self, x: torch.Tensor, concept_surprise: torch.Tensor, scheduled: bool,
             surprise_trigger: bool = True, trigger_threshold: float = 0.7,
             w_token: float = 0.5, w_concept: float = 0.5):
        """Read a correction and (maybe) write, using the schedule + surprise trigger.

        Parameters
        ----------
        x                : (B, S, dim) the frozen block's output.
        concept_surprise : (B,) global concept perplexity for this batch.
        scheduled        : True if the depth schedule permits an update this step.
        surprise_trigger : if True, an unusually surprising batch can trigger an
                           off-schedule update (so update frequency really does
                           depend on token/concept perplexity).
        trigger_threshold: mean gate above which a surprise-triggered update fires.

        Returns
        -------
        correction : (B, S, dim) to add to the frozen output.
        stats      : dict of scalars for logging.
        """
        correction, key, error, tok, gate = self.read_and_assess(
            x, concept_surprise, w_token, w_concept)
        gate_mean = float(gate.mean())

        do_update = scheduled or (surprise_trigger and gate_mean > trigger_threshold)
        if do_update:
            self.write(key, error, gate)

        stats = {
            "updated": 1.0 if do_update else 0.0,
            "scheduled": 1.0 if scheduled else 0.0,
            "gate_mean": gate_mean,
            "token_surprise_mean": float(tok.mean()),
            "memory_norm": float(self.W.norm()),
        }
        return correction, stats

    @torch.no_grad()
    def reset(self) -> None:
        """Wipe the memory (use at a task boundary or before a clean eval)."""
        self.W.zero_()
        self.S.zero_()
