"""
An EEG-shaped transformer backbone (CBraMod / LaBraM family)
============================================================

PLAIN ENGLISH
-------------
Real EEG foundation models (CBraMod, LaBraM, EEGPT) all do the same three things:

  1. **Patchify**: cut each channel's time series into short patches and embed
     each patch into a token (plus channel + time position embeddings).
  2. **Transformer blocks** over that token grid.
  3. A head (for pretraining, a head that *reconstructs* masked patches).

This file implements exactly that shape, small enough to pretrain on CPU. It
exposes the same ``.blocks`` / ``.dim`` / ``.embed`` / ``.readout`` interface
that :class:`RetrofitModel` wraps, so a retrofit works identically on this
backbone and on a real EEG FM.

Why we pretrain it (see ``pretrain.py``): the J-space hypothesis is that the
*middle* of a **trained** network is a privileged global workspace. A randomly
initialised model has no such structure, so the middle-vs-ends bet can only be
tested after the backbone has actually learned something. We can't download real
CBraMod/LaBraM weights in this sandbox (the network policy blocks HuggingFace),
so we pretrain a small model here instead. ``load_pretrained`` documents the
real-weights path for a machine that *does* have network + GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .reference_transformer import TransformerBlock


class EEGPatchEmbed(nn.Module):
    """Cut ``(B, C, T)`` EEG into ``(B, C*n_time, dim)`` patch tokens."""

    def __init__(self, patch_len: int, dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, dim)

    def forward(self, x: torch.Tensor):
        b, c, t = x.shape
        if t % self.patch_len != 0:
            raise ValueError(f"n_times ({t}) must be divisible by patch_len ({self.patch_len})")
        n_time = t // self.patch_len
        patches = x.reshape(b, c, n_time, self.patch_len)     # (B,C,n_time,P)
        tokens = self.proj(patches).reshape(b, c * n_time, -1)  # (B, C*n_time, dim)
        return tokens, (c, n_time)


class EEGTransformer(nn.Module):
    """Small EEG foundation-model-shaped transformer with a reconstruction head."""

    def __init__(self, n_channels: int = 8, n_times: int = 128, patch_len: int = 16,
                 dim: int = 64, depth: int = 9, num_heads: int = 4):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.patch_len = patch_len
        self.dim = dim
        self.depth = depth
        self.n_time = n_times // patch_len
        self.n_tokens = n_channels * self.n_time

        self.patch = EEGPatchEmbed(patch_len, dim)
        self.chan_emb = nn.Parameter(torch.randn(1, n_channels, 1, dim) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(1, 1, self.n_time, dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.recon_head = nn.Linear(dim, patch_len)  # reconstruct a patch

    # -- interface used by RetrofitModel --------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        tokens, (c, nt) = self.patch(x)
        pos = (self.chan_emb + self.time_emb).reshape(1, c * nt, self.dim)
        return tokens + pos

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return self.recon_head(self.norm(h))  # (B, n_tokens, patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.readout(h)

    # -- helpers for masked-reconstruction pretraining ------------------
    def patch_targets(self, x: torch.Tensor) -> torch.Tensor:
        """Ground-truth patches ``(B, n_tokens, patch_len)`` for reconstruction loss."""
        b, c, t = x.shape
        return x.reshape(b, c, self.n_time, self.patch_len).reshape(b, self.n_tokens, self.patch_len)

    def apply_mask(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace masked token embeddings with the learned mask token."""
        m = mask.unsqueeze(-1).to(tokens.dtype)  # (B,S,1)
        return tokens * (1 - m) + self.mask_token.view(1, 1, -1) * m


def load_pretrained(name: str = "cbramod", **kwargs) -> EEGTransformer:
    """Documented path to load *real* CBraMod/LaBraM weights (needs network + the
    upstream repo). Not runnable in this sandbox because HuggingFace is blocked.

    On a machine with access:
      1. ``pip install huggingface_hub safetensors`` and clone the upstream repo
         (``wjq-learning/CBraMod`` or ``935963004/LaBraM``) for the exact module
         definitions — their attention (CBraMod is criss-cross) differs from the
         vanilla blocks here, so use *their* model class to load weights.
      2. Download weights, e.g. for CBraMod:
         ``from huggingface_hub import hf_hub_download``
         ``ckpt = hf_hub_download("weighting666/CBraMod", "pretrained_weights.pth")``
      3. Instantiate their model, ``load_state_dict(torch.load(ckpt))``, then hand
         its ``.blocks`` (or equivalent block list) and hidden ``dim`` to
         :class:`~nested_torch.FrozenRetrofit`. The retrofit is architecture-
         agnostic: any ``nn.ModuleList`` of ``block(x)->x`` modules works.
    """
    raise NotImplementedError(
        "Real EEG-FM weights can't be fetched here (HuggingFace is blocked by the "
        "environment network policy). Pretrain locally with nested_torch.pretrain "
        "for a runnable model, or follow this docstring on a networked GPU box.")
