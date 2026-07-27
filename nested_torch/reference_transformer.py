"""
A tiny reference transformer to stand in for a frozen EEG foundation model
==========================================================================

PLAIN ENGLISH
-------------
To test the retrofit we need *some* frozen transformer. A real EEG foundation
model (CBraMod, LaBraM, EEGPT) is a stack of standard transformer blocks over
embedded EEG patches — exactly the shape below, just bigger and pretrained.

So this file gives you:

- :class:`TransformerBlock` — a bog-standard pre-norm block, ``block(x) -> x``.
- :class:`ReferenceTransformer` — patch/vector embedding + a stack of blocks +
  a read-out head. It exposes ``.blocks`` (an ``nn.ModuleList``), which is the
  exact interface :class:`FrozenRetrofit` wraps.
- :class:`RetrofitModel` — glues a **frozen** backbone's embedding and read-out
  onto a :class:`FrozenRetrofit` over its blocks, giving a full model you can
  call end-to-end.

To use a real EEG model instead, hand :class:`FrozenRetrofit` that model's list
of transformer blocks and its ``dim``; nothing else changes. Everything is
device-agnostic: ``model.to("cuda")`` runs it on GPU, and it falls back to CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .frozen_retrofit import FrozenRetrofit


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block: ``x -> x`` (same shape in and out)."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ReferenceTransformer(nn.Module):
    """A small transformer over continuous inputs (e.g. EEG patch vectors)."""

    def __init__(self, in_dim: int, dim: int = 64, depth: int = 8, num_heads: int = 4,
                 max_len: int = 64, out_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_len = max_len
        self.input_proj = nn.Linear(in_dim, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim if out_dim is not None else in_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[1]
        if s > self.max_len:
            raise ValueError(f"sequence length {s} exceeds max_len {self.max_len}")
        return self.input_proj(x) + self.pos_emb[:, :s, :]

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.readout(h)


class RetrofitModel(nn.Module):
    """A frozen backbone + a :class:`FrozenRetrofit` over its blocks = full model.

    The backbone's embedding and read-out are frozen and reused; only the
    attached per-block memories adapt (online, via the delta rule).
    """

    def __init__(self, backbone: ReferenceTransformer, retrofit: FrozenRetrofit,
                 freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.retrofit = retrofit
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    @classmethod
    def build(cls, backbone: ReferenceTransformer, **retrofit_cfg) -> "RetrofitModel":
        """Convenience: wrap ``backbone.blocks`` in a retrofit with the given config."""
        retrofit = FrozenRetrofit(backbone.blocks, dim=backbone.dim, **retrofit_cfg)
        return cls(backbone, retrofit)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, adapt: bool = True) -> torch.Tensor:
        h = self.backbone.embed(x)
        h = self.retrofit(h, adapt=adapt)
        return self.backbone.readout(h)

    @torch.no_grad()
    def predictive_error(self, x: torch.Tensor) -> float:
        h = self.backbone.embed(x)
        return self.retrofit.predictive_error(h)

    def reset_memory(self, **kw) -> None:
        self.retrofit.reset_memory(**kw)

    def adaptation_report(self) -> dict:
        return self.retrofit.adaptation_report()
