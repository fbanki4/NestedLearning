"""
Toy structured-sequence tasks for validating the retrofit on CPU/GPU
====================================================================

These are deliberately tiny and synthetic so the demos run in seconds without a
GPU or any real dataset. Each "task" draws sequences whose tokens live in a
particular random low-dimensional subspace. Two tasks with *different* subspaces
are two different data distributions — switching from one to the other is what
lets us measure catastrophic forgetting in the block memories.

When you move to real EEG, you swap these generators for a dataloader over the
foundation-model benchmark datasets (SEED, PhysioNet-MI, CHB-MIT, ...) and keep
everything else.
"""

from __future__ import annotations

import torch


def make_subspace_basis(in_dim: int, rank: int, seed: int) -> torch.Tensor:
    """An orthonormal ``(in_dim, rank)`` basis = the 'concept subspace' of a task."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(in_dim, rank, generator=g)
    q, _ = torch.linalg.qr(a)
    return q[:, :rank]


def sample_task(basis: torch.Tensor, batch: int, seq: int, noise: float = 0.05,
                seed: int | None = None, device=None) -> torch.Tensor:
    """Sample ``(batch, seq, in_dim)`` sequences living in ``basis``'s subspace."""
    in_dim, rank = basis.shape
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    c = torch.randn(batch, seq, rank, generator=g)
    x = c @ basis.t()
    if noise > 0:
        x = x + noise * torch.randn(batch, seq, in_dim, generator=g)
    if device is not None:
        x = x.to(device)
    return x


def make_eeg_batch(batch: int, n_channels: int, n_times: int, freqs, fs: float = 128.0,
                   noise: float = 0.1, seed: int | None = None, device=None) -> torch.Tensor:
    """Synthetic EEG-like batch: ``(batch, n_channels, n_times)``.

    Each channel is a sum of oscillations at the given ``freqs`` (Hz) with random
    per-sample, per-channel amplitude and phase — a crude but EEG-plausible signal
    (EEG is dominated by band-limited rhythms). Two 'tasks' with disjoint ``freqs``
    (e.g. alpha vs beta bands) are genuinely different distributions, which is what
    lets us pretrain on a rich band and then measure forgetting across bands.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    f = torch.as_tensor(list(freqs), dtype=torch.float32)          # (F,)
    t = torch.arange(n_times, dtype=torch.float32) / fs            # (T,)
    amps = 0.5 + torch.rand(batch, n_channels, len(f), generator=g)   # (B,C,F)
    phases = 2 * torch.pi * torch.rand(batch, n_channels, len(f), generator=g)
    angles = (2 * torch.pi * f[None, None, :, None] * t[None, None, None, :]
              + phases[..., None])                                  # (B,C,F,T)
    sig = (amps[..., None] * torch.sin(angles)).sum(dim=2)          # (B,C,T)
    if noise > 0:
        sig = sig + noise * torch.randn(batch, n_channels, n_times, generator=g)
    if device is not None:
        sig = sig.to(device)
    return sig
