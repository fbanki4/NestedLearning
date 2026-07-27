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
