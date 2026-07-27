"""
Self-supervised pretraining for the EEG backbone (masked patch reconstruction)
==============================================================================

A miniature of how real EEG foundation models are pretrained: randomly mask a
fraction of patch tokens, run the transformer, and reconstruct the *masked*
patches from context (an MAE-style objective). This teaches the backbone real
structure so that — unlike a random init — its middle blocks integrate context,
which is the precondition for testing the J-space "fast middle" hypothesis.

Runs in well under a minute on CPU for the small default model.
"""

from __future__ import annotations

from typing import Callable, List

import torch

from .eeg_backbone import EEGTransformer


def pretrain_masked(backbone: EEGTransformer, data_fn: Callable[[int], torch.Tensor],
                    steps: int = 400, mask_ratio: float = 0.5, lr: float = 1e-3,
                    device: str = "cpu", log_every: int = 100, seed: int = 0) -> List[float]:
    """Pretrain ``backbone`` in place. ``data_fn(step) -> (B, C, T)`` on ``device``."""
    torch.manual_seed(seed)
    backbone.to(device).train()
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    losses: List[float] = []

    for step in range(steps):
        x = data_fn(step)
        tokens = backbone.embed(x)                      # (B,S,dim)
        b, s, _ = tokens.shape
        mask = torch.rand(b, s, device=x.device) < mask_ratio
        mask[:, 0] = True                               # guarantee >=1 masked token

        h = backbone.apply_mask(tokens, mask)
        for block in backbone.blocks:
            h = block(h)
        recon = backbone.readout(h)                     # (B,S,P)
        target = backbone.patch_targets(x)              # (B,S,P)

        m = mask.unsqueeze(-1).to(recon.dtype)          # (B,S,1)
        denom = (m.sum() * recon.shape[-1]).clamp_min(1.0)
        loss = ((recon - target) ** 2 * m).sum() / denom

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  pretrain step {step:4d}  masked-recon MSE = {losses[-1]:.4f}")

    backbone.eval()
    return losses


def save_backbone(backbone: EEGTransformer, path: str) -> None:
    torch.save(backbone.state_dict(), path)


def load_backbone(backbone: EEGTransformer, path: str, device: str = "cpu") -> EEGTransformer:
    backbone.load_state_dict(torch.load(path, map_location=device))
    backbone.to(device).eval()
    return backbone
