"""
FrozenRetrofit — retroactively give a frozen transformer multi-speed memories
=============================================================================

PLAIN ENGLISH
-------------
You hand this class the *already-trained* transformer blocks of any model (an
EEG foundation model like CBraMod/LaBraM, or the toy transformer in
``reference_transformer.py``). It:

1. **Freezes** every block — their weights never change.
2. Bolts a small :class:`BlockMemory` onto each block (see ``fast_weights.py``).
3. Uses a :class:`FrequencySchedule` (see ``frequency_schedule.py``) to decide
   how often each block's memory is allowed to update. With the default
   ``"workspace"`` schedule the *middle* blocks update most often — matching the
   J-space global workspace — while the first and last blocks stay nearly frozen.
4. On top of the schedule, an unusually surprising batch (high token/concept
   perplexity) can trigger an off-schedule update, so the update frequency truly
   depends on perplexity.

The forward pass runs under ``no_grad``: this is *test-time / online* adaptation
via the local delta rule, not backprop. The frozen model is recovered exactly
when memories are empty, and the model departs from it only as it adapts.

WHY THIS SHOULD HELP CONTINUAL LEARNING
---------------------------------------
Catastrophic forgetting happens when learning task B overwrites the weights that
encoded task A. Here the frozen weights can never be overwritten, and only the
*middle* memories are highly plastic. Early blocks (input feature extractors)
and late blocks (read-out) barely move, so the representations tasks depend on
stay put — plasticity is concentrated in the workspace where concepts are
combined. That is the bet this code lets you test.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .fast_weights import BlockMemory
from .frequency_schedule import FrequencySchedule, build_schedule
from .surprise import ConceptCodebook


class FrozenRetrofit(nn.Module):
    """Wrap a frozen ``nn.ModuleList`` of transformer blocks with multi-speed memory."""

    def __init__(self, blocks: nn.ModuleList, dim: int,
                 schedule: Optional[FrequencySchedule] = None,
                 schedule_name: str = "workspace", min_period: int = 1,
                 max_period: int = 32, num_heads: int = 4, lr: float = 0.1,
                 momentum: float = 0.9, forget: float = 0.05,
                 normalize_surprise: bool = True, num_concepts: int = 16,
                 w_token: float = 0.5, w_concept: float = 0.5,
                 surprise_trigger: bool = True, trigger_threshold: float = 0.7,
                 freeze: bool = True):
        super().__init__()
        self.dim = dim
        self.blocks = blocks
        n = len(blocks)

        if freeze:
            for p in self.blocks.parameters():
                p.requires_grad_(False)
            self.blocks.eval()

        if schedule is None:
            schedule = build_schedule(n, name=schedule_name,
                                      min_period=min_period, max_period=max_period)
        if schedule.num_layers != n:
            raise ValueError(
                f"schedule has {schedule.num_layers} layers but backbone has {n} blocks")
        self.schedule = schedule

        self.memories = nn.ModuleList([
            BlockMemory(dim, num_heads=num_heads, lr=lr, momentum=momentum,
                        forget=forget, normalize_surprise=normalize_surprise)
            for _ in range(n)
        ])
        self.concepts = ConceptCodebook(dim, num_concepts=num_concepts)

        self.w_token = w_token
        self.w_concept = w_concept
        self.surprise_trigger = surprise_trigger
        self.trigger_threshold = trigger_threshold

        self.register_buffer("step_count", torch.zeros((), dtype=torch.long))
        self._reset_log()

    # -- logging ---------------------------------------------------------
    def _reset_log(self) -> None:
        n = len(self.blocks)
        self._update_counts = [0 for _ in range(n)]
        self._forward_calls = 0
        self._gate_sums = [0.0 for _ in range(n)]
        self._policy_logps: List[torch.Tensor] = []

    def start_episode(self) -> None:
        """Clear collected policy log-probs (call at the start of an RL episode)."""
        self._policy_logps = []

    def episode_logps(self) -> List[torch.Tensor]:
        """Log-probs of the controller's update decisions since ``start_episode``."""
        return self._policy_logps

    def _block_features(self, i: int, n: int, tok: torch.Tensor,
                        concept: torch.Tensor) -> torch.Tensor:
        """Detached per-block feature vector the RL controller decides from."""
        depth = i / (n - 1) if n > 1 else 0.5
        mnorm = float(self.memories[i].W.norm()) / (self.memories[i].W.numel() ** 0.5)
        vals = [depth, float(tok.mean()), float(concept.mean()), mnorm]
        return torch.tensor(vals, dtype=torch.float32, device=self.memories[i].W.device)

    def adaptation_report(self) -> Dict[str, object]:
        """Per-block summary: how often each block actually updated, etc."""
        n = len(self.blocks)
        per_block = []
        for i in range(n):
            calls = max(self._forward_calls, 1)
            per_block.append({
                "block": i,
                "period": self.schedule.periods[i],
                "plasticity": round(self.schedule.plasticity[i], 3),
                "updates": self._update_counts[i],
                "update_rate": round(self._update_counts[i] / calls, 3),
                "mean_gate": round(self._gate_sums[i] / calls, 3),
                "memory_norm": round(float(self.memories[i].W.norm()), 4),
            })
        return {
            "schedule": self.schedule.name,
            "forward_calls": self._forward_calls,
            "workspace_blocks": self.schedule.workspace_layers(),
            "per_block": per_block,
        }

    # -- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor, adapt: bool = True, controller=None) -> torch.Tensor:
        """Run the frozen blocks with additive memory corrections.

        Parameters
        ----------
        x          : (B, S, dim) embedded input (run the backbone's embedding first).
        adapt      : if True, memories may read *and* write. If False (eval),
                     memories still contribute learned corrections but do not change.
        controller : optional RL policy. When given (and ``adapt``), the policy
                     decides *per block* whether to write this step, and its
                     log-probs are collected via ``episode_logps`` so an outer
                     REINFORCE loop can train it. The frozen blocks and the delta-
                     rule writes stay under ``no_grad``; only the controller's own
                     forward builds a gradient graph. With no controller this is
                     the plain schedule + surprise-trigger path.
        """
        with torch.no_grad():
            z = x.mean(dim=1)                                    # (B, dim)
            concept = self.concepts(z, update=adapt)["novelty"]  # (B,)

        step = int(self.step_count)
        n = len(self.blocks)
        for i, block in enumerate(self.blocks):
            with torch.no_grad():
                h = block(x)                                     # frozen block
                correction, key, error, tok, gate = self.memories[i].read_and_assess(
                    h, concept, self.w_token, self.w_concept)
                gate_mean = float(gate.mean())

            if adapt and controller is not None:
                feats = self._block_features(i, n, tok, concept)
                do_update, logp = controller.act(feats)          # logp keeps grad
                self._policy_logps.append(logp)
            else:
                do_update = adapt and (step % self.schedule.periods[i] == 0)
                if (not do_update and adapt and self.surprise_trigger
                        and gate_mean > self.trigger_threshold):
                    do_update = True

            if do_update:
                self.memories[i].write(key, error, gate)         # no_grad inside

            x = h + correction
            self._update_counts[i] += int(bool(do_update))
            self._gate_sums[i] += gate_mean

        self._forward_calls += 1
        if adapt:
            self.step_count += 1
        return x

    @torch.no_grad()
    def predictive_error(self, x: torch.Tensor) -> float:
        """Mean squared memory-prediction error over blocks, WITHOUT adapting.

        This is the mechanism's own objective: how well each block's memory
        predicts its value from its key. Low = this data is well remembered.
        We use it as the per-task 'loss' in the continual-learning experiment,
        so a rise on task A after training task B *is* catastrophic forgetting.
        """
        errs: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            h = block(x)
            error, _ = self.memories[i]._error_and_key(h)
            errs.append(error.pow(2).sum(dim=-1).mean())
            x = h + self.memories[i].read(h)
        return float(torch.stack(errs).mean())

    # -- memory control --------------------------------------------------
    @torch.no_grad()
    def reset_memory(self, reset_concepts: bool = False, reset_log: bool = True) -> None:
        """Wipe all block memories (e.g. between independent runs)."""
        for m in self.memories:
            m.reset()
        self.step_count.zero_()
        if reset_concepts:
            self.concepts.prototypes.normal_(0, 0.02)
            self.concepts.counts.zero_()
        if reset_log:
            self._reset_log()
