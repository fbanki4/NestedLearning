"""
RL controller: learn *where and when* a frozen model should be plastic
======================================================================

PLAIN ENGLISH
-------------
The frequency schedule (``frequency_schedule.py``) is a hand-designed guess at
which blocks should adapt — the "workspace" schedule guesses "the middle". This
module replaces the guess with a **learned policy**.

At every adaptation step, for every block, a tiny policy network looks at a few
features (how deep the block is, how surprising the current input is at token and
concept level, how full the block's memory already is) and outputs the
probability of writing to that block's memory *now*. We sample those decisions,
let the memories adapt across a two-task stream, then score the whole episode by
a **reward that punishes forgetting**:

    reward = -(errA_afterA + errB_afterB)          # learn BOTH tasks well
             - forget_penalty * max(0, forgetting)  # but heavily punish forgetting A
             - cost * update_fraction               # and prefer sparing updates

The ``forget_penalty`` term is the crucial one. Without it, "update everything"
wins: it learns the new task so well that it offsets the forgetting, and the
policy just collapses to the ``uniform`` schedule. Penalizing the *rise* in
old-task error (``forgetting = errA_afterB - errA_afterA``) is what forces the
policy to find a selective, forgetting-aware allocation.

We train the policy with REINFORCE (policy gradient) plus a moving-average
baseline for variance reduction. Crucially, the memory writes stay non-
differentiable (delta rule, ``no_grad``); the gradient flows **only** through the
policy's log-probabilities. So this is genuine RL *on top of* a frozen foundation
model — the thing the project set out to do.

The interesting question this lets you ask: **does the learned policy rediscover
a "fast middle" (workspace) allocation, or something better?**
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch
import torch.nn as nn


class PolicyController(nn.Module):
    """Maps per-block features -> probability of writing to that block's memory."""

    def __init__(self, feat_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.deterministic = False  # trainer flips this for eval rollouts

    def update_prob(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feats)).squeeze(-1)

    def act(self, feats: torch.Tensor):
        """Return ``(do_update: bool, logp: Tensor)``. ``logp`` keeps grad."""
        p = self.update_prob(feats).clamp(1e-4, 1 - 1e-4)
        dist = torch.distributions.Bernoulli(probs=p)
        if self.deterministic:
            action = (p > 0.5).float()
        else:
            action = dist.sample()
        logp = dist.log_prob(action)
        return bool(action.item()), logp


class ReinforceTrainer:
    """Train a :class:`PolicyController` with REINFORCE + moving-average baseline."""

    def __init__(self, controller: PolicyController, lr: float = 1e-2,
                 update_cost: float = 0.05, forget_penalty: float = 2.0,
                 baseline_decay: float = 0.9):
        self.controller = controller
        self.opt = torch.optim.Adam(controller.parameters(), lr=lr)
        self.update_cost = float(update_cost)
        self.forget_penalty = float(forget_penalty)
        self.baseline_decay = float(baseline_decay)
        self.baseline: float | None = None

    def run_episode(self, model, taskA_fn: Callable[[int], torch.Tensor],
                    taskB_fn: Callable[[int], torch.Tensor], eval_a: torch.Tensor,
                    eval_b: torch.Tensor, steps: int, deterministic: bool = False) -> Dict:
        """One A-then-B stream with the policy in control. Returns metrics + logps."""
        self.controller.deterministic = deterministic
        n_blocks = len(model.retrofit.blocks)
        model.reset_memory(reset_concepts=True)
        model.retrofit.start_episode()

        for t in range(steps):
            model(taskA_fn(t), adapt=True, controller=self.controller)
        err_a_after_a = model.predictive_error(eval_a)

        for t in range(steps):
            model(taskB_fn(t), adapt=True, controller=self.controller)
        err_a_after_b = model.predictive_error(eval_a)
        err_b_after_b = model.predictive_error(eval_b)

        logps = model.retrofit.episode_logps()
        decisions = n_blocks * 2 * steps
        updates = sum(model.retrofit._update_counts)
        update_frac = updates / max(decisions, 1)
        forgetting = err_a_after_b - err_a_after_a
        reward = (-(err_a_after_a + err_b_after_b)
                  - self.forget_penalty * max(0.0, forgetting)
                  - self.update_cost * update_frac)
        return {
            "reward": reward,
            "logps": logps,
            "errA_afterA": err_a_after_a,
            "errA_afterB": err_a_after_b,
            "errB_afterB": err_b_after_b,
            "forgetting": forgetting,
            "update_frac": update_frac,
        }

    def train_step(self, ep: Dict) -> float:
        """One REINFORCE update from a (training) episode's result."""
        reward, logps = ep["reward"], ep["logps"]
        if not logps:
            return 0.0
        if self.baseline is None:
            self.baseline = reward
        advantage = reward - self.baseline
        loss = -advantage * torch.stack(logps).sum()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
        return float(loss.detach())

    def train(self, model, taskA_fn, taskB_fn, eval_a, eval_b, steps: int = 40,
              episodes: int = 60, log_every: int = 10) -> List[Dict]:
        history = []
        for ep_i in range(episodes):
            ep = self.run_episode(model, taskA_fn, taskB_fn, eval_a, eval_b, steps,
                                  deterministic=False)
            self.train_step(ep)
            history.append(ep)
            if log_every and (ep_i % log_every == 0 or ep_i == episodes - 1):
                print(f"  episode {ep_i:3d}  reward={ep['reward']:+.3f}  "
                      f"forgetting={ep['forgetting']:+.3f}  "
                      f"errA|B={ep['errA_afterB']:.3f}  errB|B={ep['errB_afterB']:.3f}  "
                      f"update_frac={ep['update_frac']:.2f}")
        return history

    @torch.no_grad()
    def block_update_profile(self, model, taskA_fn, taskB_fn, eval_a, eval_b,
                             steps: int = 40) -> List[float]:
        """Per-block update rate under the trained (deterministic) policy — i.e.
        the plasticity profile the policy discovered. Compare to the 'workspace'
        U-shape."""
        self.run_episode(model, taskA_fn, taskB_fn, eval_a, eval_b, steps,
                         deterministic=True)
        report = model.adaptation_report()
        return [b["update_rate"] for b in report["per_block"]]
