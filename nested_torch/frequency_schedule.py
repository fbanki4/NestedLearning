"""
Depth-dependent update-frequency schedules
===========================================

PLAIN ENGLISH
-------------
Nested Learning says: a model has many "memories" that update at different
speeds. Fast memories grab new information; slow memories protect old
knowledge. In the original Continuum Memory System (CMS) the *levels* of a
brand-new model update at different speeds.

Here we do something the Nested Learning paper only sketches as an "ad hoc"
option: take an **already-trained, frozen** transformer and retroactively give
each of its blocks its own update speed. So block 0 might refresh its attached
memory every step, block 5 only every 32 steps, and so on.

THE ONE NEW IDEA IN THIS FILE
-----------------------------
Which blocks should be fast and which should be slow?

The plain CMS answer is "early = fast, later = slow" (a monotonic ramp). We
instead make the **middle** blocks the fast ones and keep the first and last
blocks slow — a U-shaped *period* curve, equivalently an ∩-shaped *frequency*
curve that peaks in the middle.

Why the middle? Anthropic's July-2026 "J-space / J-lens" interpretability work
(a Jacobian lens onto the residual stream) found that a small, sparse
"global workspace" — a privileged set of directions the model reads, reports,
and reasons with — lives *only in the middle block* of a transformer, echoing
Global Workspace Theory. If that middle region is where concepts are actively
held and combined, then that is exactly where we want plasticity when we adapt
to new tasks: let the workspace re-organize quickly while the early feature
extractors and the late read-out layers stay stable (which protects old tasks
from catastrophic forgetting).

VOCABULARY
----------
- ``period[i]``     : block i's memory may update once every ``period[i]``
                     forward steps. Small period = high frequency = plastic.
- ``plasticity[i]`` : a 0..1 curve, 1 where the block is most plastic. It is
                     the shape we derive ``period`` from, and is handy for
                     logging / optionally scaling the learning rate.

This module has **no torch dependency** on purpose — it is pure Python so the
schedule can be inspected and unit-tested without a GPU or even torch present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

# The schedules we support. "workspace" is the novel, J-space-aligned one.
SCHEDULES = ("workspace", "edge_fast", "uniform", "shallow_fast", "deep_fast")


@dataclass
class FrequencySchedule:
    """The per-block update plan for a retrofit.

    Attributes
    ----------
    name        : which schedule generated this.
    periods     : ``periods[i]`` = update block i's memory every this many steps.
    plasticity  : ``plasticity[i]`` in [0, 1], the shape periods were derived
                  from (1.0 = most plastic / fastest block).
    min_period  : the fastest period used (applied at the plasticity peak).
    max_period  : the slowest period used (applied at plasticity ~ 0).
    """

    name: str
    periods: List[int]
    plasticity: List[float]
    min_period: int
    max_period: int

    @property
    def num_layers(self) -> int:
        return len(self.periods)

    def workspace_layers(self, k: Optional[int] = None) -> List[int]:
        """Return the indices of the ``k`` most-plastic blocks (the 'workspace').

        With no ``k`` it returns every block whose plasticity is at least half
        of the maximum — a reasonable definition of "the fast middle".
        """
        order = sorted(range(self.num_layers), key=lambda i: self.plasticity[i], reverse=True)
        if k is not None:
            return sorted(order[:k])
        peak = max(self.plasticity) if self.plasticity else 0.0
        thresh = 0.5 * peak
        return [i for i in range(self.num_layers) if self.plasticity[i] >= thresh]

    def summary(self) -> str:
        """A little ASCII bar chart of the frequency profile (nice for demos)."""
        lines = [f"FrequencySchedule('{self.name}', {self.num_layers} blocks, "
                 f"period {self.min_period}..{self.max_period})"]
        peak = max(self.plasticity) if self.plasticity else 1.0
        peak = peak if peak > 0 else 1.0
        for i in range(self.num_layers):
            bar = "#" * int(round(20 * self.plasticity[i] / peak))
            lines.append(
                f"  block {i:2d}  period={self.periods[i]:4d}  "
                f"plasticity={self.plasticity[i]:.2f}  |{bar:<20}|"
            )
        ws = self.workspace_layers()
        lines.append(f"  workspace (fast middle) blocks: {ws}")
        return "\n".join(lines)


def _normalized_depths(num_layers: int) -> List[float]:
    """Map block index -> depth in [0, 1]. A single block sits at the center."""
    if num_layers <= 1:
        return [0.5]
    return [i / (num_layers - 1) for i in range(num_layers)]


def _plasticity_curve(name: str, num_layers: int, center: float, width: float) -> List[float]:
    """Return a 0..1 plasticity value per block for the chosen schedule shape."""
    depths = _normalized_depths(num_layers)

    if name == "uniform":
        return [1.0] * num_layers

    if name == "shallow_fast":  # early blocks fast (this is vanilla-CMS-like)
        return [1.0 - d for d in depths]

    if name == "deep_fast":  # later blocks fast
        return [d for d in depths]

    # Gaussian bump centered at ``center`` with std ``width``.
    bump = [math.exp(-((d - center) ** 2) / (2.0 * width ** 2)) for d in depths]

    if name == "workspace":  # peak in the middle -> the J-space global workspace
        return bump

    if name == "edge_fast":  # the control/ablation: fast at the ends, slow middle
        return [1.0 - b for b in bump]

    raise ValueError(f"unknown schedule '{name}', expected one of {SCHEDULES}")


def build_schedule(
    num_layers: int,
    name: str = "workspace",
    min_period: int = 1,
    max_period: int = 32,
    center: float = 0.5,
    width: float = 0.25,
) -> FrequencySchedule:
    """Build a per-block update schedule.

    Parameters
    ----------
    num_layers : number of transformer blocks in the frozen backbone.
    name       : one of ``SCHEDULES``. ``"workspace"`` is the J-space-aligned
                 U-shaped default (fast middle, slow ends).
    min_period : the *fastest* update period, applied where plasticity == 1
                 (the middle for ``"workspace"``). 1 = update every step.
    max_period : the *slowest* update period, applied where plasticity == 0
                 (the ends for ``"workspace"``).
    center     : location of the plasticity peak in [0, 1] (0.5 = the middle).
                 Only used by ``workspace`` / ``edge_fast``.
    width      : std of the Gaussian bump (fraction of total depth). Larger =
                 a wider fast region.

    Returns
    -------
    FrequencySchedule with integer ``periods`` and the underlying ``plasticity``.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    if min_period < 1 or max_period < 1:
        raise ValueError("periods must be >= 1")
    if min_period > max_period:
        raise ValueError("min_period must be <= max_period")
    if not (0.0 <= center <= 1.0):
        raise ValueError("center must be in [0, 1]")
    if width <= 0:
        raise ValueError("width must be > 0")

    plasticity = _plasticity_curve(name, num_layers, center, width)

    # Interpolate period between max_period (plasticity 0) and min_period
    # (plasticity 1). High plasticity -> small period -> frequent updates.
    periods: List[int] = []
    for p in plasticity:
        period = max_period + (min_period - max_period) * p
        periods.append(max(1, int(round(period))))

    return FrequencySchedule(
        name=name,
        periods=periods,
        plasticity=plasticity,
        min_period=min_period,
        max_period=max_period,
    )
