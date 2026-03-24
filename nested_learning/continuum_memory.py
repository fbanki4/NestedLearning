"""
Continuum Memory System (CMS) — Multi-Frequency Memory
========================================================

PLAIN ENGLISH:
    Think about how YOUR memory works:
    - Working memory: what you're thinking about RIGHT NOW (seconds)
    - Short-term memory: what happened this morning (hours)
    - Long-term memory: your childhood, your skills (years)

    Each type of memory updates at a different speed. Working memory changes
    every moment. Long-term memory barely changes day to day.

    Standard neural networks have only TWO memory types:
    - Attention (short-term): processes the current input window
    - Weights (long-term): fixed after training

    The Continuum Memory System adds memory at MANY different speeds.
    It's a chain of small MLPs, each updating at a different frequency:
    - Level 0: updates every step      (like working memory)
    - Level 1: updates every 10 steps  (like short-term memory)
    - Level 2: updates every 100 steps (like long-term memory)

    The output of each level feeds into the next, so fast memories inform
    slow memories, and slow memories provide context for fast ones.

WHY THIS HELPS:
    Without CMS, a model learning a new task might overwrite knowledge
    from old tasks (catastrophic forgetting). With CMS:
    - New information goes into fast-updating levels
    - Established knowledge is in slow-updating levels (protected)
    - The system naturally separates transient patterns from lasting ones

MATH:
    For L levels with chunk sizes C_0 < C_1 < ... < C_{L-1}:

    Level l has an MLP f_l with parameters theta_l.
    theta_l is updated only once every C_l steps.

    Forward pass (nested composition):
        y_0 = f_0(x)                    (fastest level processes input)
        y_1 = f_1(y_0)                  (next level processes level 0's output)
        ...
        y_{L-1} = f_{L-1}(y_{L-2})      (slowest level)
        output = y_{L-1}

    Update schedule:
        At step t, only update level l if (t % C_l == 0)

    This creates a hierarchy where deeper levels capture slower-changing patterns.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Optional, Tuple


class ContinuumMLP(nn.Module):
    """A single MLP block in the Continuum Memory System.

    Each ContinuumMLP is one "level" of memory, processing at its own frequency.
    It's a standard MLP but we control WHEN it gets its gradients.

    Parameters:
        dim:       Input/output dimension
        hidden_dim: Hidden layer dimension
        level:     Which CMS level (0 = fastest, higher = slower)
    """
    dim: int
    hidden_dim: int
    level: int = 0

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input tensor, shape (batch, seq, dim)
        Returns:
            output: Processed tensor, same shape
        """
        residual = x

        x = nn.Dense(self.hidden_dim, name="up")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, name="down")(x)

        # Residual connection — the level refines the input, not replaces it
        return x + residual


class ContinuumMemorySystem(nn.Module):
    """Multi-frequency memory system — the core of Nested Learning.

    A chain of MLP blocks where each block updates at a different frequency.
    Level 0 updates every step (fast/working memory).
    Level 1 updates every chunk_sizes[1] steps (medium-term).
    Level 2 updates every chunk_sizes[2] steps (long-term).
    ...and so on.

    The key insight: this is EXACTLY like stacking optimization levels.
    Each level is its own optimization problem with its own update rate.

    Parameters:
        dim:         Model dimension
        hidden_dim:  Hidden dimension for each MLP block
        num_levels:  How many memory frequencies (THIS IS THE OPEN PROBLEM —
                     how many levels should you use? See scaling_exploration.py)
        chunk_sizes: Update frequency for each level. Level l updates every
                     chunk_sizes[l] steps. If None, uses exponential schedule:
                     [1, 8, 64, 512, ...] = [8^0, 8^1, 8^2, 8^3, ...]
        use_residual: If True, add residual connections between levels (stable).
                      If False, pure nested composition per the paper.
    """
    dim: int
    hidden_dim: int = 256
    num_levels: int = 3
    chunk_sizes: Optional[List[int]] = None
    use_residual: bool = True

    def setup(self):
        # Default chunk sizes: exponentially increasing
        if self.chunk_sizes is not None:
            self._chunk_sizes = list(self.chunk_sizes)
        else:
            self._chunk_sizes = [8 ** i for i in range(self.num_levels)]
            # e.g., 3 levels -> [1, 8, 64]

        # One MLP per level
        self.levels = [
            ContinuumMLP(
                dim=self.dim,
                hidden_dim=self.hidden_dim,
                level=i,
                name=f"level_{i}",
            )
            for i in range(self.num_levels)
        ]

        # Learnable mixing weights: how much each level contributes
        # (helps the model learn what balance of fast/slow memory works best)
        self.level_gates = [
            nn.Dense(1, name=f"gate_{i}")
            for i in range(self.num_levels)
        ]

    def __call__(self, x, step: int = 0, training: bool = True):
        """Process input through the multi-frequency memory chain.

        Args:
            x: Input tensor, shape (batch, seq, dim)
            step: Current training step (determines which levels update)
            training: Whether we're training (affects which levels get gradients)

        Returns:
            output: Processed tensor, shape (batch, seq, dim)
            level_outputs: List of each level's output (for analysis/visualization)
        """
        level_outputs = []
        current = x

        for i in range(self.num_levels):
            # Determine if this level should update at this step
            should_update = (step % self._chunk_sizes[i] == 0)

            # Forward pass through this level's MLP
            if training and not should_update:
                # Don't compute gradients for this level at this step
                # This is the key mechanism: slow levels only learn occasionally
                level_out = jax.lax.stop_gradient(self.levels[i](current))
            else:
                level_out = self.levels[i](current)

            # Compute gate weight (learned importance of this level)
            gate = jax.nn.sigmoid(self.level_gates[i](current).mean(axis=-1, keepdims=True))

            if self.use_residual:
                # Residual: blend this level's output with its input
                current = gate * level_out + (1 - gate) * current
            else:
                # Paper-exact: pure composition (output of level i feeds level i+1)
                current = level_out

            level_outputs.append(level_out)

        return current, level_outputs

    def get_update_schedule(self, num_steps: int) -> dict:
        """Visualize which levels update at which steps.

        Returns a dictionary mapping each level to its update steps.
        Useful for understanding the multi-frequency behavior.

        Example for 3 levels with chunk_sizes [1, 8, 64]:
            Level 0: updates at steps 0, 1, 2, 3, 4, ...        (every step)
            Level 1: updates at steps 0, 8, 16, 24, 32, ...     (every 8th)
            Level 2: updates at steps 0, 64, 128, 192, ...      (every 64th)
        """
        schedule = {}
        for i, chunk_size in enumerate(self._chunk_sizes):
            update_steps = list(range(0, num_steps, chunk_size))
            schedule[f"level_{i} (every {chunk_size} steps)"] = update_steps
        return schedule


class FullyNestedCMS(nn.Module):
    """Paper-exact Fully Nested Continuum Memory System.

    In this version, each level's MLP is composed inside the next level's,
    creating a true nested function: f_L(...f_2(f_1(f_0(x)))...)

    The nesting means slower levels "wrap around" faster levels, providing
    a context that modulates how fast levels process their inputs.

    This is mathematically equivalent to having L nested optimization levels.
    """
    dim: int
    hidden_dim: int = 256
    num_levels: int = 3
    chunk_sizes: Optional[List[int]] = None

    def setup(self):
        if self.chunk_sizes is not None:
            self._chunk_sizes = list(self.chunk_sizes)
        else:
            self._chunk_sizes = [8 ** i for i in range(self.num_levels)]

        self.levels = [
            ContinuumMLP(
                dim=self.dim,
                hidden_dim=self.hidden_dim,
                level=i,
                name=f"level_{i}",
            )
            for i in range(self.num_levels)
        ]

    def __call__(self, x, step: int = 0, training: bool = True):
        """Pure nested composition: y = f_L(...f_1(f_0(x))...).

        Each level feeds directly into the next. No mixing, no gates.
        This is the cleanest expression of the nested optimization idea.
        """
        current = x
        level_outputs = []

        for i in range(self.num_levels):
            should_update = (step % self._chunk_sizes[i] == 0)

            if training and not should_update:
                out = jax.lax.stop_gradient(self.levels[i](current))
            else:
                out = self.levels[i](current)

            current = out
            level_outputs.append(out)

        return current, level_outputs
