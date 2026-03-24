"""
Self-Modifying Layers — Layers That Rewrite Their Own Weights
==============================================================

PLAIN ENGLISH:
    Normally, a neural network's weights are FROZEN during inference (prediction).
    Training updates them, but once deployed, they're fixed. The model can't
    adapt to new data it sees.

    Self-modifying layers break this rule. During the forward pass itself —
    while making predictions — the layer also updates its own weights based
    on what it sees. It's like a student who learns from each exam question
    WHILE taking the exam.

    The update rule is the "delta rule" from classical neuroscience:
        "If you predicted wrong, adjust your weights to be less wrong next time"

    Concretely, if the layer computes y = x @ W, and y doesn't match what
    it expected, it adjusts W right there and then:
        W_new = W - lr * (prediction_error) @ input^T

    This makes the model a "self-referential learner" — it learns its own
    update algorithm. The HOPE architecture chains multiple such layers,
    creating a system that can continuously adapt to new data even after
    training is complete.

MATH (Delta Rule):
    Forward pass:
        y = x @ W

    Self-modification (applied AFTER computing y, BEFORE next token):
        error = W @ x - target
        W = W - lr * error @ x^T

    Paper-exact version (no normalization):
        W = W - lr * (W @ x) @ x^T

    Stable version (with normalization):
        W = W - lr * (W @ x) @ x^T / (x^T @ x)

WHY THIS IS "NESTED":
    The self-modification is itself an optimization loop running inside
    the forward pass. So we have:
        - Level 0: the forward pass (fastest, runs every token)
        - Level 1: the self-modification (also every token, but changes weights)
        - Level 2: backpropagation training (slowest, runs every batch)

    Three nested optimization levels, each with its own learning rate
    and information flow. This is Nested Learning in action.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple


class SelfModifyingLinear(nn.Module):
    """A linear layer that modifies its own weights during the forward pass.

    Each time data flows through, the layer adjusts its weights using the
    delta rule. This creates an inner optimization loop INSIDE the model.

    Parameters:
        features:    Output dimension
        mod_lr:      Self-modification learning rate (how aggressively to adapt)
        normalized:  If True, normalize by input magnitude (more stable).
                     If False, use paper-exact unnormalized delta rule.
    """
    features: int
    mod_lr: float = 0.01
    normalized: bool = True

    @nn.compact
    def __call__(self, x, fast_weights=None):
        """
        Args:
            x: Input tensor, shape (batch, dim) or (batch, seq, dim)
            fast_weights: Optional override weights from previous self-modification.
                          If None, uses the base (slow) weights from training.

        Returns:
            output: Linear projection result
            new_fast_weights: Updated weights after self-modification
        """
        in_dim = x.shape[-1]

        # Base weights (these are the "slow weights" updated by backprop)
        W = self.param("kernel", nn.initializers.lecun_normal(), (in_dim, self.features))
        b = self.param("bias", nn.initializers.zeros, (self.features,))

        # Use fast weights if provided (from previous self-modification)
        if fast_weights is not None:
            W_current = fast_weights["W"]
            b_current = fast_weights["b"]
        else:
            W_current = W
            b_current = b

        # Forward pass: y = x @ W + b
        output = x @ W_current + b_current

        # --- SELF-MODIFICATION: Update weights based on what we just saw ---
        # Delta rule: W -= lr * (W @ x^T) @ x / ||x||^2
        # This pushes W to minimize ||W @ x||^2 — an autoencoding objective
        # that makes the weights encode the input distribution.

        # For batched inputs, use the mean across batch
        if x.ndim == 3:
            # (batch, seq, dim) -> average over batch and seq for update
            x_flat = x.reshape(-1, in_dim)  # (batch*seq, dim)
        else:
            x_flat = x  # (batch, dim)

        # Compute the delta rule update
        # prediction = x @ W (what the layer currently outputs)
        prediction = x_flat @ W_current  # (n, features)

        if self.normalized:
            # Normalized: divide by ||x||^2 for stability
            x_norm_sq = jnp.sum(x_flat ** 2, axis=-1, keepdims=True) + 1e-8
            # grad_W = x^T @ prediction / ||x||^2, averaged over samples
            delta_W = (x_flat.T @ prediction) / x_norm_sq.mean()
        else:
            # Paper-exact: no normalization
            delta_W = x_flat.T @ prediction

        # Average over the number of samples
        delta_W = delta_W / x_flat.shape[0]

        # Apply the modification
        new_W = W_current - self.mod_lr * delta_W
        new_b = b_current  # Bias typically not modified

        new_fast_weights = {"W": new_W, "b": new_b}

        return output, new_fast_weights


class SelfModifyingMLP(nn.Module):
    """An MLP where each layer self-modifies its weights.

    This is a stack of SelfModifyingLinear layers — each one independently
    adjusts its weights during the forward pass.

    The result is a network that gets better at processing the current
    data distribution AS it processes it. Like an employee who gets better
    at their job every hour, not just at annual reviews.
    """
    hidden_dim: int
    output_dim: int
    num_layers: int = 2
    mod_lr: float = 0.01
    normalized: bool = True

    @nn.compact
    def __call__(self, x, fast_weights_list=None):
        """
        Args:
            x: Input, shape (batch, dim) or (batch, seq, dim)
            fast_weights_list: List of fast_weights dicts, one per layer

        Returns:
            output: MLP output
            new_fast_weights_list: Updated fast weights for all layers
        """
        if fast_weights_list is None:
            fast_weights_list = [None] * self.num_layers

        new_fast_weights_list = []

        for i in range(self.num_layers):
            out_dim = self.hidden_dim if i < self.num_layers - 1 else self.output_dim
            layer = SelfModifyingLinear(
                features=out_dim,
                mod_lr=self.mod_lr,
                normalized=self.normalized,
                name=f"layer_{i}",
            )
            x, fast_w = layer(x, fast_weights_list[i])
            new_fast_weights_list.append(fast_w)

            if i < self.num_layers - 1:
                x = nn.gelu(x)

        return x, new_fast_weights_list
