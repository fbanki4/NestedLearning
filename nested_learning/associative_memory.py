"""
Associative Memory — The Foundation of Nested Learning
=======================================================

PLAIN ENGLISH:
    Imagine a notebook where you jot down facts as (question, answer) pairs.
    When someone asks a question, you flip through the notebook and find the
    closest matching question, then read back the answer.

    That's associative memory: store (key, value) pairs, retrieve by similarity.

    What's special here is HOW we write to the notebook. Instead of just
    appending entries, we use gradient descent — the same algorithm that trains
    neural networks. Each new fact slightly adjusts ALL existing entries so the
    notebook as a whole becomes more accurate.

    The "surprise" mechanism means: if a new fact is very different from what
    the notebook already knows (high loss = high surprise), we write it down
    more aggressively. Boring, predictable facts barely change the notebook.

MATH (from the Titans paper, Behrouz et al. 2025):
    The memory M is a small neural network (an MLP).
    For each input x_t, we compute:
        k_t = x_t @ W_K          (key)
        v_t = x_t @ W_V          (value)
        q_t = x_t @ W_Q          (query)

    Loss (how surprised the memory is):
        l(M; x_t) = ||M(k_t) - v_t||^2

    Surprise signal (gradient of loss = how much to update):
        S_t = eta * S_{t-1} - theta * grad_M l(M; x_t)
              ^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
              past surprise   momentary surprise (current gradient)

    Memory update (with forgetting):
        M_t = (1 - alpha_t) * M_{t-1} + S_t

    Retrieval:
        y_t = M(q_t)    (just a forward pass, no weight update)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple


class MemoryNetwork(nn.Module):
    """A small MLP that serves as the associative memory.

    Think of this as the "notebook" — its weights encode all the stored facts.
    When you pass a key through it, the output is the recalled value.
    """
    hidden_dim: int
    output_dim: int
    num_layers: int = 2

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers - 1):
            x = nn.Dense(self.hidden_dim, name=f"layer_{i}")(x)
            x = nn.gelu(x)
        x = nn.Dense(self.output_dim, name=f"layer_{self.num_layers - 1}")(x)
        return x


class AssociativeMemory(nn.Module):
    """Gradient-based associative memory from the Titans architecture.

    This is the building block that makes Nested Learning possible.
    The key insight: this memory module IS an optimization problem.
    Writing to it = doing gradient descent. Reading from it = forward pass.

    Parameters:
        dim:         Dimension of input tokens
        memory_dim:  Internal dimension of the memory network
        num_heads:   Number of independent memory heads (like multi-head attention)
        lr:          How fast to write new memories (learning rate for memory updates)
        momentum:    How much past surprise carries forward (0 = only current, 1 = full history)
    """
    dim: int
    memory_dim: int = 64
    num_heads: int = 4
    lr: float = 0.01
    momentum: float = 0.9
    normalize_surprise: bool = False

    @nn.compact
    def __call__(self, x, memory_state=None):
        """Process a sequence through the associative memory.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            memory_state: Optional tuple of (memory_params, surprise_momentum)
                          from a previous call (for continuing across chunks)

        Returns:
            output: Retrieved memories, shape (batch, seq_len, dim)
            new_memory_state: Updated state to pass to the next call
        """
        batch_size, seq_len, _ = x.shape
        head_dim = self.dim // self.num_heads

        # Project input into keys, queries, values
        # Keys = what we write with, Queries = what we read with, Values = what we store
        keys = nn.Dense(self.dim, name="key_proj")(x)      # (batch, seq, dim)
        queries = nn.Dense(self.dim, name="query_proj")(x)  # (batch, seq, dim)
        values = nn.Dense(self.dim, name="value_proj")(x)   # (batch, seq, dim)

        # Reshape for multi-head: (batch, seq, num_heads, head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, head_dim)
        queries = queries.reshape(batch_size, seq_len, self.num_heads, head_dim)
        values = values.reshape(batch_size, seq_len, self.num_heads, head_dim)

        # Initialize memory as a simple linear map per head: W of shape (head_dim, head_dim)
        # This is M(k) = k @ W, so M is a linear associative memory
        if memory_state is None:
            # Start with small random-ish weights (identity + noise feels like "blank notebook")
            memory_W = jnp.zeros((batch_size, self.num_heads, head_dim, head_dim))
            surprise_S = jnp.zeros((batch_size, self.num_heads, head_dim, head_dim))
        else:
            memory_W, surprise_S = memory_state

        # Learnable forget gate: alpha controls how much old memory to keep
        # alpha near 0 = keep everything, alpha near 1 = forget fast
        alpha_logit = self.param(
            "forget_gate",
            nn.initializers.constant(-2.0),  # sigmoid(-2) ≈ 0.12, so mostly remembering
            (self.num_heads,),
        )
        alpha = jax.nn.sigmoid(alpha_logit)  # (num_heads,)

        # Process each token sequentially (this is the recurrent part)
        outputs = []
        for t in range(seq_len):
            k_t = keys[:, t, :, :]    # (batch, heads, head_dim)
            q_t = queries[:, t, :, :]  # (batch, heads, head_dim)
            v_t = values[:, t, :, :]   # (batch, heads, head_dim)

            # --- RETRIEVE: Read from memory using query ---
            # y_t = M(q_t) = q_t @ W  (no weight update during read)
            y_t = jnp.einsum("bhd,bhde->bhe", q_t, memory_W)  # (batch, heads, head_dim)

            # --- COMPUTE SURPRISE: How wrong is the memory about this input? ---
            # predicted = M(k_t) = k_t @ W
            predicted = jnp.einsum("bhd,bhde->bhe", k_t, memory_W)  # (batch, heads, head_dim)
            # error = predicted - actual (this IS the gradient for linear memory)
            error = predicted - v_t  # (batch, heads, head_dim)

            # Gradient of ||M(k) - v||^2 w.r.t. W is: 2 * (M(k) - v) @ k^T
            # For a linear memory M(k) = k @ W, grad_W = k^T @ error
            grad_W = jnp.einsum("bhd,bhe->bhde", k_t, error)  # (batch, heads, hd, hd)

            # --- UPDATE SURPRISE with momentum ---
            # S_t = eta * S_{t-1} - theta * grad
            # This is exactly SGD with momentum! The paper's key insight.
            #
            # Optional normalization: divide by input norm so that random
            # noise (which has high norm) doesn't produce outsized surprise.
            # Without this, noise triggers MORE surprise than structured
            # novel data, causing the memory to waste capacity on garbage.
            if self.normalize_surprise:
                input_norm = jnp.linalg.norm(k_t, axis=-1, keepdims=True)  # (batch, heads, 1)
                input_norm = jnp.maximum(input_norm, 1e-6)[:, :, :, None]  # (batch, heads, 1, 1)
                grad_W = grad_W / input_norm
            surprise_S = self.momentum * surprise_S - self.lr * grad_W

            # --- UPDATE MEMORY with forgetting ---
            # M_t = (1 - alpha) * M_{t-1} + S_t
            alpha_expanded = alpha[None, :, None, None]  # broadcast to (1, heads, 1, 1)
            memory_W = (1.0 - alpha_expanded) * memory_W + surprise_S

            outputs.append(y_t)

        # Stack outputs: (batch, seq_len, heads, head_dim) -> (batch, seq_len, dim)
        output = jnp.stack(outputs, axis=1)
        output = output.reshape(batch_size, seq_len, self.dim)

        # Final projection to mix head outputs
        output = nn.Dense(self.dim, name="output_proj")(output)

        return output, (memory_W, surprise_S)


class LinearAssociativeMemory(nn.Module):
    """A simpler version: linear attention reinterpreted as associative memory.

    Instead of softmax attention (which needs the full sequence), linear attention
    keeps a running matrix M = sum(k_i^T @ v_i) and retrieves with y = q @ M.

    This is the simplest associative memory — no surprise gating, no forgetting.
    Useful as a baseline to compare against the full Titans memory.

    MATH:
        M_t = M_{t-1} + k_t^T @ v_t    (write: outer product accumulation)
        y_t = q_t @ M_t                 (read: matrix-vector product)
    """
    dim: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, x, memory_state=None):
        batch_size, seq_len, _ = x.shape
        head_dim = self.dim // self.num_heads

        keys = nn.Dense(self.dim, name="key_proj")(x)
        queries = nn.Dense(self.dim, name="query_proj")(x)
        values = nn.Dense(self.dim, name="value_proj")(x)

        # Feature map: use elu+1 to keep values positive (standard for linear attention)
        keys = jax.nn.elu(keys) + 1.0
        queries = jax.nn.elu(queries) + 1.0

        keys = keys.reshape(batch_size, seq_len, self.num_heads, head_dim)
        queries = queries.reshape(batch_size, seq_len, self.num_heads, head_dim)
        values = values.reshape(batch_size, seq_len, self.num_heads, head_dim)

        if memory_state is None:
            memory_M = jnp.zeros((batch_size, self.num_heads, head_dim, head_dim))
        else:
            memory_M = memory_state

        outputs = []
        for t in range(seq_len):
            k_t = keys[:, t]    # (batch, heads, hd)
            q_t = queries[:, t]
            v_t = values[:, t]

            # Write: M += k^T @ v (outer product)
            memory_M = memory_M + jnp.einsum("bhd,bhe->bhde", k_t, v_t)

            # Read: y = q @ M
            y_t = jnp.einsum("bhd,bhde->bhe", q_t, memory_M)

            # Normalize
            z_t = jnp.einsum("bhd,bhd->bh", q_t, k_t)  # normalizer
            y_t = y_t / (z_t[:, :, None] + 1e-6)

            outputs.append(y_t)

        output = jnp.stack(outputs, axis=1).reshape(batch_size, seq_len, self.dim)
        output = nn.Dense(self.dim, name="output_proj")(output)
        return output, memory_M
