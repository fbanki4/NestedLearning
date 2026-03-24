"""
Training Utilities for Nested Learning
========================================

PLAIN ENGLISH:
    Training a Nested Learning model is trickier than a standard Transformer
    because we have MULTIPLE optimization levels running simultaneously:

    1. Self-modification:    Happens inside the forward pass (no extra work)
    2. Associative memory:   Happens inside the forward pass (no extra work)
    3. CMS multi-frequency:  We need to mask gradients based on step count
    4. Backpropagation:      Standard gradient descent on model parameters

    The training loop handles all of this. The key trick is the CMS scheduling:
    at each step, only certain CMS levels receive gradient updates.

    We also optionally use the Deep Momentum GD optimizer instead of Adam,
    which adds yet another optimization level (the optimizer's own MLP is
    trained at each step too).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Any, Callable, Dict, Optional, Tuple
from functools import partial


def cross_entropy_loss(logits, targets):
    """Standard cross-entropy loss for language modeling.

    Compares the model's predicted token probabilities against the actual
    next tokens. Lower = better predictions.
    """
    # Shift: logits[..., :-1] predicts targets[..., 1:]
    # (predict next token from current token)
    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:]

    # One-hot encode targets
    vocab_size = logits.shape[-1]
    one_hot = jax.nn.one_hot(shift_targets, vocab_size)

    # Cross-entropy
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)

    return jnp.mean(loss)


def create_train_state(model, rng_key, learning_rate=1e-4, weight_decay=0.01):
    """Initialize model parameters and optimizer.

    Args:
        model: A HOPE model instance
        rng_key: JAX random key for initialization
        learning_rate: Learning rate for Adam optimizer
        weight_decay: Weight decay for regularization

    Returns:
        params: Initialized model parameters
        optimizer: Optax optimizer
        opt_state: Optimizer state
    """
    # Create dummy input to initialize parameters
    dummy_input = jnp.ones((1, 16), dtype=jnp.int32)
    params = model.init(rng_key, dummy_input, training=False)

    # AdamW optimizer with linear warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=100,
        decay_steps=10000,
        end_value=learning_rate * 0.1,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(schedule, weight_decay=weight_decay),
    )

    opt_state = optimizer.init(params)

    return params, optimizer, opt_state


@partial(jax.jit, static_argnums=(1,))
def train_step(state, model, batch, step):
    """One training step with nested optimization.

    This is where the magic happens. The forward pass internally runs
    self-modification and memory updates (levels 0-1). The CMS handles
    multi-frequency updates (levels 2+). And backprop updates the base
    weights (outermost level).

    Args:
        state: Tuple of (params, opt_state, memory_states, fast_weights)
        model: The HOPE model (static — passed via static_argnums)
        batch: Token IDs, shape (batch_size, seq_len)
        step: Current training step (for CMS scheduling)

    Returns:
        new_state: Updated state
        metrics: Dict of loss and other metrics
    """
    params, opt_state, memory_states, fast_weights, optimizer = state

    def loss_fn(params):
        logits, new_mem, new_fw, cms_outs = model.apply(
            params,
            batch,
            memory_states=memory_states,
            fast_weights_list=fast_weights,
            step=step,
            training=True,
            rngs={"dropout": jax.random.PRNGKey(step)},
        )
        loss = cross_entropy_loss(logits, batch)
        return loss, (new_mem, new_fw, cms_outs)

    # Compute loss and gradients
    (loss, (new_mem, new_fw, cms_outs)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params)

    # Update parameters with optimizer
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    new_state = (new_params, new_opt_state, new_mem, new_fw, optimizer)
    metrics = {"loss": loss}

    return new_state, metrics


def train_loop(
    model,
    train_data,
    num_steps: int = 1000,
    batch_size: int = 8,
    seq_len: int = 128,
    learning_rate: float = 1e-4,
    rng_seed: int = 42,
    log_every: int = 50,
):
    """Full training loop for a HOPE model.

    Args:
        model:         HOPE model instance
        train_data:    Token IDs array, shape (num_tokens,)
        num_steps:     Number of training steps
        batch_size:    Batch size
        seq_len:       Sequence length
        learning_rate: Learning rate
        rng_seed:      Random seed
        log_every:     Print loss every N steps

    Returns:
        params:  Trained parameters
        losses:  List of (step, loss) tuples for plotting
    """
    rng_key = jax.random.PRNGKey(rng_seed)
    init_key, data_key = jax.random.split(rng_key)

    # Initialize
    params, optimizer, opt_state = create_train_state(
        model, init_key, learning_rate
    )

    memory_states = [None] * model.num_layers
    fast_weights = [None] * model.num_layers

    losses = []
    num_tokens = len(train_data)

    print(f"Training HOPE model:")
    print(f"  Parameters: {sum(p.size for p in jax.tree.leaves(params)):,}")
    print(f"  Steps: {num_steps}, Batch: {batch_size}, Seq: {seq_len}")
    print(f"  CMS levels: {model.cms_num_levels}")
    print()

    for step in range(num_steps):
        # Sample a random batch of sequences
        data_key, batch_key = jax.random.split(data_key)
        starts = jax.random.randint(
            batch_key, (batch_size,), 0, num_tokens - seq_len
        )
        batch = jnp.stack([train_data[s : s + seq_len] for s in starts])

        # Training step (all nested levels run simultaneously)
        state = (params, opt_state, memory_states, fast_weights, optimizer)
        state, metrics = train_step(state, model, batch, step)
        params, opt_state, memory_states, fast_weights, optimizer = state

        loss = float(metrics["loss"])
        losses.append((step, loss))

        if step % log_every == 0:
            print(f"  Step {step:5d} | Loss: {loss:.4f}")

    print(f"\nTraining complete. Final loss: {losses[-1][1]:.4f}")
    return params, losses
