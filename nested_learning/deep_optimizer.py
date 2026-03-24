"""
Deep Momentum Gradient Descent (DMGD) — The "Deep Optimizer"
=============================================================

PLAIN ENGLISH:
    Normal optimizers like SGD with momentum are simple: they keep a running
    average of past gradients (momentum) and use that to update weights.

    The Nested Learning paper shows that this momentum IS itself an associative
    memory — it's a linear map that compresses gradient history. So why not
    replace it with something smarter?

    Deep Momentum GD replaces the simple linear momentum with a small neural
    network (an MLP). This MLP takes in the current gradient and the previous
    momentum state, and outputs a better update direction.

    The MLP is trained with its OWN loss function — making this a NESTED
    optimization problem:
        - Outer level: train the model weights using the MLP's output
        - Inner level: train the MLP to produce good update directions

    This is the paper's central insight: adding an optimization level is
    the same as adding a layer. The optimizer IS part of the architecture.

WHY THIS MATTERS:
    Standard momentum treats all gradient dimensions equally and uses a fixed
    decay rate. The learned MLP can:
    - Pay more attention to important gradient dimensions
    - Adapt its momentum behavior based on the loss landscape
    - Learn non-linear compression of gradient history
    - Act differently in different parts of parameter space

MATH:
    Standard SGD with momentum:
        m_t = beta * m_{t-1} + grad_t          (linear combination)
        theta_t = theta_{t-1} - lr * m_t       (parameter update)

    Deep Momentum GD:
        m_t = MLP([grad_t, m_{t-1}])            (learned combination)
        theta_t = theta_{t-1} - lr * m_t        (parameter update)

    The MLP's own loss (inner optimization):
        l_inner = ||MLP([grad_t, m_{t-1}]) - grad_t||^2
        (i.e., the MLP should at minimum reconstruct the current gradient,
         but it can also incorporate useful information from m_{t-1})
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Optional, Tuple, NamedTuple
import optax


class MemoryMLP(nn.Module):
    """The learned momentum network — replaces simple linear momentum.

    Takes in [current_gradient, previous_momentum] and outputs the new momentum.
    This is the "deep" part of Deep Momentum GD.

    Think of it as: instead of "new_momentum = 0.9 * old + gradient" (fixed rule),
    we have "new_momentum = NeuralNet(old, gradient)" (learned rule).
    """
    hidden_dim: int = 128
    output_dim: int = None  # defaults to input gradient dim
    num_layers: int = 2

    @nn.compact
    def __call__(self, gradient, momentum):
        """
        Args:
            gradient: Current gradient, shape (dim,) or (batch, dim)
            momentum: Previous momentum state, same shape as gradient
        Returns:
            new_momentum: The learned update direction
        """
        # Concatenate gradient and momentum — the MLP sees both
        x = jnp.concatenate([gradient, momentum], axis=-1)
        out_dim = self.output_dim or gradient.shape[-1]

        for i in range(self.num_layers - 1):
            x = nn.Dense(self.hidden_dim, name=f"layer_{i}")(x)
            x = nn.gelu(x)

        x = nn.Dense(out_dim, name=f"layer_{self.num_layers - 1}")(x)
        return x


class DeepMomentumState(NamedTuple):
    """State carried between optimization steps."""
    momentum: Any          # Current momentum state (pytree matching params)
    memory_params: Any     # Parameters of the MemoryMLP
    memory_opt_state: Any  # Optimizer state for training the MemoryMLP
    step: int              # Current step count


class DeepMomentumGD:
    """Deep Momentum Gradient Descent optimizer.

    This is an optax-compatible optimizer that replaces standard momentum
    with a learned neural network (MemoryMLP).

    Usage:
        optimizer = DeepMomentumGD(param_dim=256, lr=1e-3)
        state = optimizer.init(params)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)

    The optimizer trains its own internal MLP at each step (nested optimization).

    Parameters:
        param_dim:    Dimension of the parameter vectors being optimized
        lr:           Outer learning rate (for the actual model parameters)
        memory_lr:    Inner learning rate (for training the MemoryMLP itself)
        hidden_dim:   Hidden dimension of the MemoryMLP
        inner_steps:  How many gradient steps to take on the MLP per outer step
    """

    def __init__(
        self,
        param_dim: int,
        lr: float = 1e-3,
        memory_lr: float = 1e-4,
        hidden_dim: int = 64,
        inner_steps: int = 1,
        rng_seed: int = 42,
    ):
        self.param_dim = param_dim
        self.lr = lr
        self.memory_lr = memory_lr
        self.hidden_dim = hidden_dim
        self.inner_steps = inner_steps
        self.rng_seed = rng_seed

        # Create the memory MLP
        self.memory_mlp = MemoryMLP(
            hidden_dim=hidden_dim,
            output_dim=param_dim,
        )

        # Inner optimizer for training the MLP
        self.inner_optimizer = optax.adam(memory_lr)

    def init(self, rng_key=None):
        """Initialize the optimizer state."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.rng_seed)

        # Initialize MLP parameters
        dummy_grad = jnp.zeros(self.param_dim)
        dummy_momentum = jnp.zeros(self.param_dim)
        memory_params = self.memory_mlp.init(rng_key, dummy_grad, dummy_momentum)

        # Initialize inner optimizer state
        memory_opt_state = self.inner_optimizer.init(memory_params)

        return DeepMomentumState(
            momentum=jnp.zeros(self.param_dim),
            memory_params=memory_params,
            memory_opt_state=memory_opt_state,
            step=0,
        )

    def _inner_loss(self, memory_params, gradient, momentum):
        """Loss for training the MemoryMLP.

        The MLP should produce an update direction that at minimum reconstructs
        the current gradient (L2 regression). This is the "inner optimization".

        L_inner = ||MLP(grad, momentum) - grad||^2

        You could also use a surrogate loss that encourages the MLP to produce
        updates that actually reduce the outer loss, but L2 regression is
        simpler and matches the paper's formulation.
        """
        predicted = self.memory_mlp.apply(memory_params, gradient, momentum)
        return jnp.mean((predicted - gradient) ** 2)

    def update(self, gradient, state: DeepMomentumState):
        """Compute the parameter update using deep momentum.

        This is where the nested optimization happens:
        1. Train the MLP on the current (gradient, momentum) pair  ← inner level
        2. Use the MLP to compute the new momentum                 ← outer level
        3. Return -lr * new_momentum as the parameter update

        Args:
            gradient: Gradient of the outer loss w.r.t. parameters, shape (param_dim,)
            state: Current optimizer state

        Returns:
            update: The parameter update to apply
            new_state: Updated optimizer state
        """
        memory_params = state.memory_params
        memory_opt_state = state.memory_opt_state
        momentum = state.momentum

        # --- INNER OPTIMIZATION: Train the MLP ---
        # This is the nested part — we're doing gradient descent INSIDE
        # the gradient descent step. Meta-learning in miniature.
        for _ in range(self.inner_steps):
            inner_loss, inner_grads = jax.value_and_grad(self._inner_loss)(
                memory_params, gradient, momentum
            )
            inner_updates, memory_opt_state = self.inner_optimizer.update(
                inner_grads, memory_opt_state
            )
            memory_params = optax.apply_updates(memory_params, inner_updates)

        # --- OUTER UPDATE: Use the trained MLP to compute new momentum ---
        new_momentum = self.memory_mlp.apply(memory_params, gradient, momentum)

        # The actual parameter update
        param_update = -self.lr * new_momentum

        new_state = DeepMomentumState(
            momentum=new_momentum,
            memory_params=memory_params,
            memory_opt_state=memory_opt_state,
            step=state.step + 1,
        )

        return param_update, new_state


class SimpleMomentumGD:
    """Standard SGD with momentum — for comparison with DeepMomentumGD.

    This is the "flat" (non-nested) baseline:
        m_t = beta * m_{t-1} + grad_t
        update = -lr * m_t
    """

    def __init__(self, lr: float = 1e-3, beta: float = 0.9):
        self.lr = lr
        self.beta = beta

    def init(self):
        return {"momentum": None, "step": 0}

    def update(self, gradient, state):
        if state["momentum"] is None:
            momentum = gradient
        else:
            momentum = self.beta * state["momentum"] + gradient

        update = -self.lr * momentum
        new_state = {"momentum": momentum, "step": state["step"] + 1}
        return update, new_state
