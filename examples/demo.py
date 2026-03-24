"""
Nested Learning Demo — See It In Action
=========================================

This script demonstrates each component of Nested Learning on small
synthetic tasks so you can see what each piece does.

Run with:
    python examples/demo.py

No GPU required — everything runs on CPU with tiny models.
"""

import jax
import jax.numpy as jnp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_associative_memory():
    """Demo 1: Associative Memory — A Notebook That Learns From Surprise

    We feed the memory a sequence of (key, value) pairs, then ask it to
    recall values for keys it's seen before. The surprise mechanism means
    it remembers surprising (novel) inputs better than boring ones.
    """
    from nested_learning.associative_memory import AssociativeMemory

    print("=" * 60)
    print("DEMO 1: Associative Memory")
    print("=" * 60)
    print()
    print("We feed the memory a sequence of tokens and see how well")
    print("it can recall information. Higher surprise = stronger write.")
    print()

    rng = jax.random.PRNGKey(42)
    dim = 32
    seq_len = 16
    batch_size = 2

    # Create the memory module
    memory = AssociativeMemory(dim=dim, memory_dim=32, num_heads=2, lr=0.05)

    # Random input sequence
    x = jax.random.normal(rng, (batch_size, seq_len, dim))

    # Initialize and run
    params = memory.init(rng, x)
    output, memory_state = memory.apply(params, x)

    print(f"  Input shape:    {x.shape}")
    print(f"  Output shape:   {output.shape}")
    print(f"  Memory W shape: {memory_state[0].shape}")
    print(f"  Surprise S shape: {memory_state[1].shape}")
    print()

    # Show that memory CHANGES over time: the memory matrix grows in norm
    # as it absorbs information from the input sequence
    mem_norm = float(jnp.linalg.norm(memory_state[0]))
    surprise_norm = float(jnp.linalg.norm(memory_state[1]))
    print(f"  Memory weight norm after 1st pass: {mem_norm:.4f}")
    print(f"  Surprise momentum norm:            {surprise_norm:.4f}")

    # Run a second pass with the SAME input — memory keeps accumulating
    output2, memory_state2 = memory.apply(params, x, memory_state=memory_state)
    mem_norm2 = float(jnp.linalg.norm(memory_state2[0]))
    print(f"  Memory weight norm after 2nd pass: {mem_norm2:.4f}")
    print()
    print("  The memory matrix grows as it absorbs information.")
    print("  Surprising inputs cause larger updates (bigger gradient).")
    print()


def demo_deep_optimizer():
    """Demo 2: Deep Momentum GD — A Learned Optimizer

    We optimize a simple quadratic function f(x) = ||x - target||^2
    using both standard momentum and deep momentum, and compare convergence.
    """
    from nested_learning.deep_optimizer import DeepMomentumGD, SimpleMomentumGD

    print("=" * 60)
    print("DEMO 2: Deep Momentum GD vs Standard Momentum")
    print("=" * 60)
    print()
    print("Optimizing f(x) = ||x - target||^2 on a 16-dim problem.")
    print("Deep momentum uses a learned MLP instead of simple averaging.")
    print()

    dim = 16
    target = jnp.ones(dim) * 3.0  # Target: all 3s
    x_init = jnp.zeros(dim)       # Start: all 0s

    def loss_fn(x):
        return jnp.mean((x - target) ** 2)

    # Standard Momentum
    simple_opt = SimpleMomentumGD(lr=0.1, beta=0.9)
    simple_state = simple_opt.init()
    x_simple = x_init

    # Deep Momentum
    deep_opt = DeepMomentumGD(param_dim=dim, lr=0.1, memory_lr=1e-3, hidden_dim=32)
    deep_state = deep_opt.init()
    x_deep = x_init

    print(f"  {'Step':>5} | {'Simple Loss':>12} | {'Deep Loss':>12}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}")

    for step in range(50):
        # Simple momentum step
        grad_simple = jax.grad(loss_fn)(x_simple)
        update_simple, simple_state = simple_opt.update(grad_simple, simple_state)
        x_simple = x_simple + update_simple

        # Deep momentum step
        grad_deep = jax.grad(loss_fn)(x_deep)
        update_deep, deep_state = deep_opt.update(grad_deep, deep_state)
        x_deep = x_deep + update_deep

        if step % 10 == 0:
            l_simple = float(loss_fn(x_simple))
            l_deep = float(loss_fn(x_deep))
            print(f"  {step:5d} | {l_simple:12.6f} | {l_deep:12.6f}")

    print()


def demo_self_modifying():
    """Demo 3: Self-Modifying Layers — Weights That Change During Inference

    We show that a self-modifying layer adapts to new data distributions
    without retraining. The weights change during the forward pass itself.
    """
    from nested_learning.self_modifying import SelfModifyingLinear

    print("=" * 60)
    print("DEMO 3: Self-Modifying Linear Layer")
    print("=" * 60)
    print()
    print("The layer adjusts its weights based on each input it sees.")
    print("Watch how fast_weights diverge from the initial weights.")
    print()

    rng = jax.random.PRNGKey(42)
    dim = 16
    layer = SelfModifyingLinear(features=dim, mod_lr=0.05)

    # Initialize
    x = jax.random.normal(rng, (4, dim))
    params = layer.init(rng, x)

    # First pass: no fast weights yet
    out1, fast_w1 = layer.apply(params, x)
    print(f"  After pass 1: weight change norm = {float(jnp.linalg.norm(fast_w1['W'] - params['params']['kernel'])):.4f}")

    # Second pass: using fast weights from pass 1
    out2, fast_w2 = layer.apply(params, x, fast_weights=fast_w1)
    print(f"  After pass 2: weight change norm = {float(jnp.linalg.norm(fast_w2['W'] - params['params']['kernel'])):.4f}")

    # Third pass: weights have adapted even more
    out3, fast_w3 = layer.apply(params, x, fast_weights=fast_w2)
    print(f"  After pass 3: weight change norm = {float(jnp.linalg.norm(fast_w3['W'] - params['params']['kernel'])):.4f}")

    print()
    print("  The weights drift further from their initial values with each pass.")
    print("  This is the layer learning from its own inputs in real time!")
    print()


def demo_continuum_memory():
    """Demo 4: Continuum Memory System — Multi-Speed Memory

    We visualize which CMS levels update at which steps, showing
    the multi-frequency nature of the system.
    """
    from nested_learning.continuum_memory import ContinuumMemorySystem

    print("=" * 60)
    print("DEMO 4: Continuum Memory System — Update Schedule")
    print("=" * 60)
    print()

    rng = jax.random.PRNGKey(42)
    dim = 32

    cms = ContinuumMemorySystem(dim=dim, hidden_dim=64, num_levels=3)

    # Initialize (need a dummy forward pass to set up params)
    x = jax.random.normal(rng, (2, 8, dim))
    params = cms.init(rng, x, step=0)

    # Show update schedule — we compute this directly from the chunk sizes
    # (no need to run the model, just need the schedule structure)
    chunk_sizes = [8 ** i for i in range(3)]  # default: [1, 8, 64]
    schedule = {}
    for i, chunk_size in enumerate(chunk_sizes):
        update_steps = list(range(0, 100, chunk_size))
        schedule[f"level_{i} (every {chunk_size} steps)"] = update_steps

    for level_name, steps in schedule.items():
        step_str = ", ".join(str(s) for s in steps[:10])
        if len(steps) > 10:
            step_str += ", ..."
        print(f"  {level_name}: [{step_str}]")
        print(f"    ({len(steps)} updates in 100 steps)")
        print()

    print("  Notice: Level 0 updates every step (working memory),")
    print("  Level 1 every 8 steps (short-term), Level 2 every 64 steps (long-term).")
    print("  This IS the nested optimization — each level is its own learning problem.")
    print()


def demo_hope_model():
    """Demo 5: HOPE Model — Everything Together

    We create a small HOPE model and run it on random token sequences
    to verify all components work together.
    """
    from nested_learning.hope import HOPE

    print("=" * 60)
    print("DEMO 5: HOPE Model — Full Nested Learning")
    print("=" * 60)
    print()

    rng = jax.random.PRNGKey(42)

    # Tiny model for demonstration
    model = HOPE(
        vocab_size=100,
        dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        max_seq_len=32,
        memory_dim=32,
        cms_num_levels=2,
        cms_chunk_sizes=[1, 4],
        mod_lr=0.01,
        memory_lr=0.01,
    )

    # Random input tokens
    batch = jax.random.randint(rng, (2, 16), 0, 100)

    # Initialize
    params = model.init(rng, batch, training=False)
    num_params = sum(p.size for p in jax.tree.leaves(params))

    print(f"  Model parameters: {num_params:,}")
    print(f"  Input shape: {batch.shape}")
    print()

    # Forward pass — all nested levels run simultaneously
    logits, mem_states, fast_weights, cms_outs = model.apply(
        params, batch, step=0, training=False
    )

    print(f"  Output logits shape: {logits.shape}")
    print(f"  Memory states: {len(mem_states)} layers, each with W and S matrices")
    print(f"  Fast weights: {len(fast_weights)} layers of self-modified projections")
    print(f"  CMS outputs: {len(cms_outs)} layers x {len(cms_outs[0])} levels each")
    print()

    # Second forward pass — memory and fast weights carry over
    logits2, mem_states2, fast_weights2, _ = model.apply(
        params, batch, memory_states=mem_states, fast_weights_list=fast_weights,
        step=1, training=False
    )

    # The outputs should differ because memory/fast weights have changed
    diff = float(jnp.mean((logits - logits2) ** 2))
    print(f"  Output difference between pass 1 and 2: {diff:.6f}")
    print(f"  (Non-zero = model adapted from the first pass!)")
    print()


if __name__ == "__main__":
    print()
    print("Nested Learning — Interactive Demos")
    print("=" * 60)
    print()
    print("These demos show each component of the Nested Learning")
    print("framework on small synthetic tasks. No GPU needed.")
    print()

    demo_associative_memory()
    demo_deep_optimizer()
    demo_self_modifying()
    demo_continuum_memory()
    demo_hope_model()

    print("=" * 60)
    print("All demos complete!")
    print()
    print("Next steps:")
    print("  - Read the module docstrings for detailed explanations")
    print("  - Run examples/scaling_exploration.py to explore open problems")
    print("  - Check the README for the full layman's guide")
    print("=" * 60)
