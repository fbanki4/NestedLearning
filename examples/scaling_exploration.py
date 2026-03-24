"""
Scaling Laws Exploration — The Open Problem
=============================================

THE OPEN PROBLEM:
    Nested Learning lets you add as many optimization levels as you want.
    Each level is like adding another layer of learning. But how many
    levels should you actually use?

    This is an open research question with no definitive answer yet.
    The paper demonstrates the framework and shows 2-3 levels work well,
    but doesn't provide a theory for the optimal number.

    This script provides tools to explore this question empirically.

WHAT WE CAN INVESTIGATE:
    1. How does the number of CMS levels affect performance?
    2. What's the compute cost of each additional level?
    3. Is there a diminishing returns point?
    4. How do chunk sizes (update frequencies) interact with depth?
    5. Does the optimal depth depend on task complexity?
    6. Are there scaling laws similar to the Chinchilla laws for Transformers?

HYPOTHESES TO TEST:
    H1: More levels = better at long-range dependencies, but diminishing returns
    H2: Optimal depth scales with log(sequence_length) — longer sequences
        need more levels to capture slower patterns
    H3: Compute scales linearly with levels but benefit scales sub-linearly
    H4: The ratio between chunk sizes matters more than absolute values
    H5: Self-modification provides most benefit at smaller model scales

Run with:
    python examples/scaling_exploration.py

This generates plots in results/ showing how performance varies with depth.
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nested_learning.hope import HOPE
from nested_learning.continuum_memory import ContinuumMemorySystem
from nested_learning.train import cross_entropy_loss


# ─────────────────────────────────────────────────────────────
# Experiment 1: CMS Depth vs. Performance
# ─────────────────────────────────────────────────────────────

def experiment_cms_depth(
    max_levels: int = 5,
    dim: int = 64,
    seq_len: int = 64,
    num_train_steps: int = 200,
    batch_size: int = 4,
    vocab_size: int = 50,
    seed: int = 42,
):
    """How does the number of CMS levels affect loss?

    We train small HOPE models with 1, 2, 3, ... CMS levels on the same
    synthetic data and compare their final loss.

    WHAT TO LOOK FOR:
    - If loss keeps dropping as we add levels: more levels = better
    - If loss plateaus: there's a sweet spot beyond which extra levels are wasteful
    - If loss increases: too many levels = overfitting or optimization difficulties

    This is the central open question: is there a scaling law for depth?
    """
    print("=" * 60)
    print("EXPERIMENT 1: CMS Depth vs Performance")
    print("=" * 60)
    print()
    print(f"Training HOPE models with 1 to {max_levels} CMS levels")
    print(f"on random token sequences (vocab={vocab_size}, seq={seq_len})")
    print()

    rng = jax.random.PRNGKey(seed)

    # Generate fixed synthetic training data
    data_key, rng = jax.random.split(rng)
    train_data = jax.random.randint(data_key, (1000,), 0, vocab_size)

    results = {}

    for num_levels in range(1, max_levels + 1):
        print(f"  --- {num_levels} CMS level(s) ---")

        model_key, rng = jax.random.split(rng)

        # Chunk sizes scale exponentially
        chunk_sizes = [4 ** i for i in range(num_levels)]

        model = HOPE(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=2,
            num_heads=4,
            ffn_dim=dim * 2,
            max_seq_len=seq_len,
            memory_dim=32,
            cms_num_levels=num_levels,
            cms_chunk_sizes=chunk_sizes,
        )

        # Initialize
        dummy_batch = jax.random.randint(model_key, (1, seq_len), 0, vocab_size)
        params = model.init(model_key, dummy_batch, training=False)
        num_params = sum(p.size for p in jax.tree.leaves(params))

        # Simple training loop (no JIT for simplicity)
        import optax
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        losses = []
        start_time = time.time()

        for step in range(num_train_steps):
            # Random batch
            batch_key, rng = jax.random.split(rng)
            starts = jax.random.randint(batch_key, (batch_size,), 0, len(train_data) - seq_len)
            batch = jnp.stack([train_data[s:s + seq_len] for s in starts])

            # Forward + backward
            def loss_fn(p):
                logits, _, _, _ = model.apply(p, batch, step=step, training=True,
                                              rngs={"dropout": jax.random.PRNGKey(step)})
                return cross_entropy_loss(logits, batch)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))

        elapsed = time.time() - start_time
        final_loss = np.mean(losses[-20:])  # Average of last 20 steps

        results[num_levels] = {
            "final_loss": final_loss,
            "num_params": num_params,
            "time_seconds": elapsed,
            "chunk_sizes": chunk_sizes,
            "losses": losses,
        }

        print(f"    Params: {num_params:,}")
        print(f"    Chunk sizes: {chunk_sizes}")
        print(f"    Final loss: {final_loss:.4f}")
        print(f"    Time: {elapsed:.1f}s")
        print()

    # Summary
    print("  " + "-" * 50)
    print(f"  {'Levels':>6} | {'Params':>10} | {'Loss':>8} | {'Time':>6}")
    print("  " + "-" * 50)
    for n, r in results.items():
        print(f"  {n:>6} | {r['num_params']:>10,} | {r['final_loss']:>8.4f} | {r['time_seconds']:>5.1f}s")
    print()

    return results


# ─────────────────────────────────────────────────────────────
# Experiment 2: Chunk Size Ratios
# ─────────────────────────────────────────────────────────────

def experiment_chunk_ratios(
    dim: int = 64,
    seq_len: int = 64,
    num_train_steps: int = 200,
    batch_size: int = 4,
    vocab_size: int = 50,
    seed: int = 42,
):
    """How do different chunk size ratios affect performance?

    With a fixed number of CMS levels (3), we try different spacing
    between the update frequencies:
    - Tight:  [1, 2, 4]     (levels update at similar rates)
    - Medium: [1, 8, 64]    (default exponential spacing)
    - Wide:   [1, 32, 1024] (very different rates)

    WHAT TO LOOK FOR:
    - If tight spacing wins: the model benefits from redundancy
    - If wide spacing wins: the model needs very different time scales
    - If medium wins: there's an optimal "spread" of update frequencies

    This tells us about the temporal structure of the learning problem.
    """
    print("=" * 60)
    print("EXPERIMENT 2: Chunk Size Ratios (3 CMS levels)")
    print("=" * 60)
    print()

    rng = jax.random.PRNGKey(seed)
    data_key, rng = jax.random.split(rng)
    train_data = jax.random.randint(data_key, (1000,), 0, vocab_size)

    configurations = {
        "tight  [1,2,4]":    [1, 2, 4],
        "medium [1,8,64]":   [1, 8, 64],
        "wide   [1,32,1024]": [1, 32, 1024],
        "linear [1,10,20]":  [1, 10, 20],
        "powers [1,4,16]":   [1, 4, 16],
    }

    results = {}

    for name, chunk_sizes in configurations.items():
        print(f"  --- {name} ---")

        model_key, rng = jax.random.split(rng)

        model = HOPE(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=2,
            num_heads=4,
            ffn_dim=dim * 2,
            max_seq_len=seq_len,
            memory_dim=32,
            cms_num_levels=3,
            cms_chunk_sizes=chunk_sizes,
        )

        dummy_batch = jax.random.randint(model_key, (1, seq_len), 0, vocab_size)
        params = model.init(model_key, dummy_batch, training=False)

        import optax
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        losses = []
        for step in range(num_train_steps):
            batch_key, rng = jax.random.split(rng)
            starts = jax.random.randint(batch_key, (batch_size,), 0, len(train_data) - seq_len)
            batch = jnp.stack([train_data[s:s + seq_len] for s in starts])

            def loss_fn(p):
                logits, _, _, _ = model.apply(p, batch, step=step, training=True,
                                              rngs={"dropout": jax.random.PRNGKey(step)})
                return cross_entropy_loss(logits, batch)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))

        final_loss = np.mean(losses[-20:])
        results[name] = {"final_loss": final_loss, "chunk_sizes": chunk_sizes, "losses": losses}
        print(f"    Final loss: {final_loss:.4f}")
        print()

    print("  " + "-" * 40)
    print(f"  {'Config':>25} | {'Loss':>8}")
    print("  " + "-" * 40)
    for name, r in sorted(results.items(), key=lambda x: x[1]["final_loss"]):
        print(f"  {name:>25} | {r['final_loss']:>8.4f}")
    print()

    return results


# ─────────────────────────────────────────────────────────────
# Experiment 3: Parameter Count vs Performance (Scaling Laws)
# ─────────────────────────────────────────────────────────────

def experiment_scaling_laws(
    dims: list = None,
    seq_len: int = 64,
    num_train_steps: int = 200,
    batch_size: int = 4,
    vocab_size: int = 50,
    seed: int = 42,
):
    """Does HOPE follow power-law scaling like Transformers?

    Transformers follow the Chinchilla scaling law: loss ~ N^(-alpha)
    where N is the parameter count and alpha ≈ 0.076.

    Does HOPE follow a similar law? Does the exponent change with the
    number of CMS levels? If so, that would mean Nested Learning changes
    the fundamental efficiency of scale.

    We train models at different sizes (varying dim) and check if
    log(loss) vs log(params) is a straight line.
    """
    print("=" * 60)
    print("EXPERIMENT 3: Scaling Laws — Loss vs Parameter Count")
    print("=" * 60)
    print()

    if dims is None:
        dims = [32, 48, 64, 96, 128]

    rng = jax.random.PRNGKey(seed)
    data_key, rng = jax.random.split(rng)
    train_data = jax.random.randint(data_key, (2000,), 0, vocab_size)

    results = {}

    for dim in dims:
        print(f"  --- dim={dim} ---")
        model_key, rng = jax.random.split(rng)

        model = HOPE(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=2,
            num_heads=4,
            ffn_dim=dim * 2,
            max_seq_len=seq_len,
            memory_dim=max(16, dim // 2),
            cms_num_levels=3,
            cms_chunk_sizes=[1, 8, 64],
        )

        dummy_batch = jax.random.randint(model_key, (1, seq_len), 0, vocab_size)
        params = model.init(model_key, dummy_batch, training=False)
        num_params = sum(p.size for p in jax.tree.leaves(params))

        import optax
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        losses = []
        for step in range(num_train_steps):
            batch_key, rng = jax.random.split(rng)
            starts = jax.random.randint(batch_key, (batch_size,), 0, len(train_data) - seq_len)
            batch = jnp.stack([train_data[s:s + seq_len] for s in starts])

            def loss_fn(p):
                logits, _, _, _ = model.apply(p, batch, step=step, training=True,
                                              rngs={"dropout": jax.random.PRNGKey(step)})
                return cross_entropy_loss(logits, batch)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))

        final_loss = np.mean(losses[-20:])
        results[dim] = {
            "final_loss": final_loss,
            "num_params": num_params,
            "log_params": np.log(num_params),
            "log_loss": np.log(final_loss) if final_loss > 0 else 0,
        }
        print(f"    Params: {num_params:,}, Loss: {final_loss:.4f}")
        print()

    # Check for power-law: fit log(loss) = -alpha * log(params) + c
    log_params = np.array([r["log_params"] for r in results.values()])
    log_losses = np.array([r["log_loss"] for r in results.values()])

    if len(log_params) > 1 and np.all(np.isfinite(log_losses)):
        # Simple linear regression
        coeffs = np.polyfit(log_params, log_losses, 1)
        alpha = -coeffs[0]  # Scaling exponent

        print(f"  Scaling law fit: loss ~ N^(-{alpha:.4f})")
        print(f"  (Chinchilla Transformers: alpha ≈ 0.076)")
        print(f"  (alpha > 0.076 means HOPE scales better than Transformers)")
        print(f"  (alpha < 0.076 means HOPE scales worse)")
        print()
        print("  NOTE: This is on a TINY synthetic task. Real scaling laws")
        print("  require much larger models and real data. This is just")
        print("  a framework for running the experiment at scale.")
    else:
        print("  Could not fit scaling law (need more data points or finite losses)")

    print()
    return results


# ─────────────────────────────────────────────────────────────
# Experiment 4: Ablation — Which Component Matters Most?
# ─────────────────────────────────────────────────────────────

def experiment_ablation(
    dim: int = 64,
    seq_len: int = 64,
    num_train_steps: int = 200,
    batch_size: int = 4,
    vocab_size: int = 50,
    seed: int = 42,
):
    """Which nested optimization level contributes the most?

    We disable each component one at a time and measure the impact:
    - No self-modification (mod_lr=0)
    - No associative memory (memory_lr=0)
    - No CMS (1 level only)
    - Full HOPE (all components)

    This tells us which levels of the nested optimization are most important.
    """
    print("=" * 60)
    print("EXPERIMENT 4: Ablation — Which Component Matters Most?")
    print("=" * 60)
    print()

    rng = jax.random.PRNGKey(seed)
    data_key, rng = jax.random.split(rng)
    train_data = jax.random.randint(data_key, (1000,), 0, vocab_size)

    configs = {
        "Full HOPE":         {"mod_lr": 0.01, "memory_lr": 0.01, "cms_num_levels": 3},
        "No self-mod":       {"mod_lr": 0.0,  "memory_lr": 0.01, "cms_num_levels": 3},
        "No assoc memory":   {"mod_lr": 0.01, "memory_lr": 0.0,  "cms_num_levels": 3},
        "No CMS (1 level)":  {"mod_lr": 0.01, "memory_lr": 0.01, "cms_num_levels": 1},
        "Only backprop":     {"mod_lr": 0.0,  "memory_lr": 0.0,  "cms_num_levels": 1},
    }

    results = {}

    for name, cfg in configs.items():
        print(f"  --- {name} ---")
        model_key, rng = jax.random.split(rng)

        model = HOPE(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=2,
            num_heads=4,
            ffn_dim=dim * 2,
            max_seq_len=seq_len,
            memory_dim=32,
            cms_num_levels=cfg["cms_num_levels"],
            mod_lr=cfg["mod_lr"],
            memory_lr=cfg["memory_lr"],
        )

        dummy_batch = jax.random.randint(model_key, (1, seq_len), 0, vocab_size)
        params = model.init(model_key, dummy_batch, training=False)

        import optax
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        losses = []
        for step in range(num_train_steps):
            batch_key, rng = jax.random.split(rng)
            starts = jax.random.randint(batch_key, (batch_size,), 0, len(train_data) - seq_len)
            batch = jnp.stack([train_data[s:s + seq_len] for s in starts])

            def loss_fn(p):
                logits, _, _, _ = model.apply(p, batch, step=step, training=True,
                                              rngs={"dropout": jax.random.PRNGKey(step)})
                return cross_entropy_loss(logits, batch)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))

        final_loss = np.mean(losses[-20:])
        results[name] = {"final_loss": final_loss, "losses": losses}
        print(f"    Final loss: {final_loss:.4f}")
        print()

    # Summary sorted by loss
    print("  " + "-" * 40)
    print(f"  {'Config':>20} | {'Loss':>8} | {'Rank':>4}")
    print("  " + "-" * 40)
    for rank, (name, r) in enumerate(
        sorted(results.items(), key=lambda x: x[1]["final_loss"]), 1
    ):
        print(f"  {name:>20} | {r['final_loss']:>8.4f} | {rank:>4}")
    print()

    return results


# ─────────────────────────────────────────────────────────────
# Plotting utility
# ─────────────────────────────────────────────────────────────

def plot_results(depth_results, chunk_results, scaling_results, ablation_results):
    """Generate plots for all experiments.

    Saves to results/ directory.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots.")
        print("Install with: pip install matplotlib")
        return

    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: CMS Depth vs Loss
    ax = axes[0, 0]
    levels = list(depth_results.keys())
    losses = [r["final_loss"] for r in depth_results.values()]
    ax.plot(levels, losses, "o-", color="tab:blue", linewidth=2, markersize=8)
    ax.set_xlabel("Number of CMS Levels")
    ax.set_ylabel("Final Loss")
    ax.set_title("How Many Memory Levels Do We Need?")
    ax.grid(True, alpha=0.3)

    # Plot 2: Chunk Size Ratios
    ax = axes[0, 1]
    names = list(chunk_results.keys())
    losses = [r["final_loss"] for r in chunk_results.values()]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(range(len(names)), losses, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.strip() for n in names], fontsize=9)
    ax.set_xlabel("Final Loss")
    ax.set_title("Which Update Frequency Ratio Works Best?")
    ax.invert_yaxis()

    # Plot 3: Scaling Laws (log-log)
    ax = axes[1, 0]
    params = [r["num_params"] for r in scaling_results.values()]
    losses = [r["final_loss"] for r in scaling_results.values()]
    ax.loglog(params, losses, "s-", color="tab:red", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Final Loss")
    ax.set_title("Scaling Law: Does HOPE Follow a Power Law?")
    ax.grid(True, alpha=0.3, which="both")

    # Plot 4: Ablation
    ax = axes[1, 1]
    names = list(ablation_results.keys())
    losses = [r["final_loss"] for r in ablation_results.values()]
    sorted_idx = np.argsort(losses)
    colors = ["tab:green" if "Full" in names[i] else "tab:gray" for i in sorted_idx]
    ax.barh(
        range(len(names)),
        [losses[i] for i in sorted_idx],
        color=colors,
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel("Final Loss")
    ax.set_title("Which Component Matters Most?")
    ax.invert_yaxis()

    plt.suptitle(
        "Nested Learning: Scaling Exploration\n"
        "(Open Problem: What is the optimal depth and frequency structure?)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("results/scaling_exploration.png", dpi=150, bbox_inches="tight")
    print("Plots saved to results/scaling_exploration.png")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("Nested Learning — Scaling Laws Exploration")
    print("=" * 60)
    print()
    print("This script explores the OPEN PROBLEM in Nested Learning:")
    print("how many optimization levels should a model have, and how")
    print("should their update frequencies be spaced?")
    print()
    print("The paper shows that adding levels helps, but doesn't provide")
    print("a theory for the optimal configuration. These experiments")
    print("give you the tools to investigate empirically.")
    print()
    print("NOTE: These are SMALL experiments on synthetic data.")
    print("Real scaling laws require larger models and real datasets.")
    print("This is a starting point for exploration, not a definitive answer.")
    print()

    depth_results = experiment_cms_depth()
    chunk_results = experiment_chunk_ratios()
    scaling_results = experiment_scaling_laws()
    ablation_results = experiment_ablation()

    plot_results(depth_results, chunk_results, scaling_results, ablation_results)

    print()
    print("=" * 60)
    print("OPEN QUESTIONS FOR FURTHER RESEARCH")
    print("=" * 60)
    print()
    print("1. DEPTH SCALING LAW: Is there a formula like")
    print("   optimal_levels = f(model_size, sequence_length, task_complexity)?")
    print()
    print("2. FREQUENCY SPACING: Should chunk sizes be exponential (4^i),")
    print("   linear (10*i), or something task-dependent?")
    print()
    print("3. INTERACTION EFFECTS: Does self-modification become more or less")
    print("   important as we add more CMS levels? Are they complementary or")
    print("   redundant optimization mechanisms?")
    print()
    print("4. COMPUTE-OPTIMAL DEPTH: Given a fixed compute budget, is it")
    print("   better to add more CMS levels or make the model wider/deeper")
    print("   in the traditional sense?")
    print()
    print("5. TASK DEPENDENCE: Do tasks with longer temporal dependencies")
    print("   (e.g., book-level language modeling) need more CMS levels")
    print("   than tasks with shorter dependencies (e.g., sentiment analysis)?")
    print()
    print("6. EMERGENT PROPERTIES: At what depth/scale do qualitatively new")
    print("   capabilities emerge? (Similar to how in-context learning")
    print("   emerges in large Transformers)")
    print()
    print("To explore these, modify the experiment parameters above and")
    print("run on larger models with real data (e.g., WikiText-103).")
    print("=" * 60)
