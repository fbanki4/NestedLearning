"""
Quick Scaling Experiment — Faster Version for CPU
===================================================

The full scaling_exploration.py takes a while on CPU because JAX
recompiles for each different model configuration. This script
uses a single compilation where possible and smaller models.

Run with:
    python examples/quick_experiment.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nested_learning.continuum_memory import ContinuumMemorySystem, FullyNestedCMS
from nested_learning.associative_memory import AssociativeMemory
from nested_learning.self_modifying import SelfModifyingLinear
import optax


def make_synthetic_task(rng, num_tokens=500, vocab_size=30, pattern_length=8):
    """Create a synthetic sequence with multi-scale patterns.

    This generates data that has patterns at MULTIPLE time scales,
    which is exactly the kind of data where CMS should shine.

    - Short patterns: repeating every ~4 tokens  (captured by fast CMS levels)
    - Medium patterns: repeating every ~16 tokens (captured by medium levels)
    - Long patterns: repeating every ~64 tokens   (captured by slow levels)
    """
    tokens = []
    for i in range(num_tokens):
        # Mix of patterns at different frequencies
        short = (i * 7 + 3) % vocab_size           # fast pattern
        medium = ((i // 4) * 13 + 5) % vocab_size  # medium pattern
        long_ = ((i // 16) * 11 + 7) % vocab_size  # slow pattern
        # Combine: token depends on all three scales
        token = (short + medium + long_) % vocab_size
        tokens.append(token)
    return jnp.array(tokens)


def train_simple_model(model, params, train_data, num_steps=80, batch_size=4,
                       seq_len=32, lr=1e-3, rng_seed=0):
    """Train a Flax model and return losses. Works for any model with __call__(x)."""
    rng = jax.random.PRNGKey(rng_seed)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    num_tokens = len(train_data)

    losses = []
    for step in range(num_steps):
        rng, batch_key = jax.random.split(rng)
        starts = jax.random.randint(batch_key, (batch_size,), 0, num_tokens - seq_len)
        batch = jnp.stack([train_data[s:s + seq_len] for s in starts])

        # Simple next-token prediction loss
        def loss_fn(p):
            # We'll use a thin wrapper — just embed, apply model, project to vocab
            return _next_token_loss(model, p, batch, step)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        losses.append(float(loss))

    return losses, params


def _next_token_loss(model, params, batch, step):
    """Wrapper that handles different model signatures."""
    import flax.linen as nn

    # For CMS models, we pass step for multi-frequency scheduling
    try:
        out, _ = model.apply(params, batch, step=step, training=True)
    except TypeError:
        out = model.apply(params, batch)
    # Shift for next-token prediction
    preds = out[:, :-1, :]
    targets = batch[:, 1:]
    vocab_size = preds.shape[-1]
    one_hot = jax.nn.one_hot(targets, vocab_size)
    log_probs = jax.nn.log_softmax(preds, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


# ═══════════════════════════════════════════════════════════════
# Experiment: CMS Depth
# ═══════════════════════════════════════════════════════════════

import flax.linen as nn


class SimpleSequenceModel(nn.Module):
    """Minimal sequence model with configurable CMS depth."""
    vocab_size: int
    dim: int
    cms_levels: int
    cms_chunk_sizes: list

    @nn.compact
    def __call__(self, x, step=0, training=True):
        batch, seq = x.shape
        # Embed
        h = nn.Embed(self.vocab_size, self.dim, name="embed")(x)
        # CMS
        if self.cms_levels > 0:
            h, level_outs = ContinuumMemorySystem(
                dim=self.dim,
                hidden_dim=self.dim * 2,
                num_levels=self.cms_levels,
                chunk_sizes=self.cms_chunk_sizes,
                name="cms",
            )(h, step=step, training=training)
        # Project to vocab
        logits = nn.Dense(self.vocab_size, name="head")(h)
        return logits, level_outs if self.cms_levels > 0 else []


def experiment_depth():
    """Experiment 1: How many CMS levels give the best loss?"""
    print("=" * 60)
    print("EXPERIMENT 1: CMS Depth vs Performance")
    print("=" * 60)
    print()

    rng = jax.random.PRNGKey(42)
    vocab = 30
    dim = 48
    num_steps = 80
    train_data = make_synthetic_task(rng, num_tokens=500, vocab_size=vocab)

    results = {}
    for n_levels in [1, 2, 3, 4, 5]:
        chunk_sizes = [4 ** i for i in range(n_levels)]
        model = SimpleSequenceModel(
            vocab_size=vocab, dim=dim,
            cms_levels=n_levels, cms_chunk_sizes=chunk_sizes,
        )

        key = jax.random.PRNGKey(n_levels)
        dummy = jax.random.randint(key, (1, 32), 0, vocab)
        params = model.init(key, dummy, step=0)
        n_params = sum(p.size for p in jax.tree.leaves(params))

        t0 = time.time()
        losses, _ = train_simple_model(model, params, train_data, num_steps=num_steps)
        elapsed = time.time() - t0
        final = np.mean(losses[-15:])

        results[n_levels] = {"loss": final, "params": n_params, "time": elapsed}
        print(f"  {n_levels} level(s) | {n_params:>7,} params | loss={final:.4f} | {elapsed:.1f}s")

    print()
    best = min(results, key=lambda k: results[k]["loss"])
    print(f"  Best: {best} level(s) with loss {results[best]['loss']:.4f}")
    print()
    return results


# ═══════════════════════════════════════════════════════════════
# Experiment: Chunk Size Ratios
# ═══════════════════════════════════════════════════════════════

def experiment_chunk_ratios():
    """Experiment 2: Which spacing between update frequencies works best?"""
    print("=" * 60)
    print("EXPERIMENT 2: Chunk Size Ratios (3 CMS levels)")
    print("=" * 60)
    print()

    rng = jax.random.PRNGKey(42)
    vocab = 30
    dim = 48
    num_steps = 80
    train_data = make_synthetic_task(rng, num_tokens=500, vocab_size=vocab)

    configs = {
        "tight  [1,2,4]":     [1, 2, 4],
        "medium [1,4,16]":    [1, 4, 16],
        "wide   [1,8,64]":    [1, 8, 64],
        "linear [1,5,10]":    [1, 5, 10],
        "extreme[1,16,256]":  [1, 16, 256],
    }

    results = {}
    for name, chunks in configs.items():
        model = SimpleSequenceModel(
            vocab_size=vocab, dim=dim, cms_levels=3, cms_chunk_sizes=chunks,
        )
        key = jax.random.PRNGKey(hash(name) % 2**31)
        dummy = jax.random.randint(key, (1, 32), 0, vocab)
        params = model.init(key, dummy, step=0)

        losses, _ = train_simple_model(model, params, train_data, num_steps=num_steps)
        final = np.mean(losses[-15:])
        results[name] = {"loss": final, "chunks": chunks}
        print(f"  {name:>25} | loss={final:.4f}")

    print()
    best = min(results, key=lambda k: results[k]["loss"])
    print(f"  Best: {best.strip()} with loss {results[best]['loss']:.4f}")
    print()
    return results


# ═══════════════════════════════════════════════════════════════
# Experiment: Ablation
# ═══════════════════════════════════════════════════════════════

class AblationModel(nn.Module):
    """Model with toggleable components for ablation study."""
    vocab_size: int
    dim: int
    use_cms: bool = True
    cms_levels: int = 3
    use_self_mod: bool = True
    use_memory: bool = True

    @nn.compact
    def __call__(self, x, step=0, training=True):
        batch, seq = x.shape
        h = nn.Embed(self.vocab_size, self.dim, name="embed")(x)

        # Optional self-modifying layer
        if self.use_self_mod:
            h_flat = h.reshape(-1, self.dim)
            h_mod, _ = SelfModifyingLinear(
                features=self.dim, mod_lr=0.01, name="self_mod"
            )(h_flat)
            h = h + h_mod.reshape(batch, seq, self.dim)

        # Optional associative memory
        if self.use_memory:
            mem_out, _ = AssociativeMemory(
                dim=self.dim, memory_dim=32, num_heads=2,
                lr=0.01, name="memory"
            )(h)
            gate = jax.nn.sigmoid(nn.Dense(self.dim, name="mem_gate")(h))
            h = h + gate * mem_out

        # Optional CMS
        if self.use_cms:
            chunks = [4 ** i for i in range(self.cms_levels)]
            h_cms, _ = ContinuumMemorySystem(
                dim=self.dim, hidden_dim=self.dim * 2,
                num_levels=self.cms_levels, chunk_sizes=chunks,
                name="cms",
            )(h, step=step, training=training)
            h = h + h_cms

        # FFN
        h2 = nn.Dense(self.dim * 2, name="ffn_up")(h)
        h2 = nn.gelu(h2)
        h2 = nn.Dense(self.dim, name="ffn_down")(h2)
        h = h + h2

        logits = nn.Dense(self.vocab_size, name="head")(h)
        return logits, []


def experiment_ablation():
    """Experiment 3: Remove each component — which matters most?"""
    print("=" * 60)
    print("EXPERIMENT 3: Ablation — Which Component Matters?")
    print("=" * 60)
    print()

    rng = jax.random.PRNGKey(42)
    vocab = 30
    dim = 48
    num_steps = 80
    train_data = make_synthetic_task(rng, num_tokens=500, vocab_size=vocab)

    configs = {
        "Full model":       {"use_cms": True,  "use_self_mod": True,  "use_memory": True},
        "No self-mod":      {"use_cms": True,  "use_self_mod": False, "use_memory": True},
        "No memory":        {"use_cms": True,  "use_self_mod": True,  "use_memory": False},
        "No CMS":           {"use_cms": False, "use_self_mod": True,  "use_memory": True},
        "Baseline (none)":  {"use_cms": False, "use_self_mod": False, "use_memory": False},
    }

    results = {}
    for name, cfg in configs.items():
        model = AblationModel(vocab_size=vocab, dim=dim, **cfg)
        key = jax.random.PRNGKey(hash(name) % 2**31)
        dummy = jax.random.randint(key, (1, 32), 0, vocab)
        params = model.init(key, dummy, step=0)
        n_params = sum(p.size for p in jax.tree.leaves(params))

        losses, _ = train_simple_model(model, params, train_data, num_steps=num_steps)
        final = np.mean(losses[-15:])
        results[name] = {"loss": final, "params": n_params}
        print(f"  {name:>18} | {n_params:>7,} params | loss={final:.4f}")

    print()

    # Compute contribution of each component
    full = results["Full model"]["loss"]
    base = results["Baseline (none)"]["loss"]
    total_improvement = base - full

    if total_improvement > 0:
        print("  Component contributions (lower loss = better):")
        for removed, name in [("No self-mod", "Self-mod"), ("No memory", "Memory"), ("No CMS", "CMS")]:
            cost = results[removed]["loss"] - full
            pct = (cost / total_improvement) * 100 if total_improvement > 0 else 0
            print(f"    {name:>10}: removing it costs +{cost:.4f} loss ({pct:.0f}% of total improvement)")
    print()
    return results


# ═══════════════════════════════════════════════════════════════
# Experiment: Scaling
# ═══════════════════════════════════════════════════════════════

def experiment_scaling():
    """Experiment 4: Does loss vs params follow a power law?"""
    print("=" * 60)
    print("EXPERIMENT 4: Scaling Law — Loss vs Parameters")
    print("=" * 60)
    print()

    rng = jax.random.PRNGKey(42)
    vocab = 30
    num_steps = 80
    train_data = make_synthetic_task(rng, num_tokens=500, vocab_size=vocab)

    results = {}
    for dim in [24, 32, 48, 64, 96]:
        model = SimpleSequenceModel(
            vocab_size=vocab, dim=dim, cms_levels=3, cms_chunk_sizes=[1, 4, 16],
        )
        key = jax.random.PRNGKey(dim)
        dummy = jax.random.randint(key, (1, 32), 0, vocab)
        params = model.init(key, dummy, step=0)
        n_params = sum(p.size for p in jax.tree.leaves(params))

        losses, _ = train_simple_model(model, params, train_data, num_steps=num_steps)
        final = np.mean(losses[-15:])
        results[dim] = {"loss": final, "params": n_params}
        print(f"  dim={dim:>3} | {n_params:>7,} params | loss={final:.4f}")

    # Fit power law: loss = a * N^(-alpha)
    log_n = np.array([np.log(r["params"]) for r in results.values()])
    log_l = np.array([np.log(r["loss"]) for r in results.values()])

    if np.all(np.isfinite(log_l)):
        coeffs = np.polyfit(log_n, log_l, 1)
        alpha = -coeffs[0]
        print()
        print(f"  Fitted power law: loss ~ N^(-{alpha:.4f})")
        print(f"  (Chinchilla Transformers: alpha ~ 0.076)")
        if alpha > 0:
            print(f"  => Each 10x increase in params reduces loss by {(1 - 10**(-alpha))*100:.1f}%")
        else:
            print(f"  => Negative alpha: loss is INCREASING with params (underfitting or not enough training)")
    print()
    return results


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("Nested Learning — Quick Scaling Experiments")
    print("=" * 60)
    print()
    print("Running 4 experiments on synthetic multi-scale data.")
    print("Each token depends on patterns at 3 time scales,")
    print("which is exactly where multi-frequency memory should help.")
    print()

    t_total = time.time()

    r1 = experiment_depth()
    r2 = experiment_chunk_ratios()
    r3 = experiment_ablation()
    r4 = experiment_scaling()

    elapsed = time.time() - t_total

    print("=" * 60)
    print(f"All experiments complete in {elapsed:.0f}s")
    print("=" * 60)
    print()
    print("INTERPRETATION GUIDE:")
    print()
    print("Exp 1 (Depth): Look for where adding levels stops helping.")
    print("  That's the 'diminishing returns' point for this data.")
    print()
    print("Exp 2 (Ratios): The best ratio tells you the temporal structure")
    print("  of the data. Tight ratios = patterns at similar speeds.")
    print("  Wide ratios = patterns at very different speeds.")
    print()
    print("Exp 3 (Ablation): The component whose removal hurts most")
    print("  is the most important optimization level for this task.")
    print()
    print("Exp 4 (Scaling): A straight line on a log-log plot means")
    print("  power law scaling. The slope tells you how efficiently")
    print("  HOPE converts parameters into performance.")
    print()
    print("REMEMBER: These are tiny models on synthetic data.")
    print("The real test is at scale with natural language.")

    # Try to plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("results", exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # 1: Depth
        ax = axes[0, 0]
        levels = list(r1.keys())
        losses = [r1[k]["loss"] for k in levels]
        ax.plot(levels, losses, "o-", color="#2196F3", linewidth=2, markersize=8)
        ax.set_xlabel("Number of CMS Levels")
        ax.set_ylabel("Final Loss")
        ax.set_title("Exp 1: How Many Levels?")
        ax.grid(True, alpha=0.3)

        # 2: Chunk ratios
        ax = axes[0, 1]
        names = [n.strip() for n in r2.keys()]
        losses = [r2[k]["loss"] for k in r2]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        ax.barh(range(len(names)), losses, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Final Loss")
        ax.set_title("Exp 2: Which Frequency Spacing?")
        ax.invert_yaxis()

        # 3: Ablation
        ax = axes[1, 0]
        names = list(r3.keys())
        losses = [r3[k]["loss"] for k in names]
        idx = np.argsort(losses)
        c = ["#4CAF50" if "Full" in names[i] else "#9E9E9E" for i in idx]
        ax.barh(range(len(names)), [losses[i] for i in idx], color=c)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([names[i] for i in idx], fontsize=9)
        ax.set_xlabel("Final Loss")
        ax.set_title("Exp 3: Which Component Matters?")
        ax.invert_yaxis()

        # 4: Scaling
        ax = axes[1, 1]
        params = [r4[k]["params"] for k in r4]
        losses = [r4[k]["loss"] for k in r4]
        ax.loglog(params, losses, "s-", color="#F44336", linewidth=2, markersize=8)
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Loss")
        ax.set_title("Exp 4: Scaling Law (log-log)")
        ax.grid(True, alpha=0.3, which="both")

        plt.suptitle("Nested Learning: Open Problem Exploration", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("results/quick_experiments.png", dpi=150, bbox_inches="tight")
        print(f"\nPlots saved to results/quick_experiments.png")
    except Exception as e:
        print(f"\nCould not generate plots: {e}")
