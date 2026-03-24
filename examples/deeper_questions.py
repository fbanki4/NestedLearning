"""
Deeper Open Questions — Things Nobody Has Tested Yet
======================================================

These go beyond "how many levels" into questions about whether the
mechanisms actually work the way the paper claims.

Run with:
    python examples/deeper_questions.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================================================================
# Q1: Does adaptation ACTUALLY help, or just change things?
# ================================================================
#
# The demo shows that HOPE's output changes between pass 1 and pass 2.
# But changing output != improving output. Does the carried-over state
# (memory + fast weights) actually make predictions BETTER?
#
# This is the most fundamental question about the whole framework.
# If carried state doesn't help, the inner optimization is just noise.

def question_does_adaptation_help():
    """Compare: model with carried state vs. fresh state on the same data.

    If adaptation helps, carried state should give lower loss on the
    NEXT chunk of the same sequence.
    """
    from nested_learning.hope import HOPE
    from nested_learning.train import cross_entropy_loss

    print("=" * 60)
    print("Q1: Does adaptation actually help prediction?")
    print("=" * 60)
    print()
    print("If carried memory/fast_weights help, the model should predict")
    print("chunk 2 better when it has already seen chunk 1.")
    print()

    rng = jax.random.PRNGKey(42)
    vocab = 30
    dim = 64
    seq_len = 32

    model = HOPE(
        vocab_size=vocab, dim=dim, num_layers=2, num_heads=4,
        ffn_dim=128, max_seq_len=seq_len, memory_dim=32,
        cms_num_levels=2, cms_chunk_sizes=[1, 4],
        mod_lr=0.01, memory_lr=0.01,
    )

    # Create a sequence with repeating structure
    # Chunk 2 is related to chunk 1, so memory of chunk 1 should help
    pattern = jnp.array([1, 5, 3, 7, 2, 8, 4, 6] * 4)  # repeating pattern
    chunk1 = pattern[:seq_len][None, :]  # (1, seq_len)
    chunk2 = pattern[:seq_len][None, :]  # same pattern (memory should help)

    # Also test with a DIFFERENT chunk2 (memory shouldn't help as much)
    rng_key = jax.random.PRNGKey(99)
    chunk2_diff = jax.random.randint(rng_key, (1, seq_len), 0, vocab)

    params = model.init(rng, chunk1, training=False)

    # --- Scenario A: Fresh state for chunk 2 ---
    logits_fresh, _, _, _ = model.apply(params, chunk2, step=0, training=False)
    loss_fresh = float(cross_entropy_loss(logits_fresh, chunk2))

    # --- Scenario B: Carry state from chunk 1 to chunk 2 ---
    _, mem_states, fast_weights, _ = model.apply(params, chunk1, step=0, training=False)
    logits_warm, _, _, _ = model.apply(
        params, chunk2,
        memory_states=mem_states, fast_weights_list=fast_weights,
        step=1, training=False,
    )
    loss_warm = float(cross_entropy_loss(logits_warm, chunk2))

    # --- Scenario C: Carry state but chunk 2 is DIFFERENT ---
    logits_warm_diff, _, _, _ = model.apply(
        params, chunk2_diff,
        memory_states=mem_states, fast_weights_list=fast_weights,
        step=1, training=False,
    )
    loss_warm_diff = float(cross_entropy_loss(logits_warm_diff, chunk2_diff))
    logits_fresh_diff, _, _, _ = model.apply(params, chunk2_diff, step=0, training=False)
    loss_fresh_diff = float(cross_entropy_loss(logits_fresh_diff, chunk2_diff))

    print(f"  SAME pattern (chunk1 == chunk2):")
    print(f"    Fresh state loss: {loss_fresh:.4f}")
    print(f"    Warm state loss:  {loss_warm:.4f}")
    print(f"    Adaptation {'HELPED' if loss_warm < loss_fresh else 'HURT'} "
          f"(delta: {loss_warm - loss_fresh:+.4f})")
    print()
    print(f"  DIFFERENT pattern (chunk2 is random):")
    print(f"    Fresh state loss: {loss_fresh_diff:.4f}")
    print(f"    Warm state loss:  {loss_warm_diff:.4f}")
    print(f"    Adaptation {'HELPED' if loss_warm_diff < loss_fresh_diff else 'HURT'} "
          f"(delta: {loss_warm_diff - loss_fresh_diff:+.4f})")
    print()
    print("  INTERPRETATION:")
    print("  - If adaptation helps on SAME pattern: memory is learning useful info")
    print("  - If adaptation hurts on DIFFERENT pattern: memory is overfitting")
    print("  - If adaptation helps on BOTH: memory learned something general")
    print()

    # --- PART 2: Repeat after training ---
    print("  Now training the model for 80 steps, then re-testing...")
    print()
    import optax
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    # Create training data from the repeating pattern
    train_data = jnp.tile(pattern[:seq_len], 10)  # longer sequence
    train_rng = jax.random.PRNGKey(0)

    def loss_fn(p, batch):
        logits, _, _, _ = model.apply(p, batch, step=0, training=True)
        return float('inf') if logits is None else cross_entropy_loss(logits, batch)

    for step in range(80):
        train_rng, batch_key, dropout_key = jax.random.split(train_rng, 3)
        starts = jax.random.randint(batch_key, (2,), 0, len(train_data) - seq_len)
        batch = jnp.stack([train_data[s:s + seq_len] for s in starts])
        loss, grads = jax.value_and_grad(
            lambda p, b: cross_entropy_loss(
                model.apply(p, b, step=step, training=True,
                            rngs={"dropout": dropout_key})[0], b
            )
        )(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    # Re-test with trained model
    logits_fresh2, _, _, _ = model.apply(params, chunk2, step=0, training=False)
    loss_fresh2 = float(cross_entropy_loss(logits_fresh2, chunk2))

    _, mem_states2, fast_weights2, _ = model.apply(params, chunk1, step=0, training=False)
    logits_warm2, _, _, _ = model.apply(
        params, chunk2,
        memory_states=mem_states2, fast_weights_list=fast_weights2,
        step=1, training=False,
    )
    loss_warm2 = float(cross_entropy_loss(logits_warm2, chunk2))

    print(f"  AFTER TRAINING (80 steps):")
    print(f"    Fresh state loss: {loss_fresh2:.4f}")
    print(f"    Warm state loss:  {loss_warm2:.4f}")
    print(f"    Adaptation {'HELPED' if loss_warm2 < loss_fresh2 else 'HURT'} "
          f"(delta: {loss_warm2 - loss_fresh2:+.4f})")
    print()

    return {
        "same_fresh": loss_fresh, "same_warm": loss_warm,
        "diff_fresh": loss_fresh_diff, "diff_warm": loss_warm_diff,
        "trained_fresh": loss_fresh2, "trained_warm": loss_warm2,
    }


# ================================================================
# Q2: Is surprise actually correlated with information content?
# ================================================================
#
# The paper says "surprising inputs get written more aggressively."
# But is "surprise" (gradient magnitude) actually measuring novelty?
# Or does random noise also produce high surprise?
#
# If noise produces high surprise, the memory will waste capacity
# storing garbage. A good surprise signal should be HIGH for
# structured-but-novel data and LOW for both familiar data AND noise.

def question_surprise_calibration():
    """Feed the memory three types of input and measure surprise for each:
    1. Familiar: the same pattern repeated
    2. Novel: a new structured pattern the memory hasn't seen
    3. Noise: random garbage

    A well-calibrated surprise signal should rank: novel > noise > familiar
    (In practice, noise might score higher than novel, which would be a problem)
    """
    from nested_learning.associative_memory import AssociativeMemory

    print("=" * 60)
    print("Q2: Is the surprise signal well-calibrated?")
    print("=" * 60)
    print()
    print("Good surprise: high for novel info, low for familiar AND noise.")
    print("Bad surprise: high for noise (wastes memory on garbage).")
    print()

    rng = jax.random.PRNGKey(42)
    dim = 32
    batch = 1
    seq_len = 16

    memory = AssociativeMemory(dim=dim, memory_dim=32, num_heads=2, lr=0.05, momentum=0.9)

    # Create three types of input
    # 1. Familiar pattern (we'll prime the memory with this)
    familiar = jnp.sin(jnp.arange(seq_len * dim).reshape(1, seq_len, dim) * 0.1)

    # 2. Novel structured pattern (different frequency, but structured)
    novel = jnp.cos(jnp.arange(seq_len * dim).reshape(1, seq_len, dim) * 0.3)

    # 3. Pure noise
    noise = jax.random.normal(jax.random.PRNGKey(99), (1, seq_len, dim))

    # Initialize and prime memory with the familiar pattern (3 passes)
    params = memory.init(rng, familiar)
    _, mem_state = memory.apply(params, familiar)
    _, mem_state = memory.apply(params, familiar, memory_state=mem_state)
    _, mem_state = memory.apply(params, familiar, memory_state=mem_state)

    # Now measure surprise for each type
    def measure_surprise(x, mem_state):
        """Run one pass and measure how much the memory changed."""
        _, new_mem_state = memory.apply(params, x, memory_state=mem_state)
        # Surprise = how much the memory weights changed
        mem_delta = float(jnp.linalg.norm(new_mem_state[0] - mem_state[0]))
        surprise_norm = float(jnp.linalg.norm(new_mem_state[1]))
        return mem_delta, surprise_norm

    fam_delta, fam_surprise = measure_surprise(familiar, mem_state)
    nov_delta, nov_surprise = measure_surprise(novel, mem_state)
    noise_delta, noise_surprise = measure_surprise(noise, mem_state)

    print(f"  {'Input Type':>12} | {'Memory Change':>13} | {'Surprise Norm':>13}")
    print(f"  {'-'*12}-+-{'-'*13}-+-{'-'*13}")
    print(f"  {'Familiar':>12} | {fam_delta:>13.4f} | {fam_surprise:>13.4f}")
    print(f"  {'Novel':>12} | {nov_delta:>13.4f} | {nov_surprise:>13.4f}")
    print(f"  {'Noise':>12} | {noise_delta:>13.4f} | {noise_surprise:>13.4f}")
    print()

    # Rank them
    scores = {"Familiar": fam_delta, "Novel": nov_delta, "Noise": noise_delta}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"  Ranking (highest surprise first): {' > '.join(n for n, _ in ranked)}")
    print()

    ideal = ranked[0][0] == "Novel" and ranked[-1][0] == "Familiar"
    if ideal:
        print("  GOOD: Novel > Noise > Familiar (surprise is well-calibrated!)")
    elif ranked[0][0] == "Noise":
        print("  CONCERN: Noise triggers most surprise (memory may store garbage).")
        print("  Testing the fix: normalize_surprise=True...")
    else:
        print(f"  MIXED: The ranking isn't ideal. Investigate why.")
    print()

    # --- PART 2: Test with normalized surprise ---
    print("  WITH normalize_surprise=True (divides gradient by input norm):")
    print()
    memory_fixed = AssociativeMemory(
        dim=dim, memory_dim=32, num_heads=2, lr=0.05, momentum=0.9,
        normalize_surprise=True,
    )
    params_fixed = memory_fixed.init(rng, familiar)
    _, mem_state_fixed = memory_fixed.apply(params_fixed, familiar)
    _, mem_state_fixed = memory_fixed.apply(params_fixed, familiar, memory_state=mem_state_fixed)
    _, mem_state_fixed = memory_fixed.apply(params_fixed, familiar, memory_state=mem_state_fixed)

    def measure_surprise_fixed(x, ms):
        _, new_ms = memory_fixed.apply(params_fixed, x, memory_state=ms)
        mem_delta = float(jnp.linalg.norm(new_ms[0] - ms[0]))
        surprise_norm = float(jnp.linalg.norm(new_ms[1]))
        return mem_delta, surprise_norm

    fam_d2, fam_s2 = measure_surprise_fixed(familiar, mem_state_fixed)
    nov_d2, nov_s2 = measure_surprise_fixed(novel, mem_state_fixed)
    noise_d2, noise_s2 = measure_surprise_fixed(noise, mem_state_fixed)

    print(f"  {'Input Type':>12} | {'Memory Change':>13} | {'Surprise Norm':>13}")
    print(f"  {'-'*12}-+-{'-'*13}-+-{'-'*13}")
    print(f"  {'Familiar':>12} | {fam_d2:>13.4f} | {fam_s2:>13.4f}")
    print(f"  {'Novel':>12} | {nov_d2:>13.4f} | {nov_s2:>13.4f}")
    print(f"  {'Noise':>12} | {noise_d2:>13.4f} | {noise_s2:>13.4f}")
    print()

    scores_fixed = {"Familiar": fam_d2, "Novel": nov_d2, "Noise": noise_d2}
    ranked_fixed = sorted(scores_fixed.items(), key=lambda x: x[1], reverse=True)
    print(f"  Fixed ranking: {' > '.join(n for n, _ in ranked_fixed)}")
    if ranked_fixed[0][0] == "Novel":
        print("  FIX WORKED: Novel now triggers the most surprise!")
    elif ranked_fixed[0][0] == "Noise":
        print("  Fix helped but noise is still highest — may need stronger normalization.")
    print()

    return scores


# ================================================================
# Q3: Catastrophic forgetting — does CMS actually prevent it?
# ================================================================
#
# This is the paper's MAIN CLAIM: multi-frequency updates should prevent
# catastrophic forgetting because slow levels preserve old knowledge
# while fast levels absorb new information.
#
# We test this directly: train on Task A, then Task B, then check
# if the model still remembers Task A.

def question_catastrophic_forgetting():
    """Train on two sequential tasks, measure retention of the first.

    Task A: predict a simple repeating pattern (1,2,3,4,5,1,2,3,4,5,...)
    Task B: predict a different pattern (5,4,3,2,1,5,4,3,2,1,...)

    After training on A then B:
    - A model that forgets: high loss on A, low loss on B
    - A model that remembers: moderate loss on both
    """
    import flax.linen as nn
    from nested_learning.continuum_memory import ContinuumMemorySystem
    import optax

    print("=" * 60)
    print("Q3: Does CMS actually prevent catastrophic forgetting?")
    print("=" * 60)
    print()
    print("Train on Task A, then Task B. Measure if Task A is forgotten.")
    print()

    rng = jax.random.PRNGKey(42)
    vocab = 10
    dim = 48
    seq_len = 32
    num_steps = 60  # per task

    # Create tasks with clear structure
    task_a = jnp.array([i % 5 for i in range(200)])     # 0,1,2,3,4,0,1,2,3,4,...
    task_b = jnp.array([4 - (i % 5) for i in range(200)])  # 4,3,2,1,0,4,3,2,1,0,...

    class TestModel(nn.Module):
        vocab_size: int
        dim: int
        use_cms: bool
        num_levels: int = 3

        @nn.compact
        def __call__(self, x, step=0, training=True):
            h = nn.Embed(self.vocab_size, self.dim, name="embed")(x)
            if self.use_cms:
                chunks = [4 ** i for i in range(self.num_levels)]
                h, _ = ContinuumMemorySystem(
                    dim=self.dim, hidden_dim=self.dim * 2,
                    num_levels=self.num_levels, chunk_sizes=chunks,
                    name="cms"
                )(h, step=step, training=training)
            else:
                # Simple FFN baseline (same param count roughly)
                for i in range(self.num_levels):
                    h2 = nn.Dense(self.dim * 2, name=f"ffn_up_{i}")(h)
                    h2 = nn.gelu(h2)
                    h2 = nn.Dense(self.dim, name=f"ffn_down_{i}")(h2)
                    h = h + h2
            return nn.Dense(self.vocab_size, name="head")(h), []

    def train_and_eval(model, task_a_data, task_b_data, num_steps_per_task):
        key = jax.random.PRNGKey(42)
        dummy = jax.random.randint(key, (1, seq_len), 0, vocab)
        params = model.init(key, dummy, step=0)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        def loss_fn(p, batch, step):
            logits, _ = model.apply(p, batch, step=step, training=True)
            shift_logits = logits[:, :-1, :]
            shift_targets = batch[:, 1:]
            one_hot = jax.nn.one_hot(shift_targets, vocab)
            log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
            return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

        rng = jax.random.PRNGKey(0)

        # Phase 1: Train on Task A
        for step in range(num_steps_per_task):
            rng, batch_key = jax.random.split(rng)
            starts = jax.random.randint(batch_key, (4,), 0, len(task_a_data) - seq_len)
            batch = jnp.stack([task_a_data[s:s + seq_len] for s in starts])
            loss, grads = jax.value_and_grad(loss_fn)(params, batch, step)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        # Measure loss on Task A after training on A
        batch_a = task_a_data[:seq_len][None, :]
        loss_a_after_a = float(loss_fn(params, batch_a, 0))

        # Phase 2: Train on Task B
        for step in range(num_steps_per_task, 2 * num_steps_per_task):
            rng, batch_key = jax.random.split(rng)
            starts = jax.random.randint(batch_key, (4,), 0, len(task_b_data) - seq_len)
            batch = jnp.stack([task_b_data[s:s + seq_len] for s in starts])
            loss, grads = jax.value_and_grad(loss_fn)(params, batch, step)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        # Measure loss on BOTH tasks after training on B
        loss_a_after_b = float(loss_fn(params, batch_a, 0))
        batch_b = task_b_data[:seq_len][None, :]
        loss_b_after_b = float(loss_fn(params, batch_b, 0))

        return loss_a_after_a, loss_a_after_b, loss_b_after_b

    # Test with CMS
    model_cms = TestModel(vocab_size=vocab, dim=dim, use_cms=True, num_levels=3)
    cms_a1, cms_a2, cms_b = train_and_eval(model_cms, task_a, task_b, num_steps)

    # Test without CMS (baseline)
    model_flat = TestModel(vocab_size=vocab, dim=dim, use_cms=False, num_levels=3)
    flat_a1, flat_a2, flat_b = train_and_eval(model_flat, task_a, task_b, num_steps)

    print(f"  {'':>20} | {'Task A':>12} | {'Task A':>12} | {'Task B':>12}")
    print(f"  {'Model':>20} | {'(after A)':>12} | {'(after A+B)':>12} | {'(after A+B)':>12}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'With CMS':>20} | {cms_a1:>12.4f} | {cms_a2:>12.4f} | {cms_b:>12.4f}")
    print(f"  {'Without CMS':>20} | {flat_a1:>12.4f} | {flat_a2:>12.4f} | {flat_b:>12.4f}")
    print()

    cms_forgetting = cms_a2 - cms_a1
    flat_forgetting = flat_a2 - flat_a1
    print(f"  Forgetting (Task A loss increase after learning B):")
    print(f"    With CMS:    {cms_forgetting:+.4f}")
    print(f"    Without CMS: {flat_forgetting:+.4f}")
    print()

    if cms_forgetting < flat_forgetting:
        print(f"  CMS forgot LESS ({cms_forgetting:+.4f} vs {flat_forgetting:+.4f})")
        print(f"  => Multi-frequency updates help preserve old knowledge!")
    else:
        print(f"  CMS forgot MORE or equally ({cms_forgetting:+.4f} vs {flat_forgetting:+.4f})")
        print(f"  => CMS didn't help here. The slow levels may need more training")
        print(f"     to learn to preserve information, or the tasks are too simple.")
    print()

    return {"cms": (cms_a1, cms_a2, cms_b), "flat": (flat_a1, flat_a2, flat_b)}


# ================================================================
# Q4: Do CMS levels actually specialize to different frequencies?
# ================================================================
#
# The claim: level 0 handles fast patterns, level 2 handles slow patterns.
# But do they actually learn to do this, or do all levels learn the same thing?
#
# We can test by creating data with patterns at known frequencies,
# then seeing which CMS level's output correlates with which frequency.

def question_level_specialization():
    """Create data with patterns at 2 known frequencies.
    Check if different CMS levels learn to capture different frequencies.

    We measure: correlation between each level's output and each frequency component.
    """
    import flax.linen as nn
    from nested_learning.continuum_memory import ContinuumMemorySystem

    print("=" * 60)
    print("Q4: Do CMS levels specialize to different frequencies?")
    print("=" * 60)
    print()
    print("Data has a FAST pattern (period 4) and a SLOW pattern (period 16).")
    print("Does level 0 (fast updates) capture the fast pattern,")
    print("and level 1 (slow updates) capture the slow pattern?")
    print()

    rng = jax.random.PRNGKey(42)
    dim = 32
    seq_len = 64

    # Create input with two known frequency components
    t = jnp.arange(seq_len).astype(float)
    fast_component = jnp.sin(2 * jnp.pi * t / 4)[:, None]   # period 4
    slow_component = jnp.sin(2 * jnp.pi * t / 16)[:, None]  # period 16

    # Input: embed both components into dim-dimensional space
    fast_dir = jax.random.normal(jax.random.PRNGKey(1), (1, dim))
    fast_dir = fast_dir / jnp.linalg.norm(fast_dir)
    slow_dir = jax.random.normal(jax.random.PRNGKey(2), (1, dim))
    slow_dir = slow_dir / jnp.linalg.norm(slow_dir)

    x = fast_component * fast_dir + slow_component * slow_dir  # (seq, dim)
    x = x[None, :, :]  # (1, seq, dim)

    # Create CMS and get per-level outputs
    cms = ContinuumMemorySystem(
        dim=dim, hidden_dim=64, num_levels=2, chunk_sizes=[1, 4],
    )
    params = cms.init(rng, x, step=0)
    _, level_outputs = cms.apply(params, x, step=0, training=False)

    # For each level output, measure correlation with fast and slow components
    print(f"  {'Level':>8} | {'Corr w/ Fast':>12} | {'Corr w/ Slow':>12} | {'Specializes to':>15}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}")

    for i, level_out in enumerate(level_outputs):
        out = level_out[0]  # remove batch dim: (seq, dim)

        # Project level output onto fast and slow directions
        proj_fast = jnp.abs(out @ fast_dir.T).mean()
        proj_slow = jnp.abs(out @ slow_dir.T).mean()

        # Compute correlation with the original components
        out_mean = out - out.mean(axis=0)
        fast_mean = (fast_component * fast_dir) - (fast_component * fast_dir).mean(axis=0)
        slow_mean = (slow_component * slow_dir) - (slow_component * slow_dir).mean(axis=0)

        corr_fast = float(jnp.abs(jnp.sum(out_mean * fast_mean)) /
                         (jnp.linalg.norm(out_mean) * jnp.linalg.norm(fast_mean) + 1e-8))
        corr_slow = float(jnp.abs(jnp.sum(out_mean * slow_mean)) /
                         (jnp.linalg.norm(out_mean) * jnp.linalg.norm(slow_mean) + 1e-8))

        specializes = "Fast" if corr_fast > corr_slow else "Slow"
        expected = "Fast" if i == 0 else "Slow"
        match = "ok" if specializes == expected else "MISMATCH"

        print(f"  Level {i:>2} | {corr_fast:>12.4f} | {corr_slow:>12.4f} | "
              f"{specializes:>8} ({match})")

    print()
    print("  Above: at initialization (untrained). Now training on this data...")
    print()

    # --- PART 2: Train the CMS, then re-check specialization ---
    import optax

    class SpecModel(nn.Module):
        dim: int
        @nn.compact
        def __call__(self, x, step=0, training=True):
            out, level_outs = ContinuumMemorySystem(
                dim=self.dim, hidden_dim=64, num_levels=2, chunk_sizes=[1, 4],
                name="cms",
            )(x, step=step, training=training)
            # Simple reconstruction target
            pred = nn.Dense(self.dim, name="head")(out)
            return pred, level_outs

    spec_model = SpecModel(dim=dim)
    spec_params = spec_model.init(rng, x, step=0)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(spec_params)

    for step in range(120):
        def recon_loss(p):
            pred, _ = spec_model.apply(p, x, step=step, training=True)
            return jnp.mean((pred - x) ** 2)
        loss, grads = jax.value_and_grad(recon_loss)(spec_params)
        updates, opt_state = optimizer.update(grads, opt_state, spec_params)
        spec_params = optax.apply_updates(spec_params, updates)

    _, trained_level_outputs = spec_model.apply(spec_params, x, step=0, training=False)

    print(f"  AFTER TRAINING (120 steps of reconstruction):")
    print(f"  {'Level':>8} | {'Corr w/ Fast':>12} | {'Corr w/ Slow':>12} | {'Specializes to':>15}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}")

    for i, level_out in enumerate(trained_level_outputs):
        out = level_out[0]
        out_mean = out - out.mean(axis=0)
        fast_mean = (fast_component * fast_dir) - (fast_component * fast_dir).mean(axis=0)
        slow_mean = (slow_component * slow_dir) - (slow_component * slow_dir).mean(axis=0)

        corr_fast = float(jnp.abs(jnp.sum(out_mean * fast_mean)) /
                         (jnp.linalg.norm(out_mean) * jnp.linalg.norm(fast_mean) + 1e-8))
        corr_slow = float(jnp.abs(jnp.sum(out_mean * slow_mean)) /
                         (jnp.linalg.norm(out_mean) * jnp.linalg.norm(slow_mean) + 1e-8))

        specializes = "Fast" if corr_fast > corr_slow else "Slow"
        expected = "Fast" if i == 0 else "Slow"
        match = "ok" if specializes == expected else "MISMATCH"

        print(f"  Level {i:>2} | {corr_fast:>12.4f} | {corr_slow:>12.4f} | "
              f"{specializes:>8} ({match})")

    print()


# ================================================================
# Q5: Memory capacity — how many facts before it breaks?
# ================================================================
#
# The associative memory is a linear map W of shape (head_dim, head_dim).
# In theory, it can store at most head_dim linearly independent key-value
# pairs perfectly. But with momentum and forgetting, the effective
# capacity could be more or less.
#
# This is analogous to Hopfield network capacity: a network with N
# neurons can store ~N/(4*log(N)) patterns. What's the equivalent
# for this gradient-based memory?

def question_memory_capacity():
    """Feed the memory increasing numbers of key-value pairs,
    then test retrieval accuracy. Find the breaking point.
    """
    from nested_learning.associative_memory import AssociativeMemory

    print("=" * 60)
    print("Q5: Memory capacity — how many facts can it hold?")
    print("=" * 60)
    print()

    rng = jax.random.PRNGKey(42)
    dim = 32
    head_dim = dim // 2  # 2 heads, so head_dim = 16

    print(f"  Head dimension: {head_dim} (theoretical max capacity: ~{head_dim} facts)")
    print()

    memory = AssociativeMemory(dim=dim, memory_dim=32, num_heads=2, lr=0.1, momentum=0.9)

    results = {}
    for num_facts in [4, 8, 12, 16, 20, 24, 32]:
        # Create num_facts distinct key-value pairs
        keys = jax.random.normal(jax.random.PRNGKey(num_facts), (1, num_facts, dim))
        # Orthogonalize keys for clean experiment
        if num_facts <= dim:
            keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)

        # Write all facts to memory
        params = memory.init(rng, keys)
        output, mem_state = memory.apply(params, keys)

        # Now query: feed the same keys, check if output matches
        retrieval, _ = memory.apply(params, keys, memory_state=mem_state)

        # Measure retrieval quality (cosine similarity between retrieved and stored output)
        # We compare the first-pass output (which serves as the "intended" values)
        # to the second-pass output (which queries the memory with state)
        cos_sim = float(jnp.mean(
            jnp.sum(output * retrieval, axis=-1) /
            (jnp.linalg.norm(output, axis=-1) * jnp.linalg.norm(retrieval, axis=-1) + 1e-8)
        ))

        mse = float(jnp.mean((output - retrieval) ** 2))
        results[num_facts] = {"cos_sim": cos_sim, "mse": mse}
        print(f"  {num_facts:>3} facts | Cosine similarity: {cos_sim:.4f} | MSE: {mse:.4f}")

    print()
    print(f"  Head dimension = {head_dim}.")
    print(f"  Theory predicts capacity ~ {head_dim} for a linear associative memory.")
    print(f"  Look for where cosine similarity drops sharply — that's the capacity limit.")
    print()

    return results


# ================================================================
# Q6: Self-modification — does it converge or diverge?
# ================================================================
#
# Self-modifying layers update their weights at each forward pass.
# Three possibilities:
#   a) Converge to a fixed point (stable — weights settle down)
#   b) Oscillate (unstable — weights bounce around)
#   c) Diverge (catastrophic — weights grow without bound)
#
# This determines whether self-modification is safe to use for
# long inference sequences. If it diverges, the model breaks
# after processing enough tokens.

def question_selfmod_stability():
    """Apply self-modification repeatedly and track weight norm.
    Does it converge, oscillate, or diverge?
    """
    from nested_learning.self_modifying import SelfModifyingLinear

    print("=" * 60)
    print("Q6: Self-modification stability over many passes")
    print("=" * 60)
    print()

    rng = jax.random.PRNGKey(42)
    dim = 32
    num_passes = 50

    # Test with different learning rates
    for mod_lr in [0.001, 0.01, 0.05, 0.1]:
        layer = SelfModifyingLinear(features=dim, mod_lr=mod_lr, normalized=True)
        x = jax.random.normal(rng, (4, dim))
        params = layer.init(rng, x)

        base_W = params["params"]["kernel"]
        fast_weights = None
        norms = []

        for t in range(num_passes):
            _, fast_weights = layer.apply(params, x, fast_weights=fast_weights)
            drift = float(jnp.linalg.norm(fast_weights["W"] - base_W))
            w_norm = float(jnp.linalg.norm(fast_weights["W"]))
            norms.append(w_norm)

        # Classify behavior
        early = np.mean(norms[:5])
        late = np.mean(norms[-5:])
        mid = np.mean(norms[20:25])

        if late > early * 2:
            behavior = "DIVERGING"
        elif abs(late - mid) / (mid + 1e-8) < 0.05:
            behavior = "CONVERGED"
        else:
            behavior = "OSCILLATING"

        print(f"  lr={mod_lr:.3f} | "
              f"norm: {norms[0]:.3f} -> {norms[24]:.3f} -> {norms[-1]:.3f} | "
              f"{behavior}")

    print()
    print("  CONVERGED = safe for long sequences")
    print("  DIVERGING = will break after enough tokens (need smaller lr)")
    print("  OSCILLATING = unpredictable, may cause instability")
    print()
    print("  This determines the maximum 'inference horizon' —")
    print("  how many tokens you can process before self-modification")
    print("  drifts the weights too far from their trained values.")
    print()


# ================================================================
# Q7: Temporal generalization — can it transfer across scales?
# ================================================================
#
# If we train a CMS model on patterns at scale X (e.g., repeating
# every 4 tokens), can it generalize to patterns at scale 2X
# (repeating every 8 tokens)?
#
# If yes: CMS learns abstract "pattern recognition at a time scale"
# If no: CMS just memorizes specific frequencies

def question_temporal_generalization():
    """Train on patterns at one frequency, test on a different frequency.
    Does the multi-scale structure help generalize?
    """
    import flax.linen as nn
    from nested_learning.continuum_memory import ContinuumMemorySystem
    import optax

    print("=" * 60)
    print("Q7: Can CMS generalize across temporal scales?")
    print("=" * 60)
    print()
    print("Train on period-4 patterns. Test on period-8 patterns.")
    print("A model that learned 'repeating structure' should transfer.")
    print()

    rng = jax.random.PRNGKey(42)
    vocab = 10
    dim = 48
    seq_len = 32

    # Training data: pattern repeating every 4 tokens
    train_pattern = jnp.array([i % 4 for i in range(200)])

    # Test data: pattern repeating every 8 tokens (related but different scale)
    test_same = jnp.array([i % 4 for i in range(200)])   # same scale
    test_diff = jnp.array([i % 8 for i in range(200)])    # different scale
    test_rand = jax.random.randint(jax.random.PRNGKey(99), (200,), 0, vocab)  # random

    class Model(nn.Module):
        vocab_size: int
        dim: int
        use_cms: bool

        @nn.compact
        def __call__(self, x, step=0, training=True):
            h = nn.Embed(self.vocab_size, self.dim, name="embed")(x)
            if self.use_cms:
                h, _ = ContinuumMemorySystem(
                    dim=self.dim, hidden_dim=self.dim * 2,
                    num_levels=3, chunk_sizes=[1, 4, 16], name="cms",
                )(h, step=step, training=training)
            else:
                h2 = nn.Dense(self.dim * 2, name="ffn_up")(h)
                h2 = nn.gelu(h2)
                h2 = nn.Dense(self.dim, name="ffn_down")(h2)
                h = h + h2
            return nn.Dense(self.vocab_size, name="head")(h), []

    def train_and_test(model, train_data, test_datasets, num_steps=100):
        key = jax.random.PRNGKey(42)
        dummy = jax.random.randint(key, (1, seq_len), 0, vocab)
        params = model.init(key, dummy, step=0)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        def loss_fn(p, batch, step):
            logits, _ = model.apply(p, batch, step=step, training=True)
            shift_logits = logits[:, :-1, :]
            shift_targets = batch[:, 1:]
            one_hot = jax.nn.one_hot(shift_targets, vocab)
            log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
            return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

        rng = jax.random.PRNGKey(0)
        for step in range(num_steps):
            rng, batch_key = jax.random.split(rng)
            starts = jax.random.randint(batch_key, (4,), 0, len(train_data) - seq_len)
            batch = jnp.stack([train_data[s:s + seq_len] for s in starts])
            loss, grads = jax.value_and_grad(loss_fn)(params, batch, step)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        # Evaluate on all test sets
        results = {}
        for name, test_data in test_datasets.items():
            batch = test_data[:seq_len][None, :]
            results[name] = float(loss_fn(params, batch, 0))
        return results

    tests = {
        "Same scale (4)": test_same,
        "New scale (8)": test_diff,
        "Random": test_rand,
    }

    print(f"  {'':>20} | ", end="")
    for name in tests:
        print(f"{name:>15} | ", end="")
    print()
    print(f"  {'-'*20}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}")

    for model_name, use_cms in [("With CMS", True), ("Without CMS", False)]:
        model = Model(vocab_size=vocab, dim=dim, use_cms=use_cms)
        results = train_and_test(model, train_pattern, tests)
        print(f"  {model_name:>20} | ", end="")
        for name in tests:
            print(f"{results[name]:>15.4f} | ", end="")
        print()

    print()
    print("  INTERPRETATION:")
    print("  - 'Same scale' should be lowest (trained on this)")
    print("  - If 'New scale' << 'Random': model DOES generalize across scales")
    print("  - If CMS generalizes better than flat: multi-scale structure helps transfer")
    print()


# ================================================================

if __name__ == "__main__":
    print()
    print("Nested Learning — Deeper Open Questions")
    print("=" * 60)
    print()
    print("These experiments test whether the mechanisms ACTUALLY work")
    print("the way the paper claims, not just whether they run.")
    print()

    t0 = time.time()

    r1 = question_does_adaptation_help()
    r2 = question_surprise_calibration()
    r3 = question_catastrophic_forgetting()
    r4 = question_level_specialization()
    r5 = question_memory_capacity()
    r6 = question_selfmod_stability()
    r7 = question_temporal_generalization()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"All questions explored in {elapsed:.0f}s")
    print("=" * 60)
    print()
    print("SUMMARY OF OPEN QUESTIONS:")
    print()
    print("Q1 (Adaptation): Does carried state improve or hurt predictions?")
    print("Q2 (Surprise): Is the surprise signal well-calibrated or fooled by noise?")
    print("Q3 (Forgetting): Does CMS actually prevent catastrophic forgetting?")
    print("Q4 (Specialization): Do CMS levels capture different frequencies?")
    print("Q5 (Capacity): How many facts before the memory breaks?")
    print("Q6 (Stability): Does self-modification converge or diverge?")
    print("Q7 (Generalization): Can CMS transfer across temporal scales?")
    print()
    print("Each question probes a MECHANISM, not just a benchmark number.")
    print("The answers tell us whether Nested Learning works FOR THE REASONS")
    print("the paper claims, or for other reasons, or not at all.")
