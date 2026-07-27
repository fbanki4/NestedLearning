# Nested Learning — A Layman's Guide + JAX Implementation

A JAX implementation of **"Nested Learning: The Illusion of Deep Learning Architectures"** by Behrouz, Razaviyayn, Zhong, and Mirrokni (Google Research, NeurIPS 2025).

This repo is designed for someone who isn't a machine learning researcher. Every module has plain-English docstrings explaining what's happening and why. The code runs on CPU with small models so you can experiment without a GPU.

---

## What Is Nested Learning? (No Jargon Version)

### The One-Sentence Version
> A model's **architecture** (how it's built) and its **optimizer** (how it learns) are secretly the same thing — just running at different speeds.

### The Analogy

Imagine a **student preparing for exams**:

| What the student does | ML equivalent | Speed |
|---|---|---|
| Reads a sentence and understands it | **Forward pass** (inference) | Milliseconds |
| Takes notes while reading | **Self-modification** (delta rule) | Every sentence |
| Reviews notes at the end of each chapter | **Associative memory** (surprise-based) | Every chapter |
| Reorganizes binder at the end of each week | **CMS level 1** (medium-term memory) | Every week |
| Writes a summary at the end of each month | **CMS level 2** (long-term memory) | Every month |
| Takes the exam and gets a grade | **Backpropagation** (training signal) | Every exam |

The student has **multiple learning loops** running at different speeds. The note-taking (fast) is informed by the binder organization (slow), which is informed by the monthly summaries (slower). Each loop is its own little optimization problem — trying to be as useful as possible within its own scope.

**That's Nested Learning.** A neural network with multiple learning loops at different speeds, all running simultaneously. The paper shows that this is not just an analogy — *mathematically*, adding a new learning loop is identical to adding a new layer.

### Why "Illusion"?

The paper's title says "The Illusion of Deep Learning Architectures." Here's what that means:

When you look at a deep neural network, you see **layers**. When you look at an optimizer like Adam, you see **momentum and learning rate schedules**. They look completely different.

But the paper proves they're the same thing:
- A **layer** transforms data
- **Momentum** transforms gradients
- Both are neural networks processing a stream of inputs
- Both have parameters that get updated over time
- The only difference is the **speed** at which they update

So the distinction between "architecture" and "optimizer" is an **illusion**. They're all just nested optimization levels running at different frequencies.

---

## The Five Building Blocks

### 1. Associative Memory (the notebook)
**File:** `nested_learning/associative_memory.py`

A small neural network that acts as a key-value store. You write facts to it (key → value pairs) and read them back later. The write operation is literally gradient descent — the same algorithm used to train neural networks.

The clever part: **surprise gating**. If a new fact is boring (the memory already knows something similar), it barely changes anything. If a new fact is surprising (high error = high gradient), it writes aggressively. This is inspired by how human memory works — you remember unexpected events more vividly.

**Key equation:**
```
memory_update = momentum * past_surprise - learning_rate * gradient_of_prediction_error
```

This is exactly SGD with momentum. The paper's insight: momentum IS memory.

### 2. Deep Momentum Optimizer (the learned study strategy)
**File:** `nested_learning/deep_optimizer.py`

Standard optimizers like Adam use a fixed formula to combine current and past gradients:
```
new_momentum = 0.9 * old_momentum + current_gradient    (fixed formula)
```

Deep Momentum replaces this with a small neural network:
```
new_momentum = NeuralNetwork(old_momentum, current_gradient)  (learned formula)
```

The neural network is trained with its own loss function at every step. This means we have **optimization inside optimization** — a nested learning problem. The inner optimizer learns how to be a good optimizer for the outer problem.

### 3. Self-Modifying Layers (the self-editing notes)
**File:** `nested_learning/self_modifying.py`

Normal neural network layers have fixed weights during inference. Self-modifying layers update their own weights as data flows through them, using the delta rule from neuroscience:

```
"If I predicted wrong, adjust my weights to be less wrong for this specific input."
```

This means the model can **adapt in real-time** to whatever data it's processing, without explicit retraining. It's like a student who gets better at a test while taking it.

### 4. Continuum Memory System (the multi-speed filing system)
**File:** `nested_learning/continuum_memory.py`

A chain of small MLPs, each updating at a different frequency:
- **Level 0:** Updates every step (working memory — what's happening right now)
- **Level 1:** Updates every 8 steps (short-term — recent patterns)
- **Level 2:** Updates every 64 steps (long-term — established knowledge)

The different speeds create natural protection against **catastrophic forgetting** — the problem where learning something new makes you forget something old. Fast memories absorb new information while slow memories preserve existing knowledge.

### 5. HOPE Model (everything together)
**File:** `nested_learning/hope.py`

Stacks all four components into a full sequence model:

```
Input tokens
    ↓
Token + Position Embeddings
    ↓
┌──────────────────────────────┐
│         HOPE Block           │  ← repeated N times
│  1. Self-Modifying Attention │  (weights change while running)
│  2. Associative Memory Gate  │  (surprise-based read/write)
│  3. Continuum Memory System  │  (multi-speed MLPs)
│  4. Feed-Forward Network     │  (standard feature mixing)
└──────────────────────────────┘
    ↓
Output predictions
```

---

## Project Structure

```
NestedLearning/
├── README.md                           ← You are here
├── requirements.txt                    ← Dependencies (JAX, Flax, Optax)
├── nested_learning/
│   ├── __init__.py                     ← Package entry point
│   ├── associative_memory.py           ← Surprise-gated key-value memory
│   ├── deep_optimizer.py               ← Learned momentum (Deep Momentum GD)
│   ├── self_modifying.py               ← Delta-rule self-modifying layers
│   ├── continuum_memory.py             ← Multi-frequency memory system (CMS)
│   ├── hope.py                         ← Full HOPE model
│   └── train.py                        ← Training utilities
└── examples/
    ├── demo.py                         ← Interactive demos of each component
    ├── quick_experiment.py             ← Fast scaling experiments (depth, ratios, ablation)
    ├── deeper_questions.py             ← 7 mechanism-probing experiments
    └── scaling_exploration.py          ← Full scaling experiments (slower, more thorough)
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive demos (no GPU needed, takes ~1 minute)
python examples/demo.py

# Run scaling experiments (takes ~5-10 minutes on CPU)
python examples/scaling_exploration.py
```

---

## The Open Problem: How Many Levels?

This is the most interesting unanswered question in Nested Learning, and we've built tools to explore it.

### The Question

The framework lets you add as many optimization levels as you want. But **how many should you use?** The paper doesn't give a definitive answer. This is analogous to the question "how deep should a neural network be?" but for optimization levels instead of layers.

### What We Know

1. **More levels = more temporal scales.** A 1-level model can only handle one speed of pattern. A 5-level model can handle patterns at 5 different speeds simultaneously.

2. **Diminishing returns are likely.** The paper shows 2-3 levels outperform 1 level. But each additional level adds parameters and compute. At some point, the cost outweighs the benefit.

3. **It probably depends on the task.** A task with long-range dependencies (e.g., understanding a novel) probably needs more levels than a task with short-range dependencies (e.g., sentiment analysis of tweets).

### What We Don't Know

1. **Is there a scaling law?** For Transformers, loss scales as a power law of parameter count (the Chinchilla law: loss ~ N^(-0.076)). Does HOPE follow a similar law? Does the exponent change with depth?

2. **What's the optimal chunk size ratio?** Should levels be spaced exponentially [1, 8, 64] or linearly [1, 10, 20]? Does this depend on the data's temporal structure?

3. **Interaction effects.** Does self-modification become more or less important as we add CMS levels? Are they complementary or redundant?

4. **Compute-optimal depth.** Given a fixed compute budget, is it better to add more levels or make the model wider?

5. **Emergent capabilities.** At what depth/scale do qualitatively new capabilities appear?

### How to Explore

The `scaling_exploration.py` script provides four experiments:

| Experiment | What it tests | Key question |
|---|---|---|
| CMS Depth | 1-5 CMS levels, same model size | Diminishing returns? |
| Chunk Ratios | Different level spacings | Exponential vs linear spacing? |
| Scaling Laws | Different model sizes | Power law exponent? |
| Ablation | Remove each component | Which level matters most? |

To run with your own configuration:
```python
from examples.scaling_exploration import experiment_cms_depth

results = experiment_cms_depth(
    max_levels=7,           # Try up to 7 levels
    dim=128,                # Larger model
    seq_len=256,            # Longer sequences
    num_train_steps=500,    # More training
)
```

For serious investigation, you'd want to:
1. Use real data (WikiText-103, The Pile, etc.)
2. Train much larger models (millions of parameters)
3. Run many random seeds for statistical significance
4. Vary sequence length to test the temporal dependency hypothesis

---

## Deeper Open Questions (and What We Found)

Beyond the scaling/depth question, we ran 7 experiments (`examples/deeper_questions.py`) that test whether the mechanisms work *for the reasons the paper claims*. Each probes a specific mechanism, not just a benchmark number.

### Q1: Does Carried State Actually Help Prediction?

**Test:** Feed the model chunk 1, carry memory + fast weights forward, measure if chunk 2 predictions improve vs. starting fresh.

**Result:** Adaptation **hurt** slightly, both before and after training. On an untrained model, carried state added +0.15 loss on same-pattern data. After 80 steps of training (loss dropped 3.4 → 0.01), the delta shrank to +0.006 — nearly neutral.

**What this means:** The model learns the pattern so well through backprop that the inner optimization has little room to help. The carried state adds a small amount of noise. This raises a question the paper doesn't address: **is inner optimization redundant when outer optimization is sufficient?** It might only shine when the outer optimizer hasn't converged — e.g., few-shot or continual learning settings.

### Q2: Is the Surprise Signal Fooled by Noise?

**Test:** Prime the memory with a pattern, then measure surprise for: (a) the familiar pattern, (b) a novel structured pattern, (c) random noise.

**Result (original):** Noise triggered the *most* surprise (15.8 memory change), far more than novel data (4.1). This means the memory aggressively stores garbage.

**The fix:** We added `normalize_surprise=True`, which divides the gradient by input norm before computing the surprise signal. After this fix, ranking became **Novel > Noise > Familiar** — exactly correct.

**Why it works:** Random noise has high norm, so the raw gradient is large even though there's nothing useful to learn. Normalizing removes this magnitude bias, letting the *direction* of the error (which captures real novelty) dominate.

This is enabled as a flag: `AssociativeMemory(normalize_surprise=True)`.

### Q3: Does CMS Actually Prevent Catastrophic Forgetting?

**Test:** Train on Task A (repeating 0,1,2,3,4...), then Task B (repeating 4,3,2,1,0...). Measure how much Task A is forgotten.

**Result:** CMS forgot 2.6x less than a flat model (loss increase of +2.48 vs +6.57). This is the clearest positive result — **multi-frequency updates genuinely protect old knowledge**.

**Why it works:** When Task B training only updates the fast CMS levels (which update every step), the slow levels (which update every 16+ steps) retain Task A patterns. It's like how your long-term memory of how to ride a bike isn't erased by learning a new phone number.

### Q4: Do CMS Levels Specialize to Different Frequencies?

**Test:** Create data with patterns at two known frequencies (period 4 and period 16). Check if fast-updating levels capture the fast pattern and slow-updating levels capture the slow pattern.

**Result:** **No.** Both levels correlated more with the fast pattern, both before and after training. Training didn't drive specialization.

**What this means:** The CMS architecture doesn't automatically sort patterns by frequency. The update schedule controls *when* gradients flow, but doesn't force *what* each level learns. For specialization to emerge, the architecture might need explicit frequency-band filtering (like a filter bank in audio processing), or much longer training sequences where the mismatch would create stronger selection pressure.

### Q5: Memory Capacity

**Test:** Store increasing numbers of key-value pairs and measure retrieval quality (cosine similarity).

**Result:** Unexpectedly, cosine similarity *increased* with more facts (0.68 at 4 facts → 0.81 at 32 facts). No sharp capacity limit was observed.

**Why this is misleading:** The metric compares first-pass output to second-pass output — it measures *consistency*, not accuracy. With many facts, the memory averages over more data, producing smoother (more consistent) outputs. A better test would need ground-truth key-value pairs and measure exact recall.

### Q6: Self-Modification Stability

**Test:** Apply self-modification 50 times with different learning rates, tracking weight norm.

**Result:** Converged at all tested learning rates (0.001 to 0.1). Weight norms decreased monotonically (5.74 → 5.37 at lr=0.1). No divergence or oscillation.

**What this means:** Self-modification is safe for long sequences. The delta rule has an implicit regularization effect — it shrinks weights toward a fixed point. This is good news: you won't get catastrophic drift during long inference.

### Q7: Temporal Generalization

**Test:** Train on period-4 patterns, test on period-8 patterns (related but different scale).

**Result:** Both CMS and flat models showed the same pattern: same scale >> new scale >> random. CMS didn't generalize better than a flat model to unseen frequencies.

**What this means:** CMS doesn't learn abstract "pattern recognition at a time scale." It memorizes specific frequencies. Generalization across scales likely requires explicit scale-invariance mechanisms (like dilated convolutions or frequency-domain processing).

### Summary Scorecard

| Question | Paper's Claim | Our Finding | Verdict |
|---|---|---|---|
| Q1: Adaptation helps | Carried state improves predictions | Neutral-to-slightly-harmful | Needs more study |
| Q2: Surprise = novelty | Surprise gates useful writes | Noise fools it (fixed with normalization) | **Flaw found + fixed** |
| Q3: CMS prevents forgetting | Multi-frequency = protection | 2.6x less forgetting vs flat | **Confirmed** |
| Q4: Levels specialize | Fast level → fast patterns | No specialization after training | **Not confirmed** |
| Q5: Bounded capacity | Linear memory has finite capacity | No clear limit observed | Metric issue |
| Q6: Self-mod is stable | Weights converge | Converges at all LRs tested | **Confirmed** |
| Q7: Scale transfer | Multi-scale = generalization | No better than flat model | **Not confirmed** |

### Caveats

These are **tiny models** (7K-117K parameters) on **synthetic data** (500 tokens) with **short training** (60-120 steps). The results tell us about the mechanisms at small scale. At large scale with real data, the dynamics could be different — particularly Q1 (where inner optimization might matter more when the outer optimizer hasn't converged) and Q4 (where longer sequences might create stronger selection pressure for specialization).

---

## How This Connects to Existing Ideas

| Concept | Traditional View | Nested Learning View |
|---|---|---|
| Neural network layers | Feature extractors | Optimization levels |
| Optimizer momentum | Running average of gradients | Associative memory |
| Learning rate schedule | Hyperparameter tuning | Another optimization level |
| Attention mechanism | Weighted token mixing | Short-term memory |
| FFN layers | Feature transformation | Long-term memory |
| Meta-learning (MAML) | Learning to learn | Just adding another nesting level |

---

## PyTorch / GPU extension: `nested_torch/`

Alongside this CPU/JAX teaching implementation, the repo now ships a
**PyTorch, GPU-ready** module, [`nested_torch/`](nested_torch/README.md), that
applies Nested Learning to a **frozen** transformer for continual learning:

- It retrofits an already-trained transformer so each block's attached memory
  updates at its own **frequency**, gated by **token and concept perplexity**.
- Its default `"workspace"` schedule makes the **middle** blocks fastest and the
  ends slow — a U-shaped frequency profile aligned with Anthropic's *J-space /
  J-lens* finding that a transformer's global workspace lives in its middle
  layers. The aim is to concentrate plasticity in the workspace while frozen
  weights and slow end-blocks resist catastrophic forgetting.
- This is the adaptation layer for the larger goal of continual/RL learning on
  an open **EEG foundation model** (CBraMod/LaBraM). See
  [`nested_torch/README.md`](nested_torch/README.md) for design, demos, measured
  results, and the roadmap (real backbone → RL controller → CL benchmark).

```bash
pip install -r requirements-torch.txt
python examples_pytorch/demo_multifrequency.py   # update-frequency profile
python examples_pytorch/cl_forgetting.py          # 2-task forgetting experiment
python tests_pytorch/test_nested_torch.py         # 8 tests (no pytest needed)
```

## References

- **Nested Learning paper:** Behrouz, Razaviyayn, Zhong, Mirrokni. *"Nested Learning: The Illusion of Deep Learning Architectures."* NeurIPS 2025. [arXiv:2512.24695](https://arxiv.org/abs/2512.24695)
- **Titans paper:** Behrouz, Zhong. *"Titans: Learning to Memorize at Test Time."* ICML 2025. [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- **Google Research blog:** [Introducing Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
- **Reference implementation (PyTorch):** [erikl2/nested-learning](https://github.com/erikl2/nested-learning)
