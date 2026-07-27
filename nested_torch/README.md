# `nested_torch` — Frozen-Model Multi-Frequency Retrofit with a J-Space Workspace Schedule

A **PyTorch, GPU-ready** implementation of the "ad hoc" idea sketched in the
Nested Learning paper: take an **already-trained, frozen** transformer and
retroactively give each of its blocks its own **update frequency**, driven by
**token and concept perplexity**. On top of that we add one new idea:

> **The middle blocks update fastest; the first and last blocks stay slow.**
> A U-shaped period curve (∩-shaped *frequency*), concentrating plasticity in
> the model's **global workspace** — the sparse, privileged mid-network subspace
> that Anthropic's July-2026 *J-space / J-lens* interpretability work found lives
> *only in the middle block* of a transformer.

This is the opposite of the vanilla Continuum Memory System schedule
("early = fast, late = slow"). The bet: in a *pretrained* model, the middle is
where concepts are held and combined, so that is exactly where you want
plasticity when adapting to new tasks — while frozen weights and near-frozen
early/late memories protect old tasks from catastrophic forgetting.

Nothing here trains by backprop. Adaptation is **online, at test time**, via the
delta rule (surprise-gated associative memory) — a direct PyTorch port of this
repo's JAX `AssociativeMemory` and `ContinuumMemorySystem`, with the memory made
**persistent across batches/tasks** (the right choice for continual learning).

---

## Why this exists (the EEG + continual-learning goal)

The larger project: do continual/reinforcement learning on an open EEG
foundation model (CBraMod, LaBraM) and beat CL benchmarks. This module is the
**adaptation layer** — a way to make a *frozen* EEG FM continually adapt without
forgetting, by concentrating plasticity in its workspace. The backbone here is a
tiny stand-in transformer so the mechanism can be validated on CPU; swapping in
a real EEG FM is a one-liner (below).

## Install

```bash
pip install -r requirements-torch.txt   # torch>=2.2, numpy
```

## Quick start

```python
import torch
from nested_torch import ReferenceTransformer, RetrofitModel

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = ReferenceTransformer(in_dim=32, dim=64, depth=9).to(device)

# Retrofit: middle blocks = fast "global workspace"; ends stay slow.
model = RetrofitModel.build(backbone, schedule_name="workspace",
                            min_period=1, max_period=32).to(device)

x = torch.randn(8, 16, 32, device=device)
y = model(x, adapt=True)              # online adaptation via the delta rule
print(model.adaptation_report())      # middle blocks updated most often
```

### Retrofit a real EEG foundation model

`FrozenRetrofit` wraps any `nn.ModuleList` of transformer blocks:

```python
from nested_torch import FrozenRetrofit
retrofit = FrozenRetrofit(eeg_model.blocks, dim=eeg_model.dim,
                          schedule_name="workspace")
h = eeg_model.embed(eeg_patches)      # backbone's own embedding
h = retrofit(h, adapt=True)           # adapt only the (frozen) blocks' memories
logits = eeg_model.head(h)            # backbone's own read-out
```

## What's in the box

| File | Role |
|---|---|
| `frequency_schedule.py` | Depth→frequency schedules. `"workspace"` = U-shaped (J-space). Baselines: `uniform`, `shallow_fast`, `deep_fast`, `edge_fast`. Pure Python. |
| `surprise.py` | **Token perplexity** (norm-normalized memory error, incl. the repo's Q2 noise fix) + **concept perplexity** (novelty vs an EMA concept codebook). |
| `fast_weights.py` | `BlockMemory` — surprise-gated delta-rule associative memory, persistent state in buffers. |
| `frozen_retrofit.py` | `FrozenRetrofit` — freezes the backbone, attaches a memory per block, runs multi-frequency + surprise-triggered updates, logs per-block adaptation. |
| `reference_transformer.py` | Tiny stand-in transformer + `RetrofitModel` wrapper (frozen embed/read-out + retrofit). |
| `toy_data.py` | Synthetic subspace tasks for CPU validation. |

## Demos

```bash
python examples_pytorch/demo_multifrequency.py   # the update-frequency profile
python examples_pytorch/cl_forgetting.py          # a 2-task forgetting experiment
python tests_pytorch/test_nested_torch.py         # 8 tests, no pytest needed
```

**`demo_multifrequency.py`** — with `workspace`, update counts peak in the middle
and taper to the ends (measured, 9-block model, 200 steps):

```
block  0  period= 14  updates=  15  rate=0.07  |##
block  3  period=  3  updates=  67  rate=0.34  |##########
block  4  period=  1  updates= 200  rate=1.00  |##############################
block  5  period=  3  updates=  67  rate=0.34  |##########
block  8  period= 14  updates=  15  rate=0.07  |##
```

**`cl_forgetting.py`** — two-task memory experiment (lower `forgetting` = better):

```
schedule    errA|afterA errA|afterB  forgetting  learnedB
frozen           3.8824      3.8824      0.0000    0.0000
workspace        2.3833      2.0622     -0.3210    0.5748
uniform          0.7706      1.2038      0.4331    0.7604
edge_fast        2.5290      2.0457     -0.4833    0.6348
```

**Honest reading.** `uniform` (fast everywhere) is the *only* schedule that
catastrophically forgets (+0.43); restricting *where* plasticity lives —
`workspace` **or** `edge_fast` — prevents forgetting (both show slight backward
transfer). This reproduces the repo's JAX "Q3" result that multi-frequency
updates protect old knowledge. It does **not** yet confirm the J-space-specific
claim that the *middle* is special: on a **randomly-initialized** backbone the
middle is not a semantic workspace, so `workspace` ≈ `edge_fast`. Testing the
middle-beats-ends hypothesis requires a **pretrained** model (CBraMod/LaBraM) —
see roadmap.

## Roadmap

1. **Real backbone.** Drop CBraMod/LaBraM blocks into `FrozenRetrofit`; re-run
   `cl_forgetting.py` on an EvoBrain-style stream of real BCI tasks. Only here
   can `workspace` vs `edge_fast` actually test the J-space hypothesis.
2. **RL layer.** The `scheduled` / `do_update` / gate decisions are exactly what
   a policy can control. Cast per-block adaptation as an RL policy whose reward
   penalizes forgetting (retained-task accuracy), turning surprise-gating into a
   *learned* continual-learning controller.
3. **Benchmark.** Report Average Accuracy + Backward Transfer vs LwF/EWC/replay
   and EvoBrain on the EEG-FM downstream suite.

## References

- Behrouz et al., *Nested Learning: The Illusion of Deep Learning Architectures*, NeurIPS 2025 (arXiv:2512.24695).
- Behrouz & Zhong, *Titans: Learning to Memorize at Test Time*, ICML 2025 (arXiv:2501.00663).
- Anthropic, *J-space / J-lens: a global workspace in the middle layers of Claude* (July 2026).
- This repo's JAX implementation and its Q2 (surprise normalization) and Q3 (multi-frequency prevents forgetting) findings — see the top-level `README.md`.
