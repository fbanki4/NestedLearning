"""
Demo 4 — RL: learn where a frozen model should be plastic (beat the fixed schedules)
====================================================================================

Instead of *guessing* which blocks should adapt (the fixed `workspace` / `edge` /
`uniform` schedules), we train a small policy with REINFORCE to *decide*, per
block per step, whether to write to that block's memory — with a reward that
punishes forgetting (see ``nested_torch/rl_controller.py``).

Pipeline: pretrain an EEG backbone -> freeze it -> train the policy on a stream
of two EEG tasks (theta/alpha band then beta/gamma band) -> compare the learned
policy against the fixed schedules on the same continual-learning protocol, and
print the per-block update profile the policy discovered.

Run:
    python examples_pytorch/rl_controller.py
Tiny synthetic model on CPU; runs in ~1-2 minutes. A mechanism demo, not a benchmark.
"""

import os

import torch

from nested_torch import (EEGTransformer, RetrofitModel, pretrain_masked,
                          PolicyController, ReinforceTrainer)
from nested_torch.toy_data import make_eeg_batch

C, T, P, DIM, DEPTH = 8, 128, 16, 64, 9
BAND_A = [4, 6, 8, 10]
BAND_B = [18, 22, 26, 30]
PRETRAIN_BANDS = [4, 6, 8, 10, 14, 18, 22, 26, 30]
# Overridable for a quick smoke run, e.g. NL_RL_EPISODES=8 NL_RL_STEPS=10 NL_PRETRAIN_STEPS=60
STEPS = int(os.environ.get("NL_RL_STEPS", 30))
EPISODES = int(os.environ.get("NL_RL_EPISODES", 60))
PRETRAIN_STEPS = int(os.environ.get("NL_PRETRAIN_STEPS", 300))


def eval_fixed(name: str, backbone, device, evA, evB) -> dict:
    """Run the fixed-schedule (no policy) CL protocol for a baseline row."""
    model = RetrofitModel.build(backbone, schedule_name=("uniform" if name == "frozen" else name),
                                min_period=1, max_period=64, num_heads=4, lr=0.15,
                                momentum=0.9, forget=0.02, surprise_trigger=False).to(device)
    model.reset_memory(reset_concepts=True)
    adapt = name != "frozen"
    if adapt:
        for t in range(60):
            model(make_eeg_batch(16, C, T, BAND_A, seed=1000 + t, device=device), adapt=True)
    errA_afterA = model.predictive_error(evA)
    if adapt:
        for t in range(60):
            model(make_eeg_batch(16, C, T, BAND_B, seed=5000 + t, device=device), adapt=True)
    return {"schedule": name, "forgetting": model.predictive_error(evA) - errA_afterA,
            "errA_afterB": model.predictive_error(evA), "errB_afterB": model.predictive_error(evB)}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    torch.manual_seed(0)
    backbone = EEGTransformer(C, T, P, dim=DIM, depth=DEPTH, num_heads=4)
    print("Pretraining backbone...")
    pretrain_masked(backbone, lambda s: make_eeg_batch(32, C, T, PRETRAIN_BANDS, seed=s, device=device),
                    steps=PRETRAIN_STEPS, lr=1e-3, device=device, log_every=150)

    evA = make_eeg_batch(64, C, T, BAND_A, seed=90001, device=device)
    evB = make_eeg_batch(64, C, T, BAND_B, seed=90002, device=device)
    taskA = lambda t: make_eeg_batch(16, C, T, BAND_A, seed=1000 + t, device=device)  # noqa: E731
    taskB = lambda t: make_eeg_batch(16, C, T, BAND_B, seed=5000 + t, device=device)  # noqa: E731

    # --- train the RL policy on the frozen pretrained backbone ---
    model = RetrofitModel.build(backbone, schedule_name="uniform", min_period=1, max_period=64,
                                num_heads=4, lr=0.15, momentum=0.9, forget=0.02,
                                surprise_trigger=False).to(device)
    torch.manual_seed(1)
    controller = PolicyController(feat_dim=4).to(device)
    trainer = ReinforceTrainer(controller, lr=2e-2, update_cost=0.05)
    print(f"\nTraining RL controller ({EPISODES} episodes)...")
    trainer.train(model, taskA, taskB, evA, evB, steps=STEPS, episodes=EPISODES, log_every=10)

    # deterministic rollout of the learned policy -> metrics + per-block profile
    ep = trainer.run_episode(model, taskA, taskB, evA, evB, steps=STEPS, deterministic=True)
    profile = [b["update_rate"] for b in model.adaptation_report()["per_block"]]

    # --- compare against fixed schedules ---
    rows = [eval_fixed(n, backbone, device, evA, evB)
            for n in ("frozen", "workspace", "uniform", "edge_fast")]
    rows.append({"schedule": "RL-policy", "forgetting": ep["forgetting"],
                 "errA_afterB": ep["errA_afterB"], "errB_afterB": ep["errB_afterB"]})

    hdr = f"\n{'schedule':<11} {'forgetting':>11} {'errA|afterB':>11} {'errB|afterB':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['schedule']:<11} {r['forgetting']:>11.4f} {r['errA_afterB']:>11.4f} "
              f"{r['errB_afterB']:>11.4f}")

    print("\nPer-block update rate the policy DISCOVERED (block 0=first ... last=output):")
    peak = max(profile) if max(profile) > 0 else 1.0
    for i, r in enumerate(profile):
        print(f"  block {i:2d}  rate={r:.2f}  |{'#' * int(round(30 * r / peak))}")
    mid = profile[len(profile) // 2]
    ends = 0.5 * (profile[0] + profile[-1])
    print(f"\nmiddle rate={mid:.2f} vs mean-of-ends={ends:.2f} -> "
          f"{'policy concentrates plasticity in the MIDDLE (workspace-like)' if mid > ends else 'policy did NOT prefer the middle'}.")
    print("Objective: lower 'forgetting' while keeping errB low. The RL row shows what a "
          "learned\nplasticity allocation achieves vs the hand-designed schedules on the "
          "same 2-task stream.")


if __name__ == "__main__":
    main()
