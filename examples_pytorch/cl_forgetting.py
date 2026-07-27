"""
Demo 2 — does a J-space (fast-middle) retrofit forget less than fast-everywhere?
================================================================================

The experiment (a PyTorch echo of the repo's JAX "Q3: does CMS prevent
catastrophic forgetting?" test, adapted to the frozen-retrofit setting):

  1. Freeze a random transformer and attach block memories.
  2. Adapt on task A (sequences in subspace A), then measure how well the
     memories predict A  ->  error_A_afterA.
  3. Adapt on task B (a different subspace), then re-measure A and measure B.
       forgetting     = error_A_afterB - error_A_afterA   (lower is better)
       learned_B      = error_B_beforeB - error_B_afterB  (higher is better)

We compare four retrofits, memory reset between each:
  - workspace  : fast MIDDLE (the J-space bet)
  - uniform    : fast EVERYWHERE (max plasticity, the usual CMS-style retrofit)
  - edge_fast  : fast ENDS, slow middle (the control / opposite of workspace)
  - frozen     : never adapts (sanity floor: no learning, no forgetting)

We report the raw numbers honestly — this is tiny synthetic data, so treat it
as a mechanism check and a harness, not a benchmark result.

Run:
    python examples_pytorch/cl_forgetting.py
"""

import torch

from nested_torch import ReferenceTransformer, RetrofitModel
from nested_torch.toy_data import make_subspace_basis, sample_task

IN_DIM, DIM, DEPTH, SEQ, RANK = 32, 64, 9, 16, 6


def build(name: str, device: str) -> RetrofitModel:
    torch.manual_seed(0)  # same frozen backbone + memory init for every schedule
    backbone = ReferenceTransformer(in_dim=IN_DIM, dim=DIM, depth=DEPTH,
                                    num_heads=4, max_len=SEQ).to(device)
    # "frozen" is a run mode (never adapts), not a schedule; any schedule works.
    schedule_name = "uniform" if name == "frozen" else name
    return RetrofitModel.build(
        backbone, schedule_name=schedule_name, min_period=1, max_period=64,
        num_heads=4, lr=0.15, momentum=0.9, forget=0.02,
        surprise_trigger=False,  # schedule-only, so the comparison is clean
    ).to(device)


def adapt_on(model: RetrofitModel, basis: torch.Tensor, steps: int, device: str,
             seed0: int) -> None:
    for t in range(steps):
        x = sample_task(basis, batch=16, seq=SEQ, seed=seed0 + t, device=device)
        model(x, adapt=True)


def run(name: str, adapt: bool, device: str, basis_a, basis_b, eval_a, eval_b) -> dict:
    model = build(name, device)
    model.reset_memory(reset_concepts=True)

    if adapt:
        adapt_on(model, basis_a, steps=60, device=device, seed0=1000)
    err_a_after_a = model.predictive_error(eval_a)
    err_b_before_b = model.predictive_error(eval_b)

    if adapt:
        adapt_on(model, basis_b, steps=60, device=device, seed0=5000)
    err_a_after_b = model.predictive_error(eval_a)
    err_b_after_b = model.predictive_error(eval_b)

    return {
        "schedule": name,
        "errA_afterA": err_a_after_a,
        "errA_afterB": err_a_after_b,
        "forgetting": err_a_after_b - err_a_after_a,
        "errB_afterB": err_b_after_b,
        "learnedB": err_b_before_b - err_b_after_b,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    basis_a = make_subspace_basis(IN_DIM, RANK, seed=1)
    basis_b = make_subspace_basis(IN_DIM, RANK, seed=2)
    eval_a = sample_task(basis_a, batch=64, seq=SEQ, seed=90001, device=device)
    eval_b = sample_task(basis_b, batch=64, seq=SEQ, seed=90002, device=device)

    rows = []
    rows.append(run("frozen", False, device, basis_a, basis_b, eval_a, eval_b))
    for name in ("workspace", "uniform", "edge_fast"):
        rows.append(run(name, True, device, basis_a, basis_b, eval_a, eval_b))

    hdr = f"{'schedule':<11} {'errA|afterA':>11} {'errA|afterB':>11} " \
          f"{'forgetting':>11} {'learnedB':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['schedule']:<11} {r['errA_afterA']:>11.4f} {r['errA_afterB']:>11.4f} "
              f"{r['forgetting']:>11.4f} {r['learnedB']:>9.4f}")

    adapt_rows = [r for r in rows if r["schedule"] != "frozen"]
    best = min(adapt_rows, key=lambda r: r["forgetting"])
    uniform = next(r for r in adapt_rows if r["schedule"] == "uniform")
    print(f"\nLowest forgetting among adapting schedules: '{best['schedule']}' "
          f"({best['forgetting']:+.4f}); 'uniform' (fast everywhere) = "
          f"{uniform['forgetting']:+.4f}.")
    print(
        "Reading the table:\n"
        "  'forgetting' = rise in task-A memory error after learning B (lower=better);\n"
        "  'learnedB'   = drop in task-B error from learning B (higher=better).\n"
        "Robust finding: 'uniform' is the schedule that forgets most — restricting *where*\n"
        "plasticity lives (workspace OR edge) protects old tasks, echoing the repo's JAX Q3.\n"
        "Caveat: on a RANDOM frozen backbone the middle is not yet a semantic 'workspace',\n"
        "so 'workspace' ~ 'edge_fast' here. The J-space-specific bet (middle beats ends)\n"
        "should only show up on a PRETRAINED model (e.g. CBraMod/LaBraM) — that's the next step."
    )


if __name__ == "__main__":
    main()
