"""
Demo 1 — the update-frequency profile is U-shaped (fast middle = the workspace)
================================================================================

What this shows
---------------
We retrofit a frozen 9-block transformer, stream random data through it with
online adaptation on, and then print how many times each block's memory actually
updated. With the ``"workspace"`` schedule the *middle* blocks update far more
often than the ends — plasticity is concentrated where Anthropic's J-space
global workspace lives. We contrast it with ``"uniform"`` and ``"shallow_fast"``.

Run:
    python examples_pytorch/demo_multifrequency.py
Runs on GPU automatically if one is present, otherwise CPU.
"""

import torch

from nested_torch import ReferenceTransformer, RetrofitModel, build_schedule


def run_schedule(name: str, depth: int, steps: int, device: str) -> None:
    torch.manual_seed(0)
    backbone = ReferenceTransformer(in_dim=32, dim=64, depth=depth, num_heads=4,
                                    max_len=32).to(device)
    model = RetrofitModel.build(
        backbone, schedule_name=name, min_period=1, max_period=16,
        num_heads=4, lr=0.1, surprise_trigger=True, trigger_threshold=0.75,
    ).to(device)

    print(f"\n=== schedule: {name} ===")
    print(model.retrofit.schedule.summary())

    for _ in range(steps):
        x = torch.randn(8, 16, 32, device=device)
        model(x, adapt=True)

    report = model.adaptation_report()
    print(f"  after {steps} steps (workspace blocks = {report['workspace_blocks']}):")
    for b in report["per_block"]:
        bar = "#" * int(round(30 * b["update_rate"]))
        print(f"    block {b['block']:2d}  period={b['period']:3d}  "
              f"updates={b['updates']:4d}  rate={b['update_rate']:.2f}  "
              f"mem|W|={b['memory_norm']:.3f}  |{bar}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    depth, steps = 9, 200
    for name in ("workspace", "uniform", "shallow_fast"):
        run_schedule(name, depth, steps, device)

    print("\nTakeaway: with 'workspace', update counts peak in the MIDDLE blocks; "
          "\n'uniform' updates everywhere; 'shallow_fast' front-loads updates.")


if __name__ == "__main__":
    main()
