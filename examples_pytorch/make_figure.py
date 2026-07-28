"""
Generate the showcase figure: results/nested_torch_poc.png

Two panels, both produced live from the code:
  (left)  measured per-block update rate for three schedules -> the "workspace"
          schedule concentrates updates in the MIDDLE blocks (the J-space bet).
  (right) catastrophic forgetting by schedule on a pretrained EEG backbone ->
          fast-everywhere (`uniform`) forgets badly; concentrating plasticity
          (workspace / edge) does not.

Run:
    pip install -e .[viz]        # or: pip install matplotlib
    python examples_pytorch/make_figure.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from nested_torch import ReferenceTransformer, EEGTransformer, RetrofitModel, pretrain_masked  # noqa: E402
from nested_torch.toy_data import make_eeg_batch  # noqa: E402

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", "nested_torch_poc.png"))
BLUE, TEAL, RED, GREY = "#2563eb", "#0d9488", "#dc2626", "#9ca3af"


def measure_update_rates(schedule: str, depth: int = 9, steps: int = 150, device: str = "cpu"):
    torch.manual_seed(0)
    bb = ReferenceTransformer(in_dim=32, dim=64, depth=depth, num_heads=4, max_len=24).to(device)
    model = RetrofitModel.build(bb, schedule_name=schedule, min_period=1, max_period=16,
                                num_heads=4, lr=0.1, surprise_trigger=False).to(device)
    model.reset_memory(reset_concepts=True)
    for _ in range(steps):
        model(torch.randn(8, 16, 32, device=device), adapt=True)
    return [b["update_rate"] for b in model.adaptation_report()["per_block"]]


def measure_forgetting(device: str = "cpu"):
    C, T, P, DIM, DEPTH = 8, 128, 16, 48, 9
    band_a, band_b = [4, 6, 8, 10], [18, 22, 26, 30]
    pre = [4, 6, 8, 10, 14, 18, 22, 26, 30]
    torch.manual_seed(0)
    bb = EEGTransformer(C, T, P, dim=DIM, depth=DEPTH, num_heads=4)
    pretrain_masked(bb, lambda s: make_eeg_batch(32, C, T, pre, seed=s, device=device),
                    steps=200, lr=1e-3, device=device, log_every=0)
    ev_a = make_eeg_batch(64, C, T, band_a, seed=90001, device=device)
    ev_b = make_eeg_batch(64, C, T, band_b, seed=90002, device=device)
    out = {}
    for name in ("frozen", "workspace", "edge_fast", "uniform"):
        model = RetrofitModel.build(bb, schedule_name=("uniform" if name == "frozen" else name),
                                    min_period=1, max_period=64, num_heads=4, lr=0.15,
                                    momentum=0.9, forget=0.02, surprise_trigger=False).to(device)
        model.reset_memory(reset_concepts=True)
        adapt = name != "frozen"
        if adapt:
            for t in range(50):
                model(make_eeg_batch(16, C, T, band_a, seed=1000 + t, device=device), adapt=True)
        err_a = model.predictive_error(ev_a)
        if adapt:
            for t in range(50):
                model(make_eeg_batch(16, C, T, band_b, seed=5000 + t, device=device), adapt=True)
        out[name] = model.predictive_error(ev_a) - err_a
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} — measuring update profiles...")
    schedules = ["workspace", "uniform", "shallow_fast"]
    rates = {s: measure_update_rates(s, device=device) for s in schedules}
    print("measuring forgetting (includes a short backbone pretraining)...")
    forg = measure_forgetting(device=device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = {"workspace": BLUE, "uniform": RED, "shallow_fast": TEAL}
    x = list(range(len(rates["workspace"])))
    for s in schedules:
        ax1.plot(x, rates[s], marker="o", lw=2, color=colors[s], label=s)
    ax1.set_title("Where plasticity lives\n(measured update rate per block)", fontweight="bold")
    ax1.set_xlabel("transformer block  (0 = input … last = output)")
    ax1.set_ylabel("fraction of steps the block updated")
    ax1.set_ylim(-0.05, 1.08)
    ax1.grid(alpha=0.3)
    ax1.legend(title="schedule", loc="upper right")
    mid = len(x) // 2
    ax1.annotate("workspace = fast MIDDLE\n(the J-space global workspace)",
                 xy=(mid, rates["workspace"][mid]), xytext=(mid - 3.4, 0.55),
                 fontsize=9, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))

    names = ["frozen", "workspace", "edge_fast", "uniform"]
    vals = [forg[n] for n in names]
    ax2.bar(names, vals, color=[GREY, BLUE, TEAL, RED])
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_title("Catastrophic forgetting by schedule\n(frozen pretrained EEG backbone)", fontweight="bold")
    ax2.set_ylabel("rise in old-task error after\nlearning the new task  (lower = better)")
    ax2.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax2.text(i, v + (0.03 * max(vals) if v >= 0 else -0.03 * max(vals)),
                 f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax2.annotate("fast-everywhere\nforgets catastrophically", xy=(3, vals[3]),
                 xytext=(1.2, vals[3] * 0.7), fontsize=9, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))

    fig.suptitle("Nested-Learning retrofit: concentrating plasticity in a frozen model resists catastrophic forgetting",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
