"""
Demo 3 — the J-space test, now on a PRETRAINED EEG backbone
===========================================================

`cl_forgetting.py` ran on a *random* transformer, where the middle is not yet a
"workspace", so `workspace` ~ `edge_fast`. Here we first **pretrain** a small
EEG-shaped transformer (masked patch reconstruction on oscillatory EEG-like
signals), *then* freeze it and run the same continual-learning comparison. This
is the setting where the middle-beats-ends bet can actually be tested: if
pretraining makes the middle a genuine integration hub, concentrating plasticity
there (`workspace`) should protect the old task better than concentrating it at
the ends (`edge_fast`) or everywhere (`uniform`).

Two tasks = two disjoint EEG frequency bands (theta/alpha vs beta/gamma), which
are genuinely different distributions.

Run:
    python examples_pytorch/cl_forgetting_pretrained.py
Honest note: tiny synthetic model on CPU — a mechanism check, not a benchmark.
"""

import torch

from nested_torch import EEGTransformer, RetrofitModel, pretrain_masked
from nested_torch.toy_data import make_eeg_batch

C, T, P, DIM, DEPTH = 8, 128, 16, 64, 9
BAND_A = [4, 6, 8, 10]      # theta / alpha
BAND_B = [18, 22, 26, 30]   # beta / low-gamma
PRETRAIN_BANDS = [4, 6, 8, 10, 14, 18, 22, 26, 30]  # rich mixture


def pretrain_backbone(device: str) -> EEGTransformer:
    torch.manual_seed(0)
    bb = EEGTransformer(n_channels=C, n_times=T, patch_len=P, dim=DIM, depth=DEPTH, num_heads=4)
    print("Pretraining backbone (masked patch reconstruction)...")
    pretrain_masked(bb, lambda s: make_eeg_batch(32, C, T, PRETRAIN_BANDS, seed=s, device=device),
                    steps=500, lr=1e-3, device=device, log_every=100)
    return bb


def run(name: str, adapt: bool, backbone: EEGTransformer, device: str, evA, evB) -> dict:
    # Fresh retrofit on the SAME pretrained (frozen) backbone for each schedule.
    model = RetrofitModel.build(backbone, schedule_name=("uniform" if name == "frozen" else name),
                                min_period=1, max_period=64, num_heads=4, lr=0.15,
                                momentum=0.9, forget=0.02, surprise_trigger=False).to(device)
    model.reset_memory(reset_concepts=True)

    def stream(bands, seed0):
        if adapt:
            for t in range(60):
                model(make_eeg_batch(16, C, T, bands, seed=seed0 + t, device=device), adapt=True)

    stream(BAND_A, 1000)
    errA_afterA = model.predictive_error(evA)
    stream(BAND_B, 5000)
    errA_afterB = model.predictive_error(evA)
    errB_afterB = model.predictive_error(evB)
    return {"schedule": name, "errA_afterA": errA_afterA, "errA_afterB": errA_afterB,
            "forgetting": errA_afterB - errA_afterA, "errB_afterB": errB_afterB}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    backbone = pretrain_backbone(device)

    evA = make_eeg_batch(64, C, T, BAND_A, seed=90001, device=device)
    evB = make_eeg_batch(64, C, T, BAND_B, seed=90002, device=device)

    rows = [run("frozen", False, backbone, device, evA, evB)]
    for name in ("workspace", "uniform", "edge_fast"):
        rows.append(run(name, True, backbone, device, evA, evB))

    hdr = f"\n{'schedule':<11} {'errA|afterA':>11} {'errA|afterB':>11} {'forgetting':>11} {'errB|afterB':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['schedule']:<11} {r['errA_afterA']:>11.4f} {r['errA_afterB']:>11.4f} "
              f"{r['forgetting']:>11.4f} {r['errB_afterB']:>11.4f}")

    adapt_rows = [r for r in rows if r["schedule"] != "frozen"]
    ws = next(r for r in adapt_rows if r["schedule"] == "workspace")
    edge = next(r for r in adapt_rows if r["schedule"] == "edge_fast")
    uni = next(r for r in adapt_rows if r["schedule"] == "uniform")
    print(f"\nforgetting:  workspace={ws['forgetting']:+.4f}  "
          f"edge_fast={edge['forgetting']:+.4f}  uniform={uni['forgetting']:+.4f}")
    verdict = ("workspace < edge_fast: the middle-beats-ends (J-space) bet HOLDS here."
               if ws["forgetting"] < edge["forgetting"] else
               "workspace >= edge_fast: J-space bet NOT supported at this scale/data — "
               "expected, since synthetic pretraining may not induce a real middle workspace; "
               "revisit with real EEG-FM weights.")
    print("Verdict:", verdict)


if __name__ == "__main__":
    main()
