"""
Tests for nested_torch. Runs either under pytest or as a plain script:

    python tests_pytorch/test_nested_torch.py     # prints PASS/FAIL per test
    pytest tests_pytorch/test_nested_torch.py

The schedule tests need no torch; the rest exercise the memory / retrofit and
run on GPU automatically when one is available.
"""

import sys

from nested_torch.frequency_schedule import build_schedule


# ---- schedule tests (no torch needed) ---------------------------------
def test_workspace_peaks_in_middle():
    sch = build_schedule(9, name="workspace", min_period=1, max_period=32)
    peak = max(range(9), key=lambda i: sch.plasticity[i])
    assert peak == 4, f"expected peak at middle block 4, got {peak}"
    assert sch.periods[4] == 1, "middle block should be the fastest (period 1)"
    assert sch.periods[0] > sch.periods[4] and sch.periods[8] > sch.periods[4]
    ws = sch.workspace_layers()
    assert 4 in ws and 0 not in ws and 8 not in ws, f"workspace={ws}"


def test_schedule_variants():
    assert len(set(build_schedule(6, "uniform").periods)) == 1
    sf = build_schedule(6, "shallow_fast").plasticity
    assert sf[0] > sf[-1], "shallow_fast should be most plastic early"
    ef = build_schedule(9, "edge_fast").plasticity
    assert ef[0] > ef[4] and ef[8] > ef[4], "edge_fast should be plastic at ends"


def test_schedule_validation():
    for bad in (lambda: build_schedule(0),
                lambda: build_schedule(4, min_period=8, max_period=2),
                lambda: build_schedule(4, name="nope")):
        try:
            bad()
        except (ValueError,):
            continue
        raise AssertionError("expected ValueError")


# ---- torch-dependent tests --------------------------------------------
def _torch():
    import torch
    return torch


def test_memory_starts_as_identity_then_learns():
    torch = _torch()
    from nested_torch import BlockMemory
    torch.manual_seed(0)
    mem = BlockMemory(dim=16, num_heads=2, lr=0.5, momentum=0.9, forget=0.0)
    x = torch.randn(4, 8, 16)

    # Empty memory -> zero correction (frozen model recovered exactly).
    assert torch.allclose(mem.read(x), torch.zeros_like(mem.read(x))), "W=0 must give 0 correction"

    @torch.no_grad()
    def recon(m, z):
        err, _ = m._error_and_key(z)
        return float(err.pow(2).sum(-1).mean())

    before = recon(mem, x)
    concept = torch.zeros(4)
    for _ in range(60):
        mem.step(x, concept, scheduled=True, surprise_trigger=False)
    after = recon(mem, x)
    assert after < 0.5 * before, f"memory failed to learn: {before:.4f} -> {after:.4f}"


def test_retrofit_recovers_frozen_model_when_not_adapting():
    torch = _torch()
    from nested_torch import ReferenceTransformer, RetrofitModel
    torch.manual_seed(0)
    backbone = ReferenceTransformer(in_dim=12, dim=32, depth=6, num_heads=4, max_len=10)
    model = RetrofitModel.build(backbone, schedule_name="workspace")
    model.reset_memory()
    x = torch.randn(3, 10, 12)
    with torch.no_grad():
        ref = backbone(x)
        got = model(x, adapt=False)  # empty memory + no writes
    assert torch.allclose(ref, got, atol=1e-5), "retrofit must equal frozen model when idle"


def test_workspace_updates_concentrate_in_middle():
    torch = _torch()
    from nested_torch import ReferenceTransformer, RetrofitModel
    torch.manual_seed(0)
    backbone = ReferenceTransformer(in_dim=16, dim=32, depth=9, num_heads=4, max_len=12)
    model = RetrofitModel.build(backbone, schedule_name="workspace",
                                min_period=1, max_period=32, surprise_trigger=False)
    model.reset_memory(reset_concepts=True)
    for _ in range(100):
        model(torch.randn(8, 12, 16), adapt=True)
    counts = [b["updates"] for b in model.adaptation_report()["per_block"]]
    assert counts[4] > counts[0] and counts[4] > counts[8], f"counts={counts}"


def test_predictive_error_drops_after_adaptation():
    torch = _torch()
    from nested_torch import ReferenceTransformer, RetrofitModel
    from nested_torch.toy_data import make_subspace_basis, sample_task
    torch.manual_seed(0)
    backbone = ReferenceTransformer(in_dim=24, dim=48, depth=7, num_heads=4, max_len=12)
    model = RetrofitModel.build(backbone, schedule_name="uniform", lr=0.15)
    model.reset_memory(reset_concepts=True)
    basis = make_subspace_basis(24, 6, seed=3)
    ev = sample_task(basis, batch=32, seq=12, seed=111)
    before = model.predictive_error(ev)
    for t in range(60):
        model(sample_task(basis, batch=16, seq=12, seed=200 + t), adapt=True)
    after = model.predictive_error(ev)
    assert after < before, f"adaptation did not reduce error: {before:.4f} -> {after:.4f}"


def test_runs_on_available_device():
    torch = _torch()
    from nested_torch import ReferenceTransformer, RetrofitModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = ReferenceTransformer(in_dim=8, dim=16, depth=4, num_heads=2, max_len=6).to(device)
    model = RetrofitModel.build(backbone, schedule_name="workspace").to(device)
    out = model(torch.randn(2, 6, 8, device=device), adapt=True)
    assert out.device.type == device and out.shape == (2, 6, 8)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
