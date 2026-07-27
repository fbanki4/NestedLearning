"""
Tests for the EEG backbone, self-supervised pretraining, and the RL controller.

    python tests_pytorch/test_rl_and_eeg.py     # PASS/FAIL per test
    pytest tests_pytorch/test_rl_and_eeg.py

Kept tiny so the whole file runs in a few seconds on CPU.
"""

import sys

import torch

from nested_torch import (EEGTransformer, RetrofitModel, pretrain_masked,
                          PolicyController, ReinforceTrainer)
from nested_torch.toy_data import make_eeg_batch


def _tiny_backbone():
    torch.manual_seed(0)
    return EEGTransformer(n_channels=4, n_times=32, patch_len=8, dim=32, depth=5, num_heads=4)


def test_eeg_backbone_shapes():
    bb = _tiny_backbone()
    x = make_eeg_batch(3, 4, 32, freqs=[4, 8], seed=1)
    tokens = bb.embed(x)
    assert tokens.shape == (3, bb.n_tokens, bb.dim), tokens.shape
    assert bb.n_tokens == 4 * (32 // 8)
    out = bb(x)
    assert out.shape == (3, bb.n_tokens, bb.patch_len), out.shape
    assert bb.patch_targets(x).shape == (3, bb.n_tokens, bb.patch_len)


def test_pretrain_reduces_loss():
    bb = _tiny_backbone()
    losses = pretrain_masked(bb, lambda s: make_eeg_batch(16, 4, 32, [4, 8, 12], seed=s),
                             steps=80, lr=2e-3, log_every=0)
    assert min(losses) < losses[0], f"pretraining did not reduce loss: {losses[0]:.3f} -> min {min(losses):.3f}"


def test_retrofit_on_eeg_backbone_adapts():
    bb = _tiny_backbone()
    model = RetrofitModel.build(bb, schedule_name="workspace", lr=0.15)
    model.reset_memory(reset_concepts=True)
    ev = make_eeg_batch(16, 4, 32, [4, 8], seed=5)
    before = model.predictive_error(ev)
    for t in range(30):
        model(make_eeg_batch(12, 4, 32, [4, 8], seed=100 + t), adapt=True)
    after = model.predictive_error(ev)
    assert after < before, f"no adaptation on EEG backbone: {before:.3f} -> {after:.3f}"


def test_policy_controller_act():
    torch.manual_seed(0)
    ctrl = PolicyController(feat_dim=4)
    feats = torch.tensor([0.5, 0.3, 0.2, 0.1])
    p = ctrl.update_prob(feats)
    assert 0.0 < float(p.detach()) < 1.0
    do_update, logp = ctrl.act(feats)
    assert isinstance(do_update, bool)
    assert logp.requires_grad, "log-prob must carry grad for REINFORCE"


def test_reinforce_gradient_flows():
    bb = _tiny_backbone()
    model = RetrofitModel.build(bb, schedule_name="uniform", lr=0.15, surprise_trigger=False)
    ctrl = PolicyController(feat_dim=4)
    trainer = ReinforceTrainer(ctrl, lr=1e-2)
    taskA = lambda t: make_eeg_batch(8, 4, 32, [4, 8], seed=1000 + t)      # noqa: E731
    taskB = lambda t: make_eeg_batch(8, 4, 32, [18, 24], seed=5000 + t)    # noqa: E731
    evA = make_eeg_batch(16, 4, 32, [4, 8], seed=7001)
    evB = make_eeg_batch(16, 4, 32, [18, 24], seed=7002)
    ep = trainer.run_episode(model, taskA, taskB, evA, evB, steps=5)
    assert len(ep["logps"]) == 5 * 5 * 2, f"expected n_blocks*steps*2 logps, got {len(ep['logps'])}"
    loss = -1.0 * torch.stack(ep["logps"]).sum()   # pretend advantage = 1
    trainer.opt.zero_grad()
    loss.backward()
    gn = sum(p.grad.norm().item() for p in ctrl.parameters() if p.grad is not None)
    assert gn > 0, "no gradient reached the policy network"


def test_reinforce_train_runs():
    bb = _tiny_backbone()
    model = RetrofitModel.build(bb, schedule_name="uniform", lr=0.15, surprise_trigger=False)
    ctrl = PolicyController(feat_dim=4)
    trainer = ReinforceTrainer(ctrl, lr=2e-2)
    taskA = lambda t: make_eeg_batch(8, 4, 32, [4, 8], seed=1000 + t)      # noqa: E731
    taskB = lambda t: make_eeg_batch(8, 4, 32, [18, 24], seed=5000 + t)    # noqa: E731
    evA = make_eeg_batch(16, 4, 32, [4, 8], seed=7001)
    evB = make_eeg_batch(16, 4, 32, [18, 24], seed=7002)
    hist = trainer.train(model, taskA, taskB, evA, evB, steps=6, episodes=5, log_every=0)
    assert len(hist) == 5 and all("reward" in h for h in hist)


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
