"""
nested_torch — PyTorch, GPU-ready retrofit of a frozen transformer with
multi-frequency Nested-Learning memories on a J-space "global workspace" schedule.

Quick start
-----------
    import torch
    from nested_torch import ReferenceTransformer, RetrofitModel, build_schedule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = ReferenceTransformer(in_dim=32, dim=64, depth=8).to(device)

    # Retrofit: middle blocks are the fast "global workspace", ends stay slow.
    model = RetrofitModel.build(backbone, schedule_name="workspace",
                                min_period=1, max_period=32).to(device)

    x = torch.randn(8, 16, 32, device=device)
    y = model(x, adapt=True)          # online adaptation via the delta rule
    print(model.adaptation_report())  # middle blocks updated most often

To retrofit a real EEG foundation model, pass its transformer blocks:
    from nested_torch import FrozenRetrofit
    retrofit = FrozenRetrofit(eeg_model.blocks, dim=eeg_model.dim,
                              schedule_name="workspace")
"""

from .frequency_schedule import FrequencySchedule, build_schedule, SCHEDULES
from .surprise import RunningStandardizer, ConceptCodebook, token_surprise, combine_surprise
from .fast_weights import BlockMemory
from .frozen_retrofit import FrozenRetrofit
from .reference_transformer import TransformerBlock, ReferenceTransformer, RetrofitModel
from .eeg_backbone import EEGTransformer, EEGPatchEmbed, load_pretrained
from .pretrain import pretrain_masked, save_backbone, load_backbone
from .rl_controller import PolicyController, ReinforceTrainer

__all__ = [
    "FrequencySchedule", "build_schedule", "SCHEDULES",
    "RunningStandardizer", "ConceptCodebook", "token_surprise", "combine_surprise",
    "BlockMemory",
    "FrozenRetrofit",
    "TransformerBlock", "ReferenceTransformer", "RetrofitModel",
    "EEGTransformer", "EEGPatchEmbed", "load_pretrained",
    "pretrain_masked", "save_backbone", "load_backbone",
    "PolicyController", "ReinforceTrainer",
]
