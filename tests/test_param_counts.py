"""The code must reproduce the parameter accounting the manuscript reports.

Appendix A.2.1 gives 3.5M adapter parameters, 7.2M for the audio branch, 13.4M
for the video branch, 24.1M trainable in total, and a 192.1M language-modelling
head.  Those figures are what settle how many adapters there are and where they
sit, so a drift here means the code and the paper have parted company.
"""

import pytest
import torch

from msa_arc.config import ModelConfig
from msa_arc.model.adapter import CrossModalAdapter, expected_adapter_parameters
from msa_arc.model.branches import LSTMBranch, expected_lstm_parameters

PAPER_ADAPTERS = 3_544_704  # reported as 3.5M
PAPER_AUDIO_BRANCH = 7_213_056  # reported as 7.2M
PAPER_VIDEO_BRANCH = 13_381_632  # reported as 13.4M
PAPER_TRAINABLE = 24_139_392  # reported as 24.1M
PAPER_LM_HEAD = 192_086_016  # reported as 192.1M


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_six_adapters_hold_the_reported_parameters() -> None:
    cfg = ModelConfig()
    assert len(cfg.adapter_layers) == 6
    one_block = CrossModalAdapter(cfg.hidden_dim, cfg.bottleneck_dim)
    assert count(one_block) * 6 == PAPER_ADAPTERS
    assert expected_adapter_parameters(cfg.hidden_dim, cfg.bottleneck_dim, 6) == PAPER_ADAPTERS


def test_audio_branch_holds_the_reported_parameters() -> None:
    cfg = ModelConfig()
    branch = LSTMBranch(cfg.audio_input_dim, cfg.hidden_dim, cfg.lstm_layers)
    assert count(branch) == PAPER_AUDIO_BRANCH
    assert (
        expected_lstm_parameters(cfg.audio_input_dim, cfg.hidden_dim, cfg.lstm_layers)
        == PAPER_AUDIO_BRANCH
    )


def test_video_branch_holds_the_reported_parameters() -> None:
    cfg = ModelConfig()
    branch = LSTMBranch(cfg.video_input_dim, cfg.hidden_dim, cfg.lstm_layers)
    assert count(branch) == PAPER_VIDEO_BRANCH


def test_audio_branch_carries_no_extra_projection() -> None:
    """The appendix's 'linear projection' must be the identity.

    A real 768x768 projection would add 590,592 parameters and break the 7.2M.
    """
    cfg = ModelConfig()
    branch = LSTMBranch(cfg.audio_input_dim, cfg.hidden_dim, cfg.lstm_layers)
    linear_layers = [m for m in branch.modules() if isinstance(m, torch.nn.Linear)]
    assert linear_layers == []


def test_trainable_total_matches_the_paper() -> None:
    assert PAPER_ADAPTERS + PAPER_AUDIO_BRANCH + PAPER_VIDEO_BRANCH == PAPER_TRAINABLE


def test_language_modelling_head_size_matches_the_paper() -> None:
    """mT5 does not tie the head to the input embedding, hence the 192.1M."""
    vocab_size, hidden_dim = 250_112, 768
    assert vocab_size * hidden_dim == PAPER_LM_HEAD


def test_adapter_starts_as_a_no_op(tiny_model_config: ModelConfig) -> None:
    adapter = CrossModalAdapter(tiny_model_config.hidden_dim, tiny_model_config.bottleneck_dim)
    states = torch.randn(2, 5, tiny_model_config.hidden_dim)
    audio = torch.randn(2, tiny_model_config.hidden_dim)
    video = torch.randn(2, tiny_model_config.hidden_dim)
    torch.testing.assert_close(adapter(states, audio, video), states)


def test_build_model_reports_and_verifies_its_own_counts(tiny_model) -> None:
    actual = tiny_model.parameter_report()
    expected = tiny_model.expected_parameter_report()
    for key, value in expected.items():
        assert actual[key] == value
    assert actual["trainable"] < actual["total"]


def test_backbone_is_frozen_and_branches_are_not(tiny_model) -> None:
    assert all(not p.requires_grad for p in tiny_model.backbone.parameters())
    assert all(p.requires_grad for p in tiny_model.adapters.parameters())
    assert all(p.requires_grad for p in tiny_model.audio_branch.parameters())


def test_adapter_layer_index_must_exist(tiny_backbone) -> None:
    from msa_arc.model import build_model

    cfg = ModelConfig(
        backbone_name="stub",
        hidden_dim=32,
        bottleneck_dim=8,
        adapter_layers=(0, 5),
        audio_input_dim=8,
        video_input_dim=16,
    )
    with pytest.raises(ValueError, match="out of range"):
        build_model(cfg, backbone=tiny_backbone)


def test_duplicate_adapter_layers_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate adapter layers"):
        ModelConfig(adapter_layers=(6, 6, 7, 8, 9, 10))


PAPER_TOTAL = 632_097_704  # reported as 632.1M
PAPER_MT5_BASE = 582_401_280
PAPER_RESNET50 = 25_557_032


def test_the_reported_total_reconciles_with_the_stage_a_resnet() -> None:
    """The appendix's 632.1M only closes once the ResNet-50 is counted.

    Stage B holds mT5-base plus the trainable modules, which comes to 606.5M.
    The remaining 25.6M is the ImageNet ResNet-50 that Stage A runs offline.
    """
    from msa_arc.constants import RESNET50_PARAMETERS

    assert RESNET50_PARAMETERS == PAPER_RESNET50
    assert PAPER_MT5_BASE + PAPER_TRAINABLE + PAPER_RESNET50 == PAPER_TOTAL
    assert round(PAPER_TRAINABLE / PAPER_TOTAL * 100, 1) == 3.8


def test_the_video_extractor_drops_the_classifier_head() -> None:
    """``fc`` is replaced by an identity, so 2,049,000 parameters never run."""
    from msa_arc.constants import (
        RESNET50_FEATURE_EXTRACTOR_PARAMETERS,
        RESNET50_PARAMETERS,
    )

    assert RESNET50_PARAMETERS - RESNET50_FEATURE_EXTRACTOR_PARAMETERS == 2_049_000


@pytest.mark.slow
def test_real_mt5_base_reproduces_every_reported_count() -> None:
    """Architecture-only check against the published mT5-base config.

    Marked slow because it reaches the network for the config file; it does not
    download the 2.3 GB weights.
    """
    from transformers import MT5Config, MT5ForConditionalGeneration

    from msa_arc.model import build_model

    backbone = MT5ForConditionalGeneration(MT5Config.from_pretrained("google/mt5-base"))
    model = build_model(ModelConfig(), backbone=backbone)
    report = model.parameter_report()

    assert report["adapters"] == PAPER_ADAPTERS
    assert report["audio_branch"] == PAPER_AUDIO_BRANCH
    assert report["video_branch"] == PAPER_VIDEO_BRANCH
    assert report["trainable"] == PAPER_TRAINABLE
    assert report["backbone_frozen"] == PAPER_MT5_BASE
    assert report["total_with_stage_a"] == PAPER_TOTAL
    assert count(backbone.lm_head) == PAPER_LM_HEAD
    assert round(model.trainable_share() * 100, 1) == 3.8
