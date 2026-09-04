"""Configuration: paper defaults, validation, and the Hydra bridge."""

import pytest

from msa_arc.config import (
    DecodeConfig,
    ExperimentConfig,
    LossConfig,
    MCVAConfig,
    ModelConfig,
    TrainConfig,
    experiment_from_mapping,
)
from msa_arc.constants import PRIMARY_SEED, SEEDS


def test_defaults_match_the_reported_hyperparameters() -> None:
    train = TrainConfig()
    assert train.learning_rate == 1e-5
    assert train.batch_size == 16
    assert train.max_epochs == 50
    assert train.weight_decay == 0.01
    assert train.grad_clip == 1.0
    assert train.early_stopping_patience == 5
    assert train.warmup_ratio == 0.1

    model = ModelConfig()
    assert model.hidden_dim == 768
    assert model.bottleneck_dim == model.hidden_dim // 4
    assert len(model.adapter_layers) == 6
    assert model.dropout == 0.1
    assert model.freeze_backbone

    loss = LossConfig()
    assert (loss.alpha, loss.beta, loss.margin) == (0.7, 0.3, 1.0)

    mcva = MCVAConfig()
    assert (mcva.tau_rev, mcva.tau_flag) == (0.7, 0.3)


def test_the_ten_seeds_and_primary_run_are_pinned() -> None:
    assert SEEDS == (17, 29, 43, 61, 79, 97, 113, 137, 163, 191)
    assert SEEDS[0] == PRIMARY_SEED
    assert len(SEEDS) == 10


def test_configs_are_immutable() -> None:
    from dataclasses import FrozenInstanceError

    cfg = TrainConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.learning_rate = 1e-3  # type: ignore[misc]


def test_beam_search_is_the_primary_decode() -> None:
    assert DecodeConfig().num_beams > 1


def test_negative_adapter_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ModelConfig(adapter_layers=(-1, 2))


def test_mapping_bridge_builds_nested_configs() -> None:
    cfg = experiment_from_mapping(
        {
            "output_dir": "outputs/run",
            "model": {"adapter_layers": [0, 1, 2], "bottleneck_dim": 64},
            "loss": {"alpha": 0.6, "beta": 0.4},
            "seeds": [17, 29],
            "features": {"audio": {"n_mels": 80}},
        }
    )
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.output_dir == "outputs/run"
    # Lists become tuples so a frozen config is not mutable in practice.
    assert cfg.model.adapter_layers == (0, 1, 2)
    assert cfg.loss.alpha == 0.6
    assert cfg.seeds == [17, 29]
    assert cfg.features.audio.n_mels == 80
    # Unspecified sections keep the paper defaults.
    assert cfg.train.learning_rate == 1e-5
    assert cfg.mcva.tau_rev == 0.7


def test_mapping_bridge_rejects_a_misspelled_key() -> None:
    """A typo in a YAML override must fail rather than be silently ignored."""
    with pytest.raises(ValueError, match="has no field"):
        experiment_from_mapping({"train": {"learnign_rate": 1e-4}})


def test_empty_mapping_yields_the_paper_configuration() -> None:
    cfg = experiment_from_mapping({})
    assert cfg.model.adapter_layers == (6, 7, 8, 9, 10, 11)
    assert cfg.seeds is None
