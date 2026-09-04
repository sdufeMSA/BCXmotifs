"""The shipped YAML must compose and must agree with the dataclass defaults."""

from pathlib import Path

import pytest

from msa_arc.config import experiment_from_mapping

hydra = pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def build(overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        raw = compose(config_name="config", overrides=overrides or [])
    return experiment_from_mapping(OmegaConf.to_container(raw, resolve=True))


def test_default_composition_reproduces_the_paper_settings() -> None:
    cfg = build()
    assert cfg.model.adapter_layers == (6, 7, 8, 9, 10, 11)
    assert cfg.model.bottleneck_dim == 192
    assert cfg.train.learning_rate == 1e-5
    assert cfg.loss.alpha == 0.7
    assert cfg.mcva.tau_rev == 0.7
    assert cfg.decode.num_beams == 4
    assert cfg.seeds == [17, 29, 43, 61, 79, 97, 113, 137, 163, 191]
    assert cfg.features.audio.n_mels == 40
    assert cfg.features.video.dim == 2048


def test_ablation_configs_switch_off_the_right_branch() -> None:
    assert build(["model=text_only"]).model.use_audio is False
    assert build(["model=text_only"]).model.use_video is False
    assert build(["model=no_audio"]).model.use_audio is False
    assert build(["model=no_audio"]).model.use_video is True
    assert build(["model=no_video"]).model.use_video is False


def test_mcva_can_be_disabled_for_the_ablation_row() -> None:
    assert build(["mcva=disabled"]).mcva.enabled is False
    # Thresholds survive so the disabled run is otherwise identical.
    assert build(["mcva=disabled"]).mcva.tau_rev == 0.7


def test_command_line_overrides_reach_the_dataclasses() -> None:
    cfg = build(["train.seed=29", "seeds=[29]", "mcva.tau_rev=0.8"])
    assert cfg.train.seed == 29
    assert cfg.seeds == [29]
    assert cfg.mcva.tau_rev == 0.8
