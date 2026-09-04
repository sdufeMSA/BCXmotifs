"""Immutable configuration objects for the MSA-ARC pipeline.

Every hyperparameter reported in the manuscript has exactly one home here, and
the default value of each field is the value the paper reports.  The dataclasses
are frozen so that a run cannot silently mutate its own configuration; Hydra
composes them from ``configs/`` and they are serialised into every output
directory alongside the results.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

from msa_arc.constants import PRIMARY_SEED


@dataclass(frozen=True)
class TextFeatureConfig:
    """Text branch preprocessing (Section 5.1.1, Appendix A.2.1)."""

    backbone_name: str = "google/mt5-base"
    max_length: int = 128
    use_jieba: bool = True
    #: Semantically empty Mandarin fillers characteristic of hesitant elderly
    #: speech; removed before the backbone's own subword tokenisation.
    discourse_particles: Tuple[str, ...] = (
        "嗯",
        "啊",
        "呃",
        "哦",
        "唉",
        "呀",
        "吧",
        "嘛",
        "呢",
        "哈",
        "诶",
        "唔",
    )
    strip_symbols: bool = True


@dataclass(frozen=True)
class AudioFeatureConfig:
    """Audio branch preprocessing (Section 5.1.1, Appendix A.2.1)."""

    sample_rate: int = 16_000
    n_mels: int = 40
    window_ms: float = 25.0
    hop_ms: float = 10.0
    denoise: bool = True
    max_frames: int = 3_000
    dim: int = 40


@dataclass(frozen=True)
class VideoFeatureConfig:
    """Video branch preprocessing (Section 5.1.1, Appendix A.2.1)."""

    fps: int = 10
    image_size: int = 224
    face_detector: str = "mediapipe"
    backbone_name: str = "resnet50"
    #: ResNet-50 penultimate pooling output.
    dim: int = 2048
    max_frames: int = 300


@dataclass(frozen=True)
class FeatureConfig:
    text: TextFeatureConfig = field(default_factory=TextFeatureConfig)
    audio: AudioFeatureConfig = field(default_factory=AudioFeatureConfig)
    video: VideoFeatureConfig = field(default_factory=VideoFeatureConfig)


@dataclass(frozen=True)
class ModelConfig:
    """Mul-mT5 architecture (Section 4.1.2, Appendix A.2.1).

    Six adapter blocks are inserted after the feed-forward sub-layer of the top
    six encoder layers (0-indexed 6..11 of 12), holding 3,544,704 parameters in
    total.  The text stream is the organising sequence and the decoder
    cross-attends over the encoder output, so injecting the acoustic and visual
    cues on the encoder side conditions everything downstream of it.
    """

    backbone_name: str = "google/mt5-base"
    hidden_dim: int = 768
    bottleneck_dim: int = 192
    adapter_layers: Tuple[int, ...] = (6, 7, 8, 9, 10, 11)
    adapter_activation: str = "gelu"
    dropout: float = 0.1

    audio_input_dim: int = 40
    video_input_dim: int = 2048
    lstm_layers: int = 2

    use_audio: bool = True
    use_video: bool = True

    freeze_backbone: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4")
        if len(set(self.adapter_layers)) != len(self.adapter_layers):
            raise ValueError(f"duplicate adapter layers: {self.adapter_layers}")
        if any(i < 0 for i in self.adapter_layers):
            raise ValueError("adapter layer indices must be non-negative")


@dataclass(frozen=True)
class LossConfig:
    """Combined objective (Section 4.1.3, Appendix A.2.1).

    ``alpha`` and ``beta`` weight generation against contrastive learning.  They
    do *not* weight the three outputs against one another: the relative
    influence of polarity, intensity and category is fixed by their token
    lengths inside the shared target string.
    """

    alpha: float = 0.7
    beta: float = 0.3
    margin: float = 1.0
    #: ``batch_hard`` selects the farthest same-class positive and the nearest
    #: different-class negative in the batch.  Deterministic given the batch,
    #: unlike random triplet sampling, so multi-seed variation reflects
    #: initialisation and batch order alone.
    negative_mining: str = "batch_hard"

    def __post_init__(self) -> None:
        if self.negative_mining not in {"batch_hard", "random"}:
            raise ValueError(f"unknown negative_mining: {self.negative_mining}")


@dataclass(frozen=True)
class DecodeConfig:
    """Decoding, fixed and reported (resolves Gate 2/A2).

    Primary decoding is beam search.  A sequence that fails to parse back into
    the triplet is re-decoded greedily; one that still fails falls back to the
    argmax of the constrained rescoring, which is parseable by construction.
    """

    num_beams: int = 4
    max_new_tokens: int = 16
    length_penalty: float = 1.0
    early_stopping: bool = True
    #: Temperature applied to the length-normalised category log-likelihoods
    #: before the softmax that produces the five-class distribution.
    probability_temperature: float = 1.0


@dataclass(frozen=True)
class MCVAConfig:
    """Reconciliation rule and confidence banding (Eqs. 1-2)."""

    tau_rev: float = 0.7
    tau_flag: float = 0.3
    enabled: bool = True
    #: Low-confidence reclassifications are written to a review queue instead of
    #: being applied automatically.
    auto_accept_bands: Tuple[str, ...] = ("high", "medium")

    def __post_init__(self) -> None:
        if not 0.0 < self.tau_flag < self.tau_rev < 1.0:
            raise ValueError(
                f"require 0 < tau_flag < tau_rev < 1, got "
                f"tau_flag={self.tau_flag}, tau_rev={self.tau_rev}"
            )


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings (Table: msa_arc_params, Appendix A.2.1)."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    max_epochs: int = 50
    warmup_ratio: float = 0.1
    lr_schedule: str = "cosine"
    grad_clip: float = 1.0
    early_stopping_metric: str = "val_macro_f1"
    early_stopping_patience: int = 5
    seed: int = PRIMARY_SEED
    num_workers: int = 4
    device: str = "cuda"
    amp: bool = False


@dataclass(frozen=True)
class DataConfig:
    """Where the manifest and the extracted feature tensors live."""

    manifest_path: str = "data/manifest.csv"
    feature_dir: str = "data/features"
    splits_path: str = "data/splits.csv"
    #: Set when splits are drawn rather than read from ``splits_path``.
    split_seed: int = 20240620
    load_audio: bool = True
    load_video: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration composed by Hydra."""

    output_dir: str = "outputs/msa_arc"
    run_name: str = "msa_arc"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    mcva: MCVAConfig = field(default_factory=MCVAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    #: When set, ``scripts/train.py`` runs once per seed and reports mean +/- SD.
    seeds: Optional[List[int]] = None


_SECTION_TYPES = {
    "data": DataConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "decode": DecodeConfig,
    "mcva": MCVAConfig,
    "train": TrainConfig,
}

_FEATURE_TYPES = {
    "text": TextFeatureConfig,
    "audio": AudioFeatureConfig,
    "video": VideoFeatureConfig,
}


def _coerce(cls, values: Optional[Mapping[str, Any]]):
    """Build one frozen config from a mapping, tolerating absent keys.

    Sequence fields are converted back to tuples: YAML and OmegaConf both hand
    back lists, and a list default would make the dataclass mutable in practice
    even though it is declared frozen.

    Args:
        cls: The dataclass to build.
        values: Field values, or ``None`` to take every default.

    Returns:
        An instance of ``cls``.

    Raises:
        ValueError: If ``values`` contains a key the dataclass does not define.
    """
    if not values:
        return cls()
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"{cls.__name__} has no field(s): {unknown}")

    kwargs: Dict[str, Any] = {}
    for name, value in values.items():
        annotation = known[name].type
        if isinstance(value, (list, tuple)) and "Tuple" in str(annotation):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def experiment_from_mapping(values: Mapping[str, Any]) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from nested plain data.

    This is the bridge from a Hydra/OmegaConf ``DictConfig`` (converted with
    ``OmegaConf.to_container``) to the frozen dataclasses the library uses, and
    it is where an unknown or misspelled config key is caught rather than
    silently ignored.

    Args:
        values: Nested mapping mirroring :class:`ExperimentConfig`.

    Returns:
        The composed configuration.
    """
    features = values.get("features") or {}
    feature_cfg = FeatureConfig(
        **{name: _coerce(cls, features.get(name)) for name, cls in _FEATURE_TYPES.items()}
    )
    sections = {name: _coerce(cls, values.get(name)) for name, cls in _SECTION_TYPES.items()}
    seeds = values.get("seeds")
    return ExperimentConfig(
        output_dir=values.get("output_dir", "outputs/msa_arc"),
        run_name=values.get("run_name", "msa_arc"),
        features=feature_cfg,
        seeds=list(seeds) if seeds else None,
        **sections,
    )


__all__ = [
    "TextFeatureConfig",
    "AudioFeatureConfig",
    "VideoFeatureConfig",
    "FeatureConfig",
    "ModelConfig",
    "LossConfig",
    "DecodeConfig",
    "MCVAConfig",
    "TrainConfig",
    "DataConfig",
    "ExperimentConfig",
    "experiment_from_mapping",
]
