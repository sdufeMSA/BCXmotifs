"""Parameter accounting and checkpointing for MSA-ARC.

Kept apart from the model itself because these are bookkeeping concerns, not
architecture, and because the counts here are the ones a reader checks against
Appendix A.2.1.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

from msa_arc.config import ModelConfig
from msa_arc.constants import RESNET50_PARAMETERS
from msa_arc.model.adapter import expected_adapter_parameters
from msa_arc.model.branches import expected_lstm_parameters

logger = logging.getLogger(__name__)


def count_parameters(module: nn.Module) -> int:
    """Total parameters held by a module."""
    return sum(p.numel() for p in module.parameters())


def parameter_report(model: nn.Module) -> Dict[str, int]:
    """Break a model's parameter count down the way Appendix A.2.1 does.

    The appendix's 632.1M total includes the ImageNet ResNet-50, which in this
    implementation runs in Stage A and produces the video features offline
    rather than being held by the model. ``total`` is therefore what the model
    holds and ``total_with_stage_a`` is the published figure, so the two are not
    silently 25.6M apart.

    Args:
        model: A :class:`~msa_arc.model.mul_mt5.MulMT5`.

    Returns:
        Counts for adapters, each branch, the trainable total, the frozen
        backbone, the Stage-B total, and the reconciliation with the paper.
    """
    adapters = count_parameters(model.adapters)
    audio = count_parameters(model.audio_branch) if model.audio_branch is not None else 0
    video = count_parameters(model.video_branch) if model.video_branch is not None else 0
    backbone = count_parameters(model.backbone)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = adapters + audio + video + backbone
    return {
        "adapters": adapters,
        "audio_branch": audio,
        "video_branch": video,
        "trainable": trainable,
        "backbone_frozen": backbone,
        "total": total,
        "stage_a_resnet50": RESNET50_PARAMETERS,
        "total_with_stage_a": total + RESNET50_PARAMETERS,
    }


def expected_parameter_report(cfg: ModelConfig) -> Dict[str, int]:
    """Closed-form counts implied by a configuration, for cross-checking.

    Args:
        cfg: The architecture configuration.

    Returns:
        Expected adapter, branch and trainable counts.
    """
    adapters = expected_adapter_parameters(
        cfg.hidden_dim, cfg.bottleneck_dim, len(cfg.adapter_layers)
    )
    audio = (
        expected_lstm_parameters(cfg.audio_input_dim, cfg.hidden_dim, cfg.lstm_layers)
        if cfg.use_audio
        else 0
    )
    video = (
        expected_lstm_parameters(cfg.video_input_dim, cfg.hidden_dim, cfg.lstm_layers)
        if cfg.use_video
        else 0
    )
    return {
        "adapters": adapters,
        "audio_branch": audio,
        "video_branch": video,
        "trainable": adapters + audio + video,
    }


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """The parameters a model actually fits, cloned onto the CPU.

    A full ``state_dict`` would carry the 582M frozen backbone parameters into
    every checkpoint. The backbone is recoverable from its public checkpoint
    name, so only the 24.1M trainable tensors are saved.

    The tensors are cloned, not merely detached. Training keeps the best-so-far
    state in memory while it continues to update the parameters in place, and a
    view onto live storage would silently track the latest weights instead of
    the best ones.

    Args:
        model: The model.

    Returns:
        Cloned CPU copies of the trainable tensors, keyed as in ``state_dict``.
    """
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        name: tensor.detach().clone().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def load_trainable_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load a checkpoint produced by :func:`trainable_state_dict`.

    Args:
        model: The model to load into.
        state: The saved tensors.

    Raises:
        ValueError: If the checkpoint carries keys the model does not define,
            which means it came from a different architecture.
    """
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise ValueError(f"checkpoint contains unknown keys: {unexpected[:5]}")
    logger.info("Loaded %d trainable tensors from checkpoint", len(state))


__all__ = [
    "count_parameters",
    "expected_parameter_report",
    "load_trainable_state_dict",
    "parameter_report",
    "trainable_state_dict",
]
