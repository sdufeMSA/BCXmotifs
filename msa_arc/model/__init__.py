"""Model registry and factory for MSA-ARC.

``build_model`` is the only construction path used by the entry points. It
takes an optional pre-built backbone so that tests can inject a small randomly
initialised mT5 and run without downloading the 2.3 GB mT5-base checkpoint.
"""

import logging
from typing import Callable, Dict, Optional, Type

import torch.nn as nn

from msa_arc.config import ModelConfig
from msa_arc.model.accounting import (
    count_parameters,
    expected_parameter_report,
    parameter_report,
)
from msa_arc.model.adapter import CrossModalAdapter, expected_adapter_parameters
from msa_arc.model.branches import LSTMBranch, expected_lstm_parameters
from msa_arc.model.mul_mt5 import MulMT5, MulMT5Output, masked_mean
from msa_arc.model.target import (
    ParseResult,
    all_category_candidates,
    category_prefix,
    parse_target,
    serialize_target,
)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Register a model class under ``name``."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


register_model("mul_mt5")(MulMT5)


def load_backbone(name: str) -> nn.Module:
    """Load a pre-trained mT5 encoder-decoder.

    Args:
        name: A HuggingFace checkpoint identifier, e.g. ``google/mt5-base``.

    Returns:
        The loaded ``MT5ForConditionalGeneration``.
    """
    from transformers import MT5ForConditionalGeneration

    logger.info("Loading backbone %s", name)
    return MT5ForConditionalGeneration.from_pretrained(name)


def build_model(
    cfg: ModelConfig,
    backbone: Optional[nn.Module] = None,
    model_type: str = "mul_mt5",
    check_parameters: bool = True,
) -> MulMT5:
    """Construct the MSA-ARC model.

    Args:
        cfg: Architecture configuration.
        backbone: Pre-loaded backbone. When ``None`` the checkpoint named by
            ``cfg.backbone_name`` is downloaded.
        model_type: Key into ``MODEL_REGISTRY``.
        check_parameters: Whether to verify the fitted parameter count against
            the closed-form figures the manuscript reports. Left on by default:
            a silent mismatch here means the code and the paper have drifted.

    Returns:
        The constructed model.

    Raises:
        ValueError: If ``model_type`` is unknown, or if ``check_parameters`` is
            set and the counts disagree.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {model_type!r}; registered: {sorted(MODEL_REGISTRY)}")
    if backbone is None:
        backbone = load_backbone(cfg.backbone_name)

    model = MODEL_REGISTRY[model_type](cfg, backbone)

    if check_parameters:
        actual = model.parameter_report()
        expected = model.expected_parameter_report()
        for key, value in expected.items():
            if actual[key] != value:
                raise ValueError(
                    f"parameter count mismatch for {key}: model holds {actual[key]:,} "
                    f"but the configuration implies {value:,}"
                )
        logger.info(
            "Trainable parameters: %s (adapters %s, audio %s, video %s); frozen backbone: %s",
            f"{actual['trainable']:,}",
            f"{actual['adapters']:,}",
            f"{actual['audio_branch']:,}",
            f"{actual['video_branch']:,}",
            f"{actual['backbone_frozen']:,}",
        )
    return model


__all__ = [
    "MODEL_REGISTRY",
    "count_parameters",
    "expected_parameter_report",
    "parameter_report",
    "CrossModalAdapter",
    "LSTMBranch",
    "MulMT5",
    "MulMT5Output",
    "ParseResult",
    "all_category_candidates",
    "build_model",
    "category_prefix",
    "expected_adapter_parameters",
    "expected_lstm_parameters",
    "load_backbone",
    "masked_mean",
    "parse_target",
    "register_model",
    "serialize_target",
]
