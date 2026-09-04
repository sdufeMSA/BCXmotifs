"""Stage A: raw interview media to de-identified per-instance tensors.

The registry lets an entry point ask for an extractor by modality name without
importing the heavy optional dependencies of the modalities it does not need.
"""

import logging
from typing import Callable, Dict, Type

from msa_arc.features.base import FeatureExtractor
from msa_arc.features.manifest import (
    ManifestError,
    ManifestStats,
    instance_key,
    labelled_subset,
    load_manifest,
    media_segment,
    validate_manifest,
)

logger = logging.getLogger(__name__)

EXTRACTOR_REGISTRY: Dict[str, Type[FeatureExtractor]] = {}


def register_extractor(
    name: str,
) -> Callable[[Type[FeatureExtractor]], Type[FeatureExtractor]]:
    """Register a feature extractor under a modality name."""

    def decorator(cls: Type[FeatureExtractor]) -> Type[FeatureExtractor]:
        if name in EXTRACTOR_REGISTRY:
            raise ValueError(f"extractor {name!r} is already registered")
        EXTRACTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_extractor(name: str) -> Type[FeatureExtractor]:
    """Look up an extractor class by modality name.

    Args:
        name: ``text``, ``audio`` or ``video``.

    Returns:
        The extractor class.

    Raises:
        ValueError: If the modality is unknown.
    """
    if name not in EXTRACTOR_REGISTRY:
        _register_builtin()
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(
            f"unknown modality {name!r}; registered: {sorted(EXTRACTOR_REGISTRY)}"
        )
    return EXTRACTOR_REGISTRY[name]


def _register_builtin() -> None:
    """Import and register the three built-in extractors on first use."""
    from msa_arc.features.audio import AudioFeatureExtractor
    from msa_arc.features.text import TextFeatureExtractor
    from msa_arc.features.video import VideoFeatureExtractor

    for name, cls in (
        ("text", TextFeatureExtractor),
        ("audio", AudioFeatureExtractor),
        ("video", VideoFeatureExtractor),
    ):
        EXTRACTOR_REGISTRY.setdefault(name, cls)


__all__ = [
    "EXTRACTOR_REGISTRY",
    "FeatureExtractor",
    "ManifestError",
    "ManifestStats",
    "get_extractor",
    "instance_key",
    "labelled_subset",
    "load_manifest",
    "media_segment",
    "register_extractor",
    "validate_manifest",
]
