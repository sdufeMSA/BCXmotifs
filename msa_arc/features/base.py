"""Base class shared by the three feature extractors.

Stage A of the pipeline turns raw interview media into de-identified per-instance
tensors.  It runs wherever the recordings live, and what the study can release
is those outputs, not the recordings.  That is why extraction is a separate
stage from training instead of a step inside the data loader.
"""

import abc
import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor(abc.ABC):
    """Turn one instance's raw input into an array on disk.

    Subclasses implement :meth:`extract`; the caching and file-naming policy is
    shared so that a re-run skips work already done and every modality writes
    into the same predictable layout.
    """

    #: Subdirectory of the feature root this extractor writes into.
    modality: str = "unknown"

    @abc.abstractmethod
    def extract(self, source: Any) -> np.ndarray:
        """Compute the feature array for one instance."""

    def output_path(self, feature_dir: Union[str, Path], key: str) -> Path:
        """Where this instance's array is stored."""
        return Path(feature_dir) / self.modality / f"{key}.npy"

    def extract_to_file(
        self,
        source: Any,
        feature_dir: Union[str, Path],
        key: str,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """Extract and cache one instance.

        Args:
            source: Modality-specific input; see the subclass.
            feature_dir: Root of the feature store.
            key: Instance key from ``msa_arc.features.manifest.instance_key``.
            overwrite: Recompute even when the output already exists.

        Returns:
            The path written, or ``None`` when the instance had no usable input.
        """
        path = self.output_path(feature_dir, key)
        if path.exists() and not overwrite:
            return path

        array = self.extract(source)
        if array is None:
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.ascontiguousarray(array))
        return path


__all__ = ["FeatureExtractor"]
