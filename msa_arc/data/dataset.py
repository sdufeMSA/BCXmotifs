"""Stage B dataset: cached feature tensors to model inputs.

Training never touches raw media.  It reads the ``.npy`` arrays Stage A wrote,
which is what makes the released artefact reproducible without the recordings.

A missing audio or video file is not an error.  Instances where no face was
detected, or where the audio channel failed, are represented by an empty
sequence and a zero length; the model's branches then contribute a zero vector,
which is the same thing the ablation configurations do deliberately.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from msa_arc.constants import ATTITUDE_TO_INDEX
from msa_arc.model.target import serialize_target

logger = logging.getLogger(__name__)


class MSAARCDataset(Dataset):
    """One (participant, service, scenario) instance per item.

    Args:
        manifest: Manifest rows for this split, carrying an ``instance_key``
            column.
        feature_dir: Root of the Stage A feature store.
        load_audio: Whether to read the audio tensors.
        load_video: Whether to read the video tensors.
        require_labels: Whether every row must carry the three human labels.
            Set for training and evaluation, cleared for inference over the
            unlabelled participants.
        audio_dim: Feature width used to build the empty array for a missing
            audio instance.
        video_dim: Same, for video.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_dir: Union[str, Path],
        load_audio: bool = True,
        load_video: bool = True,
        require_labels: bool = True,
        audio_dim: int = 40,
        video_dim: int = 2048,
    ) -> None:
        self.frame = manifest.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        self.load_audio = load_audio
        self.load_video = load_video
        self.require_labels = require_labels
        self.audio_dim = audio_dim
        self.video_dim = video_dim

        if require_labels:
            self._assert_labelled()
        self._warn_missing_features()

    def _assert_labelled(self) -> None:
        missing = self.frame["label_category"].isna()
        if bool(missing.any()):
            examples = self.frame.loc[missing, "instance_key"].head(5).tolist()
            raise ValueError(
                f"{int(missing.sum())} rows lack labels but require_labels is set; "
                f"first few: {examples}"
            )

    def _warn_missing_features(self) -> None:
        """Report absent tensors once, at construction, rather than per batch."""
        counts = {"text": 0, "audio": 0, "video": 0}
        for key in self.frame["instance_key"]:
            for modality in counts:
                if modality == "audio" and not self.load_audio:
                    continue
                if modality == "video" and not self.load_video:
                    continue
                if not self._feature_path(modality, key).exists():
                    counts[modality] += 1
        if counts["text"]:
            raise FileNotFoundError(
                f"{counts['text']} instances have no text features under "
                f"{self.feature_dir / 'text'}; run scripts/prepare_features.py first"
            )
        for modality in ("audio", "video"):
            if counts[modality]:
                logger.warning(
                    "%d of %d instances have no %s features; those branches "
                    "receive a zero vector for them",
                    counts[modality],
                    len(self.frame),
                    modality,
                )

    def _feature_path(self, modality: str, key: str) -> Path:
        return self.feature_dir / modality / f"{key}.npy"

    def _load_sequence(self, modality: str, key: str, dim: int) -> np.ndarray:
        path = self._feature_path(modality, key)
        if not path.exists():
            return np.zeros((0, dim), dtype=np.float32)
        return np.load(path).astype(np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Assemble one instance.

        Returns:
            A dict with token ids and mask, the two feature sequences, the
            serialised target string when labels are present, and the
            identifiers needed to trace a prediction back to its interview.
        """
        row = self.frame.iloc[index]
        key = row["instance_key"]

        text = np.load(self._feature_path("text", key))
        item: Dict[str, Any] = {
            "instance_key": key,
            "participant_id": str(row["participant_id"]),
            "service_id": str(row["service_id"]),
            "scenario": int(row["scenario"]),
            "input_ids": text[0].astype(np.int64),
            "attention_mask": text[1].astype(np.int64),
            "audio": (
                self._load_sequence("audio", key, self.audio_dim)
                if self.load_audio
                else np.zeros((0, self.audio_dim), dtype=np.float32)
            ),
            "video": (
                self._load_sequence("video", key, self.video_dim)
                if self.load_video
                else np.zeros((0, self.video_dim), dtype=np.float32)
            ),
            "divergence_pattern": (
                None
                if pd.isna(row.get("divergence_pattern"))
                else str(row["divergence_pattern"])
            ),
        }

        if pd.notna(row.get("label_category")):
            item["label_polarity"] = str(row["label_polarity"])
            item["label_intensity"] = float(row["label_intensity"])
            item["label_category"] = str(row["label_category"])
            item["label_category_index"] = ATTITUDE_TO_INDEX[str(row["label_category"])]
            item["target_text"] = serialize_target(
                str(row["label_polarity"]),
                float(row["label_intensity"]),
                str(row["label_category"]),
            )
        return item


__all__ = ["MSAARCDataset"]
