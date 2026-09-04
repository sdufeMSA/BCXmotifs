"""Batch assembly.

Text is already padded to a fixed length by Stage A.  The two feature sequences
are ragged, so they are padded to the batch maximum and their true lengths are
carried alongside; the LSTM branches pack on those lengths so padding never
reaches the recurrence.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from msa_arc.losses.generation import IGNORE_INDEX


def pad_sequences(
    sequences: Sequence[np.ndarray], dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of ``(len, dim)`` arrays to the batch maximum.

    Args:
        sequences: Per-instance feature sequences, possibly of length zero.
        dim: Feature width, needed to shape the tensor when every sequence is
            empty.

    Returns:
        ``(batch, max_len, dim)`` padded tensor and ``(batch,)`` true lengths.
    """
    lengths = [int(sequence.shape[0]) for sequence in sequences]
    max_length = max(max(lengths), 1)
    padded = np.zeros((len(sequences), max_length, dim), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        if sequence.shape[0]:
            padded[index, : sequence.shape[0]] = sequence
    return torch.from_numpy(padded), torch.tensor(lengths, dtype=torch.long)


class Collator:
    """Turn a list of dataset items into model-ready tensors.

    Args:
        tokenizer: Tokenizer used to encode the unified target strings.
        audio_dim: Audio feature width.
        video_dim: Video feature width.
        max_target_length: Cap on the tokenised target sequence.
    """

    def __init__(
        self,
        tokenizer: Any,
        audio_dim: int = 40,
        video_dim: int = 2048,
        max_target_length: int = 24,
    ) -> None:
        self.tokenizer = tokenizer
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.max_target_length = max_target_length

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate one batch.

        Returns:
            A dict of tensors plus the identifier lists needed to write
            predictions back out per instance.
        """
        batch: Dict[str, Any] = {
            "input_ids": torch.tensor(
                np.stack([item["input_ids"] for item in items]), dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                np.stack([item["attention_mask"] for item in items]), dtype=torch.long
            ),
            "instance_key": [item["instance_key"] for item in items],
            "participant_id": [item["participant_id"] for item in items],
            "service_id": [item["service_id"] for item in items],
            "scenario": [item["scenario"] for item in items],
            "divergence_pattern": [item["divergence_pattern"] for item in items],
        }

        audio, audio_lengths = pad_sequences([item["audio"] for item in items], self.audio_dim)
        video, video_lengths = pad_sequences([item["video"] for item in items], self.video_dim)
        batch.update(
            audio_features=audio,
            audio_lengths=audio_lengths,
            video_features=video,
            video_lengths=video_lengths,
        )

        if "target_text" in items[0]:
            batch["labels"] = self.encode_targets([item["target_text"] for item in items])
            batch["target_text"] = [item["target_text"] for item in items]
            batch["label_category"] = [item["label_category"] for item in items]
            batch["label_polarity"] = [item["label_polarity"] for item in items]
            batch["label_intensity"] = torch.tensor(
                [item["label_intensity"] for item in items], dtype=torch.float
            )
            batch["label_category_index"] = torch.tensor(
                [item["label_category_index"] for item in items], dtype=torch.long
            )
        return batch

    def encode_targets(self, targets: Sequence[str]) -> torch.Tensor:
        """Tokenise target strings, masking padding out of the loss.

        Args:
            targets: Unified target strings.

        Returns:
            ``(batch, target_len)`` label ids with ``IGNORE_INDEX`` at padding.
        """
        encoded = self.tokenizer(
            list(targets),
            max_length=self.max_target_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"]
        return labels.masked_fill(encoded["attention_mask"] == 0, IGNORE_INDEX)


def move_to_device(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    """Move every tensor in a batch to ``device``, leaving metadata alone."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


__all__ = ["Collator", "move_to_device", "pad_sequences"]
