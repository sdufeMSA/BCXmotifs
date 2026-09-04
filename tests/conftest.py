"""Shared fixtures.

The tests must run offline and in seconds, so neither the 2.3 GB mT5-base
checkpoint nor its SentencePiece model is downloaded.  A character-level stub
tokenizer and a randomly-initialised two-layer mT5 stand in for them; both
expose exactly the interface the library uses, which is what makes the
dependency injection in ``build_model`` and the decode functions worth having.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 |.,-_+:;!?"


class StubTokenizer:
    """Character-level stand-in for the mT5 SentencePiece tokenizer.

    Ids 0, 1 and 2 are pad, eos and unk, matching the roles those ids play in
    the real tokenizer, so code that special-cases them behaves the same way.
    """

    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def __init__(self) -> None:
        self._to_id = {ch: index + 3 for index, ch in enumerate(_CHARSET)}
        self._to_char = {index: ch for ch, index in self._to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._to_id) + 3

    def _encode(self, text: str) -> List[int]:
        return [self._to_id.get(ch, self.unk_token_id) for ch in text] + [self.eos_token_id]

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        max_length: int = 128,
        padding: str = "longest",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [self._encode(t) for t in texts]
        if truncation:
            encoded = [ids[:max_length] for ids in encoded]

        width = max_length if padding == "max_length" else max(len(i) for i in encoded)
        ids = np.zeros((len(encoded), width), dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        for row, sequence in enumerate(encoded):
            ids[row, : len(sequence)] = sequence
            mask[row, : len(sequence)] = 1

        if return_tensors == "np":
            return {"input_ids": ids, "attention_mask": mask}

        import torch

        return {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
        }

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> List[str]:
        """Detokenise, dropping the special ids the way the real tokenizer does."""
        results = []
        for sequence in sequences:
            ids = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
            characters = [
                self._to_char[i] for i in ids if i in self._to_char or not skip_special_tokens
            ]
            results.append("".join(characters))
        return results


@pytest.fixture(scope="session")
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


@pytest.fixture
def tiny_model_config(tokenizer: StubTokenizer):
    """A ModelConfig scaled down to the tiny backbone."""
    from msa_arc.config import ModelConfig

    return ModelConfig(
        backbone_name="stub",
        hidden_dim=32,
        bottleneck_dim=8,
        adapter_layers=(0, 1),
        dropout=0.0,
        audio_input_dim=8,
        video_input_dim=16,
        lstm_layers=2,
    )


@pytest.fixture
def tiny_backbone(tokenizer: StubTokenizer):
    """A randomly-initialised two-layer mT5 with a 32-dim hidden size."""
    import torch
    from transformers import MT5Config, MT5ForConditionalGeneration

    torch.manual_seed(0)
    config = MT5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        d_ff=64,
        d_kv=8,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
        relative_attention_num_buckets=8,
        dropout_rate=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    return MT5ForConditionalGeneration(config)


@pytest.fixture
def tiny_model(tiny_model_config, tiny_backbone):
    """A MulMT5 wired around the tiny backbone."""
    from msa_arc.model import build_model

    return build_model(tiny_model_config, backbone=tiny_backbone)


@pytest.fixture
def synthetic_corpus(tmp_path: Path, tiny_model_config):
    """A small Stage-A output: manifest, splits and feature tensors."""
    import pandas as pd

    from msa_arc.constants import ATTITUDE_CLASSES, SCENARIOS, SERVICE_IDS
    from msa_arc.features.manifest import instance_key

    rng = np.random.default_rng(0)
    root = tmp_path / "corpus"
    feature_dir = root / "features"
    for modality in ("text", "audio", "video"):
        (feature_dir / modality).mkdir(parents=True, exist_ok=True)

    services = list(SERVICE_IDS[:2])
    rows = []
    participants = [f"P{i:03d}" for i in range(4)]
    for index, participant in enumerate(participants):
        for service in services:
            for scenario in SCENARIOS:
                key = instance_key(participant, service, scenario)
                category = ATTITUDE_CLASSES[rng.integers(0, len(ATTITUDE_CLASSES))]
                np.save(
                    feature_dir / "text" / f"{key}.npy",
                    np.stack(
                        [
                            rng.integers(3, 20, size=16).astype(np.int64),
                            np.ones(16, dtype=np.int64),
                        ]
                    ),
                )
                np.save(
                    feature_dir / "audio" / f"{key}.npy",
                    rng.normal(size=(5, tiny_model_config.audio_input_dim)).astype(np.float32),
                )
                np.save(
                    feature_dir / "video" / f"{key}.npy",
                    rng.normal(size=(3, tiny_model_config.video_input_dim)).astype(np.float32),
                )
                rows.append(
                    {
                        "participant_id": participant,
                        "service_id": service,
                        "scenario": scenario,
                        "transcript": f"text {key}",
                        "label_polarity": ["positive", "negative", "neutral"][index % 3],
                        "label_intensity": round(float(rng.uniform(-1, 1)), 2),
                        "label_category": category,
                        "divergence_pattern": "none",
                    }
                )

    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    pd.DataFrame(
        {
            "participant_id": participants,
            "split": ["train", "train", "validation", "test"],
        }
    ).to_csv(root / "splits.csv", index=False)
    return root
