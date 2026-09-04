"""Assembling the three split loaders from a manifest and a feature store.

Shared by the training and evaluation entry points so that both see exactly the
same instances in the same order.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from torch.utils.data import DataLoader

from msa_arc.config import DataConfig, ModelConfig, TrainConfig
from msa_arc.constants import SPLIT_NAMES
from msa_arc.data.collate import Collator
from msa_arc.data.dataset import MSAARCDataset
from msa_arc.data.splits import (
    apply_splits,
    class_counts,
    draw_participant_splits,
    load_splits,
    save_splits,
)
from msa_arc.features.manifest import labelled_subset, load_manifest

logger = logging.getLogger(__name__)


def prepare_manifest(data_cfg: DataConfig) -> pd.DataFrame:
    """Load the manifest, keep the annotated rows, and attach splits.

    A frozen split file is used when one exists, so that a re-run reproduces the
    published partition; otherwise a fresh partition is drawn and written, which
    is what a first run on a new corpus does.

    Args:
        data_cfg: Paths and loading flags.

    Returns:
        The annotated manifest rows carrying a ``split`` column.
    """
    manifest = load_manifest(data_cfg.manifest_path)
    annotated = labelled_subset(manifest)
    logger.info(
        "Manifest holds %d annotated instances from %d participants",
        len(annotated),
        annotated["participant_id"].nunique(),
    )

    splits_path = Path(data_cfg.splits_path)
    if splits_path.exists():
        splits = load_splits(splits_path)
    else:
        logger.warning(
            "No splits file at %s; drawing a fresh participant-level partition "
            "with seed %d and writing it there",
            splits_path,
            data_cfg.split_seed,
        )
        splits = draw_participant_splits(
            annotated["participant_id"].unique(), seed=data_cfg.split_seed
        )
        save_splits(splits, splits_path)

    assigned = apply_splits(annotated, splits)
    logger.info("Per-split class counts:\n%s", class_counts(assigned).to_string())
    return assigned


def build_dataloaders(
    manifest: pd.DataFrame,
    tokenizer: Any,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    """Build one loader per split.

    Args:
        manifest: Annotated manifest carrying a ``split`` column.
        tokenizer: Tokenizer used to encode the unified target strings.
        data_cfg: Feature-store paths and modality flags.
        model_cfg: Supplies the feature widths used to shape empty tensors.
        train_cfg: Supplies batch size and worker count.
        shuffle_train: Whether to shuffle the training split.

    Returns:
        Mapping from split name to loader, omitting splits with no instances.
    """
    collator = Collator(
        tokenizer=tokenizer,
        audio_dim=model_cfg.audio_input_dim,
        video_dim=model_cfg.video_input_dim,
    )
    loaders: Dict[str, DataLoader] = {}
    for split in SPLIT_NAMES:
        rows = manifest[manifest["split"] == split]
        if rows.empty:
            logger.warning("split %r has no instances; skipping", split)
            continue
        dataset = MSAARCDataset(
            rows,
            feature_dir=data_cfg.feature_dir,
            load_audio=data_cfg.load_audio and model_cfg.use_audio,
            load_video=data_cfg.load_video and model_cfg.use_video,
            require_labels=True,
            audio_dim=model_cfg.audio_input_dim,
            video_dim=model_cfg.video_input_dim,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=shuffle_train and split == "train",
            num_workers=train_cfg.num_workers,
            collate_fn=collator,
            drop_last=False,
        )
        logger.info("Split %-10s %5d instances", split, len(dataset))
    return loaders


def split_frame(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    """The manifest rows of one split, for joining labels onto predictions."""
    return manifest[manifest["split"] == split].reset_index(drop=True)


__all__ = ["build_dataloaders", "prepare_manifest", "split_frame"]
