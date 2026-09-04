"""Participant-level partitioning (Section 5.1.1).

Splitting at the instance level would let the 56 instances of one participant
land in different splits, and the model could then memorise speaker-specific
prosody and facial style.  The partition is therefore drawn over participants:
188 train, 24 validation, 23 test.

A consequence worth stating, because a reader will notice it in the class
counts, is that the class proportions come out close but not identical across
the three splits.  Equalising them exactly would require an instance-level
partition and would reintroduce the leakage the design exists to prevent.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from msa_arc.constants import SPLIT_NAMES, SPLIT_PARTICIPANTS

logger = logging.getLogger(__name__)


class SplitError(ValueError):
    """Raised when a partition would leak participants across splits."""


def draw_participant_splits(
    participant_ids: Sequence[str],
    sizes: Optional[Dict[str, int]] = None,
    seed: int = 20240620,
) -> pd.DataFrame:
    """Draw a fresh participant-level partition.

    Args:
        participant_ids: The annotated participants to partition.
        sizes: Participants per split; defaults to the paper's 188/24/23.
        seed: RNG seed, so the partition is reproducible and can be published.

    Returns:
        A frame with columns ``participant_id`` and ``split``.

    Raises:
        SplitError: If the requested sizes do not match the number of
            participants supplied.
    """
    sizes = sizes or dict(SPLIT_PARTICIPANTS)
    unique = sorted({str(p) for p in participant_ids})
    required = sum(sizes.values())
    if len(unique) != required:
        raise SplitError(
            f"have {len(unique)} annotated participants but the requested split "
            f"sizes total {required}: {sizes}"
        )

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique))

    rows: List[Dict[str, str]] = []
    cursor = 0
    for name in SPLIT_NAMES:
        count = sizes[name]
        for participant in shuffled[cursor : cursor + count]:
            rows.append({"participant_id": str(participant), "split": name})
        cursor += count

    frame = pd.DataFrame(rows)
    logger.info(
        "Drew participant-level splits with seed %d: %s",
        seed,
        frame["split"].value_counts().to_dict(),
    )
    return frame


def load_splits(path: Union[str, Path]) -> pd.DataFrame:
    """Read a frozen split file.

    Args:
        path: CSV with ``participant_id`` and ``split`` columns.

    Returns:
        The split assignment.

    Raises:
        FileNotFoundError: If the file does not exist.
        SplitError: If a participant appears more than once.
    """
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"splits file not found: {split_path}")

    frame = pd.read_csv(split_path, dtype={"participant_id": str})
    missing = {"participant_id", "split"} - set(frame.columns)
    if missing:
        raise SplitError(f"splits file is missing columns: {sorted(missing)}")

    duplicated = frame["participant_id"].duplicated()
    if bool(duplicated.any()):
        examples = frame.loc[duplicated, "participant_id"].head(5).tolist()
        raise SplitError(f"participants assigned to more than one split: {examples}")
    return frame


def save_splits(frame: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write the split assignment so a reader can reproduce the partition."""
    split_path = Path(path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["split", "participant_id"]).to_csv(split_path, index=False)
    logger.info("Wrote %d split assignments to %s", len(frame), split_path)
    return split_path


def apply_splits(manifest: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Attach split labels to manifest rows and check for leakage.

    Args:
        manifest: A loaded manifest.
        splits: Participant-level split assignment.

    Returns:
        The manifest with a ``split`` column, restricted to participants that
        appear in ``splits``.

    Raises:
        SplitError: If any participant ends up in more than one split.
    """
    assigned = manifest.drop(columns=["split"], errors="ignore").merge(
        splits, on="participant_id", how="inner", validate="many_to_one"
    )
    assert_no_leakage(assigned)

    counts = assigned.groupby("split")["participant_id"].nunique().to_dict()
    logger.info(
        "Applied splits: %s participants, %d instances",
        counts,
        len(assigned),
    )
    return assigned


def assert_no_leakage(frame: pd.DataFrame) -> None:
    """Fail loudly if one participant's instances span several splits.

    Args:
        frame: A manifest carrying a ``split`` column.

    Raises:
        SplitError: If any participant appears in more than one split.
    """
    per_participant = frame.groupby("participant_id")["split"].nunique()
    leaking = per_participant[per_participant > 1]
    if len(leaking) > 0:
        raise SplitError(
            f"{len(leaking)} participants appear in more than one split; "
            f"first few: {leaking.index[:5].tolist()}"
        )


def class_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-split, per-class instance counts.

    Reproduces the split/class table of Section 5.1.1, which a reader uses to
    check that the participant-level partition left the classes roughly, though
    not exactly, balanced.

    Args:
        frame: A manifest carrying ``split`` and ``label_category`` columns.

    Returns:
        Split-by-category counts with a ``Total`` column.
    """
    labelled = frame[frame["label_category"].notna()]
    table = pd.crosstab(labelled["split"], labelled["label_category"])
    table["Total"] = table.sum(axis=1)
    return table


__all__ = [
    "SplitError",
    "apply_splits",
    "assert_no_leakage",
    "class_counts",
    "draw_participant_splits",
    "load_splits",
    "save_splits",
]
