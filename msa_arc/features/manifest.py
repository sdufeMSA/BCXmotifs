"""The manifest: the contract between the interview corpus and this code.

One row per (participant, service, scenario) instance: 56 rows per participant,
being 28 services times the functional and dysfunctional scenarios.

The media columns accept two layouts, because a corpus may have been segmented
either way:

* **pre-segmented clips**: ``audio_path``/``video_path`` point at one file per
  instance and the offset columns are left blank;
* **full-session recordings**: the paths point at the whole 30-40 minute
  session and ``audio_start_sec``/``audio_end_sec`` (likewise for video) mark
  where the instance sits inside it.

Offsets may be given for one modality and omitted for the other.  Validation
rejects anything that would silently corrupt a result: unknown service ids,
duplicated instances, labels outside their space, offsets that run backwards.
What is genuinely optional stays optional.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from msa_arc.constants import (
    ATTITUDE_CLASSES,
    DIVERGENCE_PATTERNS,
    INSTANCES_PER_PARTICIPANT,
    INTENSITY_MAX,
    INTENSITY_MIN,
    POLARITY_CLASSES,
    SCENARIOS,
    SERVICE_IDS,
    SPLIT_NAMES,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Sequence[str] = (
    "participant_id",
    "service_id",
    "scenario",
)

MEDIA_COLUMNS: Sequence[str] = (
    "transcript",
    "transcript_path",
    "audio_path",
    "audio_start_sec",
    "audio_end_sec",
    "video_path",
    "video_start_sec",
    "video_end_sec",
)

LABEL_COLUMNS: Sequence[str] = (
    "label_polarity",
    "label_intensity",
    "label_category",
)

OPTIONAL_COLUMNS: Sequence[str] = (
    *MEDIA_COLUMNS,
    *LABEL_COLUMNS,
    "divergence_pattern",
    "split",
    "annotated",
)

ALL_COLUMNS: Sequence[str] = (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)


@dataclass(frozen=True)
class ManifestStats:
    """Summary of a validated manifest, logged before any run starts."""

    n_rows: int
    n_participants: int
    n_annotated_participants: int
    n_labelled_rows: int
    complete_participants: int
    incomplete_participants: List[str]


class ManifestError(ValueError):
    """Raised when a manifest violates the contract."""


def instance_key(participant_id: str, service_id: str, scenario: int) -> str:
    """Stable identifier for one instance, used as the feature file stem."""
    return f"{participant_id}__{service_id}__f{int(scenario)}"


def load_manifest(path: Union[str, Path], validate: bool = True) -> pd.DataFrame:
    """Read and optionally validate a manifest CSV.

    Args:
        path: Path to ``manifest.csv``.
        validate: Whether to run :func:`validate_manifest`.

    Returns:
        The manifest as a DataFrame with every optional column present, filled
        with ``NA`` where the file omitted it, so downstream code never has to
        test for column existence.

    Raises:
        FileNotFoundError: If the file does not exist.
        ManifestError: If validation fails.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path, dtype={"participant_id": str, "service_id": str})
    for column in ALL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["scenario"] = pd.to_numeric(frame["scenario"], errors="coerce").astype("Int64")
    for column in (
        "label_intensity",
        "audio_start_sec",
        "audio_end_sec",
        "video_start_sec",
        "video_end_sec",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["instance_key"] = [
        instance_key(row.participant_id, row.service_id, row.scenario)
        for row in frame.itertuples()
    ]

    if validate:
        stats = validate_manifest(frame)
        logger.info(
            "Manifest %s: %d rows, %d participants (%d annotated, %d labelled rows)",
            manifest_path,
            stats.n_rows,
            stats.n_participants,
            stats.n_annotated_participants,
            stats.n_labelled_rows,
        )
        if stats.incomplete_participants:
            logger.warning(
                "%d participants do not have all %d instances; first few: %s",
                len(stats.incomplete_participants),
                INSTANCES_PER_PARTICIPANT,
                stats.incomplete_participants[:5],
            )
    return frame


def validate_manifest(frame: pd.DataFrame) -> ManifestStats:
    """Check a manifest against the contract.

    Args:
        frame: A loaded manifest.

    Returns:
        Summary statistics.

    Raises:
        ManifestError: On any violation that would corrupt a downstream result.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ManifestError(f"manifest is missing required columns: {missing}")

    _check_membership(frame, "service_id", SERVICE_IDS, "service id")
    _check_membership(frame, "scenario", SCENARIOS, "scenario")

    duplicated = frame["instance_key"].duplicated()
    if bool(duplicated.any()):
        examples = frame.loc[duplicated, "instance_key"].head(5).tolist()
        raise ManifestError(
            f"{int(duplicated.sum())} duplicated (participant, service, scenario) "
            f"instances; first few: {examples}"
        )

    _validate_labels(frame)
    _validate_offsets(frame)
    _validate_optional_membership(frame, "divergence_pattern", DIVERGENCE_PATTERNS)
    _validate_optional_membership(frame, "split", SPLIT_NAMES)

    has_media = (
        frame["transcript"].notna()
        | frame["transcript_path"].notna()
        | frame["audio_path"].notna()
        | frame["video_path"].notna()
    )
    if not bool(has_media.any()):
        raise ManifestError(
            "no row carries any of transcript, transcript_path, audio_path or "
            "video_path; the manifest describes no data"
        )

    counts = frame.groupby("participant_id").size()
    incomplete = counts[counts != INSTANCES_PER_PARTICIPANT]
    labelled = frame["label_category"].notna()

    return ManifestStats(
        n_rows=len(frame),
        n_participants=int(frame["participant_id"].nunique()),
        n_annotated_participants=int(frame.loc[labelled, "participant_id"].nunique()),
        n_labelled_rows=int(labelled.sum()),
        complete_participants=int((counts == INSTANCES_PER_PARTICIPANT).sum()),
        incomplete_participants=[str(p) for p in incomplete.index.tolist()],
    )


def _check_membership(frame: pd.DataFrame, column: str, allowed: Sequence, label: str) -> None:
    """Reject values outside a closed vocabulary."""
    values = frame[column].dropna()
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ManifestError(f"unknown {label}(s) in column {column!r}: {unknown[:10]}")


def _validate_optional_membership(
    frame: pd.DataFrame, column: str, allowed: Sequence[str]
) -> None:
    """Same check, but tolerating a wholly absent column."""
    if frame[column].notna().any():
        _check_membership(frame, column, allowed, column)


def _validate_labels(frame: pd.DataFrame) -> None:
    """Labels must be complete per row and inside their label spaces."""
    _validate_optional_membership(frame, "label_polarity", POLARITY_CLASSES)
    _validate_optional_membership(frame, "label_category", ATTITUDE_CLASSES)

    intensity = frame["label_intensity"].dropna()
    out_of_range = intensity[(intensity < INTENSITY_MIN) | (intensity > INTENSITY_MAX)]
    if len(out_of_range) > 0:
        raise ManifestError(
            f"{len(out_of_range)} label_intensity values outside "
            f"[{INTENSITY_MIN}, {INTENSITY_MAX}]; first few: "
            f"{out_of_range.head(5).tolist()}"
        )

    present = frame[list(LABEL_COLUMNS)].notna()
    partial = present.any(axis=1) & ~present.all(axis=1)
    if bool(partial.any()):
        examples = frame.loc[partial, "instance_key"].head(5).tolist()
        raise ManifestError(
            f"{int(partial.sum())} rows carry some but not all of "
            f"{list(LABEL_COLUMNS)}; the three labels are elicited together and "
            f"must be present together. First few: {examples}"
        )


def _validate_offsets(frame: pd.DataFrame) -> None:
    """Offsets must be non-negative, ordered, and paired with a media path."""
    for modality in ("audio", "video"):
        start = frame[f"{modality}_start_sec"]
        end = frame[f"{modality}_end_sec"]
        path = frame[f"{modality}_path"]

        one_sided = start.notna() ^ end.notna()
        if bool(one_sided.any()):
            examples = frame.loc[one_sided, "instance_key"].head(5).tolist()
            raise ManifestError(
                f"{int(one_sided.sum())} rows give only one of "
                f"{modality}_start_sec/{modality}_end_sec; first few: {examples}"
            )

        both = start.notna() & end.notna()
        if bool((both & path.isna()).any()):
            raise ManifestError(f"rows give {modality} offsets but no {modality}_path")
        bad = both & ((start < 0) | (end <= start))
        if bool(bad.any()):
            examples = frame.loc[bad, "instance_key"].head(5).tolist()
            raise ManifestError(
                f"{int(bad.sum())} rows have non-increasing or negative {modality} "
                f"offsets; first few: {examples}"
            )


def media_segment(
    row: pd.Series, modality: str
) -> Optional[Dict[str, Union[str, float, None]]]:
    """Resolve one row's media reference for a modality.

    Args:
        row: A manifest row.
        modality: ``audio`` or ``video``.

    Returns:
        ``{"path": ..., "start": ..., "end": ...}`` with ``start``/``end`` set to
        ``None`` for a pre-segmented clip, or ``None`` when the row has no media
        for this modality.
    """
    path = row.get(f"{modality}_path")
    if path is None or pd.isna(path):
        return None
    start = row.get(f"{modality}_start_sec")
    end = row.get(f"{modality}_end_sec")
    return {
        "path": str(path),
        "start": None if pd.isna(start) else float(start),
        "end": None if pd.isna(end) else float(end),
    }


def labelled_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows carrying all three human labels, i.e. the 235-participant corpus."""
    return frame[frame[list(LABEL_COLUMNS)].notna().all(axis=1)].copy()


__all__ = [
    "ALL_COLUMNS",
    "LABEL_COLUMNS",
    "MEDIA_COLUMNS",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "ManifestError",
    "ManifestStats",
    "instance_key",
    "labelled_subset",
    "load_manifest",
    "media_segment",
    "validate_manifest",
]
