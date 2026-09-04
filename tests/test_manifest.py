"""The manifest contract: what is rejected, and what is legitimately optional."""

import pandas as pd
import pytest

from msa_arc.features.manifest import (
    ManifestError,
    instance_key,
    labelled_subset,
    load_manifest,
    media_segment,
    validate_manifest,
)


def base_rows() -> list[dict]:
    return [
        {
            "participant_id": "P001",
            "service_id": "s1",
            "scenario": 1,
            "transcript": "a",
            "label_polarity": "positive",
            "label_intensity": 0.8,
            "label_category": "Like",
        },
        {
            "participant_id": "P001",
            "service_id": "s1",
            "scenario": 0,
            "transcript": "b",
            "label_polarity": "negative",
            "label_intensity": -0.6,
            "label_category": "Dislike",
        },
    ]


def write(tmp_path, rows) -> str:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_a_minimal_manifest_loads(tmp_path) -> None:
    frame = load_manifest(write(tmp_path, base_rows()))
    assert len(frame) == 2
    assert frame["instance_key"].tolist() == ["P001__s1__f1", "P001__s1__f0"]


def test_optional_columns_are_materialised(tmp_path) -> None:
    frame = load_manifest(write(tmp_path, base_rows()))
    for column in ("audio_path", "video_start_sec", "divergence_pattern", "split"):
        assert column in frame.columns


def test_missing_file_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_manifest(tmp_path / "nope.csv")


def test_unknown_service_id_is_rejected(tmp_path) -> None:
    rows = base_rows()
    rows[0]["service_id"] = "s99"
    with pytest.raises(ManifestError, match="unknown service id"):
        load_manifest(write(tmp_path, rows))


def test_unknown_scenario_is_rejected(tmp_path) -> None:
    rows = base_rows()
    rows[0]["scenario"] = 2
    with pytest.raises(ManifestError, match="unknown scenario"):
        load_manifest(write(tmp_path, rows))


def test_duplicate_instances_are_rejected(tmp_path) -> None:
    rows = base_rows() + [base_rows()[0]]
    with pytest.raises(ManifestError, match="duplicated"):
        load_manifest(write(tmp_path, rows))


def test_partial_labels_are_rejected(tmp_path) -> None:
    """The three labels are elicited together and must be present together."""
    rows = base_rows()
    rows[0].pop("label_intensity")
    with pytest.raises(ManifestError, match="some but not all"):
        load_manifest(write(tmp_path, rows))


def test_intensity_outside_the_scale_is_rejected(tmp_path) -> None:
    rows = base_rows()
    rows[0]["label_intensity"] = 1.4
    with pytest.raises(ManifestError, match="outside"):
        load_manifest(write(tmp_path, rows))


def test_unknown_attitude_label_is_rejected(tmp_path) -> None:
    rows = base_rows()
    rows[0]["label_category"] = "Adores"
    with pytest.raises(ManifestError, match="unknown label_category"):
        load_manifest(write(tmp_path, rows))


def test_a_one_sided_offset_is_rejected(tmp_path) -> None:
    rows = base_rows()
    rows[0]["audio_path"] = "session.wav"
    rows[0]["audio_start_sec"] = 10.0
    with pytest.raises(ManifestError, match="only one of"):
        load_manifest(write(tmp_path, rows))


def test_a_backwards_offset_is_rejected(tmp_path) -> None:
    rows = base_rows()
    rows[0].update(audio_path="session.wav", audio_start_sec=20.0, audio_end_sec=10.0)
    with pytest.raises(ManifestError, match="non-increasing"):
        load_manifest(write(tmp_path, rows))


def test_offsets_without_a_path_are_rejected(tmp_path) -> None:
    rows = base_rows()
    rows[0].update(audio_start_sec=1.0, audio_end_sec=2.0)
    with pytest.raises(ManifestError, match="no audio_path"):
        load_manifest(write(tmp_path, rows))


def test_a_manifest_describing_no_media_is_rejected(tmp_path) -> None:
    rows = [{k: v for k, v in row.items() if k != "transcript"} for row in base_rows()]
    with pytest.raises(ManifestError, match="describes no data"):
        load_manifest(write(tmp_path, rows))


def test_pre_segmented_clips_are_accepted(tmp_path) -> None:
    rows = base_rows()
    for index, row in enumerate(rows):
        row["audio_path"] = f"clip_{index}.wav"
        row["video_path"] = f"clip_{index}.mp4"
    frame = load_manifest(write(tmp_path, rows))
    segment = media_segment(frame.iloc[0], "audio")
    assert segment == {"path": "clip_0.wav", "start": None, "end": None}


def test_session_recordings_with_offsets_are_accepted(tmp_path) -> None:
    rows = base_rows()
    rows[0].update(audio_path="session.wav", audio_start_sec=10.0, audio_end_sec=25.5)
    rows[1].update(audio_path="session.wav", audio_start_sec=25.5, audio_end_sec=40.0)
    frame = load_manifest(write(tmp_path, rows))
    assert media_segment(frame.iloc[0], "audio") == {
        "path": "session.wav",
        "start": 10.0,
        "end": 25.5,
    }


def test_mixed_layouts_across_modalities_are_accepted(tmp_path) -> None:
    """Audio may be pre-segmented while video comes from a session recording."""
    rows = base_rows()
    rows[0].update(
        audio_path="clip.wav",
        video_path="session.mp4",
        video_start_sec=3.0,
        video_end_sec=9.0,
    )
    frame = load_manifest(write(tmp_path, rows))
    assert media_segment(frame.iloc[0], "audio")["start"] is None
    assert media_segment(frame.iloc[0], "video")["start"] == 3.0


def test_unlabelled_rows_are_allowed_and_separable(tmp_path) -> None:
    """The 661 unannotated participants belong in the manifest too."""
    rows = base_rows() + [
        {
            "participant_id": "P002",
            "service_id": "s1",
            "scenario": 1,
            "transcript": "c",
        }
    ]
    frame = load_manifest(write(tmp_path, rows))
    assert len(frame) == 3
    assert len(labelled_subset(frame)) == 2


def test_incomplete_participants_warn_rather_than_fail(tmp_path, caplog) -> None:
    with caplog.at_level("WARNING"):
        load_manifest(write(tmp_path, base_rows()))
    assert "do not have all 56 instances" in caplog.text


def test_stats_count_participants_and_labels(tmp_path) -> None:
    frame = load_manifest(write(tmp_path, base_rows()), validate=False)
    stats = validate_manifest(frame)
    assert stats.n_rows == 2
    assert stats.n_participants == 1
    assert stats.n_annotated_participants == 1
    assert stats.n_labelled_rows == 2


def test_instance_key_is_stable() -> None:
    assert instance_key("P001", "s12", 0) == "P001__s12__f0"
