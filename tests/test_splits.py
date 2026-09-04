"""Participant-level partitioning must not leak a speaker across splits."""

import pandas as pd
import pytest

from msa_arc.data.splits import (
    SplitError,
    apply_splits,
    assert_no_leakage,
    class_counts,
    draw_participant_splits,
    load_splits,
    save_splits,
)


def test_draw_produces_the_paper_split_sizes() -> None:
    participants = [f"P{i:04d}" for i in range(235)]
    splits = draw_participant_splits(participants)
    assert splits["split"].value_counts().to_dict() == {
        "train": 188,
        "validation": 24,
        "test": 23,
    }
    assert splits["participant_id"].nunique() == 235


def test_draw_is_deterministic_given_a_seed() -> None:
    participants = [f"P{i:04d}" for i in range(235)]
    first = draw_participant_splits(participants, seed=17)
    second = draw_participant_splits(participants, seed=17)
    pd.testing.assert_frame_equal(first, second)
    third = draw_participant_splits(participants, seed=29)
    assert not first.equals(third)


def test_draw_rejects_a_participant_count_that_does_not_add_up() -> None:
    with pytest.raises(SplitError, match="split sizes total"):
        draw_participant_splits([f"P{i}" for i in range(100)])


def test_applying_splits_never_leaks_a_participant() -> None:
    manifest = pd.DataFrame(
        {
            "participant_id": ["A", "A", "B", "B", "C", "C"],
            "service_id": ["s1", "s1", "s1", "s1", "s1", "s1"],
            "scenario": [1, 0, 1, 0, 1, 0],
        }
    )
    splits = pd.DataFrame(
        {"participant_id": ["A", "B", "C"], "split": ["train", "validation", "test"]}
    )
    assigned = apply_splits(manifest, splits)
    assert len(assigned) == 6
    assert_no_leakage(assigned)


def test_leakage_is_detected() -> None:
    leaking = pd.DataFrame({"participant_id": ["A", "A"], "split": ["train", "test"]})
    with pytest.raises(SplitError, match="more than one split"):
        assert_no_leakage(leaking)


def test_duplicate_assignment_is_rejected_on_load(tmp_path) -> None:
    path = tmp_path / "splits.csv"
    pd.DataFrame({"participant_id": ["A", "A"], "split": ["train", "test"]}).to_csv(
        path, index=False
    )
    with pytest.raises(SplitError, match="more than one split"):
        load_splits(path)


def test_splits_round_trip(tmp_path) -> None:
    splits = draw_participant_splits([f"P{i:04d}" for i in range(235)], seed=17)
    path = save_splits(splits, tmp_path / "splits.csv")
    reloaded = load_splits(path)
    assert set(reloaded["participant_id"]) == set(splits["participant_id"])


def test_class_counts_tabulate_each_split() -> None:
    frame = pd.DataFrame(
        {
            "participant_id": ["A", "A", "B", "B"],
            "split": ["train", "train", "test", "test"],
            "label_category": ["Like", "Dislike", "Like", "Neutral"],
        }
    )
    table = class_counts(frame)
    assert table.loc["train", "Total"] == 2
    assert table.loc["test", "Like"] == 1
