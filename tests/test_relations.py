"""Intensity calibration and the empirical relationship among the three outputs."""

import math

import pytest

from msa_arc.constants import ATTITUDE_CLASSES, POLARITY_CLASSES
from msa_arc.evaluation.calibration import intensity_calibration
from msa_arc.evaluation.relations import intensity_by_attitude, polarity_by_attitude


def test_calibration_bins_span_the_label_range() -> None:
    result = intensity_calibration([0.0] * 10, [0.0] * 10, n_bins=9)
    assert len(result.bins) == 9
    assert result.bins[0].lower == pytest.approx(-1.0)
    assert result.bins[-1].upper == pytest.approx(1.0)
    assert result.n_total == 10


def test_perfect_predictions_have_zero_calibration_error() -> None:
    values = [-0.9, -0.5, -0.1, 0.0, 0.2, 0.6, 0.95]
    result = intensity_calibration(values, values)
    assert result.ece == pytest.approx(0.0, abs=1e-12)
    for b in result.bins:
        if b.count:
            assert b.discrepancy == pytest.approx(0.0, abs=1e-12)


def test_a_constant_offset_becomes_the_calibration_error() -> None:
    labels = [-0.8, -0.4, 0.0, 0.4, 0.8]
    predictions = [v + 0.1 for v in labels]
    result = intensity_calibration(predictions, labels)
    assert result.ece == pytest.approx(0.1, abs=1e-9)


def test_the_extreme_of_the_scale_is_counted() -> None:
    """A prediction of exactly -1 belongs to the first bin, not to nothing."""
    result = intensity_calibration([-1.0], [-1.0])
    assert sum(b.count for b in result.bins) == 1
    assert result.bins[0].count == 1


def test_empty_bins_report_nan_rather_than_zero() -> None:
    result = intensity_calibration([0.9], [0.9])
    empty = [b for b in result.bins if b.count == 0]
    assert empty
    assert all(math.isnan(b.mean_predicted) for b in empty)


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        intensity_calibration([0.1, 0.2], [0.1])


def test_contingency_table_keeps_the_canonical_shape() -> None:
    """Absent combinations stay as zeros so the table shape is stable."""
    table = polarity_by_attitude(["positive", "negative"], ["Like", "Dislike"])
    assert list(table.columns) == list(ATTITUDE_CLASSES) + ["Total"]
    assert list(table.index) == list(POLARITY_CLASSES) + ["Total"]
    assert table.loc["positive", "Like"] == 1
    assert table.loc["neutral", "Neutral"] == 0
    assert table.loc["Total", "Total"] == 2


def test_contingency_table_records_the_divergent_cells() -> None:
    """A negative polarity under a surface Like is the sarcasm pattern."""
    table = polarity_by_attitude(
        ["negative", "negative", "positive"], ["Like", "Like", "Like"]
    )
    assert table.loc["negative", "Like"] == 2
    assert table.loc["positive", "Like"] == 1


def test_intensity_summary_covers_every_category() -> None:
    summary = intensity_by_attitude([0.9, 0.7, -0.8], ["Like", "Like", "Dislike"])
    assert list(summary.index) == list(ATTITUDE_CLASSES)
    assert summary.loc["Like", "n"] == 2
    assert summary.loc["Like", "mean"] == pytest.approx(0.8)
    assert summary.loc["Dislike", "median"] == pytest.approx(-0.8)
    assert math.isnan(summary.loc["Neutral", "mean"])


def test_relation_tables_reject_ragged_input() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        polarity_by_attitude(["positive"], ["Like", "Dislike"])
    with pytest.raises(ValueError, match="length mismatch"):
        intensity_by_attitude([0.5], ["Like", "Dislike"])
