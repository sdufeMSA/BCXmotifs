"""Calibration of the five-class distribution.

Section 6.4 perturbs low-confidence predictions, which only means something if
the confidence is calibrated: a prediction the model reports at 0.6 should be
right about 60% of the time.  The five-class distribution comes from constrained
rescoring, so its calibration is a property of that derivation and has to be
measured rather than assumed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from msa_arc.constants import ATTITUDE_TO_INDEX


@dataclass
class CalibrationBin:
    """One equal-width confidence bin of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass
class CalibrationMetrics:
    """Calibration summary of a probability matrix.

    Attributes:
        ece: Expected calibration error, the count-weighted mean gap between
            confidence and accuracy across bins.
        mce: Maximum calibration error, the worst such gap.
        brier: Multiclass Brier score against the one-hot truth.
        nll: Mean negative log-likelihood of the true class.
        bins: The reliability diagram, ready to plot.
    """

    ece: float
    mce: float
    brier: float
    nll: float
    bins: List[CalibrationBin] = field(default_factory=list)


def calibration_metrics(
    probabilities: Sequence[Sequence[float]],
    y_true: Sequence[str],
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Measure how well the predicted probabilities are calibrated.

    Args:
        probabilities: ``(n, 5)`` distributions over ``ATTITUDE_CLASSES``.
        y_true: Human attitude labels.
        n_bins: Number of equal-width confidence bins.

    Returns:
        The calibration metrics.

    Raises:
        ValueError: If the inputs disagree in length.
    """
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.shape[0] != len(y_true):
        raise ValueError(
            f"length mismatch: {matrix.shape[0]} probability rows vs {len(y_true)} labels"
        )

    truth_index = np.asarray([ATTITUDE_TO_INDEX[label] for label in y_true])
    predicted_index = matrix.argmax(axis=1)
    confidence = matrix.max(axis=1)
    correct = (predicted_index == truth_index).astype(float)

    one_hot = np.zeros_like(matrix)
    one_hot[np.arange(matrix.shape[0]), truth_index] = 1.0
    brier = float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1)))
    true_probability = matrix[np.arange(matrix.shape[0]), truth_index]
    nll = float(-np.mean(np.log(np.clip(true_probability, 1e-12, None))))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        # Include the right edge in the final bin so confidence 1.0 is counted.
        selected = (
            (confidence > lower) & (confidence <= upper)
            if index > 0
            else (confidence >= lower) & (confidence <= upper)
        )
        count = int(selected.sum())
        if count == 0:
            bins.append(CalibrationBin(lower, upper, 0, float("nan"), float("nan")))
            continue
        mean_confidence = float(confidence[selected].mean())
        accuracy = float(correct[selected].mean())
        gap = abs(mean_confidence - accuracy)
        ece += count / matrix.shape[0] * gap
        mce = max(mce, gap)
        bins.append(CalibrationBin(lower, upper, count, mean_confidence, accuracy))

    return CalibrationMetrics(ece=ece, mce=mce, brier=brier, nll=nll, bins=bins)


@dataclass
class IntensityBin:
    """One equal-width bin of the intensity calibration table.

    Attributes:
        index: 1-based bin number, matching the manuscript's table.
        lower: Lower edge of the bin on ``[-1, 1]``.
        upper: Upper edge.
        mean_predicted: Mean predicted intensity of the instances in the bin.
        mean_labelled: Mean human-labelled intensity of the same instances.
        count: How many instances fell in the bin.
        discrepancy: ``mean_predicted - mean_labelled``.
    """

    index: int
    lower: float
    upper: float
    mean_predicted: float
    mean_labelled: float
    count: int
    discrepancy: float


@dataclass
class IntensityCalibration:
    """Calibration of the continuous intensity output.

    Attributes:
        bins: One entry per equal-width bin, in increasing order.
        ece: Sample-weighted mean absolute discrepancy across bins.
        n_total: Instances scored.
    """

    bins: List[IntensityBin] = field(default_factory=list)
    ece: float = 0.0
    n_total: int = 0


def intensity_calibration(
    y_pred: Sequence[float],
    y_true: Sequence[float],
    n_bins: int = 9,
) -> IntensityCalibration:
    """Bin the intensity output and compare predicted against labelled means.

    Predictions are grouped into equal-width bins over ``[-1, 1]``; the
    discrepancy of a bin is its predicted minus its labelled mean, and the
    expected calibration error is the sample-weighted mean of the absolute
    discrepancies.

    Args:
        y_pred: Predicted intensities.
        y_true: Human intensity labels.
        n_bins: Number of equal-width bins; the manuscript uses nine.

    Returns:
        The binned calibration.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    predicted = np.asarray(y_pred, dtype=float)
    labelled = np.asarray(y_true, dtype=float)
    if predicted.shape != labelled.shape:
        raise ValueError(
            f"length mismatch: {predicted.size} predictions vs {labelled.size} labels"
        )

    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bins: List[IntensityBin] = []
    weighted_error = 0.0

    for index in range(n_bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        # The first bin owns its lower edge so that a prediction of exactly -1
        # is counted; every other bin is left-open.
        selected = (
            (predicted >= lower) & (predicted <= upper)
            if index == 0
            else (predicted > lower) & (predicted <= upper)
        )
        count = int(selected.sum())
        if count == 0:
            bins.append(
                IntensityBin(
                    index + 1, lower, upper, float("nan"), float("nan"), 0, float("nan")
                )
            )
            continue
        mean_predicted = float(predicted[selected].mean())
        mean_labelled = float(labelled[selected].mean())
        discrepancy = mean_predicted - mean_labelled
        weighted_error += count * abs(discrepancy)
        bins.append(
            IntensityBin(
                index=index + 1,
                lower=lower,
                upper=upper,
                mean_predicted=mean_predicted,
                mean_labelled=mean_labelled,
                count=count,
                discrepancy=discrepancy,
            )
        )

    total = int(predicted.size)
    return IntensityCalibration(
        bins=bins,
        ece=weighted_error / total if total else 0.0,
        n_total=total,
    )


def confidence_accuracy_table(
    probabilities: Sequence[Sequence[float]],
    y_true: Sequence[str],
    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> List[Dict[str, float]]:
    """Accuracy and coverage at a series of confidence thresholds.

    Useful for the deployment question the framework raises: how much of the
    corpus can be auto-accepted, and how accurate is that part.

    Args:
        probabilities: ``(n, 5)`` distributions.
        y_true: Human attitude labels.
        thresholds: Confidence cut-offs.

    Returns:
        One row per threshold with coverage, accuracy and the raw counts behind
        them.
    """
    matrix = np.asarray(probabilities, dtype=float)
    truth_index = np.asarray([ATTITUDE_TO_INDEX[label] for label in y_true])
    predicted_index = matrix.argmax(axis=1)
    confidence = matrix.max(axis=1)
    correct = predicted_index == truth_index

    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        selected = confidence >= threshold
        count = int(selected.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": count / matrix.shape[0] if matrix.shape[0] else 0.0,
                "n_selected": count,
                "n_correct": int(correct[selected].sum()),
                "accuracy": float(correct[selected].mean()) if count else float("nan"),
            }
        )
    return rows


__all__ = [
    "CalibrationBin",
    "CalibrationMetrics",
    "IntensityBin",
    "IntensityCalibration",
    "calibration_metrics",
    "confidence_accuracy_table",
    "intensity_calibration",
]
