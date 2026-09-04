"""Per-output evaluation indicators (Section 5.2.1).

MSA-ARC emits three coupled outputs and the MCVA rule consumes all three, so
reporting only attitude-category accuracy would leave the two signals that drive
the reconciliation unevaluated.  Each output is scored against its own labels:

* polarity: accuracy, macro-F1, 3x3 confusion matrix;
* intensity: MAE, RMSE, Pearson and Spearman correlation;
* attitude: accuracy, macro-F1, per-class precision/recall/F1/support, and the
  5x5 confusion matrix in raw counts.

Confusion matrices are returned as counts, never as percentages.  A reader
checking 1,190/1,288 against a percentage figure cannot recover the numerator.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from msa_arc.constants import ATTITUDE_CLASSES, DIVERGENCE_SUBSET, POLARITY_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Scores for one nominal output.

    Attributes:
        accuracy: Overall accuracy.
        macro_f1: Macro-averaged F1.
        n_correct: Raw number correct, so a percentage can be checked.
        n_total: Raw number of instances.
        per_class: Precision, recall, F1 and support per class.
        confusion: Rows are true labels, columns predictions, in ``labels``
            order.
        labels: The label space, fixing the confusion-matrix axes.
    """

    accuracy: float
    macro_f1: float
    n_correct: int
    n_total: int
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)


@dataclass
class RegressionMetrics:
    """Scores for the continuous intensity output."""

    mae: float
    rmse: float
    pearson: float
    spearman: float
    n_total: int


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> ClassificationMetrics:
    """Score a nominal output against its labels.

    Args:
        y_true: Human labels.
        y_pred: Model predictions.
        labels: The label space, which fixes the confusion-matrix axes even when
            a class happens to be absent from a split.

    Returns:
        The metrics.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} labels vs {len(y_pred)} predictions")
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    label_list = list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=label_list)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_list, zero_division=0
    )

    n_correct = int(np.trace(matrix))
    n_total = int(matrix.sum())
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(label_list)
    }
    return ClassificationMetrics(
        accuracy=n_correct / n_total if n_total else 0.0,
        macro_f1=float(np.mean(f1)),
        n_correct=n_correct,
        n_total=n_total,
        per_class=per_class,
        confusion=matrix.astype(int).tolist(),
        labels=label_list,
    )


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> RegressionMetrics:
    """Score the continuous intensity output.

    Both Pearson and Spearman are reported: the first measures linear agreement
    with the annotators' slider positions, the second measures whether the model
    orders instances by intensity the way the annotators did.

    Args:
        y_true: Human intensity labels on ``[-1, 1]``.
        y_pred: Predicted intensities.

    Returns:
        The metrics. Correlations are ``nan`` when a series has zero variance.
    """
    from scipy import stats

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    errors = prediction - truth

    def _safe_corr(fn) -> float:
        if truth.size < 2 or np.std(truth) == 0 or np.std(prediction) == 0:
            return float("nan")
        return float(fn(truth, prediction)[0])

    return RegressionMetrics(
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        pearson=_safe_corr(stats.pearsonr),
        spearman=_safe_corr(stats.spearmanr),
        n_total=int(truth.size),
    )


def attitude_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> ClassificationMetrics:
    """Score the attitude output over the five categories."""
    return classification_metrics(y_true, y_pred, ATTITUDE_CLASSES)


def polarity_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> ClassificationMetrics:
    """Score the polarity output over the three classes."""
    return classification_metrics(y_true, y_pred, POLARITY_CLASSES)


def divergence_subset_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    patterns: Sequence[Optional[str]],
) -> Dict[str, ClassificationMetrics]:
    """Score each verbal-nonverbal divergence pattern separately.

    Section 6.3 reports macro-averaged accuracy over the four divergence
    patterns, which requires each pattern to be scored on its own rather than
    pooled.

    Args:
        y_true: Human attitude labels.
        y_pred: Predicted attitude categories.
        patterns: Divergence pattern of each instance, ``None`` or ``"none"``
            for the sincere-endorsement baseline.

    Returns:
        Metrics per pattern, plus an ``all_divergence`` entry pooling the four.
    """
    results: Dict[str, ClassificationMetrics] = {}
    pattern_array = np.asarray([p if p else "none" for p in patterns], dtype=object)
    truth = np.asarray(list(y_true), dtype=object)
    prediction = np.asarray(list(y_pred), dtype=object)

    for pattern in DIVERGENCE_SUBSET:
        selected = pattern_array == pattern
        if not selected.any():
            logger.warning("no instances carry divergence pattern %r", pattern)
            continue
        results[pattern] = attitude_metrics(truth[selected], prediction[selected])

    pooled = np.isin(pattern_array, list(DIVERGENCE_SUBSET))
    if pooled.any():
        results["all_divergence"] = attitude_metrics(truth[pooled], prediction[pooled])
    return results


def macro_average_accuracy(metrics: Dict[str, ClassificationMetrics]) -> float:
    """Mean accuracy over the four divergence patterns.

    Args:
        metrics: Output of :func:`divergence_subset_metrics`.

    Returns:
        The macro average over the patterns present, excluding the pooled entry.
    """
    accuracies = [m.accuracy for name, m in metrics.items() if name in DIVERGENCE_SUBSET]
    return float(np.mean(accuracies)) if accuracies else float("nan")


__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
    "attitude_metrics",
    "classification_metrics",
    "divergence_subset_metrics",
    "macro_average_accuracy",
    "polarity_metrics",
    "regression_metrics",
]
