"""Evaluation: metrics, calibration, per-run bundles and across-seed reports."""

from msa_arc.evaluation.bundle import (
    EvaluationBundle,
    apply_mcva,
    evaluate_predictions,
    fit_bands_on_validation,
    mcva_summary,
    predictions_to_frame,
)
from msa_arc.evaluation.calibration import (
    CalibrationMetrics,
    IntensityCalibration,
    calibration_metrics,
    confidence_accuracy_table,
    intensity_calibration,
)
from msa_arc.evaluation.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    attitude_metrics,
    classification_metrics,
    divergence_subset_metrics,
    macro_average_accuracy,
    polarity_metrics,
    regression_metrics,
)
from msa_arc.evaluation.relations import intensity_by_attitude, polarity_by_attitude
from msa_arc.evaluation.report import (
    aggregate_runs,
    confusion_frame,
    intensity_calibration_frame,
    per_class_frame,
    polarity_confusion_frame,
    scalar_metrics,
    write_run,
    write_summary,
)

__all__ = [
    "CalibrationMetrics",
    "ClassificationMetrics",
    "EvaluationBundle",
    "RegressionMetrics",
    "aggregate_runs",
    "apply_mcva",
    "attitude_metrics",
    "calibration_metrics",
    "classification_metrics",
    "confidence_accuracy_table",
    "IntensityCalibration",
    "confusion_frame",
    "divergence_subset_metrics",
    "evaluate_predictions",
    "intensity_by_attitude",
    "intensity_calibration",
    "intensity_calibration_frame",
    "fit_bands_on_validation",
    "macro_average_accuracy",
    "mcva_summary",
    "per_class_frame",
    "polarity_by_attitude",
    "polarity_confusion_frame",
    "polarity_metrics",
    "predictions_to_frame",
    "regression_metrics",
    "scalar_metrics",
    "write_run",
    "write_summary",
]
