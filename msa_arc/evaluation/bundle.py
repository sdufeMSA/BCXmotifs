"""Single-run evaluation bundle: predictions in, every reported number out.

One call produces the whole set of figures a run contributes to the paper --
the three per-output scores, the with- and without-MCVA attitude results, the
divergence-subset breakdown, the confidence-band accuracy of the reconciliation
rule, and the calibration of the five-class distribution.

Keeping this in one place is what stops the with-MCVA and without-MCVA numbers
from being computed by two code paths that could drift apart; they differ here
only by which category column is scored.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from msa_arc.config import MCVAConfig
from msa_arc.evaluation.calibration import (
    CalibrationMetrics,
    IntensityCalibration,
    calibration_metrics,
    intensity_calibration,
)
from msa_arc.evaluation.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    attitude_metrics,
    divergence_subset_metrics,
    macro_average_accuracy,
    polarity_metrics,
    regression_metrics,
)
from msa_arc.inference.decode import InstancePrediction
from msa_arc.mcva.confidence import (
    ConfidenceBands,
    band_reconciliations,
    fit_confidence_bands,
)
from msa_arc.mcva.reconcile import reconcile_batch

logger = logging.getLogger(__name__)


@dataclass
class EvaluationBundle:
    """Everything one run reports.

    Attributes:
        attitude: Attitude scores after MCVA reconciliation.
        attitude_without_mcva: The same scores on the surface categories, which
            is the "without MCVA" ablation row.
        polarity: Polarity scores.
        intensity: Intensity scores.
        divergence: Per-pattern attitude scores on the divergence subset.
        divergence_macro_accuracy: Macro average over the four patterns.
        calibration: Calibration of the five-class distribution.
        intensity_calibration: Binned calibration of the continuous intensity
            output, as the manuscript's intensity-calibration table reports it.
        mcva: Reconciliation counts, band accuracy and the review queue size.
        n_instances: Raw instance count backing every figure above.
    """

    attitude: ClassificationMetrics
    attitude_without_mcva: ClassificationMetrics
    polarity: ClassificationMetrics
    intensity: RegressionMetrics
    divergence: Dict[str, ClassificationMetrics]
    divergence_macro_accuracy: float
    calibration: CalibrationMetrics
    intensity_calibration: IntensityCalibration
    mcva: Dict[str, Any] = field(default_factory=dict)
    n_instances: int = 0


def predictions_to_frame(predictions: Sequence[InstancePrediction]) -> pd.DataFrame:
    """Flatten predictions into a frame, one row per instance."""
    return pd.DataFrame(
        [
            {
                "instance_key": p.instance_key,
                "participant_id": p.participant_id,
                "service_id": p.service_id,
                "scenario": p.scenario,
                "pred_polarity": p.polarity,
                "pred_intensity": p.intensity,
                "pred_category_surface": p.category,
                "raw_text": p.raw_text,
                "decode_stage": p.decode_stage,
                **{f"prob_{i}": value for i, value in enumerate(p.probabilities)},
            }
            for p in predictions
        ]
    )


def apply_mcva(
    frame: pd.DataFrame,
    bands: ConfidenceBands,
    cfg: Optional[MCVAConfig] = None,
) -> pd.DataFrame:
    """Attach reconciled categories, branch ids, confidences and bands.

    Args:
        frame: Prediction frame from :func:`predictions_to_frame`.
        bands: Cut-points fitted on the validation split.
        cfg: Threshold configuration.

    Returns:
        The frame with ``pred_category``, ``mcva_branch``, ``mcva_confidence``,
        ``mcva_band`` and ``mcva_applied`` columns added.
    """
    cfg = cfg or MCVAConfig()
    reconciliations = reconcile_batch(
        frame["pred_category_surface"].tolist(),
        frame["pred_polarity"].tolist(),
        frame["pred_intensity"].tolist(),
        cfg,
    )
    banded = band_reconciliations(
        reconciliations, frame["pred_intensity"].tolist(), bands, cfg
    )

    result = frame.copy()
    result["mcva_branch"] = [b.reconciliation.branch for b in banded]
    result["mcva_proposed"] = [b.reconciliation.category for b in banded]
    result["mcva_confidence"] = [b.confidence for b in banded]
    result["mcva_band"] = [b.band for b in banded]
    result["mcva_applied"] = [b.applied for b in banded]
    result["pred_category"] = [b.final_category for b in banded]
    return result


def fit_bands_on_validation(
    frame: pd.DataFrame, cfg: Optional[MCVAConfig] = None, strict: bool = False
) -> ConfidenceBands:
    """Fit the confidence tertiles on validation-split reclassifications.

    Args:
        frame: Prediction frame for the validation split only.
        cfg: Threshold configuration.
        strict: Raise rather than fall back when the split produced too few
            reclassifications to calibrate.

    Returns:
        The fitted bands, to be frozen before the test split is scored. When too
        few reclassifications were available the returned bands route every
        reclassification to human review and carry ``n_fitted == 0``.
    """
    from msa_arc.mcva.confidence import reconciliation_confidence

    cfg = cfg or MCVAConfig()
    reconciliations = reconcile_batch(
        frame["pred_category_surface"].tolist(),
        frame["pred_polarity"].tolist(),
        frame["pred_intensity"].tolist(),
        cfg,
    )
    confidences = [
        reconciliation_confidence(r.branch, intensity, cfg)
        for r, intensity in zip(
            reconciliations, frame["pred_intensity"].tolist(), strict=False
        )
        if r.changed
    ]
    return fit_confidence_bands([c for c in confidences if c is not None], strict=strict)


def mcva_summary(frame: pd.DataFrame, truth: Sequence[str]) -> Dict[str, Any]:
    """Count and score what the reconciliation rule actually did.

    Args:
        frame: Prediction frame after :func:`apply_mcva`.
        truth: Human attitude labels, in frame order.

    Returns:
        Interventions per branch, per-band accuracy of the reclassification, how
        many instances the rule moved from wrong to right and right to wrong,
        and the size of the review queue.
    """
    working = frame.copy()
    working["truth"] = list(truth)

    proposed = working[working["mcva_branch"] != 0]
    applied = working[working["mcva_applied"]]
    queued = proposed[~proposed["mcva_applied"]]

    surface_correct = working["pred_category_surface"] == working["truth"]
    final_correct = working["pred_category"] == working["truth"]

    per_band: Dict[str, Dict[str, float]] = {}
    for band, group in proposed.groupby("mcva_band", dropna=True):
        hits = int((group["mcva_proposed"] == group["truth"]).sum())
        per_band[str(band)] = {
            "n": int(len(group)),
            "n_correct": hits,
            "accuracy": hits / len(group) if len(group) else float("nan"),
        }

    return {
        "n_proposed": int(len(proposed)),
        "n_applied": int(len(applied)),
        "n_review_queue": int(len(queued)),
        "per_branch": proposed["mcva_branch"].value_counts().sort_index().to_dict(),
        "per_band": per_band,
        "n_wrong_to_right": int((~surface_correct & final_correct).sum()),
        "n_right_to_wrong": int((surface_correct & ~final_correct).sum()),
    }


def evaluate_predictions(
    predictions: Sequence[InstancePrediction],
    truth: pd.DataFrame,
    bands: ConfidenceBands,
    cfg: Optional[MCVAConfig] = None,
) -> tuple[EvaluationBundle, pd.DataFrame]:
    """Score one run end to end.

    Args:
        predictions: Decoded predictions for the split.
        truth: Manifest rows for the same split, carrying the three label
            columns, ``divergence_pattern`` and ``instance_key``.
        bands: Confidence cut-points fitted on the validation split.
        cfg: Threshold configuration.

    Returns:
        The bundle and the merged per-instance frame, which is written out so a
        reader can recompute every number in the bundle.

    Raises:
        ValueError: If predictions and labels do not cover the same instances.
    """
    frame = apply_mcva(predictions_to_frame(predictions), bands, cfg)

    label_columns = [
        "instance_key",
        "label_polarity",
        "label_intensity",
        "label_category",
        "divergence_pattern",
    ]
    merged = frame.merge(truth[label_columns], on="instance_key", how="inner")
    if len(merged) != len(frame):
        raise ValueError(
            f"{len(frame) - len(merged)} predicted instances have no labels; "
            "predictions and manifest disagree"
        )

    attitude = attitude_metrics(merged["label_category"], merged["pred_category"])
    without_mcva = attitude_metrics(merged["label_category"], merged["pred_category_surface"])
    divergence = divergence_subset_metrics(
        merged["label_category"],
        merged["pred_category"],
        merged["divergence_pattern"].tolist(),
    )
    probability_columns = [c for c in merged.columns if c.startswith("prob_")]

    bundle = EvaluationBundle(
        attitude=attitude,
        attitude_without_mcva=without_mcva,
        polarity=polarity_metrics(merged["label_polarity"], merged["pred_polarity"]),
        intensity=regression_metrics(merged["label_intensity"], merged["pred_intensity"]),
        divergence=divergence,
        divergence_macro_accuracy=macro_average_accuracy(divergence),
        calibration=calibration_metrics(
            merged[probability_columns].to_numpy(), merged["label_category"]
        ),
        intensity_calibration=intensity_calibration(
            merged["pred_intensity"], merged["label_intensity"]
        ),
        mcva=mcva_summary(merged, merged["label_category"]),
        n_instances=len(merged),
    )
    logger.info(
        "Attitude accuracy %d/%d = %.4f (without MCVA %d/%d = %.4f)",
        attitude.n_correct,
        attitude.n_total,
        attitude.accuracy,
        without_mcva.n_correct,
        without_mcva.n_total,
        without_mcva.accuracy,
    )
    return bundle, merged


__all__ = [
    "EvaluationBundle",
    "apply_mcva",
    "evaluate_predictions",
    "fit_bands_on_validation",
    "mcva_summary",
    "predictions_to_frame",
]
