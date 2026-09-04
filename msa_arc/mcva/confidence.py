"""Reconciliation confidence and human-review banding (Eq. 2, Section 4.1.4).

Applying Eq. 1 as a hard switch would treat an instance whose intensity barely
crosses a threshold identically to one that clears it decisively, which is
untenable when the reconciled category propagates into the Kano matrix and then
into service-bundle recommendations.  Every reclassification therefore carries a
confidence score.

The confidence is the margin by which the intensity *magnitude* clears the
boundary that triggered the branch, normalised by the width of the admissible
interval:

    c = (|r_hat| - tau_lo) / (tau_hi - tau_lo),    c in [0, 1]

The magnitude is what enters the numerator, since three of the four branches
fire on negative intensity.  Per-branch bounds are given by :func:`branch_bounds`,
with ``tau_hi = 1`` for the branches that are unbounded above.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from msa_arc.config import MCVAConfig
from msa_arc.mcva.reconcile import NO_RECONCILIATION, Reconciliation

logger = logging.getLogger(__name__)

BANDS: Tuple[str, ...] = ("low", "medium", "high")

#: A cut-point above every attainable confidence. Eq. 2 is clipped to [0, 1], so
#: any finite value greater than 1 sends every reclassification to the low band.
#: A finite sentinel is used rather than infinity because these cut-points are
#: written to ``history.json``, and ``Infinity`` is not valid JSON.
ABOVE_MAX_CONFIDENCE: float = 2.0


def branch_bounds(cfg: MCVAConfig) -> Dict[int, Tuple[float, float]]:
    """Intensity bounds ``(tau_lo, tau_hi)`` of each Eq. 1 branch.

    Args:
        cfg: Threshold configuration.

    Returns:
        Mapping from branch number to its admissible intensity interval. Branch
        2 is the only bounded-above branch; the others run to 1.
    """
    return {
        1: (cfg.tau_rev, 1.0),
        2: (cfg.tau_flag, cfg.tau_rev),
        3: (cfg.tau_flag, 1.0),
        4: (cfg.tau_flag, 1.0),
    }


def reconciliation_confidence(
    branch: int, intensity: float, cfg: Optional[MCVAConfig] = None
) -> Optional[float]:
    """Confidence of one reclassification.

    Args:
        branch: The Eq. 1 branch that fired.
        intensity: The generated intensity ``r_hat``.
        cfg: Threshold configuration.

    Returns:
        The margin by which ``|r_hat|`` clears the boundary that triggered the
        branch, normalised by the width of the admissible interval and clipped
        to ``[0, 1]``. Returns ``None`` when no branch fired, since an
        untouched prediction has no reconciliation to be confident about.
    """
    if branch == NO_RECONCILIATION:
        return None
    cfg = cfg or MCVAConfig()
    low, high = branch_bounds(cfg)[branch]
    value = (abs(intensity) - low) / (high - low)
    return min(max(value, 0.0), 1.0)


@dataclass(frozen=True)
class ConfidenceBands:
    """Tertile cut-points fitted on the validation split.

    Attributes:
        lower: Confidence below this is ``low``.
        upper: Confidence at or above this is ``high``.
        n_fitted: How many validation reclassifications the cut-points came
            from. Zero marks the uncalibrated fallback, in which every
            reclassification is routed to human review.
    """

    lower: float
    upper: float
    n_fitted: int

    @property
    def fitted(self) -> bool:
        """Whether real validation data produced these cut-points."""
        return self.n_fitted > 0

    @classmethod
    def review_everything(cls) -> "ConfidenceBands":
        """Cut-points that send every reclassification to human review.

        Used when the validation split produced too few reclassifications to
        fit tertiles. Auto-accepting on cut-points that were never calibrated
        would be worse than reviewing everything, and ``n_fitted == 0`` records
        in the run artefacts that this is what happened.
        """
        return cls(lower=ABOVE_MAX_CONFIDENCE, upper=ABOVE_MAX_CONFIDENCE, n_fitted=0)

    def band_of(self, confidence: float) -> str:
        """Assign a confidence to ``low``, ``medium`` or ``high``."""
        if confidence < self.lower:
            return "low"
        if confidence >= self.upper:
            return "high"
        return "medium"


def fit_confidence_bands(
    confidences: Sequence[float], strict: bool = False
) -> ConfidenceBands:
    """Fit tertile cut-points on validation-split reclassifications.

    The cut-points are a pipeline decision made outside the model-fitting loop,
    so they are fitted on the validation participants only and frozen before the
    test split is scored.

    A run can legitimately produce too few reclassifications to fit tertiles --
    an ablation with the rule effectively inert, or a small validation split.
    Raising there would discard a completed training run, so the default is to
    fall back to :meth:`ConfidenceBands.review_everything` with a warning; the
    resulting ``n_fitted == 0`` is written into the run artefacts, so the
    fallback cannot pass unnoticed.

    Args:
        confidences: Confidence values of every reclassification the rule made
            on the validation split.
        strict: Raise instead of falling back. Use when a caller genuinely
            requires calibrated bands.

    Returns:
        The fitted :class:`ConfidenceBands`, or the review-everything fallback.

    Raises:
        ValueError: If ``strict`` and fewer than three values are supplied.
    """
    values = sorted(float(c) for c in confidences)
    if len(values) < 3:
        message = (
            f"only {len(values)} validation reclassification(s); too few to fit "
            f"confidence tertiles"
        )
        if strict:
            raise ValueError(message)
        logger.warning(
            "%s. Falling back to routing every reclassification to human review "
            "(n_fitted=0); no reclassification will be applied automatically.",
            message,
        )
        return ConfidenceBands.review_everything()
    lower = _quantile(values, 1 / 3)
    upper = _quantile(values, 2 / 3)
    logger.info(
        "Fitted confidence tertiles on %d validation reclassifications: "
        "low < %.4f <= medium < %.4f <= high",
        len(values),
        lower,
        upper,
    )
    return ConfidenceBands(lower=lower, upper=upper, n_fitted=len(values))


def _quantile(sorted_values: List[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


@dataclass(frozen=True)
class BandedReconciliation:
    """A reconciliation together with its confidence, band and disposition.

    Attributes:
        reconciliation: The Eq. 1 outcome.
        confidence: Eq. 2 value, or ``None`` when no branch fired.
        band: ``low``/``medium``/``high``, or ``None`` when no branch fired.
        applied: Whether the reclassification enters the Kano matrix
            automatically. Low-confidence cases are held back for adjudication.
        final_category: The category that enters the Kano matrix given the
            disposition: the reconciled one when applied, the surface one
            otherwise. Human adjudication overwrites this downstream.
    """

    reconciliation: Reconciliation
    confidence: Optional[float]
    band: Optional[str]
    applied: bool
    final_category: str


def band_reconciliations(
    reconciliations: Iterable[Reconciliation],
    intensities: Sequence[float],
    bands: ConfidenceBands,
    cfg: Optional[MCVAConfig] = None,
) -> List[BandedReconciliation]:
    """Score, band and dispose of a batch of reconciliations.

    High- and medium-confidence reclassifications are accepted automatically.
    Low-confidence ones are not applied: they go to a review queue and are
    adjudicated by an annotator who sees the transcript, audio and video but not
    the direction of the proposed change.

    Args:
        reconciliations: Eq. 1 outcomes.
        intensities: The generated intensity of each instance, in the same order.
        bands: Cut-points fitted on the validation split.
        cfg: Threshold configuration; ``cfg.auto_accept_bands`` decides which
            bands are applied without review.

    Returns:
        One :class:`BandedReconciliation` per instance.
    """
    cfg = cfg or MCVAConfig()
    results: List[BandedReconciliation] = []
    for reconciliation, intensity in zip(reconciliations, intensities, strict=False):
        if not reconciliation.changed:
            results.append(
                BandedReconciliation(
                    reconciliation=reconciliation,
                    confidence=None,
                    band=None,
                    applied=False,
                    final_category=reconciliation.surface_category,
                )
            )
            continue

        confidence = reconciliation_confidence(reconciliation.branch, intensity, cfg)
        band = bands.band_of(confidence) if confidence is not None else None
        applied = band in cfg.auto_accept_bands
        results.append(
            BandedReconciliation(
                reconciliation=reconciliation,
                confidence=confidence,
                band=band,
                applied=applied,
                final_category=(
                    reconciliation.category if applied else reconciliation.surface_category
                ),
            )
        )
    return results


__all__ = [
    "ABOVE_MAX_CONFIDENCE",
    "BANDS",
    "BandedReconciliation",
    "ConfidenceBands",
    "band_reconciliations",
    "branch_bounds",
    "fit_confidence_bands",
    "reconciliation_confidence",
]
