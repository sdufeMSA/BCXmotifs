"""Eq. 1 branches, Eq. 2 confidence, and the review-queue disposition."""

import pytest

from msa_arc.config import MCVAConfig
from msa_arc.mcva.confidence import (
    ConfidenceBands,
    band_reconciliations,
    branch_bounds,
    fit_confidence_bands,
    reconciliation_confidence,
)
from msa_arc.mcva.reconcile import NO_RECONCILIATION, reconcile, reconcile_batch

CFG = MCVAConfig()


def test_branch_one_demotes_strong_negative_endorsement_to_dislike() -> None:
    result = reconcile("Like", "negative", -0.9, CFG)
    assert (result.category, result.branch, result.changed) == ("Dislike", 1, True)
    assert reconcile("Essential", "negative", -0.75, CFG).category == "Dislike"


def test_branch_two_demotes_moderate_negative_endorsement_to_live_with() -> None:
    result = reconcile("Like", "negative", -0.5, CFG)
    assert (result.category, result.branch) == ("Live with", 2)


def test_branch_three_demotes_negative_neutral_to_live_with() -> None:
    result = reconcile("Neutral", "negative", -0.6, CFG)
    assert (result.category, result.branch) == ("Live with", 3)


@pytest.mark.parametrize("polarity", ["positive", "neutral"])
def test_branch_four_promotes_aging_denial_to_essential(polarity: str) -> None:
    result = reconcile("Dislike", polarity, 0.55, CFG)
    assert (result.category, result.branch) == ("Essential", 4)


@pytest.mark.parametrize(
    ("category", "polarity", "intensity"),
    [
        ("Like", "positive", 0.9),  # sincere endorsement, nothing to correct
        ("Like", "negative", -0.2),  # below tau_flag
        ("Neutral", "neutral", 0.0),  # politeness: handled upstream, not here
        ("Dislike", "negative", -0.9),  # consistent dislike
        ("Live with", "negative", -0.9),  # already the demoted class
    ],
)
def test_otherwise_branch_leaves_the_surface_category_alone(
    category: str, polarity: str, intensity: float
) -> None:
    result = reconcile(category, polarity, intensity, CFG)
    assert result.branch == NO_RECONCILIATION
    assert result.category == category
    assert not result.changed


def test_no_branch_ever_assigns_neutral() -> None:
    """Every correction moves away from the midpoint; none moves toward it."""
    from msa_arc.constants import ATTITUDE_CLASSES, POLARITY_CLASSES

    for category in ATTITUDE_CLASSES:
        for polarity in POLARITY_CLASSES:
            for intensity in [i / 20 for i in range(-20, 21)]:
                result = reconcile(category, polarity, intensity, CFG)
                if result.changed:
                    assert result.category != "Neutral"


def test_disabling_the_rule_is_the_without_mcva_ablation() -> None:
    disabled = MCVAConfig(enabled=False)
    result = reconcile("Like", "negative", -0.9, disabled)
    assert result.category == "Like"
    assert result.branch == NO_RECONCILIATION


def test_confidence_uses_the_magnitude_and_stays_in_the_unit_interval() -> None:
    """The published Eq. 2 omits the absolute value and goes negative here."""
    for branch, intensity in [(1, -0.9), (2, -0.5), (3, -0.6), (4, 0.55)]:
        confidence = reconciliation_confidence(branch, intensity, CFG)
        assert confidence is not None
        assert 0.0 <= confidence <= 1.0


def test_confidence_bounds_match_the_branch_that_fired() -> None:
    bounds = branch_bounds(CFG)
    assert bounds[1] == (0.7, 1.0)
    assert bounds[2] == (0.3, 0.7)
    assert bounds[3] == (0.3, 1.0)
    assert bounds[4] == (0.3, 1.0)
    # An intensity exactly on the lower bound clears it by nothing.
    assert reconciliation_confidence(1, -0.7, CFG) == pytest.approx(0.0)
    assert reconciliation_confidence(1, -1.0, CFG) == pytest.approx(1.0)
    assert reconciliation_confidence(2, -0.5, CFG) == pytest.approx(0.5)


def test_confidence_is_none_when_no_branch_fired() -> None:
    assert reconciliation_confidence(NO_RECONCILIATION, -0.9, CFG) is None


def test_too_few_observations_route_everything_to_review() -> None:
    """A run must not be discarded because the rule rarely fired on validation."""
    bands = fit_confidence_bands([0.1, 0.2])
    assert not bands.fitted
    assert bands.n_fitted == 0
    assert bands.band_of(0.0) == "low"
    assert bands.band_of(1.0) == "low"


def test_strict_mode_still_refuses_to_guess() -> None:
    with pytest.raises(ValueError, match="too few to fit"):
        fit_confidence_bands([0.1, 0.2], strict=True)


def test_the_review_everything_fallback_applies_nothing() -> None:
    banded = band_reconciliations(
        reconcile_batch(["Like"], ["negative"], [-0.95], CFG),
        [-0.95],
        ConfidenceBands.review_everything(),
        CFG,
    )
    assert banded[0].band == "low"
    assert not banded[0].applied
    assert banded[0].final_category == "Like"


def test_fitted_bands_report_their_sample_size() -> None:
    bands = fit_confidence_bands([0.1, 0.4, 0.7, 0.9])
    assert bands.fitted
    assert bands.n_fitted == 4


def test_tertiles_split_the_validation_confidences() -> None:
    bands = fit_confidence_bands([i / 100 for i in range(0, 100)])
    assert 0.0 < bands.lower < bands.upper < 1.0
    assert bands.band_of(0.05) == "low"
    assert bands.band_of(0.5) == "medium"
    assert bands.band_of(0.99) == "high"


def test_low_confidence_reclassifications_are_queued_not_applied() -> None:
    bands = ConfidenceBands(lower=0.4, upper=0.8, n_fitted=90)
    categories = ["Like", "Like", "Like"]
    polarities = ["negative"] * 3
    # Confidences on branch 1 (bounds 0.7..1.0): 0.10 low, 0.50 medium, 0.90 high.
    intensities = [-0.73, -0.85, -0.97]

    banded = band_reconciliations(
        reconcile_batch(categories, polarities, intensities, CFG),
        intensities,
        bands,
        CFG,
    )
    assert [b.band for b in banded] == ["low", "medium", "high"]
    assert [b.applied for b in banded] == [False, True, True]
    # The queued instance keeps its surface category until a human adjudicates.
    assert banded[0].final_category == "Like"
    assert banded[1].final_category == "Dislike"


def test_batch_reconciliation_rejects_ragged_input() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        reconcile_batch(["Like", "Neutral"], ["negative"], [-0.9], CFG)


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="tau_flag < tau_rev"):
        MCVAConfig(tau_rev=0.3, tau_flag=0.7)
