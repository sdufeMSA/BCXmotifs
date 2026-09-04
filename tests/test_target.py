"""The unified target string must round-trip, and must fail loudly when it can't."""

import pytest

from msa_arc.constants import ATTITUDE_CLASSES, POLARITY_CLASSES
from msa_arc.model.target import (
    all_category_candidates,
    category_prefix,
    format_intensity,
    parse_target,
    serialize_target,
)


@pytest.mark.parametrize("polarity", POLARITY_CLASSES)
@pytest.mark.parametrize("category", ATTITUDE_CLASSES)
@pytest.mark.parametrize("intensity", [-1.0, -0.85, -0.3, 0.0, 0.3, 0.85, 1.0])
def test_round_trip(polarity: str, category: str, intensity: float) -> None:
    parsed = parse_target(serialize_target(polarity, intensity, category))
    assert parsed.ok
    assert parsed.polarity == polarity
    assert parsed.category == category
    assert parsed.intensity == pytest.approx(intensity, abs=0.005)


def test_intensity_is_rendered_to_two_decimals() -> None:
    assert serialize_target("positive", 0.856, "Like") == "positive | 0.86 | Like"
    assert format_intensity(-0.5) == "-0.50"


def test_intensity_is_clamped_to_the_label_range() -> None:
    assert format_intensity(3.0) == "1.00"
    assert format_intensity(-9.0) == "-1.00"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "positive | Like",
        "positive | 0.85",
        "happy | 0.85 | Like",
        "positive | 0.85 | Adores",
        "positive 0.85 Like",
        "some free-form model chatter",
    ],
)
def test_malformed_strings_do_not_parse(text: str) -> None:
    parsed = parse_target(text)
    assert not parsed.ok
    assert parsed.category is None
    assert parsed.raw == text


def test_whitespace_variation_is_tolerated() -> None:
    assert parse_target("  positive|0.85|Like ").ok
    assert parse_target("positive   |   0.85   |   Live with").category == "Live with"


def test_unknown_labels_are_rejected_at_serialisation() -> None:
    with pytest.raises(ValueError, match="unknown polarity"):
        serialize_target("happy", 0.5, "Like")
    with pytest.raises(ValueError, match="unknown category"):
        serialize_target("positive", 0.5, "Adores")


def test_candidates_share_a_prefix_and_differ_only_in_category() -> None:
    candidates = all_category_candidates("negative", -0.42)
    prefix = category_prefix("negative", -0.42)
    assert len(candidates) == len(ATTITUDE_CLASSES)
    assert all(c.startswith(prefix) for c in candidates)
    assert [c[len(prefix) :] for c in candidates] == list(ATTITUDE_CLASSES)
