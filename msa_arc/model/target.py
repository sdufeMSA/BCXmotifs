"""The unified target string that couples the three MSA-ARC outputs.

Section 4.1.3 of the manuscript specifies a single target sequence

    ``<polarity> | <intensity> | <category>``

with intensity rendered to two decimal places.  The model is trained to
generate that string token by token, which is why no multi-task weighting
between the three outputs is required: their relative influence is fixed by
their token lengths in the shared string.

This module owns both directions of that contract.  Serialisation is used to
build training targets; parsing is used at decode time and is deliberately
strict, because a silently mis-parsed string would corrupt the MCVA rule and
then the Kano matrix downstream.  ``ParseResult`` therefore reports failure
explicitly rather than guessing.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from msa_arc.constants import (
    ATTITUDE_CLASSES,
    INTENSITY_DECIMALS,
    INTENSITY_MAX,
    INTENSITY_MIN,
    POLARITY_CLASSES,
)

SEPARATOR = " | "

#: Matches exactly the strings ``serialize_target`` produces, tolerating only
#: whitespace variation around the separators.
_TARGET_RE = re.compile(
    r"^\s*(?P<polarity>{polarities})\s*\|\s*"
    r"(?P<intensity>[+-]?\d+(?:\.\d+)?)\s*\|\s*"
    r"(?P<category>{categories})\s*$".format(
        polarities="|".join(re.escape(p) for p in POLARITY_CLASSES),
        categories="|".join(re.escape(c) for c in ATTITUDE_CLASSES),
    )
)


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing one generated string.

    Attributes:
        ok: Whether the string matched the target grammar.
        polarity: One of ``POLARITY_CLASSES``, or ``None`` on failure.
        intensity: Value clamped to ``[-1, 1]``, or ``None`` on failure.
        category: One of ``ATTITUDE_CLASSES``, or ``None`` on failure.
        raw: The string that was parsed, kept for the failure audit log.
    """

    ok: bool
    polarity: Optional[str]
    intensity: Optional[float]
    category: Optional[str]
    raw: str


def format_intensity(value: float) -> str:
    """Render an intensity to the fixed-width form used in the target string."""
    clamped = min(max(float(value), INTENSITY_MIN), INTENSITY_MAX)
    return f"{clamped:.{INTENSITY_DECIMALS}f}"


def serialize_target(polarity: str, intensity: float, category: str) -> str:
    """Build the unified target string for one labelled instance.

    Args:
        polarity: One of ``POLARITY_CLASSES``.
        intensity: Continuous value on ``[-1, 1]``; values outside are clamped.
        category: One of ``ATTITUDE_CLASSES``.

    Returns:
        The target string, e.g. ``"positive | 0.85 | Like"``.

    Raises:
        ValueError: If ``polarity`` or ``category`` is outside its label space.
            Training targets are never guessed at.
    """
    if polarity not in POLARITY_CLASSES:
        raise ValueError(f"unknown polarity {polarity!r}; expected one of {POLARITY_CLASSES}")
    if category not in ATTITUDE_CLASSES:
        raise ValueError(f"unknown category {category!r}; expected one of {ATTITUDE_CLASSES}")
    return f"{polarity}{SEPARATOR}{format_intensity(intensity)}{SEPARATOR}{category}"


def parse_target(text: str) -> ParseResult:
    """Parse a generated string back into the triplet.

    Args:
        text: A decoded model output.

    Returns:
        A :class:`ParseResult`. ``ok`` is ``False`` whenever the string does not
        match the grammar exactly; callers must then re-decode or fall back
        rather than treat the instance as predicted.
    """
    match = _TARGET_RE.match(text)
    if match is None:
        return ParseResult(ok=False, polarity=None, intensity=None, category=None, raw=text)

    intensity = float(match.group("intensity"))
    intensity = min(max(intensity, INTENSITY_MIN), INTENSITY_MAX)
    return ParseResult(
        ok=True,
        polarity=match.group("polarity"),
        intensity=intensity,
        category=match.group("category"),
        raw=text,
    )


def category_prefix(polarity: str, intensity: float) -> str:
    """The shared prefix that precedes the category in the target string.

    The constrained rescoring of :mod:`msa_arc.inference.probabilities` holds
    this prefix fixed across all five category candidates, so its log-likelihood
    is identical for every candidate and cancels in the softmax.  Only the
    category continuation has to be scored.

    Args:
        polarity: The decoded polarity.
        intensity: The decoded intensity.

    Returns:
        The prefix string, ending with the separator.
    """
    return f"{polarity}{SEPARATOR}{format_intensity(intensity)}{SEPARATOR}"


def all_category_candidates(polarity: str, intensity: float) -> List[str]:
    """Full target strings for all five categories under a fixed prefix.

    Returns:
        One string per element of ``ATTITUDE_CLASSES``, in that order.
    """
    prefix = category_prefix(polarity, intensity)
    return [f"{prefix}{category}" for category in ATTITUDE_CLASSES]


__all__ = [
    "SEPARATOR",
    "ParseResult",
    "all_category_candidates",
    "category_prefix",
    "format_intensity",
    "parse_target",
    "serialize_target",
]
