"""Empirical relationship among the three MSA-ARC outputs.

The three labels are elicited independently, so the way they line up on the
test set is an empirical finding rather than a definitional mapping.  Reporting
Reporting it lets a reader check that the outputs are non-redundant: that a
negative polarity under a surface Like really does occur, and is the sarcasm
pattern the reconciliation rule exists to catch.

Both tables are computed on the *human labels*, not the predictions.
"""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from msa_arc.constants import ATTITUDE_CLASSES, POLARITY_CLASSES

logger = logging.getLogger(__name__)


def polarity_by_attitude(polarities: Sequence[str], categories: Sequence[str]) -> pd.DataFrame:
    """Contingency table of polarity against attitude category.

    Args:
        polarities: Polarity labels.
        categories: Attitude category labels.

    Returns:
        Counts with polarity as rows and attitude as columns, both in their
        canonical order, with ``Total`` margins. Absent combinations appear as
        zeros rather than being dropped, so the table shape is stable across
        splits.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(polarities) != len(categories):
        raise ValueError(
            f"length mismatch: {len(polarities)} polarities vs {len(categories)} categories"
        )
    table = pd.crosstab(
        pd.Categorical(list(polarities), categories=list(POLARITY_CLASSES)),
        pd.Categorical(list(categories), categories=list(ATTITUDE_CLASSES)),
        dropna=False,
    )
    table = table.reindex(
        index=list(POLARITY_CLASSES), columns=list(ATTITUDE_CLASSES), fill_value=0
    )
    table.index.name = "polarity"
    table.columns.name = "attitude"
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    return table


def intensity_by_attitude(
    intensities: Sequence[float], categories: Sequence[str]
) -> pd.DataFrame:
    """Distribution of intensity within each attitude category.

    Args:
        intensities: Intensity labels on ``[-1, 1]``.
        categories: Attitude category labels.

    Returns:
        One row per attitude category with count, mean, standard deviation, and
        the three quartiles of the intensity label.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(intensities) != len(categories):
        raise ValueError(
            f"length mismatch: {len(intensities)} intensities vs {len(categories)} categories"
        )
    frame = pd.DataFrame(
        {"attitude": list(categories), "intensity": np.asarray(intensities, dtype=float)}
    )
    grouped = frame.groupby("attitude")["intensity"]
    summary = pd.DataFrame(
        {
            "n": grouped.count(),
            "mean": grouped.mean(),
            "sd": grouped.std(ddof=1),
            "q25": grouped.quantile(0.25),
            "median": grouped.median(),
            "q75": grouped.quantile(0.75),
        }
    )
    return summary.reindex(list(ATTITUDE_CLASSES))


__all__ = ["intensity_by_attitude", "polarity_by_attitude"]
