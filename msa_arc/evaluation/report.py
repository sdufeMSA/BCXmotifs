"""Aggregating runs and writing results to disk.

Two different kinds of number are reported, and this module keeps them apart by
construction:

* **the primary frozen run** (``constants.PRIMARY_SEED``) supplies every raw
  count: the confusion matrices, the MCVA intervention counts, the
  numerator and denominator behind an accuracy;
* **the ten-seed protocol** supplies mean +/- SD for every scalar metric.

``aggregate_runs`` refuses to produce a mean over a set of runs that does not
include the primary seed, so a table can never quote a mean whose companion raw
counts came from a run outside the set.
"""

import json
import logging
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

from msa_arc.constants import PRIMARY_SEED
from msa_arc.evaluation.bundle import EvaluationBundle
from msa_arc.evaluation.relations import intensity_by_attitude, polarity_by_attitude

logger = logging.getLogger(__name__)


def _to_serialisable(value: Any) -> Any:
    """Convert dataclasses and numpy scalars into JSON-safe values.

    Non-finite floats become ``null``. A correlation over a zero-variance series
    and an empty calibration bin both legitimately produce ``nan``, and Python
    would write that as the bare token ``NaN``, which no strict JSON parser
    accepts, so the metrics files would be unreadable from jq, JavaScript or R.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_serialisable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _to_serialisable(value.tolist())
    return value


def dump_json(payload: Any, path: Path) -> None:
    """Write JSON that a strict parser will accept.

    Args:
        payload: Any nested structure of dataclasses, mappings and scalars.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_serialisable(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def scalar_metrics(bundle: EvaluationBundle) -> Dict[str, float]:
    """Flatten a bundle to the scalars that get averaged across seeds.

    Args:
        bundle: One run's evaluation.

    Returns:
        A flat mapping of metric name to value. Counts are deliberately absent:
        averaging a confusion-matrix cell across seeds would produce a number
        that describes no actual run.
    """
    metrics = {
        "attitude_accuracy": bundle.attitude.accuracy,
        "attitude_macro_f1": bundle.attitude.macro_f1,
        "attitude_accuracy_without_mcva": bundle.attitude_without_mcva.accuracy,
        "attitude_macro_f1_without_mcva": bundle.attitude_without_mcva.macro_f1,
        "polarity_accuracy": bundle.polarity.accuracy,
        "polarity_macro_f1": bundle.polarity.macro_f1,
        "intensity_mae": bundle.intensity.mae,
        "intensity_rmse": bundle.intensity.rmse,
        "intensity_pearson": bundle.intensity.pearson,
        "intensity_spearman": bundle.intensity.spearman,
        "divergence_macro_accuracy": bundle.divergence_macro_accuracy,
        "calibration_ece": bundle.calibration.ece,
        "calibration_brier": bundle.calibration.brier,
    }
    for pattern, pattern_metrics in bundle.divergence.items():
        metrics[f"divergence_{pattern}_accuracy"] = pattern_metrics.accuracy
    return metrics


def aggregate_runs(
    bundles: Mapping[int, EvaluationBundle],
    require_primary: bool = True,
) -> pd.DataFrame:
    """Mean and standard deviation of every scalar metric across seeds.

    Args:
        bundles: Evaluation bundles keyed by random seed.
        require_primary: Refuse to aggregate a set that omits the primary seed,
            since the paper's raw counts come from that run and the mean must
            describe the same set of runs.

    Returns:
        A frame indexed by metric with ``mean``, ``sd``, ``n_runs`` and the
        primary run's own value.

    Raises:
        ValueError: If no bundles are supplied, or the primary seed is missing
            while ``require_primary`` is set.
    """
    if not bundles:
        raise ValueError("no runs to aggregate")
    if require_primary and PRIMARY_SEED not in bundles:
        raise ValueError(
            f"primary seed {PRIMARY_SEED} is missing from the aggregated runs "
            f"{sorted(bundles)}; the reported raw counts come from that run"
        )

    frame = pd.DataFrame({seed: scalar_metrics(b) for seed, b in bundles.items()}).T
    summary = pd.DataFrame(
        {
            "mean": frame.mean(axis=0),
            "sd": frame.std(axis=0, ddof=1) if len(frame) > 1 else 0.0,
            "n_runs": len(frame),
        }
    )
    if PRIMARY_SEED in bundles:
        summary["primary_run"] = frame.loc[PRIMARY_SEED]
    return summary


def _labelled_confusion(metrics) -> pd.DataFrame:
    """Turn a metrics object's confusion matrix into a margined frame."""
    frame = pd.DataFrame(metrics.confusion, index=metrics.labels, columns=metrics.labels)
    frame.index.name = "true"
    frame.columns.name = "predicted"
    frame["Total"] = frame.sum(axis=1)
    frame.loc["Total"] = frame.sum(axis=0)
    return frame


def confusion_frame(bundle: EvaluationBundle, with_mcva: bool = True) -> pd.DataFrame:
    """The attitude confusion matrix in raw counts, labelled on both axes.

    Args:
        bundle: The primary run's evaluation.
        with_mcva: Whether to take the reconciled matrix or the surface one.

    Returns:
        Rows are human labels, columns are predictions, with ``Total`` margins.
    """
    return _labelled_confusion(bundle.attitude if with_mcva else bundle.attitude_without_mcva)


def polarity_confusion_frame(bundle: EvaluationBundle) -> pd.DataFrame:
    """The 3x3 polarity confusion matrix in raw counts."""
    return _labelled_confusion(bundle.polarity)


def per_class_frame(bundle: EvaluationBundle, with_mcva: bool = True) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support."""
    metrics = bundle.attitude if with_mcva else bundle.attitude_without_mcva
    return pd.DataFrame(metrics.per_class).T


def intensity_calibration_frame(bundle: EvaluationBundle) -> pd.DataFrame:
    """The binned intensity calibration, laid out as the manuscript prints it."""
    calibration = bundle.intensity_calibration
    frame = pd.DataFrame(
        [
            {
                "bin": b.index,
                "lower": b.lower,
                "upper": b.upper,
                "mean_predicted": b.mean_predicted,
                "mean_labelled": b.mean_labelled,
                "n": b.count,
                "discrepancy": b.discrepancy,
            }
            for b in calibration.bins
        ]
    )
    frame.attrs["ece"] = calibration.ece
    return frame


def write_run(
    output_dir: Union[str, Path],
    seed: int,
    bundle: EvaluationBundle,
    predictions: pd.DataFrame,
    decode_report: Optional[Any] = None,
) -> Path:
    """Persist one run's predictions, metrics and confusion matrices.

    Args:
        output_dir: Root output directory.
        seed: The run's random seed; becomes the subdirectory name.
        bundle: The run's evaluation.
        predictions: Per-instance prediction frame.
        decode_report: Optional decoding report, written alongside so the
            parse-failure rate is auditable.

    Returns:
        The directory written.
    """
    run_dir = Path(output_dir) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(run_dir / "predictions.csv", index=False)
    confusion_frame(bundle, with_mcva=True).to_csv(run_dir / "confusion_with_mcva.csv")
    confusion_frame(bundle, with_mcva=False).to_csv(run_dir / "confusion_without_mcva.csv")
    per_class_frame(bundle, with_mcva=True).to_csv(run_dir / "per_class.csv")
    polarity_confusion_frame(bundle).to_csv(run_dir / "confusion_polarity.csv")
    intensity_calibration_frame(bundle).to_csv(
        run_dir / "intensity_calibration.csv", index=False
    )
    polarity_by_attitude(predictions["label_polarity"], predictions["label_category"]).to_csv(
        run_dir / "output_relation_polarity_by_attitude.csv"
    )
    intensity_by_attitude(
        predictions["label_intensity"], predictions["label_category"]
    ).to_csv(run_dir / "output_relation_intensity_by_attitude.csv")

    review_queue = predictions[
        (predictions["mcva_branch"] != 0) & (~predictions["mcva_applied"])
    ]
    # The adjudicator must not see which way the rule wanted to move the label.
    review_columns = [
        c
        for c in review_queue.columns
        if c not in {"mcva_proposed", "mcva_branch", "mcva_band"}
    ]
    review_queue[review_columns].to_csv(run_dir / "review_queue.csv", index=False)

    payload: Dict[str, Any] = {"seed": seed, "metrics": _to_serialisable(bundle)}
    if decode_report is not None:
        payload["decode"] = {
            "n_instances": decode_report.n_instances,
            "n_greedy_retries": decode_report.n_greedy_retries,
            "n_fallbacks": decode_report.n_fallbacks,
            "parse_failure_rate": decode_report.parse_failure_rate,
            "failures": decode_report.failures,
        }
    dump_json(payload, run_dir / "metrics.json")
    logger.info("Wrote run artefacts for seed %d to %s", seed, run_dir)
    return run_dir


def write_summary(
    output_dir: Union[str, Path],
    bundles: Mapping[int, EvaluationBundle],
) -> Path:
    """Write the across-seed summary table.

    Args:
        output_dir: Root output directory.
        bundles: Evaluation bundles keyed by seed.

    Returns:
        The path of the summary CSV.
    """
    summary = aggregate_runs(bundles)
    path = Path(output_dir) / "summary_across_seeds.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path)
    logger.info(
        "Wrote %d-seed summary to %s (primary run: seed %d)",
        len(bundles),
        path,
        PRIMARY_SEED,
    )
    return path


__all__ = [
    "aggregate_runs",
    "dump_json",
    "confusion_frame",
    "intensity_calibration_frame",
    "per_class_frame",
    "polarity_confusion_frame",
    "scalar_metrics",
    "write_run",
    "write_summary",
]
