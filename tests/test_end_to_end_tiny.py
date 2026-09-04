"""The whole Stage-B pipeline on a synthetic corpus and a tiny backbone.

Exercises the whole chain without the interview corpus: manifest -> splits ->
loaders -> training step -> decode -> MCVA -> metrics -> written artefacts.
"""

import json

import pytest
import torch

from msa_arc.config import DataConfig, DecodeConfig, LossConfig, MCVAConfig, TrainConfig
from msa_arc.data.dataset import MSAARCDataset
from msa_arc.data.loaders import build_dataloaders, prepare_manifest, split_frame
from msa_arc.evaluation.bundle import (
    evaluate_predictions,
    predictions_to_frame,
)
from msa_arc.evaluation.report import aggregate_runs, confusion_frame, write_run
from msa_arc.inference.decode import decode_dataset
from msa_arc.mcva.confidence import ConfidenceBands
from msa_arc.train import build_optimizer, build_scheduler, train_one_epoch, training_step


@pytest.fixture
def data_cfg(synthetic_corpus) -> DataConfig:
    return DataConfig(
        manifest_path=str(synthetic_corpus / "manifest.csv"),
        feature_dir=str(synthetic_corpus / "features"),
        splits_path=str(synthetic_corpus / "splits.csv"),
    )


@pytest.fixture
def train_cfg() -> TrainConfig:
    return TrainConfig(batch_size=4, num_workers=0, device="cpu", max_epochs=1)


def test_manifest_and_splits_load(data_cfg: DataConfig) -> None:
    manifest = prepare_manifest(data_cfg)
    assert set(manifest["split"]) == {"train", "validation", "test"}
    # Participant-level partition: no participant may straddle two splits.
    assert manifest.groupby("participant_id")["split"].nunique().max() == 1


def test_loaders_cover_every_split(data_cfg, train_cfg, tokenizer, tiny_model_config) -> None:
    manifest = prepare_manifest(data_cfg)
    loaders = build_dataloaders(manifest, tokenizer, data_cfg, tiny_model_config, train_cfg)
    assert set(loaders) == {"train", "validation", "test"}
    batch = next(iter(loaders["train"]))
    assert batch["input_ids"].shape[0] <= train_cfg.batch_size
    assert batch["labels"].shape[0] == batch["input_ids"].shape[0]
    assert batch["audio_features"].shape[-1] == tiny_model_config.audio_input_dim


def test_dataset_rejects_unlabelled_rows_when_labels_are_required(
    data_cfg, tiny_model_config
) -> None:
    import pandas as pd

    from msa_arc.features.manifest import load_manifest

    manifest = load_manifest(data_cfg.manifest_path)
    manifest.loc[0, "label_category"] = pd.NA
    with pytest.raises(ValueError, match="lack labels"):
        MSAARCDataset(manifest, data_cfg.feature_dir, audio_dim=8, video_dim=16)


def test_missing_text_features_are_a_hard_error(tmp_path, data_cfg, tiny_model_config) -> None:
    from msa_arc.features.manifest import load_manifest

    manifest = load_manifest(data_cfg.manifest_path)
    with pytest.raises(FileNotFoundError, match="no text features"):
        MSAARCDataset(manifest, tmp_path / "empty", audio_dim=8, video_dim=16)


def test_training_step_produces_finite_gradients(
    data_cfg, train_cfg, tokenizer, tiny_model, tiny_model_config
) -> None:
    manifest = prepare_manifest(data_cfg)
    loaders = build_dataloaders(manifest, tokenizer, data_cfg, tiny_model_config, train_cfg)
    batch = next(iter(loaders["train"]))

    loss, components = training_step(tiny_model, batch, LossConfig())
    loss.backward()

    assert torch.isfinite(loss)
    assert components["generation"] > 0
    grads = [p.grad for p in tiny_model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_one_epoch_runs_and_updates_the_adapters(
    data_cfg, train_cfg, tokenizer, tiny_model, tiny_model_config
) -> None:
    manifest = prepare_manifest(data_cfg)
    loaders = build_dataloaders(manifest, tokenizer, data_cfg, tiny_model_config, train_cfg)
    optimizer = build_optimizer(tiny_model, train_cfg)
    scheduler = build_scheduler(optimizer, len(loaders["train"]), train_cfg)

    # The up-projection is zero-initialised, so it is the first tensor to take a
    # gradient; the down-projection only starts moving once the up-projection is
    # non-zero. Checking the up-projection therefore tests the first step, and
    # exact inequality is the right comparison at a learning rate of 1e-5.
    before = tiny_model.adapters[0].up.weight.detach().clone()
    components = train_one_epoch(
        tiny_model,
        loaders["train"],
        optimizer,
        scheduler,
        torch.device("cpu"),
        train_cfg,
        LossConfig(),
    )
    assert "loss" in components
    assert not torch.equal(before, tiny_model.adapters[0].up.weight)
    assert torch.isfinite(tiny_model.adapters[0].up.weight).all()


def test_full_evaluation_writes_every_artefact(
    tmp_path, data_cfg, train_cfg, tokenizer, tiny_model, tiny_model_config
) -> None:
    manifest = prepare_manifest(data_cfg)
    loaders = build_dataloaders(manifest, tokenizer, data_cfg, tiny_model_config, train_cfg)
    device = torch.device("cpu")
    decode_cfg = DecodeConfig(num_beams=2, max_new_tokens=8)

    predictions, report = decode_dataset(
        tiny_model, tokenizer, loaders["test"], device, decode_cfg
    )
    assert len(predictions) == len(loaders["test"].dataset)

    # An untrained model may propose no reclassification at all, so the bands
    # are supplied directly rather than fitted here.
    bands = ConfidenceBands(lower=0.33, upper=0.66, n_fitted=99)
    bundle, frame = evaluate_predictions(
        predictions, split_frame(manifest, "test"), bands, MCVAConfig()
    )

    assert bundle.n_instances == len(predictions)
    assert 0.0 <= bundle.attitude.accuracy <= 1.0
    assert bundle.attitude.n_total == bundle.n_instances
    assert bundle.polarity.n_total == bundle.n_instances
    assert bundle.intensity.n_total == bundle.n_instances

    matrix = confusion_frame(bundle)
    assert matrix.loc["Total", "Total"] == bundle.n_instances

    run_dir = write_run(tmp_path / "out", 17, bundle, frame, report)
    for name in (
        "predictions.csv",
        "confusion_with_mcva.csv",
        "confusion_without_mcva.csv",
        "per_class.csv",
        "review_queue.csv",
        "metrics.json",
    ):
        assert (run_dir / name).exists(), name

    payload = json.loads((run_dir / "metrics.json").read_text())
    assert payload["seed"] == 17
    assert "parse_failure_rate" in payload["decode"]


def test_review_queue_hides_the_proposed_direction(
    tmp_path, data_cfg, train_cfg, tokenizer, tiny_model, tiny_model_config
) -> None:
    """The adjudicator sees the instance, not which way the rule wanted to move it."""
    import pandas as pd

    manifest = prepare_manifest(data_cfg)
    loaders = build_dataloaders(manifest, tokenizer, data_cfg, tiny_model_config, train_cfg)
    predictions, report = decode_dataset(
        tiny_model, tokenizer, loaders["test"], torch.device("cpu"), DecodeConfig(num_beams=1)
    )
    bundle, frame = evaluate_predictions(
        predictions,
        split_frame(manifest, "test"),
        ConfidenceBands(lower=0.9, upper=0.95, n_fitted=99),
        MCVAConfig(),
    )
    run_dir = write_run(tmp_path / "out", 17, bundle, frame, report)
    queue = pd.read_csv(run_dir / "review_queue.csv")
    for hidden in ("mcva_proposed", "mcva_branch", "mcva_band"):
        assert hidden not in queue.columns


def test_aggregation_requires_the_primary_seed(
    data_cfg, train_cfg, tokenizer, tiny_model, tiny_model_config
) -> None:
    manifest = prepare_manifest(data_cfg)
    loaders = build_dataloaders(manifest, tokenizer, data_cfg, tiny_model_config, train_cfg)
    predictions, _ = decode_dataset(
        tiny_model, tokenizer, loaders["test"], torch.device("cpu"), DecodeConfig(num_beams=1)
    )
    bundle, _ = evaluate_predictions(
        predictions,
        split_frame(manifest, "test"),
        ConfidenceBands(lower=0.33, upper=0.66, n_fitted=99),
        MCVAConfig(),
    )
    with pytest.raises(ValueError, match="primary seed"):
        aggregate_runs({29: bundle, 43: bundle})

    summary = aggregate_runs({17: bundle, 29: bundle})
    assert "attitude_accuracy" in summary.index
    assert "primary_run" in summary.columns


def test_predictions_frame_carries_the_class_probabilities(
    data_cfg, train_cfg, tokenizer, tiny_model, tiny_model_config
) -> None:
    manifest = prepare_manifest(data_cfg)
    loaders = build_dataloaders(manifest, tokenizer, data_cfg, tiny_model_config, train_cfg)
    predictions, _ = decode_dataset(
        tiny_model, tokenizer, loaders["test"], torch.device("cpu"), DecodeConfig(num_beams=1)
    )
    frame = predictions_to_frame(predictions)
    probability_columns = [c for c in frame.columns if c.startswith("prob_")]
    assert len(probability_columns) == 5
    assert frame[probability_columns].sum(axis=1).round(4).eq(1.0).all()


def test_written_json_is_strictly_parseable(
    tmp_path, data_cfg, train_cfg, tokenizer, tiny_model, tiny_model_config
) -> None:
    """`NaN` and `Infinity` are Python conveniences, not valid JSON.

    A zero-variance correlation and an empty calibration bin both produce
    ``nan``, and the uncalibrated confidence bands used to produce ``inf``.
    Written as bare tokens, they make the metrics files unreadable from jq,
    JavaScript or R.
    """
    manifest = prepare_manifest(data_cfg)
    loaders = build_dataloaders(manifest, tokenizer, data_cfg, tiny_model_config, train_cfg)
    predictions, report = decode_dataset(
        tiny_model, tokenizer, loaders["test"], torch.device("cpu"), DecodeConfig(num_beams=1)
    )
    bundle, frame = evaluate_predictions(
        predictions,
        split_frame(manifest, "test"),
        ConfidenceBands.review_everything(),
        MCVAConfig(),
    )
    run_dir = write_run(tmp_path / "out", 17, bundle, frame, report)

    def strict(constant: str) -> None:
        raise AssertionError(f"non-JSON constant written: {constant}")

    payload = json.loads((run_dir / "metrics.json").read_text(), parse_constant=strict)
    assert payload["seed"] == 17


def test_uncalibrated_bands_use_a_finite_sentinel() -> None:
    import math

    bands = ConfidenceBands.review_everything()
    assert math.isfinite(bands.lower) and math.isfinite(bands.upper)
    assert bands.band_of(1.0) == "low"


def test_run_emits_the_intensity_and_relation_tables(
    tmp_path, data_cfg, train_cfg, tokenizer, tiny_model, tiny_model_config
) -> None:
    """The manuscript's intensity-calibration and output-relation tables."""
    import pandas as pd

    manifest = prepare_manifest(data_cfg)
    loaders = build_dataloaders(manifest, tokenizer, data_cfg, tiny_model_config, train_cfg)
    predictions, report = decode_dataset(
        tiny_model, tokenizer, loaders["test"], torch.device("cpu"), DecodeConfig(num_beams=1)
    )
    bundle, frame = evaluate_predictions(
        predictions,
        split_frame(manifest, "test"),
        ConfidenceBands.review_everything(),
        MCVAConfig(),
    )
    run_dir = write_run(tmp_path / "out", 17, bundle, frame, report)

    calibration = pd.read_csv(run_dir / "intensity_calibration.csv")
    assert len(calibration) == 9
    assert calibration["n"].sum() == bundle.n_instances

    contingency = pd.read_csv(
        run_dir / "output_relation_polarity_by_attitude.csv", index_col=0
    )
    assert contingency.loc["Total", "Total"] == bundle.n_instances

    distribution = pd.read_csv(
        run_dir / "output_relation_intensity_by_attitude.csv", index_col=0
    )
    assert distribution["n"].sum() == bundle.n_instances
