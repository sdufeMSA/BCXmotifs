#!/usr/bin/env python
"""Stage B: train and evaluate MSA-ARC over the ten-seed protocol.

One run per seed.  Each run trains on the 188 training participants, selects a
checkpoint on the 24 validation participants, fits the MCVA confidence tertiles
on that same validation split, and only then scores the 23 test participants.

Every artefact a table in the paper needs is written under ``output_dir``:
per-instance predictions, raw-count confusion matrices with and without MCVA,
per-class precision/recall/F1/support, the review queue, and the across-seed
summary that separates the primary frozen run from the ten-seed mean.

Examples::

    python scripts/train.py                       # all ten seeds
    python scripts/train.py seeds=[17]            # primary run only
    python scripts/train.py model=text_only       # ablation
    python scripts/train.py mcva=disabled         # the "without MCVA" row
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msa_arc.config import ExperimentConfig, experiment_from_mapping  # noqa: E402
from msa_arc.data.loaders import build_dataloaders, prepare_manifest, split_frame  # noqa: E402
from msa_arc.evaluation.bundle import (  # noqa: E402
    EvaluationBundle,
    evaluate_predictions,
    fit_bands_on_validation,
    predictions_to_frame,
)
from msa_arc.evaluation.report import dump_json, write_run, write_summary  # noqa: E402
from msa_arc.inference.decode import decode_dataset  # noqa: E402
from msa_arc.model import build_model  # noqa: E402
from msa_arc.train import train_model  # noqa: E402
from msa_arc.utils.logging import configure_logging  # noqa: E402
from msa_arc.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger("train")


def resolve_device(requested: str) -> torch.device:
    """Fall back to CPU when CUDA was asked for but is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def run_single_seed(
    cfg: ExperimentConfig,
    seed: int,
    manifest,
    tokenizer,
) -> tuple[EvaluationBundle, object]:
    """Train one model and evaluate it on the test split.

    Args:
        cfg: The composed experiment configuration.
        seed: Random seed for this run.
        manifest: Annotated manifest carrying a ``split`` column.
        tokenizer: Tokenizer used for the unified target strings.

    Returns:
        The evaluation bundle and the per-instance prediction frame.
    """
    set_seed(seed)
    train_cfg = type(cfg.train)(**{**cfg.train.__dict__, "seed": seed})
    device = resolve_device(train_cfg.device)

    loaders = build_dataloaders(manifest, tokenizer, cfg.data, cfg.model, train_cfg)
    for split in ("train", "validation", "test"):
        if split not in loaders:
            raise RuntimeError(f"split {split!r} is empty; cannot run the protocol")

    model = build_model(cfg.model)
    model, history = train_model(
        model=model,
        tokenizer=tokenizer,
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        train_cfg=train_cfg,
        loss_cfg=cfg.loss,
        decode_cfg=cfg.decode,
    )
    logger.info(
        "Seed %d: best epoch %d with validation macro-F1 %.4f",
        seed,
        history.best_epoch + 1,
        history.best_metric,
    )

    # Confidence tertiles are fitted on validation and frozen before test.
    validation_predictions, _ = decode_dataset(
        model, tokenizer, loaders["validation"], device, cfg.decode
    )
    bands = fit_bands_on_validation(predictions_to_frame(validation_predictions), cfg.mcva)

    test_predictions, decode_report = decode_dataset(
        model, tokenizer, loaders["test"], device, cfg.decode
    )
    bundle, frame = evaluate_predictions(
        test_predictions, split_frame(manifest, "test"), bands, cfg.mcva
    )

    run_dir = write_run(cfg.output_dir, seed, bundle, frame, decode_report)
    dump_json(
        {
            "epochs": history.epochs,
            "best_epoch": history.best_epoch,
            "best_metric": history.best_metric,
            "stopped_early": history.stopped_early,
            "confidence_bands": {
                "lower": bands.lower,
                "upper": bands.upper,
                "n_fitted": bands.n_fitted,
                "fitted": bands.fitted,
            },
        },
        run_dir / "history.json",
    )
    torch.save(model.trainable_state_dict(), run_dir / "checkpoint.pt")
    return bundle, frame


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    configure_logging(log_file=Path(raw_cfg.output_dir) / "train.log")
    cfg = experiment_from_mapping(OmegaConf.to_container(raw_cfg, resolve=True))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(raw_cfg), encoding="utf-8"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    manifest = prepare_manifest(cfg.data)

    seeds = cfg.seeds or [cfg.train.seed]
    bundles: Dict[int, EvaluationBundle] = {}
    for seed in seeds:
        logger.info("=== seed %d (%d of %d) ===", seed, seeds.index(seed) + 1, len(seeds))
        bundle, _ = run_single_seed(cfg, seed, manifest, tokenizer)
        bundles[seed] = bundle

    if len(bundles) > 1:
        write_summary(cfg.output_dir, bundles)
    logger.info("Done. Artefacts under %s", output_dir.resolve())


if __name__ == "__main__":
    main()
