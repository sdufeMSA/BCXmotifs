#!/usr/bin/env python
"""Evaluate a saved MSA-ARC checkpoint without retraining.

Use this to regenerate the tables from a released checkpoint, to score a split
other than test, or to re-run the reconciliation under different thresholds
while holding the model fixed:

    python scripts/evaluate.py --checkpoint outputs/msa_arc/seed_17/checkpoint.pt
    python scripts/evaluate.py --checkpoint ... --tau-rev 0.8 --tau-flag 0.4

The confidence tertiles are always refitted on the validation split under
whatever thresholds are in force, since a different threshold produces a
different set of reclassifications and the old cut-points would not describe it.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msa_arc.config import MCVAConfig, experiment_from_mapping  # noqa: E402
from msa_arc.data.loaders import build_dataloaders, prepare_manifest, split_frame  # noqa: E402
from msa_arc.evaluation.bundle import (  # noqa: E402
    evaluate_predictions,
    fit_bands_on_validation,
    predictions_to_frame,
)
from msa_arc.evaluation.report import dump_json, write_run  # noqa: E402
from msa_arc.inference.decode import decode_dataset  # noqa: E402
from msa_arc.model import build_model  # noqa: E402
from msa_arc.utils.logging import configure_logging  # noqa: E402
from msa_arc.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trainable-state checkpoint")
    parser.add_argument(
        "--config",
        default=None,
        help="resolved_config.yaml from the training run; defaults to the one "
        "beside the checkpoint",
    )
    parser.add_argument("--split", default="test", help="Split to score")
    parser.add_argument("--output-dir", default=None, help="Where to write artefacts")
    parser.add_argument("--seed", type=int, default=None, help="Seed label for outputs")
    parser.add_argument("--tau-rev", type=float, default=None, help="Override tau_rev")
    parser.add_argument("--tau-flag", type=float, default=None, help="Override tau_flag")
    parser.add_argument(
        "--no-mcva", action="store_true", help="Score the surface categories only"
    )
    return parser.parse_args()


def locate_config(checkpoint: Path, explicit: str | None) -> Path:
    """Find the resolved config that produced a checkpoint."""
    if explicit:
        return Path(explicit)
    for candidate in (
        checkpoint.parent / "resolved_config.yaml",
        checkpoint.parent.parent / "resolved_config.yaml",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no resolved_config.yaml found near {checkpoint}; pass --config explicitly"
    )


def main() -> int:
    args = parse_args()
    configure_logging()

    checkpoint_path = Path(args.checkpoint)
    cfg = experiment_from_mapping(
        OmegaConf.to_container(
            OmegaConf.load(locate_config(checkpoint_path, args.config)), resolve=True
        )
    )

    mcva_values = dict(cfg.mcva.__dict__)
    if args.tau_rev is not None:
        mcva_values["tau_rev"] = args.tau_rev
    if args.tau_flag is not None:
        mcva_values["tau_flag"] = args.tau_flag
    if args.no_mcva:
        mcva_values["enabled"] = False
    mcva_cfg = MCVAConfig(**mcva_values)

    seed = args.seed if args.seed is not None else cfg.train.seed
    set_seed(seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_name)
    manifest = prepare_manifest(cfg.data)
    loaders = build_dataloaders(
        manifest, tokenizer, cfg.data, cfg.model, cfg.train, shuffle_train=False
    )
    if args.split not in loaders:
        logger.error("split %r has no instances", args.split)
        return 2

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model)
    model.load_trainable_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(device)

    validation_predictions, _ = decode_dataset(
        model, tokenizer, loaders["validation"], device, cfg.decode
    )
    bands = fit_bands_on_validation(predictions_to_frame(validation_predictions), mcva_cfg)

    predictions, decode_report = decode_dataset(
        model, tokenizer, loaders[args.split], device, cfg.decode
    )
    bundle, frame = evaluate_predictions(
        predictions, split_frame(manifest, args.split), bands, mcva_cfg
    )

    output_dir = Path(args.output_dir or (checkpoint_path.parent / f"eval_{args.split}"))
    run_dir = write_run(output_dir, seed, bundle, frame, decode_report)
    dump_json(mcva_values, run_dir / "mcva_config.json")
    logger.info(
        "%s: attitude accuracy %d/%d = %.4f",
        args.split,
        bundle.attitude.n_correct,
        bundle.attitude.n_total,
        bundle.attitude.accuracy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
