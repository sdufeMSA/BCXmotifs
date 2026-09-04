#!/usr/bin/env python
"""Stage A: raw interview media -> de-identified per-instance feature tensors.

Run this where the recordings live.  Its outputs are the ``.npy`` tensors that
Stage B trains on, and they are what the study can release: the recordings carry
identifiable faces and voices, the pooled features do not reconstruct them.

Example::

    python scripts/prepare_features.py \\
        --manifest data/manifest.csv \\
        --feature-dir data/features \\
        --modalities text,audio,video

The script is resumable.  An instance whose tensor already exists is skipped
unless ``--overwrite`` is given, so a run interrupted after 600 of 896
participants picks up where it stopped.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msa_arc.config import FeatureConfig  # noqa: E402
from msa_arc.features.audio import AudioFeatureExtractor  # noqa: E402
from msa_arc.features.manifest import load_manifest, media_segment  # noqa: E402
from msa_arc.features.text import TextFeatureExtractor  # noqa: E402
from msa_arc.features.video import VideoFeatureExtractor  # noqa: E402
from msa_arc.utils.logging import configure_logging  # noqa: E402

logger = logging.getLogger("prepare_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument(
        "--feature-dir", required=True, help="Output root for the feature tensors"
    )
    parser.add_argument(
        "--modalities",
        default="text,audio,video",
        help="Comma-separated subset of text,audio,video",
    )
    parser.add_argument(
        "--tokenizer",
        default="google/mt5-base",
        help="Tokenizer checkpoint for the text branch",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device for the ResNet-50 video encoder"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute tensors that already exist"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N rows"
    )
    parser.add_argument("--log-file", default=None, help="Optional log file")
    return parser.parse_args()


def build_extractors(
    modalities: List[str], cfg: FeatureConfig, tokenizer_name: str, device: str
) -> Dict[str, object]:
    """Instantiate only the extractors that were asked for.

    The audio and video extractors pull in heavy optional dependencies, so a
    text-only run must not construct them.
    """
    extractors: Dict[str, object] = {}
    if "text" in modalities:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        extractors["text"] = TextFeatureExtractor(cfg.text, tokenizer)
    if "audio" in modalities:
        extractors["audio"] = AudioFeatureExtractor(cfg.audio)
    if "video" in modalities:
        extractors["video"] = VideoFeatureExtractor(cfg.video, device=device)
    return extractors


def main() -> int:
    args = parse_args()
    configure_logging(log_file=args.log_file)

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    unknown = set(modalities) - {"text", "audio", "video"}
    if unknown:
        logger.error("unknown modalities: %s", sorted(unknown))
        return 2

    manifest = load_manifest(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)

    extractors = build_extractors(modalities, FeatureConfig(), args.tokenizer, args.device)
    feature_dir = Path(args.feature_dir)

    written = dict.fromkeys(modalities, 0)
    skipped = dict.fromkeys(modalities, 0)
    failed: List[str] = []

    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        key = row.instance_key
        row_dict = row._asdict()

        for modality in modalities:
            extractor = extractors[modality]
            source = (
                row_dict.get("transcript")
                if modality == "text"
                else media_segment(row_dict, modality)
            )
            try:
                path = extractor.extract_to_file(
                    source, feature_dir, key, overwrite=args.overwrite
                )
            except Exception:
                logger.exception("failed to extract %s features for %s", modality, key)
                failed.append(f"{key}:{modality}")
                continue
            if path is None:
                skipped[modality] += 1
            else:
                written[modality] += 1

        if position % 200 == 0:
            logger.info("Processed %d/%d instances", position, len(manifest))

    logger.info("Written: %s", written)
    logger.info("Skipped (no usable input): %s", skipped)
    if failed:
        logger.error("%d extractions failed; first few: %s", len(failed), failed[:10])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
