#!/usr/bin/env python
"""Generate a synthetic corpus so the pipeline can be exercised without data.

The real interviews cannot be released, and a collaborator preparing the corpus
needs something to check their manifest against.  This script writes a complete
Stage-A output: manifest, split file and feature tensors in the exact layout
``scripts/train.py`` expects, filled with random data whose labels are weakly
correlated with the features, so a smoke run produces a loss that moves.

Nothing here is a substitute for the corpus.  It exists so that the pipeline
can be verified end to end on any machine, with or without the real tensors.

Example::

    python scripts/make_synthetic_fixture.py --out data/synthetic --participants 6
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msa_arc.constants import (  # noqa: E402
    ATTITUDE_CLASSES,
    DIVERGENCE_PATTERNS,
    SCENARIOS,
    SERVICE_IDS,
)
from msa_arc.features.manifest import instance_key  # noqa: E402
from msa_arc.utils.logging import configure_logging  # noqa: E402

logger = logging.getLogger("make_synthetic_fixture")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output root")
    parser.add_argument(
        "--participants", type=int, default=6, help="Number of synthetic participants"
    )
    parser.add_argument(
        "--services",
        type=int,
        default=len(SERVICE_IDS),
        help="Services per participant; lower it for a fast smoke test",
    )
    parser.add_argument("--text-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--audio-frames", type=int, default=40)
    parser.add_argument("--video-frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20240620)
    return parser.parse_args()


def intensity_for(category: str, rng: np.random.Generator) -> tuple[str, float]:
    """Draw a polarity and intensity consistent with an attitude category.

    Real annotations are not independent across the three outputs, and a fixture
    whose labels were independent would let a model score well by ignoring two
    of them. The draws here keep the same broad coupling the label protocol
    describes, including occasional divergence.
    """
    if category in {"Like", "Essential"}:
        polarity = "positive" if rng.random() > 0.15 else "negative"
        magnitude = rng.uniform(0.4, 1.0)
    elif category == "Neutral":
        polarity = "neutral" if rng.random() > 0.2 else "negative"
        magnitude = rng.uniform(0.0, 0.35)
    elif category == "Live with":
        polarity = "negative" if rng.random() > 0.3 else "neutral"
        magnitude = rng.uniform(0.3, 0.7)
    else:  # Dislike
        polarity = "negative" if rng.random() > 0.15 else "positive"
        magnitude = rng.uniform(0.6, 1.0)

    sign = -1.0 if polarity == "negative" else (1.0 if polarity == "positive" else 0.0)
    return polarity, round(float(sign * magnitude), 2)


def main() -> int:
    args = parse_args()
    configure_logging()
    rng = np.random.default_rng(args.seed)

    root = Path(args.out)
    feature_dir = root / "features"
    for modality in ("text", "audio", "video"):
        (feature_dir / modality).mkdir(parents=True, exist_ok=True)

    services = list(SERVICE_IDS[: args.services])
    rows: List[dict] = []

    for index in range(args.participants):
        participant = f"P{index:04d}"
        for service in services:
            for scenario in SCENARIOS:
                key = instance_key(participant, service, scenario)
                category = str(rng.choice(ATTITUDE_CLASSES))
                polarity, intensity = intensity_for(category, rng)

                # A class-dependent offset gives the features something to learn.
                offset = ATTITUDE_CLASSES.index(category)
                ids = rng.integers(2, args.vocab_size, size=args.text_length)
                ids[0] = 2 + offset
                mask = np.ones(args.text_length, dtype=np.int64)
                np.save(feature_dir / "text" / f"{key}.npy", np.stack([ids, mask]))

                audio = rng.normal(offset * 0.1, 1.0, size=(args.audio_frames, 40))
                np.save(feature_dir / "audio" / f"{key}.npy", audio.astype(np.float32))

                video = rng.normal(offset * 0.1, 1.0, size=(args.video_frames, 2048))
                np.save(feature_dir / "video" / f"{key}.npy", video.astype(np.float32))

                rows.append(
                    {
                        "participant_id": participant,
                        "service_id": service,
                        "scenario": scenario,
                        "transcript": f"synthetic transcript for {key}",
                        "audio_path": f"synthetic/{participant}.wav",
                        "audio_start_sec": "",
                        "audio_end_sec": "",
                        "video_path": f"synthetic/{participant}.mp4",
                        "video_start_sec": "",
                        "video_end_sec": "",
                        "label_polarity": polarity,
                        "label_intensity": intensity,
                        "label_category": category,
                        "divergence_pattern": str(rng.choice(DIVERGENCE_PATTERNS)),
                    }
                )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / "manifest.csv", index=False)

    # At least one participant per split, so every loader is non-empty.
    participants = sorted(manifest["participant_id"].unique())
    if len(participants) < 3:
        raise SystemExit("need at least 3 participants to fill three splits")
    n_validation = max(1, len(participants) // 6)
    n_test = max(1, len(participants) // 6)
    assignments = (
        ["validation"] * n_validation
        + ["test"] * n_test
        + ["train"] * (len(participants) - n_validation - n_test)
    )
    pd.DataFrame({"participant_id": participants, "split": assignments}).to_csv(
        root / "splits.csv", index=False
    )

    logger.info(
        "Wrote %d instances for %d participants to %s",
        len(manifest),
        len(participants),
        root.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
