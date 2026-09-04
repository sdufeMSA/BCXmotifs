# MSA-ARC

Reference implementation of the **MSA-ARC** component of *From Voices to Service
Bundles: A Multimodal Sentiment and Biclustering Framework for Smart Elderly
Care*.

MSA-ARC recognises older adults' attitudes toward home-care services from the
text, audio and video of face-to-face interviews. An mT5 backbone is fused with
prosodic and facial cues through six cross-modal adapters, and generates one
target string per instance:

```
positive | 0.85 | Like
```

Polarity, intensity and attitude category are carried together. A downstream
reconciliation rule (**MCVA**) lets the polarity-intensity signal override a
surface category that the verbal channel overstates. That is what recovers the
sarcasm, aging-denial and reluctant-acceptance cases a text-only model reads at
face value.

---

## Contents

| | |
|---|---|
| [Two stages](#two-stages) | Why extraction and training are separate |
| [Verify in 30 seconds](#verify-in-30-seconds) | No data, no downloads |
| [Install](#install) | Pinned versions |
| [Reproduce](#reproduce) | One command |
| [Manifest](#manifest) | The input the pipeline reads |
| [Output → manuscript](#output--manuscript) | Which file backs which table |
| [Runtime and hardware](#runtime-and-hardware) | Measured and estimated |
| [Parameter accounting](#parameter-accounting) | Checked against the paper |
| [Data availability](#data-availability) | What can and cannot be released |

---

## Two stages

The pipeline splits where the study's data statement splits.

**Stage A, feature extraction.** Runs where the recordings live. Turns raw
interview media into de-identified per-instance tensors. Its outputs, not the
recordings, are what the study can release: the interviews carry identifiable
faces and voices; pooled ResNet-50 features and Mel-spectrograms do not
reconstruct them.

**Stage B, training and evaluation.** Runs on those tensors, and recomputes
every reported number without access to the raw media.

```
manifest.csv ──► [Stage A]  prepare_features.py
                     │
                     ▼
        features/{text,audio,video}/*.npy      ← releasable
                     │
                     ▼
             [Stage B]  train.py → evaluate.py
                     │
                     ▼
        predictions · confusion matrices · metrics
```

---

## Verify in 30 seconds

Confirms the code runs end to end before any data exists:

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
pytest
```

**244 tests, ~9 s, fully offline.** They run against a randomly-initialised
small mT5 and a character-level stub tokenizer, so nothing downloads the 2.3 GB
mT5-base checkpoint. `pytest -m slow` adds one check against the published
mT5-base config.

---

## Install

```bash
uv venv --python 3.12
uv pip install -e .              # Stage B: train and evaluate on feature tensors
uv pip install -e ".[features]"  # + Stage A: raw media extractors
uv pip install -e ".[dev]"       # + pytest, ruff, mypy
```

Versions this release was verified with:

| | Verified | Declared minimum |
|---|---|---|
| Python | 3.12.12 | ≥ 3.10, < 3.13 |
| PyTorch | 2.14.0 | ≥ 2.0 |
| transformers | 4.57.6 | ≥ 4.35, < 5 |
| numpy | 2.5.2 | ≥ 1.24 |
| pandas | 3.0.5 | ≥ 2.0 |
| scikit-learn | 1.9.0 | ≥ 1.3 |
| scipy | 1.18.1 | ≥ 1.10 |
| hydra-core | 1.3.6 | ≥ 1.3 |
| omegaconf | 2.3.1 | ≥ 2.3 |

Stage A additionally needs `jieba`, `librosa`, `soundfile`, `noisereduce`,
`opencv-python`, `torchvision` and a face-detection backend. Stage B needs none
of them, which is why reproducing the results does not require them.

---

## Reproduce

Prepare `data/manifest.csv` as specified under [Manifest](#manifest), then:

```bash
# Stage A: at the collection site, on the raw recordings
python scripts/prepare_features.py \
    --manifest data/manifest.csv \
    --feature-dir data/features \
    --modalities text,audio,video \
    --device cuda

# Stage B: the whole ten-seed protocol, one command
python scripts/train.py
```

That single command trains on the 188 training participants, selects a
checkpoint on the 24 validation participants, fits the MCVA confidence tertiles
on that same validation split, scores the 23 test participants, and writes every
artefact below, once per seed, followed by the across-seed summary.

Variants:

```bash
python scripts/train.py seeds=[17]         # primary run only
python scripts/train.py model=text_only    # ablation: no audio, no video
python scripts/train.py model=no_audio
python scripts/train.py model=no_video
python scripts/train.py mcva=disabled      # the "without MCVA" row

# Re-score a saved checkpoint under different thresholds, without retraining
python scripts/evaluate.py \
    --checkpoint outputs/msa_arc/seed_17/checkpoint.pt \
    --tau-rev 0.8 --tau-flag 0.4
```

Any configuration field can be overridden on the command line
(`train.learning_rate=2e-5`). A misspelled key raises rather than being silently
ignored.

---

## Manifest

`data/manifest.csv` is the only interface between the interview corpus and this
code. One row per (participant, service, scenario) instance: 56 rows per
participant, being 28 services times the functional and dysfunctional scenarios.
A worked example covering every case is in `data/manifest_example.csv`.

**Required**

| Column | Values |
|---|---|
| `participant_id` | Any stable de-identified id, consistent across a participant's 56 rows |
| `service_id` | `s1` … `s28` |
| `scenario` | `1` functional, `0` dysfunctional |

**Media, at least one per row**

| Column | Notes |
|---|---|
| `transcript` | Verbatim Mandarin transcript; cleaning happens in Stage A |
| `transcript_path` | Alternative to `transcript` when the text lives in a file |
| `audio_path`, `video_path` | A per-instance clip, or a whole-session recording |
| `audio_start_sec`, `audio_end_sec` | Offsets into a session recording; blank for a pre-segmented clip |
| `video_start_sec`, `video_end_sec` | Likewise for video |

Both layouts are accepted and may be mixed across modalities within one row.
Offsets are in seconds from the start of the file, and `end` must exceed
`start`; give both or neither.

**Labels, all three or none**

| Column | Values |
|---|---|
| `label_polarity` | `positive`, `negative`, `neutral` |
| `label_intensity` | Continuous on `[-1, 1]` |
| `label_category` | `Like`, `Essential`, `Neutral`, `Live with`, `Dislike` |

A row carrying one or two of the three is rejected, since a partially labelled
instance would drop out of one per-output metric but not the others. Rows
without labels are expected: participants who were never annotated belong in the
manifest, and inference runs over them.

**Optional**

| Column | Values |
|---|---|
| `divergence_pattern` | `none`, `aging_denial`, `politeness`, `reluctant_acceptance`, `sarcasm` |
| `split` | `train`, `validation`, `test`; normally left blank, `data/splits.csv` is authoritative |

**Validate before running**

```bash
python -c "from msa_arc.features.manifest import load_manifest; load_manifest('data/manifest.csv')"
```

Rejected: an unknown `service_id`, a `scenario` outside `{0, 1}`, a duplicated
instance, a partially labelled row, an intensity outside `[-1, 1]`, a one-sided
or backwards media offset. Warned about but accepted: a participant with fewer
than 56 instances, and instances with no audio or video features, which
contribute a zero vector to that branch.

### splits.csv

Two columns, one row per annotated participant. Written on first run if absent,
and read thereafter so the partition is reproduced exactly.

```csv
participant_id,split
P0001,train
P0189,validation
P0213,test
```

The partition is drawn over participants, never over instances: a participant's
56 rows always land in the same split. The pipeline refuses to run on a split
file that places one participant in two splits.

---

## Output → manuscript

```
outputs/msa_arc/
├── resolved_config.yaml                          every hyperparameter used
├── summary_across_seeds.csv                      mean ± SD, primary run alongside
└── seed_17/
    ├── predictions.csv
    ├── metrics.json
    ├── confusion_with_mcva.csv
    ├── confusion_without_mcva.csv
    ├── confusion_polarity.csv
    ├── per_class.csv
    ├── intensity_calibration.csv
    ├── output_relation_polarity_by_attitude.csv
    ├── output_relation_intensity_by_attitude.csv
    ├── review_queue.csv
    ├── history.json
    └── checkpoint.pt
```

| Manuscript | File | Note |
|---|---|---|
| **Table 11** per-output performance | `metrics.json` → `polarity`, `intensity`, `attitude`; `per_class.csv` | |
| **Table 12** intensity calibration | `intensity_calibration.csv` | 9 equal-width bins over [−1, 1]; ECE in the frame's `ece` attribute |
| **Table 13** relationship among the three outputs | `output_relation_polarity_by_attitude.csv` (Panel A), `output_relation_intensity_by_attitude.csv` (Panel B) | computed on human labels, not predictions |
| **Table 14** ablation | run `model=text_only`, `no_audio`, `no_video`, `mcva=disabled`; compare `attitude_accuracy` | |
| **Table 15** MCVA validation | `metrics.json` → `mcva` (per-branch counts, per-band accuracy, wrong→right and right→wrong); `review_queue.csv` | adjudicated cases come from `review_queue.csv` |
| **Table 16** divergence subset | `metrics.json` → `divergence`, `divergence_macro_accuracy` | requires `divergence_pattern` in the manifest |
| **Figure 3** attitude confusion matrix | `confusion_with_mcva.csv` | **raw counts**, both margins |
| **Table 25** polarity confusion matrix | `confusion_polarity.csv` | raw counts |
| **Table 26** attitude confusion, without MCVA | `confusion_without_mcva.csv` | raw counts |
| ten-seed mean ± SD | `summary_across_seeds.csv` | |

**Seed 17 is the primary frozen run.** Every raw count in the manuscript comes
from it; scalar metrics are reported as mean ± SD over the ten seeds
`{17, 29, 43, 61, 79, 97, 113, 137, 163, 191}`. The two are kept apart by
construction: `aggregate_runs` refuses to average a set of runs that omits the
primary seed, so a mean can never be printed beside counts from a different set.

`review_queue.csv` deliberately **omits** `mcva_proposed`, `mcva_branch` and
`mcva_band`. The adjudicator sees the instance but not the direction the rule
wanted to move the label, so the adjudication stays independent of the model's
opinion.

---

## Runtime and hardware

**Measured** on Apple M-series (arm64), CPU only:

| Step | Time |
|---|---|
| Test suite (244 tests, offline) | ~9 s |
| Synthetic fixture + one-seed smoke run | ~5 s |

**Full corpus.** Reported runs use an Intel Core i7-10700K, 32 GB RAM and one
NVIDIA RTX 3090 (24 GB). Training is 658 optimisation steps per epoch (10,528
instances at batch size 16), at most 50 epochs with early stopping at patience
5, over 10 seeds. Only 24.1M parameters are updated; the 582M backbone is frozen
and runs in inference mode. Each run records its own wall-clock time in
`outputs/msa_arc/train.log`.

---

## Parameter accounting

Asserted in `tests/test_param_counts.py`, including against the published
mT5-base configuration:

| Component | Parameters | Manuscript |
|---|---|---|
| 6 cross-modal adapters (2304 → 192 → 768) | 3,544,704 | 3.5M |
| Audio LSTM (40 → 768, 2 layers) | 7,213,056 | 7.2M |
| Video LSTM (2048 → 768, 2 layers) | 13,381,632 | 13.4M |
| **Trainable** | **24,139,392** | **24.1M** |
| mT5 language-modelling head (250,112 × 768) | 192,086,016 | 192.1M |
| mT5-base, frozen | 582,401,280 | — |
| ResNet-50 (Stage A), frozen | 25,557,032 | — |
| **Total** | **632,097,704** | **632.1M** |
| Trainable share | 3.82% | 3.8% |

The ResNet-50 runs in Stage A rather than inside the model, so
`MulMT5.parameter_report()` reports the Stage-B total (606.5M) and the total
including Stage A (632.1M) as separate entries.

---

## What this release covers

The complete MSA-ARC component: feature extraction for all three modalities, the
Mul-mT5 model with adapter-based cross-modal fusion, the combined generation and
contrastive objective, decoding, MCVA reconciliation with confidence banding and
a human-review queue, per-output evaluation, calibration, and the ten-seed
reporting protocol.

---

## Data availability

The raw interview recordings are **not released**. They contain identifiable
face video and voice of older adults, and the consent obtained covers research
use by the study team only; public release would breach both that consent and
the approval under which the data were collected.

Everything releasable is here: the code, the configuration files, the
participant-level split identifiers, the per-class label counts, and the
de-identified feature tensors, so the reported results can be recomputed end to
end without the raw media.

---

## Licence

MIT. See [`LICENSE`](LICENSE).
