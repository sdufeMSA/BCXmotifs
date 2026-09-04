"""Fixed vocabularies and identifiers shared across the MSA-ARC pipeline.

Every constant here is pinned to a specific location in the manuscript so that
a reader can check the code against the paper without guessing.  Nothing in
this module may be changed by a configuration file: these are the label spaces
and identifiers the annotation protocol was run with.
"""

from typing import Dict, Final, Tuple

# Attitude categories (Table: expression-gap; Section 4.1.4)

ATTITUDE_CLASSES: Final[Tuple[str, ...]] = (
    "Like",
    "Essential",
    "Neutral",
    "Live with",
    "Dislike",
)

ATTITUDE_CODES: Final[Dict[str, str]] = {
    "Like": "L",
    "Essential": "E",
    "Neutral": "N",
    "Live with": "Li",
    "Dislike": "D",
}

CODE_TO_ATTITUDE: Final[Dict[str, str]] = {v: k for k, v in ATTITUDE_CODES.items()}

ATTITUDE_TO_INDEX: Final[Dict[str, int]] = {name: i for i, name in enumerate(ATTITUDE_CLASSES)}

# Sentiment polarity (Section 4.1.4)

POLARITY_CLASSES: Final[Tuple[str, ...]] = ("positive", "negative", "neutral")

POLARITY_TO_INDEX: Final[Dict[str, int]] = {name: i for i, name in enumerate(POLARITY_CLASSES)}

# Short forms used inside Eq. 1 of the manuscript.
POLARITY_SHORT: Final[Dict[str, str]] = {
    "positive": "pos",
    "negative": "neg",
    "neutral": "neu",
}

# Sentiment intensity (Section 4.1.4)

INTENSITY_MIN: Final[float] = -1.0
INTENSITY_MAX: Final[float] = 1.0
INTENSITY_DECIMALS: Final[int] = 2

# Experimental protocol (Appendix A.2.1)

SEEDS: Final[Tuple[int, ...]] = (17, 29, 43, 61, 79, 97, 113, 137, 163, 191)

#: The single run whose raw counts back every confusion matrix and every MCVA
#: intervention count in the paper.  Multi-seed figures are reported as
#: mean +/- SD over ``SEEDS`` and must never be mixed with these counts.
PRIMARY_SEED: Final[int] = 17

SPLIT_NAMES: Final[Tuple[str, ...]] = ("train", "validation", "test")

#: Participant counts of the frozen participant-level partition (Section 5.1.1).
SPLIT_PARTICIPANTS: Final[Dict[str, int]] = {
    "train": 188,
    "validation": 24,
    "test": 23,
}

# Stage-A feature extractor (Appendix A.2.1)

#: Parameters of the ImageNet ResNet-50 checkpoint, classifier head included.
#: The appendix's 632.1M total is the sum of the frozen mT5-base (582,401,280),
#: the 24,139,392 trainable parameters, and this figure. The ResNet-50 runs in
#: Stage A of this implementation and is not held by the Stage-B model, so
#: ``MulMT5.parameter_report`` reports the reconciliation rather than silently
#: falling 25.6M short of the published number.
RESNET50_PARAMETERS: Final[int] = 25_557_032

#: The same network after ``fc`` is replaced by an identity, which is how the
#: video extractor uses it. The 2,049,000-parameter classifier is never run.
RESNET50_FEATURE_EXTRACTOR_PARAMETERS: Final[int] = 23_508_032

# Service catalogue (Table: service_items)

SCENARIOS: Final[Tuple[int, ...]] = (1, 0)  # 1 = functional, 0 = dysfunctional

SCENARIO_NAMES: Final[Dict[int, str]] = {1: "functional", 0: "dysfunctional"}

SERVICE_NAMES: Final[Dict[str, str]] = {
    "s1": "personal hygiene nursing",
    "s2": "living care",
    "s3": "group dining",
    "s4": "meals on wheels",
    "s5": "door to door bath",
    "s6": "outdoor bath",
    "s7": "home cleaning",
    "s8": "goods cleaning",
    "s9": "centralized washing",
    "s10": "home washing",
    "s11": "escorted walk",
    "s12": "escort out",
    "s13": "agency purchasing goods",
    "s14": "agency collecting goods",
    "s15": "agency payment",
    "s16": "group rehabilitation",
    "s17": "individual rehabilitation",
    "s18": "psychological support service",
    "s19": "cultural & entertaining services",
    "s20": "agency dispense medicines",
    "s21": "accompany for medical treatment",
    "s22": "legal aid",
    "s23": "social interaction service",
    "s24": "smart monitoring",
    "s25": "emergency rescue",
    "s26": "safety reminder",
    "s27": "health management",
    "s28": "home appliance maintenance",
}

SERVICE_IDS: Final[Tuple[str, ...]] = tuple(f"s{i}" for i in range(1, 29))

N_SERVICES: Final[int] = len(SERVICE_IDS)

#: 28 services x 2 scenarios.  A participant record that does not contain
#: exactly this many instances is rejected by the manifest validator.
INSTANCES_PER_PARTICIPANT: Final[int] = N_SERVICES * len(SCENARIOS)

# Verbal-nonverbal divergence patterns (Table: expression-gap)

#: ``none`` is the sincere-endorsement baseline (row 1); the remaining four are
#: the divergence patterns evaluated as a subset in Section 6.3.
DIVERGENCE_PATTERNS: Final[Tuple[str, ...]] = (
    "none",
    "aging_denial",
    "politeness",
    "reluctant_acceptance",
    "sarcasm",
)

DIVERGENCE_SUBSET: Final[Tuple[str, ...]] = (
    "aging_denial",
    "politeness",
    "reluctant_acceptance",
    "sarcasm",
)

__all__ = [
    "ATTITUDE_CLASSES",
    "ATTITUDE_CODES",
    "CODE_TO_ATTITUDE",
    "ATTITUDE_TO_INDEX",
    "POLARITY_CLASSES",
    "POLARITY_TO_INDEX",
    "POLARITY_SHORT",
    "INTENSITY_MIN",
    "INTENSITY_MAX",
    "INTENSITY_DECIMALS",
    "SEEDS",
    "PRIMARY_SEED",
    "SPLIT_NAMES",
    "RESNET50_FEATURE_EXTRACTOR_PARAMETERS",
    "RESNET50_PARAMETERS",
    "SPLIT_PARTICIPANTS",
    "SCENARIOS",
    "SCENARIO_NAMES",
    "SERVICE_NAMES",
    "SERVICE_IDS",
    "N_SERVICES",
    "INSTANCES_PER_PARTICIPANT",
    "DIVERGENCE_PATTERNS",
    "DIVERGENCE_SUBSET",
]
