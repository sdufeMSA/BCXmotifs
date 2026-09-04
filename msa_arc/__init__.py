"""MSA-ARC: multimodal attitude recognition for smart elderly care.

Reference implementation for the MSA-ARC component of *From Voices to Service
Bundles: A Multimodal Sentiment and Biclustering Framework for Smart Elderly
Care*.

The pipeline runs in two stages, split where the study's data statement splits:

* **Stage A** (:mod:`msa_arc.features`) turns raw interview media into
  de-identified per-instance tensors. It runs where the recordings live and its
  outputs, not the recordings, are what can be released.
* **Stage B** (:mod:`msa_arc.data`, :mod:`msa_arc.model`, :mod:`msa_arc.train`,
  :mod:`msa_arc.inference`, :mod:`msa_arc.mcva`, :mod:`msa_arc.evaluation`)
  trains and evaluates on those tensors, and reproduces every reported number
  without access to the raw media.

Typical use::

    from msa_arc import ExperimentConfig, build_model, train_model
"""

__version__ = "0.1.0"

from msa_arc.config import (
    DataConfig,
    DecodeConfig,
    ExperimentConfig,
    FeatureConfig,
    LossConfig,
    MCVAConfig,
    ModelConfig,
    TrainConfig,
)
from msa_arc.constants import (
    ATTITUDE_CLASSES,
    POLARITY_CLASSES,
    PRIMARY_SEED,
    SEEDS,
    SERVICE_IDS,
)

__all__ = [
    "ATTITUDE_CLASSES",
    "DataConfig",
    "DecodeConfig",
    "ExperimentConfig",
    "FeatureConfig",
    "LossConfig",
    "MCVAConfig",
    "ModelConfig",
    "POLARITY_CLASSES",
    "PRIMARY_SEED",
    "SEEDS",
    "SERVICE_IDS",
    "TrainConfig",
    "__version__",
]
