"""Deterministic seeding for the ten-seed protocol of Appendix A.2.1."""

import logging
import os
import random
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every source of randomness used by a training run.

    Args:
        seed: The seed to apply. The manuscript uses ``msa_arc.constants.SEEDS``.
        deterministic: Whether to force deterministic cuDNN kernels. Costs
            throughput, and is what makes a run reproducible from its seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        logger.warning("numpy not installed; skipping numpy seeding")

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dependency
        logger.warning("torch not installed; skipping torch seeding")

    logger.info("Seeded all RNGs with %d (deterministic=%s)", seed, deterministic)


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """Give every DataLoader worker its own deterministic stream."""
    seed = (base_seed if base_seed is not None else 0) + worker_id
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:  # pragma: no cover
        pass


__all__ = ["set_seed", "worker_init_fn"]
