"""Training objectives."""

from msa_arc.losses.contrastive import contrastive_loss, pairwise_squared_distances
from msa_arc.losses.generation import (
    IGNORE_INDEX,
    combined_loss,
    generation_loss,
    sequence_log_probabilities,
)

__all__ = [
    "IGNORE_INDEX",
    "combined_loss",
    "contrastive_loss",
    "generation_loss",
    "pairwise_squared_distances",
    "sequence_log_probabilities",
]
