"""Class-aware contrastive loss (Section 4.1.3, Appendix A.2.1).

The term pulls together fused representations that share an attitude category
and pushes apart those that do not, which matters most for adjacent categories
such as Neutral and Live with that get confused when an elderly participant
answers politely or concessively.

Two implementation details:

1. The distance is the squared Euclidean distance between L2-normalised fused
   representations.  The representation itself is the masked mean of the
   adapter-augmented encoder states. That pooling is used only by this loss,
   never by prediction.
2. Positive and negative selection is batch-hard: the farthest same-class and
   the nearest different-class instance in the batch.  Batch-hard mining is
   deterministic given the batch, so multi-seed variation reflects
   initialisation and batch order rather than triplet sampling noise.

An anchor with no in-batch positive contributes nothing, and the loss is
averaged over the anchors that do have one, so no pair is ever fabricated.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def pairwise_squared_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between L2-normalised rows.

    For unit vectors ``||a - b||^2 = 2 - 2 a.b``, which avoids the numerical
    noise of squaring a subtraction.

    Args:
        embeddings: ``(batch, dim)``, already L2-normalised.

    Returns:
        ``(batch, batch)`` distance matrix with a zero diagonal.
    """
    similarity = embeddings @ embeddings.t()
    distances = (2.0 - 2.0 * similarity).clamp(min=0.0)
    return distances.fill_diagonal_(0.0)


def contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    negative_mining: str = "batch_hard",
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Triplet loss over attitude categories.

    Args:
        embeddings: ``(batch, dim)`` fused representations.
        labels: ``(batch,)`` integer attitude category indices.
        margin: Triplet margin ``m``; the paper uses 1.0.
        negative_mining: ``batch_hard`` or ``random``.
        generator: RNG for ``random`` mining, so a run stays reproducible.

    Returns:
        Scalar loss, or ``None`` when no anchor in the batch has both a positive
        and a negative. Callers must treat ``None`` as "no contrastive term for
        this step" rather than as zero loss.

    Raises:
        ValueError: If ``negative_mining`` is unknown.
    """
    if negative_mining not in {"batch_hard", "random"}:
        raise ValueError(f"unknown negative_mining: {negative_mining!r}")

    normalized = F.normalize(embeddings, p=2, dim=-1)
    distances = pairwise_squared_distances(normalized)

    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    identity = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positive_mask = same_class & ~identity
    negative_mask = ~same_class

    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not bool(valid.any()):
        return None

    if negative_mining == "batch_hard":
        # Farthest positive: hardest to pull together.
        positive_distances = distances.masked_fill(~positive_mask, float("-inf"))
        hardest_positive = positive_distances.max(dim=1).values
        # Nearest negative: hardest to push apart.
        negative_distances = distances.masked_fill(~negative_mask, float("inf"))
        hardest_negative = negative_distances.min(dim=1).values
    else:
        hardest_positive = _sample_masked(distances, positive_mask, generator)
        hardest_negative = _sample_masked(distances, negative_mask, generator)

    triplet = F.relu(hardest_positive - hardest_negative + margin)
    return triplet[valid].mean()


def _sample_masked(
    distances: torch.Tensor,
    mask: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Pick one distance uniformly at random from each row's masked entries."""
    weights = mask.to(distances.dtype)
    # Rows with no candidate get a uniform fallback; the caller filters them out
    # through ``valid`` before the mean is taken.
    weights = torch.where(
        weights.sum(dim=1, keepdim=True) > 0, weights, torch.ones_like(weights)
    )
    indices = torch.multinomial(weights, num_samples=1, generator=generator)
    return distances.gather(1, indices).squeeze(1)


__all__ = ["contrastive_loss", "pairwise_squared_distances"]
