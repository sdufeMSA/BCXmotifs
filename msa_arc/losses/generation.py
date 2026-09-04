"""Token-level generation loss over the unified target sequence.

Section 4.1.3 is explicit that the three outputs are not produced by three task
heads with three objectives.  They are serialised into one target string and the
cross-entropy runs over the tokens of that string, so polarity, intensity and
attitude category are supervised jointly and their relative influence is fixed
by their token lengths rather than by tunable coefficients.
"""

from typing import Optional

import torch
import torch.nn.functional as F

#: Positions carrying this label are excluded from the loss.
IGNORE_INDEX = -100


def generation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Mean cross-entropy over the target tokens.

    Args:
        logits: ``(batch, target_len, vocab)`` decoder logits.
        labels: ``(batch, target_len)`` target ids with ``IGNORE_INDEX`` at
            padding positions.
        label_smoothing: Passed through to ``F.cross_entropy``. The paper uses
            none; the argument exists so an ablation can switch it on without
            editing this function.

    Returns:
        Scalar loss.
    """
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        label_smoothing=label_smoothing,
    )


def sequence_log_probabilities(
    logits: torch.Tensor,
    labels: torch.Tensor,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Log-likelihood of each target sequence in the batch.

    Args:
        logits: ``(batch, target_len, vocab)``.
        labels: ``(batch, target_len)`` with ``IGNORE_INDEX`` at padding.
        length_normalize: Divide by the number of scored tokens. Required when
            comparing candidates of different token lengths, which is exactly
            the case for the five attitude category names.

    Returns:
        ``(batch,)`` log-likelihoods.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    mask = labels != IGNORE_INDEX
    safe_labels = labels.masked_fill(~mask, 0)

    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~mask, 0.0)

    totals = token_log_probs.sum(dim=-1)
    if not length_normalize:
        return totals
    counts = mask.sum(dim=-1).clamp(min=1)
    return totals / counts


def combined_loss(
    generation: torch.Tensor,
    contrastive: Optional[torch.Tensor],
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Weighted sum of the two objective terms.

    ``L = alpha * L_CE + beta * L_Contrastive``. ``contrastive`` may be ``None``
    for a batch in which no anchor had an in-batch positive, in which case the
    generation term stands alone rather than being paired with a fabricated one.

    Args:
        generation: Scalar generation loss.
        contrastive: Scalar contrastive loss, or ``None``.
        alpha: Weight on the generation term.
        beta: Weight on the contrastive term.

    Returns:
        Scalar total loss.
    """
    if contrastive is None:
        return alpha * generation
    return alpha * generation + beta * contrastive


__all__ = [
    "IGNORE_INDEX",
    "combined_loss",
    "generation_loss",
    "sequence_log_probabilities",
]
