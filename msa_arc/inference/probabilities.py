"""Normalised five-class probabilities from a generative model.

The uncertainty analysis perturbs predictions using the maximum softmax
probability and the second-highest-probability category, and a text-to-text
model has no five-way softmax to read directly.

This module derives one.  Holding the decoded polarity and intensity
fixed as a prefix, the five candidate target strings differ only in their
category, so the model's log-likelihood of each candidate is a score for that
category.  Scoring is restricted to the tokens where the candidates actually
diverge, found as the longest common token prefix of the five so that nothing is
assumed about how SentencePiece segments the boundary.  Scores are then
normalised by the number of scored tokens, so categories with longer names are
not penalised.  A softmax over the five length-normalised log-likelihoods gives

    P(c | instance) = softmax_c( log p(candidate_c) / n_c )

which is a genuine distribution over the five categories and is what Section 6.4
perturbs.
"""

import logging
from typing import Any, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from msa_arc.constants import ATTITUDE_CLASSES
from msa_arc.losses.generation import IGNORE_INDEX
from msa_arc.model.target import all_category_candidates

logger = logging.getLogger(__name__)

N_CLASSES = len(ATTITUDE_CLASSES)


def longest_common_prefix_length(sequences: torch.Tensor, mask: torch.Tensor) -> int:
    """Number of leading positions on which every candidate agrees.

    Args:
        sequences: ``(n_candidates, seq_len)`` token ids.
        mask: ``(n_candidates, seq_len)`` marking real tokens.

    Returns:
        The length of the shared prefix. Positions from here on are where the
        candidates differ and are therefore the only ones worth scoring.
    """
    length = 0
    max_length = int(mask.sum(dim=1).min().item())
    for position in range(max_length):
        column = sequences[:, position]
        if not bool((column == column[0]).all()):
            break
        length += 1
    return length


def build_candidate_labels(
    polarities: Sequence[str],
    intensities: Sequence[float],
    tokenizer: Any,
    max_length: int = 24,
    device: Any = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenise the five category candidates for every instance.

    Args:
        polarities: Decoded polarity per instance.
        intensities: Decoded intensity per instance.
        tokenizer: Tokenizer used for the targets.
        max_length: Cap on the tokenised candidate length.
        device: Device to place the tensors on.

    Returns:
        ``labels`` of shape ``(batch * 5, seq_len)`` with ``IGNORE_INDEX`` at
        padding and at the shared prefix, and ``scored`` of the same shape
        marking the positions that entered the score. Candidates are ordered
        instance-major, matching ``ATTITUDE_CLASSES`` within each instance.
    """
    texts: List[str] = []
    for polarity, intensity in zip(polarities, intensities, strict=False):
        texts.extend(all_category_candidates(polarity, float(intensity)))

    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)

    batch_size = len(polarities)
    seq_len = ids.size(1)
    scored = mask.clone().bool()

    grouped_ids = ids.view(batch_size, N_CLASSES, seq_len)
    grouped_mask = mask.view(batch_size, N_CLASSES, seq_len).bool()
    grouped_scored = scored.view(batch_size, N_CLASSES, seq_len)

    for index in range(batch_size):
        shared = longest_common_prefix_length(grouped_ids[index], grouped_mask[index])
        grouped_scored[index, :, :shared] = False

    labels = ids.masked_fill(~scored, IGNORE_INDEX)
    return labels, scored


@torch.no_grad()
def category_probabilities(
    model: Any,
    encoder_outputs: BaseModelOutput,
    attention_mask: torch.Tensor,
    polarities: Sequence[str],
    intensities: Sequence[float],
    tokenizer: Any,
    temperature: float = 1.0,
    max_length: int = 24,
) -> torch.Tensor:
    """Score the five categories against a cached encoder output.

    Args:
        model: A :class:`~msa_arc.model.mul_mt5.MulMT5`.
        encoder_outputs: Output of ``model.encode`` for this batch.
        attention_mask: Encoder attention mask for this batch.
        polarities: Decoded polarity per instance.
        intensities: Decoded intensity per instance.
        tokenizer: Tokenizer used for the targets.
        temperature: Softmax temperature over the length-normalised scores.
        max_length: Cap on the tokenised candidate length.

    Returns:
        ``(batch, 5)`` probabilities, columns ordered as ``ATTITUDE_CLASSES``.
    """
    device = attention_mask.device
    batch_size = len(polarities)

    labels, scored = build_candidate_labels(
        polarities, intensities, tokenizer, max_length=max_length, device=device
    )

    expanded_states = encoder_outputs.last_hidden_state.repeat_interleave(N_CLASSES, dim=0)
    expanded_mask = attention_mask.repeat_interleave(N_CLASSES, dim=0)
    expanded_outputs = BaseModelOutput(last_hidden_state=expanded_states)

    logits = model.decoder_logits(expanded_outputs, expanded_mask, labels)

    log_probs = F.log_softmax(logits.float(), dim=-1)
    safe_labels = labels.masked_fill(~scored, 0)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~scored, 0.0)

    counts = scored.sum(dim=-1).clamp(min=1)
    normalized = token_log_probs.sum(dim=-1) / counts
    scores = normalized.view(batch_size, N_CLASSES)
    return F.softmax(scores / max(temperature, 1e-6), dim=-1)


def top_two(
    probabilities: torch.Tensor,
) -> Tuple[List[str], List[float], List[str], List[float]]:
    """Split a probability matrix into its top-two categories.

    Section 6.4 flips a low-confidence prediction to its runner-up, so both the
    maximum and the second-highest category are needed per instance.

    Args:
        probabilities: ``(batch, 5)`` distribution over ``ATTITUDE_CLASSES``.

    Returns:
        ``(top_categories, top_probabilities, second_categories,
        second_probabilities)``.
    """
    values, indices = probabilities.topk(2, dim=-1)
    top_categories = [ATTITUDE_CLASSES[i] for i in indices[:, 0].tolist()]
    second_categories = [ATTITUDE_CLASSES[i] for i in indices[:, 1].tolist()]
    return (
        top_categories,
        values[:, 0].tolist(),
        second_categories,
        values[:, 1].tolist(),
    )


__all__ = [
    "N_CLASSES",
    "build_candidate_labels",
    "category_probabilities",
    "longest_common_prefix_length",
    "top_two",
]
