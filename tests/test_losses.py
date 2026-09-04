"""Objective terms: token-level generation loss and class-aware contrastive loss."""

import pytest
import torch

from msa_arc.losses.contrastive import contrastive_loss, pairwise_squared_distances
from msa_arc.losses.generation import (
    IGNORE_INDEX,
    combined_loss,
    generation_loss,
    sequence_log_probabilities,
)


def test_generation_loss_ignores_padding() -> None:
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[3, 4, IGNORE_INDEX, IGNORE_INDEX], [5, 6, 7, IGNORE_INDEX]])
    loss = generation_loss(logits, labels)
    assert loss.ndim == 0 and torch.isfinite(loss)

    # Changing the logits under a masked position must not change the loss.
    perturbed = logits.clone()
    perturbed[0, 2] += 100.0
    assert generation_loss(perturbed, labels) == pytest.approx(float(loss), abs=1e-6)


def test_perfect_predictions_drive_the_loss_to_zero() -> None:
    labels = torch.tensor([[1, 2, 3]])
    logits = torch.full((1, 3, 8), -50.0)
    for position, token in enumerate(labels[0]):
        logits[0, position, token] = 50.0
    assert generation_loss(logits, labels) < 1e-6


def test_sequence_log_probabilities_are_length_normalised() -> None:
    logits = torch.randn(3, 5, 12)
    labels = torch.tensor(
        [
            [1, 2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
            [1, 2, 3, IGNORE_INDEX, IGNORE_INDEX],
            [1, 2, 3, 4, 5],
        ]
    )
    normalised = sequence_log_probabilities(logits, labels, length_normalize=True)
    totals = sequence_log_probabilities(logits, labels, length_normalize=False)
    assert normalised.shape == (3,)
    assert normalised[2] == pytest.approx(float(totals[2] / 5), abs=1e-5)


def test_combined_loss_drops_the_contrastive_term_when_it_is_absent() -> None:
    generation = torch.tensor(2.0)
    assert combined_loss(generation, None, 0.7, 0.3) == pytest.approx(1.4)
    assert combined_loss(generation, torch.tensor(1.0), 0.7, 0.3) == pytest.approx(1.7)


def test_pairwise_distances_of_normalised_vectors() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(4, 6), dim=-1)
    distances = pairwise_squared_distances(embeddings)
    assert torch.allclose(distances.diagonal(), torch.zeros(4), atol=1e-6)
    assert torch.allclose(distances, distances.t(), atol=1e-6)
    assert bool((distances >= -1e-6).all())


def test_contrastive_loss_is_none_without_an_in_batch_positive() -> None:
    """Anchors with no same-class partner contribute nothing rather than a fake pair."""
    embeddings = torch.randn(3, 8)
    labels = torch.tensor([0, 1, 2])
    assert contrastive_loss(embeddings, labels) is None


def test_contrastive_loss_rewards_class_structure() -> None:
    tight = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    scrambled = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01], [0.01, 0.99]])
    labels = torch.tensor([0, 0, 1, 1])
    assert contrastive_loss(tight, labels) < contrastive_loss(scrambled, labels)


def test_contrastive_loss_is_deterministic_under_batch_hard_mining() -> None:
    embeddings = torch.randn(8, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    first = contrastive_loss(embeddings, labels, negative_mining="batch_hard")
    second = contrastive_loss(embeddings, labels, negative_mining="batch_hard")
    assert float(first) == float(second)


def test_unknown_mining_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown negative_mining"):
        contrastive_loss(
            torch.randn(4, 4), torch.tensor([0, 0, 1, 1]), negative_mining="magic"
        )
