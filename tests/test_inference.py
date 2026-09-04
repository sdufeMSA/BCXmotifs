"""Decoding, the fallback chain, and the five-class probability derivation."""

import torch

from msa_arc.config import DecodeConfig
from msa_arc.constants import ATTITUDE_CLASSES
from msa_arc.inference.decode import DecodeReport, decode_batch
from msa_arc.inference.probabilities import (
    build_candidate_labels,
    category_probabilities,
    longest_common_prefix_length,
    top_two,
)
from msa_arc.losses.generation import IGNORE_INDEX


def make_batch(model_config, batch_size: int = 3, seq_len: int = 10):
    torch.manual_seed(2)
    return {
        "input_ids": torch.randint(3, 20, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "audio_features": torch.randn(batch_size, 4, model_config.audio_input_dim),
        "audio_lengths": torch.tensor([4] * batch_size),
        "video_features": torch.randn(batch_size, 3, model_config.video_input_dim),
        "video_lengths": torch.tensor([3] * batch_size),
        "instance_key": [f"P000__s1__f{i}" for i in range(batch_size)],
        "participant_id": ["P000"] * batch_size,
        "service_id": ["s1"] * batch_size,
        "scenario": [1, 0, 1][:batch_size],
    }


def test_common_prefix_stops_where_candidates_diverge() -> None:
    sequences = torch.tensor([[5, 6, 7, 1], [5, 6, 9, 1], [5, 6, 8, 1]])
    mask = torch.ones_like(sequences)
    assert longest_common_prefix_length(sequences, mask) == 2


def test_candidate_labels_mask_the_shared_prefix(tokenizer) -> None:
    labels, scored = build_candidate_labels(["positive", "negative"], [0.8, -0.4], tokenizer)
    assert labels.shape[0] == 2 * len(ATTITUDE_CLASSES)
    # Every candidate of an instance masks the identical prefix.
    grouped = scored.view(2, len(ATTITUDE_CLASSES), -1)
    for instance in range(2):
        first_scored = grouped[instance].float().argmax(dim=1)
        assert bool((first_scored == first_scored[0]).all())
    assert bool((labels[~scored] == IGNORE_INDEX).all())


def test_probabilities_form_a_distribution(tiny_model, tokenizer, tiny_model_config) -> None:
    batch = make_batch(tiny_model_config)
    encoder_outputs = tiny_model.encode(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        audio_features=batch["audio_features"],
        audio_lengths=batch["audio_lengths"],
        video_features=batch["video_features"],
        video_lengths=batch["video_lengths"],
    )
    probabilities = category_probabilities(
        model=tiny_model,
        encoder_outputs=encoder_outputs,
        attention_mask=batch["attention_mask"],
        polarities=["positive", "negative", "neutral"],
        intensities=[0.8, -0.5, 0.0],
        tokenizer=tokenizer,
    )
    assert probabilities.shape == (3, len(ATTITUDE_CLASSES))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)
    assert bool((probabilities >= 0).all())


def test_temperature_flattens_the_distribution(
    tiny_model, tokenizer, tiny_model_config
) -> None:
    batch = make_batch(tiny_model_config, batch_size=1)
    encoder_outputs = tiny_model.encode(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    kwargs = {
        "model": tiny_model,
        "encoder_outputs": encoder_outputs,
        "attention_mask": batch["attention_mask"],
        "polarities": ["positive"],
        "intensities": [0.8],
        "tokenizer": tokenizer,
    }
    sharp = category_probabilities(temperature=0.1, **kwargs)
    flat = category_probabilities(temperature=10.0, **kwargs)
    assert float(sharp.max()) > float(flat.max())


def test_top_two_reports_the_runner_up() -> None:
    probabilities = torch.tensor([[0.5, 0.2, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.2, 0.5]])
    top, top_p, second, second_p = top_two(probabilities)
    assert top == ["Like", "Dislike"]
    assert second == ["Essential", "Live with"]
    assert top_p[0] > second_p[0]


def test_decode_batch_always_yields_a_valid_category(
    tiny_model, tokenizer, tiny_model_config
) -> None:
    """An untrained model generates noise; the fallback must still produce a label."""
    predictions, report = decode_batch(
        tiny_model, tokenizer, make_batch(tiny_model_config), DecodeConfig(num_beams=2)
    )
    assert len(predictions) == 3
    for prediction in predictions:
        assert prediction.category in ATTITUDE_CLASSES
        assert prediction.polarity in {"positive", "negative", "neutral"}
        assert -1.0 <= prediction.intensity <= 1.0
        assert len(prediction.probabilities) == len(ATTITUDE_CLASSES)
        assert prediction.decode_stage in {"beam", "greedy", "fallback"}
    assert report.n_instances == 3


def test_report_tracks_the_parse_failure_rate() -> None:
    report = DecodeReport(n_instances=1000, n_greedy_retries=3, n_fallbacks=1)
    assert report.parse_failure_rate == 0.003
    assert DecodeReport().parse_failure_rate == 0.0


def test_reports_merge() -> None:
    first = DecodeReport(n_instances=10, n_greedy_retries=1, failures=[{"a": "b"}])
    first.merge(DecodeReport(n_instances=5, n_greedy_retries=2, n_fallbacks=1))
    assert (first.n_instances, first.n_greedy_retries, first.n_fallbacks) == (15, 3, 1)
    assert len(first.failures) == 1
