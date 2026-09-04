"""The adapters must actually inject the non-text cues, and generation must run."""

import torch

from msa_arc.model.mul_mt5 import masked_mean


def make_batch(tokenizer, model_config, batch_size: int = 3, seq_len: int = 12):
    torch.manual_seed(1)
    return {
        "input_ids": torch.randint(3, 20, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "audio_features": torch.randn(batch_size, 5, model_config.audio_input_dim),
        "audio_lengths": torch.tensor([5, 4, 3]),
        "video_features": torch.randn(batch_size, 4, model_config.video_input_dim),
        "video_lengths": torch.tensor([4, 4, 2]),
    }


def test_encode_returns_the_backbone_shape(tiny_model, tokenizer, tiny_model_config) -> None:
    batch = make_batch(tokenizer, tiny_model_config)
    encoded = tiny_model.encode(**batch)
    assert encoded.last_hidden_state.shape == (3, 12, tiny_model_config.hidden_dim)


def test_adapters_change_the_representation_once_trained(
    tiny_model, tokenizer, tiny_model_config
) -> None:
    """Adapters start as a no-op; a non-zero up-projection must take effect."""
    batch = make_batch(tokenizer, tiny_model_config)
    before = tiny_model.encode(**batch).last_hidden_state.clone()

    with torch.no_grad():
        for adapter in tiny_model.adapters:
            adapter.up.weight.normal_(std=0.5)
    after = tiny_model.encode(**batch).last_hidden_state

    assert not torch.allclose(before, after)


def test_different_audio_produces_a_different_encoding(
    tiny_model, tokenizer, tiny_model_config
) -> None:
    with torch.no_grad():
        for adapter in tiny_model.adapters:
            adapter.up.weight.normal_(std=0.5)

    batch = make_batch(tokenizer, tiny_model_config)
    first = tiny_model.encode(**batch).last_hidden_state.clone()
    batch["audio_features"] = torch.randn_like(batch["audio_features"]) * 5
    second = tiny_model.encode(**batch).last_hidden_state
    assert not torch.allclose(first, second)


def test_forward_produces_logits_and_a_pooled_vector(
    tiny_model, tokenizer, tiny_model_config
) -> None:
    batch = make_batch(tokenizer, tiny_model_config)
    labels = torch.randint(3, 20, (3, 6))
    output = tiny_model(labels=labels, **batch)
    assert output.logits.shape[:2] == (3, 6)
    assert output.pooled.shape == (3, tiny_model_config.hidden_dim)


def test_gradients_reach_the_branches_but_not_the_backbone(
    tiny_model, tokenizer, tiny_model_config
) -> None:
    from msa_arc.losses.generation import generation_loss

    batch = make_batch(tokenizer, tiny_model_config)
    labels = torch.randint(3, 20, (3, 6))
    output = tiny_model(labels=labels, **batch)
    generation_loss(output.logits, labels).backward()

    assert any(
        p.grad is not None and bool((p.grad != 0).any())
        for p in tiny_model.adapters.parameters()
    )
    assert all(p.grad is None for p in tiny_model.backbone.parameters())


def test_generate_runs_with_beam_search(tiny_model, tokenizer, tiny_model_config) -> None:
    batch = make_batch(tokenizer, tiny_model_config)
    generated = tiny_model.generate(num_beams=2, max_new_tokens=8, do_sample=False, **batch)
    assert generated.shape[0] == 3
    assert len(tokenizer.batch_decode(generated)) == 3


def test_masked_mean_ignores_padding() -> None:
    states = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    torch.testing.assert_close(masked_mean(states, mask), torch.tensor([[2.0, 2.0]]))


def test_trainable_state_dict_excludes_the_backbone(tiny_model) -> None:
    state = tiny_model.trainable_state_dict()
    assert state
    assert not any(name.startswith("backbone.") for name in state)


def test_checkpoint_round_trips(tiny_model, tokenizer, tiny_model_config) -> None:
    with torch.no_grad():
        for adapter in tiny_model.adapters:
            adapter.up.weight.normal_(std=0.5)
    state = tiny_model.trainable_state_dict()

    batch = make_batch(tokenizer, tiny_model_config)
    expected = tiny_model.encode(**batch).last_hidden_state.clone()

    with torch.no_grad():
        for adapter in tiny_model.adapters:
            adapter.up.weight.zero_()
    tiny_model.load_trainable_state_dict(state)
    torch.testing.assert_close(tiny_model.encode(**batch).last_hidden_state, expected)


def test_generate_leaves_the_caller_encoder_outputs_untouched(
    tiny_model, tokenizer, tiny_model_config
) -> None:
    """Beam search expands encoder states; it must not reshape our copy.

    ``decode_batch`` reuses one encoder pass for the beam decode, the greedy
    retry and the constrained rescoring. If ``generate`` reshaped the shared
    object, every later use would silently see ``batch * num_beams`` rows.
    """
    batch = make_batch(tokenizer, tiny_model_config)
    encoder_outputs = tiny_model.encode(**batch)
    before = encoder_outputs.last_hidden_state.shape

    tiny_model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        encoder_outputs=encoder_outputs,
        num_beams=2,
        do_sample=False,
        max_new_tokens=6,
    )
    assert encoder_outputs.last_hidden_state.shape == before
