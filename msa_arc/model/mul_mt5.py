"""Mul-mT5: the MSA-ARC model (Section 4.1.2, Appendix A.2.1).

MSA-ARC is a full encoder-decoder model.  The encoder consumes the
adapter-augmented text representation and the decoder generates the unified
target string of :mod:`msa_arc.model.target`.  No task-specific classification
head is attached: prediction is a generation step and the output layer is the
frozen pre-trained language-modelling head.

Only two groups of parameters are updated: the six cross-modal adapters and the
two LSTM branches.  Everything else is frozen, including the mT5
language-modelling head and the ImageNet-pretrained ResNet-50 that produced the
video features.

Adapters are attached with forward hooks rather than by rewriting the block
list, so the backbone's ``state_dict`` keys are untouched and a checkpoint
saved by this code loads into a stock ``MT5ForConditionalGeneration``.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from msa_arc.config import ModelConfig
from msa_arc.model import accounting
from msa_arc.model.adapter import CrossModalAdapter
from msa_arc.model.branches import LSTMBranch

logger = logging.getLogger(__name__)


@dataclass
class MulMT5Output:
    """What a forward pass returns.

    Attributes:
        logits: ``(batch, target_len, vocab)`` decoder logits over the unified
            target sequence.
        pooled: ``(batch, hidden)`` masked mean of the adapter-augmented encoder
            states.  Consumed only by the contrastive term; prediction never
            uses a pooled text vector.
        encoder_outputs: The encoder output, reusable for decoding and rescoring
            so the encoder never runs twice on the same instance.
    """

    logits: torch.Tensor
    pooled: torch.Tensor
    encoder_outputs: BaseModelOutput


def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average token states over the non-padding positions.

    Args:
        states: ``(batch, seq_len, hidden)``.
        mask: ``(batch, seq_len)`` with 1 for real tokens.

    Returns:
        ``(batch, hidden)``.
    """
    weights = mask.unsqueeze(-1).to(states.dtype)
    return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)


class MulMT5(nn.Module):
    """Adapter-fused multimodal mT5.

    Args:
        cfg: Architecture configuration.
        backbone: A loaded ``MT5ForConditionalGeneration``. Injected rather than
            constructed so tests can supply a small randomly-initialised model
            and run offline.
    """

    def __init__(self, cfg: ModelConfig, backbone: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone

        hidden_dim = cfg.hidden_dim
        self.audio_branch = (
            LSTMBranch(cfg.audio_input_dim, hidden_dim, cfg.lstm_layers, cfg.dropout)
            if cfg.use_audio
            else None
        )
        self.video_branch = (
            LSTMBranch(cfg.video_input_dim, hidden_dim, cfg.lstm_layers, cfg.dropout)
            if cfg.use_video
            else None
        )

        self.adapters = nn.ModuleList(
            CrossModalAdapter(
                hidden_dim=hidden_dim,
                bottleneck_dim=cfg.bottleneck_dim,
                activation=cfg.adapter_activation,
                dropout=cfg.dropout,
            )
            for _ in cfg.adapter_layers
        )

        self._audio_context: Optional[torch.Tensor] = None
        self._video_context: Optional[torch.Tensor] = None
        self._hook_handles: List[Any] = []
        self._attach_adapters()

        if cfg.freeze_backbone:
            self.freeze_backbone()

    # adapter wiring

    def _attach_adapters(self) -> None:
        """Hook one adapter after the feed-forward sub-layer of each chosen block."""
        blocks = self.backbone.encoder.block
        n_blocks = len(blocks)
        for adapter, layer_index in zip(self.adapters, self.cfg.adapter_layers, strict=False):
            if layer_index >= n_blocks:
                raise ValueError(
                    f"adapter layer {layer_index} is out of range for an encoder "
                    f"with {n_blocks} blocks"
                )
            handle = blocks[layer_index].register_forward_hook(self._make_hook(adapter))
            self._hook_handles.append(handle)
        logger.info(
            "Attached %d cross-modal adapters to encoder layers %s",
            len(self.adapters),
            list(self.cfg.adapter_layers),
        )

    def _make_hook(self, adapter: CrossModalAdapter):
        """Build the forward hook that applies ``adapter`` to a block's output."""

        def hook(module: nn.Module, inputs: Any, output: Any) -> Any:
            if self._audio_context is None or self._video_context is None:
                # No multimodal context set: behave as the unmodified backbone.
                return output
            hidden = output[0]
            updated = adapter(hidden, self._audio_context, self._video_context)
            return (updated,) + tuple(output[1:])

        return hook

    def _zero_context(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return torch.zeros(batch_size, self.cfg.hidden_dim, device=device, dtype=dtype)

    # freezing

    def freeze_backbone(self) -> None:
        """Freeze every backbone parameter, language-modelling head included."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "MulMT5":
        """Keep the frozen backbone in eval mode so its dropout stays off."""
        super().train(mode)
        if self.cfg.freeze_backbone:
            self.backbone.eval()
        return self

    # forward paths

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        video_lengths: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        """Run the encoder with acoustic and visual cues injected.

        Args:
            input_ids: ``(batch, seq_len)`` mT5 token ids.
            attention_mask: ``(batch, seq_len)``.
            audio_features: ``(batch, frames, 40)`` Mel-spectrogram.
            audio_lengths: ``(batch,)`` true frame counts.
            video_features: ``(batch, frames, 2048)`` ResNet-50 features.
            video_lengths: ``(batch,)`` true frame counts.

        Returns:
            The encoder output, with adapters already applied.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        dtype = next(self.adapters.parameters()).dtype

        if self.audio_branch is not None and audio_features is not None:
            audio_vector = self.audio_branch(audio_features, audio_lengths)
        else:
            audio_vector = self._zero_context(batch_size, device, dtype)

        if self.video_branch is not None and video_features is not None:
            video_vector = self.video_branch(video_features, video_lengths)
        else:
            video_vector = self._zero_context(batch_size, device, dtype)

        self._audio_context = audio_vector
        self._video_context = video_vector
        try:
            encoder_outputs = self.backbone.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        finally:
            self._audio_context = None
            self._video_context = None
        return encoder_outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        video_lengths: Optional[torch.Tensor] = None,
    ) -> MulMT5Output:
        """Encode, then decode the unified target sequence.

        Args:
            labels: ``(batch, target_len)`` target token ids with ``-100`` at
                padding positions. The backbone derives ``decoder_input_ids``
                from these by shifting right.

        Returns:
            A :class:`MulMT5Output`.
        """
        encoder_outputs = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_lengths=audio_lengths,
            video_features=video_features,
            video_lengths=video_lengths,
        )
        outputs = self.backbone(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        pooled = masked_mean(encoder_outputs.last_hidden_state, attention_mask)
        return MulMT5Output(
            logits=outputs.logits,
            pooled=pooled,
            encoder_outputs=encoder_outputs,
        )

    def decoder_logits(
        self,
        encoder_outputs: BaseModelOutput,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Score given target sequences against a cached encoder output.

        Used by the constrained rescoring that turns the generative model into a
        normalised five-class distribution.

        Args:
            encoder_outputs: Output of :meth:`encode`.
            attention_mask: Encoder attention mask, matching the batch of
                ``encoder_outputs``.
            labels: ``(batch, target_len)`` candidate target ids.

        Returns:
            ``(batch, target_len, vocab)`` logits.
        """
        outputs = self.backbone(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
        video_lengths: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Generate target strings.

        Beam search expands the encoder states to ``batch * num_beams``, and
        ``generate`` performs that expansion by assigning into the
        ``encoder_outputs`` object it is handed. A caller that reuses the same
        object afterwards, for the greedy retry or for the constrained
        rescoring, would find it silently reshaped. A shallow copy is passed so
        the caller's object is left as the encoder produced it.

        Args:
            encoder_outputs: Optional cached encoder output. Supplying it keeps
                the encoder from running twice when the caller also needs the
                states for rescoring.
            **generate_kwargs: Forwarded to ``MT5ForConditionalGeneration.generate``.

        Returns:
            ``(batch, generated_len)`` token ids.
        """
        if encoder_outputs is None:
            encoder_outputs = self.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_features,
                audio_lengths=audio_lengths,
                video_features=video_features,
                video_lengths=video_lengths,
            )
        return self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state
            ),
            **generate_kwargs,
        )

    # accounting

    def parameter_report(self) -> Dict[str, int]:
        """Parameter breakdown; see :func:`msa_arc.model.accounting.parameter_report`."""
        return accounting.parameter_report(self)

    def expected_parameter_report(self) -> Dict[str, int]:
        """Closed-form counts implied by this model's configuration."""
        return accounting.expected_parameter_report(self.cfg)

    def trainable_share(self) -> float:
        """Fraction of parameters that are fitted, counting the Stage-A ResNet-50.

        Returns:
            The 3.8% the appendix reports, expressed as a fraction.
        """
        report = self.parameter_report()
        return report["trainable"] / report["total_with_stage_a"]

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Only the parameters this model actually fits, cloned onto the CPU."""
        return accounting.trainable_state_dict(self)

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Load a checkpoint produced by :meth:`trainable_state_dict`."""
        accounting.load_trainable_state_dict(self, state)


__all__ = ["MulMT5", "MulMT5Output", "masked_mean"]
