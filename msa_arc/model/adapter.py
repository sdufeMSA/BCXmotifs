"""Cross-modal adapter block (Section 4.1.2, Appendix A.2.1).

The block extends the parameter-efficient adapter of Houlsby et al. (2019) from
single-modal fine-tuning to cross-modal injection.  The textual stream stays the
organising sequence; the global acoustic and visual vectors are broadcast to
token level, concatenated onto every token, and pushed through a bottleneck
whose output is added back residually:

    F^{(c,l)} = F^{(l-1)} || 1 (x) X^a || 1 (x) X^v
    F^{d,l}   = sigma(W^d F^{(c,l)} + b^d)
    F^{u,l}   = W^u F^{d,l} + b^u
    F^{(l)}   = F^{(l-1)} + F^{u,l}

With ``hidden_dim=768`` and ``bottleneck_dim=192`` one block holds 590,784
parameters and six blocks hold 3,544,704.
:func:`msa_arc.model.build_model` asserts that count at construction.
"""

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class CrossModalAdapter(nn.Module):
    """One bottleneck adapter that injects acoustic and visual cues.

    Args:
        hidden_dim: Backbone hidden size ``d_h``.
        bottleneck_dim: Adapter bottleneck ``d_b``; the paper uses ``d_h / 4``.
        activation: Non-linearity applied after the down-projection.
        dropout: Dropout applied to the bottleneck activations.
    """

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; expected one of {sorted(_ACTIVATIONS)}"
            )
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # Input is [text || audio || video], hence three times the hidden size.
        self.down = nn.Linear(3 * hidden_dim, bottleneck_dim)
        self.activation = _ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)

        self._init_near_identity()

    def _init_near_identity(self) -> None:
        """Start the block close to a no-op.

        A randomly initialised adapter would perturb the frozen backbone
        representation before it has learned anything useful.  Zeroing the
        up-projection makes the residual branch start at exactly zero, so the
        first forward pass reproduces the frozen backbone.
        """
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        text_states: torch.Tensor,
        audio_vector: torch.Tensor,
        video_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Inject the two non-text vectors into the token sequence.

        Args:
            text_states: ``(batch, seq_len, hidden_dim)`` textual stream.
            audio_vector: ``(batch, hidden_dim)`` global prosodic representation.
            video_vector: ``(batch, hidden_dim)`` global facial representation.

        Returns:
            ``(batch, seq_len, hidden_dim)`` residually updated text states.
        """
        seq_len = text_states.size(1)
        audio = audio_vector.unsqueeze(1).expand(-1, seq_len, -1)
        video = video_vector.unsqueeze(1).expand(-1, seq_len, -1)

        concatenated = torch.cat([text_states, audio, video], dim=-1)
        bottleneck = self.dropout(self.activation(self.down(concatenated)))
        return text_states + self.up(bottleneck)

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, bottleneck_dim={self.bottleneck_dim}"


def expected_adapter_parameters(hidden_dim: int, bottleneck_dim: int, n_blocks: int) -> int:
    """Closed-form parameter count, used to check the code against the paper.

    Args:
        hidden_dim: Backbone hidden size.
        bottleneck_dim: Adapter bottleneck size.
        n_blocks: Number of adapter blocks.

    Returns:
        Total number of adapter parameters.
    """
    down = 3 * hidden_dim * bottleneck_dim + bottleneck_dim
    up = bottleneck_dim * hidden_dim + hidden_dim
    return n_blocks * (down + up)


__all__ = ["CrossModalAdapter", "expected_adapter_parameters"]
