"""Non-text encoding branches (Section 4.1.1, Appendix A.2.1).

Both branches are 2-layer LSTMs whose last hidden state is the global
representation of the modality:

* audio: 40-bank Mel-spectrogram -> LSTM(40 -> 768) -> ``X^a``
* video: ResNet-50 frame features -> LSTM(2048 -> 768) -> ``X^v``

Both branches emit ``d_h = 768`` directly, so no projection is needed to make
them dimensionally compatible with the text stream.  The audio branch therefore
holds 7,213,056 parameters and the video branch 13,381,632.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMBranch(nn.Module):
    """Encode a padded sequence into a single vector via its last hidden state.

    Args:
        input_dim: Per-timestep feature dimension (40 for audio, 2048 for video).
        hidden_dim: LSTM hidden size; must equal the backbone hidden size so the
            branch output can be concatenated into the adapter without a
            projection.
        num_layers: Number of stacked LSTM layers.
        dropout: Inter-layer dropout; adds no parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the branch.

        Args:
            features: ``(batch, max_len, input_dim)`` padded feature sequence.
            lengths: ``(batch,)`` true lengths. When given, the sequence is
                packed so that padding never reaches the recurrence and the
                returned state is the state at the true final timestep.

        Returns:
            ``(batch, hidden_dim)`` last hidden state of the top layer.
        """
        if lengths is None:
            _, (hidden, _) = self.lstm(features)
            return hidden[-1]

        # pack_padded_sequence rejects zero lengths; an instance with no frames
        # is clamped to one padded step and contributes a near-zero vector.
        safe_lengths = lengths.clamp(min=1).to(device="cpu", dtype=torch.int64)
        packed = pack_padded_sequence(
            features, safe_lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]

    def unpacked_sequence(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Full per-timestep output; kept for diagnostics, unused in training."""
        safe_lengths = lengths.clamp(min=1).to(device="cpu", dtype=torch.int64)
        packed = pack_padded_sequence(
            features, safe_lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        padded, _ = pad_packed_sequence(output, batch_first=True)
        return padded

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}"


def expected_lstm_parameters(input_dim: int, hidden_dim: int, num_layers: int) -> int:
    """Closed-form parameter count for a stacked unidirectional LSTM.

    Each layer holds ``4 * hidden * (in + hidden)`` weights plus two bias
    vectors of ``4 * hidden``. With the paper's settings this returns 7,213,056
    for the audio branch and 13,381,632 for the video branch.

    Args:
        input_dim: Input feature dimension of the first layer.
        hidden_dim: Hidden size.
        num_layers: Number of stacked layers.

    Returns:
        Total number of parameters.
    """
    total = 0
    for layer in range(num_layers):
        in_dim = input_dim if layer == 0 else hidden_dim
        total += 4 * hidden_dim * (in_dim + hidden_dim) + 8 * hidden_dim
    return total


__all__ = ["LSTMBranch", "expected_lstm_parameters"]
