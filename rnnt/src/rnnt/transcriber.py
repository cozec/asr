"""RNN-T encoder ("transcriber"), following He et al. 2019 (arXiv:1811.06621) §3.1.

The paper's encoder is eight uni-directional LSTM layers of 2048 units, each followed by
a 640-dimensional projection, with layer normalisation on every layer and a time-reduction
layer (N=2) inserted after the second layer.

This implements that structure at a width trainable on one machine (see configs). Being
uni-directional, it is streaming by construction: `infer` carries per-layer (h, c) across
chunks, and the `_Transcriber` interface it satisfies is torchaudio's, so the model drops
straight into `torchaudio.models.RNNT` and `RNNTBeamSearch`.
"""

import torch
import torch.nn as nn
from torchaudio.models.rnnt import _Transcriber


class TimeReduction(nn.Module):
    """Concatenate N adjacent frames, reducing the frame rate by N (paper §3.1).

    The paper stacks N inputs so output frame i+1 is
    [h_{iN}; h_{iN+1}; ...; h_{(i+1)N-1}], and reports a 1.7x overall speedup with no
    accuracy loss when inserted after the second encoder layer.
    """

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """x: (B, T, D) -> (B, T // N, D * N)."""
        b, t, d = x.size()
        usable = (t // self.factor) * self.factor
        # Dropping the remainder keeps every output frame built from exactly N inputs.
        x = x[:, :usable].reshape(b, usable // self.factor, d * self.factor)
        return x, torch.div(lengths, self.factor, rounding_mode="floor").clamp(min=1)


class _LSTMLayer(nn.Module):
    """One uni-directional LSTM + projection + layer norm (paper §3.1).

    The projection after each LSTM is what keeps the recurrent connections cheap: the
    paper uses 2048 units projected to 640.
    """

    def __init__(self, input_dim: int, hidden_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.layer_norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state=None):
        x, state = self.lstm(x, state)
        return self.dropout(self.layer_norm(self.proj(x))), state


class LSTMTranscriber(_Transcriber, nn.Module):
    """Stacked LSTM encoder with a time-reduction layer (paper §3.1).

    Satisfies torchaudio's `_Transcriber` interface:
      forward(input, lengths)          -> (encodings, lengths)                # training
      infer(input, lengths, states)    -> (encodings, lengths, states)        # streaming
    """

    def __init__(self, input_dim: int = 80, num_layers: int = 6, hidden_dim: int = 1024,
                 proj_dim: int = 320, time_reduction_factor: int = 2,
                 time_reduction_after: int = 2, dropout: float = 0.1):
        super().__init__()
        self.time_reduction_after = time_reduction_after
        self.time_reduction = TimeReduction(time_reduction_factor)

        layers, dim = [], input_dim
        for i in range(num_layers):
            layers.append(_LSTMLayer(dim, hidden_dim, proj_dim, dropout))
            dim = proj_dim
            # Frames are concatenated here, so the next layer sees factor x the width.
            if i + 1 == time_reduction_after:
                dim = proj_dim * time_reduction_factor
        self.layers = nn.ModuleList(layers)
        # `dim` now holds what the next layer would consume, i.e. the stack's output
        # width. It is proj_dim normally, but proj_dim * factor when time reduction
        # lands after the final layer.
        self.output_dim = dim

    def _run(self, x: torch.Tensor, lengths: torch.Tensor, states=None):
        """Shared by forward and infer; `states` is None for full-context training."""
        new_states = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, None if states is None else states[i])
            new_states.append(state)
            if i + 1 == self.time_reduction_after:
                x, lengths = self.time_reduction(x, lengths)
        return x, lengths, new_states

    def forward(self, input: torch.Tensor, lengths: torch.Tensor):
        encodings, out_lengths, _ = self._run(input, lengths, None)
        return encodings, out_lengths

    def infer(self, input: torch.Tensor, lengths: torch.Tensor, states):
        """Streaming step. `states` is the list of per-layer (h, c) from the last call.

        Note the time-reduction layer drops a trailing frame when a chunk carries an odd
        number of frames, so chunk sizes should be chosen as multiples of the reduction
        factor (the configs do this).
        """
        return self._run(input, lengths, states)
