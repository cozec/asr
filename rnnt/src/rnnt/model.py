"""Stateless prediction network, joint network, and RNN-T assembly.

The prediction network follows "RNN-Transducer with Stateless Prediction Network"
(Ghodsi et al., ICASSP 2020) rather than the 2x LSTM(2048)+640-projection network of
He et al. §3.1: prediction is conditioned only on the last symbol(s), "acting effectively
as a 2-gram LM on the output subword set", at comparable WER.

Two consequences worth stating:

* It removes recurrent state from the decode loop, which makes He et al. §3.3's
  prediction-network state caching unnecessary -- there is nothing left to cache.
* Ghodsi §3.4 finds this costs little with wordpieces but regresses badly with
  graphemes: seeing one symbol back, the model cannot tell whether it already emitted
  one 'o' or two in "food", so it matches 'foo*d'. This project uses wordpieces.
"""

import torch
import torch.nn as nn
import torchaudio

from .transcriber import LSTMTranscriber

# Note on base classes: torchaudio's `_Transcriber` is a genuine ABC (interface), so the
# encoder inherits it. `_Predictor` and `_Joiner`, despite the naming, are *concrete*
# LSTM/joiner implementations rather than interfaces -- inheriting them would drag in
# their constructors. `RNNT` duck-types its components, so plain nn.Modules suffice.


class StatelessPredictor(nn.Module):
    """Embedding over the last `context_size` tokens, then a Conv1d (Ghodsi et al. 2020).

    `context_size=1` is the paper's pure 2-gram formulation (embedding of the last symbol
    only); `context_size=2` matches k2/icefall's `pruned_transducer_stateless` default.

    The "state" required by torchaudio's interface is not an RNN state here -- it is
    simply the trailing `context_size - 1` token ids, which is what makes streaming
    decode produce results identical to a full-sequence forward pass.
    """

    def __init__(self, num_symbols: int, output_dim: int, symbol_embedding_dim: int = 320,
                 context_size: int = 2, blank: int = 0, dropout: float = 0.1):
        super().__init__()
        assert context_size >= 1, "context_size must be at least 1"
        self.context_size = context_size
        self.blank = blank
        self.embedding = nn.Embedding(num_symbols, symbol_embedding_dim)
        # groups=1 (dense) rather than icefall's grouped conv: our widths are small, and
        # it avoids a divisibility constraint on the embedding dim.
        self.conv = nn.Conv1d(symbol_embedding_dim, symbol_embedding_dim,
                              kernel_size=context_size, bias=False)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(symbol_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(symbol_embedding_dim, output_dim)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor, state=None):
        """input: (B, U) token ids. Returns ((B, U, output_dim), lengths, state)."""
        b = input.size(0)
        pad = self.context_size - 1
        if pad:
            if state is None:
                # Start of an utterance: history is all blanks (the transducer's SOS).
                prev = torch.full((b, pad), self.blank, dtype=torch.long,
                                  device=input.device)
            else:
                prev = state[0][0].to(input.device)
            full = torch.cat([prev, input], dim=1)
        else:
            full = input

        x = self.embedding(full).transpose(1, 2)        # (B, D, pad + U)
        x = self.conv(x).transpose(1, 2)                # (B, U, D)
        x = self.dropout(self.relu(self.layer_norm(x)))
        out = self.output_proj(x)

        new_state = [[full[:, full.size(1) - pad:]]] if pad else [[torch.empty(b, 0, dtype=torch.long, device=input.device)]]
        return out, lengths, new_state


class Joiner(nn.Module):
    """Feed-forward joint network (He et al. §3.1: a single 640-unit layer).

    Mirrors the joint network verified in the conformer project: project both streams to
    a common width, add, tanh, then project to the vocabulary.
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = "tanh"):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = {"tanh": nn.Tanh(), "relu": nn.ReLU()}[activation]

    def forward(self, source_encodings: torch.Tensor, source_lengths: torch.Tensor,
                target_encodings: torch.Tensor, target_lengths: torch.Tensor):
        """(B, T, D) x (B, U, D) -> ((B, T, U, V), source_lengths, target_lengths)."""
        joint = source_encodings.unsqueeze(2) + target_encodings.unsqueeze(1)
        return self.linear(self.activation(joint)), source_lengths, target_lengths


def build_rnnt(cfg: dict, num_symbols: int) -> torchaudio.models.RNNT:
    """Assemble the three components into torchaudio's RNNT container.

    Using torchaudio's container is what lets `RNNTBeamSearch` -- including its streaming
    `infer(..., state, hypothesis)` path -- drive this model unmodified, exactly as it
    drives the pretrained Emformer pipeline.
    """
    m = cfg["model"]
    transcriber = LSTMTranscriber(
        input_dim=m["input_dim"] * m["frame_stack"],
        num_layers=m["num_layers"],
        hidden_dim=m["hidden_dim"],
        proj_dim=m["proj_dim"],
        time_reduction_factor=m["time_reduction_factor"],
        time_reduction_after=m["time_reduction_after"],
        dropout=m["dropout"],
    )
    predictor = StatelessPredictor(
        num_symbols=num_symbols,
        output_dim=m["joint_dim"],
        symbol_embedding_dim=m["symbol_embedding_dim"],
        context_size=m["context_size"],
        dropout=m["dropout"],
    )
    # The transcriber's final width is the projection dim, unless time reduction is the
    # very last thing it does; project it to the joint width.
    transcriber_proj = nn.Linear(transcriber.output_dim, m["joint_dim"])
    joiner = Joiner(m["joint_dim"], num_symbols)
    return torchaudio.models.RNNT(_ProjectedTranscriber(transcriber, transcriber_proj),
                                  predictor, joiner)


class _ProjectedTranscriber(nn.Module):
    """Wraps the encoder with a linear projection to the joint-network width."""

    def __init__(self, transcriber: LSTMTranscriber, proj: nn.Linear):
        super().__init__()
        self.transcriber = transcriber
        self.proj = proj

    def forward(self, input: torch.Tensor, lengths: torch.Tensor):
        x, lengths = self.transcriber(input, lengths)
        return self.proj(x), lengths

    def infer(self, input: torch.Tensor, lengths: torch.Tensor, states):
        x, lengths, states = self.transcriber.infer(input, lengths, states)
        return self.proj(x), lengths, states
