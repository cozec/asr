"""Conformer ASR models: the paper's RNN-T, plus a CTC variant.

The paper (§3.2) uses a Transducer with a single-LSTM-layer decoder for all three
model sizes; `ConformerTransducer` is that model. `ConformerCTC` swaps the decoder for
a linear CTC head -- far cheaper to train and decode, which is what makes local
single-machine runs practical. Both share the same `ConformerEncoder`.
"""

import torch
import torch.nn as nn

from .encoder import ConformerEncoder


class ConformerCTC(nn.Module):
    """Conformer encoder + linear CTC head."""

    def __init__(self, num_classes: int, input_dim: int = 80, encoder_dim: int = 144,
                 num_layers: int = 16, num_heads: int = 4, ffn_expansion: int = 4,
                 conv_expansion: int = 2, conv_kernel_size: int = 31,
                 dropout: float = 0.1, half_step_residual: bool = True):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_dim, encoder_dim, num_layers, num_heads, ffn_expansion,
            conv_expansion, conv_kernel_size, dropout, half_step_residual,
        )
        self.fc = nn.Linear(encoder_dim, num_classes)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        """Returns (log_probs (B, T', V), output_lengths (B,))."""
        encoded, out_lengths = self.encoder(inputs, input_lengths)
        return self.fc(encoded).log_softmax(dim=-1), out_lengths


class TransducerDecoder(nn.Module):
    """Prediction network: label embedding + single LSTM layer (paper §3.2, Table 1)."""

    def __init__(self, num_classes: int, hidden_dim: int = 320, num_layers: int = 1,
                 blank_id: int = 0, dropout: float = 0.1):
        super().__init__()
        self.blank_id = blank_id
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, targets: torch.Tensor, hidden=None):
        """targets: (B, U) label ids. Returns (B, U, hidden_dim) and LSTM state."""
        outputs, hidden = self.lstm(self.embedding(targets), hidden)
        return outputs, hidden


class JointNetwork(nn.Module):
    """Joint network combining encoder and decoder states into logits over the vocab."""

    def __init__(self, num_classes: int, encoder_dim: int, decoder_dim: int,
                 joint_dim: int = 320):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, joint_dim)
        self.decoder_proj = nn.Linear(decoder_dim, joint_dim)
        self.out = nn.Linear(joint_dim, num_classes)

    def forward(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor):
        """(B, T, De) x (B, U, Dd) -> (B, T, U, V) logits."""
        enc = self.encoder_proj(encoder_out).unsqueeze(2)    # (B, T, 1, J)
        dec = self.decoder_proj(decoder_out).unsqueeze(1)    # (B, 1, U, J)
        return self.out(torch.tanh(enc + dec))


class ConformerTransducer(nn.Module):
    """The paper's model: Conformer encoder + LSTM prediction net + joint network."""

    def __init__(self, num_classes: int, input_dim: int = 80, encoder_dim: int = 144,
                 num_layers: int = 16, num_heads: int = 4, ffn_expansion: int = 4,
                 conv_expansion: int = 2, conv_kernel_size: int = 31,
                 decoder_dim: int = 320, decoder_layers: int = 1, joint_dim: int = 320,
                 dropout: float = 0.1, half_step_residual: bool = True,
                 blank_id: int = 0):
        super().__init__()
        self.blank_id = blank_id
        self.encoder = ConformerEncoder(
            input_dim, encoder_dim, num_layers, num_heads, ffn_expansion,
            conv_expansion, conv_kernel_size, dropout, half_step_residual,
        )
        self.decoder = TransducerDecoder(num_classes, decoder_dim, decoder_layers,
                                         blank_id, dropout)
        self.joint = JointNetwork(num_classes, encoder_dim, decoder_dim, joint_dim)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor,
                targets: torch.Tensor, target_lengths: torch.Tensor):
        """Returns (logits (B, T', U+1, V), encoder output lengths).

        `targets` must already be prepended with a blank (the transducer's SOS), so
        its length axis is U+1 -- see `rnnt_collate` in the data pipeline.
        """
        encoded, out_lengths = self.encoder(inputs, input_lengths)
        decoded, _ = self.decoder(targets)
        return self.joint(encoded, decoded), out_lengths
