"""Conformer encoder: convolution subsampling + N Conformer blocks (paper §2)."""

import torch
import torch.nn as nn

from .modules import (
    ConvolutionModule,
    Conv2dSubsampling,
    FeedForwardModule,
    MultiHeadedSelfAttentionModule,
    RelPositionalEncoding,
)


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) lengths -> (B, max_len) bool mask, True on valid frames."""
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


class ConformerBlock(nn.Module):
    """A Conformer block (paper §2.4, Eq. 1).

        x~ = x  + 1/2 FFN(x)
        x' = x~ +     MHSA(x~)
        x''= x' +     Conv(x')
        y  = LayerNorm(x'' + 1/2 FFN(x''))

    The half-step residual weights on the macaron FFN pair are the paper's default
    (Table 5 shows full-step residuals cost 0.2 WER on test-other).
    """

    def __init__(self, dim: int, num_heads: int, ffn_expansion: int = 4,
                 conv_expansion: int = 2, conv_kernel_size: int = 31,
                 dropout: float = 0.1, half_step_residual: bool = True):
        super().__init__()
        self.residual_factor = 0.5 if half_step_residual else 1.0
        self.ffn1 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.attn = MultiHeadedSelfAttentionModule(dim, num_heads, dropout)
        self.conv = ConvolutionModule(dim, conv_kernel_size, conv_expansion, dropout)
        self.ffn2 = FeedForwardModule(dim, ffn_expansion, dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.residual_factor * self.ffn1(x)
        x = x + self.attn(x, pos_emb, mask)
        x = x + self.conv(x, mask)
        x = x + self.residual_factor * self.ffn2(x)
        return self.layer_norm(x)


class ConformerEncoder(nn.Module):
    """Full encoder stack (paper Fig. 1).

    SpecAugment is applied in the data pipeline, not here.
    """

    def __init__(self, input_dim: int = 80, encoder_dim: int = 144, num_layers: int = 16,
                 num_heads: int = 4, ffn_expansion: int = 4, conv_expansion: int = 2,
                 conv_kernel_size: int = 31, dropout: float = 0.1,
                 half_step_residual: bool = True):
        super().__init__()
        self.subsample = Conv2dSubsampling(input_dim, encoder_dim)
        self.input_proj = nn.Linear(self.subsample.out_dim, encoder_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.pos_encoding = RelPositionalEncoding(encoder_dim)
        self.layers = nn.ModuleList([
            ConformerBlock(encoder_dim, num_heads, ffn_expansion, conv_expansion,
                           conv_kernel_size, dropout, half_step_residual)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """x: (B, T, input_dim) filterbanks; lengths: (B,) valid frame counts.

        Returns (B, T//4, encoder_dim) encodings and their subsampled lengths.
        """
        x, out_lengths = self.subsample(x, lengths)
        x = self.input_dropout(self.input_proj(x))

        mask = lengths_to_mask(out_lengths, x.size(1))
        pos_emb = self.pos_encoding(x.size(1)).to(x.dtype)

        for layer in self.layers:
            x = layer(x, pos_emb, mask)

        # Padded positions carry BatchNorm/LayerNorm bias; zero them so downstream
        # pooling or loss code never sees junk.
        return x.masked_fill(~mask.unsqueeze(-1), 0.0), out_lengths
