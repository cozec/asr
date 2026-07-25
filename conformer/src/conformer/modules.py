"""Conformer sub-modules, following Gulati et al. 2020 (arXiv:2005.08100).

Section references in the docstrings point at the paper. The four sub-modules of a
Conformer block live here; `encoder.py` assembles them into the macaron sandwich.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """x * sigmoid(x). Used in the FFN and convolution modules (paper Fig. 2, Fig. 4)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Conv2dSubsampling(nn.Module):
    """Convolution subsampling to 1/4 the input frame rate (paper Fig. 1).

    Two 3x3 convolutions with stride 2 take the 10ms-rate filterbank input down to a
    40ms rate, which is what the Conformer blocks operate on.
    """

    def __init__(self, in_dim: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        # Frequency axis shrinks the same way the time axis does.
        self.out_dim = out_channels * (((in_dim - 1) // 2 - 1) // 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """(B, T, F) -> (B, T//4, out_dim), with lengths subsampled to match."""
        x = self.conv(x.unsqueeze(1))                       # (B, C, T', F')
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        # Mirrors the conv arithmetic above: floor((n - 3)/2) + 1 == (n - 1)//2 for k=3,s=2.
        out_lengths = ((lengths - 1) // 2 - 1) // 2
        return x, out_lengths.clamp(min=1)


class FeedForwardModule(nn.Module):
    """Feed forward module (paper §2.3, Fig. 4).

    Pre-norm residual unit: LayerNorm -> Linear(d, 4d) -> Swish -> Dropout ->
    Linear(4d, d) -> Dropout. The residual add and its 1/2 factor are applied by the
    caller (`ConformerBlock`), per Eq. 1.
    """

    def __init__(self, dim: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d whose batch statistics ignore padded frames.

    The paper's convolution module uses BatchNorm (§2.2). With variable-length batches
    a plain BatchNorm averages over padding too, which drags the mean toward zero and
    makes a sequence's output depend on how much padding its batch-mates forced onto
    it. Restricting the statistics to valid frames keeps training-mode outputs
    padding-invariant; in eval mode this is exactly nn.BatchNorm1d.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, D, T); mask: (B, T) True on valid frames."""
        if mask is None or not self.training:
            return super().forward(x)

        m = mask.unsqueeze(1).to(x.dtype)                   # (B, 1, T)
        count = m.sum() * 1.0
        mean = (x * m).sum(dim=(0, 2)) / count
        centred = (x - mean.view(1, -1, 1)) * m
        var = (centred ** 2).sum(dim=(0, 2)) / count

        if self.track_running_stats:
            with torch.no_grad():
                # Running variance follows the nn.BatchNorm convention of the
                # unbiased estimate; the normalisation below uses the biased one.
                unbiased = var * count / (count - 1) if count > 1 else var
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased)
                self.num_batches_tracked += 1

        normalised = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
        if self.affine:
            normalised = normalised * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return normalised


class ConvolutionModule(nn.Module):
    """Convolution module (paper §2.2, Fig. 2).

    LayerNorm -> pointwise conv (expansion 2) -> GLU -> depthwise conv -> BatchNorm ->
    Swish -> pointwise conv -> Dropout.

    `mask` zeroes padded frames before the depthwise convolution so that padding cannot
    bleed into valid frames through the kernel window (and cannot skew the BatchNorm
    statistics).
    """

    def __init__(self, dim: int, kernel_size: int = 31, expansion_factor: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * expansion_factor, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=0)
        self.batch_norm = MaskedBatchNorm1d(dim)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        # "same" padding, split asymmetrically so even kernel sizes (the paper uses 32)
        # still preserve the sequence length.
        self.pad = (kernel_size // 2, (kernel_size - 1) // 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D); mask: (B, T) True on valid frames."""
        x = self.layer_norm(x).transpose(1, 2)              # (B, D, T)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(1), 0.0)
        x = F.glu(self.pointwise_conv1(x), dim=1)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(F.pad(x, self.pad))
        x = self.swish(self.batch_norm(x, mask))
        x = self.pointwise_conv2(x)
        return self.dropout(x.transpose(1, 2))


class RelPositionalEncoding(nn.Module):
    """Relative sinusoidal positional encoding, Transformer-XL style (paper §2.1).

    Produces embeddings for every relative offset in [T-1, ..., -(T-1)], i.e. length
    2T-1, ordered from the largest positive offset down to the largest negative one.
    That ordering is what makes the `_rel_shift` in `RelPositionMultiHeadAttention`
    line each score up with R_{i-j}, the Transformer-XL relative offset of key j as
    seen from query i.
    """

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        self.register_buffer("pe", self._build(max_len, dim), persistent=False)

    def _build(self, length: int, dim: int) -> torch.Tensor:
        pe_pos = torch.zeros(length, dim)
        pe_neg = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim)
        )
        pe_pos[:, 0::2] = torch.sin(position * div_term)
        pe_pos[:, 1::2] = torch.cos(position * div_term)
        pe_neg[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_neg[:, 1::2] = torch.cos(-1 * position * div_term)
        # [T-1 ... 0] then [-1 ... -(T-1)]  ->  (2T-1, dim)
        pe_pos = torch.flip(pe_pos, [0])
        return torch.cat([pe_pos, pe_neg[1:]], dim=0)

    def forward(self, length: int) -> torch.Tensor:
        if length * 2 - 1 > self.pe.size(0):
            self.pe = self._build(length, self.dim).to(self.pe.device, self.pe.dtype)
        center = self.pe.size(0) // 2
        return self.pe[center - length + 1: center + length].unsqueeze(0)


class RelPositionMultiHeadAttention(nn.Module):
    """Multi-head self-attention with relative position bias (paper §2.1).

    Implements the Transformer-XL decomposition: the attention logit is
    (q + u)·k  +  rel_shift((q + v)·pos), where u and v are the learned content and
    position bias vectors.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.u_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.v_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, 2T-1) -> (B, H, T, T), picking R_{i-j} for each query/key pair.

        Input column n holds the score against relative offset T-1-n. Left-padding by
        one and reinterpreting the strides shifts row i by i columns, so that output
        (i, j) lands on input column T-1+i-j, i.e. offset i-j. Verified directly by
        `tests/test_rel_shift.py`.
        """
        b, h, t, n = x.size()
        x = F.pad(x, (1, 0))                                # (B, H, T, 2T)
        x = x.view(b, h, n + 1, t)
        return x[:, :, 1:].view(b, h, t, n)[:, :, :, :t]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, D); pos_emb: (1, 2T-1, D); mask: (B, T) True on valid frames."""
        b, t, _ = x.size()

        q = self.query_proj(x).view(b, t, self.num_heads, self.head_dim)
        k = self.key_proj(x).view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value_proj(x).view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        p = self.pos_proj(pos_emb).view(1, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q_u = (q + self.u_bias).transpose(1, 2)             # (B, H, T, d)
        q_v = (q + self.v_bias).transpose(1, 2)

        content_score = torch.matmul(q_u, k.transpose(2, 3))            # (B, H, T, T)
        pos_score = self._rel_shift(torch.matmul(q_v, p.transpose(2, 3)))
        score = (content_score + pos_score) / self.scale

        if mask is not None:
            # Mask keys only; a fully-masked query row cannot occur because every
            # sequence has at least one valid frame.
            score = score.masked_fill(~mask[:, None, None, :], float("-inf"))

        attn = self.dropout(torch.softmax(score, dim=-1))
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.dim)
        return self.out_proj(context)


class MultiHeadedSelfAttentionModule(nn.Module):
    """Pre-norm residual wrapper around relative MHSA (paper §2.1, Fig. 3)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attention = RelPositionMultiHeadAttention(dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.dropout(self.attention(self.layer_norm(x), pos_emb, mask))
