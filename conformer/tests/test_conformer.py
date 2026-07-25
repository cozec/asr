"""Correctness tests for the Conformer implementation.

    python tests/test_conformer.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from conformer import ConformerCTC, ConformerTransducer
from conformer.encoder import lengths_to_mask
from conformer.modules import (
    ConvolutionModule,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def test_rel_shift():
    """_rel_shift must map output (i, j) onto relative offset i - j."""
    b, h, t = 1, 1, 5
    n = 2 * t - 1
    # Column n encodes offset t-1-n; store the offset itself as the value.
    offsets = torch.tensor([t - 1 - k for k in range(n)], dtype=torch.float32)
    x = offsets.view(1, 1, 1, n).expand(b, h, t, n).contiguous()

    shifted = RelPositionMultiHeadAttention._rel_shift(x)
    expected = torch.tensor([[i - j for j in range(t)] for i in range(t)],
                            dtype=torch.float32).view(1, 1, t, t)
    check("rel_shift selects offset i-j", torch.equal(shifted, expected),
          f"got\n{shifted[0, 0].int().tolist()}")


def test_positional_encoding_shape():
    pe = RelPositionalEncoding(16)
    emb = pe(7)
    check("rel pos encoding is length 2T-1", emb.shape == (1, 13, 16), str(tuple(emb.shape)))
    # Centre element is offset 0: sin(0)=0, cos(0)=1.
    centre = emb[0, 6]
    check("centre embedding is offset zero",
          torch.allclose(centre[0::2], torch.zeros(8), atol=1e-6)
          and torch.allclose(centre[1::2], torch.ones(8), atol=1e-6))


def test_padding_invariance():
    """The core guarantee: extra padding must not change results for valid frames.

    This is what masking in attention and the convolution module buys us; without it
    the depthwise conv and softmax leak padded positions into real ones.
    """
    torch.manual_seed(0)
    model = ConformerCTC(num_classes=32, input_dim=80, encoder_dim=64, num_layers=2,
                         num_heads=4, conv_kernel_size=15, dropout=0.0).eval()

    torch.manual_seed(1)
    short = torch.randn(1, 100, 80)
    lengths = torch.tensor([100])

    with torch.no_grad():
        out_a, len_a = model(short, lengths)
        padded = torch.cat([short, torch.randn(1, 60, 80) * 5], dim=1)
        out_b, len_b = model(padded, lengths)

    valid = int(len_a[0])
    diff = (out_a[0, :valid] - out_b[0, :valid]).abs().max().item()
    check("output lengths ignore padding", int(len_b[0]) == valid)
    check("encoder output is padding-invariant", diff < 1e-4, f"max diff {diff:.2e}")


def test_batch_invariance():
    """An utterance must decode the same alone as inside a mixed-length batch."""
    torch.manual_seed(0)
    model = ConformerCTC(num_classes=32, input_dim=80, encoder_dim=64, num_layers=2,
                         num_heads=4, conv_kernel_size=15, dropout=0.0).eval()

    torch.manual_seed(2)
    a = torch.randn(120, 80)
    b = torch.randn(60, 80)
    batch = torch.zeros(2, 120, 80)
    batch[0, :120], batch[1, :60] = a, b

    with torch.no_grad():
        out_batch, len_batch = model(batch, torch.tensor([120, 60]))
        out_solo, len_solo = model(b.unsqueeze(0), torch.tensor([60]))

    valid = int(len_solo[0])
    diff = (out_batch[1, :valid] - out_solo[0, :valid]).abs().max().item()
    check("batching does not change results", diff < 1e-4, f"max diff {diff:.2e}")


def test_conv_module_causality_of_padding():
    """Zeroed padding must not shift the convolution output on valid frames."""
    torch.manual_seed(0)
    conv = ConvolutionModule(dim=32, kernel_size=32, dropout=0.0).eval()
    x = torch.randn(1, 50, 32)
    mask = lengths_to_mask(torch.tensor([50]), 50)

    with torch.no_grad():
        out_a = conv(x, mask)
        x_pad = torch.cat([x, torch.randn(1, 20, 32) * 10], dim=1)
        mask_pad = lengths_to_mask(torch.tensor([50]), 70)
        out_b = conv(x_pad, mask_pad)

    diff = (out_a - out_b[:, :50]).abs().max().item()
    check("even kernel (32) preserves length", out_a.shape == x.shape, str(tuple(out_a.shape)))
    check("conv module masks padding", diff < 1e-5, f"max diff {diff:.2e}")


def test_conv_module_padding_invariance_in_train_mode():
    """Training-mode BatchNorm must take its statistics from valid frames only.

    A plain nn.BatchNorm1d averages over the padded region too, so the same utterance
    produces different activations depending on its batch-mates' lengths.
    """
    torch.manual_seed(0)
    conv = ConvolutionModule(dim=32, kernel_size=32, dropout=0.0).train()
    x = torch.randn(1, 50, 32)

    out_a = conv(x, lengths_to_mask(torch.tensor([50]), 50))
    x_pad = torch.cat([x, torch.randn(1, 150, 32) * 3], dim=1)
    out_b = conv(x_pad, lengths_to_mask(torch.tensor([50]), 200))

    diff = (out_a - out_b[:, :50]).abs().max().item()
    check("train-mode BatchNorm ignores padding", diff < 1e-4, f"max diff {diff:.2e}")


def test_masked_batchnorm_matches_unpadded_reference():
    """With no padding, MaskedBatchNorm1d must equal nn.BatchNorm1d exactly."""
    from conformer.modules import MaskedBatchNorm1d

    torch.manual_seed(0)
    x = torch.randn(4, 16, 30)
    masked = MaskedBatchNorm1d(16).train()
    plain = torch.nn.BatchNorm1d(16).train()

    out_masked = masked(x, lengths_to_mask(torch.full((4,), 30), 30))
    out_plain = plain(x)
    diff = (out_masked - out_plain).abs().max().item()
    check("masked BN == nn.BatchNorm1d when nothing is padded", diff < 1e-5,
          f"max diff {diff:.2e}")
    stat_diff = max((masked.running_mean - plain.running_mean).abs().max().item(),
                    (masked.running_var - plain.running_var).abs().max().item())
    check("masked BN running stats match too", stat_diff < 1e-5, f"max diff {stat_diff:.2e}")


def test_subsampling_rate():
    model = ConformerCTC(num_classes=32, encoder_dim=32, num_layers=1, num_heads=4).eval()
    with torch.no_grad():
        out, lengths = model(torch.randn(2, 400, 80), torch.tensor([400, 200]))
    # Two stride-2 convs -> roughly 1/4 the frames (40ms from a 10ms rate).
    check("subsampling is ~4x", abs(int(lengths[0]) - 100) <= 2 and out.size(1) == int(lengths[0]),
          f"400 -> {int(lengths[0])}, 200 -> {int(lengths[1])}")


def test_ctc_loss_decreases():
    """The model should be able to overfit a single batch."""
    torch.manual_seed(0)
    model = ConformerCTC(num_classes=20, input_dim=80, encoder_dim=64, num_layers=2,
                         num_heads=4, conv_kernel_size=15, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    feats = torch.randn(2, 200, 80)
    feat_lens = torch.tensor([200, 200])
    targets = torch.randint(1, 20, (2, 8))
    target_lens = torch.tensor([8, 8])

    losses = []
    for _ in range(30):
        log_probs, out_lens = model(feats, feat_lens)
        loss = loss_fn(log_probs.transpose(0, 1), targets, out_lens, target_lens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    check("CTC loss decreases when overfitting a batch", losses[-1] < losses[0] * 0.6,
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")


def test_transducer_shapes():
    model = ConformerTransducer(num_classes=32, encoder_dim=64, num_layers=2,
                                num_heads=4, decoder_dim=64, joint_dim=64).eval()
    feats, feat_lens = torch.randn(2, 200, 80), torch.tensor([200, 160])
    targets, target_lens = torch.randint(1, 32, (2, 7)), torch.tensor([7, 5])
    sos = torch.zeros(2, 1, dtype=torch.long)
    with torch.no_grad():
        logits, out_lens = model(feats, feat_lens, torch.cat([sos, targets], 1), target_lens)
    check("transducer joint output is (B, T, U+1, V)",
          logits.shape == (2, int(out_lens[0]), 8, 32), str(tuple(logits.shape)))


def test_gradients_flow():
    model = ConformerCTC(num_classes=20, encoder_dim=32, num_layers=2, num_heads=4)
    log_probs, out_lens = model(torch.randn(2, 150, 80), torch.tensor([150, 120]))
    log_probs.sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    check("every parameter receives a gradient", not missing, f"missing: {missing[:5]}")


if __name__ == "__main__":
    print("Conformer implementation tests\n")
    for fn in [test_rel_shift, test_positional_encoding_shape, test_subsampling_rate,
               test_conv_module_causality_of_padding,
               test_conv_module_padding_invariance_in_train_mode,
               test_masked_batchnorm_matches_unpadded_reference,
               test_padding_invariance, test_batch_invariance, test_transducer_shapes,
               test_gradients_flow, test_ctc_loss_decreases]:
        print(f"{fn.__name__}:")
        fn()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)
