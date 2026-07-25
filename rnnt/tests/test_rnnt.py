"""Correctness tests for the RNN-T implementation.

    python tests/test_rnnt.py

The two that matter most are the streaming-equivalence tests: a streaming model is only
correct if feeding audio in chunks gives the same answer as feeding it all at once.
"""

import os
import sys

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rnnt import LSTMTranscriber, StatelessPredictor, TimeReduction, build_rnnt
from stream_features import StreamingFeatureExtractor, offline_features

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def load_cfg():
    with open(os.path.join(ROOT, "configs", "rnnt_small.yaml")) as fh:
        return yaml.safe_load(fh)


def test_time_reduction():
    tr = TimeReduction(2)
    x = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
    out, lengths = tr(x, torch.tensor([8, 6]))
    check("time reduction halves the frame rate", out.shape == (2, 4, 6), str(tuple(out.shape)))
    check("time reduction concatenates adjacent frames",
          torch.equal(out[0, 0], torch.cat([x[0, 0], x[0, 1]])))
    check("time reduction scales lengths", lengths.tolist() == [4, 3], str(lengths.tolist()))


def test_transcriber_shapes_and_causality():
    enc = LSTMTranscriber(input_dim=240, num_layers=4, hidden_dim=128, proj_dim=64,
                          dropout=0.0).eval()
    x = torch.randn(1, 40, 240)
    with torch.no_grad():
        out_a, len_a = enc(x, torch.tensor([40]))
        # Appending future audio must not change earlier outputs: the LSTM is uni-directional.
        out_b, _ = enc(torch.cat([x, torch.randn(1, 20, 240)], 1), torch.tensor([60]))
    check("encoder subsamples by the reduction factor", int(len_a[0]) == 20, str(int(len_a[0])))
    diff = (out_a - out_b[:, :out_a.size(1)]).abs().max().item()
    check("encoder is causal (future audio cannot change the past)", diff < 1e-5,
          f"max diff {diff:.2e}")


def test_transcriber_streaming_matches_offline():
    """Chunked `infer` with carried state must equal one full-context `forward`."""
    enc = LSTMTranscriber(input_dim=240, num_layers=4, hidden_dim=128, proj_dim=64,
                          dropout=0.0).eval()
    x = torch.randn(1, 48, 240)
    with torch.no_grad():
        full, _ = enc(x, torch.tensor([48]))
        chunks, states = [], None
        for i in range(0, 48, 8):                      # 8 is a multiple of the factor 2
            piece = x[:, i:i + 8]
            out, _, states = enc.infer(piece, torch.tensor([piece.size(1)]), states)
            chunks.append(out)
        streamed = torch.cat(chunks, dim=1)
    check("streamed encoder output has the same shape", streamed.shape == full.shape,
          f"{tuple(streamed.shape)} vs {tuple(full.shape)}")
    diff = (streamed - full).abs().max().item()
    check("streaming encoder == offline encoder", diff < 1e-5, f"max diff {diff:.2e}")


def test_stateless_predictor_context():
    """Output must depend only on the last `context_size` symbols (Ghodsi et al. 2020)."""
    pred = StatelessPredictor(num_symbols=100, output_dim=32, symbol_embedding_dim=32,
                              context_size=2, dropout=0.0).eval()
    a = torch.tensor([[5, 6, 7, 8, 9]])
    b = torch.tensor([[1, 2, 3, 8, 9]])               # differs only in distant history
    with torch.no_grad():
        out_a, _, _ = pred(a, torch.tensor([5]))
        out_b, _, _ = pred(b, torch.tensor([5]))
    last = (out_a[:, -1] - out_b[:, -1]).abs().max().item()
    early = (out_a[:, 0] - out_b[:, 0]).abs().max().item()
    check("predictor ignores history beyond context_size", last < 1e-6, f"diff {last:.2e}")
    check("predictor does depend on recent symbols", early > 1e-6, f"diff {early:.2e}")


def test_predictor_streaming_matches_offline():
    """Token-by-token prediction with carried state must equal a full-sequence pass."""
    pred = StatelessPredictor(num_symbols=100, output_dim=32, symbol_embedding_dim=32,
                              context_size=2, dropout=0.0).eval()
    tokens = torch.tensor([[4, 11, 27, 3, 9, 15]])
    with torch.no_grad():
        full, _, _ = pred(tokens, torch.tensor([tokens.size(1)]))
        outs, state = [], None
        for i in range(tokens.size(1)):
            out, _, state = pred(tokens[:, i:i + 1], torch.tensor([1]), state)
            outs.append(out)
        streamed = torch.cat(outs, dim=1)
    diff = (streamed - full).abs().max().item()
    check("streaming predictor == offline predictor", diff < 1e-6, f"max diff {diff:.2e}")


def test_streaming_features_match_offline():
    """The frontend must produce identical features chunked or whole (paper §6.2)."""
    torch.manual_seed(0)
    wave = torch.randn(16000)                          # 1 s at 16 kHz
    offline = offline_features(wave)

    extractor = StreamingFeatureExtractor()
    pieces = [extractor(wave[i:i + 1600]) for i in range(0, 16000, 1600)]
    streamed = torch.cat([p for p in pieces if p.numel()], dim=0)

    n = min(streamed.size(0), offline.size(0))
    check("streaming frontend yields the same frame count", abs(streamed.size(0) - offline.size(0)) <= 1,
          f"{streamed.size(0)} vs {offline.size(0)}")
    diff = (streamed[:n] - offline[:n]).abs().max().item()
    check("streaming features == offline features", diff < 1e-3, f"max diff {diff:.2e}")


def test_feature_frame_rate():
    """80 mels x 3 stacked frames at a 30 ms rate (paper §6.2)."""
    feats = offline_features(torch.randn(16000))
    check("stacked feature dim is 80*3", feats.size(1) == 240, str(feats.size(1)))
    # 1 s at a 30 ms frame rate -> ~33 frames
    check("frame rate is ~30 ms", 31 <= feats.size(0) <= 34, f"{feats.size(0)} frames for 1 s")


def test_model_assembles_and_decodes():
    from torchaudio.models import RNNTBeamSearch

    model = build_rnnt(load_cfg(), 1025).eval()
    params = sum(p.numel() for p in model.parameters())
    x, xl = torch.randn(2, 60, 240), torch.tensor([60, 48])
    targets, tl = torch.randint(1, 1025, (2, 7)), torch.tensor([7, 5])
    out, src_len, _, _ = model(x, xl, targets, tl)
    check("RNNT joint output is (B, T, U, V)", out.shape == (2, 30, 7, 1025), str(tuple(out.shape)))
    check("model is a trainable size", params < 30e6, f"{params/1e6:.2f}M params")

    with torch.no_grad():
        hyps, state = RNNTBeamSearch(model, blank=0).infer(
            torch.randn(1, 32, 240), torch.tensor([32]), beam_width=4)
    check("torchaudio RNNTBeamSearch drives our model", len(hyps) > 0 and state is not None)


def test_rnnt_loss_decreases():
    """Overfit a single batch to confirm gradients actually train the model."""
    import torchaudio

    torch.manual_seed(0)
    cfg = load_cfg()
    cfg["model"].update(num_layers=2, hidden_dim=128, proj_dim=64, joint_dim=64,
                        symbol_embedding_dim=64, dropout=0.0)
    model = build_rnnt(cfg, 20)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x, xl = torch.randn(2, 40, 240), torch.tensor([40, 40])
    targets = torch.randint(1, 20, (2, 5))
    tl = torch.tensor([5, 5])
    sos = torch.zeros(2, 1, dtype=torch.long)

    losses = []
    for _ in range(30):
        logits, src_len, _, _ = model(x, xl, torch.cat([sos, targets], 1),
                                      tl + 1)
        loss = torchaudio.functional.rnnt_loss(
            logits=logits.float(), targets=targets.int(), logit_lengths=src_len.int(),
            target_lengths=tl.int(), blank=0, reduction="mean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    check("RNN-T loss decreases when overfitting a batch", losses[-1] < losses[0] * 0.6,
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")


if __name__ == "__main__":
    print("RNN-T implementation tests\n")
    for fn in [test_time_reduction, test_feature_frame_rate,
               test_streaming_features_match_offline,
               test_transcriber_shapes_and_causality,
               test_transcriber_streaming_matches_offline,
               test_stateless_predictor_context,
               test_predictor_streaming_matches_offline,
               test_model_assembles_and_decodes, test_rnnt_loss_decreases]:
        print(f"{fn.__name__}:")
        fn()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)
