# RNN-T — Streaming End-to-end Speech Recognition For Mobile Devices

PyTorch implementation of **[Streaming End-to-end Speech Recognition For Mobile
Devices](https://arxiv.org/abs/1811.06621)** (He et al., Google, ICASSP 2019) — the
on-device RNN-T that runs faster than real time on a Pixel — plus a **live streaming
demo** that runs on this MacBook.

## The demo

```bash
# on a file (deterministic, no microphone needed)
python src/stream_demo.py --source file \
    --audio ../data/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac

# live microphone
python src/stream_demo.py --source mic
```

Text appears word by word as audio streams in:

```
> mister quilter is the
> mister quilter is the apostle of the
> mister quilter is the apostle of the middle classes and we are glad to welcome his gospel

audio 5.9s | compute 0.9s | RTF 0.154
per-chunk latency: p50 24 ms, p90 29 ms (chunk is 160 ms of audio)
```

Measured over 25 dev-clean utterances, decoded chunk-by-chunk:

| | |
|---|---|
| **WER (streaming)** | **5.99%** |
| **RTF** | **0.138** — 7× faster than real time |
| Chunk latency | p50 24 ms per 160 ms chunk |

For comparison, the paper's Table 4 reports RT90 of 1.43 (float) and 0.51 (quantized) on
a Pixel. A laptop is not a phone, so this is not a like-for-like number — but it is
comfortably real-time, which is what the demo needs.

```bash
python scripts/eval_streaming.py --num 25       # reproduces the table above
```

## Two models, one streaming loop

The paper's model was trained on **35M utterances (~27,500 h)** of Google-internal voice
search on 8×8 TPUs, at ~120M params. That is not reproducible here, so the project splits
in two, meeting at one interface:

| | Backend | Status |
|---|---|---|
| **Demo today** | `torchaudio` pretrained Emformer RNN-T (LibriSpeech 960 h) | Working, numbers above |
| **Our implementation** | The paper's architecture, trainable locally | Implemented + tested; training is the long pole |

Both satisfy torchaudio's `RNNT` component interfaces, so
`torchaudio.models.RNNTBeamSearch` — including its streaming `infer(..., state,
hypothesis)` path — drives either one unmodified. Swapping models is a flag:

```bash
python src/stream_demo.py --source mic --model ours --checkpoint exp/rnnt_small/epoch9.pt
```

## Architecture

| | Paper (§3.1, §6.2) | Ours | Why |
|---|---|---|---|
| Encoder | 8 × LSTM(2048) + 640 proj | 6 × LSTM(768) + 256 proj | ~120M → 21.6M, trainable locally |
| Per-layer projection | yes | yes | keeps recurrent connections cheap |
| Layer norm | every LSTM layer | yes | paper Table 1: 6% relative WER |
| Time reduction | N=2 after layer 2 | same | paper: 1.7× speedup, no accuracy loss |
| Predictor | 2 × LSTM(2048) + 640 proj | **stateless** (embedding + Conv1d) | Ghodsi et al. 2020 |
| Joint | feed-forward, 640 | 256 | matched to encoder |
| Output | 4,096 wordpieces | 1k wordpieces | 960 h corpus, not 27,500 h |
| Features | 80 log-mel, 25/10 ms, ×3 stack → 30 ms | same | paper §6.2 |

Encoder output frame rate is 30 ms × 2 = **60 ms**. The paper notes this is fine for
RNN-T while it degrades CTC phoneme models.

### The improvement we adopted: stateless prediction network

[Ghodsi et al. 2020](https://storage.googleapis.com/gweb-research2023-media/pubtools/5775.pdf)
replaces the paper's 2×LSTM predictor with an embedding over the last symbol(s) — "a
2-gram LM on the output subword set" — at comparable WER. Their finding is that the
predictor "does not function as the LM in classical ASR"; it "merely helps the model
align to the input audio."

Two consequences:

- It removes recurrent state from the decode loop, which makes the paper's §3.3
  prediction-network **state caching unnecessary** — there is nothing left to cache.
- **It only works with wordpieces.** §3.4 reports significant regressions on graphemes:
  seeing one symbol back, the model cannot tell whether it already emitted one `o` or
  two in `food`, so it matches `foo*d`. This is why the vocabulary is wordpieces —
  see [FINDINGS.md](FINDINGS.md).

## Verification

`python tests/test_rnnt.py` — 18 assertions, all passing. A streaming model is only
correct if chunked inference equals whole-utterance inference, so three tests assert
exactly that, and all three match **exactly** (0.00e+00):

- `test_streaming_features_match_offline` — the frontend carries a waveform tail (25 ms
  windows overlap chunks) and a frame tail (stacking spans boundaries)
- `test_transcriber_streaming_matches_offline` — chunked `infer` with carried LSTM state
  equals one full-context `forward`
- `test_predictor_streaming_matches_offline` — token-by-token equals full-sequence

Plus: encoder causality (future audio cannot change past outputs), time-reduction frame
arithmetic, stateless-predictor context bounds, and an overfit-a-batch check that the
RNN-T loss actually trains (63.2 → 10.1 in 30 steps).

## Training

```bash
python scripts/train_tokenizer.py --config configs/rnnt_small.yaml   # 1k wordpieces
python src/train.py --config configs/rnnt_small.yaml
```

Adam (β 0.9/0.98, ε 1e-9), transformer LR schedule, warmup 2,000 steps (scaled for
100 h, per the conformer project's finding that the paper's 10k warmup is calibrated for
a much larger corpus). RNN-T loss has no MPS kernel, so it is computed on CPU while the
model stays on the GPU — handled automatically.

## Layout

```
src/rnnt/       transcriber.py (LSTM stack + time reduction), model.py (stateless
                predictor, joiner, RNNT assembly)
src/            stream_features.py (streaming-safe frontend), stream_demo.py (the demo),
                train.py
src/data/       fbank + SpecAugment, LibriSpeech dataset, tokenizer (from conformer/)
scripts/        train_tokenizer.py, eval_streaming.py
tests/          test_rnnt.py — 18 assertions
```

## Deliberately out of scope

int8 quantization (§3.4), FastEmit, pruned RNN-T loss, contextual-biasing FST (§4), and
TTS-based text normalization (§5). All are real parts of the paper or its successors;
none are needed for the streaming demo, and each is noted rather than silently dropped.
