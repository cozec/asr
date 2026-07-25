# RNN-T — findings & session state

Working notes for [README.md](README.md). Everything measured on this machine (Apple M5,
16 GB, macOS 25.5, Python 3.11, torch 2.13.0 / torchaudio 2.11.0).

Last updated: 2026-07-25.

---

## TL;DR

| | |
|---|---|
| **Streaming demo** | **Working.** 5.99% WER, RTF 0.138, p50 24 ms/chunk on dev-clean. |
| Our implementation | Done, 21.62M params, 18/18 tests passing. |
| Our model's training | Smoke-tested only (loss 975.9 → 775.0 over 15 steps). Not trained. |
| Improvement adopted | Stateless prediction network (Ghodsi et al. 2020). |

## 1. Prior-art search

| Source | Verdict |
|---|---|
| `torchaudio.models.RNNT` + `RNNTBeamSearch` | **Reused.** Still present in 2.11.0 (the Emformer tutorials only document up to 2.6, so this needed checking). `RNNT` is a container of `(transcriber, predictor, joiner)`, and `RNNTBeamSearch.infer(input, length, beam_width, state, hypothesis)` carries state across chunks — a tested streaming decoder we did not have to write. |
| `torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH` | **Reused** as the demo backbone. 16 kHz, 80 mels, 160 ms chunks, 40 ms right-context lookahead, ~300 MB download. |
| [k2-fsa/icefall](https://github.com/k2-fsa/icefall) | Reference only — source for the stateless predictor's shape (`nn.Embedding` + `nn.Conv1d`, `context_size=2`). Full k2 is awkward to build on Apple Silicon. |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa) | Reference only — production on-device streaming runtime, not a paper implementation. |
| `conformer/src/data/*` | Copied (fbank + SpecAugment, LibriSpeech dataset, tokenizer) to keep the folder self-contained. |

## 2. Paper facts (arXiv:1811.06621)

Architecture (§3.1, §6.2): 8 × LSTM(2048) + 640 projection encoder, layer norm on every
layer, time reduction **N=2 after layer 2** (1.7× speedup, no accuracy loss); prediction
network 2 × LSTM(2048) + 640 proj; joint network 640 units; 4,096 wordpieces or <100
graphemes; **117M params (graphemes) / 120M (wordpieces)**. Features: 80-dim log-mel,
25 ms window / 10 ms shift, stacked 3 frames and downsampled to a **30 ms** frame rate.

Training data: **35M utterances (~27,500 h)** of anonymized Google voice-search and
dictation traffic, on 8×8 TPU slices, batch 4,096. Not reproducible.

Results (Table 1, VS/IME WER): grapheme 8.1/4.9 → +layer norm 7.6/4.6 → +larger batch
7.5/4.4 → **+wordpiece 6.8/4.0**; CTC baseline 9.3/5.3.
Table 4 (RT90): float wordpiece 1.43 → asymmetric int8 1.03 → **symmetric int8 0.51**,
model size 4× smaller.

## 3. The improvement: stateless prediction network

[Ghodsi et al. 2020](https://storage.googleapis.com/gweb-research2023-media/pubtools/5775.pdf)
replaces the recurrent predictor with an embedding over the last symbol(s) — "effectively
a 2-gram LM on the output subword set." Their conclusion: the predictor "does not
function as the LM in classical ASR. Instead, it merely helps the model align to the
input audio." Table 2 supports this — tuning the encoder alone (17 WER) beats tuning the
predictor + joint (24).

**Two things this changes for us:**

1. **It moots the paper's §3.3 state caching.** That optimization caches prediction-network
   LSTM states across decoding steps (the paper reports saving 50–60% of predictor
   compute). With a stateless predictor there is no recurrent state to cache.
2. **It constrains the vocabulary.** §3.4: stateless prediction "causes significant
   regressions for the grapheme models" — seeing one symbol back, the model cannot tell
   whether it has already emitted one `o` or two in `food`, so it matches `foo*d`. With
   wordpieces there is "little or no WER regression."

So **stateless + wordpiece is the combination that works.** Had we chosen characters, the
two decisions would have actively conflicted. Worth remembering if the vocabulary is ever
revisited.

## 4. Measured results

**Streaming demo** (pretrained Emformer RNN-T, 25 dev-clean utterances, 206 s audio,
decoded chunk-by-chunk through the same backend the live demo uses):

| | |
|---|---|
| WER | **5.99%** |
| RTF | **0.138** (7× faster than real time) |
| Per-chunk latency | p50 24 ms, p90 29 ms, for 160 ms chunks |

First utterance is an exact match to the reference transcript.

**Our model** (`--model ours`): runs end-to-end through the identical loop at RTF 0.025,
p50 11 ms per 480 ms chunk. Transcript is empty because the checkpoint has had 15
training steps — an RNN-T that has only learned "blank is most likely" is exactly what
that produces. The code path is verified; the model is not trained.

## 5. Verification

18/18 assertions in `tests/test_rnnt.py`. The load-bearing ones are the three streaming
equivalences, all **exact** (0.00e+00 / 2.4e-07):

| Test | What it protects |
|---|---|
| `test_streaming_features_match_offline` | Frontend carries a waveform tail (25 ms windows overlap chunk edges) *and* a frame tail (3-frame stacking spans boundaries). Either one missing corrupts features at every chunk boundary. |
| `test_transcriber_streaming_matches_offline` | Per-layer LSTM `(h, c)` carried correctly across `infer` calls. |
| `test_predictor_streaming_matches_offline` | Token-history state equals a full-sequence pass. |

Plus encoder causality, time-reduction arithmetic, stateless-predictor context bounds,
and overfit-a-batch (RNN-T loss 63.2 → 10.1 in 30 steps).

## 6. Gotchas hit (and fixed)

- **`_Predictor` and `_Joiner` are not interfaces.** Despite the naming, only
  `_Transcriber` is an ABC in torchaudio; the other two are *concrete* LSTM/joiner
  implementations, so inheriting them drags in their constructors. `RNNT` duck-types its
  components, so plain `nn.Module`s work.
- **`output_dim` bug when time reduction lands last.** If `time_reduction_after ==
  num_layers`, the encoder's output width doubles. The initial code hard-coded
  `proj_dim`. Caught by the overfit test, fixed by tracking the running dim.
- **Closures can't be DataLoader collate functions on macOS.** Workers are spawned, not
  forked, so `rnnt_collate` had to become a picklable class. Conformer never hit this
  because its default CTC collate is a module-level function.
- **`torchaudio.load` needs torchcodec in 2.11.** Used `soundfile` instead, consistent
  with the rest of the project.
- **RNN-T loss has no MPS kernel** (same as CTC). Computed on CPU; model stays on GPU.

## 7. Resume

State: demo working, tokenizer built on train-clean-100, manifests cached, one throwaway
15-step checkpoint in `exp/rnnt_small/`. Nothing running.

```bash
cd /Users/adam/interviews/asr/rnnt
.venv/bin/python tests/test_rnnt.py                      # 18 assertions
.venv/bin/python scripts/eval_streaming.py --num 25      # WER + RTF
.venv/bin/python src/stream_demo.py --source mic         # live demo
```

Microphone confirmed present ("MacBook Pro Microphone"); macOS will prompt for permission
on first mic use. The `--source file` path exercises everything except capture.

### If we keep going

- **Train our model properly** — the obvious next step. Expect throughput similar to the
  conformer project (~35 min/epoch on 100 h); note one stall during the smoke run
  (step 9→12 jumped 57 s → 278 s), consistent with the swap pressure documented in
  `conformer/FINDINGS.md`.
- **int8 quantization (§3.4)** — the paper's headline on-device result (RT90 1.43 → 0.51,
  4× smaller). `torch.ao.quantization.quantize_dynamic` on the LSTM layers is the natural
  local analogue, and would make the demo's RTF directly comparable to Table 4.
- **Endpointing / VAD** — the demo currently streams until Ctrl-C; the paper cares about
  end-of-utterance latency.
- **FastEmit** — reduces emission delay, which is what makes a live demo feel responsive.
