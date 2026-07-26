# wav2vec2 — findings & session state

Working notes for [README.md](README.md). All numbers measured on this machine (Apple M5,
16 GB, macOS 25.5, Python 3.11, torch 2.13.0 / torchaudio 2.11.0, **CPU**).

Last updated: 2026-07-25.

---

## TL;DR

| | |
|---|---|
| Tutorial reproduction | **Exact** — `I\|HAD\|THAT\|CURIOSITY\|BESIDE\|ME\|AT\|THIS\|MOMENT\|` |
| Tests | 10/10 passing |
| Best WER | **1.64%** (beam 50 + 4-gram LM), vs 2.15% greedy |
| Headline finding | Beam search alone buys **nothing**; the LM is where the gain is |

## 1. What the tutorial does

Five steps, all reproduced in `src/pipeline_demo.py`: load the bundle (weights + sample
rate + 29 labels); load and resample audio; `model.extract_features()` → 12 per-layer
tensors; `model()` → emission logits `(1, 169, 29)` at 49.7 frames/s (wav2vec2's 20 ms
stride); greedy CTC decode.

The model is `WAV2VEC2_ASR_BASE_960H`, 94.4M params: self-supervised pretraining on 960 h
of *unlabeled* LibriSpeech, then fine-tuned with CTC on 960 h of *labeled* LibriSpeech.

## 2. Decoder comparison (the interesting part)

50 utterances, test-clean, `WAV2VEC2_ASR_BASE_960H`. Acoustic model run once; every
decoder re-scores the *same* emissions, so differences are purely decoding.

| decoder | lexicon | LM | WER% | decode s | ×greedy |
|---|---|---|---|---|---|
| greedy | – | – | 2.15 | 0.01 | 1× |
| beam 5 | no | no | 2.15 | 0.10 | 8× |
| beam 50 | no | no | 2.15 | 1.07 | 87× |
| beam 5 | yes | 4-gram | 2.46 | 0.10 | 8× |
| beam 50 | yes | 4-gram | **1.64** | 0.88 | 72× |

**Beam search without an LM is exactly equal to greedy** — not close, equal, at both beam
5 and beam 50. The reason is structural: CTC models outputs as conditionally independent
given the audio, so the highest-probability path is the concatenation of per-frame
argmaxes. Beam search explores alternative *alignments* of the same acoustic scores,
which collapse to the same output string. Without outside information there is nothing
for the wider search to find. Anyone reaching for beam search to improve a CTC model,
without also adding an LM, is buying an 87× slowdown for nothing.

**The LM is the whole gain**: 2.15 → 1.64 (24% relative) at beam 50.

**Beam width and LM must be added together.** Beam 5 + LM is *worse than greedy* (2.46 vs
2.15). The LM reorders hypotheses, so the beam must be wide enough to still contain the
one the LM will promote. A narrow beam prunes it away before the LM can score it.

Acoustic model cost for context: RTF 0.015. Even the 72× decoder is a small share of
total runtime, so "87× slower decoding" is less alarming than it sounds — but it is also
buying nothing without the LM.

## 3. Labeled-data efficiency

Same 94M params, same self-supervised pretraining; only the labeled fine-tuning set
differs. 30 utterances, test-clean, greedy.

| labeled | WER% | RTF |
|---|---|---|
| 10 min | 44.41 | 0.014 |
| 100 h | 6.06 | 0.013 |
| 960 h | 2.33 | 0.013 |

The qualitative failure is the point — with 10 minutes of labels the model writes
`TERNEIPS` for `TURNIPS`, `PATATOWS` for `POTATOES`, `BROSED` for `BRUISED`. Phonetically
right, orthographically wrong: the self-supervised representation already encodes the
sounds, and labeled data is buying spelling. That is wav2vec 2.0's thesis, visible in
three transcripts.

Note the 960 h number differs slightly between experiments (2.33% on 30 utterances here,
2.15% on 50 in §2) — small-sample variation, not a discrepancy. Neither is a full
test-clean run (2620 utterances).

## 4. Gotchas hit

- **`torchaudio.utils.download_asset` is gone in 2.11.** The tutorial depends on it. The
  underlying URL (`download.pytorch.org/torchaudio/tutorial-assets/...`) still serves, so
  `pipeline_demo.download_asset` fetches it with `requests`.
- **`torchaudio.load` requires torchcodec in 2.11** — used `soundfile`, as in the other
  projects here.
- **torch.hub's downloader stalls.** Repeatedly died partway through 360 MB checkpoints
  (0% CPU, frozen `.partial` files), and left duplicate partials when interrupted. curl
  fetched the same file at 28 MB/s in 12 s. If a run appears hung during model download:

  ```bash
  pkill -f <script>; rm -f ~/.cache/torch/hub/checkpoints/*.partial
  curl -fL --retry 5 --retry-all-errors -o ~/.cache/torch/hub/checkpoints/<name>.pth \
       https://download.pytorch.org/torchaudio/models/<name>.pth
  ```
  Pre-fetching all needed checkpoints this way makes the experiments reliable.
- **`ctc_decoder` needs `flashlight-text`**, which is not a torchaudio dependency. It
  installs cleanly from a wheel on Apple Silicon (no build).
- **A test of mine was wrong, not the code.** The greedy-decoder test expected `HE|TO`
  while indexing label 4, which is `A`, not `O`. Fixed the expectation. The decoder was
  independently confirmed correct by the exact tutorial-transcript match.

## 5. Resume

```bash
cd /Users/adam/interviews/asr/wav2vec2_asr
.venv/bin/python tests/test_wav2vec2.py                        # 10 assertions
.venv/bin/python src/pipeline_demo.py                          # tutorial + plots
.venv/bin/python scripts/compare_decoders.py --num 50 --with-lm
.venv/bin/python scripts/evaluate.py --num 30 --bundles WAV2VEC2_ASR_BASE_10M WAV2VEC2_ASR_BASE_100H WAV2VEC2_ASR_BASE_960H
```

Cached: all three BASE ASR checkpoints (360 MB each), the librispeech-4-gram LM (~3 GB),
and the tutorial audio in `data/`.

### If we keep going

- **Full test-clean / test-other** (2620 / 2939 utterances) for numbers comparable to the
  paper, rather than 30-50 utterance samples.
- **Tune `lm_weight` / `word_score`** — currently torchaudio's tutorial defaults (3.23,
  -0.26), not tuned on a dev set.
- **Sweep beam width** between 5 and 50 to locate where LM beam search overtakes greedy.
- **LARGE / LV60K bundles** — same script, larger models, notably better WER.
