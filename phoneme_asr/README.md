# Phoneme-based ASR for edge devices — landscape survey

Prior-art search before building anything: what exists for **phoneme-level speech
recognition** that is small enough to run **on-device**. Stars and dates are from the
GitHub API on 2026-07-26; model sizes marked *(measured)* were checked locally rather
than quoted.

## Why phonemes for edge, specifically

Three properties make phone units attractive when the compute budget is small, and they
compound:

1. **The output vocabulary collapses.** ~40 phones (English) or ~120 (IPA, multilingual)
   against 1k–10k word pieces. That shrinks the softmax layer, and more importantly the
   **decoding graph** — the WFST is the memory bottleneck on a microcontroller, not the
   acoustic model.
2. **One model covers many languages.** Phones are shared across languages in a way word
   pieces are not, so a single small model generalizes instead of shipping one per
   language.
3. **Downstream tasks often want phones anyway** — forced alignment, pronunciation
   scoring, keyword spotting, lip sync. Going phones→words is a separate, cheap step.

The cost: you need a pronunciation lexicon or a grapheme-to-phoneme step to recover
words, and out-of-vocabulary words need that lexicon extended.

## What exists

### Phoneme recognition models

| project | scale | notes |
|---|---|---|
| **[Tiny Transducer](https://arxiv.org/abs/2101.06856)** (ICASSP 2021) | **0.9M params** after SVD | The extreme edge case. DFSMN encoder + CNN **stateless** predictor, WFST decoding with phone-synchronous blank skipping. Beats a larger conventional hybrid by 9–20% relative on-device. Explicitly phone-based *because* it keeps the decoding graph tiny. |
| **[CUPE](https://arxiv.org/abs/2508.15316)** (2025) | **30.1M params, 115 MB fp32 / ~29 MB int8** *(measured)* | Contextless Universal Phoneme Encoder. Processes fixed **120 ms** windows independently — roughly one phoneme — so it learns acoustics free of language-model context. Competitive cross-lingual with fewer params. Already in this repo: it is the encoder behind [BFA](../basic_force_alignment/). |
| **[Allosaurus](https://github.com/xinjli/allosaurus)** | 737★, last push 2024 | Universal phone recognizer for **2000+ languages**. Language-independent encoder + phone predictor with a per-language *allophone* layer mapping allophones to phonemes. Research-grade, no longer actively developed. |
| **[wav2vec2 espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)** | 418k + 316k downloads | Facebook's IPA-output phoneme models (`xlsr-53` multilingual, `lv-60` English). By far the most-used phoneme ASR weights. **~1 GB — too big for edge as-is**; a distillation or quantization target rather than a deployable. |
| **[Charsiu](https://github.com/lingjzhu/charsiu)** | 347★, **stale since 2022** | Neural phonetic aligner. Listed for completeness; unmaintained. |

### On-device runtimes (what you would actually deploy into)

| project | scale | notes |
|---|---|---|
| **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)** | 13.8k★, active | The strongest option. ONNX Runtime, streaming transducers, ships for Android/iOS/RPi/embedded Linux, and has a broad pretrained zoo. From the k2/icefall lineage. |
| **[vosk-api](https://github.com/alphacep/vosk-api)** | 15.0k★, active | Kaldi-based offline recognition with ~50 MB per-language models. Mature bindings for Android, iOS, Raspberry Pi. |
| **[moonshine](https://github.com/moonshine-ai/moonshine)** | 10.5k★, very active | 27M-param ASR aimed squarely at edge; the *Flavors of Moonshine* paper ([arXiv:2509.02523](https://arxiv.org/pdf/2509.02523)) reports tiny specialized models matching Whisper Medium (28× larger) on 6 languages. **Character/word output, not phonemes** — relevant as the size bar to beat. |
| **[PocketSphinx](https://github.com/cmusphinx/pocketsphinx)** | 4.3k★, active | The classic embedded recognizer, phoneme/HMM-based, runs on hardware nothing else will. Pre-neural accuracy. |
| **[icefall](https://github.com/k2-fsa/icefall)** | 1.5k★, active | Training recipes (Zipformer, pruned RNN-T) that feed sherpa-onnx. Where you would train a custom phoneme transducer. |
| **[ExecuTorch](https://github.com/pytorch/executorch)** | 4.8k★, active | PyTorch's on-device runtime — the deployment path if the model is authored in PyTorch. |

## The size ladder

Everything below was measured on this machine, from artifacts already downloaded for the
other projects in this repo:

| | size | on-device? |
|---|---|---|
| Tiny Transducer (paper) | **0.9M params** | microcontroller |
| CUPE | 30.1M / 115 MB fp32 / **~29 MB int8** | phone, comfortably |
| Vosk per-language model | ~50 MB | phone |
| Gentle's Kaldi models | 191 MB | borderline |
| wav2vec2-base ASR | 360 MB | no, without compression |
| MFA conda environment | 1.3 GB | no |

The jump from CUPE to wav2vec2-base is **12×** for the same task family. That gap is the
whole design space of this project.

## Gaps worth building into

The survey turned up two things that are *not* well covered:

**No small, current, deployable phoneme recognizer with a maintained edge story.** The
pieces exist separately — Tiny Transducer is the right architecture but has no public
implementation; CUPE is small and current but ships as a research checkpoint inside an
aligner; sherpa-onnx is the right runtime but its zoo is word/BPE transducers. Nobody has
joined them.

**No honest edge benchmark.** Claims are quoted at wildly different operating points
(SVD-compressed vs fp32, RTF on unnamed hardware, PER on different phone sets). A
like-for-like comparison of PER against size and latency, on one machine with one phone
set, does not appear to exist publicly.

## Where this connects to the rest of the repo

Two pieces of the work are already here:

- [`wav2vec2_fine_tune/`](../wav2vec2_fine_tune/) has a **complete but untrained** TIMIT
  phoneme-recognition pipeline (39-phone folded set, PER scoring). It is the obvious
  baseline: fine-tune, measure PER, then quantize and measure the size/accuracy curve.
- [`basic_force_alignment/`](../basic_force_alignment/) already runs **CUPE** via BFA, so
  the most interesting small phoneme encoder is installed and working.

A defensible project from here is the missing benchmark: train the TIMIT phoneme model,
compare it against CUPE and a wav2vec2 phoneme model at matched phone sets, and report
**PER vs. model size vs. latency** on identical hardware — with int8 quantization and an
ONNX/ExecuTorch export as the edge path.

## Sources

Papers: [Tiny Transducer](https://arxiv.org/abs/2101.06856) ·
[CUPE](https://arxiv.org/abs/2508.15316) ·
[Allosaurus](https://arxiv.org/pdf/2002.11800) ·
[Flavors of Moonshine](https://arxiv.org/pdf/2509.02523) ·
[BFA](https://arxiv.org/abs/2509.23147)

Repos: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ·
[vosk-api](https://github.com/alphacep/vosk-api) ·
[moonshine](https://github.com/moonshine-ai/moonshine) ·
[allosaurus](https://github.com/xinjli/allosaurus) ·
[pocketsphinx](https://github.com/cmusphinx/pocketsphinx) ·
[icefall](https://github.com/k2-fsa/icefall) ·
[executorch](https://github.com/pytorch/executorch) ·
[charsiu](https://github.com/lingjzhu/charsiu)
