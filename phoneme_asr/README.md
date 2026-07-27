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
| **[Multilingual-PR](https://github.com/ASR-project/Multilingual-PR)** | 266★, **stale since 2022**, no license | Student project comparing wav2vec2 / HuBERT / WavLM for phoneme recognition across 5 languages. Its numbers are a useful reference (see below), but the code is 4 years stale and unlicensed. |

### On-device runtimes (what you would actually deploy into)

| project | scale | notes |
|---|---|---|
| **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)** | 13.8k★, active | The strongest option. ONNX Runtime, streaming transducers, ships for Android/iOS/RPi/embedded Linux, and has a broad pretrained zoo. From the k2/icefall lineage. |
| **[vosk-api](https://github.com/alphacep/vosk-api)** | 15.0k★, active | Kaldi-based offline recognition with ~50 MB per-language models. Mature bindings for Android, iOS, Raspberry Pi. |
| **[moonshine](https://github.com/moonshine-ai/moonshine)** | 10.5k★, active daily; **108 MB fp32 / ~27 MB int8** *(measured)* | Edge-first ASR, MIT weights. *Flavors of Moonshine* ([arXiv:2509.02523](https://arxiv.org/pdf/2509.02523)) reports tiny specialized models matching Whisper Medium, 28× larger, on 6 languages. **Subword output (`vocab_size: 32768`), not phonemes** — so it sets the size bar rather than providing a PER. Best candidate to adapt: see [below](#the-concrete-candidate-a-phoneme-head-on-moonshines-encoder). |
| **[PocketSphinx](https://github.com/cmusphinx/pocketsphinx)** | 4.3k★, active | The classic embedded recognizer, phoneme/HMM-based, runs on hardware nothing else will. Pre-neural accuracy. |
| **[icefall](https://github.com/k2-fsa/icefall)** | 1.5k★, active | Training recipes (Zipformer, pruned RNN-T) that feed sherpa-onnx. Where you would train a custom phoneme transducer. |
| **[ExecuTorch](https://github.com/pytorch/executorch)** | 4.8k★, active | PyTorch's on-device runtime — the deployment path if the model is authored in PyTorch. |

## The size ladder

Everything below was measured on this machine, from artifacts already downloaded for the
other projects in this repo:

| | size | on-device? |
|---|---|---|
| Tiny Transducer (paper) | **0.9M params** | microcontroller |
| moonshine-tiny | 108 MB fp32 / **~27 MB int8** | phone, comfortably |
| CUPE | 30.1M / 115 MB fp32 / **~29 MB int8** | phone, comfortably |
| moonshine-base | 246 MB fp32 / ~62 MB int8 | phone |
| Vosk per-language model | ~50 MB | phone |
| Gentle's Kaldi models | 191 MB | borderline |
| wav2vec2-base ASR | 360 MB | no, without compression |
| MFA conda environment | 1.3 GB | no |

The jump from CUPE to wav2vec2-base is **12×** for the same task family. That gap is the
whole design space of this project. Note that moonshine-tiny and CUPE land in the same
place — ~27 vs ~29 MB int8 — but only one of them emits phones, which is the whole point
of [the adaptation below](#the-concrete-candidate-a-phoneme-head-on-moonshines-encoder).

## Benchmarks

**Yes — and phoneme recognition has a better-established benchmark than word ASR does.**
This was missing from the first version of this survey.

### TIMIT PER — the canonical one

TIMIT (1988) with **phone error rate** on the core test set is *the* phoneme recognition
benchmark, and the results ladder runs decades deep. The protocol is fixed and everyone
follows it:

* train on the 3696-utterance set (SA1/SA2 dialect sentences excluded)
* evaluate on the 192-utterance **core test set** (24 speakers)
* recognize 61 phones, **fold to 39** (Lee & Hon 1989) *before* scoring
* PER = (S + I + D) / N, Levenshtein over the folded phone sequence

The ladder, as reported in wav2vec 2.0 Table 3 ([Baevski et al.
2020](https://arxiv.org/abs/2006.11477)):

| model | dev PER | test PER |
|---|---|---|
| CNN + TD-filterbanks | 15.6 | 18.0 |
| PASE+ | – | 17.2 |
| Li-GRU + fMLLR | – | 14.9 |
| wav2vec | 12.9 | 14.7 |
| vq-wav2vec | 9.6 | 11.6 |
| **wav2vec 2.0 LARGE (LS-960), no LM** | **7.4** | **8.3** |
| *[this repo](../wav2vec2_fine_tune/) — wav2vec2-**base**, 20 epochs* | – | *11.52* |

That folding step matters more than it looks: PER on the 61-phone set is not comparable
to PER on the folded 39, and papers do not always say which they report. Any number
quoted without the phone set is unusable.

### Multilingual-PR: the frozen-vs-fine-tuned number

[ASR-project/Multilingual-PR](https://github.com/ASR-project/Multilingual-PR) is the most
directly useful set of published phoneme numbers found in this search, even though the
repo itself is stale. It fine-tunes three English-pretrained SSL models — wav2vec2 Base,
HuBERT Large, WavLM Base/Large — for phoneme recognition on **Common Voice 6.1** in five
non-English languages, with IPA targets from `phonemizer`.

| regime | avg test PER | best model |
|---|---|---|
| **Fine-tuned** | **17.36** | HuBERT Large |
| **Frozen features** (train only the head) | **28.31** | WavLM Large |

Per language, fine-tuned: Italian 12.67 (62 h) · Turkish 14.19 (2.5 h) · Dutch 16.49
(13 h) · Russian 18.88 (16 h) · Swedish 19.38 (3 h).

Two things worth taking from it:

**Freezing the encoder costs ~11 PER points** — 28.31 vs 17.36. That is the price of
linear probing, quantified. Relevant here because freezing is the cheap option for edge
work, and this says how much accuracy it buys back to fine-tune.

**Turkish at 2.5 h scores 14.19, better than Russian at 16 h.** Hours of data matter less
than the match between the pretrained model and the target language's phonology. Their
Swedish scarcity curve makes the same point: 10 min → 39.38 PER, 3 h → 32.68 PER frozen,
so the first few hours buy a lot and the curve flattens fast.

**These numbers do not sit on the TIMIT ladder.** Different corpus (Common Voice, not
TIMIT), different phone set (IPA via `phonemizer`, not folded ARPAbet-39), different
languages. Useful as a cross-lingual reference and for the frozen/fine-tuned ratio; not
as a substitute for a TIMIT PER.

### Other corpora

| corpus | used for |
|---|---|
| **Buckeye** | conversational American English with hand-corrected phonetic labels. BFA evaluates boundary recall here alongside TIMIT — closer to real speech than TIMIT's read sentences. |
| **AlloVera / UCLA Phonetic Corpus** | multilingual phone recognition; the basis for Allosaurus's 2000-language claim. Evaluation is by allophone-to-phoneme mapping rather than a single PER. |
| **Common Voice + espeak G2P** | how the `facebook/*-espeak-cv-ft` models were fine-tuned and evaluated — IPA targets generated by espeak rather than hand-labelled, so it measures agreement with a G2P system, not with phoneticians. |

### What is actually missing

The accuracy axis is well covered. What no public benchmark reports is **PER against
model size and latency on named hardware**. Published claims sit at incompatible
operating points: Tiny Transducer quotes params after SVD compression, CUPE quotes
cross-lingual GER, BFA quotes a speed multiple over MFA, and none share a phone set or a
device. That is the gap this folder could close, not the accuracy ladder itself.

### We are already set up to enter the ladder

**Done.** [`wav2vec2_fine_tune/`](../wav2vec2_fine_tune/) implements this protocol exactly
— 3696/192 splits, 61→39 folding verified to yield exactly 39 phones — and a 20-epoch
`wav2vec2-base` fine-tune scored **11.52% PER**, level with vq-wav2vec and ahead of
wav2vec. 2.4 hours on an M5.

That beat the 12–20% I predicted before running it. The lesson is the one the survey is
built on: the ladder is public, the protocol is fixed, and running the thing costs less
than arguing about where it would land.

## Gaps worth building into

The survey turned up two things that are *not* well covered:

**No small, current, deployable phoneme recognizer with a maintained edge story.** The
pieces exist separately — Tiny Transducer is the right architecture but has no public
implementation; CUPE is small and current but ships as a research checkpoint inside an
aligner; sherpa-onnx is the right runtime but its zoo is word/BPE transducers. Nobody has
joined them.

**No edge benchmark.** The *accuracy* ladder is well established (see
[Benchmarks](#benchmarks) above) — what is missing is the size and latency axis measured
like-for-like, on one machine with one phone set.

### The concrete candidate: a phoneme head on Moonshine's encoder

[Moonshine](https://github.com/moonshine-ai/moonshine) is the best-positioned base for
closing the first gap, because it is the only project in this survey that is
simultaneously small, current, and licensed for reuse.

| | |
|---|---|
| size | **108 MB fp32 / ~27 MB int8** (tiny), 246 MB / ~62 MB (base) *(measured)* |
| architecture | 6 encoder + 6 decoder layers, hidden 288 |
| activity | 10.5k★, pushed the day of this survey, 196k downloads on tiny |
| licence | **MIT** on the weights — reuse is clean |
| paper | [arXiv:2410.15608](https://arxiv.org/abs/2410.15608) |

Its actual innovation is variable-length input: Whisper pads every clip to 30 s, Moonshine
does not, which is where most of the latency win on short utterances comes from. That
property matters more for edge than parameter count does.

**But it is not a phoneme model.** `vocab_size: 32768` — a subword vocabulary, so it emits
text and a WER, and nothing it produces lands on the TIMIT ladder.

The build is therefore: **keep the encoder, replace the 32k-token decoder with a phoneme
CTC head.** That is the same surgery as `wav2vec2_fine_tune` step 2 — a fresh
`nn.Linear(hidden, num_phones)` on a pretrained trunk — against a trunk that is 4× smaller
and already has a streaming edge runtime behind it. Moonshine-tiny's encoder at ~27 MB
int8 lands where CUPE does, but with maintenance and an ONNX path that CUPE lacks.

Sequencing matters: this is only worth doing *after* there is a TIMIT PER from the
existing pipeline, because without a baseline number there is nothing to say whether the
smaller trunk cost accuracy or not.

## Where this connects to the rest of the repo

Two pieces of the work are already here:

- [`wav2vec2_fine_tune/`](../wav2vec2_fine_tune/) has a **trained** TIMIT phoneme model at
  **11.52% PER** on the standard protocol. That is the baseline: next is quantizing it and
  measuring the size/accuracy curve, which is the axis nobody publishes.
- [`basic_force_alignment/`](../basic_force_alignment/) already runs **CUPE** via BFA, so
  the most interesting small phoneme encoder is installed and working.

With the baseline in hand, the missing benchmark is now: quantize this model to int8,
compare it against CUPE and a Moonshine-encoder variant at the same 39-phone set, and
report **PER vs. model size vs. latency** on identical hardware.

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
