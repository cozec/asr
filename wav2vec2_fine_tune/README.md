# wav2vec2 fine-tuning on TIMIT — ASR and phoneme recognition

Two experiments on the same corpus and the same pretrained checkpoint:

1. **Step 1** — reproduce [Fine-Tune Wav2Vec2 for English ASR with 🤗
   Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english): fine-tune
   `facebook/wav2vec2-base` on TIMIT for character-level ASR, scored by **WER**.
2. **Step 2** — fine-tune the same checkpoint for **phoneme recognition** using TIMIT's
   `.PHN` annotations, scored by **PER** on the standard folded 39-phone set.

The recipe is shared; only the labels and the metric differ, so both live in
[`src/finetune.py`](src/finetune.py) behind `--task asr|phoneme`.

```bash
python src/finetune.py --task asr      --epochs 30    # step 1
python src/finetune.py --task phoneme  --epochs 30    # step 2
```

## The data

**TIMIT is licensed (LDC93S1) and cannot be downloaded from the blog's
`load_dataset("timit_asr")`** — that loader is a script with no data attached and expects
a local copy. This project reads the corpus off disk instead, which also exposes the
phonetic annotations the blog never uses and step 2 needs.

> ⚠️ The copy used here came from a public HuggingFace mirror. TIMIT is distributed by
> the LDC under a paid licence; those mirrors are redistributions, not licensed
> channels. Use an institutional LDC copy for anything beyond personal experimentation.

[`src/timit.py`](src/timit.py) applies the standard TIMIT conventions, none of which the
blog handles:

| convention | effect |
|---|---|
| **SA1/SA2 excluded** — all 630 speakers read the same two dialect sentences, so keeping them lets the model memorise test text | train 4620 → **3696**, test 1680 → **1344** |
| **61 → 39 phone folding** (Lee & Hon, 1989), the standard PER protocol | verified to yield exactly **39** phones |
| **core test set** available (24 speakers) | `--eval-split core_test` → 192 utterances |

Audio is NIST SPHERE rather than RIFF; `soundfile` reads it directly.

## Step 2: how phoneme recognition differs

Only two things change from step 1:

- **Labels** come from `.PHN` (frame-aligned phones) instead of `.TXT`, folded 61 → 39.
- **Scoring** is PER instead of WER — edit distance over phone sequences.

One implementation wrinkle: `Wav2Vec2CTCTokenizer` tokenises text into *characters*, so a
multi-character symbol like `aa` can never be produced. `PhoneCoder` maps each phone to a
private-use codepoint, so the stock CTC tokenizer and decoder work unmodified, and
inverts the mapping for scoring.

## Adaptations for transformers 5.x

The blog targets transformers 4.x. Five of its APIs have since changed, and the notebook
fails on each without these:

| blog | transformers 5.14 |
|---|---|
| `processor.as_target_processor()` | **removed** — call the tokenizer directly |
| `model.freeze_feature_extractor()` | renamed `freeze_feature_encoder()` |
| `TrainingArguments(evaluation_strategy=)` | renamed `eval_strategy` |
| `TrainingArguments(group_by_length=True)` | **removed** — we sort by duration ourselves |
| `datasets.load_metric("wer")` | **removed** — `jiwer` computes WER/PER |

`notebook_login()` and the hub upload are omitted, as requested.

## Apple Silicon notes

- **MPS has no `aten::_ctc_loss` kernel.** Unlike a hand-written loop, the loss is
  computed inside `Wav2Vec2ForCTC.forward`, so it cannot simply be run on CPU. The script
  sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before torch loads, which falls that single op
  back to CPU while the rest stays on the GPU.
- **`fp16` is off** (unsupported on MPS); the blog's GPU recipe uses it.
- Batch 8 × 4 accumulation reproduces the blog's effective batch of 32 within 16 GB.

## Layout

```
src/timit.py       corpus loader: splits, SA exclusion, 61->39 folding, PhoneCoder
src/finetune.py    shared recipe for both tasks
scripts/run_both.sh  runs step 2 after step 1 finishes
exp/               vocab.json, checkpoints, metrics.json per task
```
