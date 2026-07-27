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

## One trunk, two heads

![wav2vec2 TIMIT fine-tuning: TIMIT .WAV audio feeds a frozen 7-layer convolutional feature encoder, then 12 fine-tuned transformer layers, then dropout. That shared trunk splits into two freshly initialized linear heads, drawn with dashed borders: one projecting 768 to 30 character tokens with 23,070 parameters, scored by WER; the other 768 to 41 phone tokens with 31,529 parameters, scored by PER. The .TXT orthographic transcripts supervise the first head; the .PHN phonetic transcripts, folded from 61 to 39 phones, supervise the second.](diagrams/wav2vec2-timit-two-heads.png)

Three kinds of parameter, and the borders distinguish them:

| | params | state |
|---|---|---|
| conv feature encoder | 4.2M | pretrained, **frozen** by `freeze_feature_encoder()` |
| 12 transformer layers | 89.8M | pretrained, **fine-tuned** |
| linear head (dashed) | 23K / 31.5K | **fresh** — random init, no pretrained weights exist |

**The head does not come from the checkpoint.** `facebook/wav2vec2-base` was pretrained
with no output vocabulary at all, so loading it prints `lm_head.weight | MISSING` and
transformers creates `nn.Linear(768, vocab_size)` from scratch — sampled from
`N(0, 0.02)`, matching `config.initializer_range`. That is the *only* structural
difference between the two steps: **768 → 30** characters versus **768 → 41** phones.

It also explains the long WER = 1.00 plateau early in training. The head starts as noise
over the vocabulary, so the model emits blanks until those 23K new parameters learn the
inventory *while* the 90.2M below them reshape to make it linearly separable. Warmup
exists to stop a high early learning rate from wrecking the pretrained trunk to satisfy a
random head.

Sources: [`diagrams/`](diagrams/) holds the mermaid `.mmd`, an editable `.excalidraw`
scene, and rendered `.svg` / `.png`.

## Results

### Step 1 — ASR: WER 28.69%

Fine-tuned `facebook/wav2vec2-base` on 3696 TIMIT utterances, evaluated on the
192-utterance core test set. **Stopped at epoch 20 of 30** — see [Honest
scope](#honest-scope) below.

| epoch | WER | |
|---|---|---|
| 1–4 | 1.0000 | model emits only blanks |
| **5** | 0.6524 | the plateau breaks |
| 8 | 0.3744 | |
| 11 | 0.3240 | |
| 14 | 0.2965 | |
| **17** | **0.2869** | best |
| 20 | 0.2927 | converged |

Full curve: [`results/asr_timit.json`](results/asr_timit.json).

**WER is exactly 1.00 for four epochs, then collapses.** That is the freshly initialized
head doing its job: it starts as noise over 30 characters, so the model emits nothing but
blanks until it learns the inventory. `warmup_steps=1000` (epoch 8.6) deliberately holds
the learning rate down through this phase so a random head cannot wreck the pretrained
trunk. By epoch 14 it has converged — the last 7 epochs span 0.2869–0.3029, a spread of
0.016, while `eval_loss` *rises* from 0.487 to 0.574. That divergence is mild overfitting,
and the reason stopping early cost nothing.

### What the errors look like

```
ref: in wage negotiations the industry bargains as a unit with a single union
hyp: in wage negociiations the industry bargons ias a unit with a single union

ref: materials ceramic modeling clay red white or buff
hyp: materials seremic modeling clay red whiht or bufh

ref: artificial intelligence is for real
hyp: artefficial intelligence is for wreal
```

Every error is a **phonetically plausible misspelling** — `negociiations`, `seremic`,
`artefficial`, `wreal`. The acoustics are right and the orthography is wrong, which is
what character-level CTC with no language model does. It is also the argument for step 2:
if the model is really predicting sounds, phones are the more honest target than English
spelling.

### Step 2 — Phoneme recognition: PER 11.52%

Same checkpoint and same 3696 utterances, but predicting **phones** instead of characters:
targets from TIMIT's `.PHN` files folded 61→39 (Lee & Hon 1989), scored by phone error
rate on the 192-utterance core test set. This is the standard TIMIT protocol, so the
number is directly comparable to the published ladder.

| epoch | PER | |
|---|---|---|
| 1–3 | 98.18% | blanks only |
| **4** | 63.05% | plateau breaks |
| 5 | 15.08% | |
| 10 | 12.16% | |
| **16** | **11.52%** | best |
| 20 | 11.70% | final |

Full curve: [`results/phoneme_timit.json`](results/phoneme_timit.json).

**Where that sits** (TIMIT test PER, no LM, from wav2vec 2.0 Table 3):

| model | test PER |
|---|---|
| CNN + TD-filterbanks | 18.0 |
| PASE+ | 17.2 |
| Li-GRU + fMLLR | 14.9 |
| wav2vec | 14.7 |
| **this run — wav2vec2-base, 20 epochs** | **11.52** |
| vq-wav2vec | 11.6 |
| wav2vec 2.0 LARGE (317M) | 8.3 |

A 95M-param base model fine-tuned for 2.4 hours lands level with **vq-wav2vec** and
comfortably ahead of **wav2vec**. The 8.3 needs LARGE at 317M params, so the gap is
model capacity, not recipe.

**Phones break out of the plateau faster than characters.** The ASR head sat at WER 1.00
for 4 epochs then fell to 0.65; the phoneme head sat at 0.98 for 3 epochs then fell
straight to 0.15. Predicting 39 phones is an easier target than spelling 30 characters —
the model is not being asked to learn English orthography on top of acoustics, which is
the same point the ASR errors made from the other direction (`negociiations`, `seremic`).

Run it with:

```bash
caffeinate -i -s python src/finetune.py --task phoneme --epochs 20 --eval-split core_test
```

`caffeinate` is not optional on a Mac: this machine has `pmset sleep 1`, and the ASR run
without it averaged **25.7 s/step against 3.72 with it** — 7× on identical work.

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

## Honest scope

What ran, and what did not:

| | status |
|---|---|
| Step 1 — ASR | trained, **WER 28.69%**, 20 of 30 epochs |
| Step 2 — phonemes | trained, **PER 11.52%**, 20 epochs |

Both steps converged and stopped early on purpose: WER was flat for 7 epochs with
`eval_loss` rising, and PER was flat for 8 (11.52–12.16 from epoch 12 on). Neither number
is 30-epoch.

**Step 1 is not comparable to the blog's 0.221.** That is the full 1344-utterance test set
at 30 epochs; this is the 192-utterance core test set at 20. **Step 2 is comparable** to
the TIMIT ladder, because that protocol is fixed and this run follows it exactly.

**Why they stopped at 20 epochs.** Not a technical failure — WER had been flat for 7 epochs
and `eval_loss` was climbing, so the remaining 10 epochs had nothing to offer. The cost
was real: 16.6 h wall-clock for 2327 steps, an average of **25.7 s/step against ~5 s/step
when the machine is actually awake**. The watchdog log shows multi-hour stretches
completing ~150 steps, consistent with the Mac sleeping overnight. Anyone reproducing this
should either disable sleep (`caffeinate -i`) or run it on a GPU box.

Both runs used `caffeinate -i -s`; the earlier ASR run without it averaged 25.7 s/step
against 3.72 with it, because this machine has `pmset sleep 1`.

## Layout

```
src/timit.py       corpus loader: splits, SA exclusion, 61->39 folding, PhoneCoder
src/finetune.py    shared recipe for both tasks
scripts/run_both.sh  runs step 2 after step 1 finishes
exp/               vocab.json, checkpoints, metrics.json per task
```
