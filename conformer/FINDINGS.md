# Conformer — findings & session state

Working notes for [README.md](README.md). Everything below is measured on this machine
(Apple M5, 16 GB, macOS 25.5, Python 3.11, torch 2.13.0 / torchaudio 2.11.0, MPS),
not quoted from anywhere.

Last updated: 2026-07-25, ~01:30.

---

## TL;DR — where things stand

| | |
|---|---|
| Implementation | Done. Encoder + CTC and RNN-T heads, S/M/L configs. |
| Tests | 15/15 passing (`python tests/test_conformer.py`). |
| Param counts vs paper Table 1 | S +0.2%, L +1.2%, M +5.6%. |
| LibriSpeech 960 h | Downloaded, md5-verified, manifests cached. |
| Tokenizer | Rebuilt on all 960 h (1024 pieces + blank). |
| **Training** | **Never run for real.** Only smoke/benchmark runs; no usable checkpoint exists. |

**The one open decision:** lower `warmup_steps` before launching a 100 h run — see
[Pending decision](#pending-decision-warmup_steps) below.

---

## 1. Prior-art search (why this is a fresh implementation)

| Project | Verdict |
|---|---|
| [sooftware/conformer](https://github.com/sooftware/conformer) (1.1k★) | Model only. Encoder never threads a padding mask into the blocks (`encoder.py:202` → `layer(outputs)`), so attention and depthwise conv read padded frames. Cloned to `reference/` as a cross-check. |
| [`torchaudio.models.Conformer`](https://docs.pytorch.org/audio/stable/generated/torchaudio.models.Conformer.html) | Absolute positional encoding, not the paper's relative scheme (ablating it costs 0.4/1.5 WER, paper Table 3). Encoder only. |
| [LuluW8071/Conformer](https://github.com/LuluW8071/Conformer) | Full pipeline but 15.94% WER vs paper's 2.7%; departs from the paper's recipe (AdamW @1e-4, ReduceLROnPlateau). |
| ESPnet / WeNet / NeMo | Solid, but the Conformer is buried under a framework. |

Conclusion: model code exists, a faithful *and* self-contained trainable version did not.

## 2. Paper facts worth keeping (arXiv:2005.08100)

Table 1 — **the WebFetch summary of this table was hallucinated**; these come from
reading the PDF directly. Trust these:

| | S | M | L |
|---|---|---|---|
| Params | 10.3M | 30.7M | 118.8M |
| Encoder layers | 16 | 16 | 17 |
| Encoder dim | 144 | 256 | 512 |
| Attention heads | 4 | 4 | 8 |
| Conv kernel | 32 | 32 | 32 |
| Decoder layers / dim | 1 / 320 | 1 / 640 | 1 / 640 |

WER without LM: S 2.7/6.3, M 2.3/5.0, L 2.1/4.3. With LM: 2.1/5.0, 2.0/4.3, 1.9/3.9.

Recipe (§3.1–3.2): 80-ch fbank @25 ms/10 ms · SpecAugment F=27, 10 time masks, pS=0.05 ·
1k WPM · Adam β=(0.9, 0.98) ε=1e-9 · L2 1e-6 · 10k warmup · peak LR 0.05/√d · dropout 0.1.
The paper's decoder is a **single-LSTM-layer RNN-T**; joint-network dims are never specified.

## 3. Verification

**Parameter counts** (`python src/param_count.py`):

```
conformer_s    8.69M enc + 1.63M dec =  10.32M   paper 10.3M   +0.2%
conformer_m   27.27M enc + 5.17M dec =  32.43M   paper 30.7M   +5.6%
conformer_l  114.86M enc + 5.33M dec = 120.19M   paper 118.8M  +1.2%
```

Encoders line up closely; the residual gap is in the joint network, whose width the
paper never gives. M is the loose one — a narrower joint would close most of the 5.6%.

**Tests** — `python tests/test_conformer.py`, 15 assertions. The load-bearing ones:
`test_rel_shift` (asserts the exact `i-j` Toeplitz matrix, i.e. Transformer-XL `R_{i-j}`),
and `test_padding_invariance` / `test_batch_invariance` (an utterance must encode
identically alone vs. padded in a batch — the property the reference lacks).

**Deliberate deviation — masked BatchNorm.** Plain BatchNorm takes statistics over
padded frames; measured drift on valid frames was **1.64**. `MaskedBatchNorm1d` restricts
statistics to valid frames → **3e-7**, and is provably identical to `nn.BatchNorm1d`
(outputs *and* running stats) when nothing is padded. This is a correct implementation of
what the paper describes, not an architecture change.

## 4. Environment gotchas (hard-won)

- **MPS has no CTC or RNN-T kernel.** `aten::_ctc_loss` and torchaudio's `rnnt_loss` are
  CPU/CUDA only. `train.py` computes the loss on CPU; autograd copies gradients back and
  the model stays on GPU. Automatic, prints a note.
- **Swap thrashing looks exactly like a hang.** Running training alongside the 60 GB
  download drove swap to 20.7/21.5 GB and produced ~16-minute stalls mid-epoch. Check
  `sysctl vm.swapusage` and `memory_pressure` before debugging code.
- **`num_workers` — a false alarm I chased.** I measured a 974s first batch with
  `num_workers=4` vs. instant with 0, concluded that forking after Metal init stalls on
  macOS, and changed the code to force 0 on MPS. **That was wrong** — it was swap noise.
  Re-tested alternating order on a quiet machine: `num_workers=4` → 1.6s, `num_workers=0`
  → 5.0s. Workers are *faster*. The change was reverted; configs keep `num_workers: 4`.
  Do not re-introduce it.
- **Memory is the limit, not compute.** Relative-position attention materialises a
  `(B, H, T, 2T-1)` score tensor per layer, so long utterances dominate. Hence
  `max_duration: 17.0` (drops 0.2% of train-clean-100).

## 5. Data

All 7 archives downloaded and **md5-verified** (`ALL PARTS OK`), via
[`../data/download_librispeech.sh`](../data/download_librispeech.sh) (resumable, skips
completed parts).

| | Utterances | Hours | Size |
|---|---|---|---|
| train-clean-100 / -360 / other-500 | 281,241 | **961.1** | 58 GB |
| dev + test (4 sets) | 11,126 | 21.3 | 1.4 GB |
| Total | 292,367 flac | | 59 GB |

961.1 h matches the paper's "970 hours" to within rounding.

Manifests are cached in `exp/manifests/*.jsonl` — the ~280k `sf.info` header probes
(~40 s) are done once, not per run.

**Tokenizer**: rebuilt on all 960 h in **24 s** (scaling is linear, ~0.073 ms/sentence:
10k→0.9s, 30k→2.4s, 90k→6.6s). Output `exp/spm_1k.model`, 1025 ids = 1024 unigram pieces
+ blank at id 0, exact round-trip, ~1.7 pieces/word. Earlier smoke runs used a dev-clean
stand-in — that is now superseded.

## 6. Measured throughput (Conformer-S CTC, MPS)

On `train-clean-100`, config batch size (8 × accum 4 = 32 utts/step), steady state and
perfectly linear over 30 steps:

| | |
|---|---|
| **2.36 s / optimizer step** | 13.6 utterances/s |
| 890 steps / epoch | 28,488 utts after the 17 s filter |
| **35 min / epoch** | |

| Epochs | Steps | Wall-clock |
|---|---|---|
| 20 | 18k | **12 h** |
| 50 | 44k | **29 h** |
| 100 (config default) | 89k | **58 h ≈ 2.4 days** |

Scaling to the full 960 h: ~9× more data → **~5.3 h/epoch**, i.e. weeks here. That is a
GPU-box job, not a laptop job.

Other measurements: pure MPS fwd+bwd 0.18 s/batch (B=4, T=800), Metal warmup 0.43 s,
fbank extraction ~2 ms/utterance.

## 7. Does it learn?

Yes — CTC loss falls steadily while still deep in warmup. Two independent runs:

```
dev-clean,        batch 4:  step 10 34.93 → step 60 23.63
train-clean-100,  batch 8:  step  5 34.19 → step 30 30.12
```

Inference plumbing verified end-to-end (argmax → collapse repeats → strip blanks →
detokenize). Output shapes: CTC `(B, T/4, 1025)` log-probs at a **40 ms** frame rate;
RNN-T joint `(B, T/4, U+1, 1025)` logits. Decoding is **greedy only** — no beam search,
no LM shallow fusion (the paper's "with LM" column uses a 3-layer LSTM LM, width 4096).

## Pending decision: `warmup_steps`

`configs/*.yaml` carry the paper's `warmup_steps: 10000`, which is calibrated for 960 h.
On 100 h that is **11.2 epochs just to reach peak LR** — over half a 20-epoch run spent
barely training.

**Recommendation for a 100 h run: set `warmup_steps: 2000`** (≈2.2 epochs). Everything
else in the recipe carries over unchanged; this one knob is tied to corpus size.

**Expected result:** ~10–15% WER on test-clean for 100 h + CTC + greedy decoding. The
paper's 2.7% needs 960 h + RNN-T + LM. Don't read the gap as a bug.

## Resume tonight

State: tokenizer built, manifests cached, **no checkpoints** (the 8-step smoke checkpoint
was deleted). Nothing is running.

```bash
cd /Users/adam/interviews/asr/conformer

# sanity (≈1 min)
.venv/bin/python tests/test_conformer.py && .venv/bin/python src/param_count.py

# 1. lower warmup for 100h  ->  configs/conformer_s.yaml: warmup_steps: 2000
# 2. launch (~12 h for 20 epochs); log_every 50 ≈ every 2 min
nohup .venv/bin/python -u src/train.py --config configs/conformer_s.yaml \
      --train-sets train-clean-100 > logs/train_100h.log 2>&1 &

# watch
tail -f logs/train_100h.log
grep -E "^epoch" logs/train_100h.log | tail

# 3. decode when checkpoints exist
.venv/bin/python src/decode.py --config configs/conformer_s.yaml \
      --checkpoint exp/conformer_s/epoch19.pt --test-sets test-clean test-other
```

`--resume exp/conformer_s/epochN.pt` picks up optimizer + LR-schedule state after an
interruption. Checkpoints are ~107 MB each, one per epoch.

### Ideas if we keep going

- **Length bucketing** — batches are padded to the longest member; sorting by duration
  would cut wasted compute noticeably. Not implemented.
- **Beam search + LM shallow fusion** — the paper's "with LM" numbers need it.
- **Checkpoint averaging** — standard for this recipe; config keys for it were removed
  as dead weight rather than left misleading.
- **RNN-T head** — implemented and shape-tested but never trained; it's the paper's
  actual model and the expensive path (its loss also falls back to CPU on MPS).
