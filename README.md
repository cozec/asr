# ASR — Automatic Speech Recognition models

One folder per model, each a self-contained implementation of its paper. Shared corpora
live in [`data/`](data/) so models don't duplicate tens of gigabytes.

## Models

| Folder | Paper | Status |
|---|---|---|
| [`conformer/`](conformer/) | [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) (Gulati et al., INTERSPEECH 2020) | Implemented — encoder + CTC/RNN-T heads, S/M/L configs, 15 tests passing |
| [`rnnt/`](rnnt/) | [Streaming End-to-end Speech Recognition For Mobile Devices](https://arxiv.org/abs/1811.06621) (He et al., ICASSP 2019) | Implemented + **live streaming demo** — 5.99% WER at RTF 0.138, 18 tests passing |
| [`wav2vec2_asr/`](wav2vec2_asr/) | [wav2vec 2.0](https://arxiv.org/abs/2006.11477) (Baevski et al., NeurIPS 2020) — via torchaudio's pipeline tutorial | Tutorial reproduced exactly; decoder and labeled-data studies — best 1.64% WER, 10 tests passing |
| [`wav2vec2_fine_tune/`](wav2vec2_fine_tune/) | [HF blog: Fine-Tune Wav2Vec2 for English ASR](https://huggingface.co/blog/fine-tune-wav2vec2-english) + TIMIT phoneme recognition | Step 1 trained — **WER 0.2869** on TIMIT core test; step 2 pipeline complete but untrained |
| [`wave2vec2_forced_alignment/`](wave2vec2_forced_alignment/) | [torchaudio: Forced Alignment with Wav2Vec2](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html) | Tutorial reproduced exactly — matching frame indices and scores on all 9 words, 5 figures |
| [`basic_force_alignment/`](basic_force_alignment/) | [Gentle](https://github.com/strob/gentle) — Kaldi-based forced aligner | Demo run on a 96 s recording — 101/105 words aligned, word + phone timings, 3 figures |
| [`phoneme_asr/`](phoneme_asr/) | Landscape survey: phoneme-based ASR for edge devices | Prior-art search — 5 phoneme models, 6 on-device runtimes, measured size ladder, gaps identified |

## Shared data

[`data/download_librispeech.sh`](data/download_librispeech.sh) fetches the full
LibriSpeech corpus (960 h train + dev/test, ~60 GB) into `data/LibriSpeech/`. It is
resumable, verifies every archive against the mirror's md5, and skips parts already
extracted — re-run it after an interruption.

```bash
bash data/download_librispeech.sh
tail -f data/download.log
```

| Subset | Hours | Archive |
|---|---|---|
| train-clean-100 / -360, train-other-500 | 960 | 6.3G / 23G / 30G |
| dev-clean, dev-other | 5.4 / 5.3 | 337M / 314M |
| test-clean, test-other | 5.4 / 5.1 | 346M / 328M |

## Conventions for new models

Each model folder is independent — its own `.venv`, `requirements.txt`, and `README.md`
explaining what the paper specifies and where the implementation departs from it.
Point configs at `../data/` rather than copying corpora.
