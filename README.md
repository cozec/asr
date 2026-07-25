# ASR — Automatic Speech Recognition models

One folder per model, each a self-contained implementation of its paper. Shared corpora
live in [`data/`](data/) so models don't duplicate tens of gigabytes.

## Models

| Folder | Paper | Status |
|---|---|---|
| [`conformer/`](conformer/) | [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) (Gulati et al., INTERSPEECH 2020) | Implemented — encoder + CTC/RNN-T heads, S/M/L configs, 15 tests passing |
| [`rnnt/`](rnnt/) | [Streaming End-to-end Speech Recognition For Mobile Devices](https://arxiv.org/abs/1811.06621) (He et al., ICASSP 2019) | Implemented + **live streaming demo** — 5.99% WER at RTF 0.138, 18 tests passing |

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
