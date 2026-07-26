#!/usr/bin/env bash
# Step 1 (ASR) then step 2 (phoneme). Evaluated on TIMIT's core test set (192
# utterances) -- the standard benchmark, and far lighter on memory than the full 1344.
cd "$(dirname "$0")/.."
set -x
.venv/bin/python -u src/finetune.py --task asr --epochs 30 --batch-size 8 --accum 4 \
    --eval-split core_test --output-dir exp/wav2vec2-timit-asr > logs/train_asr.log 2>&1
.venv/bin/python -u src/finetune.py --task phoneme --epochs 30 --batch-size 8 --accum 4 \
    --eval-split core_test --output-dir exp/wav2vec2-timit-phoneme > logs/train_phoneme.log 2>&1
