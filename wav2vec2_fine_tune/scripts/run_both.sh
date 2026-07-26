#!/usr/bin/env bash
# Step 2 (phoneme) after step 1 (ASR) finishes. Both: 30 epochs, batch 8 x accum 4
# (the blog's effective 32), evaluated on TIMIT's 192-utterance core test set.
cd "$(dirname "$0")/.."
while pgrep -f "finetune.py --task asr" > /dev/null; do sleep 60; done
echo "[$(date '+%H:%M')] ASR finished -> starting phoneme"
.venv/bin/python -u src/finetune.py --task phoneme --epochs 30 --batch-size 8 --accum 4 \
    --eval-split core_test --output-dir exp/wav2vec2-timit-phoneme > logs/train_phoneme.log 2>&1
echo "[$(date '+%H:%M')] phoneme finished"
