"""Train the 1k word-piece SentencePiece model on LibriSpeech transcripts (paper §3.2).

    python scripts/train_tokenizer.py --config configs/conformer_s.yaml
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data.dataset import load_manifest_cached
from data.tokenizer import train_sentencepiece


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sets", nargs="+", default=None,
                        help="transcript sources; defaults to the config's train sets")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    sets = args.sets or cfg["data"]["train_sets"]
    print(f"collecting transcripts from {sets} ...")
    # Via the manifest cache: a plain scan would re-probe every FLAC header with
    # sf.info (~40s over the 960h set) purely to get durations we don't need here.
    manifest_dir = os.path.join(
        os.path.dirname(cfg["train"]["save_dir"].rstrip("/")) or "exp", "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    texts = [u.text for u in load_manifest_cached(cfg["data"]["root"], sets, manifest_dir)]
    print(f"  {len(texts)} utterances")

    model_prefix = os.path.splitext(cfg["data"]["tokenizer"])[0]
    path = train_sentencepiece(texts, model_prefix, cfg["data"]["vocab_size"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
