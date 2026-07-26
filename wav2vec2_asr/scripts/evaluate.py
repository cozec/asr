"""Measure wav2vec2 WER on LibriSpeech, and compare bundles by labeled-data budget.

The tutorial stops at one qualitative transcript. This puts numbers on it using the
LibriSpeech copy in ../data, and compares the ASR bundles that differ only in how much
*labeled* data was used to fine-tune the same self-supervised representation -- which is
the central claim of the wav2vec2 paper (Baevski et al. 2020, arXiv:2006.11477).

    python scripts/evaluate.py --num 50
    python scripts/evaluate.py --num 50 --bundles WAV2VEC2_ASR_BASE_10M WAV2VEC2_ASR_BASE_100H WAV2VEC2_ASR_BASE_960H
    python scripts/evaluate.py --num 50 --subset test-other
"""

import argparse
import json
import os
import sys
import time

import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from decoder import GreedyCTCDecoder, to_words
from pipeline_demo import load_audio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRISPEECH = os.path.join(ROOT, "..", "data", "LibriSpeech")

# Labeled fine-tuning budget per bundle -- the axis the wav2vec2 paper varies.
LABELED_HOURS = {
    "WAV2VEC2_ASR_BASE_10M": "10 min",
    "WAV2VEC2_ASR_BASE_100H": "100 h",
    "WAV2VEC2_ASR_BASE_960H": "960 h",
    "WAV2VEC2_ASR_LARGE_10M": "10 min",
    "WAV2VEC2_ASR_LARGE_100H": "100 h",
    "WAV2VEC2_ASR_LARGE_960H": "960 h",
    "WAV2VEC2_ASR_LARGE_LV60K_10M": "10 min",
    "WAV2VEC2_ASR_LARGE_LV60K_100H": "100 h",
    "WAV2VEC2_ASR_LARGE_LV60K_960H": "960 h",
}


def scan_librispeech(subset: str, limit: int):
    """Collect (path, reference) pairs from a LibriSpeech subset."""
    subset_dir = os.path.join(LIBRISPEECH, subset)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"missing {subset_dir}; run ../data/download_librispeech.sh")
    items = []
    for dirpath, _, filenames in sorted(os.walk(subset_dir)):
        for name in sorted(f for f in filenames if f.endswith(".trans.txt")):
            with open(os.path.join(dirpath, name)) as fh:
                for line in fh:
                    utt_id, _, text = line.strip().partition(" ")
                    path = os.path.join(dirpath, f"{utt_id}.flac")
                    if os.path.exists(path):
                        items.append((utt_id, path, text))
                    if len(items) >= limit:
                        return items
    return items


def evaluate(bundle_name: str, items, device) -> dict:
    import jiwer

    bundle = getattr(torchaudio.pipelines, bundle_name)
    model = bundle.get_model().to(device).eval()
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())

    refs, hyps, audio_seconds, compute_seconds = [], [], 0.0, 0.0
    for i, (_, path, text) in enumerate(items, 1):
        waveform = load_audio(path, bundle.sample_rate).to(device)
        audio_seconds += waveform.size(1) / bundle.sample_rate
        t0 = time.perf_counter()
        with torch.inference_mode():
            emission, _ = model(waveform)
        hyp = to_words(decoder(emission[0]))
        compute_seconds += time.perf_counter() - t0
        refs.append(text.upper())
        hyps.append(hyp.upper())
        print(f"\r  {bundle_name}: {i}/{len(items)}", end="", flush=True)

    print()
    return {
        "bundle": bundle_name,
        "labeled": LABELED_HOURS.get(bundle_name, "?"),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "wer": jiwer.wer(refs, hyps) * 100,
        "rtf": compute_seconds / audio_seconds,
        "audio_s": audio_seconds,
        "example_ref": refs[0],
        "example_hyp": hyps[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", nargs="+", default=["WAV2VEC2_ASR_BASE_960H"])
    parser.add_argument("--subset", default="test-clean")
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    items = scan_librispeech(args.subset, args.num)
    print(f"{len(items)} utterances from {args.subset}, device={device}\n")

    results = [evaluate(name, items, device) for name in args.bundles]

    print(f"\n{'bundle':<32}{'labeled':>9}{'params':>9}{'WER%':>8}{'RTF':>8}")
    print("-" * 66)
    for r in results:
        print(f"{r['bundle']:<32}{r['labeled']:>9}{r['params_m']:>8.0f}M"
              f"{r['wer']:>8.2f}{r['rtf']:>8.3f}")

    print(f"\nexample ({args.subset}):")
    print(f"  ref: {results[0]['example_ref'][:95]}")
    for r in results:
        print(f"  {r['labeled']:>6}: {r['example_hyp'][:95]}")

    out = args.out or os.path.join(ROOT, "results", f"{args.subset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"subset": args.subset, "num_utterances": len(items),
                   "results": results}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
