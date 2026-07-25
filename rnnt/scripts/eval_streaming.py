"""Measure streaming WER and real-time factor over a set of LibriSpeech utterances.

    python scripts/eval_streaming.py --num 20
    python scripts/eval_streaming.py --num 20 --model ours --checkpoint exp/rnnt_small/epoch9.pt

Decoding runs chunk-by-chunk through exactly the same backend the live demo uses, so the
numbers reflect streaming behaviour rather than a full-context pass.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data.dataset import load_manifest_cached
from stream_demo import OursBackend, PretrainedBackend, iter_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pretrained", "ours"], default="pretrained")
    parser.add_argument("--config", default="configs/rnnt_small.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--subset", default="dev-clean")
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--root", default="../data/LibriSpeech")
    args = parser.parse_args()

    import jiwer

    backend = (OursBackend(args.config, args.checkpoint) if args.model == "ours"
               else PretrainedBackend())

    utterances = load_manifest_cached(args.root, [args.subset], "exp/manifests")[:args.num]
    refs, hyps, audio_seconds, compute_seconds = [], [], 0.0, 0.0

    for i, utt in enumerate(utterances, 1):
        backend.reset()
        transcript = ""
        t0 = time.perf_counter()
        for chunk in iter_file(utt.path, backend.chunk_samples, backend.sample_rate, False):
            transcript = backend.process(chunk)
        compute_seconds += time.perf_counter() - t0
        audio_seconds += utt.duration
        refs.append(utt.text.upper())
        hyps.append(transcript.strip().upper())
        print(f"\r  {i}/{len(utterances)}", end="", flush=True)

    wer = jiwer.wer(refs, hyps)
    print(f"\n\nmodel   : {backend.name}")
    print(f"subset  : {args.subset}, {len(refs)} utterances, {audio_seconds:.0f}s audio")
    print(f"WER     : {wer * 100:.2f}%   (streaming, chunk-by-chunk)")
    print(f"RTF     : {compute_seconds / audio_seconds:.3f}")
    print(f"\nexample:\n  ref: {refs[0][:90]}\n  hyp: {hyps[0][:90]}")


if __name__ == "__main__":
    main()
