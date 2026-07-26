"""Compare CTC decoding strategies on wav2vec2 emissions.

The tutorial decodes greedily -- "simply pick up the best hypothesis at each time step"
with no language model. That is the cheapest option and the weakest. This measures what
beam search buys, and separates the two things people conflate:

  1. **search**: greedy takes the argmax per frame; beam search keeps N hypotheses and
     can recover when the locally-best path is globally wrong.
  2. **the language model**: a lexicon constrains output to real words, and an n-gram LM
     scores word sequences. This is usually where most of the gain comes from.

    python scripts/compare_decoders.py --num 50
    python scripts/compare_decoders.py --num 50 --with-lm        # adds the 4-gram LM (~3 GB download)
    python scripts/compare_decoders.py --num 50 --beam-sizes 1 5 50 500
"""

import argparse
import json
import os
import sys
import time

import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from decoder import GreedyCTCDecoder, to_words
from evaluate import scan_librispeech
from pipeline_demo import load_audio


def compute_emissions(bundle, items, device):
    """Run the acoustic model once; every decoder then re-scores the same emissions."""
    model = bundle.get_model().to(device).eval()
    emissions, refs, audio_seconds, acoustic_seconds = [], [], 0.0, 0.0
    for i, (_, path, text) in enumerate(items, 1):
        waveform = load_audio(path, bundle.sample_rate).to(device)
        audio_seconds += waveform.size(1) / bundle.sample_rate
        t0 = time.perf_counter()
        with torch.inference_mode():
            emission, _ = model(waveform)
        acoustic_seconds += time.perf_counter() - t0
        emissions.append(emission.cpu())
        refs.append(text.upper())
        print(f"\r  acoustic model: {i}/{len(items)}", end="", flush=True)
    print()
    return emissions, refs, audio_seconds, acoustic_seconds


def run_greedy(emissions, labels):
    decoder = GreedyCTCDecoder(labels=labels)
    t0 = time.perf_counter()
    hyps = [to_words(decoder(e[0])).upper() for e in emissions]
    return hyps, time.perf_counter() - t0


def run_beam(emissions, decoder, use_words: bool):
    t0 = time.perf_counter()
    hyps = []
    for e in emissions:
        best = decoder(e)[0][0]
        if use_words:
            hyps.append(" ".join(best.words).upper())
        else:
            # Lexicon-free: the decoder returns tokens, so join and split on the
            # word-separator exactly as greedy decoding does.
            hyps.append(to_words("".join(decoder.idxs_to_tokens(best.tokens))).upper())
    return hyps, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="WAV2VEC2_ASR_BASE_960H")
    parser.add_argument("--subset", default="test-clean")
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--beam-sizes", type=int, nargs="+", default=[5, 50])
    parser.add_argument("--with-lm", action="store_true",
                        help="add lexicon + 4-gram LM beam search (~3 GB download)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import jiwer
    from torchaudio.models.decoder import ctc_decoder

    device = torch.device(args.device)
    bundle = getattr(torchaudio.pipelines, args.bundle)
    labels = bundle.get_labels()
    items = scan_librispeech(args.subset, args.num)
    print(f"{len(items)} utterances from {args.subset}, bundle {args.bundle}\n")

    emissions, refs, audio_seconds, acoustic_seconds = compute_emissions(bundle, items, device)
    print(f"acoustic model: {acoustic_seconds:.1f}s for {audio_seconds:.0f}s audio "
          f"(RTF {acoustic_seconds / audio_seconds:.3f})\n")

    rows = []

    hyps, seconds = run_greedy(emissions, labels)
    rows.append(("greedy", "-", "-", jiwer.wer(refs, hyps) * 100, seconds, hyps[0]))

    for beam_size in args.beam_sizes:
        decoder = ctc_decoder(lexicon=None, tokens=list(labels), blank_token="-",
                              sil_token="|", beam_size=beam_size)
        hyps, seconds = run_beam(emissions, decoder, use_words=False)
        rows.append((f"beam {beam_size}", "no", "no",
                     jiwer.wer(refs, hyps) * 100, seconds, hyps[0]))

    if args.with_lm:
        from torchaudio.models.decoder import download_pretrained_files

        print("fetching librispeech-4-gram lexicon + LM ...")
        files = download_pretrained_files("librispeech-4-gram")
        for beam_size in args.beam_sizes:
            decoder = ctc_decoder(lexicon=files.lexicon, tokens=files.tokens,
                                  lm=files.lm, beam_size=beam_size,
                                  lm_weight=3.23, word_score=-0.26)
            hyps, seconds = run_beam(emissions, decoder, use_words=True)
            rows.append((f"beam {beam_size}", "yes", "4-gram",
                         jiwer.wer(refs, hyps) * 100, seconds, hyps[0]))

    print(f"\n{'decoder':<14}{'lexicon':>9}{'LM':>9}{'WER%':>9}{'decode s':>11}{'x greedy':>10}")
    print("-" * 62)
    greedy_seconds = rows[0][4]
    for name, lexicon, lm, wer, seconds, _ in rows:
        print(f"{name:<14}{lexicon:>9}{lm:>9}{wer:>9.2f}{seconds:>11.2f}"
              f"{seconds / greedy_seconds:>9.0f}x")

    print(f"\nref     : {refs[0][:95]}")
    for name, _, _, _, _, hyp in rows:
        print(f"{name:<8}: {hyp[:95]}")

    out = os.path.join(ROOT, "results", f"decoders_{args.subset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"bundle": args.bundle, "subset": args.subset,
                   "num_utterances": len(items),
                   "acoustic_rtf": acoustic_seconds / audio_seconds,
                   "rows": [{"decoder": n, "lexicon": lx, "lm": lm, "wer": w,
                             "decode_seconds": s} for n, lx, lm, w, s, _ in rows]},
                  fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
