"""Visualize Gentle's forced-alignment output.

Gentle emits JSON with word- and phone-level timings; this turns that into the same
three views used for the wav2vec2 aligner, so the two are directly comparable.

    python plot_alignment.py                       # uses gentle/examples/
    python plot_alignment.py --json a.json --audio a.mp3
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(ROOT, "plots")


def load(json_path, audio_path):
    result = json.load(open(json_path))
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    return result, audio.mean(axis=1), sample_rate


def save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote plots/{name}")


def plot_words_over_spectrogram(result, audio, sample_rate, t0=6.0, t1=26.0):
    """Word boundaries over the spectrogram -- the check that alignment is real."""
    words = [w for w in result["words"]
             if w["case"] == "success" and w["end"] > t0 and w["start"] < t1]

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2]})

    times = np.arange(len(audio)) / sample_rate
    mask = (times >= t0) & (times <= t1)
    ax1.plot(times[mask], audio[mask], linewidth=0.4, color="#4a6fa5")
    ax1.set_ylabel("amplitude")
    ax1.set_title(f"Gentle word alignment, {t0:.0f}-{t1:.0f}s")
    for w in words:
        ax1.axvspan(w["start"], w["end"], alpha=0.12, color="#e8674a")

    ax2.specgram(audio, Fs=sample_rate, NFFT=512, noverlap=384, cmap="viridis")
    for i, w in enumerate(words):
        ax2.axvspan(w["start"], w["end"], facecolor="none", edgecolor="white",
                    linewidth=0.9, hatch="//" if i % 2 else None)
        ax2.annotate(w["word"], ((w["start"] + w["end"]) / 2, sample_rate * 0.46),
                     ha="center", fontsize=7.5, rotation=45, annotation_clip=False)
    ax2.set_xlim(t0, t1)
    ax2.set_ylim(0, sample_rate * 0.4)
    ax2.set_xlabel("time [second]")
    ax2.set_ylabel("frequency [Hz]")
    save(fig, "1_words_over_spectrogram.png")


def plot_phone_detail(result, audio, sample_rate, num_words=6):
    """Phone-level timings -- what Gentle gives beyond word boundaries.

    Six words covers "I am sitting in a room" and stops before the 3s pause that
    follows, which would otherwise leave most of the panel empty.
    """
    words = [w for w in result["words"] if w["case"] == "success"][:num_words]
    t0, t1 = words[0]["start"] - 0.1, words[-1]["end"] + 0.1

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.specgram(audio, Fs=sample_rate, NFFT=512, noverlap=384, cmap="viridis")
    ax1.set_xlim(t0, t1)
    ax1.set_ylim(0, sample_rate * 0.4)
    ax1.set_ylabel("frequency [Hz]")
    ax1.set_title("Phone-level alignment")

    # Phones carry durations only, so absolute times accumulate from the word start.
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for wi, w in enumerate(words):
        ax1.axvline(w["start"], color="white", linewidth=1.2)
        ax1.annotate(w["word"], (w["start"], sample_rate * 0.42), fontsize=9,
                     color="white", annotation_clip=False)
        t = w["start"]
        for pi, ph in enumerate(w["phones"]):
            dur = ph["duration"]
            ax2.barh(0, dur, left=t, height=0.6,
                     color=colors[(wi * 3 + pi) % 20], edgecolor="black", linewidth=0.4)
            if dur > 0.035:
                # Strip Gentle's _B/_I/_E/_S position tags for readability.
                ax2.annotate(ph["phone"].split("_")[0], (t + dur / 2, 0),
                             ha="center", va="center", fontsize=7)
            t += dur
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("time [second]")
    ax2.set_ylabel("phones")
    save(fig, "2_phone_detail.png")


def plot_coverage(result, audio, sample_rate):
    """Which words landed, which did not, across the whole recording."""
    words = result["words"]
    ok = [w for w in words if w["case"] == "success"]
    bad = [w for w in words if w["case"] != "success"]
    duration = len(audio) / sample_rate

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(14, 6),
                                   gridspec_kw={"height_ratios": [1, 1]})

    # Every aligned word as a bar on the timeline.
    for w in ok:
        ax1.barh(0, w["end"] - w["start"], left=w["start"], height=0.5,
                 color="#4aa06a", edgecolor="white", linewidth=0.3)
    ax1.set_xlim(0, duration)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.set_xlabel("time [second]")
    ax1.set_title(f"Coverage: {len(ok)} of {len(words)} words aligned "
                  f"({len(bad)} not found in audio)")

    # Word durations: how long each aligned word lasted.
    durs = [w["end"] - w["start"] for w in ok]
    ax2.hist(durs, bins=30, color="#4a6fa5", edgecolor="white")
    ax2.axvline(np.median(durs), color="#e8674a", linewidth=2,
                label=f"median {np.median(durs):.2f}s")
    ax2.set_xlabel("word duration [second]")
    ax2.set_ylabel("count")
    ax2.legend()
    ax2.set_title("Aligned word durations")

    fig.tight_layout()
    save(fig, "3_coverage.png")

    return ok, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=os.path.join(ROOT, "gentle/examples/lucier_alignment.json"))
    parser.add_argument("--audio", default=os.path.join(ROOT, "gentle/examples/data/lucier.mp3"))
    args = parser.parse_args()

    result, audio, sample_rate = load(args.json, args.audio)
    print(f"audio : {len(audio)/sample_rate:.1f}s @ {sample_rate} Hz")
    print(f"words : {len(result['words'])}\n")

    print("figures:")
    plot_words_over_spectrogram(result, audio, sample_rate)
    plot_phone_detail(result, audio, sample_rate)
    ok, bad = plot_coverage(result, audio, sample_rate)

    print(f"\naligned  : {len(ok)}/{len(result['words'])}")
    print(f"span     : {ok[0]['start']:.2f}s - {ok[-1]['end']:.2f}s")
    print(f"unaligned: {[w['word'] for w in bad]}")
    print("\nfirst words:")
    for w in ok[:6]:
        phones = ",".join(p["phone"].split("_")[0] for p in w["phones"])
        print(f"  {w['start']:6.2f} - {w['end']:6.2f}  {w['word']:<10} [{phones}]")


if __name__ == "__main__":
    main()
