"""Compare four forced aligners on the same audio.

Gentle (Kaldi), MFA (Kaldi), BFA (CUPE+CTC) and the wav2vec2 CTC aligner all produce
word timings for data/demo_audio.wav. This puts them on one axis so the disagreements
are visible rather than buried in four JSON files.

    python compare_aligners.py
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))

# wav2vec2 numbers come from ../wave2vec2_forced_alignment/forced_alignment_demo.py,
# which aligns this same clip with a CTC trellis.
WAV2VEC2 = [("hello", 0.02, 0.40), ("world", 0.55, 0.87), ("today", 0.95, 1.33)]


def parse_textgrid(path):
    """Minimal Praat TextGrid reader: {tier_name: [(start, end, label), ...]}."""
    txt = open(path, encoding="utf-8").read()
    tiers = {}
    for tm in re.finditer(r'name = "([^"]+)"(.*?)(?=name = "|\Z)', txt, re.S):
        intervals = []
        for m in re.finditer(
                r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "([^"]*)"',
                tm.group(2)):
            if m.group(3).strip():
                intervals.append((float(m.group(1)), float(m.group(2)), m.group(3)))
        tiers[tm.group(1)] = intervals
    return tiers


def load_all():
    """Return {aligner: {"words": [...], "phones": [...]}} with times in seconds."""
    out = {}

    gentle = json.load(open(os.path.join(ROOT, "results/demo_audio_alignment.json")))
    words, phones = [], []
    for w in gentle["words"]:
        if w["case"] != "success":
            continue
        words.append((w["word"], w["start"], w["end"]))
        t = w["start"]
        for p in w["phones"]:
            phones.append((p["phone"].split("_")[0], t, t + p["duration"]))
            t += p["duration"]
    out["Gentle (Kaldi)"] = {"words": words, "phones": phones}

    mfa = parse_textgrid(os.path.join(ROOT, "results/mfa_out/demo_audio.TextGrid"))
    out["MFA (Kaldi)"] = {
        "words": [(l, a, b) for a, b, l in mfa["words"]],
        # Strip ARPAbet stress digits so the labels line up with the others.
        "phones": [(re.sub(r"\d", "", l), a, b) for a, b, l in mfa["phones"]],
    }

    bfa = json.load(open(os.path.join(ROOT, "results/demo_bfa.json")))["segments"][0]
    out["BFA (CUPE+CTC)"] = {
        "words": [(w["word"], w["start_ms"] / 1000, w["end_ms"] / 1000)
                  for w in bfa["words_ts"]],
        "phones": [(p["ipa_label"], p["start_ms"] / 1000, p["end_ms"] / 1000)
                   for p in bfa["phoneme_ts"]],
    }

    out["wav2vec2 (CTC)"] = {"words": WAV2VEC2, "phones": []}
    return out


def plot(all_results, audio, sample_rate, out_path):
    names = list(all_results)
    colors = {"Gentle (Kaldi)": "#4aa06a", "MFA (Kaldi)": "#4a6fa5",
              "BFA (CUPE+CTC)": "#e8674a", "wav2vec2 (CTC)": "#c98b4a"}

    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2] + [1] * len(names)})

    axes[0].specgram(audio, Fs=sample_rate, NFFT=512, noverlap=384, cmap="viridis")
    axes[0].set_ylim(0, sample_rate * 0.4)
    axes[0].set_ylabel("Hz")
    axes[0].set_title('Four aligners on the same clip: "hello world today"')

    for ax, name in zip(axes[1:], names):
        for label, a, b in all_results[name]["words"]:
            ax.barh(0, b - a, left=a, height=0.55, color=colors[name],
                    edgecolor="black", linewidth=0.6)
            ax.annotate(label, ((a + b) / 2, 0), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        for _, a, b in all_results[name]["phones"]:
            ax.plot([a, a], [-0.42, -0.3], color="black", linewidth=0.7)
        ax.set_ylim(-0.5, 0.4)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8.5)

    axes[-1].set_xlabel("time [second]")
    axes[-1].set_xlim(0, len(audio) / sample_rate)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out_path, ROOT)}")


def main():
    results = load_all()
    audio, sample_rate = sf.read(os.path.join(ROOT, "data/demo_audio.wav"),
                                 dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)

    print(f"{'aligner':<18}{'word':<8}{'start':>7}{'end':>7}{'phones':>8}")
    print("-" * 50)
    for name, r in results.items():
        for i, (label, a, b) in enumerate(r["words"]):
            n = sum(1 for _, pa, _ in r["phones"] if a - 1e-6 <= pa < b)
            print(f"{name if i == 0 else '':<18}{label:<8}{a:>7.2f}{b:>7.2f}"
                  f"{(n if r['phones'] else '-'):>8}")

    # Pairwise boundary agreement against Gentle, the reference here.
    ref = {w: (a, b) for w, a, b in results["Gentle (Kaldi)"]["words"]}
    print(f"\n{'vs Gentle':<18}{'start dev':>11}{'end dev':>10}  (mean abs, ms)")
    print("-" * 50)
    for name, r in results.items():
        if name == "Gentle (Kaldi)":
            continue
        ds = [abs(a - ref[w][0]) for w, a, _ in r["words"] if w in ref]
        de = [abs(b - ref[w][1]) for w, _, b in r["words"] if w in ref]
        print(f"{name:<18}{np.mean(ds)*1000:>10.0f}{np.mean(de)*1000:>10.0f}")

    print("\nfigures:")
    os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)
    plot(results, audio, sample_rate, os.path.join(ROOT, "plots/4_aligner_comparison.png"))


if __name__ == "__main__":
    main()
