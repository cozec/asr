"""Reproduction of torchaudio's "Forced Alignment with Wav2Vec2" tutorial.

https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

Faithful to the tutorial's algorithm and its five figures, with two changes forced by
current torchaudio:

* `torchaudio.utils.download_asset` no longer exists, so the asset is fetched from the
  URL it used to wrap.
* `torchaudio.load` routes through TorchCodec from 2.9, which is in requirements.txt.

Every figure is written to plots/ rather than shown, so the script runs headless.

    python forced_alignment_tutorial.py
"""

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")                       # write files, never open a window
import matplotlib.pyplot as plt
import torch
import torchaudio

ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(ROOT, "plots")

ASSET = "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
ASSET_URL = f"https://download.pytorch.org/torchaudio/tutorial-assets/{ASSET}"
TRANSCRIPT = "|I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def download_asset(url: str = ASSET_URL) -> str:
    """Stand-in for the removed torchaudio.utils.download_asset."""
    import requests

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    path = os.path.join(ROOT, "data", os.path.basename(url))
    if not os.path.exists(path):
        print(f"downloading {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    return path


def get_trellis(emission, tokens, blank_id=0):
    """Alignment probability lattice (tutorial section 'Generate alignment probability').

    Two transitions per frame: stay on the token, which emits a blank, or advance to the
    next token, which emits that token.
    """
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis


def backtrack(trellis, emission, tokens, blank_id=0):
    """Walk the lattice backwards along the most likely path."""
    t, j = trellis.size(0) - 1, trellis.size(1) - 1
    path = [Point(j, t, emission[t, blank_id].exp().item())]

    while j > 0:
        assert t > 0
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        t -= 1
        if changed > stayed:
            j -= 1
        path.append(Point(j, t, (p_change if changed > stayed else p_stay).exp().item()))

    # Leading frames belong to the first token.
    while t > 0:
        path.append(Point(j, t - 1, emission[t - 1, blank_id].exp().item()))
        t -= 1

    return path[::-1]


def merge_repeats(path, transcript):
    """Collapse consecutive frames on the same token into one labelled segment."""
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(Segment(transcript[path[i1].token_index],
                                path[i1].time_index, path[i2 - 1].time_index + 1, score))
        i1 = i2
    return segments


def merge_words(segments, separator="|"):
    """Group character segments into words on the separator token."""
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(s.score * s.length for s in segs) / sum(s.length for s in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


# --------------------------------------------------------------------- figures

def save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote plots/{name}")


def plot_emission(emission, labels):
    fig, ax = plt.subplots(figsize=(12, 6))
    img = ax.imshow(emission.T, aspect="auto", interpolation="nearest")
    ax.set_title("Frame-wise class probability")
    ax.set_xlabel("Time (frame)")
    ax.set_ylabel("Labels")
    # The tutorial leaves these as indices; real labels make the plot readable.
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([("- (blank)" if l == "-" else "| (space)" if l == "|" else l)
                        for l in labels], fontsize=7, fontfamily="monospace")
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    save(fig, "1_emission.png")


def plot_trellis(trellis):
    fig, ax = plt.subplots(figsize=(12, 5))
    img = ax.imshow(trellis.T, origin="lower", aspect="auto")
    ax.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5), color="white")
    ax.annotate("+ Inf", (trellis.size(0) - trellis.size(1) / 5, trellis.size(1) / 3),
                color="white")
    ax.set_title("Trellis: alignment probability lattice")
    ax.set_xlabel("Time (frame)")
    ax.set_ylabel("Token index")
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    save(fig, "2_trellis.png")


def plot_trellis_with_path(trellis, path):
    trellis_with_path = trellis.clone()
    for p in path:
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(trellis_with_path.T, origin="lower", aspect="auto")
    ax.set_title("The path found by backtracking")
    ax.set_xlabel("Time (frame)")
    ax.set_ylabel("Token index")
    save(fig, "3_trellis_with_path.png")


def plot_trellis_with_segments(trellis, segments, transcript, path):
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start:seg.end, i] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")
    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

    ax2.set_title("Label probability with and without repetition")
    xs, hs, ws = [], [], []
    for seg in segments:
        if seg.label != "|":
            xs.append((seg.end + seg.start) / 2 + 0.4)
            hs.append(seg.score)
            ws.append(seg.end - seg.start)
            ax2.annotate(seg.label, (seg.start + 0.8, -0.07))
    ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in path:
        if transcript[p.token_index] != "|":
            xs.append(p.time_index + 1)
            hs.append(p.score)
    ax2.bar(xs, hs, width=0.5, alpha=0.5)
    ax2.axhline(0, color="black")
    ax2.grid(True, axis="y")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Time (frame)")
    save(fig, "4_trellis_with_segments.png")


def plot_alignments(trellis, segments, word_segments, waveform, sample_rate):
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start:seg.end, i] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(13, 9))
    ax1.imshow(trellis_with_path.T, origin="lower", aspect="auto")
    ax1.set_facecolor("lightgray")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Alignment path with word boundaries")
    for word in word_segments:
        ax1.axvspan(word.start - 0.5, word.end - 0.5, edgecolor="white", facecolor="none")
    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i - 0.7), size="small")
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 3), size="small")

    ratio = waveform.size(0) / sample_rate / trellis.size(0)
    ax2.specgram(waveform, Fs=sample_rate)
    for word in word_segments:
        x0, x1 = ratio * word.start, ratio * word.end
        ax2.axvspan(x0, x1, facecolor="none", edgecolor="white", hatch="/")
        ax2.annotate(f"{word.score:.2f}", (x0, sample_rate * 0.51), annotation_clip=False)
    for seg in segments:
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, sample_rate * 0.55),
                         annotation_clip=False)
    ax2.set_xlabel("time [second]")
    ax2.set_yticks([])
    save(fig, "5_alignments.png")


def save_word_audio(waveform, word_segments, trellis_size, sample_rate):
    """The tutorial's final step: cut each word out of the audio."""
    out = os.path.join(ROOT, "plots", "words")
    os.makedirs(out, exist_ok=True)
    ratio = waveform.size(0) / trellis_size
    for i, word in enumerate(word_segments):
        x0, x1 = int(ratio * word.start), int(ratio * word.end)
        seg = waveform[x0:x1].unsqueeze(0)
        torchaudio.save(os.path.join(out, f"{i:02d}_{word.label}.wav"), seg, sample_rate)
    print(f"  wrote {len(word_segments)} word clips to plots/words/")


def main():
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()

    path_wav = download_asset()
    waveform, sample_rate = torchaudio.load(path_wav)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        sample_rate = bundle.sample_rate

    with torch.inference_mode():
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)
    emission = emissions[0].cpu().detach()

    print(f"audio     : {os.path.basename(path_wav)} "
          f"({waveform.size(1) / sample_rate:.2f}s @ {sample_rate} Hz)")
    print(f"emission  : {tuple(emission.shape)}")
    print(f"transcript: {TRANSCRIPT}\n")

    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in TRANSCRIPT]

    trellis = get_trellis(emission, tokens)
    path = backtrack(trellis, emission, tokens)
    segments = merge_repeats(path, TRANSCRIPT)
    word_segments = merge_words(segments)

    print("figures:")
    plot_emission(emission, labels)
    plot_trellis(trellis)
    plot_trellis_with_path(trellis, path)
    plot_trellis_with_segments(trellis, segments, TRANSCRIPT, path)
    plot_alignments(trellis, segments, word_segments, waveform[0], sample_rate)
    save_word_audio(waveform[0], word_segments, trellis.size(0), sample_rate)

    ratio = waveform.size(1) / sample_rate / trellis.size(0)
    print("\nWord segments:")
    for word in word_segments:
        print(f"{word}  ->  {word.start * ratio:5.2f}s - {word.end * ratio:5.2f}s")


if __name__ == "__main__":
    main()
