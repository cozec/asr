"""Reproduction of torchaudio's "Speech Recognition with Wav2Vec2" tutorial.

https://docs.pytorch.org/audio/2.8/tutorials/speech_recognition_pipeline_tutorial.html

Walks the tutorial's steps in order -- load the pipeline, extract per-layer transformer
features, classify them into label logits, then greedily CTC-decode -- and writes the
three figures it plots into plots/.

    python src/pipeline_demo.py                      # the tutorial's own audio sample
    python src/pipeline_demo.py --audio some.flac    # any file
    python src/pipeline_demo.py --bundle WAV2VEC2_ASR_BASE_100H
"""

import argparse
import os
import sys

import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decoder import GreedyCTCDecoder, to_words

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The tutorial fetches this via torchaudio.utils.download_asset, which no longer exists
# in torchaudio 2.11; the underlying URL is still served, so we fetch it directly.
TUTORIAL_ASSET = "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
TUTORIAL_ASSET_URL = f"https://download.pytorch.org/torchaudio/tutorial-assets/{TUTORIAL_ASSET}"


def download_asset(url: str = TUTORIAL_ASSET_URL, dest_dir: str = None) -> str:
    """Stand-in for the removed torchaudio.utils.download_asset."""
    import requests

    dest_dir = dest_dir or os.path.join(ROOT, "data")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(url))
    if not os.path.exists(path):
        print(f"downloading {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(response.content)
    return path


def load_audio(path: str, target_sample_rate: int) -> torch.Tensor:
    """Load and resample to the bundle's rate (tutorial step 2).

    soundfile rather than torchaudio.load: 2.11 routes load() through torchcodec, which
    this project does not depend on.
    """
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data).T                     # (channels, samples)
    waveform = waveform.mean(0, keepdim=True)               # to mono
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        print(f"resampled {sample_rate} -> {target_sample_rate} Hz")
    return waveform


def plot_waveform(waveform, sample_rate, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3))
    times = torch.arange(waveform.size(1)) / sample_rate
    ax.plot(times, waveform[0], linewidth=0.5)
    ax.set(xlabel="time (s)", ylabel="amplitude", title="Waveform")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_features(features, path):
    """One heatmap per transformer layer (tutorial step 3)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(features), 1, figsize=(14, 2.0 * len(features)))
    for i, feats in enumerate(features):
        axes[i].imshow(feats[0].cpu().T, interpolation="nearest", aspect="auto")
        axes[i].set_title(f"Feature from transformer layer {i + 1}", fontsize=9)
        axes[i].set_ylabel("dim")
        if i < len(features) - 1:
            axes[i].set_xticks([])
    axes[-1].set_xlabel("frame (time axis)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_emission(emission, path, labels=None):
    """Emission heatmap, with the actual character labels on the y-axis.

    wav2vec2's vocabulary is 29 CTC labels: '-' is the blank and '|' the word
    separator; the rest are letters and the apostrophe. Naming them makes the plot
    readable -- the bright top row is blank dominating, as CTC output does.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(emission[0].cpu().T, interpolation="nearest", aspect="auto")
    ax.set(title="Classification result (emission)", xlabel="frame (time axis)")

    if labels is not None:
        pretty = []
        for label in labels:
            if label == "-":
                pretty.append("- (blank)")
            elif label == "|":
                pretty.append("| (space)")
            else:
                pretty.append(label)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(pretty, fontsize=8, fontfamily="monospace")
        ax.set_ylabel("CTC label")
    else:
        ax.set_ylabel("class")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="WAV2VEC2_ASR_BASE_960H")
    parser.add_argument("--audio", default=None, help="defaults to the tutorial's sample")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    torch.random.manual_seed(0)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Step 1: the pipeline bundles weights, expected sample rate and label set.
    bundle = getattr(torchaudio.pipelines, args.bundle)
    print(f"bundle      : {args.bundle}")
    print(f"sample rate : {bundle.sample_rate}")
    labels = bundle.get_labels()
    print(f"labels ({len(labels)}) : {labels}")

    model = bundle.get_model().to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"| device {device}\n")

    # Step 2: audio in, resampled to the bundle's rate.
    path = args.audio or download_asset()
    waveform = load_audio(path, bundle.sample_rate).to(device)
    duration = waveform.size(1) / bundle.sample_rate
    print(f"audio       : {path}  ({duration:.2f}s)")

    # Step 3: per-transformer-layer features.
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)
    print(f"features    : {len(features)} transformer layers, "
          f"each {tuple(features[0].shape)}")

    # Step 4: classify features into label logits ("emission").
    with torch.inference_mode():
        emission, _ = model(waveform)
    print(f"emission    : {tuple(emission.shape)}  "
          f"({emission.size(1) / duration:.1f} frames/s)")

    # Step 5-6: greedy CTC decode.
    decoder = GreedyCTCDecoder(labels=labels)
    transcript = decoder(emission[0])
    print(f"\ntranscript  : {transcript}")
    print(f"as words    : {to_words(transcript)}")

    if not args.no_plots:
        plots_dir = os.path.join(ROOT, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        tag = args.bundle.lower()
        plot_waveform(waveform.cpu(), bundle.sample_rate,
                      os.path.join(plots_dir, f"{tag}_waveform.png"))
        plot_features(features, os.path.join(plots_dir, f"{tag}_features.png"))
        plot_emission(emission, os.path.join(plots_dir, f"{tag}_emission.png"), labels)
        print(f"\nplots       : {plots_dir}/{tag}_{{waveform,features,emission}}.png")


if __name__ == "__main__":
    main()
