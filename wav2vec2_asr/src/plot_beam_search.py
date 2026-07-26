"""Draw CTC prefix beam search unrolling over our own emissions.

Same idea as the classic figure in Hannun's "Sequence Modeling with CTC", but every
node, edge and probability here comes from running an actual prefix beam search on the
wav2vec2 emissions for the tutorial audio -- not from a schematic.

The default window (frames 33-36) is chosen because it is where this utterance is
genuinely uncertain: '|' 0.85 vs blank 0.15, then blank 0.61 vs '|' 0.39. Most frames
are ~1.0 confident and would produce a tree with nothing to show.

    python src/plot_beam_search.py
    python src/plot_beam_search.py --start-frame 44 --steps 5
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_demo import TUTORIAL_ASSET, download_asset, load_audio

NEG_INF = -float("inf")


def logsumexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi, lo = max(a, b), min(a, b)
    import math

    return hi + math.log1p(math.exp(lo - hi))


def step(beam, frame_logprobs, labels, blank: int, topk: int, beam_width: int):
    """One timestep of CTC prefix beam search.

    beam maps prefix -> (log p ending in blank, log p ending in a non-blank).
    Returns the next beam plus the edges taken, for drawing.
    """
    candidates = frame_logprobs.topk(topk)
    proposals = list(zip(candidates.indices.tolist(), candidates.values.tolist()))

    nxt = defaultdict(lambda: [NEG_INF, NEG_INF])
    edges = []                                   # (src_prefix, label_id, dst_prefix)

    for prefix, (p_b, p_nb) in beam.items():
        total = logsumexp(p_b, p_nb)
        for label, logp in proposals:
            if label == blank:
                # Blank never extends the string: the prefix stays put.
                nxt[prefix][0] = logsumexp(nxt[prefix][0], total + logp)
                edges.append((prefix, label, prefix))
                continue

            char = labels[label]
            if prefix and char == prefix[-1]:
                # Repeat of the last character. Two distinct paths:
                #   - no blank in between  -> collapses back into the same prefix
                #   - a blank in between   -> genuinely doubles the character
                nxt[prefix][1] = logsumexp(nxt[prefix][1], p_nb + logp)
                edges.append((prefix, label, prefix))
                extended = prefix + char
                nxt[extended][1] = logsumexp(nxt[extended][1], p_b + logp)
                edges.append((prefix, label, extended))
            else:
                extended = prefix + char
                nxt[extended][1] = logsumexp(nxt[extended][1], total + logp)
                edges.append((prefix, label, extended))

    ranked = sorted(nxt.items(), key=lambda kv: -logsumexp(*kv[1]))[:beam_width]
    return dict(ranked), proposals, edges


def run(logprobs, labels, blank: int, start_frame: int, steps: int, topk: int,
        beam_width: int):
    """Warm the beam up to `start_frame`, then record `steps` timesteps."""
    beam = {"": [0.0, NEG_INF]}
    for t in range(start_frame):
        beam, _, _ = step(beam, logprobs[t], labels, blank, topk, beam_width)

    trace = []
    for t in range(start_frame, start_frame + steps):
        before = beam
        beam, proposals, edges = step(beam, logprobs[t], labels, blank, topk, beam_width)
        trace.append({"frame": t, "before": before, "after": beam,
                      "proposals": proposals, "edges": edges})
    return trace


def show(prefix: str) -> str:
    return "λ" if prefix == "" else prefix


def draw(trace, labels, blank: int, out_path: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    steps = len(trace)
    col = 3.4                                     # horizontal span of one timestep
    fig, ax = plt.subplots(figsize=(3.6 * steps + 2.4, 8.6))
    ax.set_xlim(-0.9, col * steps + 1.6)
    ax.set_ylim(-0.9, 8.3)
    ax.axis("off")

    blue, red, grey = "#5b9bd5", "#e0685c", "#c9c9c9"

    def prefix_node(x, y, prefix, radius=0.19):
        """Draw a prefix as a chain of circles, one per emitted character."""
        chars = list(prefix) if prefix else ["λ"]
        for i, ch in enumerate(chars):
            cx = x + i * (radius * 2.35)
            ax.add_patch(Circle((cx, y), radius, facecolor=blue, edgecolor="#3d6f9e",
                                linewidth=1.1, zorder=3))
            ax.text(cx, y, ch, ha="center", va="center", fontsize=9.5,
                    color="white", fontweight="bold", zorder=4)
        return x + (len(chars) - 1) * (radius * 2.35)

    def ext_node(x, y, text, kept, radius=0.19):
        ax.add_patch(Circle((x, y), radius, facecolor=red if kept else grey,
                            edgecolor="#b04a3f" if kept else "#a8a8a8",
                            linewidth=1.1, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold", zorder=4)

    # Vertical slots for each timestep's hypotheses. Kept well below the header band
    # so multi-character prefixes never collide with the column labels.
    def slots(n, top=6.0, bottom=0.7):
        if n == 1:
            return [(top + bottom) / 2]
        gap = (top - bottom) / (n - 1)
        return [top - i * gap for i in range(n)]

    hyp_pos = {}                                  # (t, prefix) -> (x_right, y)
    ext_pos = {}                                  # (t, prefix, label) -> (x, y)

    for t, entry in enumerate(trace):
        x_h = t * col
        befores = list(entry["before"])
        ys = slots(len(befores))

        ax.text(x_h, 7.85, f"T = {t + 1}", fontsize=12, fontweight="bold", va="bottom")
        ax.text(x_h, 7.55, f"frame {entry['frame']}", fontsize=8.5, color="#666666",
                va="bottom")
        ax.text(x_h, 6.85, "current\nhypotheses", fontsize=8, color="#888888",
                va="bottom", linespacing=1.4)
        ax.text(x_h + 1.75, 6.85, "proposed\nextensions", fontsize=8, color="#888888",
                va="bottom", ha="center", linespacing=1.4)

        for prefix, y in zip(befores, ys):
            right = prefix_node(x_h, y, prefix)
            hyp_pos[(t, prefix)] = (right, y)

            # Fan out this hypothesis' proposals.
            n = len(entry["proposals"])
            spread = 0.42
            for j, (label, logp) in enumerate(entry["proposals"]):
                ey = y + (n - 1) / 2 * spread - j * spread
                ex = x_h + 1.75
                text = "ε" if label == blank else labels[label]
                dsts = {d for (s, l, d) in entry["edges"] if s == prefix and l == label}
                kept = any(d in entry["after"] for d in dsts)
                ax.plot([right + 0.22, ex - 0.22], [y, ey], color="#444444",
                        linewidth=1.0, zorder=1)
                ext_node(ex, ey, text, kept)
                ax.text(ex + 0.30, ey, f"{torch.tensor(logp).exp():.2f}", fontsize=7,
                        color="#777777", va="center")
                ext_pos[(t, prefix, label)] = (ex, ey)

    # Final column: the hypotheses the last step produced.
    x_final = steps * col
    finals = list(trace[-1]["after"])
    ys = slots(len(finals))
    ax.text(x_final, 7.85, f"T = {steps + 1}", fontsize=12, fontweight="bold", va="bottom")
    ax.text(x_final, 6.85, "current\nhypotheses", fontsize=8, color="#888888",
            va="bottom", linespacing=1.4)
    for prefix, y in zip(finals, ys):
        prefix_node(x_final, y, prefix)
        hyp_pos[(steps, prefix)] = (x_final, y)

    # Dashed edges: each surviving extension flows into the next timestep's hypothesis.
    merge_targets = defaultdict(list)
    for t, entry in enumerate(trace):
        for (src, label, dst) in entry["edges"]:
            if dst not in entry["after"] or (t, src, label) not in ext_pos:
                continue
            if (t + 1, dst) not in hyp_pos:
                continue
            x0, y0 = ext_pos[(t, src, label)]
            x1, y1 = hyp_pos[(t + 1, dst)]
            ax.add_patch(FancyArrowPatch((x0 + 0.22, y0), (x1 - 0.24, y1),
                                         arrowstyle="-", linestyle=(0, (4, 3)),
                                         linewidth=0.8, color="#bdbdbd", alpha=0.75,
                                         zorder=0))
            merge_targets[(t + 1, dst)].append((src, label))

    # Call out a genuine merge, the point of the figure.
    for (t, dst), sources in merge_targets.items():
        if len(sources) > 1 and t == steps:
            x, y = hyp_pos[(t, dst)]
            ax.annotate(f"{len(sources)} extensions merge\ninto the same prefix",
                        xy=(x - 0.26, y - 0.22), xytext=(x + 0.35, y - 1.35),
                        fontsize=8, color="#444444", ha="center", linespacing=1.4,
                        arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.9))
            break

    ax.text(-0.75, -0.6, "ε = blank    '|' = word separator    "
                         "numbers are per-frame probabilities    "
                         "red = kept in beam, grey = pruned",
            fontsize=8, color="#666666")
    ax.set_title("CTC prefix beam search on wav2vec2 emissions  "
                 "(tutorial audio, \"I HAD THAT ...\")",
                 fontsize=13, fontweight="bold", pad=16)

    fig.tight_layout()
    fig.savefig(out_path, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="WAV2VEC2_ASR_BASE_960H")
    parser.add_argument("--audio", default=None)
    parser.add_argument("--start-frame", type=int, default=33)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--topk", type=int, default=3, help="proposed extensions per step")
    parser.add_argument("--beam-width", type=int, default=3)
    args = parser.parse_args()

    bundle = getattr(torchaudio.pipelines, args.bundle)
    model = bundle.get_model().eval()
    labels = bundle.get_labels()
    blank = 0

    waveform = load_audio(args.audio or download_asset(), bundle.sample_rate)
    with torch.inference_mode():
        emission, _ = model(waveform)
    logprobs = emission[0].log_softmax(-1)

    trace = run(logprobs, labels, blank, args.start_frame, args.steps, args.topk,
                args.beam_width)

    for t, entry in enumerate(trace):
        props = "  ".join(
            f"{'ε' if i == blank else labels[i]}:{torch.tensor(v).exp():.2f}"
            for i, v in entry["proposals"])
        print(f"T={t + 1} (frame {entry['frame']})  propose[{props}]")
        for prefix in entry["after"]:
            print(f"           -> {show(prefix)!r}")

    out = os.path.join(ROOT, "plots", f"{args.bundle.lower()}_beam_search.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    draw(trace, labels, blank, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
