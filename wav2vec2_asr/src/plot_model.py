"""Draw the wav2vec2 architecture, with every number read off the loaded model.

Nothing here is hardcoded: layer shapes, strides, parameter counts, the total
downsampling factor and the receptive field are all introspected from the checkpoint, so
the diagram cannot drift from the model it describes.

    python src/plot_model.py
    python src/plot_model.py --bundle WAV2VEC2_ASR_LARGE_960H
"""

import argparse
import os
import sys

import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def describe(model, sample_rate: int) -> dict:
    """Pull the architecture out of the model object."""
    convs = [layer.conv for layer in model.feature_extractor.conv_layers]
    transformer = model.encoder.transformer
    layer0 = transformer.layers[0]

    # Total stride: how many input samples one output frame advances by.
    total_stride = 1
    for conv in convs:
        total_stride *= conv.stride[0]

    # Receptive field, folded back through the stack: r = (r - 1) * stride + kernel.
    receptive_field = 1
    for conv in reversed(convs):
        receptive_field = (receptive_field - 1) * conv.stride[0] + conv.kernel_size[0]

    def count(module):
        return sum(p.numel() for p in module.parameters())

    return {
        "convs": [(c.in_channels, c.out_channels, c.kernel_size[0], c.stride[0])
                  for c in convs],
        "conv_out": convs[-1].out_channels,
        "total_stride": total_stride,
        "receptive_field": receptive_field,
        "stride_ms": total_stride / sample_rate * 1000,
        "receptive_ms": receptive_field / sample_rate * 1000,
        "frame_rate": sample_rate / total_stride,
        "proj_in": model.encoder.feature_projection.projection.in_features,
        "embed_dim": layer0.attention.embed_dim,
        "num_heads": layer0.attention.num_heads,
        "ffn_dim": layer0.feed_forward.intermediate_dense.out_features,
        "num_layers": len(transformer.layers),
        "vocab": model.aux.out_features,
        "params_fe": count(model.feature_extractor),
        "params_enc": count(model.encoder),
        "params_head": count(model.aux),
        "params_total": count(model),
    }


def draw(spec: dict, bundle_name: str, sample_rate: int, out_path: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 13))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16.1)
    ax.axis("off")

    def box(y, height, label, sub, colour, edge, x=1.6, width=5.2, fontsize=11):
        """Title and caption are placed proportionally so short boxes never clip."""
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08",
                                    facecolor=colour, edgecolor=edge, linewidth=1.6))
        if not sub:
            ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold")
            return
        ax.text(x + width / 2, y + height * 0.70, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")
        ax.text(x + width / 2, y + height * 0.32, sub, ha="center", va="center",
                fontsize=8.6, color="#333333", linespacing=1.5)

    def arrow(y0, y1, x=4.2):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                     mutation_scale=17, linewidth=1.5, color="#555555"))

    def note(y, text, colour="#444444"):
        ax.text(7.2, y, text, ha="left", va="center", fontsize=8.6, color=colour,
                linespacing=1.6)

    ax.text(5, 15.75, "wav2vec 2.0 for ASR", ha="center", fontsize=16, fontweight="bold")
    ax.text(5, 15.35, f"{bundle_name}   ·   {spec['params_total'] / 1e6:.1f}M parameters",
            ha="center", fontsize=10, color="#555555")

    # --- input -----------------------------------------------------------------
    box(0.35, 0.95, "Raw waveform", f"16 kHz mono, no hand-designed features",
        "#eef2f7", "#7d8fa8")
    note(0.82, f"(1, N) samples")
    arrow(1.30, 1.85)

    # --- convolutional feature encoder -----------------------------------------
    conv_lines = "   ".join(
        f"k{k}/s{s}" for (_, _, k, s) in spec["convs"])
    box(1.85, 2.5, f"Convolutional feature encoder  ({len(spec['convs'])} layers)",
        f"Conv1d 1→{spec['conv_out']}, then {spec['conv_out']}→{spec['conv_out']}\n"
        f"{conv_lines}\n"
        f"GELU + group norm",
        "#fdece8", "#e8674a")
    note(3.55, f"{spec['params_fe'] / 1e6:.1f}M params")
    note(3.15, f"stride {spec['total_stride']}x  →  {spec['stride_ms']:.0f} ms hop\n"
               f"receptive field {spec['receptive_field']} samples\n"
               f"= {spec['receptive_ms']:.0f} ms per frame", "#b03a22")
    note(2.35, f"→ {spec['frame_rate']:.0f} frames/s", "#b03a22")
    arrow(4.35, 4.9)

    # --- feature projection ------------------------------------------------------
    box(4.9, 1.0, "Feature projection",
        f"Linear {spec['proj_in']} → {spec['embed_dim']}", "#f6f0e8", "#c98b4a")
    note(5.40, f"(B, T, {spec['embed_dim']})")
    arrow(5.90, 6.3)

    # --- transformer -------------------------------------------------------------
    box(6.3, 3.3, f"Transformer encoder  × {spec['num_layers']}",
        f"self-attention: {spec['num_heads']} heads, d = {spec['embed_dim']}\n"
        f"feed-forward: {spec['embed_dim']} → {spec['ffn_dim']} → {spec['embed_dim']}\n"
        f"GELU, layer norm, residual\n\n"
        f"convolutional positional embedding\n(relative, not sinusoidal)",
        "#e9f5ec", "#4aa06a")
    note(9.0, f"{spec['params_enc'] / 1e6:.1f}M params")
    note(8.1, "self-supervised\npretraining happens\nhere: masked latent\nprediction with a\n"
              "contrastive loss\nover quantized units", "#2f7a4d")
    arrow(9.6, 10.15)

    # --- CTC head ----------------------------------------------------------------
    box(10.15, 0.95, "CTC head",
        f"Linear {spec['embed_dim']} → {spec['vocab']}", "#e8eef8", "#4a6fa5")
    note(10.62, f"{spec['params_head'] / 1e3:.0f}K params\nadded at fine-tuning", "#2f4f7a")
    arrow(11.10, 11.65)

    # --- output ------------------------------------------------------------------
    box(11.65, 0.95, f"Character logits",
        f"{spec['vocab']} labels @ {spec['frame_rate']:.0f} frames/s",
        "#eef2f7", "#7d8fa8")
    note(12.12, f"(B, T, {spec['vocab']})")
    arrow(12.60, 13.15)

    box(13.15, 0.95, "Greedy / beam CTC decoding",
        "collapse repeats → strip blanks", "#f4f4f4", "#999999")
    arrow(14.10, 14.55)
    ax.text(4.2, 14.78, "\"I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT\"",
            ha="center", fontsize=10, style="italic", color="#222222")

    # The two-stage story, which is the whole point of wav2vec2.
    ax.text(0.15, 7.9, "pretrained\non unlabeled\naudio", ha="left", va="center",
            fontsize=9, color="#2f7a4d", fontweight="bold", linespacing=1.6)
    ax.text(0.15, 10.6, "added +\nfine-tuned\non labels", ha="left", va="center",
            fontsize=9, color="#2f4f7a", fontweight="bold", linespacing=1.6)
    ax.plot([1.35, 1.35], [1.85, 9.6], color="#4aa06a", linewidth=2.5, alpha=0.55)
    ax.plot([1.35, 1.35], [10.15, 11.1], color="#4a6fa5", linewidth=2.5, alpha=0.55)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="WAV2VEC2_ASR_BASE_960H")
    args = parser.parse_args()

    bundle = getattr(torchaudio.pipelines, args.bundle)
    model = bundle.get_model().eval()
    spec = describe(model, bundle.sample_rate)

    print(f"{args.bundle}: {spec['params_total'] / 1e6:.1f}M params")
    print(f"  feature encoder : {len(spec['convs'])} conv layers, "
          f"stride {spec['total_stride']}x -> {spec['stride_ms']:.0f} ms hop, "
          f"receptive field {spec['receptive_ms']:.0f} ms")
    print(f"  transformer     : {spec['num_layers']} layers, d={spec['embed_dim']}, "
          f"{spec['num_heads']} heads, ffn {spec['ffn_dim']}")
    print(f"  CTC head        : {spec['embed_dim']} -> {spec['vocab']}")

    # Cross-check the derived frame rate against a real forward pass.
    with torch.inference_mode():
        emission, _ = model(torch.zeros(1, bundle.sample_rate))
    print(f"  verified        : 1 s of audio -> {emission.size(1)} frames "
          f"(derived {spec['frame_rate']:.0f})")

    out = os.path.join(ROOT, "plots", f"{args.bundle.lower()}_architecture.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    draw(spec, args.bundle, bundle.sample_rate, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
