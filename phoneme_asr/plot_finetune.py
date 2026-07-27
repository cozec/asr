"""Plot the TIMIT phoneme fine-tune curves from ../wav2vec2_fine_tune.

Three panels: PER against the published ladder, the loss curves, and the two together
so the plateau-then-collapse is visible in one place.

    python plot_finetune.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FT = os.path.join(ROOT, "..", "wav2vec2_fine_tune", "results")

# TIMIT test PER, no LM, from wav2vec 2.0 Table 3 (Baevski et al. 2020).
# (name, PER, label y-offset) -- offsets keep 18.0/17.2 and 14.9/14.7 from colliding.
LADDER = [("CNN + TD-filterbanks", 18.0, 0.55), ("PASE+", 17.2, -0.35),
          ("Li-GRU + fMLLR", 14.9, 0.42), ("wav2vec", 14.7, -0.42),
          ("vq-wav2vec", 11.6, 0.0), ("wav2vec 2.0 LARGE", 8.3, 0.0)]


def main():
    ev = json.load(open(os.path.join(FT, "phoneme_timit.json")))["curve"]
    tr = json.load(open(os.path.join(FT, "phoneme_trainloss.json")))["train_loss"]

    epochs = [r["epoch"] for r in ev]
    per = [r["per"] * 100 for r in ev]
    eval_loss = [r["eval_loss"] for r in ev]
    best = min(ev, key=lambda r: r["per"])

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    # --- PER against the ladder ------------------------------------------------
    for name, v, dy in LADDER:
        ax1.axhline(v, color="#bbbbbb", linewidth=0.8, linestyle=(0, (5, 4)))
        ax1.annotate(f"{name}  {v}", (20.4, v + dy), fontsize=7.5, color="#777777",
                     va="center", annotation_clip=False)
    ax1.plot(epochs, per, "o-", color="#e8674a", linewidth=2, markersize=4,
             label="this run (wav2vec2-base)")
    ax1.plot(best["epoch"], best["per"] * 100, "*", color="#4aa06a", markersize=18,
             zorder=5, label=f"best {best['per']*100:.2f}% @ epoch {int(best['epoch'])}")
    ax1.set_yscale("log")
    ax1.set_yticks([8, 10, 12, 15, 20, 30, 50, 100])
    ax1.set_yticklabels(["8", "10", "12", "15", "20", "30", "50", "100"])
    ax1.set_ylabel("phone error rate (%, log scale)")
    ax1.set_title("TIMIT phoneme recognition: wav2vec2-base fine-tune vs the published ladder")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right")

    # --- losses ------------------------------------------------------------------
    ax2.plot([p["epoch"] for p in tr], [p["loss"] for p in tr], color="#9aa7b5",
             linewidth=1.1, label="train loss (SpecAugment active)")
    ax2.plot(epochs, eval_loss, "o-", color="#4a6fa5", linewidth=2, markersize=4,
             label="eval loss")
    ax2.axvline(1000 * 32 / 3696, color="#c98b4a", linewidth=1.4, linestyle=(0, (4, 3)))
    ax2.annotate("warmup ends\n(step 1000)", (1000 * 32 / 3696 + 0.2, 40),
                 fontsize=8, color="#c98b4a")
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("CTC loss (log scale)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper right")

    out = os.path.join(ROOT, "plots", "timit_phoneme_finetune.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote plots/timit_phoneme_finetune.png")
    print(f"best {best['per']*100:.2f}% @ epoch {int(best['epoch'])}, "
          f"final {ev[-1]['per']*100:.2f}%")


if __name__ == "__main__":
    main()
