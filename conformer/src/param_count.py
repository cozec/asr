"""Compare parameter counts of the S/M/L configs against paper Table 1.

    python src/param_count.py
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import build_model

# Paper Table 1, "Num Params (M)".
PAPER = {"conformer_s": 10.3, "conformer_m": 30.7, "conformer_l": 118.8}
VOCAB_SIZE = 1024          # paper §3.2: 1k word-piece model


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"{'model':<14}{'encoder':>10}{'decoder+joint':>15}{'total':>10}"
          f"{'paper':>9}{'delta':>9}")
    print("-" * 67)

    for name, expected in PAPER.items():
        with open(os.path.join(here, "configs", f"{name}.yaml")) as fh:
            cfg = yaml.safe_load(fh)
        # Table 1 counts the transducer model, so force that head regardless of config.
        cfg["model"]["head"] = "transducer"
        model = build_model(cfg, VOCAB_SIZE)

        encoder = sum(p.numel() for p in model.encoder.parameters())
        total = sum(p.numel() for p in model.parameters())
        delta = (total / 1e6 - expected) / expected * 100
        print(f"{name:<14}{encoder / 1e6:>9.2f}M{(total - encoder) / 1e6:>14.2f}M"
              f"{total / 1e6:>9.2f}M{expected:>8.1f}M{delta:>8.1f}%")


if __name__ == "__main__":
    main()
