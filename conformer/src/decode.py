"""Decode LibriSpeech with a trained Conformer and report WER.

Usage:
    python src/decode.py --config configs/conformer_s.yaml \
        --checkpoint exp/conformer_s/epoch9.pt --test-sets test-clean test-other
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import LibriSpeechDataset, ctc_collate, load_manifest_cached
from train import build_model, load_tokenizer, pick_device


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor,
                      blank_id: int = 0) -> list[list[int]]:
    """Collapse repeats then strip blanks, per utterance."""
    best = log_probs.argmax(dim=-1)
    hypotheses = []
    for seq, length in zip(best, lengths):
        seq = seq[:length].tolist()
        collapsed, previous = [], None
        for token in seq:
            if token != previous and token != blank_id:
                collapsed.append(token)
            previous = token
        hypotheses.append(collapsed)
    return hypotheses


@torch.no_grad()
def transducer_greedy_decode(model, features, feature_lengths, blank_id: int = 0,
                             max_symbols_per_step: int = 5) -> list[list[int]]:
    """Standard greedy transducer search: emit until blank, then advance time."""
    encoded, out_lengths = model.encoder(features, feature_lengths)
    hypotheses = []
    for b in range(encoded.size(0)):
        hidden, tokens = None, []
        token = torch.full((1, 1), blank_id, dtype=torch.long, device=encoded.device)
        decoder_out, hidden = model.decoder(token, hidden)
        for t in range(int(out_lengths[b])):
            for _ in range(max_symbols_per_step):
                logits = model.joint(encoded[b:b + 1, t:t + 1], decoder_out)
                pred = int(logits[0, 0, 0].argmax())
                if pred == blank_id:
                    break
                tokens.append(pred)
                token = torch.tensor([[pred]], device=encoded.device)
                decoder_out, hidden = model.decoder(token, hidden)
        hypotheses.append(tokens)
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-sets", nargs="+", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    import jiwer

    device = pick_device(args.device)
    tokenizer = load_tokenizer(cfg)
    head = cfg["model"]["head"]

    model = build_model(cfg, tokenizer.vocab_size).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    feature_cfg = {k: cfg["data"][k] for k in ("num_mel_bins", "frame_length", "frame_shift")}
    os.makedirs(args.output_dir, exist_ok=True)

    manifest_dir = os.path.join("exp", "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    for subset in (args.test_sets or cfg["data"]["test_sets"]):
        utterances = load_manifest_cached(cfg["data"]["root"], [subset], manifest_dir)
        dataset = LibriSpeechDataset(utterances, tokenizer, feature_cfg, specaug_cfg=None)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=ctc_collate)

        references = {u.utt_id: u.text for u in utterances}
        hypotheses = {}

        with torch.no_grad():
            for feats, feat_lens, _, _, utt_ids in loader:
                feats, feat_lens = feats.to(device), feat_lens.to(device)
                if head == "ctc":
                    log_probs, out_lens = model(feats, feat_lens)
                    decoded = ctc_greedy_decode(log_probs, out_lens, tokenizer.blank_id)
                else:
                    decoded = transducer_greedy_decode(model, feats, feat_lens,
                                                       tokenizer.blank_id)
                for utt_id, tokens in zip(utt_ids, decoded):
                    hypotheses[utt_id] = tokenizer.decode(tokens)

        ordered = sorted(hypotheses)
        wer = jiwer.wer([references[u] for u in ordered], [hypotheses[u] for u in ordered])
        print(f"{subset}: WER {wer * 100:.2f}%  ({len(ordered)} utterances)")

        out_path = os.path.join(args.output_dir, f"{subset}.hyp.txt")
        with open(out_path, "w") as fh:
            for utt_id in ordered:
                fh.write(f"{utt_id}\t{hypotheses[utt_id]}\n")
        print(f"  hypotheses -> {out_path}")


if __name__ == "__main__":
    main()
