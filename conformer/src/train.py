"""Train a Conformer on LibriSpeech.

Usage:
    python src/train.py --config configs/conformer_s.yaml
    python src/train.py --config configs/conformer_s.yaml --train-sets dev-clean --max-steps 50
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conformer import ConformerCTC, ConformerTransducer
from data.dataset import (
    LibriSpeechDataset,
    ctc_collate,
    load_manifest_cached,
    rnnt_collate,
)
from data.tokenizer import CharTokenizer, Tokenizer


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TransformerLRSchedule:
    """Transformer schedule (paper §3.2): 10k warmup, peak lr = lr_scale / sqrt(d).

    Linear warmup to the peak, then inverse-square-root decay.
    """

    def __init__(self, optimizer, encoder_dim: int, warmup_steps: int,
                 lr_scale: float = 0.05):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.peak_lr = lr_scale / (encoder_dim ** 0.5)
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            lr = self.peak_lr * self.step_num / self.warmup_steps
        else:
            lr = self.peak_lr * (self.warmup_steps / self.step_num) ** 0.5
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {"step_num": self.step_num}

    def load_state_dict(self, state: dict) -> None:
        self.step_num = state["step_num"]


def build_model(cfg: dict, vocab_size: int):
    m = cfg["model"]
    common = dict(
        num_classes=vocab_size,
        input_dim=m["input_dim"],
        encoder_dim=m["encoder_dim"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        ffn_expansion=m["ffn_expansion"],
        conv_expansion=m["conv_expansion"],
        conv_kernel_size=m["conv_kernel_size"],
        dropout=m["dropout"],
        half_step_residual=m["half_step_residual"],
    )
    if m["head"] == "ctc":
        return ConformerCTC(**common)
    if m["head"] == "transducer":
        return ConformerTransducer(
            decoder_dim=m["decoder_dim"], decoder_layers=m["decoder_layers"],
            joint_dim=m["joint_dim"], **common,
        )
    raise ValueError(f"unknown head: {m['head']}")


def load_tokenizer(cfg: dict):
    path = cfg["data"]["tokenizer"]
    if path and os.path.exists(path):
        return Tokenizer(path)
    print(f"[warn] tokenizer {path!r} not found -- falling back to characters. "
          f"Run scripts/train_tokenizer.py for the paper's 1k word-piece vocabulary.")
    return CharTokenizer()


def rnnt_loss_fn(logits, targets, logit_lengths, target_lengths, blank_id):
    import torchaudio

    return torchaudio.functional.rnnt_loss(
        logits=logits.float(),
        targets=targets.int(),
        logit_lengths=logit_lengths.int(),
        target_lengths=target_lengths.int(),
        blank=blank_id,
        reduction="mean",
    )


# Neither aten::_ctc_loss nor torchaudio's rnnt_loss has an MPS kernel, so on Apple
# Silicon the loss runs on CPU. Autograd copies the gradient back to the MPS tensors,
# leaving the whole model on the GPU -- only the loss itself falls back.
LOSS_DEVICE_FALLBACK = torch.device("cpu")


def needs_cpu_loss(device: torch.device) -> bool:
    return device.type == "mps"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-sets", nargs="+", default=None,
                        help="override the config's training subsets")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop early; useful for smoke tests")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None,
                        help="override the config's DataLoader worker count")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.log_every:
        cfg["train"]["log_every"] = args.log_every

    torch.manual_seed(cfg["train"]["seed"])
    device = pick_device(args.device)
    head = cfg["model"]["head"]
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = load_tokenizer(cfg)
    blank_id = tokenizer.blank_id

    train_sets = args.train_sets or cfg["data"]["train_sets"]
    print(f"loading {train_sets} from {cfg['data']['root']} ...")
    manifest_dir = os.path.join(os.path.dirname(save_dir.rstrip("/")) or "exp", "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    utterances = load_manifest_cached(cfg["data"]["root"], train_sets, manifest_dir)
    total_hours = sum(u.duration for u in utterances) / 3600
    print(f"  {len(utterances)} utterances, {total_hours:.1f} h")

    feature_cfg = {k: cfg["data"][k] for k in ("num_mel_bins", "frame_length", "frame_shift")}
    dataset = LibriSpeechDataset(
        utterances, tokenizer, feature_cfg, cfg["specaug"], cfg["data"]["max_duration"],
    )
    collate = ctc_collate if head == "ctc" else rnnt_collate(blank_id)
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=(args.num_workers if args.num_workers is not None
                     else cfg["train"]["num_workers"]),
        collate_fn=collate,
        drop_last=True,
    )

    model = build_model(cfg, tokenizer.vocab_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg['model']['name']} ({head}), {num_params / 1e6:.2f}M params, "
          f"device={device}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
        betas=tuple(cfg["train"]["betas"]),
        eps=float(cfg["train"]["eps"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = TransformerLRSchedule(
        optimizer, cfg["model"]["encoder_dim"], cfg["train"]["warmup_steps"],
        cfg["train"]["lr_scale"],
    )
    ctc_loss = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
    loss_device = LOSS_DEVICE_FALLBACK if needs_cpu_loss(device) else device
    if loss_device != device:
        print(f"note: {device.type} has no CTC/RNN-T kernel -- computing the loss on "
              f"{loss_device.type} (model stays on {device.type})")

    start_epoch, global_step = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, global_step = state["epoch"] + 1, state["step"]
        print(f"resumed from {args.resume} at epoch {start_epoch}, step {global_step}")

    accum = cfg["train"]["accum_grad"]
    stop = False

    for epoch in range(start_epoch, cfg["train"]["max_epochs"]):
        model.train()
        running_loss, num_batches, epoch_start = 0.0, 0, time.time()

        for i, batch in enumerate(loader):
            if head == "ctc":
                feats, feat_lens, targets, target_lens, _ = batch
                feats, feat_lens = feats.to(device), feat_lens.to(device)
                log_probs, out_lens = model(feats, feat_lens)
                # CTCLoss wants (T, B, V).
                loss = ctc_loss(
                    log_probs.transpose(0, 1).to(loss_device), targets.to(loss_device),
                    out_lens.cpu(), target_lens,
                )
            else:
                feats, feat_lens, targets, decoder_in, target_lens, _ = batch
                feats, feat_lens = feats.to(device), feat_lens.to(device)
                logits, out_lens = model(feats, feat_lens, decoder_in.to(device),
                                         target_lens.to(device))
                loss = rnnt_loss_fn(logits.to(loss_device), targets.to(loss_device),
                                    out_lens.to(loss_device),
                                    target_lens.to(loss_device), blank_id)

            (loss / accum).backward()
            running_loss += loss.item()
            num_batches += 1

            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg["train"]["grad_clip"])
                lr = scheduler.step()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg["train"]["log_every"] == 0:
                    avg = running_loss / max(num_batches, 1)
                    print(f"epoch {epoch} step {global_step} loss {avg:.4f} "
                          f"lr {lr:.3e} ({time.time() - epoch_start:.0f}s)", flush=True)

                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        ckpt = os.path.join(save_dir, f"epoch{epoch}.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "config": cfg,
        }, ckpt)
        print(f"epoch {epoch} done: loss {running_loss / max(num_batches, 1):.4f}, "
              f"saved {ckpt}", flush=True)

        if stop:
            print(f"stopping at step {global_step} (--max-steps)")
            break


if __name__ == "__main__":
    main()
