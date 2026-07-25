"""Train the RNN-T on LibriSpeech.

    python src/train.py --config configs/rnnt_small.yaml
    python src/train.py --config configs/rnnt_small.yaml --train-sets dev-clean --max-steps 30 --log-every 5

Structure follows the conformer project's trainer (same LR schedule, manifest caching and
MPS loss fallback); the loss is RNN-T rather than CTC.
"""

import argparse
import os
import sys
import time

import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import LibriSpeechDataset, load_manifest_cached, rnnt_collate
from data.tokenizer import CharTokenizer, Tokenizer
from rnnt import build_rnnt


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TransformerLRSchedule:
    """Linear warmup to `lr_scale / sqrt(dim)`, then inverse-square-root decay."""

    def __init__(self, optimizer, dim: int, warmup_steps: int, lr_scale: float = 0.05):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.peak_lr = lr_scale / (dim ** 0.5)
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


def load_tokenizer(cfg: dict):
    path = cfg["data"]["tokenizer"]
    if path and os.path.exists(path):
        return Tokenizer(path)
    print(f"[warn] tokenizer {path!r} not found -- falling back to characters. Note that "
          f"Ghodsi et al. §3.4 warns the stateless predictor regresses badly on "
          f"graphemes; run scripts/train_tokenizer.py for the wordpiece vocabulary.")
    return CharTokenizer()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-sets", nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
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
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = load_tokenizer(cfg)
    blank_id = tokenizer.blank_id

    train_sets = args.train_sets or cfg["data"]["train_sets"]
    print(f"loading {train_sets} from {cfg['data']['root']} ...")
    manifest_dir = os.path.join(os.path.dirname(save_dir.rstrip("/")) or "exp", "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    utterances = load_manifest_cached(cfg["data"]["root"], train_sets, manifest_dir)
    print(f"  {len(utterances)} utterances, "
          f"{sum(u.duration for u in utterances) / 3600:.1f} h")

    feature_cfg = {k: cfg["data"][k] for k in ("num_mel_bins", "frame_length", "frame_shift")}
    feature_cfg["frame_stack"] = cfg["model"]["frame_stack"]
    feature_cfg["frame_stride"] = cfg["model"]["frame_stride"]

    dataset = LibriSpeechDataset(utterances, tokenizer, feature_cfg, cfg["specaug"],
                                 cfg["data"]["max_duration"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=(args.num_workers if args.num_workers is not None
                     else cfg["train"]["num_workers"]),
        collate_fn=rnnt_collate(blank_id),
        drop_last=True,
    )

    model = build_rnnt(cfg, tokenizer.vocab_size).to(device)
    print(f"model: {cfg['model']['name']}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, device={device}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0, betas=tuple(cfg["train"]["betas"]),
        eps=float(cfg["train"]["eps"]), weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = TransformerLRSchedule(optimizer, cfg["model"]["proj_dim"],
                                      cfg["train"]["warmup_steps"], cfg["train"]["lr_scale"])

    # torchaudio's rnnt_loss has no MPS kernel; run it on CPU and let autograd copy the
    # gradient back, keeping the model itself on the GPU.
    loss_device = torch.device("cpu") if device.type == "mps" else device
    if loss_device != device:
        print(f"note: {device.type} has no RNN-T kernel -- computing the loss on cpu "
              f"(model stays on {device.type})")

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
            feats, feat_lens, targets, decoder_in, target_lens, _ = batch
            feats, feat_lens = feats.to(device), feat_lens.to(device)

            logits, src_lens, _, _ = model(feats, feat_lens, decoder_in.to(device),
                                           (target_lens + 1).to(device))
            loss = torchaudio.functional.rnnt_loss(
                logits=logits.to(loss_device).float(),
                targets=targets.to(loss_device).int(),
                logit_lengths=src_lens.to(loss_device).int(),
                target_lengths=target_lens.to(loss_device).int(),
                blank=blank_id,
                reduction="mean",
            )

            (loss / accum).backward()
            running_loss += loss.item()
            num_batches += 1

            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                lr = scheduler.step()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg["train"]["log_every"] == 0:
                    print(f"epoch {epoch} step {global_step} "
                          f"loss {running_loss / max(num_batches, 1):.4f} lr {lr:.3e} "
                          f"({time.time() - epoch_start:.0f}s)", flush=True)
                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        ckpt = os.path.join(save_dir, f"epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "step": global_step, "config": cfg}, ckpt)
        print(f"epoch {epoch} done: loss {running_loss / max(num_batches, 1):.4f}, "
              f"saved {ckpt}", flush=True)
        if stop:
            print(f"stopping at step {global_step} (--max-steps)")
            break


if __name__ == "__main__":
    main()
