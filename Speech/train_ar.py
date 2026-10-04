"""
Stage 2: trains the autoregressive anomaly detection model (models/ar_model.py) on
S(i,·) sequences extracted from REAL video only (extract_features.py's `ar_train`
output). Soft cross-entropy against the sync model's own delay distribution (paper
Eq. 5) — the best-performing feature/loss combination in Table 3.

Matches the paper's stated optimizer (Sec. 4.1/A.5): Adam, lr 1e-3, weight decay
1e-6, linear warmup then cosine decay, batch 16, dropout 0.1 (set on the model).

Meant to run on Kaggle after extract_features.py has populated data/av_features/;
also runs on CPU for a short --steps smoke run.

Usage:
    python train_ar.py --steps 8000
"""

import argparse
import itertools
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import AR_BATCH_SIZE, AR_CKPT_DIR, AR_LR, AR_WARMUP_STEPS, AR_WEIGHT_DECAY, SEQUENCE_LENGTH
from datasets import FeatureSequenceDataset
from models.ar_model import ARModel, step_losses
from train_sync import get_device


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def make_lr_lambda(warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train the autoregressive anomaly detection model.")
    parser.add_argument("--split", type=str, default="ar_train")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=AR_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--lr", type=float, default=AR_LR)
    parser.add_argument("--weight-decay", type=float, default=AR_WEIGHT_DECAY)
    parser.add_argument("--warmup-steps", type=int, default=AR_WARMUP_STEPS)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=AR_CKPT_DIR / "ar_model.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    ds = FeatureSequenceDataset(args.split, seq_len=args.seq_len, full_sequence=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    stream = _cycle(loader)

    model = ARModel().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from {args.resume}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(args.warmup_steps, args.steps))

    args.out.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        seq = next(stream).to(device)  # (B, seq_len, DELAY_BINS)

        losses = step_losses(model, seq)
        loss = losses.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if step % args.log_every == 0 or step == args.steps:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"[ar {step}/{args.steps}] loss={loss.item():.4f} lr={lr_now:.2e} elapsed={elapsed:.0f}s")

        if step % args.save_every == 0 or step == args.steps:
            torch.save(model.state_dict(), args.out)

    print(f"Saved AR model to {args.out}")


if __name__ == "__main__":
    main()
