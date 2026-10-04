"""
Stage 1: trains the audio-visual synchronization model (models/sync_model.py) on real
GRID video, following the paper's two-phase curriculum (Appendix A.5):

  Phase 1 ("cross_video"): negatives are other videos' audio in the same batch — an
    easy "does this audio belong to this video at all" contrastive task.
  Phase 2 ("window"): negatives are the SAME video's audio at other delays in
    [-tau, tau] — a hard "which exact frame does this audio align to" task, and
    literally the paper's Eq. 1 softmax window used later for feature extraction.

This is meant to run on a Kaggle GPU (see kaggle/KAGGLE_INSTRUCTIONS.md) — it also
runs on CPU for a short --phase1-steps/--phase2-steps smoke run, just slowly.

Usage (Kaggle, full run):
    python train_sync.py --phase1-steps 3000 --phase2-steps 6000

Usage (quick local check that it trains at all, once you've preprocessed a couple
of speakers):
    python train_sync.py --phase1-steps 20 --phase2-steps 20 --batch-size-phase1 4 --batch-size-phase2 4
"""

import argparse
import itertools
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import SYNC_BATCH_PHASE1, SYNC_BATCH_PHASE2, SYNC_CKPT_DIR, SYNC_LR
from datasets import SyncDataset, sync_collate_cross_video, sync_collate_window
from models.sync_model import SyncModel


def get_device(requested):
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def run_phase1(model, opt, device, split, steps, batch_size, num_workers, log_every):
    ds = SyncDataset(split, mode="cross_video")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, collate_fn=sync_collate_cross_video,
    )
    stream = _cycle(loader)
    target = None

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        video_clip, audio_clip = next(stream)
        video_clip, audio_clip = video_clip.to(device), audio_clip.to(device)
        if target is None or target.numel() != video_clip.size(0):
            target = torch.arange(video_clip.size(0), device=device)

        logits = model.cross_video_logits(video_clip, audio_clip)
        loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps:
            acc = (logits.argmax(dim=1) == target).float().mean().item()
            elapsed = time.time() - t0
            print(f"[phase1 {step}/{steps}] loss={loss.item():.4f} acc={acc:.3f} elapsed={elapsed:.0f}s")


def run_phase2(model, opt, device, split, steps, batch_size, num_workers, log_every):
    ds = SyncDataset(split, mode="window")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, collate_fn=sync_collate_window,
    )
    stream = _cycle(loader)

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        video_clip, audio_window, target = next(stream)
        video_clip, audio_window, target = video_clip.to(device), audio_window.to(device), target.to(device)

        logits = model.window_logits(video_clip, audio_window)
        loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == steps:
            acc = (logits.argmax(dim=1) == target).float().mean().item()
            elapsed = time.time() - t0
            print(f"[phase2 {step}/{steps}] loss={loss.item():.4f} acc={acc:.3f} elapsed={elapsed:.0f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the audio-visual synchronization model.")
    parser.add_argument("--split", type=str, default="sync_train")
    parser.add_argument("--phase1-steps", type=int, default=3000)
    parser.add_argument("--phase2-steps", type=int, default=6000)
    parser.add_argument("--batch-size-phase1", type=int, default=SYNC_BATCH_PHASE1)
    parser.add_argument("--batch-size-phase2", type=int, default=SYNC_BATCH_PHASE2)
    parser.add_argument("--lr", type=float, default=SYNC_LR)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu; default auto-detects.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--out", type=Path, default=SYNC_CKPT_DIR / "sync_model.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    model = SyncModel().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from {args.resume}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.phase1_steps > 0:
        run_phase1(model, opt, device, args.split, args.phase1_steps,
                   args.batch_size_phase1, args.num_workers, args.log_every)
    if args.phase2_steps > 0:
        run_phase2(model, opt, device, args.split, args.phase2_steps,
                   args.batch_size_phase2, args.num_workers, args.log_every)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved sync model to {args.out}")


if __name__ == "__main__":
    main()
