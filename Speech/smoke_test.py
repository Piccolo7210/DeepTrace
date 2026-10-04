"""
CPU-only shape/gradient smoke test — no GRID data, no extra dependencies beyond
what's already in requirements.txt (torch/torchvision/numpy). Builds random tensors
matching every shape the real pipeline produces and runs one forward+backward step
of the sync model (both curriculum phases) and the AR model, asserting finite losses
and correct output shapes.

The point is to catch architecture/shape bugs for free before spending Kaggle GPU
time on the real thing. It says nothing about whether the models will actually learn
anything useful from real audio-visual data — only that the code runs.

Usage:
    python smoke_test.py
"""

import time

import torch
import torch.nn.functional as F

from config import AUDIO_SR, CROP_SIZE, DELAY_BINS, SEQUENCE_LENGTH, VISUAL_CLIP_LEN
from models.ar_model import ARModel, step_losses
from models.sync_model import SyncModel

AUDIO_CLIP_SAMPLES = int(round(0.2 * AUDIO_SR))
BATCH = 2


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        raise AssertionError(name)


def test_sync_model():
    print("SyncModel")
    torch.manual_seed(0)
    model = SyncModel()

    video_clip = torch.rand(BATCH, VISUAL_CLIP_LEN, 3, CROP_SIZE, CROP_SIZE)
    audio_clip = torch.randn(BATCH, AUDIO_CLIP_SAMPLES) * 0.1
    audio_window = torch.randn(BATCH, DELAY_BINS, AUDIO_CLIP_SAMPLES) * 0.1

    t0 = time.time()
    logits_cv = model.cross_video_logits(video_clip, audio_clip)
    check("cross_video_logits shape", tuple(logits_cv.shape) == (BATCH, BATCH))
    loss_cv = F.cross_entropy(logits_cv, torch.arange(BATCH))
    check("cross_video loss finite", torch.isfinite(loss_cv).item())
    loss_cv.backward()
    check("cross_video grads finite", all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    model.zero_grad()

    logits_win = model.window_logits(video_clip, audio_window)
    check("window_logits shape", tuple(logits_win.shape) == (BATCH, DELAY_BINS))
    target = torch.full((BATCH,), DELAY_BINS // 2, dtype=torch.long)
    loss_win = F.cross_entropy(logits_win, target)
    check("window loss finite", torch.isfinite(loss_win).item())
    loss_win.backward()
    check("window grads finite", all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    print(f"  ({time.time() - t0:.1f}s)")


def test_ar_model():
    print("ARModel")
    torch.manual_seed(0)
    model = ARModel()

    logits = torch.randn(BATCH, SEQUENCE_LENGTH, DELAY_BINS)
    seq = F.softmax(logits, dim=-1)

    t0 = time.time()
    losses = step_losses(model, seq)
    check("step_losses shape", tuple(losses.shape) == (BATCH, SEQUENCE_LENGTH - 1))
    check("step_losses finite", torch.isfinite(losses).all().item())
    check("step_losses nonnegative", (losses >= 0).all().item())

    losses.mean().backward()
    check("AR grads finite", all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    print(f"  ({time.time() - t0:.1f}s)")


def main():
    test_sync_model()
    test_ar_model()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
