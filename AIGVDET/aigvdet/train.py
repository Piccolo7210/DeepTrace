"""Single-branch trainer for AIGVDet (RGB or optical-flow ResNet-50).

Reproduces the official recipe - BCE loss, Adam(1e-4), LR / 10 on a
validation-accuracy plateau, stop below 1e-6 - on top of AMP, resumable
checkpoints and Drive sync. Run as::

    python -m aigvdet.train --branch rgb --data-root DIR --index hf_index.csv --out-dir DIR
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import remote
from .config import (
    ADAM_BETAS,
    BATCH_SIZE,
    BRANCHES,
    BUDGET_FRAMES_PER_VIDEO,
    LEARNING_RATE,
    LR_DECAY_FACTOR,
    LR_FLOOR,
    LR_PATIENCE,
    MAX_EPOCHS,
    MIN_DELTA_AFTER_DROP,
    MIN_DELTA_FIRST,
    SEED,
)
from .data import FrameDataset, VideoBalancedSampler, make_loader, select_frames
from .metrics import binary_metrics, video_level
from .models import build_branch
from .preprocess import normalise_on_device

LOGGER = logging.getLogger("aigvdet.train")
LOG_EVERY_STEPS = 50
EPOCH_ONE_MIN_ACC = 0.7  # gate G2: validation accuracy expected after the first epoch
PROFILES = {"paper": None, "budget": BUDGET_FRAMES_PER_VIDEO}  # frames per video per epoch
_SYNCED_FILES = ("last.pt", "best.pt", "history.jsonl", "status.json", "train.log")


class PlateauAction(enum.Enum):
    """What the schedule asks the trainer to do after an epoch."""

    CONTINUE = "continue"
    DROP_LR = "drop_lr"
    STOP = "stop"


class PlateauController:
    """Official ``EarlyStopping`` + ``adjust_learning_rate`` behaviour (decision D5).

    After ``LR_PATIENCE`` epochs without a validation-accuracy gain of at least
    ``min_delta`` the learning rate is divided by ``LR_DECAY_FACTOR`` and the
    reference is reset; training stops once the next rate would fall below
    ``LR_FLOOR``. Best-checkpoint tracking lives in the trainer, not here.
    """

    def __init__(self, lr: float = LEARNING_RATE) -> None:
        self.lr = lr
        self._reference: float | None = None
        self._stale_epochs = 0
        self._min_delta = MIN_DELTA_FIRST

    def step(self, val_acc: float) -> PlateauAction:
        if self._reference is None or val_acc >= self._reference + self._min_delta:
            self._reference = val_acc
            self._stale_epochs = 0
            return PlateauAction.CONTINUE
        self._stale_epochs += 1
        if self._stale_epochs < LR_PATIENCE:
            return PlateauAction.CONTINUE
        next_lr = self.lr / LR_DECAY_FACTOR
        if next_lr < LR_FLOOR * 0.999:  # tolerance for float rounding (1e-5 / 10)
            return PlateauAction.STOP
        self.lr = next_lr
        self._reference = None
        self._stale_epochs = 0
        self._min_delta = MIN_DELTA_AFTER_DROP
        return PlateauAction.DROP_LR

    def state_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "reference": self._reference,
            "stale_epochs": self._stale_epochs,
            "min_delta": self._min_delta,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lr = state["lr"]
        self._reference = state["reference"]
        self._stale_epochs = state["stale_epochs"]
        self._min_delta = state["min_delta"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fixed 448x448 inputs


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _limit_frames(frames: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Stratified random subsample (debugging only); ``limit <= 0`` keeps everything."""
    if limit <= 0:
        return frames
    parts = [
        group.sample(n=min(len(group), max(limit // 2, 1)), random_state=seed)
        for _, group in frames.groupby("label")
    ]
    return pd.concat(parts).reset_index(drop=True)


class BranchTrainer:
    """Trains one branch (``rgb`` or ``flow``) with resumable checkpoints."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        if args.batch_size % args.accum_steps:
            raise ValueError("--batch-size must be divisible by --accum-steps")
        self.args = args
        self.device = device
        self.use_amp = device.type == "cuda"
        self.accum_steps = args.accum_steps
        self.out_dir: Path = args.out_dir
        self.remote_dir: str | None = args.remote_dir

        index = pd.read_csv(args.index)
        self.train_frames = _limit_frames(
            select_frames(index, "train", args.branch), args.debug_frames, args.seed
        )
        self.val_frames = _limit_frames(
            select_frames(index, "val", args.branch), args.debug_frames, args.seed
        )
        self.sampler = VideoBalancedSampler(
            self.train_frames, PROFILES[args.profile], seed=args.seed
        )
        self.train_loader = make_loader(
            FrameDataset(self.train_frames, args.data_root, train=True),
            batch_size=args.batch_size // args.accum_steps,
            num_workers=args.num_workers,
            sampler=self.sampler,
            drop_last=True,
        )
        self.val_loader = make_loader(
            FrameDataset(self.val_frames, args.data_root, train=False),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        self.model = build_branch(pretrained=not args.no_pretrained).to(
            device, memory_format=torch.channels_last
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.controller = PlateauController()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.controller.lr, betas=ADAM_BETAS
        )
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        self.best_acc = -1.0
        self.best_epoch = -1

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self) -> str:
        """Train until the schedule stops, the epoch cap or the time budget is hit."""
        start_epoch = self._resume()
        if self._schedule_complete(start_epoch):
            return "already_finished"
        deadline = time.time() + self.args.max_hours * 3600
        reason = "max_epochs"
        LOGGER.info(
            "Training %s branch: %d train frames, %d val frames, epoch %d -> %d",
            self.args.branch,
            len(self.train_frames),
            len(self.val_frames),
            start_epoch,
            self.args.max_epochs,
        )
        for epoch in range(start_epoch, self.args.max_epochs):
            started = time.time()
            lr_used = self.optimizer.param_groups[0]["lr"]
            train_stats = self._train_epoch(epoch)
            val_stats = self._validate()
            action = self.controller.step(val_stats["frame_acc"])
            is_best = val_stats["frame_acc"] > self.best_acc
            if is_best:
                self.best_acc, self.best_epoch = val_stats["frame_acc"], epoch
            if action is PlateauAction.DROP_LR:
                self._set_lr(self.controller.lr)
            elapsed = time.time() - started
            self._record(epoch, lr_used, train_stats, val_stats, action, elapsed)
            self._save_checkpoints(epoch, val_stats, is_best)
            self._check_gate_g2(epoch, val_stats)
            if action is PlateauAction.STOP:
                reason = "lr_floor"
                break
            if time.time() + elapsed > deadline:
                reason = "time_budget"
                break
        self._write_status(reason)
        return reason

    def overfit_check(self, steps: int) -> bool:
        """Gate G2: the network must overfit one fixed batch without NaN."""
        images, labels = next(iter(self.train_loader))
        images, labels = self._to_device(images), labels.to(self.device)
        self.model.train()
        losses: list[float] = []
        for step in range(1, steps + 1):
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits = self.model(images).squeeze(1)
            loss = self.criterion(logits.float(), labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(loss.item())
            if step % 10 == 0:
                LOGGER.info("overfit step %3d  loss %.4f", step, losses[-1])
        passed = all(math.isfinite(v) for v in losses) and losses[-1] < 0.5 * losses[0]
        LOGGER.info(
            "G2 overfit check %s (loss %.4f -> %.4f)",
            "PASSED" if passed else "FAILED",
            losses[0],
            losses[-1],
        )
        return passed

    # ------------------------------------------------------------------ #
    # Epoch loops
    # ------------------------------------------------------------------ #
    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
        )

    def _to_device(self, images: torch.Tensor) -> torch.Tensor:
        batch = normalise_on_device(images, self.device)
        return batch.contiguous(memory_format=torch.channels_last)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.sampler.set_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device)
        seen = 0
        started = time.time()
        for step, (images, labels) in enumerate(self.train_loader, start=1):
            images = self._to_device(images)
            labels = labels.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(images).squeeze(1)
            loss = self.criterion(logits.float(), labels)
            self.scaler.scale(loss / self.accum_steps).backward()
            if step % self.accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.detach() * len(labels)
            correct += ((logits.detach() > 0) == (labels > 0.5)).sum()
            seen += len(labels)
            if step % LOG_EVERY_STEPS == 0:
                self._log_progress(epoch, step, float(loss_sum), float(correct), seen, started)
        if seen == 0:
            raise RuntimeError("The training loader produced no batches")
        mean_loss = float(loss_sum) / seen
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"Non-finite training loss in epoch {epoch}")
        elapsed = time.time() - started
        return {
            "loss": mean_loss,
            "acc": float(correct) / seen,
            "images": seen,
            "img_per_s": seen / elapsed,
        }

    def _log_progress(
        self, epoch: int, step: int, loss_sum: float, correct: float, seen: int, started: float
    ) -> None:
        if not math.isfinite(loss_sum):
            raise FloatingPointError(f"Non-finite loss in epoch {epoch}, step {step}")
        LOGGER.info(
            "epoch %2d step %5d/%d  loss %.4f  acc %.4f  %.1f img/s  lr %.0e",
            epoch,
            step,
            len(self.train_loader),
            loss_sum / seen,
            correct / seen,
            seen / (time.time() - started),
            self.optimizer.param_groups[0]["lr"],
        )

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        probabilities = []
        for images, _ in self.val_loader:
            with self._autocast():
                logits = self.model(self._to_device(images)).squeeze(1)
            probabilities.append(torch.sigmoid(logits.float()).cpu())
        table = self.val_frames[["label", "video_id"]].assign(prob=torch.cat(probabilities).numpy())
        frame = binary_metrics(table["label"], table["prob"])
        videos = video_level(table)
        video = binary_metrics(videos["label"], videos["prob"])
        return {
            "frame_acc": frame["acc"],
            "frame_auc": frame["auc"],
            "frame_ap": frame["ap"],
            "video_acc": video["acc"],
            "video_auc": video["auc"],
        }

    # ------------------------------------------------------------------ #
    # Bookkeeping: history, checkpoints, status, resume
    # ------------------------------------------------------------------ #
    def _record(
        self, epoch: int, lr: float, train: dict, val: dict, action: PlateauAction, seconds: float
    ) -> None:
        entry = {
            "epoch": epoch,
            "lr": lr,
            "train": train,
            "val": val,
            "action": action.value,
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "epoch_seconds": round(seconds, 1),
        }
        with open(self.out_dir / "history.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        LOGGER.info(
            "epoch %2d done  train loss %.4f acc %.4f | val frame acc %.4f auc %.4f "
            "| video acc %.4f auc %.4f | %s | %.0fs",
            epoch,
            train["loss"],
            train["acc"],
            val["frame_acc"],
            val["frame_auc"],
            val["video_acc"],
            val["video_auc"],
            action.value,
            seconds,
        )

    def _save_checkpoints(self, epoch: int, val: dict, is_best: bool) -> None:
        if is_best:
            weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            _atomic_save(
                {"branch": self.args.branch, "epoch": epoch, "val": val, "model": weights},
                self.out_dir / "best.pt",
            )
        payload = {
            "branch": self.args.branch,
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "controller": self.controller.state_dict(),
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "rng_cpu": torch.get_rng_state(),
        }
        if self.use_amp:
            payload["rng_cuda"] = torch.cuda.get_rng_state()
        _atomic_save(payload, self.out_dir / "last.pt")
        self._sync_to_remote(("best.pt",) if is_best else ())
        self._sync_to_remote(("last.pt", "history.jsonl", "train.log"))

    def _check_gate_g2(self, epoch: int, val: dict) -> None:
        if epoch == 0 and val["frame_acc"] <= EPOCH_ONE_MIN_ACC:
            LOGGER.warning(
                "Gate G2: epoch-1 validation accuracy %.3f <= %.2f - check the data and the "
                "learning rate before continuing",
                val["frame_acc"],
                EPOCH_ONE_MIN_ACC,
            )

    def _write_status(self, reason: str) -> None:
        state = "paused" if reason == "time_budget" else "finished"
        status = {
            "state": state,
            "reason": reason,
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
        }
        (self.out_dir / "status.json").write_text(json.dumps(status))
        self._sync_to_remote(("status.json",))
        LOGGER.info(
            "Training %s (%s); best val frame acc %.4f at epoch %d",
            state,
            reason,
            self.best_acc,
            self.best_epoch,
        )

    def _schedule_complete(self, next_epoch: int) -> bool:
        """True if the LR floor was reached earlier or the epoch cap leaves nothing to run."""
        path = self.out_dir / "status.json"
        floor_reached = path.exists() and json.loads(path.read_text()).get("reason") == "lr_floor"
        return floor_reached or next_epoch >= self.args.max_epochs

    def _sync_to_remote(self, names: tuple[str, ...]) -> None:
        if not self.remote_dir:
            return
        for name in names:
            remote.upload_file(self.out_dir / name, f"{self.remote_dir}/{name}")

    def _resume(self) -> int:
        """Restore the latest checkpoint (local first, then Drive); returns the next epoch."""
        if self.remote_dir and not (self.out_dir / "last.pt").exists():
            for name in _SYNCED_FILES:
                remote.download_file(f"{self.remote_dir}/{name}", self.out_dir / name)
        last = self.out_dir / "last.pt"
        if not last.exists():
            return 0
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["branch"] != self.args.branch:
            raise ValueError(f"{last} belongs to the {state['branch']!r} branch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.controller.load_state_dict(state["controller"])
        self.best_acc, self.best_epoch = state["best_acc"], state["best_epoch"]
        torch.set_rng_state(state["rng_cpu"])
        if self.use_amp and "rng_cuda" in state:
            torch.cuda.set_rng_state(state["rng_cuda"])
        self._set_lr(self.controller.lr)
        LOGGER.info("Resumed from epoch %d (lr %.0e)", state["epoch"], self.controller.lr)
        return state["epoch"] + 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="hf_index.csv from K1")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--remote-dir", default=None, help="rclone folder for checkpoints")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="paper")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accum-steps", type=int, default=1, help="use 2 if batch 64 hits OOM")
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--max-hours", type=float, default=11.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overfit-steps", type=int, default=0, help="run gate G2 and exit")
    parser.add_argument("--debug-frames", type=int, default=0, help="cap frames per split")
    parser.add_argument("--no-pretrained", action="store_true", help="offline smoke tests")
    return parser.parse_args(argv)


def _configure_logging(log_file: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(args.out_dir / "train.log")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s | args: %s", device, vars(args))
    trainer = BranchTrainer(args, device)
    if args.overfit_steps:
        return 0 if trainer.overfit_check(args.overfit_steps) else 1
    LOGGER.info("Training stopped: %s", trainer.fit())
    return 0


if __name__ == "__main__":
    sys.exit(main())
