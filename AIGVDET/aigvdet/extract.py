"""Test-time extraction worker: scores manifest videos shard by shard on one GPU.

Each worker takes every ``workers``-th manifest row, processes it in shards,
and writes one ``probs_<shard>.csv`` (one row per video x frame pair) that is
uploaded to Drive as soon as it is complete. A shard whose CSV already exists
(locally or on Drive) is skipped, so a dead session costs at most one shard. The
shard name carries a hash of its video keys, so changing the manifest (more
videos, another sample) recomputes exactly the shards whose content changed.

Run as ``python -m aigvdet.extract --manifest ... --worker 0 --workers 2 ...``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

from . import remote
from .config import MAX_FLOW_PIXELS, MAX_VIDEO_FRAMES, PAIRS_PER_VIDEO, RAFT_UPDATES
from .data import read_rgb
from .flow import load_raft
from .models import load_branch
from .scoring import PairScorer
from .video_io import pick_pair_indices, probe_video, read_frames, resolution_bucket

LOGGER = logging.getLogger("aigvdet.extract")
SHARD_SIZE = 50
_RECOVERABLE = (ValueError, RuntimeError, OSError, cv2.error, torch.cuda.OutOfMemoryError)
_META_COLUMNS = ("video_key", "set", "kind", "label", "half")


def _frames_from_video(path: Path, k: int) -> tuple[object, list[int], dict]:
    frames = read_frames(path, MAX_VIDEO_FRAMES)
    height, width = frames[0].shape[:2]
    info = probe_video(path)
    details = {"fps": info.fps, "n_frames": len(frames), "width": width, "height": height}
    return frames, pick_pair_indices(len(frames), k), details


def _frames_from_hf(index: pd.DataFrame, data_root: Path, video_id: str, k: int):
    part = index[
        (index["split"] == "val")
        & (index["kind"] == "rgb")
        & (index["label"] == 0)
        & (index["video_id"] == video_id)
    ].sort_values("frame_idx")
    pairs = pick_pair_indices(len(part), k)
    needed = {j for i in pairs for j in (i, i + 1)}
    frames = {j: read_rgb(data_root / part.iloc[j]["path"]) for j in needed}
    first = part.iloc[0]
    details = {
        "fps": float("nan"),
        "n_frames": len(part),
        "width": int(first["width"]),
        "height": int(first["height"]),
    }
    return frames, pairs, details


def score_video(
    scorer: PairScorer,
    row: pd.Series,
    local_path: Path | None,
    index: pd.DataFrame,
    data_root: Path,
    k: int,
) -> list[dict]:
    """Per-pair rows of one manifest video (raises a recoverable error on bad clips)."""
    if row["source"] == "hf_frames":
        frames, pairs, details = _frames_from_hf(index, data_root, row["relative"], k)
    else:
        frames, pairs, details = _frames_from_video(local_path, k)
    if not pairs:
        raise ValueError("fewer than 2 frames: no adjacent pair")
    details["bucket"] = resolution_bucket(details["width"], details["height"])
    meta = {name: row[name] for name in _META_COLUMNS}
    return [{**meta, **details, **pair} for pair in scorer.score_pairs(frames, pairs)]


def _content_hash(shard: pd.DataFrame) -> str:
    """Short hash of a shard's video keys: a changed manifest never reuses stale shards."""
    return hashlib.md5("\n".join(shard["video_key"]).encode()).hexdigest()[:8]


def _download_shard(shard: pd.DataFrame, cache: Path) -> dict[str, Path]:
    """Fetch the Drive videos of a shard; returns ``video_key -> local path``."""
    local_paths: dict[str, Path] = {}
    drive_rows = shard[shard["source"] == "drive"]
    for number, (root, group) in enumerate(drive_rows.groupby("remote_root")):
        target = cache / f"root{number}"
        remote.copy_files(root, group["relative"].tolist(), target)
        for _, row in group.iterrows():
            local_paths[row["video_key"]] = target / row["relative"]
    return local_paths


def process_shard(
    scorer: PairScorer,
    shard: pd.DataFrame,
    args: argparse.Namespace,
    index: pd.DataFrame,
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one shard; returns (probabilities, failures)."""
    cache = args.out_dir / f"cache_w{args.worker}"
    local_paths = _download_shard(shard, cache)
    rows, failures = [], []
    started = time.time()
    for number, (_, row) in enumerate(shard.iterrows(), start=1):
        try:
            rows += score_video(
                scorer, row, local_paths.get(row["video_key"]), index, args.data_root, args.k_pairs
            )
        except _RECOVERABLE as error:
            LOGGER.warning("%s: %s failed: %s", name, row["video_key"], error)
            failures.append({"video_key": row["video_key"], "error": str(error)[:200]})
            torch.cuda.empty_cache()
        if number % 10 == 0 or number == len(shard):
            LOGGER.info("%s: %d/%d videos, %.0fs", name, number, len(shard), time.time() - started)
    shutil.rmtree(cache, ignore_errors=True)  # videos are only needed while scoring
    return pd.DataFrame(rows), pd.DataFrame(failures, columns=["video_key", "error"])


def build_scorer(args: argparse.Namespace, device: torch.device) -> PairScorer:
    return PairScorer(
        load_branch(args.rgb_ckpt, device),
        load_branch(args.flow_ckpt, device),
        load_raft(device, pretrained=not args.no_pretrained),
        device,
        updates=args.raft_updates,
        max_flow_pixels=args.max_flow_pixels,
    )


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    index = pd.read_csv(args.index) if args.index else pd.DataFrame()
    mine = manifest.iloc[args.worker :: args.workers]
    scorer = build_scorer(args, device)
    LOGGER.info("Worker %d/%d on %s: %d videos", args.worker, args.workers, device, len(mine))
    for number, start in enumerate(range(0, len(mine), args.shard_size)):
        shard = mine.iloc[start : start + args.shard_size]
        name = f"w{args.worker}_s{number:03d}_{_content_hash(shard)}"
        probs_file = args.out_dir / f"probs_{name}.csv"
        remote_file = f"{args.remote_dir}/{probs_file.name}" if args.remote_dir else None
        if probs_file.exists() or (remote_file and remote.exists(remote_file)):
            LOGGER.info("%s already done - skipped", name)
            continue
        probs, failures = process_shard(scorer, shard, args, index, name)
        probs.to_csv(probs_file, index=False)
        failures.to_csv(args.out_dir / f"failures_{name}.csv", index=False)
        if remote_file:
            remote.upload_file(probs_file, remote_file)
            remote.upload_file(
                args.out_dir / f"failures_{name}.csv", f"{args.remote_dir}/failures_{name}.csv"
            )
        LOGGER.info("%s done: %d pair rows, %d failed videos", name, len(probs), len(failures))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score test-manifest videos on one GPU.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=None, help="hf_index.csv (for R1 reals)")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--remote-dir", default=None, help="rclone folder for shard CSVs")
    parser.add_argument("--rgb-ckpt", type=Path, required=True)
    parser.add_argument("--flow-ckpt", type=Path, required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--k-pairs", type=int, default=PAIRS_PER_VIDEO)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--raft-updates", type=int, default=RAFT_UPDATES)
    parser.add_argument("--max-flow-pixels", type=int, default=MAX_FLOW_PIXELS)
    parser.add_argument("--no-pretrained", action="store_true", help="offline smoke tests")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
