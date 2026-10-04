"""
Stage 2 prep: runs a trained synchronization model (train_sync.py's output) over
every utterance in the given splits and saves the per-frame time-delay distribution
S(i,·) (paper Eq. 1) as a (T, DELAY_BINS) .npy sequence — the input the AR anomaly
model (train_ar.py) trains and scores on.

Every frame of a clip is scored (edge frames use the same zero-padded audio/video
windows as training, via datasets.py's helpers), so a clip with T frames yields a
full-length (T, DELAY_BINS) sequence, not just the interior frames.

Run on whichever device trained the sync model (GPU strongly recommended — this does
T * DELAY_BINS audio encodes per clip). Resumable: an utterance whose output .npy
already exists is skipped.

Usage:
    python extract_features.py --sync-checkpoint models/sync/sync_model.pt --splits ar_train,heldout_real
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import FEATURES_DIR, SYNC_CKPT_DIR, TAU
from datasets import _audio_window, _load_wav, _processed_paths, _video_window, load_manifest_rows
from models.sync_model import SyncModel
from train_sync import get_device


@torch.no_grad()
def extract_clip_features(model, frames, wav, device, batch_size=32):
    T = len(frames)
    out = np.zeros((T, 2 * TAU + 1), dtype=np.float32)
    for start in range(0, T, batch_size):
        chunk = range(start, min(start + batch_size, T))

        video_batch = torch.stack([torch.from_numpy(_video_window(frames, i)) for i in chunk]).to(device)
        audio_batch = torch.stack(
            [
                torch.from_numpy(np.stack([_audio_window(wav, i + d) for d in range(-TAU, TAU + 1)], axis=0))
                for i in chunk
            ]
        ).to(device)

        logits = model.window_logits(video_batch, audio_batch)
        dist = F.softmax(logits, dim=1).cpu().numpy()
        out[start:start + len(chunk)] = dist
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Extract S(i,.) delay-distribution sequences from a trained sync model.")
    parser.add_argument("--sync-checkpoint", type=Path, default=SYNC_CKPT_DIR / "sync_model.pt")
    parser.add_argument("--splits", type=str, default="ar_train,heldout_real")
    parser.add_argument("--batch-size", type=int, default=32, help="Frames per forward-pass chunk.")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    if not args.sync_checkpoint.exists():
        raise SystemExit(f"No sync checkpoint at {args.sync_checkpoint} — run train_sync.py first.")

    model = SyncModel().to(device)
    model.load_state_dict(torch.load(args.sync_checkpoint, map_location=device))
    model.eval()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    counts = {}
    for split in splits:
        rows = load_manifest_rows(split)
        for row in tqdm(rows, desc=f"Extracting [{split}]"):
            frames_path, wav_path = _processed_paths(row)
            if not (frames_path.exists() and wav_path.exists()):
                counts["missing_processed"] = counts.get("missing_processed", 0) + 1
                continue

            out_path = FEATURES_DIR / split / row["speaker"] / f"{row['utt_id']}.npy"
            if out_path.exists():
                counts["skipped"] = counts.get("skipped", 0) + 1
                continue

            frames = np.load(frames_path)
            wav = _load_wav(wav_path)
            features = extract_clip_features(model, frames, wav, device, args.batch_size)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, features)
            counts["extracted"] = counts.get("extracted", 0) + 1

    print("\nDone.")
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")


if __name__ == "__main__":
    main()
