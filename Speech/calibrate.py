"""
Calibrates the anomaly-score threshold on `heldout_real` — real GRID clips the AR
model never trained on (paper Sec. 4.2's rationale for a 3-way split: don't score a
model on the same real data it learned the "normal" distribution from).

The per-clip anomaly score is the AR model's cumulative prediction loss (paper
Eq. 4), averaged over frames so it's comparable across clips of different lengths.
The threshold is set at a target false-positive rate on this real-only distribution
(mirrors src/calibrate_target_fpr.py in the image module): the score below which
(1 - target_fpr) of real clips fall.

Requires extract_features.py to have already produced heldout_real's feature
sequences. Writes models/threshold.json, which detect.py reads.

Usage:
    python calibrate.py --ar-checkpoint models/ar/ar_model.pt --target-fpr 0.05
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import AR_CKPT_DIR, RESULTS_DIR, THRESHOLD_PATH
from datasets import FeatureSequenceDataset
from models.ar_model import ARModel, sequence_anomaly_score
from train_sync import get_device


@torch.no_grad()
def score_split(model, split, device):
    ds = FeatureSequenceDataset(split, full_sequence=True)
    scores, stems = [], []
    for seq, stem in ds:
        total, losses = sequence_anomaly_score(model, seq.to(device))
        scores.append(total / max(1, len(losses)))  # mean per-frame loss, length-invariant
        stems.append(stem)
    return np.array(scores), stems


def plot_histogram(scores, threshold, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=30, alpha=0.7, label="heldout_real")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4f}")
    ax.set_xlabel("Mean per-frame anomaly score (soft-CE nats)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate the anomaly-score threshold on held-out real clips.")
    parser.add_argument("--ar-checkpoint", type=Path, default=AR_CKPT_DIR / "ar_model.pt")
    parser.add_argument("--split", type=str, default="heldout_real")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    if not args.ar_checkpoint.exists():
        raise SystemExit(f"No AR checkpoint at {args.ar_checkpoint} — run train_ar.py first.")

    model = ARModel().to(device)
    model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    model.eval()

    scores, stems = score_split(model, args.split, device)
    print(f"{len(scores)} clips from split={args.split!r}")
    print(f"score mean={scores.mean():.4f} std={scores.std():.4f} range=[{scores.min():.4f}, {scores.max():.4f}]")

    threshold = float(np.percentile(scores, (1 - args.target_fpr) * 100))
    achieved_fpr = float(np.mean(scores > threshold))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = RESULTS_DIR / "calibration_histogram.png"
    plot_histogram(scores, threshold, hist_path)

    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(
            {
                "threshold": threshold,
                "metric": "mean_per_frame_soft_ce",
                "calibration_split": args.split,
                "target_fpr": args.target_fpr,
                "achieved_fpr": achieved_fpr,
                "num_clips": int(len(scores)),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "ar_checkpoint": str(args.ar_checkpoint),
                "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    print(f"\nThreshold (score above this = flagged as anomalous/fake) = {threshold:.4f}")
    print(f"Achieved FPR on {args.split} = {achieved_fpr:.2%} (target {args.target_fpr:.2%})")
    print(f"Saved threshold to {THRESHOLD_PATH}")
    print(f"Saved histogram to {hist_path}")


if __name__ == "__main__":
    main()
