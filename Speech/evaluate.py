"""
Sanity-check evaluation: scores `heldout_real` clips against make_synthetic_fakes.py's
desynced-audio clips, and reports AP/AUC (mirrors src/evaluate.py's format for the
image module). Runs the full inference pipeline (sync model -> S(i,.) sequence -> AR
model -> cumulative loss) on both sets.

This is NOT the paper's FakeAVCeleb/KoDF evaluation — it only confirms the pipeline's
anomaly score reacts to audio-visual desync at all. A real forensics benchmark needs
an actual manipulated-video dataset (see Speech/README.md).

Usage:
    python evaluate.py --sync-checkpoint models/sync/sync_model.pt --ar-checkpoint models/ar/ar_model.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from config import AR_CKPT_DIR, PROCESSED_DIR, SYNC_CKPT_DIR, SYNTHETIC_FAKES_DIR
from datasets import _load_wav, usable_utterances
from extract_features import extract_clip_features
from models.ar_model import ARModel, sequence_anomaly_score
from models.sync_model import SyncModel
from train_sync import get_device


@torch.no_grad()
def score_pairs(sync_model, ar_model, pairs, device):
    scores = []
    for frames_path, wav_path in pairs:
        frames = np.load(frames_path)
        wav = _load_wav(wav_path)
        feats = extract_clip_features(sync_model, frames, wav, device)
        total, losses = sequence_anomaly_score(ar_model, torch.from_numpy(feats).to(device))
        scores.append(total / max(1, len(losses)))
    return np.array(scores)


def real_pairs(split):
    return [
        (PROCESSED_DIR / r["speaker"] / r["utt_id"] / "frames.npy",
         PROCESSED_DIR / r["speaker"] / r["utt_id"] / "audio.wav")
        for r in usable_utterances(split)
    ]


def fake_pairs(fake_dir):
    return [
        (p, p.parent / "audio.wav")
        for p in sorted(Path(fake_dir).rglob("frames.npy"))
        if (p.parent / "audio.wav").exists()
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="AP/AUC of real vs. synthetic-fake anomaly scores.")
    parser.add_argument("--sync-checkpoint", type=Path, default=SYNC_CKPT_DIR / "sync_model.pt")
    parser.add_argument("--ar-checkpoint", type=Path, default=AR_CKPT_DIR / "ar_model.pt")
    parser.add_argument("--real-split", type=str, default="heldout_real")
    parser.add_argument("--fake-dir", type=Path, default=SYNTHETIC_FAKES_DIR)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    sync_model = SyncModel().to(device)
    sync_model.load_state_dict(torch.load(args.sync_checkpoint, map_location=device))
    sync_model.eval()

    ar_model = ARModel().to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    ar_model.eval()

    real = real_pairs(args.real_split)
    fake = fake_pairs(args.fake_dir)
    if not real or not fake:
        raise SystemExit(
            f"Need both real ({len(real)} found in split={args.real_split!r}) and fake "
            f"({len(fake)} found under {args.fake_dir}) clips — run preprocess.py and "
            f"make_synthetic_fakes.py first."
        )

    print(f"Scoring {len(real)} real + {len(fake)} synthetic-fake clips...")
    real_scores = score_pairs(sync_model, ar_model, real, device)
    fake_scores = score_pairs(sync_model, ar_model, fake, device)

    y_true = np.concatenate([np.zeros_like(real_scores), np.ones_like(fake_scores)])
    y_score = np.concatenate([real_scores, fake_scores])
    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    print(f"\nReal:  n={len(real_scores)}  mean={real_scores.mean():.4f}  "
          f"range=[{real_scores.min():.4f}, {real_scores.max():.4f}]")
    print(f"Fake:  n={len(fake_scores)}  mean={fake_scores.mean():.4f}  "
          f"range=[{fake_scores.min():.4f}, {fake_scores.max():.4f}]")
    print(f"\nAP  = {ap:.4f}")
    print(f"AUC = {auc:.4f}")
    print(
        "\n(Sanity check only — 'fake' here is desynced audio, not a real manipulation. "
        "AP/AUC well above 0.5 means the trained pipeline is at least sensitive to "
        "audio-visual misalignment; it says nothing about performance on actual deepfakes.)"
    )


if __name__ == "__main__":
    main()
