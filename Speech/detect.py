"""
Single-video inference: runs the full pipeline (sync model -> S(i,.) sequence -> AR
model) on one clip and reports a verdict against the calibrated threshold
(calibrate.py). Mirrors src/detect.py's role in the image module.

Two input modes:
  --video PATH     a raw video file (any format ffmpeg/OpenCV can read, with audio).
                    Runs the same face-crop + audio-resample steps as preprocess.py,
                    so this needs opencv-python/mediapipe/ffmpeg installed.
  --processed-dir PATH   a directory already containing frames.npy + audio.wav
                    (e.g. from preprocess.py or make_synthetic_fakes.py). No extra
                    dependencies needed in this mode.

Usage:
    python detect.py --video path/to/clip.mp4 --plot
    python detect.py --processed-dir data/grid_processed/s5/bbaf2n --plot
"""

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import AR_CKPT_DIR, RESULTS_DIR, SYNC_CKPT_DIR, THRESHOLD_PATH
from datasets import _load_wav
from extract_features import extract_clip_features
from models.ar_model import ARModel, sequence_anomaly_score
from models.sync_model import SyncModel
from train_sync import get_device

DEFAULT_THRESHOLD = 1.0  # nats/frame; arbitrary until calibrate.py has run


def load_from_video(video_path, crop_size, margin, sample_rate):
    from preprocess import crop_frames, detect_face_boxes, extract_audio, read_frames

    frames_raw = read_frames(video_path)
    if not frames_raw:
        raise RuntimeError(f"No frames decoded from {video_path}.")
    boxes = detect_face_boxes(frames_raw)
    frames = crop_frames(frames_raw, boxes, crop_size, margin)
    if frames is None:
        raise RuntimeError(f"No face detected in {video_path}.")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        extract_audio(video_path, wav_path, sample_rate)
        wav = _load_wav(wav_path)
    return frames, wav


def load_from_processed(processed_dir):
    processed_dir = Path(processed_dir)
    frames = np.load(processed_dir / "frames.npy")
    wav = _load_wav(processed_dir / "audio.wav")
    return frames, wav


def load_threshold(threshold_path):
    path = Path(threshold_path)
    if path.exists():
        with open(path) as f:
            return json.load(f)["threshold"], True
    return DEFAULT_THRESHOLD, False


def plot_report(features, losses, score, threshold, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].imshow(features.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[0].set_ylabel("Delay bin (S(i,.))")
    axes[0].set_title("Synchronization feature sequence")

    axes[1].plot(losses.numpy() if torch.is_tensor(losses) else losses)
    axes[1].axhline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.3f}")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Per-frame loss")
    axes[1].set_title(f"Anomaly score = {score:.4f}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def detect(frames, wav, sync_model, ar_model, device, threshold, calibrated):
    features = extract_clip_features(sync_model, frames, wav, device)
    total, losses = sequence_anomaly_score(ar_model, torch.from_numpy(features).to(device))
    score = total / max(1, len(losses))

    return {
        "num_frames": int(len(frames)),
        "mean_per_frame_score": score,
        "cumulative_score": total,
        "threshold": threshold,
        "threshold_calibrated": calibrated,
        "verdict": "Anomalous (audio-visual desync)" if score > threshold else "Consistent with real",
        "features": features,
        "losses": losses,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run audio-visual anomaly detection on one clip.")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--sync-checkpoint", type=Path, default=SYNC_CKPT_DIR / "sync_model.pt")
    parser.add_argument("--ar-checkpoint", type=Path, default=AR_CKPT_DIR / "ar_model.pt")
    parser.add_argument("--threshold-path", type=Path, default=THRESHOLD_PATH)
    parser.add_argument("--crop-size", type=int, default=112)
    parser.add_argument("--margin", type=float, default=1.6)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.video and not args.processed_dir:
        raise SystemExit("Pass either --video or --processed-dir.")

    device = get_device(args.device)

    sync_model = SyncModel().to(device)
    sync_model.load_state_dict(torch.load(args.sync_checkpoint, map_location=device))
    sync_model.eval()

    ar_model = ARModel().to(device)
    ar_model.load_state_dict(torch.load(args.ar_checkpoint, map_location=device))
    ar_model.eval()

    threshold, calibrated = load_threshold(args.threshold_path)
    if not calibrated:
        print(f"Warning: no calibrated threshold at {args.threshold_path}; using default {DEFAULT_THRESHOLD}. "
              f"Run calibrate.py once you have a trained AR model and heldout_real features.")

    if args.video:
        frames, wav = load_from_video(args.video, args.crop_size, args.margin, args.sample_rate)
        stem = args.video.stem
    else:
        frames, wav = load_from_processed(args.processed_dir)
        stem = args.processed_dir.name

    result = detect(frames, wav, sync_model, ar_model, device, threshold, calibrated)

    print(f"\n{stem}")
    print(f"  frames = {result['num_frames']}")
    print(f"  mean per-frame anomaly score = {result['mean_per_frame_score']:.4f}")
    print(f"  threshold = {result['threshold']:.4f} (calibrated={result['threshold_calibrated']})")
    print(f"  verdict = {result['verdict']}")

    if args.plot:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = args.out_dir / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}_analysis.png"
        plot_report(result["features"], result["losses"], result["mean_per_frame_score"], threshold, plot_path)
        print(f"  saved plot: {plot_path}")


if __name__ == "__main__":
    main()
