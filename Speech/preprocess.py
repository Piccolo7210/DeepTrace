"""
Turns each raw GRID utterance (an .mpg + a .wav, listed in manifest.csv) into what the
sync model actually trains on: a fixed-size, face-cropped frame stack, and a 16kHz mono
wav (paper Sec. 4.1 hyperparameters: 25fps video, 16kHz audio).

Face cropping uses MediaPipe's short-range face detector. GRID is a static camera with
one frontal, well-lit speaker per clip, so a single smoothed bounding box (median center
+ size across the clip) is enough — no per-frame face tracker needed, and it avoids
crop jitter that would otherwise show up as spurious visual motion to the sync model.

Audio is resampled with ffmpeg (must be on PATH — see Speech/README.md), independent of
whichever sample rate the GRID download used (25kHz "endpointed" or 50kHz "raw").

Resumable like src/download_real.py: an utterance already fully processed
(frames.npy + audio.wav both present) is skipped on rerun.

Usage:
    python preprocess.py                     # process every row in manifest.csv
    python preprocess.py --split sync_train  # process just one split
    python preprocess.py --speakers s1,s2
"""

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import AUDIO_SR, CROP_MARGIN, CROP_SIZE, MANIFEST_PATH, PROCESSED_DIR

_face_detector = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        import mediapipe as mp

        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
    return _face_detector


def read_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def detect_face_boxes(frames):
    """Runs the face detector per frame; returns a list of (cx, cy, side) in pixels,
    one per frame that had a detection (frames with no detection are skipped)."""
    detector = _get_face_detector()
    boxes = []
    for frame in frames:
        h, w, _ = frame.shape
        result = detector.process(frame)
        if not result.detections:
            continue
        box = result.detections[0].location_data.relative_bounding_box
        cx = (box.xmin + box.width / 2) * w
        cy = (box.ymin + box.height / 2) * h
        side = max(box.width * w, box.height * h)
        boxes.append((cx, cy, side))
    return boxes


def crop_frames(frames, boxes, crop_size, margin):
    """Crops every frame to a single smoothed square box (median over per-frame
    detections), so the crop doesn't jitter frame-to-frame. Returns a
    (T, crop_size, crop_size, 3) uint8 array, or None if no face was ever detected."""
    if not boxes:
        return None
    cx = float(np.median([b[0] for b in boxes]))
    cy = float(np.median([b[1] for b in boxes]))
    side = float(np.median([b[2] for b in boxes])) * margin

    half = side / 2
    out = np.zeros((len(frames), crop_size, crop_size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        h, w, _ = frame.shape
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        x1, y1 = int(round(cx + half)), int(round(cy + half))

        pad_l, pad_t = max(0, -x0), max(0, -y0)
        pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
        x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)

        crop = frame[y0c:y1c, x0c:x1c]
        if pad_l or pad_t or pad_r or pad_b:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        out[i] = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return out


def extract_audio(video_path, out_wav, sample_rate):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — install it and retry (see Speech/README.md).")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-ar", str(sample_rate), "-ac", "1", "-vn",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def process_one(video_path, audio_path, out_dir, crop_size, margin, sample_rate):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.npy"
    wav_path = out_dir / "audio.wav"

    if frames_path.exists() and wav_path.exists():
        return "skipped"

    frames = read_frames(video_path)
    if not frames:
        return "no_frames"
    boxes = detect_face_boxes(frames)
    cropped = crop_frames(frames, boxes, crop_size, margin)
    if cropped is None:
        return "no_face"
    np.save(frames_path, cropped)

    extract_audio(audio_path, wav_path, sample_rate)
    return "success"


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess GRID utterances into face crops + resampled audio.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--split", type=str, default=None, help="Only process this split (sync_train/ar_train/heldout_real).")
    parser.add_argument("--speakers", type=str, default=None, help="Comma-separated speaker filter.")
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--margin", type=float, default=CROP_MARGIN)
    parser.add_argument("--sample-rate", type=int, default=AUDIO_SR)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"No manifest at {args.manifest} — run manifest.py first.")

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    if args.speakers:
        wanted = set(s.strip() for s in args.speakers.split(","))
        rows = [r for r in rows if r["speaker"] in wanted]

    if not rows:
        raise SystemExit("No manifest rows matched the given --split/--speakers filters.")

    counts = {}
    for row in tqdm(rows, desc="Preprocessing"):
        out_dir = PROCESSED_DIR / row["speaker"] / row["utt_id"]
        try:
            status = process_one(
                Path(row["video_path"]), Path(row["audio_path"]), out_dir,
                args.crop_size, args.margin, args.sample_rate,
            )
        except Exception as e:
            status = f"error:{type(e).__name__}"
        counts[status] = counts.get(status, 0) + 1

    print("\nDone.")
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")
    if any(k not in ("success", "skipped") for k in counts):
        print(
            "\nSome utterances failed (no_face / no_frames / error:*) — these are "
            "excluded automatically downstream since their processed files don't exist."
        )


if __name__ == "__main__":
    main()
