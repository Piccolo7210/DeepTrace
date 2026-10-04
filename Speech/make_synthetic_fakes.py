"""
Placeholder evaluation data: since no manipulated-video dataset (FakeAVCeleb/KoDF-
style) is available yet, this builds a cheap "obviously desynced" fake set out of
`heldout_real` clips, by shifting the audio track well beyond the trained delay
window (tau frames) relative to the video. This is NOT a deepfake — it's a sanity
check that the trained pipeline's anomaly score actually reacts to audio-visual
misalignment before spending effort on a real forensics benchmark. See
Speech/README.md for the caveat and what a real evaluation would need instead
(FakeAVCeleb/KoDF, or Wav2Lip-manipulated clips as in the paper's Appendix A.7).

Output goes to data/synthetic_fakes/<speaker>/<utt_id>/{frames.npy,audio.wav}, in
the same layout preprocess.py uses, so evaluate.py can run the same feature
extraction + scoring code over both real and synthetic-fake directories.

Usage:
    python make_synthetic_fakes.py --split heldout_real --shift-frames 20
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from config import AUDIO_SR, PROCESSED_DIR, SYNTHETIC_FAKES_DIR, TAU
from datasets import _load_wav, usable_utterances

DEFAULT_SHIFT_FRAMES = TAU + 5  # frames of delay well outside [-tau, tau]


def shift_audio(wav, shift_samples):
    """Rolls the waveform by shift_samples, zero-filling the vacated edge (no wraparound)."""
    shifted = np.zeros_like(wav)
    if shift_samples >= 0:
        n = max(0, len(wav) - shift_samples)
        shifted[shift_samples:shift_samples + n] = wav[:n]
    else:
        n = max(0, len(wav) + shift_samples)
        shifted[:n] = wav[-shift_samples:-shift_samples + n]
    return shifted


def parse_args():
    parser = argparse.ArgumentParser(description="Build a synthetic desynced-audio fake set from real clips.")
    parser.add_argument("--split", type=str, default="heldout_real")
    parser.add_argument("--shift-frames", type=int, default=DEFAULT_SHIFT_FRAMES,
                         help="Audio delay in video frames (converted to samples at AUDIO_SR); "
                              "should exceed TAU so it falls outside the trained delay window.")
    parser.add_argument("--out-dir", type=Path, default=SYNTHETIC_FAKES_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.shift_frames <= TAU:
        print(f"Warning: --shift-frames={args.shift_frames} is within the trained window "
              f"(tau={TAU}) — the desync may not be detectable as intended.")

    shift_samples = int(round(args.shift_frames / 25 * AUDIO_SR))
    rows = usable_utterances(args.split)
    if not rows:
        raise SystemExit(f"No preprocessed utterances for split={args.split!r} under {PROCESSED_DIR}.")

    for row in tqdm(rows, desc="Synthesizing desynced fakes"):
        src_dir = PROCESSED_DIR / row["speaker"] / row["utt_id"]
        out_dir = args.out_dir / row["speaker"] / row["utt_id"]
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(src_dir / "frames.npy", out_dir / "frames.npy")

        wav = _load_wav(src_dir / "audio.wav")
        shifted = shift_audio(wav, shift_samples)
        wavfile.write(out_dir / "audio.wav", AUDIO_SR, (np.clip(shifted, -1.0, 1.0) * 32767).astype(np.int16))

    print(f"\nWrote {len(rows)} synthetic-fake clips to {args.out_dir} "
          f"(audio shifted by {args.shift_frames} frames / {shift_samples} samples).")


if __name__ == "__main__":
    main()
