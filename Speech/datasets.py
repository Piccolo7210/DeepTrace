"""
PyTorch datasets for the two training stages.

Stage 1 (sync model, paper Sec. 3.1 / Eq. 1-2): `SyncDataset` reads the face-cropped
frames.npy + 16kHz audio.wav produced by preprocess.py and returns short video/audio
windows around a random anchor frame. Two modes match the two-phase curriculum
described in the paper's Appendix A.5:
  - "cross_video": one (video, audio) pair per item, at true alignment (delay 0).
    The training loop batches these and treats every other item's audio as a negative
    — easy, cross-video negatives.
  - "window": one video window plus the full stack of `DELAY_BINS` candidate audio
    windows at every delay in [-tau, tau] from the *same* video — hard, within-video
    negatives. This is also exactly the computation extract_features.py needs at
    inference time (paper Eq. 1's softmax window), so training in this mode and
    extracting features later use the identical audio-windowing code path.

Stage 2 (AR anomaly model, paper Sec. 3.2): `FeatureSequenceDataset` reads the
per-utterance S(i,:) sequences saved by extract_features.py and samples fixed-length
windows for training, or returns whole sequences for calibration/detection.
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    AUDIO_SR,
    DELAY_BINS,
    FEATURES_DIR,
    FPS,
    MANIFEST_PATH,
    PROCESSED_DIR,
    SEQUENCE_LENGTH,
    TAU,
    VISUAL_CLIP_LEN,
)

AUDIO_CLIP_SAMPLES = int(round(0.2 * AUDIO_SR))  # paper Sec. 4.1: 0.2s audio clips


def load_manifest_rows(split=None, manifest_path=MANIFEST_PATH):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split is not None:
        rows = [r for r in rows if r["split"] == split]
    return rows


def _processed_paths(row):
    d = PROCESSED_DIR / row["speaker"] / row["utt_id"]
    return d / "frames.npy", d / "audio.wav"


def usable_utterances(split, manifest_path=MANIFEST_PATH):
    """Manifest rows whose preprocessed frames+audio actually exist on disk."""
    usable = []
    for row in load_manifest_rows(split, manifest_path):
        frames_path, wav_path = _processed_paths(row)
        if frames_path.exists() and wav_path.exists():
            usable.append(row)
    return usable


_PCM_SCALE = {"int16": 32768.0, "int32": 2147483648.0}


def _load_wav(wav_path):
    """Reads the 16-bit PCM wav ffmpeg produces via scipy (already a dependency of
    the main repo), so training/extraction code needs no extra audio package."""
    from scipy.io import wavfile

    sr, wav = wavfile.read(str(wav_path))
    if sr != AUDIO_SR:
        raise RuntimeError(f"{wav_path} is {sr}Hz, expected {AUDIO_SR}Hz — was it preprocessed?")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    scale = _PCM_SCALE.get(str(wav.dtype))
    wav = wav.astype(np.float32) / scale if scale else wav.astype(np.float32)
    return wav


def _audio_window(wav, frame_idx, fps=FPS, sr=AUDIO_SR, clip_samples=AUDIO_CLIP_SAMPLES):
    """0.2s audio centered on the timestamp of video frame `frame_idx`, zero-padded
    at the clip boundaries."""
    center_sample = int(round((frame_idx / fps) * sr))
    start = center_sample - clip_samples // 2
    end = start + clip_samples

    out = np.zeros(clip_samples, dtype=np.float32)
    src_start, src_end = max(0, start), min(len(wav), end)
    dst_start = src_start - start
    if src_end > src_start:
        out[dst_start:dst_start + (src_end - src_start)] = wav[src_start:src_end]
    return out


def _video_window(frames, frame_idx, clip_len=VISUAL_CLIP_LEN):
    """clip_len consecutive frames centered on frame_idx, edge-padded at boundaries.
    frames: (T, H, W, 3) uint8. Returns (clip_len, 3, H, W) float32 in [0, 1]."""
    half = clip_len // 2
    idxs = np.clip(np.arange(frame_idx - half, frame_idx - half + clip_len), 0, len(frames) - 1)
    clip = frames[idxs].astype(np.float32) / 255.0
    return np.transpose(clip, (0, 3, 1, 2))  # (T, H, W, 3) -> (T, 3, H, W)


class SyncDataset(Dataset):
    def __init__(self, split, mode="cross_video", manifest_path=MANIFEST_PATH):
        assert mode in ("cross_video", "window")
        self.mode = mode
        self.rows = usable_utterances(split, manifest_path)
        if not self.rows:
            raise RuntimeError(
                f"No preprocessed utterances found for split={split!r} under {PROCESSED_DIR}. "
                f"Run manifest.py then preprocess.py first."
            )

    def __len__(self):
        return len(self.rows)

    def _valid_frame_range(self, n_frames):
        margin = max(VISUAL_CLIP_LEN // 2, TAU if self.mode == "window" else 0)
        lo, hi = margin, n_frames - 1 - margin
        return (lo, hi) if hi >= lo else (0, n_frames - 1)

    def __getitem__(self, idx):
        row = self.rows[idx]
        frames_path, wav_path = _processed_paths(row)
        frames = np.load(frames_path)
        wav = _load_wav(wav_path)

        lo, hi = self._valid_frame_range(len(frames))
        frame_idx = random.randint(lo, hi)
        video_clip = torch.from_numpy(_video_window(frames, frame_idx))

        if self.mode == "cross_video":
            audio_clip = torch.from_numpy(_audio_window(wav, frame_idx))
            return video_clip, audio_clip

        # "window" mode: stack candidate audio windows at every delay in [-TAU, TAU]
        window = np.stack(
            [_audio_window(wav, frame_idx + d) for d in range(-TAU, TAU + 1)], axis=0
        )
        return video_clip, torch.from_numpy(window), TAU  # target index = zero delay


def sync_collate_cross_video(batch):
    videos, audios = zip(*batch)
    return torch.stack(videos), torch.stack(audios)


def sync_collate_window(batch):
    videos, audio_windows, targets = zip(*batch)
    return torch.stack(videos), torch.stack(audio_windows), torch.tensor(targets, dtype=torch.long)


class FeatureSequenceDataset(Dataset):
    """Reads per-utterance (T, DELAY_BINS) sequences from FEATURES_DIR/<split>/.
    In training mode (`full_sequence=False`) each item is a random length-`seq_len`
    window (edge-padded if the clip is shorter); in eval mode the whole sequence is
    returned as-is for calibrate.py/detect.py to score end-to-end."""

    def __init__(self, split, seq_len=SEQUENCE_LENGTH, full_sequence=False):
        self.dir = FEATURES_DIR / split
        self.paths = sorted(self.dir.rglob("*.npy"))
        if not self.paths:
            raise RuntimeError(f"No feature files under {self.dir} — run extract_features.py first.")
        self.seq_len = seq_len
        self.full_sequence = full_sequence

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        seq = np.load(self.paths[idx]).astype(np.float32)  # (T, DELAY_BINS)
        if self.full_sequence:
            return torch.from_numpy(seq), self.paths[idx].stem

        T = len(seq)
        if T >= self.seq_len:
            start = random.randint(0, T - self.seq_len)
            window = seq[start:start + self.seq_len]
        else:
            pad = np.repeat(seq[-1:], self.seq_len - T, axis=0)
            window = np.concatenate([seq, pad], axis=0)
        return torch.from_numpy(window)
