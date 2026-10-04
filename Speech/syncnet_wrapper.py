"""
Raw video -> face track -> S(i,.) sync-feature sequence, backed by the vendored
syncnet_python (vendor/syncnet_python/) + the official pretrained syncnet_v2.model /
sfd_face.pth checkpoints in models/. This is the feature-extraction front-end for
detect_pretrained.py, and mirrors exactly what Speech/kaggle/speechcode.ipynb did to
produce the training data for models/speech_forensics_model.pt.

Two stages:

  1. Face detect + track + crop (run_face_crop): shells out to the vendored
     run_pipeline.py as a subprocess. This can't be imported in-process -- it runs
     `opt = parser.parse_args()` at module level with no `if __name__ == "__main__":`
     guard, then executes the whole scene-detect/face-detect/track/crop pipeline
     unconditionally as a side effect of the `import` statement itself (would stomp
     this process's own sys.argv and start deleting directories / shelling out to
     ffmpeg / running S3FD on import). This repo already solved the same shape of
     problem once in kaggle/score_dual_gpu.py, for the same reason: subprocess, not
     multiprocessing.Process or a plain import.
  2. Sync feature extraction (extract_sync_features): SyncNetInstance has no such
     issue -- it's a plain class with no module-level side effects -- so it runs
     in-process, loaded once via load_syncnet() and reused across analyses.

Requires ffmpeg on PATH (both stages shell out to it) and a video file with an
embedded audio track -- SyncNetInstance.evaluate() extracts audio directly from
whatever video file it's given.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    MIN_FACE_SIZE_PX,
    MIN_TRACK_FRAMES,
    SFD_CKPT_PATH,
    SYNCNET_CKPT_PATH,
    SYNCNET_VENDOR_DIR,
    TAU,
)


def load_syncnet(device="cpu"):
    """Constructs a SyncNetInstance and loads syncnet_v2.model. Call once and reuse
    across analyses (mirrors app/engine.py's load-once pattern for the AEROBLADE
    autoencoders) -- construction + weight loading is the expensive part, not eval."""
    vendor_dir = str(SYNCNET_VENDOR_DIR)
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    from SyncNetInstance import SyncNetInstance  # vendored; see module docstring

    net = SyncNetInstance(device=device)
    net.loadParameters(str(SYNCNET_CKPT_PATH))
    net.eval()
    return net


class _Opt:
    """SyncNetInstance.evaluate() reads exactly these four attributes off its `opt`
    argument (vendor/syncnet_python/SyncNetInstance.py) -- nothing else."""

    def __init__(self, tmp_dir, reference, batch_size, vshift):
        self.tmp_dir = str(tmp_dir)
        self.reference = reference
        self.batch_size = batch_size
        self.vshift = vshift


def run_face_crop(video_path, work_dir, reference="clip",
                   min_track=MIN_TRACK_FRAMES, min_face_size=MIN_FACE_SIZE_PX):
    """Runs the vendored run_pipeline.py (scene-detect -> S3FD face-detect -> IOU
    face-tracking -> crop) as a subprocess against video_path. Returns the resulting
    pycrop/<reference>/*.avi paths -- each one a face track with audio muxed back in
    by the pipeline's own crop step. Raises RuntimeError (with the subprocess's
    stderr tail, matching detect.py's existing error convention) if none were
    produced -- e.g. no face detected, or every track shorter than min_track.
    """
    work_dir = Path(work_dir)
    env = dict(os.environ)
    env["SFD_WEIGHTS_PATH"] = str(SFD_CKPT_PATH)

    cmd = [
        sys.executable, "run_pipeline.py",
        "--videofile", str(Path(video_path).resolve()),
        "--reference", reference,
        "--data_dir", str(work_dir.resolve()),
        "--min_track", str(min_track),
        "--min_face_size", str(min_face_size),
        "--overwrite",
    ]
    result = subprocess.run(
        cmd, cwd=str(SYNCNET_VENDOR_DIR), env=env,
        capture_output=True, text=True,
    )

    crop_dir = work_dir / "pycrop" / reference
    avis = sorted(crop_dir.glob("*.avi")) if crop_dir.is_dir() else []
    if not avis:
        stderr_tail = (result.stderr or "")[-1500:]
        raise RuntimeError(
            f"No usable face track found in {video_path} "
            f"(min_track={min_track} frames, min_face_size={min_face_size}px). "
            f"If this is a real, non-GRID clip, try loosening those two settings "
            f"first.\nrun_pipeline.py stderr tail:\n{stderr_tail}"
        )
    return avis


def select_longest_track(avi_paths):
    """Multi-face videos yield multiple tracks; picks the one with the most frames
    (the most-visible face) rather than the first one found -- a deliberate
    improvement over the training notebook's own avis[0], which just took whichever
    track glob() happened to sort first. Fine for GRID's single-speaker clips, not a
    safe assumption for an arbitrary user upload."""
    if len(avi_paths) == 1:
        return avi_paths[0]

    def frame_count(path):
        cap = cv2.VideoCapture(str(path))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    return max(avi_paths, key=frame_count)


@dataclass
class SyncFeatures:
    """S(i,.) plus the two scalars SyncNet computes alongside it. The scalars are
    kept rather than discarded because they're independently informative about
    audio-visual alignment and cost nothing extra: on a correctly-synced clip
    offset lands at 0 with a high confidence, and on a desynced one the offset moves
    and confidence collapses (measured on a GRID clip: offset 0 / conf 5.46 real vs
    offset -15 / conf 0.82 after shifting the audio 1.5s)."""

    features: np.ndarray  # (T, 2*tau+1) float32, rows sum to 1
    offset: int           # best global A/V offset in frames; 0 = perfectly synced
    confidence: float     # SyncNet's own sync confidence; higher = more confident
    num_face_tracks: int = 1


def extract_sync_features(net, avi_path, work_dir, reference="clip", tau=TAU, batch_size=20):
    """Runs the already-loaded SyncNetInstance over one face-track .avi (video +
    audio muxed together by run_face_crop). Returns a SyncFeatures whose .features
    is S(i,.): (T, 2*tau+1) float32, softmax(-L2 distance) over 2*tau+1 offsets per
    frame (paper Eq. 1) -- same quantity, same conversion, as speechcode.ipynb's
    extract_worker.
    """
    opt = _Opt(tmp_dir=Path(work_dir) / "synctmp", reference=reference,
               batch_size=batch_size, vshift=tau)
    offset, conf, dists = net.evaluate(opt, videofile=str(Path(avi_path).resolve()))
    probs = F.softmax(torch.from_numpy(-np.stack(dists)), dim=-1).numpy()
    return SyncFeatures(
        features=probs.astype(np.float32),
        offset=int(offset),
        confidence=float(conf),
    )


def extract_from_video(net, video_path, work_dir, tau=TAU,
                        min_track=MIN_TRACK_FRAMES, min_face_size=MIN_FACE_SIZE_PX,
                        batch_size=20):
    """Convenience wrapper: run_face_crop -> select_longest_track ->
    extract_sync_features. Returns a SyncFeatures with num_face_tracks filled in."""
    avis = run_face_crop(video_path, work_dir, min_track=min_track, min_face_size=min_face_size)
    track = select_longest_track(avis)
    extracted = extract_sync_features(net, track, work_dir, tau=tau, batch_size=batch_size)
    extracted.num_face_tracks = len(avis)
    return extracted
