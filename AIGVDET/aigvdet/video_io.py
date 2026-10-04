"""Video decoding and frame-pair selection for test-time scoring."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import MAX_VIDEO_FRAMES, RESOLUTION_BUCKET_EDGES


@dataclass(frozen=True)
class VideoInfo:
    """Container metadata of a video file."""

    fps: float
    frame_count: int
    width: int
    height: int


def probe_video(path: Path) -> VideoInfo:
    """Read fps, frame count and frame size from the container."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        return VideoInfo(
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def read_frames(path: Path, max_frames: int = MAX_VIDEO_FRAMES) -> list[np.ndarray]:
    """Decode up to ``max_frames`` frames as RGB uint8 arrays."""
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No decodable frames in {path}")
    return frames


def evenly_spaced(count: int, k: int) -> list[int]:
    """Up to ``k`` evenly spaced indices from ``range(count)`` (all of them if ``k >= count``)."""
    if count <= 0 or k <= 0:
        return []
    chosen = np.round(np.linspace(0, count - 1, min(k, count))).astype(int)
    return np.unique(chosen).tolist()


def pick_pair_indices(n_frames: int, k: int) -> list[int]:
    """Start indices ``i`` of up to ``k`` evenly spaced adjacent pairs ``(i, i + 1)`` (D8)."""
    return evenly_spaced(n_frames - 1, k)


def resolution_bucket(width: int, height: int) -> str:
    """Bucket by the short side: ``<448``, ``448-719`` or ``>=720`` pixels."""
    short_side = min(width, height)
    low, high = RESOLUTION_BUCKET_EDGES
    if short_side < low:
        return f"<{low}"
    if short_side < high:
        return f"{low}-{high - 1}"
    return f">={high}"

def read_selected_frames(path: Path, indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Decode only the frames at ``indices`` (sequential grab/retrieve) as RGB uint8 arrays.

    Frames that cannot be decoded (a container that over-reports its length) are
    simply absent from the result; the caller decides what to do about them.
    """
    wanted = set(indices)
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(path))
    frames: dict[int, np.ndarray] = {}
    try:
        for position in range(max(wanted) + 1):
            if not capture.grab():
                break
            if position in wanted:
                ok, bgr = capture.retrieve()
                if ok:
                    frames[position] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    return frames
