"""Video -> 8 canonical frames (uint8, T x 3 x 224 x 224). Shared by Kaggle and the local app.

Canonical sampler (plan §4.3b + official NSG-VD stride rule):
    W = min(N, FRAME_WINDOW);  stride = W // T;  indices = 0, stride, ..., (T-1)*stride
Spatial policy (plan §4.5): bottom watermark band removed from EVERY video of BOTH classes ->
centre square crop -> 224x224 with PIL LANCZOS (the resampler the official code uses).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from nsgvd import config as C
from nsgvd.utils import natural_key

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".mpg", ".mpeg", ".wmv", ".flv", ".3gp", ".ts"}
GIF_EXTS = {".gif"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_RESAMPLE = getattr(Image, "Resampling", Image)   # Pillow >= 9.1 moved the enum
LANCZOS, BILINEAR = _RESAMPLE.LANCZOS, _RESAMPLE.BILINEAR


class VideoDecodeError(Exception):
    """The clip cannot be used (unreadable, or fewer than T frames)."""


def sample_indices(n_available: int, t: int | None = None, window: int | None = None) -> list[int]:
    t = t or C.T_FRAMES
    window = window or C.FRAME_WINDOW
    w = min(int(n_available), window)
    if w < t:
        raise VideoDecodeError(f"only {n_available} decodable frames (< {t})")
    stride = max(1, w // t)
    return [i * stride for i in range(t)]


def spatial_transform(rgb: np.ndarray, size: int | None = None, bottom_frac: float | None = None) -> np.ndarray:
    """(H, W, 3) uint8 -> (size, size, 3) uint8: watermark band -> centre square -> LANCZOS."""
    size = size or C.FEATURE_SIZE
    bottom_frac = C.WATERMARK_BOTTOM_FRAC if bottom_frac is None else bottom_frac
    h, w = rgb.shape[:2]
    keep_h = max(1, h - int(round(h * bottom_frac)))
    img = rgb[:keep_h]
    side = min(keep_h, w)
    top, left = (keep_h - side) // 2, (w - side) // 2
    img = np.ascontiguousarray(img[top:top + side, left:left + side])
    return np.asarray(Image.fromarray(img).resize((size, size), LANCZOS), dtype=np.uint8)


def _to_tchw(frames: list[np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(np.stack(frames).transpose(0, 3, 1, 2))


def _open_cv2(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise VideoDecodeError("cannot open video")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)   # honour rotation metadata (plan §7.4)
    return cap


def _read_video(path: Path, size: int, bottom_frac: float) -> tuple[np.ndarray, dict]:
    import cv2

    cap = _open_cv2(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc >> (8 * k)) & 0xFF) for k in range(4)).strip("\x00 ") or "?"
        n = 0   # pass 1: count decodable frames inside the window (metadata is unreliable)
        while n < C.FRAME_WINDOW and cap.grab():
            n += 1
    finally:
        cap.release()
    idx = sample_indices(n)
    wanted, frames, raw_hw = set(idx), {}, (0, 0)
    cap = _open_cv2(path)
    try:
        for i in range(idx[-1] + 1):   # pass 2: decode only up to the last needed frame
            if not cap.grab():
                raise VideoDecodeError(f"stream ended at frame {i} on second pass")
            if i in wanted:
                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    raise VideoDecodeError(f"cannot decode frame {i}")
                raw_hw = bgr.shape[:2]
                frames[i] = spatial_transform(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size, bottom_frac)
    finally:
        cap.release()
    info = {"n_window": n, "n_meta": n_meta, "fps": round(fps, 3), "height": int(raw_hw[0]),
            "width": int(raw_hw[1]), "codec": codec, "indices": idx}
    return _to_tchw([frames[i] for i in idx]), info


def _read_gif(path: Path, size: int, bottom_frac: float) -> tuple[np.ndarray, dict]:
    with Image.open(path) as im:
        n_total = int(getattr(im, "n_frames", 1))
        idx = sample_indices(n_total)
        frames = []
        for i in idx:   # increasing order: Pillow composites GIF frames correctly
            im.seek(i)
            frames.append(spatial_transform(np.asarray(im.convert("RGB")), size, bottom_frac))
        duration_ms = float(im.info.get("duration") or 0)
        width, height = im.size
    info = {"n_window": min(n_total, C.FRAME_WINDOW), "n_meta": n_total,
            "fps": round(1000.0 / duration_ms, 3) if duration_ms > 0 else 0.0,
            "height": height, "width": width, "codec": "gif", "indices": idx}
    return _to_tchw(frames), info


def _read_frame_folder(path: Path, size: int, bottom_frac: float) -> tuple[np.ndarray, dict]:
    files = sorted((p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=lambda p: natural_key(p.name))
    idx = sample_indices(len(files))
    frames = []
    for i in idx:
        with Image.open(files[i]) as im:
            frames.append(spatial_transform(np.asarray(im.convert("RGB")), size, bottom_frac))
    with Image.open(files[0]) as im:
        width, height = im.size
    info = {"n_window": min(len(files), C.FRAME_WINDOW), "n_meta": len(files), "fps": 0.0,
            "height": height, "width": width, "codec": files[0].suffix.lower().lstrip("."), "indices": idx}
    return _to_tchw(frames), info


def read_clip(path: str | Path, size: int | None = None, bottom_frac: float | None = None) -> tuple[np.ndarray, dict]:
    """Decode one clip (video file, GIF, or folder of frames) -> ((T,3,S,S) uint8, info)."""
    path = Path(path)
    size = size or C.FEATURE_SIZE
    bottom_frac = C.WATERMARK_BOTTOM_FRAC if bottom_frac is None else bottom_frac
    if not path.exists():
        raise VideoDecodeError("file missing")
    if path.is_dir():
        return _read_frame_folder(path, size, bottom_frac)
    if path.suffix.lower() in GIF_EXTS:
        return _read_gif(path, size, bottom_frac)
    return _read_video(path, size, bottom_frac)


def augment_clip(frames: np.ndarray, variant: str, seed: int) -> np.ndarray:
    """Deterministic nuisance variants of a (T,3,S,S) uint8 clip; same rule for both classes."""
    if variant == "clean":
        return frames
    rng = np.random.default_rng(seed)
    hwc = frames.transpose(0, 2, 3, 1)
    out = []
    if variant == "jpeg":
        lo, hi = C.JPEG_QUALITY_RANGE
        quality = int(rng.integers(lo, hi + 1))       # one quality per clip, like a real re-encode
        for f in hwc:
            buf = io.BytesIO()
            Image.fromarray(f).save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            with Image.open(buf) as im:
                out.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    elif variant == "rescale":
        s = hwc.shape[1]
        small = max(8, int(round(s * C.RESCALE_FACTOR)))
        for f in hwc:
            im = Image.fromarray(f).resize((small, small), LANCZOS).resize((s, s), BILINEAR)
            out.append(np.asarray(im, dtype=np.uint8))
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return _to_tchw(out)
