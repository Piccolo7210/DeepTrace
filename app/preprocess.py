"""
Normalizes an arbitrary user upload into the 512x512 PNG form the AEROBLADE
threshold was calibrated on, before handing it to src/detect.py.

Why this exists rather than passing the raw upload straight to detect(): the
calibration corpus (data/real/, data/generated/) is uniformly 512x512, produced by
src/download_real.py's own resize-then-center-crop recipe. src/reconstruct.py's
Resize((512, 512)) squashes anything non-square to fit, silently distorting the
aspect ratio; that never mattered for the calibration corpus because every image in
it already was square, but a phone photo or screenshot from a GUI user usually
isn't. Pre-cropping here keeps src/reconstruct.py's calibration-critical behaviour
completely untouched while still giving the app well-behaved input.
"""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_SIZE = 512  # kept in sync with src/config.py's IMAGE_SIZE by the caller (engine.py)


@dataclass
class PrepResult:
    prepared_path: Path
    original_size: tuple  # (w, h) of the upload, before any resizing
    was_upscaled: bool  # shorter side was below IMAGE_SIZE -> resize is an upscale
    was_cropped: bool  # original aspect ratio wasn't already square


def _center_crop(img, size):
    """Same arithmetic as src/download_real.py's _center_crop, so an app upload and
    the calibration corpus are cropped identically for any given aspect ratio."""
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def prepare_input(src_path, out_dir, size=IMAGE_SIZE):
    """
    Loads src_path, flattens it to RGB, resizes its shorter side to `size` with
    LANCZOS resampling, center-crops to size x size, and saves the result as
    prepared_input.png in out_dir.

    Returns a PrepResult. Note `was_upscaled` describes the *original* upload's
    resolution — src/detect.py's own low-resolution check runs on the file it's
    given, and since the app hands it this already-512x512 PNG, that check would
    never fire. The caller (engine.py) uses was_upscaled to surface the warning the
    app is otherwise responsible for raising itself.
    """
    src_path = Path(src_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        img = img.convert("RGB")  # flattens alpha/palette/CMYK, matches load_image_tensor
        original_size = img.size
        w, h = original_size

        was_upscaled = min(w, h) < size
        was_cropped = w != h

        scale = size / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img = _center_crop(img, size)

        prepared_path = out_dir / "prepared_input.png"
        img.save(prepared_path, "PNG")

    return PrepResult(
        prepared_path=prepared_path,
        original_size=original_size,
        was_upscaled=was_upscaled,
        was_cropped=was_cropped,
    )
