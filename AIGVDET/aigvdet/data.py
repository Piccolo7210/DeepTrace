"""Training-data handling: download, indexing, validation and loading.

Layout produced by :func:`extract_archives` and consumed everywhere else::

    <data_root>/<split>/<kind>/[<archive stem>/]<0_real|1_fake>/<video_id>_<frame>.jpg

``split`` is ``train`` or ``val``; ``kind`` is ``rgb`` or ``flow``.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import (
    AUG_BLUR_PROB,
    AUG_BLUR_SIGMA,
    AUG_FLIP_PROB,
    AUG_JPEG_PROB,
    AUG_JPEG_QUALITY,
    CLASS_DIRS,
    CROP_SIZE,
    EXPECTED_FRAMES_PER_VIDEO,
    EXPECTED_VIDEOS_PER_CLASS,
    HF_ARCHIVES,
    HF_REPO_ID,
    HF_REVISION,
    IMAGE_EXTENSIONS,
    LABEL_FAKE,
    LABEL_REAL,
)
from .preprocess import center_crop_or_pad, jpeg_round_trip, random_crop, to_chw_tensor

INDEX_COLUMNS = ["path", "kind", "label", "split", "video_id", "frame_idx", "width", "height"]
_FRAME_NAME = re.compile(r"^(?P<video>.+)_(?P<frame>\d+)$")
_EXTRACTION_MARKER = ".extracted.json"


# --------------------------------------------------------------------------- #
# Download and extraction
# --------------------------------------------------------------------------- #
def download_archives(destination: Path) -> Path:
    """Download the four pinned HF archives into ``destination``."""
    from huggingface_hub import snapshot_download  # imported lazily: optional at inference

    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=str(destination),
        allow_patterns=["train/*.zip", "val/*.zip"],
        max_workers=8,
    )
    return Path(local_dir)


def _extract_one(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(target)
    archive.unlink()  # free disk space as soon as the archive is unpacked


def is_extracted(data_root: Path) -> bool:
    """True if ``data_root`` holds a completed extraction of the pinned revision."""
    marker = data_root / _EXTRACTION_MARKER
    return marker.exists() and json.loads(marker.read_text()).get("revision") == HF_REVISION


def extract_archives(archive_root: Path, data_root: Path) -> None:
    """Unzip every archive into ``<data_root>/<split>/<kind>/`` and delete the zips."""
    jobs = [
        (archive_root / relative, data_root / split / kind)
        for (split, kind), relative in HF_ARCHIVES.items()
    ]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for future in [pool.submit(_extract_one, *job) for job in jobs]:
            future.result()  # re-raises extraction errors
    (data_root / _EXTRACTION_MARKER).write_text(json.dumps({"revision": HF_REVISION}))


# --------------------------------------------------------------------------- #
# Indexing and validation
# --------------------------------------------------------------------------- #
def _describe_frame(path: Path, root: Path) -> dict | None:
    """Return the index row of one image file, or None if it is not a dataset frame."""
    if path.name.startswith(".") or "__MACOSX" in path.parts:
        return None
    relative = path.relative_to(root)
    label = CLASS_DIRS.get(path.parent.name)
    match = _FRAME_NAME.match(path.stem)
    if label is None or match is None or len(relative.parts) < 4:
        return None
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": relative.as_posix(),
        "kind": relative.parts[1],
        "label": label,
        "split": relative.parts[0],
        "video_id": match["video"],
        "frame_idx": int(match["frame"]),
        "width": width,
        "height": height,
    }


def index_frames(root: Path, workers: int = 32) -> pd.DataFrame:
    """Scan ``root`` and return one row per frame (see ``INDEX_COLUMNS``)."""
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [row for row in pool.map(lambda p: _describe_frame(p, root), files) if row]
    frame = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    return frame.sort_values(["split", "kind", "label", "video_id", "frame_idx"]).reset_index(
        drop=True
    )


def validate_index(index: pd.DataFrame) -> list[str]:
    """Gate G0: return a list of problems (empty list means the data is as expected)."""
    problems: list[str] = []
    for split, expected_videos in EXPECTED_VIDEOS_PER_CLASS.items():
        for kind, expected_frames in EXPECTED_FRAMES_PER_VIDEO.items():
            part = index[(index["split"] == split) & (index["kind"] == kind)]
            for label in (LABEL_REAL, LABEL_FAKE):
                counts = part[part["label"] == label].groupby("video_id").size()
                tag = f"{split}/{kind}/label={label}"
                if len(counts) != expected_videos:
                    problems.append(f"{tag}: {len(counts)} videos, expected {expected_videos}")
                wrong = counts[counts != expected_frames]
                if len(wrong):
                    problems.append(f"{tag}: {len(wrong)} videos without {expected_frames} frames")
    small = index[(index["width"] < CROP_SIZE) | (index["height"] < CROP_SIZE)]
    if len(small):
        problems.append(f"{len(small)} images smaller than {CROP_SIZE}px on a side")
    return problems


def select_frames(index: pd.DataFrame, split: str, kind: str) -> pd.DataFrame:
    """Rows of one split/kind with a fresh 0..n-1 index (required by the sampler)."""
    mask = (index["split"] == split) & (index["kind"] == kind)
    return index[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Augmentation (official order: blur -> JPEG -> crop -> flip)
# --------------------------------------------------------------------------- #
def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Spatial Gaussian blur of an HWC image (channels are not mixed)."""
    return gaussian_filter(image, sigma=(sigma, sigma, 0))


def augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the training-time augmentation chain to a full frame."""
    if rng.random() < AUG_BLUR_PROB:
        image = gaussian_blur(image, AUG_BLUR_SIGMA)
    if rng.random() < AUG_JPEG_PROB:
        image = jpeg_round_trip(image, AUG_JPEG_QUALITY)
    image = random_crop(image, rng)
    if rng.random() < AUG_FLIP_PROB:
        image = image[:, ::-1]
    return image


def read_rgb(path: Path) -> np.ndarray:
    """Read an image file as an RGB ``uint8`` array."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Dataset, sampler, loader
# --------------------------------------------------------------------------- #
class FrameDataset(Dataset):
    """Independent frames with a binary label (official ``ImageFolder`` semantics, D3)."""

    def __init__(self, frames: pd.DataFrame, root: Path, train: bool) -> None:
        self._paths = [root / relative for relative in frames["path"]]
        self._labels = frames["label"].to_numpy(dtype=np.float32)
        self._train = train
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self._paths)

    def _worker_rng(self) -> np.random.Generator:
        if self._rng is None:  # created lazily so every DataLoader worker gets its own seed
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = read_rgb(self._paths[position])
        if self._train:
            image = augment(image, self._worker_rng())
        else:
            image = center_crop_or_pad(image)
        return to_chw_tensor(image), torch.tensor(self._labels[position])


class VideoBalancedSampler(Sampler[int]):
    """Shuffled frame sampler; optionally draws a fixed number of frames per video.

    ``frames_per_video=None`` yields every frame once per epoch (``paper`` profile);
    an integer yields that many random frames per video (``budget`` profile).
    Requires ``frames`` to carry a default RangeIndex.
    """

    def __init__(self, frames: pd.DataFrame, frames_per_video: int | None, seed: int) -> None:
        self._groups = list(frames.groupby(["label", "video_id"]).indices.values())
        self._total = len(frames)
        self._per_video = frames_per_video
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        if self._per_video is None:
            return self._total
        return sum(min(self._per_video, len(group)) for group in self._groups)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed + self._epoch)
        if self._per_video is None:
            return iter(rng.permutation(self._total).tolist())
        chosen = [
            rng.choice(group, size=min(self._per_video, len(group)), replace=False)
            for group in self._groups
        ]
        return iter(rng.permutation(np.concatenate(chosen)).tolist())


def _init_worker(_: int) -> None:
    cv2.setNumThreads(0)  # one decode thread per worker; parallelism comes from workers


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    sampler: Sampler[int] | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader with the settings used throughout the project."""
    options = {}
    if num_workers > 0:
        options = {"persistent_workers": True, "prefetch_factor": 4}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=_init_worker,
        **options,
    )
