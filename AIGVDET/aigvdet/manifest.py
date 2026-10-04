"""Test manifest: the videos scored by K4, sampled with a fixed seed from the Drive folders."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from . import remote
from .config import LABEL_FAKE, LABEL_REAL

MANIFEST_COLUMNS = [
    "video_key",
    "set",
    "kind",
    "label",
    "source",
    "remote_root",
    "relative",
    "half",
]
GENERATIONS = ("T2V", "I2V")
SET_FOLDER_SUFFIX = "_mp4"
VIDEO_PATTERN = "*.mp4"
SAMPLE_SEED = 42
HF_REAL_POOL = "R1 HF-val"


def sample_names(names: list[str], count: int, seed: int = SAMPLE_SEED) -> list[str]:
    """Seeded sample without replacement; everything (sorted) if ``count`` covers it."""
    ordered = sorted(names)
    if count >= len(ordered):
        return ordered
    chosen = np.random.default_rng(seed).choice(ordered, size=count, replace=False)
    return sorted(chosen.tolist())


def calibration_half(key: str) -> str:
    """Deterministic ``calib`` / ``report`` split of the reals (by hash of the key)."""
    return "calib" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 2 == 0 else "report"


def _drive_rows(
    set_name: str, kind: str, label: int, root: str, relatives: list[str]
) -> list[dict]:
    rows = []
    for relative in relatives:
        key = f"{set_name}/{PurePosixPath(relative).stem}"
        half = calibration_half(key) if label == LABEL_REAL else ""
        rows.append(
            {
                "video_key": key,
                "set": set_name,
                "kind": kind,
                "label": label,
                "source": "drive",
                "remote_root": root,
                "relative": relative,
                "half": half,
            }
        )
    return rows


def fake_rows(test_root: str, per_set: int) -> list[dict]:
    """Seeded sample of up to ``per_set`` videos from every generator set under ``test_root``."""
    rows: list[dict] = []
    for generation in GENERATIONS:
        for folder in remote.list_dirs(f"{test_root}/{generation}"):
            name = f"{generation} {folder.removesuffix(SET_FOLDER_SUFFIX)}"
            directory = f"{generation}/{folder}"
            files = remote.list_names(
                f"{test_root}/{directory}", recursive=True, include=VIDEO_PATTERN
            )
            chosen = sample_names(files, per_set)
            relatives = [f"{directory}/{name_}" for name_ in chosen]
            rows += _drive_rows(name, "fake", LABEL_FAKE, test_root, relatives)
    return rows


def real_pool_rows(pool: str, root: str, per_pool: int) -> list[dict]:
    """Seeded sample of up to ``per_pool`` real videos from a Drive folder."""
    files = remote.list_names(root, recursive=True, include=VIDEO_PATTERN)
    return _drive_rows(pool, "real", LABEL_REAL, root, sample_names(files, per_pool))


def hf_real_rows(index: pd.DataFrame) -> list[dict]:
    """The HF validation reals (pool R1): scored from their stored JPEG frames, no decoding."""
    part = index[(index["split"] == "val") & (index["kind"] == "rgb") & (index["label"] == 0)]
    rows = []
    for video_id in sorted(part["video_id"].unique()):
        rows.append(
            {
                "video_key": f"{HF_REAL_POOL}/{video_id}",
                "set": HF_REAL_POOL,
                "kind": "real",
                "label": LABEL_REAL,
                "source": "hf_frames",
                "remote_root": "",
                "relative": video_id,
                "half": "",
            }
        )
    return rows


def build_manifest(
    test_root: str, per_set: int, real_pools: dict[str, str], per_pool: int, index: pd.DataFrame
) -> pd.DataFrame:
    """Fakes from every generator set + reals from R1 and every reachable Drive pool."""
    rows = fake_rows(test_root, per_set) + hf_real_rows(index)
    for pool, root in real_pools.items():
        try:
            rows += real_pool_rows(pool, root, per_pool)
        except remote.RcloneError as error:
            print(f"Real pool '{pool}' skipped: {error}")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
