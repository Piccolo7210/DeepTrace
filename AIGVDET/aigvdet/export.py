"""Packaging of the trained branches and everything local inference needs (plan K6).

Besides the weights, the export carries *parity samples*: a few video frame pairs
together with the probabilities Kaggle computed for them (see ``verification``).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import remote
from .data import read_rgb
from .flow import load_raft
from .models import load_branch
from .video_io import evenly_spaced, read_frames

PARITY_SETS = ("T2V moonvalley", "T2V hotshot", "I2V pika", "R2 Kinetics", "R1 HF-val")
PAIRS_PER_PARITY_VIDEO = 3


def package_hash(package_dir: Path) -> str:
    """Hash of the package sources, so an export can be tied to the code that made it."""
    digest = hashlib.sha256()
    for source in sorted(package_dir.glob("*.py")):
        digest.update(source.name.encode() + source.read_bytes())
    return digest.hexdigest()[:16]


def export_branch_weights(checkpoint: Path, target: Path) -> dict:
    """Write a plain fp32 state-dict from a trainer checkpoint and prove it reloads."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.save(payload["model"], target)
    load_branch(target)
    return {"epoch": payload["epoch"], "val": payload.get("val")}


def save_raft_weights(target: Path) -> None:
    """Save the pretrained RAFT-large (``C_T_V1``) weights and prove the file reloads."""
    torch.save(load_raft("cpu").state_dict(), target)
    load_raft("cpu", weights_path=target)


def pick_parity_rows(
    pairs: pd.DataFrame,
    sets: tuple[str, ...] = PARITY_SETS,
    per_video: int = PAIRS_PER_PARITY_VIDEO,
) -> pd.DataFrame:
    """Per-pair rows of one video from each set (names matched case-insensitively)."""
    chosen = []
    lowered = pairs["set"].str.lower()
    for set_name in sets:
        part = pairs[lowered == set_name.lower()]
        if part.empty:
            continue
        video = part[part["video_key"] == sorted(part["video_key"].unique())[0]]
        video = video.sort_values("pair_idx")
        chosen.append(video.iloc[evenly_spaced(len(video), per_video)])
    return pd.concat(chosen, ignore_index=True) if chosen else pairs.iloc[0:0]


def _load_frames(
    row: pd.Series, needed: set[int], cache: Path, index: pd.DataFrame, data_root: Path
) -> dict[int, np.ndarray]:
    """Frames ``needed`` of one manifest video (decoded from Drive or read from the HF frames)."""
    if row["source"] == "hf_frames":
        part = index[
            (index["split"] == "val")
            & (index["kind"] == "rgb")
            & (index["label"] == 0)
            & (index["video_id"] == row["relative"])
        ].sort_values("frame_idx")
        return {j: read_rgb(data_root / part.iloc[j]["path"]) for j in needed}
    remote.copy_files(row["remote_root"], [row["relative"]], cache)
    decoded = read_frames(cache / row["relative"])
    return {j: decoded[j] for j in needed}


def build_parity_samples(
    rows: pd.DataFrame, manifest: pd.DataFrame, index: pd.DataFrame, data_root: Path, cache: Path
) -> dict[str, np.ndarray]:
    """Frame pairs plus Kaggle's raw-input probabilities for the chosen rows (npz-ready)."""
    manifest = manifest.set_index("video_key")
    samples: dict[str, np.ndarray] = {}
    for number, (key, group) in enumerate(rows.groupby("video_key", sort=False)):
        pair_idx = group["pair_idx"].astype(int).tolist()
        needed = {j for i in pair_idx for j in (i, i + 1)}
        frames = _load_frames(manifest.loc[key], needed, cache, index, data_root)
        prefix = f"s{number}"
        samples[f"{prefix}_key"] = np.array(key)
        samples[f"{prefix}_pair_idx"] = np.array(pair_idx)
        samples[f"{prefix}_frames"] = np.stack(
            [np.stack([frames[i], frames[i + 1]]) for i in pair_idx]
        )
        samples[f"{prefix}_p_rgb"] = group["p_rgb_raw"].to_numpy(dtype=float)
        samples[f"{prefix}_p_flow"] = group["p_flow_raw"].to_numpy(dtype=float)
    shutil.rmtree(cache, ignore_errors=True)
    return samples
