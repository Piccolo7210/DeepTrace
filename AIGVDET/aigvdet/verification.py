"""Verification of an exported model bundle: checksums and Kaggle parity samples.

Used on Kaggle when the bundle is written and locally before it is trusted
(plan gate G5): file integrity first, then a re-score of stored frame pairs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FUSION_ALPHA
from .scoring import PairScorer

CHECKSUM_FILE = "SHA256SUMS.txt"
PARITY_TOLERANCE = 0.03  # plan gate G5: largest allowed |delta p| per pair


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(directory: Path) -> list[Path]:
    """Write ``SHA256SUMS.txt`` for every file below ``directory``; returns all files."""
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != CHECKSUM_FILE)
    lines = [f"{sha256_of(p)}  {p.relative_to(directory).as_posix()}" for p in files]
    (directory / CHECKSUM_FILE).write_text("\n".join(lines) + "\n")
    return [*files, directory / CHECKSUM_FILE]


def verify_checksums(directory: Path) -> list[str]:
    """Problems found when checking ``directory`` against its checksum file (empty = intact)."""
    listing = directory / CHECKSUM_FILE
    if not listing.exists():
        return [f"{CHECKSUM_FILE} is missing"]
    problems = []
    for line in listing.read_text().splitlines():
        if not line.strip():
            continue
        expected, _, relative = line.partition("  ")
        path = directory / relative.strip()
        if not path.exists():
            problems.append(f"{relative.strip()}: missing")
        elif sha256_of(path) != expected.strip().lower():
            problems.append(f"{relative.strip()}: checksum mismatch (truncated or corrupt?)")
    return problems


def load_samples(path: Path) -> dict[str, np.ndarray]:
    """Read ``parity_samples.npz`` (plain arrays only, no pickles)."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def verify_parity(samples: dict[str, np.ndarray], scorer: PairScorer) -> pd.DataFrame:
    """Re-score the stored frame pairs and return the |delta p| per pair and branch."""
    records = []
    for name in sorted(k for k in samples if k.endswith("_key")):
        prefix = name.removesuffix("_key")
        for position, pair_idx in enumerate(samples[f"{prefix}_pair_idx"]):
            first, second = samples[f"{prefix}_frames"][position]
            new = scorer.score_pair(first, second)
            old_rgb = samples[f"{prefix}_p_rgb"][position]
            old_flow = samples[f"{prefix}_p_flow"][position]
            fused_old = FUSION_ALPHA * old_rgb + (1 - FUSION_ALPHA) * old_flow
            fused_new = FUSION_ALPHA * new["p_rgb_raw"] + (1 - FUSION_ALPHA) * new["p_flow_raw"]
            records.append(
                {
                    "video": str(samples[name]),
                    "pair_idx": int(pair_idx),
                    "d_rgb": abs(new["p_rgb_raw"] - old_rgb),
                    "d_flow": abs(new["p_flow_raw"] - old_flow),
                    "d_fused": abs(fused_new - fused_old),
                }
            )
    return pd.DataFrame(records)
