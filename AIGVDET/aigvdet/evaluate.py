"""Benchmark tables from cached per-pair probabilities (decision D7).

Everything here works on the ``probs_*.csv`` files written by ``extract.py``;
no model is run. Video score = mean over the video's pairs; the fused score is
``alpha * p_rgb + (1 - alpha) * p_flow`` (paper Eq. 4-5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import FUSION_ALPHA

SIGNALS = ("rgb", "flow", "fused")
THRESHOLDS = (0.5, 0.1)  # code default and the paper's text
_META = [
    "video_key",
    "set",
    "kind",
    "label",
    "half",
    "fps",
    "n_frames",
    "width",
    "height",
    "bucket",
]

# Paper Table III (ACC %, AUC %) per generator set - indicative only: the paper's reals differ.
PAPER_TABLE_III = {
    "T2V moonvalley": (95.1, 100.0),
    "T2V videocraft": (91.3, 97.0),
    "T2V pika": (89.5, 95.5),
    "T2V neverends": (89.5, 95.7),
    "T2V emu": (94.1, 99.3),
    "T2V videopoet": (93.8, 97.7),
    "T2V hotshot": (93.3, 98.0),
    "T2V sora": (82.3, 92.1),
    "I2V moonvalley": (89.0, 95.5),
    "I2V pika": (89.2, 96.1),
    "I2V neverends": (91.0, 96.6),
}


def load_probs(directory: Path, keys: set[str] | None = None) -> pd.DataFrame:
    """Concatenate every ``probs_*.csv`` shard in ``directory``.

    ``keys`` restricts the result to the current manifest's videos, so shards left
    over from an older manifest cannot leak in; duplicate (video, pair) rows from
    overlapping old and new shards are dropped.
    """
    files = sorted(directory.glob("probs_*.csv"))
    tables = [pd.read_csv(f, keep_default_na=False) for f in files if f.stat().st_size > 1]
    tables = [t for t in tables if len(t)]
    if not tables:
        raise FileNotFoundError(f"No probability shards in {directory}")
    pairs = pd.concat(tables, ignore_index=True)
    if keys is not None:
        pairs = pairs[pairs["video_key"].isin(keys)]
    return pairs.drop_duplicates(["video_key", "pair_idx"]).reset_index(drop=True)


def video_table(pairs: pd.DataFrame, variant: str, alpha: float = FUSION_ALPHA) -> pd.DataFrame:
    """One row per video: mean per-pair probabilities of both branches and their fusion."""
    scores = pairs.groupby("video_key")[[f"p_rgb_{variant}", f"p_flow_{variant}"]].mean()
    scores.columns = ["rgb", "flow"]
    scores["fused"] = alpha * scores["rgb"] + (1 - alpha) * scores["flow"]
    scores["n_pairs"] = pairs.groupby("video_key").size()
    meta = pairs.drop_duplicates("video_key").set_index("video_key")[_META[1:]]
    return meta.join(scores).reset_index()


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (NaN with a single class)."""
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, rounds: int = 500, seed: int = 0
) -> tuple[float, float, float]:
    """AUC with a 95% percentile interval, resampling videos within each class."""
    point = auc(labels, scores)
    if np.isnan(point):
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    fake = np.flatnonzero(labels == 1)
    real = np.flatnonzero(labels == 0)
    draws = []
    for _ in range(rounds):
        chosen = np.concatenate([rng.choice(fake, len(fake)), rng.choice(real, len(real))])
        draws.append(auc(labels[chosen], scores[chosen]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, float(low), float(high)


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Fake recall (TPR), real specificity (TNR) and their mean (balanced accuracy)."""
    predicted = scores > threshold
    tpr = float(predicted[labels == 1].mean()) if (labels == 1).any() else float("nan")
    tnr = float((~predicted[labels == 0]).mean()) if (labels == 0).any() else float("nan")
    return {"tpr": tpr, "tnr": tnr, "bacc": (tpr + tnr) / 2}


def paper_reference(set_name: str) -> tuple[float, float]:
    """Paper Table III (ACC, AUC) for a generator set, matched by name; NaN if absent."""
    lowered = set_name.lower()
    for key, value in PAPER_TABLE_III.items():
        generation, generator = key.split(" ")
        if lowered.startswith(generation.lower()) and generator in lowered:
            return value
    return (float("nan"), float("nan"))


def benchmark_table(videos: pd.DataFrame, rounds: int = 500) -> pd.DataFrame:
    """One row per (generator set, real pool) with AUC (+95% CI) and threshold metrics."""
    rows = []
    reals = dict(tuple(videos[videos["kind"] == "real"].groupby("set")))
    for set_name, fakes in videos[videos["kind"] == "fake"].groupby("set"):
        paper_acc, paper_auc = paper_reference(set_name)
        for pool, real_part in reals.items():
            both = pd.concat([fakes, real_part])
            labels = both["label"].to_numpy()
            row = {
                "set": set_name,
                "real_pool": pool,
                "n_fake": len(fakes),
                "n_real": len(real_part),
                "paper_acc": paper_acc,
                "paper_auc": paper_auc,
            }
            for signal in SIGNALS:
                scores = both[signal].to_numpy()
                point, low, high = bootstrap_auc(labels, scores, rounds)
                row.update({f"{signal}_auc": point, f"{signal}_auc_lo": low})
                row[f"{signal}_auc_hi"] = high
                for threshold in THRESHOLDS:
                    for name, value in threshold_metrics(labels, scores, threshold).items():
                        row[f"{signal}_{name}_{threshold}"] = value
            rows.append(row)
    return pd.DataFrame(rows)
