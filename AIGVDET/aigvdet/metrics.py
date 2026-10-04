"""Classification metrics at frame and video level."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import DECISION_THRESHOLD


def binary_metrics(
    labels: np.ndarray | pd.Series,
    probs: np.ndarray | pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Accuracy at ``threshold``, ROC-AUC and average precision (NaN if one class only)."""
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probs, dtype=float)
    accuracy = float(((y_prob > threshold).astype(int) == y_true).mean())
    if len(np.unique(y_true)) < 2:
        return {"acc": accuracy, "auc": float("nan"), "ap": float("nan")}
    return {
        "acc": accuracy,
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
    }


def video_level(frame_table: pd.DataFrame) -> pd.DataFrame:
    """Average per-frame probabilities into one row per video.

    ``frame_table`` needs the columns ``label``, ``video_id`` and ``prob``.
    """
    grouped = frame_table.groupby(["label", "video_id"], sort=False)["prob"]
    summary = grouped.agg(["mean", "size"]).reset_index()
    return summary.rename(columns={"mean": "prob", "size": "n_frames"})
