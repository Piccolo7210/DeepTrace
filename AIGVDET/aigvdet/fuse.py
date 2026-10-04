"""Frame-pair -> video aggregation, alpha fusion and verdict bands (paper Eq. 4-5)."""

from __future__ import annotations

import numpy as np

from .config import FUSION_ALPHA

VERDICT_FLAGGED = "AI-generated"
VERDICT_UNCERTAIN = "Uncertain"
VERDICT_CLEAR = "Consistent with real"


def fuse(
    p_rgb: float | np.ndarray, p_flow: float | np.ndarray, alpha: float = FUSION_ALPHA
) -> float | np.ndarray:
    """Weighted decision-level fusion ``alpha * p_rgb + (1 - alpha) * p_flow``."""
    return alpha * np.asarray(p_rgb, dtype=float) + (1 - alpha) * np.asarray(p_flow, dtype=float)


def video_score(pair_scores: np.ndarray | list[float]) -> float:
    """Video-level probability: the mean over the analysed frame pairs."""
    return float(np.mean(pair_scores))


def verdict(score: float, threshold: float, band: float) -> tuple[str, str, bool]:
    """Return ``(label, level, is_flagged)`` for a video score.

    ``is_flagged`` is the plain threshold decision. The label adds an *uncertain*
    zone of +-``band`` around the threshold: above it the clip is called
    AI-generated, below it consistent with real.
    """
    if score > threshold + band:
        return VERDICT_FLAGGED, "flagged", True
    if score < threshold - band:
        return VERDICT_CLEAR, "clear", False
    return VERDICT_UNCERTAIN, "uncertain", score > threshold
