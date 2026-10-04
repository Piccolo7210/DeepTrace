"""CPU video detector: loads the exported bundle and scores a clip (plan section 9).

The pipeline mirrors the benchmark: adjacent frame pairs -> RAFT flow map ->
centre crops -> RGB and flow ResNet-50 -> alpha fusion -> mean over pairs ->
threshold. Differences from the benchmark (pairs spread over the whole clip,
reduced-scale flow, fewer pairs) are reported as caveats in every result.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from .config import CROP_SIZE, MAX_FLOW_PIXELS, MAX_VIDEO_FRAMES, RAFT_UPDATES
from .flow import load_raft
from .fuse import fuse, verdict, video_score
from .models import load_branch
from .scoring import PairScorer
from .verification import load_samples, verify_checksums, verify_parity
from .video_io import pick_pair_indices, probe_video, read_selected_frames

MIN_FRAMES = 5  # fewer decodable frames than this are rejected
CALIBRATION_PAIRS = 16  # pairs per video behind the shipped threshold
HIGH_FPS = 36.0  # clips above this frame rate were not validated (plan D17)
WINDOWS = ("whole", "first95")  # frame window the pairs are drawn from
PARITY_FILE = "parity_samples.npz"
META_FILE = "aigvdet_meta.json"


class Mode(NamedTuple):
    """A speed/accuracy setting: flow scale, RAFT updates and number of pairs."""

    flow_scale: float
    updates: int
    pairs: int
    description: str


# Provisional: only ``benchmark`` matches the validated protocol. The Kaggle speed
# ablation (plan D16) decides which reduced settings keep their accuracy.
MODES = {
    "benchmark": Mode(1.0, RAFT_UPDATES, CALIBRATION_PAIRS, "validated protocol, slowest"),
    "accurate": Mode(1.0, RAFT_UPDATES, 8, "native flow, 20 updates, 8 pairs"),
    "standard": Mode(0.75, 12, 12, "0.75x flow, 12 updates, 12 pairs (accuracy unmeasured)"),
    "quick": Mode(0.5, 12, 8, "0.5x flow, 12 updates, 8 pairs (accuracy unmeasured)"),
}


def _track_extremes(extremes: dict[str, dict], pair_idx: int, score: float, crops: dict) -> None:
    """Keep the crops of only the highest- and lowest-scoring pair seen so far."""
    entry = {"pair_idx": pair_idx, "score": score, **crops}
    if "most_suspicious" not in extremes or score > extremes["most_suspicious"]["score"]:
        extremes["most_suspicious"] = entry
    if "least_suspicious" not in extremes or score < extremes["least_suspicious"]["score"]:
        extremes["least_suspicious"] = entry


class ExportError(RuntimeError):
    """The model bundle is missing, incomplete or corrupt."""


def default_models_dir() -> Path:
    """``<project>/models`` next to the ``aigvdet`` package."""
    return Path(__file__).resolve().parents[1] / "models"


class AIGVDetDetector:
    """Two ResNet-50 branches plus RAFT, ready to score video clips on the CPU."""

    def __init__(self, rgb_model, flow_model, raft_model, meta: dict, models_dir: Path, device):
        self._models = (rgb_model, flow_model, raft_model)
        self.meta = meta
        self.models_dir = models_dir
        self.device = device

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"]["value"])

    def _scorer(self, updates: int, scale_limit: float, conditioned: bool = False) -> PairScorer:
        rgb, flow, raft = self._models
        return PairScorer(
            rgb,
            flow,
            raft,
            self.device,
            updates=updates,
            max_flow_pixels=self.meta["flow"].get("max_flow_pixels", MAX_FLOW_PIXELS),
            scale_limit=scale_limit,
            conditioned=conditioned,
        )

    def parity(self) -> pd.DataFrame:
        """Gate G5: re-score the stored Kaggle frame pairs with the exact benchmark settings."""
        samples = load_samples(self.models_dir / PARITY_FILE)
        scorer = self._scorer(self.meta["flow"]["raft_updates"], 1.0)
        return verify_parity(samples, scorer)

    def analyze(
        self,
        path: Path,
        mode: str = "accurate",
        window: str = "whole",
        progress: Callable[[int, int], None] | None = None,
        pairs: int | None = None,
        flow_scale: float | None = None,
        updates: int | None = None,
        keep_crops: bool = False,
    ) -> dict:
        """Score one clip. ``window='first95'`` reproduces the benchmark's frame window.

        ``keep_crops=True`` adds ``result["crops"]``: the 448x448 RGB and flow crops of the most
        and least suspicious pair (uint8 arrays, for the report figure). Off by default.
        """
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; choose from {sorted(MODES)}")
        if window not in WINDOWS:
            raise ValueError(f"Unknown window {window!r}; choose from {WINDOWS}")
        started = time.time()
        setting = MODES[mode]
        pairs = pairs or setting.pairs
        flow_scale = flow_scale or setting.flow_scale
        updates = updates or setting.updates

        info = probe_video(path)
        frame_count = info.frame_count
        if window == "first95":
            frame_count = min(frame_count, MAX_VIDEO_FRAMES)
        if frame_count < MIN_FRAMES:
            raise ValueError(f"Clip too short: {frame_count} frames (minimum {MIN_FRAMES})")
        indices = pick_pair_indices(frame_count, pairs)
        frames = read_selected_frames(path, {j for i in indices for j in (i, i + 1)})
        usable = [i for i in indices if i in frames and i + 1 in frames]
        if not usable:
            raise ValueError("None of the selected frames could be decoded")

        scorer = self._scorer(updates, flow_scale)
        alpha = float(self.meta["fusion_alpha"])
        rows = []
        extremes: dict[str, dict] = {}
        for done, index in enumerate(usable, start=1):
            row = scorer.score_pair(frames[index], frames[index + 1], keep_crops=keep_crops)
            crops = row.pop("crops", None)
            rows.append({"pair_idx": index, **row})
            if crops is not None:
                pair_score = float(fuse(row["p_rgb_raw"], row["p_flow_raw"], alpha))
                _track_extremes(extremes, index, pair_score, crops)
            if progress:
                progress(done, len(usable))
        table = pd.DataFrame(rows)

        fused = fuse(table["p_rgb_raw"], table["p_flow_raw"], alpha)
        score = video_score(fused)
        band = float(self.meta["threshold"]["uncertain_band"])
        label, level, flagged = verdict(score, self.threshold, band)
        height, width = frames[usable[0]].shape[:2]
        result = {
            "p_fake": score,
            "p_spatial": video_score(table["p_rgb_raw"]),
            "p_temporal": video_score(table["p_flow_raw"]),
            "alpha": alpha,
            "threshold": self.threshold,
            "threshold_source": self.meta["threshold"]["source"],
            "uncertain_band": band,
            "verdict": label,
            "verdict_level": level,
            "is_flagged": flagged,
            "pairs_used": len(usable),
            "pair_indices": usable,
            "frame_probs": {
                "spatial": table["p_rgb_raw"].tolist(),
                "temporal": table["p_flow_raw"].tolist(),
                "fused": fused.tolist(),
            },
            "flow_scales": table["flow_scale"].tolist(),
            "fps": info.fps,
            "resolution": [width, height],
            "frames_in_window": frame_count,
            "duration_seconds": frame_count / info.fps if info.fps > 0 else None,
            "mode": mode,
            "window": window,
            "settings": {"pairs": pairs, "flow_scale": flow_scale, "updates": updates},
            "seconds": time.time() - started,
        }
        if extremes:
            result["crops"] = extremes
        result["caveats"] = self._caveats(result, len(indices) - len(usable))
        result["limits"] = self.meta.get("known_limits", [])
        calibration = self.meta.get("calibration", {})
        result["calibration"] = {  # the bundle's own test numbers, for honest UI text
            "mean_fake_recall": calibration.get("mean_fake_recall_at_cal"),
            "fake_recall_by_set": calibration.get("fake_recall_by_set", {}),
            "fpr_on_report": calibration.get("fpr_on_report"),
        }
        return result

    def _caveats(self, result: dict, undecodable_pairs: int) -> list[str]:
        """Plain-language reasons why this result may differ from the benchmark."""
        notes = []
        width, height = result["resolution"]
        if min(width, height) < CROP_SIZE:
            rate = self.meta.get("calibration", {}).get("fpr_on_report")
            tail = (
                f" Real clips like this were flagged {rate:.0%} of the time in testing."
                if rate
                else ""
            )
            notes.append(
                f"Small clip ({width}x{height}): the networks saw zero-padded crops.{tail}"
            )
        if result["fps"] > HIGH_FPS:
            notes.append(f"Frame rate {result['fps']:.0f} fps is above the range that was tested.")
        if result["pairs_used"] < CALIBRATION_PAIRS:
            notes.append(
                f"Only {result['pairs_used']} frame pairs analysed (threshold calibrated on "
                f"{CALIBRATION_PAIRS}); the score is noisier."
            )
        lowest_scale = min(result["flow_scales"])
        if lowest_scale < 1.0:
            notes.append(
                f"Optical flow computed at reduced scale (down to {lowest_scale:.2f}x); "
                "accuracy at reduced scale has not been measured."
            )
        if result["window"] == "whole":
            notes.append(
                f"Pairs are spread over the whole clip; the benchmark used the first "
                f"{MAX_VIDEO_FRAMES} frames."
            )
        if undecodable_pairs:
            notes.append(f"{undecodable_pairs} selected frame pair(s) could not be decoded.")
        return notes


def load_detector(
    models_dir: Path | None = None, device: str = "cpu", verify: bool = True
) -> AIGVDetDetector:
    """Load the exported bundle; verifies checksums first so a truncated file fails loudly."""
    directory = Path(models_dir) if models_dir else default_models_dir()
    if not directory.is_dir():
        raise ExportError(f"{directory} does not exist - run tools\\pull_export.ps1")
    if verify:
        problems = verify_checksums(directory)
        if problems:
            listing = "\n  ".join(problems)
            raise ExportError(
                f"Model files failed verification:\n  {listing}\nRun tools\\pull_export.ps1"
            )
    meta = json.loads((directory / META_FILE).read_text())
    return AIGVDetDetector(
        load_branch(directory / "aigvdet_rgb.pth", device),
        load_branch(directory / "aigvdet_flow.pth", device),
        load_raft(device, weights_path=directory / "raft_large_C_T_V1.pth"),
        meta,
        directory,
        device,
    )
