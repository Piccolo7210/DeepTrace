"""Per-pair scoring: RAFT flow map + both ResNet-50 branches, raw and JPEG-conditioned."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn

from .config import COND_FLOW_JPEG_QUALITY, COND_RGB_JPEG_QUALITY, MAX_FLOW_PIXELS, RAFT_UPDATES
from .flow import clamp_scale, flow_color_map, flow_scale
from .preprocess import center_crop_or_pad, jpeg_round_trip, normalise_on_device, to_chw_tensor


def map_statistics(flow_crop: np.ndarray) -> dict[str, float]:
    """Mean HSV saturation and value (0-255) of a flow colour map - a cheap recipe fingerprint."""
    hsv = cv2.cvtColor(flow_crop, cv2.COLOR_RGB2HSV)
    return {"map_saturation": float(hsv[..., 1].mean()), "map_value": float(hsv[..., 2].mean())}


class PairScorer:
    """Scores adjacent frame pairs with the RGB branch, the flow branch and RAFT.

    Every pair yields four probabilities (``p_rgb_raw``, ``p_rgb_jpeg``,
    ``p_flow_raw``, ``p_flow_jpeg``): the *jpeg* variants round-trip the full
    frame / flow map through JPEG before the centre crop (decision D11). Local
    inference sets ``conditioned=False`` to compute the raw variants only, and
    ``scale_limit < 1`` to run RAFT at reduced scale for speed.
    """

    def __init__(
        self,
        rgb_model: nn.Module,
        flow_model: nn.Module,
        raft_model: nn.Module,
        device: torch.device | str,
        updates: int = RAFT_UPDATES,
        max_flow_pixels: int | None = MAX_FLOW_PIXELS,
        scale_limit: float = 1.0,
        conditioned: bool = True,
    ) -> None:
        self.rgb_model = rgb_model
        self.flow_model = flow_model
        self.raft_model = raft_model
        self.device = torch.device(device)
        self.updates = updates
        self.max_flow_pixels = max_flow_pixels
        self.scale_limit = scale_limit
        self.conditioned = conditioned

    @torch.no_grad()
    def _probabilities(self, model: nn.Module, crops: list[np.ndarray]) -> list[float]:
        batch = torch.stack([to_chw_tensor(crop) for crop in crops])
        logits = model(normalise_on_device(batch, self.device)).squeeze(1)
        return torch.sigmoid(logits.float()).cpu().tolist()

    def flow_probability(self, flow_map: np.ndarray) -> float:
        """Flow-branch probability of an existing colour map (e.g. a stored training map)."""
        return self._probabilities(self.flow_model, [center_crop_or_pad(flow_map)])[0]

    def score_pair(
        self, frame_a: np.ndarray, frame_b: np.ndarray, keep_crops: bool = False
    ) -> dict[str, float]:
        """Probabilities for one pair; ``keep_crops`` adds the raw crops under ``"crops"``."""
        height, width = frame_a.shape[:2]
        scale = min(flow_scale(height, width, self.max_flow_pixels), self.scale_limit)
        scale = clamp_scale(scale, height, width)
        flow_map = flow_color_map(
            self.raft_model, frame_a, frame_b, self.device, self.updates, scale
        )
        rgb_crops = [center_crop_or_pad(frame_a)]
        flow_crops = [center_crop_or_pad(flow_map)]
        if self.conditioned:
            rgb_crops.append(center_crop_or_pad(jpeg_round_trip(frame_a, COND_RGB_JPEG_QUALITY)))
            flow_crops.append(center_crop_or_pad(jpeg_round_trip(flow_map, COND_FLOW_JPEG_QUALITY)))
        p_rgb = self._probabilities(self.rgb_model, rgb_crops)
        p_flow = self._probabilities(self.flow_model, flow_crops)
        row = {"p_rgb_raw": p_rgb[0], "p_flow_raw": p_flow[0], "flow_scale": scale}
        if self.conditioned:
            row.update({"p_rgb_jpeg": p_rgb[1], "p_flow_jpeg": p_flow[1]})
        row.update(map_statistics(flow_crops[0]))
        if keep_crops:
            row["crops"] = {"rgb": rgb_crops[0], "flow": flow_crops[0]}
        return row

    def score_pairs(
        self, frames: Sequence[np.ndarray] | Mapping[int, np.ndarray], pair_indices: list[int]
    ) -> list[dict[str, float]]:
        """Score pairs ``(i, i + 1)`` for every start index ``i`` in ``pair_indices``."""
        rows = []
        for index in pair_indices:
            row = self.score_pair(frames[index], frames[index + 1])
            rows.append({"pair_idx": index, **row})
        return rows
