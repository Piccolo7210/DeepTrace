"""RAFT optical flow with the authors' ``demo.py`` recipe (plan decision D2).

Frames are padded to a multiple of 8 with replicated borders (RAFT's
``InputPadder``), flow is estimated with 20 updates, colour-coded with the
official wheel and - as in the authors' demo - *not* cropped back to the
original size, so a 1184x636 frame yields a 1184x640 map.

RAFT's memory grows with the square of the pixel count. Frames above
``max_pixels`` are therefore processed at a reduced scale: the flow field is
estimated on the shrunken pair, resized to the native padded size with the
vectors rescaled, and only then colour-coded, so the crop geometry is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from .config import RAFT_UPDATES
from .flow_viz import flow_to_image

_PADDING_MULTIPLE = 8
_MIN_SIDE = 128  # torchvision's RAFT needs feature maps of at least 16 px (= 128 px / 8)


def load_raft(
    device: torch.device | str, weights_path: Path | None = None, pretrained: bool = True
) -> nn.Module:
    """RAFT-large in eval mode: ``C_T_V1`` weights, a local state-dict, or random (tests)."""
    if weights_path is not None:
        model = raft_large(weights=None, progress=False)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    else:
        weights = Raft_Large_Weights.C_T_V1 if pretrained else None
        model = raft_large(weights=weights, progress=False)
    return model.to(device).eval()


def pad_to_multiple_of_8(batch: torch.Tensor) -> torch.Tensor:
    """Replicate-pad an NCHW batch like RAFT's ``InputPadder`` (split evenly on both sides)."""
    height, width = batch.shape[-2:]
    pad_h = (-height) % _PADDING_MULTIPLE
    pad_w = (-width) % _PADDING_MULTIPLE
    padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
    return functional.pad(batch, padding, mode="replicate")


def _to_raft_input(frame: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> CHW float in [-1, 1], the range torchvision's RAFT expects."""
    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0


def padded_size(height: int, width: int) -> tuple[int, int]:
    """Size after replicate-padding to a multiple of 8."""
    return height + (-height) % _PADDING_MULTIPLE, width + (-width) % _PADDING_MULTIPLE


def flow_scale(height: int, width: int, max_pixels: int | None) -> float:
    """Linear scale (<= 1) that brings a frame within ``max_pixels``; 1.0 if it already fits."""
    if max_pixels is None or height * width <= max_pixels:
        return 1.0
    return float(np.sqrt(max_pixels / (height * width)))


def resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize an (N, 2, H, W) flow field to ``size``, rescaling the vectors."""
    height, width = flow.shape[-2:]
    resized = functional.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    resized[:, 0] *= size[1] / width
    resized[:, 1] *= size[0] / height
    return resized

def clamp_scale(scale: float, height: int, width: int) -> float:
    """Raise ``scale`` just enough that the shrunken frame still meets RAFT's minimum size."""
    return min(1.0, max(scale, _MIN_SIDE / min(height, width)))


@torch.no_grad()
def flow_color_map(
    model: nn.Module,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    device: torch.device | str,
    updates: int = RAFT_UPDATES,
    scale: float = 1.0,
) -> np.ndarray:
    """Colour map (RGB uint8, native padded size) of the flow from ``frame_a`` to ``frame_b``.

    ``scale < 1`` runs RAFT on shrunken frames (memory) and resizes the field back.
    """
    height, width = frame_a.shape[:2]
    if scale < 1.0:
        size = (max(round(width * scale), 1), max(round(height * scale), 1))
        frame_a = cv2.resize(frame_a, size, interpolation=cv2.INTER_AREA)
        frame_b = cv2.resize(frame_b, size, interpolation=cv2.INTER_AREA)
    pair = torch.stack([_to_raft_input(frame_a), _to_raft_input(frame_b)]).to(device)
    pair = pad_to_multiple_of_8(pair)
    padded_height, padded_width = pair.shape[-2:]
    if min(padded_height, padded_width) < _MIN_SIDE:
        raise ValueError(
            f"{padded_width}x{padded_height} frames are too small for RAFT (minimum {_MIN_SIDE}px)"
        )
    flow = model(pair[:1], pair[1:], num_flow_updates=updates)[-1]
    if scale < 1.0:
        flow = resize_flow(flow, padded_size(height, width))
    return flow_to_image(flow[0].permute(1, 2, 0).float().cpu().numpy())
