"""Optical-flow colour coding used by the authors' ``demo.py`` (RAFT ``flow_viz.py``).

Adapted from RAFT's ``core/utils/flow_viz.py`` (Tom Runia, MIT licence), itself
based on the Baker et al. colour wheel. The flow field is normalised by the
largest flow magnitude of the frame, so the map encodes direction and *relative*
magnitude only.
"""

from __future__ import annotations

import numpy as np

_RY, _YG, _GC, _CB, _BM, _MR = 15, 6, 4, 11, 13, 6


def make_colorwheel() -> np.ndarray:
    """Return the (55, 3) colour wheel of Baker et al. with values in 0..255."""
    ncols = _RY + _YG + _GC + _CB + _BM + _MR
    wheel = np.zeros((ncols, 3))
    col = 0
    wheel[0:_RY, 0] = 255
    wheel[0:_RY, 1] = np.floor(255 * np.arange(0, _RY) / _RY)
    col += _RY
    wheel[col : col + _YG, 0] = 255 - np.floor(255 * np.arange(0, _YG) / _YG)
    wheel[col : col + _YG, 1] = 255
    col += _YG
    wheel[col : col + _GC, 1] = 255
    wheel[col : col + _GC, 2] = np.floor(255 * np.arange(0, _GC) / _GC)
    col += _GC
    wheel[col : col + _CB, 1] = 255 - np.floor(255 * np.arange(_CB) / _CB)
    wheel[col : col + _CB, 2] = 255
    col += _CB
    wheel[col : col + _BM, 2] = 255
    wheel[col : col + _BM, 0] = np.floor(255 * np.arange(0, _BM) / _BM)
    col += _BM
    wheel[col : col + _MR, 2] = 255 - np.floor(255 * np.arange(_MR) / _MR)
    wheel[col : col + _MR, 0] = 255
    return wheel


def flow_uv_to_colors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Colour-code a normalised flow field (radius <= 1 inside the wheel) as RGB uint8."""
    image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    wheel = make_colorwheel()
    ncols = wheel.shape[0]
    radius = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(-v, -u) / np.pi
    position = (angle + 1) / 2 * (ncols - 1)
    k0 = np.floor(position).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    fraction = position - k0
    for channel in range(wheel.shape[1]):
        column = wheel[:, channel]
        colour = (1 - fraction) * (column[k0] / 255.0) + fraction * (column[k1] / 255.0)
        inside = radius <= 1
        colour[inside] = 1 - radius[inside] * (1 - colour[inside])
        colour[~inside] = colour[~inside] * 0.75  # out of range
        image[:, :, channel] = np.floor(255 * colour)
    return image


def flow_to_image(flow_uv: np.ndarray) -> np.ndarray:
    """Colour-code an (H, W, 2) flow field as an (H, W, 3) RGB uint8 image."""
    if flow_uv.ndim != 3 or flow_uv.shape[2] != 2:
        raise ValueError(f"Expected an (H, W, 2) flow field, got {flow_uv.shape}")
    u, v = flow_uv[:, :, 0], flow_uv[:, :, 1]
    radius_max = np.max(np.sqrt(np.square(u) + np.square(v)))
    epsilon = 1e-5
    return flow_uv_to_colors(u / (radius_max + epsilon), v / (radius_max + epsilon))
