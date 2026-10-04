"""Image preprocessing shared by training, evaluation and the app.

All images are ``uint8`` RGB arrays of shape (H, W, 3) until they are turned
into a tensor; normalisation happens on the compute device.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .config import CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD


def _pad_to_minimum(image: np.ndarray, size: int) -> np.ndarray:
    """Zero-pad symmetrically so both sides are at least ``size`` (torchvision rule)."""
    height, width = image.shape[:2]
    pad_h, pad_w = max(size - height, 0), max(size - width, 0)
    if pad_h == 0 and pad_w == 0:
        return image
    padding = ((pad_h // 2, (pad_h + 1) // 2), (pad_w // 2, (pad_w + 1) // 2), (0, 0))
    return np.pad(image, padding, mode="constant", constant_values=0)


def center_crop_or_pad(image: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Centre-crop to ``size`` x ``size``, zero-padding frames that are smaller."""
    image = _pad_to_minimum(image, size)
    height, width = image.shape[:2]
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return image[top : top + size, left : left + size]


def random_crop(image: np.ndarray, rng: np.random.Generator, size: int = CROP_SIZE) -> np.ndarray:
    """Uniformly random ``size`` x ``size`` crop (zero-padding small frames)."""
    image = _pad_to_minimum(image, size)
    height, width = image.shape[:2]
    top = int(rng.integers(0, height - size + 1))
    left = int(rng.integers(0, width - size + 1))
    return image[top : top + size, left : left + size]


def jpeg_round_trip(image: np.ndarray, quality: int) -> np.ndarray:
    """Encode/decode an RGB image as JPEG with OpenCV (the plan's single JPEG implementation)."""
    ok, buffer = cv2.imencode(".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(decoded[:, :, ::-1])


def to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    """HWC ``uint8`` array -> contiguous CHW ``uint8`` tensor."""
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def normalise_on_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a ``uint8`` NCHW batch to ``device`` and apply ImageNet normalisation."""
    batch = batch.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (batch - mean) / std
