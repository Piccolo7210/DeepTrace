"""ResNet-50 binary classifier used by both AIGVDet branches."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .config import FC_INIT_STD

_FEATURE_DIM = 2048


def build_branch(pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained ResNet-50 with a single-logit head (``fc: 2048 -> 1``).

    The state-dict keys match the authors' released checkpoints, so those
    load directly with :func:`load_branch`.
    """
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(_FEATURE_DIM, 1)
    nn.init.normal_(model.fc.weight, mean=0.0, std=FC_INIT_STD)
    return model


def _unwrap_state_dict(payload: dict) -> dict:
    """Accept a plain state-dict or the authors' ``{"model": state_dict, ...}`` format."""
    state = payload.get("model", payload)
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_branch(path: Path, device: torch.device | str = "cpu") -> nn.Module:
    """Build a branch (without downloading ImageNet weights) and load a checkpoint into it."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_branch(pretrained=False)
    model.load_state_dict(_unwrap_state_dict(payload))
    return model.to(device).eval()
