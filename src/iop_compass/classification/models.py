"""View-classification model factory."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .dataset import VIEW_CLASSES


def build_resnet18(
    num_classes: int = len(VIEW_CLASSES), pretrained: bool = True
) -> nn.Module:
    """ResNet18 with a fresh ``num_classes``-way head.

    ImageNet weights are read from the repository-local ``TORCH_HOME`` cache so
    compute nodes never need network access.
    """
    weights = None
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, path: str | Path, strict: bool = True) -> nn.Module:
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=strict)
    return model


def torch_home() -> str:
    return os.environ.get("TORCH_HOME", "")
