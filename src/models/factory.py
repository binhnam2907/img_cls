from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.resnet import RESNET_VARIANTS, build_resnet
from src.models.simple_cnn import SimpleCNN

SUPPORTED_MODELS = (
    list(RESNET_VARIANTS) + ["simple_cnn"]
)


def build_model(cfg: dict[str, Any]) -> nn.Module:
    name = cfg["model"]["name"]
    num_classes = cfg["model"].get("num_classes", 10)
    dropout = cfg["model"].get("dropout", 0.0)

    if name in RESNET_VARIANTS:
        return build_resnet(
            name,
            num_classes=num_classes,
            pretrained=cfg["model"].get(
                "pretrained", False,
            ),
            dropout=dropout,
        )

    if name == "simple_cnn":
        return SimpleCNN(
            num_classes=num_classes, dropout=dropout,
        )

    raise ValueError(
        f"Unknown model '{name}'. "
        f"Choose from: {SUPPORTED_MODELS}"
    )
