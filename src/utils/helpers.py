from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(
    preference: str = "auto",
) -> torch.device:
    if preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(
    model: torch.nn.Module,
    trainable_only: bool = True,
) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def save_checkpoint(
    state: dict[str, Any],
    directory: str | Path,
    filename: str = "checkpoint.pth",
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    torch.save(state, filepath)
    return filepath


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    return torch.load(
        path, map_location=device, weights_only=False,
    )
