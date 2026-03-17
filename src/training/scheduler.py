from __future__ import annotations

from typing import Any

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)

SUPPORTED_SCHEDULERS = [
    "step", "cosine", "plateau", "none",
]


def _build_step(
    opt: optim.Optimizer, cfg: dict,
) -> StepLR:
    return StepLR(
        opt,
        step_size=cfg.get("step_size", 30),
        gamma=cfg.get("gamma", 0.1),
    )


def _build_cosine(
    opt: optim.Optimizer, cfg: dict, epochs: int,
) -> CosineAnnealingLR:
    return CosineAnnealingLR(
        opt,
        T_max=cfg.get("T_max", epochs),
        eta_min=cfg.get("eta_min", 1e-6),
    )


def _build_plateau(
    opt: optim.Optimizer, cfg: dict,
) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        opt, mode="max",
        patience=cfg.get("patience", 10),
        factor=cfg.get("factor", 0.1),
    )


_BUILDERS = {
    "step": lambda o, c, _e: _build_step(o, c),
    "cosine": lambda o, c, e: _build_cosine(o, c, e),
    "plateau": lambda o, c, _e: _build_plateau(o, c),
}


def build_scheduler(
    optimizer: optim.Optimizer, cfg: dict[str, Any],
) -> LRScheduler | None:
    sch_cfg = cfg["training"]["scheduler"]
    name = sch_cfg.get("name", "none").lower()
    epochs = cfg["training"]["epochs"]

    if name == "none":
        return None

    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            f"Choose from: {SUPPORTED_SCHEDULERS}"
        )

    return builder(optimizer, sch_cfg, epochs)
