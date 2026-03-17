from __future__ import annotations

from typing import Any

import torch.nn as nn
import torch.optim as optim


def _build_sgd(
    params, lr: float, wd: float, opt_cfg: dict,
) -> optim.SGD:
    return optim.SGD(
        params, lr=lr, weight_decay=wd,
        momentum=opt_cfg.get("momentum", 0.9),
        nesterov=opt_cfg.get("nesterov", False),
    )


_REGISTRY: dict[str, type[optim.Optimizer]] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
}

SUPPORTED_OPTIMIZERS = ["sgd"] + list(_REGISTRY)


def build_optimizer(
    model: nn.Module, cfg: dict[str, Any],
) -> optim.Optimizer:
    opt_cfg = cfg["training"]["optimizer"]
    name = opt_cfg["name"].lower()
    lr = opt_cfg.get("lr", 1e-3)
    wd = opt_cfg.get("weight_decay", 0.0)

    if name == "sgd":
        return _build_sgd(
            model.parameters(), lr, wd, opt_cfg,
        )

    if name in _REGISTRY:
        return _REGISTRY[name](
            model.parameters(),
            lr=lr, weight_decay=wd,
        )

    raise ValueError(
        f"Unknown optimizer '{name}'. "
        f"Choose from: {SUPPORTED_OPTIMIZERS}"
    )
