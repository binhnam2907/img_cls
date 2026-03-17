"""Mixup and CutMix data augmentation for imbalance.

Both create virtual training samples by interpolating
between pairs, which acts as a strong regularizer and
helps the model generalize across imbalanced classes.

References:
  - Mixup: Zhang et al., 2018
  - CutMix: Yun et al., 2019
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MixupOutput:
    images: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam: float


def mixup(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.4,
) -> MixupOutput:
    """Mixup: linear interpolation of two samples.

    x_mix = lam * x_i + (1 - lam) * x_j
    Loss  = lam * L(pred, y_i) + (1-lam) * L(pred, y_j)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed = lam * images + (1 - lam) * images[index]
    return MixupOutput(
        images=mixed,
        targets_a=targets,
        targets_b=targets[index],
        lam=lam,
    )


def cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
) -> MixupOutput:
    """CutMix: replace a rectangular patch with
    another sample's patch.

    The loss is weighted by the area ratio (lam).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    _, _, h, w = images.shape
    bbx1, bby1, bbx2, bby2 = _rand_bbox(
        h, w, lam,
    )

    mixed = images.clone()
    mixed[:, :, bbx1:bbx2, bby1:bby2] = (
        images[index, :, bbx1:bbx2, bby1:bby2]
    )

    area_ratio = (
        (bbx2 - bbx1) * (bby2 - bby1)
    ) / (h * w)
    lam = 1.0 - area_ratio

    return MixupOutput(
        images=mixed,
        targets_a=targets,
        targets_b=targets[index],
        lam=lam,
    )


def mixup_criterion(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    mix_out: MixupOutput,
) -> torch.Tensor:
    """Compute mixed loss from MixupOutput."""
    return (
        mix_out.lam * criterion(
            logits, mix_out.targets_a,
        )
        + (1 - mix_out.lam) * criterion(
            logits, mix_out.targets_b,
        )
    )


def _rand_bbox(
    h: int, w: int, lam: float,
) -> tuple[int, int, int, int]:
    """Random bounding box for CutMix."""
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cx = np.random.randint(h)
    cy = np.random.randint(w)

    x1 = max(0, cx - cut_h // 2)
    y1 = max(0, cy - cut_w // 2)
    x2 = min(h, cx + cut_h // 2)
    y2 = min(w, cy + cut_w // 2)

    return x1, y1, x2, y2
