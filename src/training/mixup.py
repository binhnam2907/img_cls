"""Mixup, CutMix, and Remix data augmentation.

References:
  - Mixup: Zhang et al., 2018
  - CutMix: Yun et al., 2019
  - Remix: Chou et al., 2020 (Sec. 2.1.1, Ref [13])
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
    lam_label: float | None = None


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


def remix(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    tau: float = 0.5,
    kappa: float = 0.9,
    class_counts: torch.Tensor | None = None,
) -> MixupOutput:
    """Remix: Mixup with separate label lambda biased
    toward minority (Chou et al., 2020).

    The feature lambda comes from Beta(alpha, alpha) as
    usual.  The label lambda is adjusted so the minority
    sample in each pair receives at least ``kappa``
    weight when the feature lambda falls below ``tau``.
    """
    lam_f = (
        np.random.beta(alpha, alpha)
        if alpha > 0 else 1.0
    )
    bs = images.size(0)
    idx = torch.randperm(bs, device=images.device)
    mixed = lam_f * images + (1 - lam_f) * images[idx]

    lam_l = lam_f
    if class_counts is not None and lam_f < tau:
        cc = class_counts.to(targets.device)
        n_a = cc[targets].float()
        n_b = cc[targets[idx]].float()
        minority_is_a = (n_a <= n_b).float().mean()
        if minority_is_a > 0.5:
            lam_l = max(lam_f, kappa)
        else:
            lam_l = min(lam_f, 1.0 - kappa)

    return MixupOutput(
        images=mixed,
        targets_a=targets,
        targets_b=targets[idx],
        lam=lam_f,
        lam_label=lam_l,
    )


def mixup_criterion(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    mix_out: MixupOutput,
) -> torch.Tensor:
    """Compute mixed loss from MixupOutput."""
    lam = (
        mix_out.lam_label
        if mix_out.lam_label is not None
        else mix_out.lam
    )
    return (
        lam * criterion(logits, mix_out.targets_a)
        + (1 - lam) * criterion(
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
