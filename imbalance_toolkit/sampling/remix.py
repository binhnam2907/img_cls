"""Modern interpolation: Remix and Balanced-Mixup.

Based on Sec. 2.1.1 of Gao et al., 2025:

Remix (Ref [13]):
    "Remix offers a flexible solution by relaxing the
     constraint of employing the same mixing factor for
     both features and labels ... bestowing greater
     importance upon minority samples."

Balanced-Mixup (Ref [15]):
    "Balanced-MixUp employs a dual-strategy approach by
     concurrently sampling imbalanced instances and
     balanced classes from the training data."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MixOutput:
    """Carrier for mixed images and label targets."""

    images: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam_feature: float
    lam_label: float


class Remix:
    """Rebalanced Mixup (Chou et al., 2020).

    Unlike vanilla Mixup which uses the same lambda for
    features and labels, Remix decouples them: the label
    lambda is biased toward the minority class, shifting
    the decision boundary toward the majority.

    Parameters
    ----------
    alpha : float
        Beta distribution parameter for feature lambda.
    tau : float
        Threshold below which ``lam_label`` is clamped
        to ``kappa`` for the minority sample.
    kappa : float
        Minimum label weight assigned to the minority
        sample when ``lam_feature < tau``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        tau: float = 0.5,
        kappa: float = 0.9,
    ) -> None:
        self.alpha = alpha
        self.tau = tau
        self.kappa = kappa

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        class_counts: torch.Tensor,
    ) -> MixOutput:
        """Mix a batch.

        Parameters
        ----------
        images : (B, C, H, W)
        targets : (B,) int labels
        class_counts : (num_classes,) per-class sample
            counts used to decide which sample in each
            pair is the minority.
        """
        lam_f = float(
            np.random.beta(self.alpha, self.alpha),
        )
        idx = torch.randperm(images.size(0))

        mixed = lam_f * images + (1 - lam_f) * images[idx]

        n_a = class_counts[targets].float()
        n_b = class_counts[targets[idx]].float()
        minority_is_a = n_a <= n_b

        lam_l = torch.where(
            minority_is_a,
            torch.clamp(
                torch.tensor(lam_f),
                min=self.kappa,
            ),
            torch.clamp(
                torch.tensor(lam_f),
                max=1.0 - self.kappa,
            ),
        ).item() if lam_f < self.tau else lam_f

        return MixOutput(
            images=mixed,
            targets_a=targets,
            targets_b=targets[idx],
            lam_feature=lam_f,
            lam_label=float(lam_l),
        )


class BalancedMixup:
    """Balanced-MixUp (Galdran et al., 2021).

    Two-sampler approach: one draws from the original
    imbalanced distribution, the other draws class-
    balanced samples.  Mixup is applied between the pair.

    Parameters
    ----------
    alpha : float
        Beta distribution parameter for lambda.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = alpha

    def __call__(
        self,
        images_imb: torch.Tensor,
        targets_imb: torch.Tensor,
        images_bal: torch.Tensor,
        targets_bal: torch.Tensor,
    ) -> MixOutput:
        """Mix imbalanced batch with a balanced batch.

        The caller is responsible for providing the
        balanced batch (e.g. via a class-balanced sampler).
        """
        lam = float(
            np.random.beta(self.alpha, self.alpha),
        )
        mixed = lam * images_imb + (1 - lam) * images_bal
        return MixOutput(
            images=mixed,
            targets_a=targets_imb,
            targets_b=targets_bal,
            lam_feature=lam,
            lam_label=lam,
        )


def remix_criterion(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    mix: MixOutput,
) -> torch.Tensor:
    """Compute mixed loss using ``lam_label``."""
    return (
        mix.lam_label * criterion(logits, mix.targets_a)
        + (1.0 - mix.lam_label)
        * criterion(logits, mix.targets_b)
    )
