"""Cost-sensitive learning losses.

Based on Sec. 2.2.1 of Gao et al., 2025:
    "Cost-sensitive learning assigns predefined costs to
     misclassified samples ... elevating certain
     misclassification errors above others."

Implements:
- ClassWeightedCE  (inverse-frequency weighting, Ref [92])
- FocalLoss        (instance-level, Algorithm 4, Ref [99])
- ClassBalancedLoss (effective number, Ref [98])
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedCE(nn.Module):
    """Cross-entropy with inverse-frequency class weights.

    w_c = N / (K * n_c)

    Parameters
    ----------
    samples_per_class : list[int]
        Per-class sample counts.
    """

    def __init__(
        self, samples_per_class: list[int],
    ) -> None:
        super().__init__()
        n = sum(samples_per_class)
        k = len(samples_per_class)
        w = torch.tensor(
            [n / (k * c) for c in samples_per_class],
            dtype=torch.float,
        )
        self.register_buffer("weight", w)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits, targets, weight=self.weight,  # type: ignore[arg-type]
        )


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples so training focuses on
    hard, misclassified instances (Algorithm 4, Ref [99]).

    Parameters
    ----------
    gamma : float
        Focusing parameter.  ``gamma = 0`` is plain CE.
    alpha : float or list[float] or None
        Per-class balancing factor.  If a scalar, applied
        uniformly; if a list, per-class weights.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.register_buffer(
                "alpha",
                torch.tensor(alpha, dtype=torch.float),
            )
        elif isinstance(alpha, (int, float)):
            self.register_buffer(
                "alpha",
                torch.tensor([alpha], dtype=torch.float),
            )
        else:
            self.alpha: torch.Tensor | None = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        ce = F.nll_loss(log_p, targets, reduction="none")

        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)
            if alpha_t.numel() == 1:
                focal = alpha_t * focal
            else:
                focal = alpha_t[targets] * focal

        return focal.mean()


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss (Cui et al., 2019).

    Reweights by effective number of samples:
        E_n = (1 - beta^n) / (1 - beta)
        w_c = 1 / E_{n_c}

    Parameters
    ----------
    samples_per_class : list[int]
        Per-class sample counts.
    beta : float
        Effective number parameter (typically 0.9999).
    loss_type : str
        ``"ce"`` for weighted CE, ``"focal"`` for
        focal variant.
    gamma : float
        Focal gamma (used only when ``loss_type="focal"``).
    """

    def __init__(
        self,
        samples_per_class: list[int],
        beta: float = 0.9999,
        loss_type: str = "ce",
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma

        effective = [
            (1.0 - beta ** n) / (1.0 - beta)
            for n in samples_per_class
        ]
        weights = [1.0 / e for e in effective]
        total = sum(weights)
        weights = [
            w / total * len(weights) for w in weights
        ]
        self.register_buffer(
            "weight",
            torch.tensor(weights, dtype=torch.float),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "focal":
            return self._focal(logits, targets)
        return F.cross_entropy(
            logits, targets, weight=self.weight,  # type: ignore[arg-type]
        )

    def _focal(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_p, targets, reduction="none")
        pt = log_p.exp().gather(
            1, targets.unsqueeze(1),
        ).squeeze(1)
        focal = ((1.0 - pt) ** self.gamma) * ce
        w = self.weight.to(logits.device)  # type: ignore[union-attr]
        return (w[targets] * focal).mean()
