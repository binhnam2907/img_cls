"""Loss functions for imbalanced classification.

Independent strategies:
  1. FocalLoss          - down-weights easy examples
  2. ClassBalancedLoss  - effective-number reweighting
  3. LabelSmoothingCE   - soft targets for calibration
  4. CBFocalLoss        - CB + Focal combined
  5. BalancedSoftmax    - log-prior logit adjustment
  6. LogitAdjustment    - parameterised logit shift
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights well-classified examples so the model
    focuses on hard, misclassified samples.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            "_alpha",
            alpha if alpha is not None
            else torch.tensor(1.0),
        )

    def forward(
        self, logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets_one_hot = F.one_hot(
            targets, num_classes=logits.size(1),
        ).float()

        alpha = self._alpha.to(logits.device)
        if alpha.dim() == 0:
            alpha_t = alpha
        else:
            alpha_t = alpha[targets].unsqueeze(1)

        p_t = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1.0 - p_t) ** self.gamma

        ce = -(
            log_probs * targets_one_hot
        ).sum(dim=1)

        loss = alpha_t.squeeze() * focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    """Class-Balanced Loss (Cui et al., 2019).

    Reweights by effective number of samples:
      E_n = (1 - beta^n) / (1 - beta)
    where n is per-class count.
    """

    def __init__(
        self,
        samples_per_class: list[int],
        beta: float = 0.9999,
        loss_type: str = "ce",
        gamma: float = 2.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.loss_type = loss_type
        self.n_classes = len(samples_per_class)

        effective_num = [
            1.0 - beta ** n for n in samples_per_class
        ]
        weights = [
            (1.0 - beta) / max(e, 1e-8)
            for e in effective_num
        ]
        total = sum(weights)
        weights = [
            w / total * self.n_classes for w in weights
        ]
        self.register_buffer(
            "_weights", torch.tensor(weights),
        )

    def forward(
        self, logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        w = self._weights.to(logits.device)

        if self.loss_type == "focal":
            return _focal_forward(
                logits, targets, w, self.gamma,
            )

        return F.cross_entropy(
            logits, targets, weight=w,
        )


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing.

    Replaces hard 0/1 targets with soft targets:
      y_smooth = (1 - eps) * y_hard + eps / K

    Improves calibration and reduces overconfidence,
    which helps when classes are imbalanced.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer(
            "_weight",
            weight if weight is not None
            else torch.tensor(0.0),
        )

    def forward(
        self, logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        w = self._weight.to(logits.device)
        use_weight = w.dim() > 0
        return F.cross_entropy(
            logits, targets,
            weight=w if use_weight else None,
            label_smoothing=self.smoothing,
        )


class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss (industry best).

    Combines effective-number reweighting (CB) with
    focal modulation for the strongest correction on
    heavily imbalanced data.
    """

    def __init__(
        self,
        samples_per_class: list[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.n_classes = len(samples_per_class)

        effective_num = [
            1.0 - beta ** n for n in samples_per_class
        ]
        weights = [
            (1.0 - beta) / max(e, 1e-8)
            for e in effective_num
        ]
        total = sum(weights)
        weights = [
            w / total * self.n_classes for w in weights
        ]
        self.register_buffer(
            "_weights", torch.tensor(weights),
        )

    def forward(
        self, logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return _focal_forward(
            logits, targets,
            self._weights.to(logits.device),
            self.gamma,
        )


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax (Ren et al., 2020).

    Adjusts logits by adding log(class_prior) before
    softmax so the decision boundary accounts for the
    training label distribution.

    Ref: Gao et al. 2025, Section 5.2.1.
    """

    def __init__(
        self,
        samples_per_class: list[int],
    ):
        super().__init__()
        freq = torch.tensor(
            samples_per_class, dtype=torch.float,
        )
        self.register_buffer(
            "_log_prior",
            freq.log() - freq.sum().log(),
        )

    def forward(
        self, logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        adjusted = logits + self._log_prior.to(
            logits.device,
        )
        return F.cross_entropy(adjusted, targets)


class LogitAdjustmentLoss(nn.Module):
    """Logit Adjustment (Menon et al., 2021).

    Shifts logits by tau * log(class_prior) to
    encourage balanced posterior predictions.

    Ref: Gao et al. 2025, Section 5.2.2.
    """

    def __init__(
        self,
        samples_per_class: list[int],
        tau: float = 1.0,
    ):
        super().__init__()
        self.tau = tau
        freq = torch.tensor(
            samples_per_class, dtype=torch.float,
        )
        self.register_buffer(
            "_log_prior",
            freq.log() - freq.sum().log(),
        )

    def forward(
        self, logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        adjusted = logits + self.tau * self._log_prior.to(
            logits.device,
        )
        return F.cross_entropy(adjusted, targets)


def _focal_forward(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Shared focal-loss forward pass."""
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    targets_oh = F.one_hot(
        targets, num_classes=logits.size(1),
    ).float()

    p_t = (probs * targets_oh).sum(dim=1)
    focal_weight = (1.0 - p_t) ** gamma

    ce = -(log_probs * targets_oh).sum(dim=1)
    alpha_t = alpha[targets]

    return (alpha_t * focal_weight * ce).mean()


def build_loss(
    cfg: dict, device: torch.device,
    samples_per_class: list[int] | None = None,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Factory: build loss from config."""
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "ce")

    match name:
        case "ce":
            if class_weights is not None:
                return nn.CrossEntropyLoss(
                    weight=class_weights.to(device),
                )
            return nn.CrossEntropyLoss()

        case "focal":
            alpha = (
                class_weights
                if class_weights is not None
                else None
            )
            return FocalLoss(
                alpha=alpha,
                gamma=loss_cfg.get("gamma", 2.0),
            ).to(device)

        case "cb":
            if samples_per_class is None:
                raise ValueError(
                    "CB loss requires samples_per_class"
                )
            return ClassBalancedLoss(
                samples_per_class,
                beta=loss_cfg.get("beta", 0.9999),
                loss_type="ce",
            ).to(device)

        case "label_smoothing":
            return LabelSmoothingCE(
                smoothing=loss_cfg.get(
                    "smoothing", 0.1,
                ),
                weight=class_weights,
            ).to(device)

        case "cb_focal":
            if samples_per_class is None:
                raise ValueError(
                    "CB Focal loss needs "
                    "samples_per_class"
                )
            return CBFocalLoss(
                samples_per_class,
                beta=loss_cfg.get("beta", 0.9999),
                gamma=loss_cfg.get("gamma", 2.0),
            ).to(device)

        case "balanced_softmax":
            if samples_per_class is None:
                raise ValueError(
                    "BalancedSoftmax needs "
                    "samples_per_class"
                )
            return BalancedSoftmaxLoss(
                samples_per_class,
            ).to(device)

        case "logit_adjust":
            if samples_per_class is None:
                raise ValueError(
                    "LogitAdjustment needs "
                    "samples_per_class"
                )
            return LogitAdjustmentLoss(
                samples_per_class,
                tau=loss_cfg.get("tau", 1.0),
            ).to(device)

        case _:
            raise ValueError(
                f"Unknown loss '{name}'. Choose from: "
                "ce, focal, cb, label_smoothing, "
                "cb_focal, balanced_softmax, "
                "logit_adjust"
            )
