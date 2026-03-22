"""Balanced Softmax and Logit Adjustment.

Based on Sec. 2.3.4 (Posterior Re-calibration) of
Gao et al., 2025:
    "Applying an additive adjustment to the logit output
     during inference:
         f_y(x) <- f_y(x) + log p_t(y) - log p_s(y)
     which forms the main idea of balanced softmax."

References
----------
- Ren et al., 2020 — Balanced Softmax (Ref [101])
- Menon et al., 2021 — Logit Adjustment (Ref [177])
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax Cross-Entropy.

    Adjusts the logits by the *log* of the training label
    prior before computing softmax, so that rare classes
    receive an implicit boost during training.

    Parameters
    ----------
    samples_per_class : list[int]
        Number of training samples per class.
    """

    def __init__(
        self, samples_per_class: list[int],
    ) -> None:
        super().__init__()
        freq = torch.tensor(
            samples_per_class, dtype=torch.float,
        )
        self.register_buffer(
            "log_prior",
            torch.log(freq / freq.sum()),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C) raw model output.
        targets : (B,)   integer labels.
        """
        adjusted = logits + self.log_prior  # type: ignore[operator]
        return F.cross_entropy(adjusted, targets)


class LogitAdjustmentLoss(nn.Module):
    """Post-hoc Logit Adjustment (Menon et al., 2021).

    Adds ``tau * log(prior)`` to logits.  When ``tau = 1``
    this recovers Balanced Softmax; tuning ``tau`` trades
    off between standard CE and full adjustment.

    Parameters
    ----------
    samples_per_class : list[int]
        Number of training samples per class.
    tau : float
        Adjustment strength.  ``tau = 0`` is plain CE.
    """

    def __init__(
        self,
        samples_per_class: list[int],
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.tau = tau
        freq = torch.tensor(
            samples_per_class, dtype=torch.float,
        )
        self.register_buffer(
            "log_prior",
            torch.log(freq / freq.sum()),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_prior  # type: ignore[operator]
        return F.cross_entropy(adjusted, targets)
