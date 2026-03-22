"""Supervised Contrastive Learning for imbalanced sets.

Based on Sec. 2.2.3 of Gao et al., 2025:
    "Supervised contrastive learning methods incorporate
     label information into the contrastive learning process
     and extend positive examples by including other samples
     from the same class along with data augmentations."

Implements SupCon (Algorithm 6, Ref [121]) with balanced
positive set construction (KCL, Ref [125]) so that each
anchor sees an equal number of positives regardless of
class frequency.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Optionally caps the positive set to ``max_positives``
    per anchor for class-balanced contrastive learning
    (KCL, Sec. 2.2.3, Ref [125]).

    Parameters
    ----------
    temperature : float
        Scaling temperature for cosine similarity.
    max_positives : int or None
        If set, randomly sample at most this many
        positives per anchor (class-balanced mode).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        max_positives: int | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.max_positives = max_positives

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, D)
            L2-normalised embedding vectors.
        labels : (B,)
            Integer class labels.
        """
        device = features.device
        b = features.size(0)

        sim = (features @ features.t()) / self.temperature
        self_mask = ~torch.eye(
            b, dtype=torch.bool, device=device,
        )

        lbl_eq = (
            labels.unsqueeze(0) == labels.unsqueeze(1)
        )
        pos_mask = lbl_eq & self_mask

        if self.max_positives is not None:
            pos_mask = _cap_positives(
                pos_mask, self.max_positives,
            )

        n_pos = pos_mask.sum(dim=1).clamp(min=1)

        logits_max = sim.detach().max(
            dim=1, keepdim=True,
        ).values
        logits = sim - logits_max

        exp_logits = torch.exp(logits) * self_mask.float()
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp(
                min=1e-8,
            ),
        )

        mean_log_prob = (
            (pos_mask.float() * log_prob).sum(dim=1)
            / n_pos
        )
        return -mean_log_prob.mean()


def _cap_positives(
    mask: torch.Tensor, k: int,
) -> torch.Tensor:
    """Keep at most *k* positives per row (anchor)."""
    b = mask.size(0)
    out = mask.clone()
    for i in range(b):
        idx = mask[i].nonzero(as_tuple=False).squeeze(-1)
        if len(idx) > k:
            keep = idx[torch.randperm(len(idx))[:k]]
            out[i] = False
            out[i, keep] = True
    return out
