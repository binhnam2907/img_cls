"""Metric Learning losses for imbalanced classification.

Based on Sec. 2.2.2 of Gao et al., 2025:
    "Metric learning aims to learn a discriminative feature
     space by designing task-specific distance metrics
     between samples, which makes samples of the same class
     closer together, while samples of different classes are
     further apart."

Implements:
- Triplet Loss (Algorithm 5, Ref [122])
- Contrastive Loss (pair-based)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet margin loss with semi-hard mining.

    For each anchor, selects the hardest positive (largest
    distance) and hardest negative (smallest distance) within
    the mini-batch.  Particularly effective for imbalanced
    sets because it forces the model to learn discriminative
    embeddings even for rare classes (Sec. 2.2.2).

    Parameters
    ----------
    margin : float
        Minimum desired gap between positive and negative
        distances.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Batch-all triplet loss.

        Parameters
        ----------
        embeddings : (B, D)  L2-normalised feature vectors.
        labels     : (B,)    integer class labels.
        """
        dist = _pairwise_l2(embeddings)
        b = embeddings.size(0)

        lbl_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = lbl_eq & ~torch.eye(
            b, dtype=torch.bool, device=labels.device,
        )
        neg_mask = ~lbl_eq

        ap = dist.unsqueeze(2)
        an = dist.unsqueeze(1)

        triplet = ap - an + self.margin

        mask = (
            pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        )
        triplet = triplet * mask.float()
        triplet = F.relu(triplet)

        n_valid = mask.sum().clamp(min=1)
        return triplet.sum() / n_valid


class ContrastiveLoss(nn.Module):
    """Pair-wise contrastive loss.

    Pulls same-class pairs together and pushes different-
    class pairs apart beyond a margin (Sec. 2.2.2).

    Parameters
    ----------
    margin : float
        Minimum distance for negative pairs.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : (B, D) feature vectors.
        labels     : (B,)   integer class labels.
        """
        dist = _pairwise_l2(embeddings)
        b = embeddings.size(0)

        same = (
            labels.unsqueeze(0) == labels.unsqueeze(1)
        ).float()
        diag = torch.eye(
            b, device=embeddings.device,
        )
        pair_mask = 1.0 - diag

        pos_loss = same * dist.pow(2)
        neg_loss = (1.0 - same) * F.relu(
            self.margin - dist,
        ).pow(2)

        loss = (pos_loss + neg_loss) * pair_mask
        n = pair_mask.sum().clamp(min=1)
        return loss.sum() / n


def _pairwise_l2(x: torch.Tensor) -> torch.Tensor:
    """(B, D) -> (B, B) pairwise Euclidean distances."""
    sq = (x * x).sum(dim=1, keepdim=True)
    dist_sq = sq + sq.t() - 2.0 * x @ x.t()
    return dist_sq.clamp(min=0).sqrt()
