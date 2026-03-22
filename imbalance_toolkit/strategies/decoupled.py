"""Decoupled Training (cRT).

Based on Sec. 2.3.1 of Gao et al., 2025:
    "cRT separates the model learning procedure into two
     distinct processes: representation learning and
     classification.  They demonstrated that the model
     could achieve robust classification performance by
     only retraining the classifier using class-balanced
     sampling."

Reference
---------
- Kang et al., 2020 (Ref [158])
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler


class DecoupledTrainer:
    """Two-stage cRT trainer.

    Stage 1 — *Representation learning*:
        Train the full model end-to-end on the original
        (imbalanced) data with instance-balanced sampling.

    Stage 2 — *Classifier re-training*:
        Freeze the feature extractor and re-train only the
        final linear layer (classifier head) using a class-
        balanced sampler.

    Parameters
    ----------
    model : nn.Module
        Must expose ``model.head`` (the linear classifier)
        and ``model.parameters()`` (full params).
    head_attr : str
        Name of the classifier submodule (default ``"head"``).
    stage1_epochs : int
        Epochs for representation learning.
    stage2_epochs : int
        Epochs for classifier re-training.
    stage1_lr : float
        Learning rate for stage 1.
    stage2_lr : float
        Learning rate for stage 2.
    device : torch.device or None
        Training device.
    """

    def __init__(
        self,
        model: nn.Module,
        head_attr: str = "head",
        stage1_epochs: int = 20,
        stage2_epochs: int = 10,
        stage1_lr: float = 1e-3,
        stage2_lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.head_attr = head_attr
        self.s1_epochs = stage1_epochs
        self.s2_epochs = stage2_epochs
        self.s1_lr = stage1_lr
        self.s2_lr = stage2_lr
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def fit(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        targets: list[int],
    ) -> dict[str, list[float]]:
        """Run both stages.

        Parameters
        ----------
        train_loader : DataLoader
            Instance-balanced loader (used in stage 1).
        criterion : nn.Module
            Loss function (e.g. CrossEntropyLoss).
        targets : list[int]
            Full list of training labels (for building
            a balanced sampler in stage 2).

        Returns
        -------
        dict with ``"stage1_loss"`` and ``"stage2_loss"``
        histories.
        """
        s1 = self._stage1(train_loader, criterion)
        bal_loader = self._balanced_loader(
            train_loader, targets,
        )
        s2 = self._stage2(bal_loader, criterion)
        return {"stage1_loss": s1, "stage2_loss": s2}

    # ── stage implementations ───────────────────

    def _stage1(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> list[float]:
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.s1_lr,
        )
        return self._train_loop(
            loader, criterion, opt, self.s1_epochs,
        )

    def _stage2(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> list[float]:
        _freeze_all_except(
            self.model, self.head_attr,
        )
        head = getattr(self.model, self.head_attr)
        opt = torch.optim.Adam(
            head.parameters(), lr=self.s2_lr,
        )
        losses = self._train_loop(
            loader, criterion, opt, self.s2_epochs,
        )
        _unfreeze(self.model)
        return losses

    def _train_loop(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Any,
        n_epochs: int,
    ) -> list[float]:
        history: list[float] = []
        self.model.train()
        for _ in range(n_epochs):
            total = 0.0
            n = 0
            for imgs, lbls in loader:
                imgs = imgs.to(self.device)
                lbls = lbls.to(self.device)
                logits = self.model(imgs)
                loss = criterion(logits, lbls)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(lbls)
                n += len(lbls)
            history.append(total / max(n, 1))
        return history

    @staticmethod
    def _balanced_loader(
        reference: DataLoader,
        targets: list[int],
    ) -> DataLoader:
        counts = Counter(targets)
        w = [1.0 / counts[t] for t in targets]
        sampler = WeightedRandomSampler(
            w, num_samples=len(w), replacement=True,
        )
        return DataLoader(
            reference.dataset,
            batch_size=reference.batch_size or 64,
            sampler=sampler,
            num_workers=reference.num_workers,
            pin_memory=reference.pin_memory,
        )


def _freeze_all_except(
    model: nn.Module, keep: str,
) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(keep)


def _unfreeze(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
