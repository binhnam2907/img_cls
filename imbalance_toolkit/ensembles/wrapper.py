"""Ensemble wrappers for imbalanced data.

Based on Sec. 2.4 of Gao et al., 2025:

Bagging (Sec. 2.4.1):
    "Bagging involves generating multiple balanced subsets
     by resampling techniques like random undersampling and
     SMOTE, and training base classifiers on these subsets
     independently."

Boosting (Sec. 2.4.2):
    "Boosting-based ensemble methods aim to enhance
     performance through iterative re-weighting of training
     samples, to focus on the minority class."

References
----------
- Over-Bagging / SMOTE-Bagging (Ref [182])
- SMOTE-Boost (Ref [188])
- RUSBoost (Ref [190])
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    Subset,
    WeightedRandomSampler,
)


class ImbalancedBagging:
    """Bagging with per-bag class re-balancing.

    Each bag is a class-balanced subset of the original
    training set (Under-Bagging variant).  Base models
    are trained independently; predictions are aggregated
    by majority vote.

    Parameters
    ----------
    base_model_fn : callable
        Factory ``() -> nn.Module`` that produces a fresh
        base model.
    n_estimators : int
        Number of bags / base models.
    bag_ratio : float
        Fraction of samples per bag relative to minority
        class size.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        base_model_fn: Any,
        n_estimators: int = 5,
        bag_ratio: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.model_fn = base_model_fn
        self.n_est = n_estimators
        self.bag_ratio = bag_ratio
        self.rng = np.random.RandomState(seed)
        self.models: list[nn.Module] = []

    def fit(
        self,
        dataset: Any,
        targets: list[int],
        train_fn: Any,
    ) -> None:
        """Train all base models.

        Parameters
        ----------
        dataset : Dataset
            Full training dataset.
        targets : list[int]
            Labels corresponding to dataset indices.
        train_fn : callable
            ``(model, loader) -> None`` that runs one
            training cycle on a model with a DataLoader.
        """
        counts = Counter(targets)
        min_n = int(
            min(counts.values()) * self.bag_ratio,
        )
        cls_indices = _group_by_class(targets)

        for _ in range(self.n_est):
            bag_idx = _sample_balanced_bag(
                cls_indices, min_n, self.rng,
            )
            subset = Subset(dataset, bag_idx)
            loader = DataLoader(
                subset, batch_size=64, shuffle=True,
            )
            model = self.model_fn()
            train_fn(model, loader)
            self.models.append(model)

    @torch.no_grad()
    def predict(
        self, inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Majority-vote prediction across all base models.

        Parameters
        ----------
        inputs : (B, ...) input batch.

        Returns
        -------
        (B,) integer predictions.
        """
        votes: list[torch.Tensor] = []
        for m in self.models:
            m.eval()
            out = m(inputs)
            votes.append(out.argmax(dim=1))
        stacked = torch.stack(votes, dim=0)
        return torch.mode(stacked, dim=0).values


class ImbalancedBoosting:
    """Adaptive boosting with sample re-weighting.

    At each round, trains a base model on the weighted
    dataset, then increases weights for misclassified
    minority samples (AdaCost-style, Sec. 2.4.3).

    Parameters
    ----------
    base_model_fn : callable
        Factory ``() -> nn.Module``.
    n_rounds : int
        Number of boosting rounds.
    cost_factor : float
        Multiplicative cost applied to minority-class
        misclassifications when updating weights.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        base_model_fn: Any,
        n_rounds: int = 5,
        cost_factor: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.model_fn = base_model_fn
        self.n_rounds = n_rounds
        self.cost_factor = cost_factor
        self.rng = np.random.RandomState(seed)
        self.models: list[nn.Module] = []
        self.alphas: list[float] = []

    def fit(
        self,
        dataset: Any,
        targets: list[int],
        train_fn: Any,
        device: torch.device | None = None,
    ) -> None:
        """Train boosted ensemble.

        Parameters
        ----------
        dataset : Dataset
            Full training dataset.
        targets : list[int]
            Labels.
        train_fn : callable
            ``(model, loader) -> None``.
        device : torch.device, optional
        """
        dev = device or torch.device("cpu")
        n = len(targets)
        w = np.ones(n, dtype=np.float64) / n

        counts = Counter(targets)
        median_count = float(np.median(
            list(counts.values()),
        ))
        is_minority = np.array([
            counts[t] < median_count for t in targets
        ])

        for _ in range(self.n_rounds):
            sampler = WeightedRandomSampler(
                w.tolist(), num_samples=n, replacement=True,
            )
            loader = DataLoader(
                dataset, batch_size=64, sampler=sampler,
            )
            model = self.model_fn()
            model.to(dev)
            train_fn(model, loader)

            preds = _predict_all(model, dataset, dev)
            correct = np.array(preds) == np.array(targets)

            err = np.sum(w * ~correct)
            err = np.clip(err, 1e-10, 1.0 - 1e-10)

            alpha = 0.5 * np.log(
                (1.0 - err) / err,
            )
            update = np.where(correct, -alpha, alpha)
            update[~correct & is_minority] *= (
                self.cost_factor
            )
            w *= np.exp(update)
            w /= w.sum()

            self.models.append(model)
            self.alphas.append(float(alpha))

    @torch.no_grad()
    def predict(
        self, inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted vote across all rounds."""
        n_cls: int | None = None
        weighted: torch.Tensor | None = None

        for m, a in zip(self.models, self.alphas):
            m.eval()
            logits = m(inputs)
            if n_cls is None:
                n_cls = logits.size(1)
                weighted = torch.zeros_like(logits)
            preds = logits.argmax(dim=1)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, preds.unsqueeze(1), 1.0)
            weighted += a * one_hot  # type: ignore[operator]

        if weighted is None:
            raise RuntimeError("No models trained.")
        return weighted.argmax(dim=1)


# ── helpers ─────────────────────────────────────────


def _group_by_class(
    targets: list[int],
) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for i, t in enumerate(targets):
        groups.setdefault(t, []).append(i)
    return groups


def _sample_balanced_bag(
    cls_indices: dict[int, list[int]],
    n_per_class: int,
    rng: np.random.RandomState,
) -> list[int]:
    bag: list[int] = []
    for cls, idx in cls_indices.items():
        k = min(n_per_class, len(idx))
        chosen = rng.choice(idx, k, replace=len(idx) < k)
        bag.extend(chosen.tolist())
    return bag


@torch.no_grad()
def _predict_all(
    model: nn.Module,
    dataset: Any,
    device: torch.device,
) -> list[int]:
    model.eval()
    loader = DataLoader(dataset, batch_size=256)
    preds: list[int] = []
    for batch in loader:
        imgs = batch[0].to(device)
        out = model(imgs).argmax(dim=1)
        preds.extend(out.cpu().tolist())
    return preds
