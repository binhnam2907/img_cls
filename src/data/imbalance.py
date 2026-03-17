"""Utilities for handling class-imbalanced datasets.

Strategies:
  1. make_imbalanced  -> subsample to create long-tail
  2. class_weights    -> weighted CrossEntropyLoss
  3. sample_weights   -> WeightedRandomSampler
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler


def make_imbalanced(
    dataset: Dataset,
    imbalance_ratio: float = 20.0,
    seed: int = 42,
) -> Subset:
    """Subsample a balanced dataset into a long-tail
    distribution.

    The most frequent class keeps all its samples;
    the least frequent class keeps
    ``max_count / imbalance_ratio`` samples.
    Intermediate classes follow exponential decay.
    """
    targets = _extract_targets(dataset)
    classes = sorted(set(targets))
    n_classes = len(classes)
    max_count = max(Counter(targets).values())

    mu = math.log(imbalance_ratio) / (n_classes - 1)

    rng = np.random.RandomState(seed)
    selected: list[int] = []

    for rank, cls_idx in enumerate(classes):
        cls_indices = [
            i for i, t in enumerate(targets)
            if t == cls_idx
        ]
        n_keep = max(
            int(max_count * math.exp(-mu * rank)),
            1,
        )
        n_keep = min(n_keep, len(cls_indices))
        chosen = rng.choice(
            cls_indices, n_keep, replace=False,
        )
        selected.extend(chosen.tolist())

    selected.sort()
    subset = Subset(dataset, selected)
    subset.targets = [targets[i] for i in selected]

    if hasattr(dataset, "classes"):
        subset.classes = dataset.classes

    return subset


def compute_class_weights(
    dataset: Dataset,
) -> torch.Tensor:
    """Inverse-frequency weights: w_c = N / (K * n_c).

    Returns a float tensor of shape (num_classes,).
    """
    targets = _extract_targets(dataset)
    counts = Counter(targets)
    n_classes = len(counts)
    total = len(targets)

    weights = torch.zeros(n_classes)
    for cls_idx, count in counts.items():
        weights[cls_idx] = total / (n_classes * count)

    return weights


def build_weighted_sampler(
    dataset: Dataset,
) -> WeightedRandomSampler:
    """Per-sample weights so rare classes are
    sampled more often."""
    targets = _extract_targets(dataset)
    counts = Counter(targets)

    class_weight = {
        cls: 1.0 / count
        for cls, count in counts.items()
    }
    sample_weights = [
        class_weight[t] for t in targets
    ]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def _extract_targets(dataset: Dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    raise AttributeError(
        "Dataset has no 'targets' or 'labels' attr. "
        "Cannot compute class weights."
    )
