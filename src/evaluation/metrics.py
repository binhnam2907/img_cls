from __future__ import annotations

import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


@torch.no_grad()
def accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[float]:
    max_k = max(topk)
    batch_size = labels.size(0)

    _, top_idx = logits.topk(
        max_k, dim=1, largest=True, sorted=True,
    )
    correct = top_idx.t().eq(
        labels.view(1, -1).expand_as(top_idx.t())
    )

    return [
        correct[:k]
        .reshape(-1)
        .float()
        .sum()
        .mul_(100.0 / batch_size)
        .item()
        for k in topk
    ]


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    report = classification_report(
        ground_truth, predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": (
            report["weighted avg"]["f1-score"]
        ),
        "confusion_matrix": confusion_matrix(
            ground_truth, predictions,
        ),
        "classification_report": report,
    }


def per_class_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    report = classification_report(
        ground_truth, predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        name: scores["precision"]
        for name, scores in report.items()
        if isinstance(scores, dict)
    }
