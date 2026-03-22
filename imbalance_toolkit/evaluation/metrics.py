"""Evaluation metrics for imbalanced classification.

Based on Sec. 4 of Gao et al., 2025:
    "Traditional metrics like accuracy can not provide a
     reliable measure in imbalanced contexts ... To counter
     this bias, it's crucial to use evaluation metrics that
     accurately reflect performance across all classes."

Implements (citing paper equations):
- Macro-F1  (Eq. 4, macro-averaged)
- G-Mean    (Eq. 6)
- PR-AUC    (area under Precision-Recall curve)
- Balanced Accuracy (Eq. 5)
- MCC       (Eq. 7)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
)


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Macro-averaged F1 (Eq. 4, Sec. 4).

    Computes F1 per class then averages — every class
    has equal weight regardless of support.
    """
    return float(
        f1_score(
            y_true, y_pred,
            average="macro", zero_division=0,
        ),
    )


def g_mean(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Geometric Mean of per-class recalls (Eq. 6).

    G-Mean = (prod_c recall_c) ^ (1/K)

    Sensitive to poor performance on any single class,
    making it ideal for imbalanced evaluation.
    """
    classes = np.unique(y_true)
    recalls: list[float] = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            recalls.append(0.0)
            continue
        recalls.append(
            float((y_pred[mask] == c).mean()),
        )
    if not recalls or 0.0 in recalls:
        return 0.0
    return float(
        np.prod(recalls) ** (1.0 / len(recalls)),
    )


def pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Macro-averaged Precision-Recall AUC.

    For multi-class, computes one-vs-rest PR-AUC per class
    and macro-averages.

    Parameters
    ----------
    y_true  : (N,) integer labels.
    y_score : (N, C) class probabilities / logits.
    """
    classes = np.unique(y_true)
    aucs: list[float] = []
    for c in classes:
        binary = (y_true == c).astype(int)
        scores = y_score[:, c]
        precision, recall, _ = precision_recall_curve(
            binary, scores,
        )
        aucs.append(float(np.trapz(precision, recall)))
    return float(np.mean(aucs)) if aucs else 0.0


def balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Balanced Accuracy (Eq. 5, Sec. 4).

    BA = mean of per-class recalls.
    """
    return float(
        balanced_accuracy_score(y_true, y_pred),
    )


def mcc_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Matthews Correlation Coefficient (Eq. 7, Sec. 4).

    Accounts for all four quadrants of the confusion matrix.
    """
    return float(matthews_corrcoef(y_true, y_pred))


def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all imbalance-aware metrics at once.

    Parameters
    ----------
    y_true  : (N,) true labels.
    y_pred  : (N,) predicted labels.
    y_score : (N, C) optional probabilities for PR-AUC.

    Returns
    -------
    dict with keys: macro_f1, g_mean, balanced_accuracy,
    mcc, pr_auc (if y_score provided).
    """
    result: dict[str, float] = {
        "macro_f1": macro_f1(y_true, y_pred),
        "g_mean": g_mean(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(
            y_true, y_pred,
        ),
        "mcc": mcc_score(y_true, y_pred),
    }
    if y_score is not None:
        result["pr_auc"] = pr_auc(y_true, y_score)
    return result
