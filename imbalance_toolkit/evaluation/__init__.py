"""Evaluation Metrics — Sec. 4 of Gao et al., 2025.

Metrics designed for imbalanced classification:
Macro-F1, G-Mean, PR-AUC, MCC, Balanced Accuracy.
"""
from imbalance_toolkit.evaluation.metrics import (  # noqa: F401
    macro_f1,
    g_mean,
    pr_auc,
    balanced_accuracy,
    mcc_score,
    full_report,
)
