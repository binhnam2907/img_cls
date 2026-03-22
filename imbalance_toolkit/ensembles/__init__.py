"""Ensemble Learning — Sec. 2.4 of Gao et al., 2025.

Bagging and boosting wrappers that integrate
data re-balancing into ensemble construction.
"""
from imbalance_toolkit.ensembles.wrapper import (  # noqa: F401
    ImbalancedBagging,
    ImbalancedBoosting,
)
