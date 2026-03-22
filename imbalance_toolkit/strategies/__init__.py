"""Training Strategies — Sec. 2.3 of Gao et al., 2025.

Decoupled training, posterior re-calibration, and
cost-sensitive learning methods.
"""
from imbalance_toolkit.strategies.decoupled import (  # noqa: F401
    DecoupledTrainer,
)
from imbalance_toolkit.strategies.balanced_softmax import (  # noqa: F401
    BalancedSoftmaxLoss,
    LogitAdjustmentLoss,
)
from imbalance_toolkit.strategies.cost_sensitive import (  # noqa: F401
    ClassWeightedCE,
    FocalLoss,
    ClassBalancedLoss,
)
