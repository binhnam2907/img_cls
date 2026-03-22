"""Feature Representation — Sec. 2.2 of Gao et al., 2025.

Loss functions for metric learning and supervised
contrastive learning, tailored for imbalanced sets.
"""
from imbalance_toolkit.representation.metric_learning import (  # noqa: F401,E501
    TripletLoss,
    ContrastiveLoss,
)
from imbalance_toolkit.representation.supcon import (  # noqa: F401
    SupConLoss,
)
