"""Data Re-balancing — Sec. 2.1 of Gao et al., 2025.

Linear generation, deep generation, and modern
interpolation methods for minority oversampling.
"""
from imbalance_toolkit.sampling.smote import (  # noqa: F401
    SMOTE,
    ADASYN,
)
from imbalance_toolkit.sampling.vae_sampler import (  # noqa: F401
    VAESampler,
)
from imbalance_toolkit.sampling.remix import (  # noqa: F401
    Remix,
    BalancedMixup,
)
