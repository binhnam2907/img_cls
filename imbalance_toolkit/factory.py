"""Config-driven Factory for the Imbalanced Data Toolkit.

Usage
-----
Switch methods via a plain dict (or YAML config)::

    cfg = {
        "method": "SMOTE",
        "k_neighbors": 5,
        "target_ratio": 1.0,
    }
    sampler = imbalance_toolkit.create(cfg)

    cfg = {"loss": "FocalLoss", "gamma": 2.0}
    criterion = imbalance_toolkit.create(cfg)

Every registered class is instantiated by extracting the
relevant kwargs from the config and forwarding them to
the constructor.
"""
from __future__ import annotations

from typing import Any

# ── sampling ────────────────────────────────────────
from imbalance_toolkit.sampling.smote import (
    SMOTE,
    ADASYN,
)
from imbalance_toolkit.sampling.vae_sampler import (
    VAESampler,
)
from imbalance_toolkit.sampling.remix import (
    Remix,
    BalancedMixup,
)

# ── representation ──────────────────────────────────
from imbalance_toolkit.representation.metric_learning import (  # noqa: E501
    TripletLoss,
    ContrastiveLoss,
)
from imbalance_toolkit.representation.supcon import (
    SupConLoss,
)

# ── strategies ──────────────────────────────────────
from imbalance_toolkit.strategies.decoupled import (
    DecoupledTrainer,
)
from imbalance_toolkit.strategies.balanced_softmax import (
    BalancedSoftmaxLoss,
    LogitAdjustmentLoss,
)
from imbalance_toolkit.strategies.cost_sensitive import (
    ClassWeightedCE,
    FocalLoss,
    ClassBalancedLoss,
)

# ── ensembles ───────────────────────────────────────
from imbalance_toolkit.ensembles.wrapper import (
    ImbalancedBagging,
    ImbalancedBoosting,
)


_REGISTRY: dict[str, type] = {
    # sampling (Sec. 2.1)
    "SMOTE": SMOTE,
    "ADASYN": ADASYN,
    "VAESampler": VAESampler,
    "Remix": Remix,
    "BalancedMixup": BalancedMixup,
    # representation (Sec. 2.2)
    "TripletLoss": TripletLoss,
    "ContrastiveLoss": ContrastiveLoss,
    "SupConLoss": SupConLoss,
    # strategies (Sec. 2.3)
    "cRT": DecoupledTrainer,
    "DecoupledTrainer": DecoupledTrainer,
    "BalancedSoftmax": BalancedSoftmaxLoss,
    "BalancedSoftmaxLoss": BalancedSoftmaxLoss,
    "LogitAdjustment": LogitAdjustmentLoss,
    "LogitAdjustmentLoss": LogitAdjustmentLoss,
    "ClassWeightedCE": ClassWeightedCE,
    "FocalLoss": FocalLoss,
    "ClassBalancedLoss": ClassBalancedLoss,
    # ensembles (Sec. 2.4)
    "ImbalancedBagging": ImbalancedBagging,
    "ImbalancedBoosting": ImbalancedBoosting,
}


def create(cfg: dict[str, Any]) -> Any:
    """Instantiate a toolkit component from *cfg*.

    The config must contain *exactly one* of the keys
    ``"method"``, ``"loss"``, or ``"trainer"`` whose value
    matches a registered class name.  All other keys
    are forwarded as constructor kwargs.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    An instance of the requested component.

    Raises
    ------
    KeyError
        If no recognised name key is found.
    ValueError
        If the name is not in the registry.

    Examples
    --------
    >>> create({"method": "SMOTE", "k_neighbors": 3})
    SMOTE(k_neighbors=3, ...)

    >>> create({"loss": "FocalLoss", "gamma": 3.0})
    FocalLoss(gamma=3.0, ...)
    """
    name = _resolve_name(cfg)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown component '{name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[name]
    kwargs = {
        k: v for k, v in cfg.items()
        if k not in _NAME_KEYS
    }
    return cls(**kwargs)


def list_available() -> list[str]:
    """Return sorted list of all registered names."""
    return sorted(_REGISTRY)


# ── internals ───────────────────────────────────────

_NAME_KEYS = {"method", "loss", "trainer"}


def _resolve_name(cfg: dict[str, Any]) -> str:
    for key in _NAME_KEYS:
        if key in cfg:
            return str(cfg[key])
    raise KeyError(
        "Config must contain one of: "
        f"{_NAME_KEYS}.  Got keys: {list(cfg)}"
    )
