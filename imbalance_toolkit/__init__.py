"""Imbalanced Data Toolkit.

A modular Python library implementing the taxonomy from:
    Gao et al., 2025.  "A comprehensive survey on imbalanced
    data learning."  Frontiers of Computer Science, 20(11).

Modules
-------
sampling        Data re-balancing (Sec. 2.1)
representation  Feature representation losses (Sec. 2.2)
strategies      Training strategies (Sec. 2.3)
ensembles       Ensemble wrappers (Sec. 2.4)
evaluation      Imbalance-aware metrics (Sec. 4)
factory         Config-driven method switching
"""
from imbalance_toolkit.factory import create  # noqa: F401

__version__ = "0.1.0"
