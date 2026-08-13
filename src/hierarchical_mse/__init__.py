"""hierarchical-mse: a regression loss for grouped data.

Ordinary MSE weights every row equally. When rows belong to groups and what you
care about is accuracy on the *group aggregates*, that is the wrong allocation:
most of the squared-error mass sits in within-group noise, so a capacity- or
regularization-constrained model chases that noise and under-fits the group-level
signal. The more rows per group, the worse the mismatch.

This package provides::

    L(g) = MSE_total(g) + lam * MSE_between(g)

with its gradient, Hessian, and a LightGBM objective. ``lam = 0`` is ordinary MSE.

    >>> import numpy as np
    >>> from hierarchical_mse import GroupIndex, power_loss
    >>> groups = np.repeat([0, 1, 2], 4)
    >>> idx = GroupIndex(groups, lam=10.0)
    >>> y = np.arange(12.0)
    >>> round(power_loss(y, np.full(12, y.mean()), idx), 4)
    118.5833

The LightGBM helpers require the optional extra::

    pip install 'hierarchical-mse[lightgbm]'

They can be imported without it; the error is raised only when you call them.
"""

from __future__ import annotations

from .groups import GroupIndex
from .lgbm import lgbm_eval, lgbm_objective
from .loss import (
    LossParts,
    grad_hess,
    init_raw_score,
    power_loss,
    power_loss_parts,
    rho_between,
)

__version__ = "0.1.0"

__all__ = [
    "GroupIndex",
    "LossParts",
    "__version__",
    "grad_hess",
    "init_raw_score",
    "lgbm_eval",
    "lgbm_objective",
    "power_loss",
    "power_loss_parts",
    "rho_between",
]
