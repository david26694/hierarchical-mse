"""The hierarchical MSE: value, its orthogonal parts, gradient and Hessian.

The loss is ordinary MSE plus a penalty on the error of the *group means*::

    L(g) = MSE_total(g) + lam * MSE_between(g)

    MSE_total(g)   = mean over rows of (y - g)^2
    MSE_between(g) = weighted mean over groups of (ybar_b - gbar_b)^2

``lam = 0`` recovers ordinary MSE. Larger ``lam`` buys accuracy on group
aggregates at the cost of accuracy on individual rows -- worthwhile whenever a
capacity- or regularization-constrained model would otherwise spend itself on
within-group noise, which is where most of the squared-error mass lives.

Gradients and Hessians are returned for the loss rescaled by ``N / 2``. That is a
monotone rescaling (identical minimizer) which puts the Hessian at ``O(1)``; see
``hessian-bound.md`` for why that matters to LightGBM. Writing
``kappa_b = lam * N * omega_b / (n_b * sum(omega))``::

    grad_i = -e_i - kappa_b * ebar_b
    hess_i = 1 + kappa_b / n_b      # exact diagonal of the true Hessian
    hess_i = 1 + kappa_b            # majorizing bound -- the default, and safer

The gradient is the ordinary residual with the group-mean residual added back,
inflated by ``kappa_b``. Since boosting split gains are computed from sums of
gradients, any split separating groups sees a signal roughly ``lam`` times larger
than it would under plain MSE. That is the entire mechanism.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from .groups import GroupIndex

__all__ = [
    "LossParts",
    "grad_hess",
    "init_raw_score",
    "power_loss",
    "power_loss_parts",
    "rho_between",
]

HessianMode = Literal["bound", "diag"]


class LossParts(NamedTuple):
    """Decomposition of the loss. See :func:`power_loss_parts`."""

    total: float
    """``mse_total + penalty`` -- equals :func:`power_loss`."""
    mse_total: float
    """Mean over rows of the squared residual."""
    mse_within: float
    """Mean over rows of the group-demeaned squared residual."""
    mse_between_rows: float
    """Size-weighted mean over groups of the squared group-mean residual.
    Satisfies ``mse_total == mse_within + mse_between_rows`` exactly (the ANOVA
    identity), because the two components are orthogonal."""
    mse_between_groups: float
    """Unweighted mean over groups of the squared group-mean residual. This is
    what the penalty measures when ``group_weights`` is uniform. It differs from
    ``mse_between_rows`` unless all groups are the same size."""
    penalty: float
    """``lam`` times the ``group_weights``-weighted mean of the squared group-mean
    residual. Equals ``lam * mse_between_groups`` for uniform weights."""


def _residual(y: np.ndarray, pred: np.ndarray, idx: GroupIndex) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if y.shape != (idx.n,):
        raise ValueError(f"y has shape {y.shape}, expected ({idx.n},) to match the GroupIndex")
    if pred.shape != (idx.n,):
        raise ValueError(f"pred has shape {pred.shape}, expected ({idx.n},)")
    return y - pred


def power_loss_parts(y: np.ndarray, pred: np.ndarray, idx: GroupIndex) -> LossParts:
    """Break the loss into its orthogonal within- and between-group components."""
    e = _residual(y, pred, idx)
    ebar = idx.group_mean(e)
    omega = idx.group_weights

    mse_total = float(np.mean(e**2))
    mse_within = float(np.mean((e - ebar[idx.codes]) ** 2))
    mse_between_rows = float(np.sum(idx.sizes * ebar**2) / idx.n)
    mse_between_groups = float(np.mean(ebar**2))
    penalty = float(idx.lam * np.sum(omega * ebar**2) / omega.sum())
    return LossParts(
        total=mse_total + penalty,
        mse_total=mse_total,
        mse_within=mse_within,
        mse_between_rows=mse_between_rows,
        mse_between_groups=mse_between_groups,
        penalty=penalty,
    )


def power_loss(y: np.ndarray, pred: np.ndarray, idx: GroupIndex) -> float:
    """The loss value, ``MSE_total + lam * MSE_between``. Lower is better."""
    e = _residual(y, pred, idx)
    ebar = idx.group_mean(e)
    omega = idx.group_weights
    return float(np.mean(e**2) + idx.lam * np.sum(omega * ebar**2) / omega.sum())


def rho_between(y: np.ndarray, pred: np.ndarray, idx: GroupIndex) -> float:
    """Correlation between group means of ``y`` and of ``pred``, across groups.

    The quantity to watch when comparing models: it is the channel through which
    a training loss affects downstream between-group accuracy. Returns ``nan``
    if either set of group means is constant.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    ybar, gbar = idx.group_mean(y), idx.group_mean(pred)
    if ybar.size < 2 or np.ptp(ybar) == 0 or np.ptp(gbar) == 0:
        return float("nan")
    return float(np.corrcoef(ybar, gbar)[0, 1])


def init_raw_score(y: np.ndarray, idx: GroupIndex) -> float:
    """The constant prediction minimizing the loss.

    LightGBM's ``boost_from_average`` does not apply to custom objectives -- the
    initial raw score is 0 -- so pass this as ``init_score``. For large ``lam`` it
    approaches the (weighted) mean of the *group* means rather than the grand
    mean of ``y``; those differ materially when group sizes are dispersed.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.shape != (idx.n,):
        raise ValueError(f"y has shape {y.shape}, expected ({idx.n},)")
    omega = idx.group_weights
    ybar_weighted = float(np.sum(omega * idx.group_mean(y)) / omega.sum())
    return float((y.mean() + idx.lam * ybar_weighted) / (1.0 + idx.lam))


def grad_hess(
    y: np.ndarray,
    pred: np.ndarray,
    idx: GroupIndex,
    hessian: HessianMode = "bound",
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and Hessian of the ``N/2``-rescaled loss w.r.t. ``pred``.

    Parameters
    ----------
    hessian
        ``"bound"`` (default) returns ``1 + kappa_b``, the largest eigenvalue of the
        true per-group Hessian block. It majorizes that block, which makes a
        boosting step a majorize-minimize step and guarantees descent at
        ``learning_rate=1.0``. It is *exact* for leaves holding a whole group and
        conservative for mixed leaves.

        ``"diag"`` returns the true diagonal ``1 + kappa_b / n_b``. This does **not**
        majorize: for a leaf holding a whole group it understates curvature by up
        to a factor ``n_b``, so leaf values overshoot by the same factor -- on
        precisely the group-separating splits the loss exists to encourage. Only
        use it with a learning rate well below ``1 / nbar``. See
        ``hessian-bound.md``.
    """
    if hessian not in ("bound", "diag"):
        raise ValueError(f"hessian must be 'bound' or 'diag', got {hessian!r}")

    e = _residual(y, pred, idx)
    ebar = idx.group_mean(e)
    kappa = idx.kappa

    grad = -e - (kappa * ebar)[idx.codes]
    per_group_hess = 1.0 + (kappa if hessian == "bound" else kappa / idx.sizes)
    hess = per_group_hess[idx.codes]
    return grad, hess
