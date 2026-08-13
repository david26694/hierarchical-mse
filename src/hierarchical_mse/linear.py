"""Exact training under the hierarchical MSE for models linear in the features.

For any model of the form ``g(x) = x @ beta + c``, the group mean of the predictions
equals the model applied to the group-mean features::

    gbar_b = mean_i g(X[b,i]) = g(Xbar_b)

That identity turns the non-separable between-group term into an ordinary weighted
squared error on **one extra row per group**, so the whole loss becomes plain
weighted least squares::

    sum_r w_r (y_aug_r - g(X_aug_r))^2  ==  N * L(g)   for every linear g

Consequently any scikit-learn estimator that is linear in the features and accepts
``sample_weight`` trains exactly under this loss with no custom optimizer::

    Ridge().fit(*augment(X, y, idx))

The identity fails for trees and other non-linear models -- ``g(mean(x)) != mean(g(x))``
-- which is why :mod:`hierarchical_mse.lgbm` needs a custom gradient instead.
"""

from __future__ import annotations

import numpy as np

from .groups import GroupIndex

__all__ = ["augment", "group_mean_matrix"]


def group_mean_matrix(X: np.ndarray, idx: GroupIndex) -> np.ndarray:
    """Group means of a 2-D array, shape ``(n_groups, n_features)``."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != idx.n:
        raise ValueError(f"X must have shape ({idx.n}, n_features), got {X.shape}")
    sums = np.column_stack(
        [np.bincount(idx.codes, weights=X[:, j], minlength=idx.n_groups) for j in range(X.shape[1])]
    )
    return sums / idx.sizes[:, None]


def augment(
    X: np.ndarray,
    y: np.ndarray,
    idx: GroupIndex,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the augmented design so weighted least squares minimizes the loss.

    Appends one row per group, holding that group's mean features and mean target,
    weighted by ``lam * N * omega_b / sum(omega)``. For uniform weights that is
    ``lam * nbar`` per group row against 1 per original row.

    Parameters
    ----------
    normalize
        Rescale all weights to sum to ``N``. This does not change the minimizer of an
        unregularized fit, but it keeps a regularization strength (``Ridge(alpha=...)``,
        ``ElasticNet(alpha=...)``) comparable to a plain-MSE fit on the same data.
        Without it the data term is roughly ``(1 + lam)`` times larger and the penalty
        is effectively that much weaker, so an MSE-vs-loss comparison would silently
        not be holding capacity fixed. Leave it on unless you know why you want it off.

    Returns
    -------
    (X_aug, y_aug, sample_weight)
        Pass straight to ``estimator.fit(*augment(X, y, idx))``. Use the estimator's
        default ``fit_intercept=True``; the intercept is handled correctly because it
        enters the group-mean rows exactly as it enters ``gbar_b``.

    Notes
    -----
    **Exact only for models linear in the features.** Using this with a tree
    ensemble trains on synthetic group-mean rows, which up-weights group structure
    but is *not* this loss.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.shape != (idx.n,):
        raise ValueError(f"y has shape {y.shape}, expected ({idx.n},)")

    X_groups = group_mean_matrix(X, idx)
    y_groups = idx.group_mean(y)

    omega = idx.group_weights
    w_groups = idx.lam * idx.n * omega / omega.sum()

    X_aug = np.vstack([X, X_groups])
    y_aug = np.concatenate([y, y_groups])
    w = np.concatenate([np.ones(idx.n), w_groups])
    if normalize:
        w = w * (idx.n / w.sum())
    return X_aug, y_aug, w
