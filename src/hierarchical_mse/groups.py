"""Group geometry: sizes, dispersion, and the per-group coefficient ``kappa``."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["GroupIndex"]


def _factorize(groups: Any) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary group labels to contiguous integer codes plus group sizes."""
    arr = np.asarray(groups)
    if arr.ndim != 1:
        raise ValueError(f"groups must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("groups must be non-empty")
    _, codes = np.unique(arr, return_inverse=True)
    codes = codes.astype(np.intp, copy=False).ravel()
    return codes, np.bincount(codes).astype(np.float64)


class GroupIndex:
    """Precomputed group structure for the hierarchical MSE.

    Built once and reused across boosting rounds, so the ``bincount``\\ s and
    ``kappa`` are not recomputed on every gradient evaluation.

    Parameters
    ----------
    groups
        1-D array of group labels, one per row, in the same order as the training
        rows. Any dtype that ``np.unique`` accepts (ints, strings, tuples) works;
        labels need not be contiguous or sorted.
    lam
        Weight on the between-group error term. ``None`` (default) uses
        ``nbar * (1 + cv2)`` computed from ``groups``.

        For general grouped regression this is simply a knob: ``lam=0`` is
        ordinary MSE, larger values buy between-group accuracy at the cost of
        within-group accuracy. In the variance-reduction setting it comes from
        (arXiv:2607.27376), ``lam`` is a *design constant* computed from the
        group-size distribution of the data you ultimately care about -- which may
        differ from the training data. Pass it explicitly in that case.

        The default is aggressive for general use, and with a high-capacity
        learner large ``lam`` can overfit the between-group term badly. See the
        "When this helps" section of the README before adopting it.
    group_weights
        Optional per-group weights ``omega_b`` (length ``n_groups``, ordered by
        sorted group label) on the between-group term. Default ``None`` is uniform.
        ``omega_b`` proportional to group size is inverse-variance weighting of the
        group mean, which mitigates the fact that ``Var(ybar_b)`` grows as
        ``1 / n_b`` -- small groups have noisy targets and uniform weighting
        over-trusts them.

    Attributes
    ----------
    codes : np.ndarray
        Integer group code per row, shape ``(n,)``.
    sizes : np.ndarray
        Rows per group, shape ``(n_groups,)``.
    nbar, cv2 : float
        Mean group size and squared coefficient of variation of group sizes.
    lam : float
        The between-group weight actually in use.
    kappa : np.ndarray
        Per-group coefficient, shape ``(n_groups,)``. See :mod:`hierarchical_mse.loss`.
    """

    __slots__ = (
        "codes",
        "sizes",
        "n",
        "n_groups",
        "nbar",
        "cv2",
        "lam",
        "group_weights",
        "kappa",
    )

    def __init__(self, groups: Any, lam: float | None = None, group_weights: Any = None):
        self.codes, self.sizes = _factorize(groups)
        self.n = int(self.codes.size)
        self.n_groups = int(self.sizes.size)
        self.nbar = self.n / self.n_groups
        self.cv2 = float(self.sizes.var() / self.nbar**2)

        if lam is None:
            self.lam = float(self.nbar * (1.0 + self.cv2))
        else:
            self.lam = float(lam)
            if self.lam < 0.0 or not np.isfinite(self.lam):
                raise ValueError(f"lam must be finite and non-negative, got {lam!r}")

        if group_weights is None:
            omega = np.ones(self.n_groups, dtype=np.float64)
        else:
            omega = np.asarray(group_weights, dtype=np.float64).ravel()
            if omega.size != self.n_groups:
                raise ValueError(f"group_weights has length {omega.size}, expected {self.n_groups}")
            if not np.all(np.isfinite(omega)) or np.any(omega < 0) or omega.sum() <= 0:
                raise ValueError("group_weights must be finite, non-negative and not all zero")
        self.group_weights = omega

        # kappa_b = lam * N * omega_b / (n_b * sum(omega)).
        # omega = 1        -> kappa_b = lam * nbar / n_b
        # omega = n_b      -> kappa_b = lam           (constant across groups)
        self.kappa = self.lam * self.n * omega / (self.sizes * omega.sum())

    def group_mean(self, v: np.ndarray) -> np.ndarray:
        """Group means of a length-``n`` vector, returned with length ``n_groups``."""
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (self.n,):
            raise ValueError(f"expected shape ({self.n},), got {v.shape}")
        return np.bincount(self.codes, weights=v, minlength=self.n_groups) / self.sizes

    def broadcast(self, v_group: np.ndarray) -> np.ndarray:
        """Expand a length-``n_groups`` vector back to one value per row."""
        return np.asarray(v_group, dtype=np.float64)[self.codes]

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (
            f"GroupIndex(n={self.n}, n_groups={self.n_groups}, nbar={self.nbar:.3g}, "
            f"cv2={self.cv2:.3g}, lam={self.lam:.6g})"
        )
