"""T2 and T3: the gradient is right, and the Hessian bound is necessary and sufficient."""

from __future__ import annotations

import numpy as np
import pytest

from hierarchical_mse import GroupIndex, grad_hess, power_loss


def _scaled_loss(y, pred, idx):
    """The loss actually differentiated by grad_hess: power_loss rescaled by N/2."""
    return (idx.n / 2.0) * power_loss(y, pred, idx)


def test_gradient_matches_finite_differences(small_data):
    """T2: analytic gradient against central differences on every row."""
    idx = GroupIndex(small_data.groups, lam=8.0)
    y = small_data.y
    pred = np.random.default_rng(0).normal(size=idx.n) * 0.5
    grad, _ = grad_hess(y, pred, idx)

    h = 1e-6
    numeric = np.empty(idx.n)
    for i in range(idx.n):
        up, down = pred.copy(), pred.copy()
        up[i] += h
        down[i] -= h
        numeric[i] = (_scaled_loss(y, up, idx) - _scaled_loss(y, down, idx)) / (2 * h)

    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-6)


def test_hessian_diag_matches_finite_differences(small_data):
    """T2: the 'diag' mode really is the diagonal of the true Hessian."""
    idx = GroupIndex(small_data.groups, lam=8.0)
    y = small_data.y
    pred = np.random.default_rng(1).normal(size=idx.n) * 0.5
    _, hess = grad_hess(y, pred, idx, hessian="diag")

    h = 1e-4
    numeric = np.empty(idx.n)
    base = _scaled_loss(y, pred, idx)
    for i in range(idx.n):
        up, down = pred.copy(), pred.copy()
        up[i] += h
        down[i] -= h
        numeric[i] = (_scaled_loss(y, up, idx) - 2 * base + _scaled_loss(y, down, idx)) / h**2

    np.testing.assert_allclose(hess, numeric, rtol=1e-3, atol=1e-4)


def _true_block(kappa_b: float, n_b: int) -> np.ndarray:
    """The exact per-group Hessian block of the N/2-rescaled loss."""
    return np.eye(n_b) + (kappa_b / n_b) * np.ones((n_b, n_b))


@pytest.mark.parametrize("n_b", [2, 5, 20])
def test_bound_majorises_true_hessian(n_b):
    """T3 (sufficient): diag(1 + kappa) - H is PSD, so the surrogate is an upper bound."""
    kappa_b = 30.0
    H = _true_block(kappa_b, n_b)
    surplus = np.diag(np.full(n_b, 1.0 + kappa_b)) - H
    assert np.linalg.eigvalsh(surplus).min() >= -1e-9


@pytest.mark.parametrize("n_b", [2, 5, 20])
def test_exact_diagonal_does_not_majorise(n_b):
    """T3 (necessary): the exact diagonal is NOT an upper bound -- this is why it overshoots.

    Without this half, defaulting to the bound would be superstition.
    """
    kappa_b = 30.0
    H = _true_block(kappa_b, n_b)
    surplus = np.diag(np.full(n_b, 1.0 + kappa_b / n_b)) - H
    assert np.linalg.eigvalsh(surplus).min() < -1e-6


@pytest.mark.parametrize("n_b", [2, 10, 50])
def test_bound_is_tight(n_b):
    """The bound is the block's top eigenvalue: the smallest safe uniform diagonal."""
    kappa_b = 30.0
    assert np.linalg.eigvalsh(_true_block(kappa_b, n_b)).max() == pytest.approx(1.0 + kappa_b)


@pytest.mark.parametrize("n_b", [2, 10, 100])
def test_group_pure_leaf_value_is_exact_under_bound(n_b):
    """The correct Newton step for a leaf holding one whole group is exactly ebar_b.

    The bound reproduces it; the exact diagonal overshoots by roughly n_b.
    """
    lam, ebar = 60.0, 1.0
    groups = np.zeros(n_b, dtype=int)
    idx = GroupIndex(groups, lam=lam)
    kappa_b = idx.kappa[0]

    y = np.full(n_b, ebar)
    grad, hess_bound = grad_hess(y, np.zeros(n_b), idx, hessian="bound")
    _, hess_diag = grad_hess(y, np.zeros(n_b), idx, hessian="diag")

    true_curvature = np.ones(n_b) @ _true_block(kappa_b, n_b) @ np.ones(n_b)
    assert hess_bound.sum() == pytest.approx(true_curvature)

    assert (-grad.sum() / hess_bound.sum()) == pytest.approx(ebar)
    overshoot = (-grad.sum() / hess_diag.sum()) / ebar
    assert overshoot > 1.5
    assert overshoot <= n_b + 1e-9


def test_diagonal_is_exact_for_group_scattered_leaves():
    """The mirror image: with one row per group there are no off-diagonal terms."""
    n_groups = 8
    idx = GroupIndex(np.arange(n_groups), lam=60.0)
    _, hess_diag = grad_hess(np.zeros(n_groups), np.zeros(n_groups), idx, hessian="diag")
    # each group contributes a 1x1 block, so 1^T H 1 == sum of diagonal
    true_curvature = sum(
        np.ones(1) @ _true_block(idx.kappa[b], 1) @ np.ones(1) for b in range(n_groups)
    )
    assert hess_diag.sum() == pytest.approx(true_curvature)


def test_lam_zero_reduces_to_l2_grad_hess(small_data):
    """T1 (pure part): at lam = 0 the formulas are LightGBM's built-in L2 exactly."""
    idx = GroupIndex(small_data.groups, lam=0.0)
    pred = np.random.default_rng(2).normal(size=idx.n)
    grad, hess = grad_hess(small_data.y, pred, idx)
    np.testing.assert_allclose(grad, pred - small_data.y)
    np.testing.assert_allclose(hess, np.ones(idx.n))


def test_gradient_is_zero_at_the_optimum(small_data):
    idx = GroupIndex(small_data.groups, lam=5.0)
    grad, _ = grad_hess(small_data.y, small_data.y, idx)
    np.testing.assert_allclose(grad, np.zeros(idx.n), atol=1e-12)


def test_invalid_hessian_mode(small_data):
    idx = GroupIndex(small_data.groups)
    with pytest.raises(ValueError, match="bound.*diag"):
        grad_hess(small_data.y, small_data.y, idx, hessian="exact")
