"""The augmentation is exactly the loss, for models linear in the features."""

from __future__ import annotations

import numpy as np
import pytest

from hierarchical_mse import GroupIndex, augment, group_mean_matrix, power_loss

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge  # noqa: E402


def test_weighted_sse_is_proportional_to_the_loss(small_data):
    """The identity the whole module rests on: same objective, up to one constant.

    If the ratio were not constant across coefficient vectors, the augmented fit
    would be minimizing something else.
    """
    idx = GroupIndex(small_data.groups, lam=9.0)
    X, y = small_data.X, small_data.y
    X_aug, y_aug, w = augment(X, y, idx)

    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(5):
        beta, c = rng.normal(size=X.shape[1]), rng.normal()
        sse = float(np.sum(w * (y_aug - (X_aug @ beta + c)) ** 2))
        ratios.append(sse / power_loss(y, X @ beta + c, idx))
    assert np.ptp(ratios) / np.mean(ratios) < 1e-12


def test_augmented_fit_minimises_the_loss(small_data):
    """OLS on the augmented rows lands on the true minimizer of the loss."""
    idx = GroupIndex(small_data.groups, lam=9.0)
    X, y = small_data.X, small_data.y
    fit = LinearRegression().fit(*augment(X, y, idx))
    best = power_loss(y, fit.predict(X), idx)

    rng = np.random.default_rng(1)
    for _ in range(20):
        beta = fit.coef_ + rng.normal(size=X.shape[1]) * 0.05
        c = fit.intercept_ + rng.normal() * 0.05
        assert power_loss(y, X @ beta + c, idx) >= best - 1e-12


def test_lam_zero_reproduces_plain_least_squares(small_data):
    """At lam = 0 the group rows carry zero weight, so it is an ordinary fit."""
    idx = GroupIndex(small_data.groups, lam=0.0)
    X, y = small_data.X, small_data.y
    augmented = LinearRegression().fit(*augment(X, y, idx))
    plain = LinearRegression().fit(X, y)
    np.testing.assert_allclose(augmented.coef_, plain.coef_, atol=1e-8)
    np.testing.assert_allclose(augmented.intercept_, plain.intercept_, atol=1e-8)


def test_group_mean_matrix(small_data):
    idx = GroupIndex(small_data.groups)
    means = group_mean_matrix(small_data.X, idx)
    assert means.shape == (idx.n_groups, small_data.X.shape[1])
    for j in range(small_data.X.shape[1]):
        np.testing.assert_allclose(means[:, j], idx.group_mean(small_data.X[:, j]))


def test_shapes_and_weights(small_data):
    idx = GroupIndex(small_data.groups, lam=4.0)
    X_aug, y_aug, w = augment(small_data.X, small_data.y, idx)
    assert X_aug.shape == (idx.n + idx.n_groups, small_data.X.shape[1])
    assert y_aug.shape == (idx.n + idx.n_groups,)
    assert w.sum() == pytest.approx(idx.n)  # normalize=True keeps the scale comparable

    _, _, raw = augment(small_data.X, small_data.y, idx, normalize=False)
    np.testing.assert_allclose(raw[: idx.n], 1.0)
    np.testing.assert_allclose(raw[idx.n :], idx.lam * idx.nbar)
    # the group-row weight is exactly n_b * kappa_b
    np.testing.assert_allclose(raw[idx.n :], idx.sizes * idx.kappa)


@pytest.mark.parametrize("estimator", [Ridge(alpha=1.0), ElasticNet(alpha=0.01, max_iter=10000)])
def test_regularized_estimators_accept_it(small_data, estimator):
    idx = GroupIndex(small_data.groups)
    estimator.fit(*augment(small_data.X, small_data.y, idx))
    assert np.isfinite(estimator.predict(small_data.X)).all()


def test_raises_on_shape_mismatch(small_data):
    idx = GroupIndex(small_data.groups)
    with pytest.raises(ValueError, match="expected"):
        augment(small_data.X, small_data.y[:-1], idx)
    with pytest.raises(ValueError, match="shape"):
        group_mean_matrix(small_data.X[:-1], idx)


def test_fixed_penalty_comparison_is_misleading():
    """At a FIXED heavy penalty the loss looks better -- and that comparison is invalid.

    Two facts, asserted together because either alone misleads:

    1. At ``alpha=3e5`` the loss-trained model really does retain more between-group
       information (higher ``rho_between``).
    2. But MSE tuned to its *own* best alpha beats the loss at *any* alpha, on
       ``rho_between``. So (1) is "degrades more gracefully when both models are
       wrecked", not "is better".

    ``normalize=True`` equalizes total sample weight but not effective regularization:
    the group-mean rows have much smaller feature variance, so at equal alpha the
    loss-trained fit is shrunk far harder. An earlier README claimed a +32% win from
    exactly this comparison; it was retracted.
    """
    from hierarchical_mse import rho_between

    from .conftest import make_grouped_data, split_by_group

    full = make_grouped_data(seed=7, n_groups=400, nbar=40.0, cv=1.5, s_between=0.15)
    train, valid = split_by_group(full, frac=0.6)
    idx_tr = GroupIndex(train.groups)
    idx_va = GroupIndex(valid.groups, lam=idx_tr.lam)

    def rho_at(alpha, use_loss):
        fit = (
            Ridge(alpha=alpha).fit(*augment(train.X, train.y, idx_tr))
            if use_loss
            else Ridge(alpha=alpha).fit(train.X, train.y)
        )
        return rho_between(valid.y, fit.predict(valid.X), idx_va)

    # (1) at one fixed, heavy alpha the loss looks better
    assert rho_at(3e5, True) > rho_at(3e5, False) + 0.02

    # (2) but tuned per method, MSE wins outright
    alphas = [1e-2, 1e0, 1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6]
    best_mse = max(rho_at(a, False) for a in alphas)
    best_loss = max(rho_at(a, True) for a in alphas)
    assert best_mse > best_loss, "if this ever fails, the retraction should be revisited"


def test_loss_training_produces_a_miscalibrated_predictor():
    """Documents *why* raw squared error can look worse despite a better fit.

    The loss deliberately reallocates capacity toward group means, which drives the
    between-group regression slope of y on g away from 1. Raw squared error
    punishes that scale error; a per-level rescale recovers it (README).
    """
    from .conftest import make_grouped_data, split_by_group

    full = make_grouped_data(seed=7, n_groups=400, nbar=40.0, cv=1.5, s_between=0.15)
    train, _ = split_by_group(full, frac=0.6)
    idx = GroupIndex(train.groups)

    def between_slope(g):
        gb, yb = idx.group_mean(g), idx.group_mean(train.y)
        gc, yc = gb - gb.mean(), yb - yb.mean()
        return (gc @ yc) / (gc @ gc)

    alpha = 3e5
    mse_slope = between_slope(Ridge(alpha=alpha).fit(train.X, train.y).predict(train.X))
    loss_slope = between_slope(
        Ridge(alpha=alpha).fit(*augment(train.X, train.y, idx)).predict(train.X)
    )
    assert loss_slope > mse_slope > 1.0


def test_loss_beats_mse_when_the_group_signal_is_expensive():
    """The existence proof: a regime where the loss genuinely wins, fairly measured.

    Each method is tuned to its own best alpha, and validation is an entirely
    independent draw of groups -- so neither the shared-hyperparameter confound nor
    a shared-groups leak can explain it.

    The contrast with `make_grouped_data` is the whole lesson: there the group signal
    is two strong clean features, MSE reaches the ceiling for free, and no loss can
    help. Here it takes 20 weak features to recover, so capacity genuinely binds.
    """
    from hierarchical_mse import rho_between

    from .conftest import make_hard_grouped_data

    train = make_hard_grouped_data(seed=0)
    valid = make_hard_grouped_data(seed=1)
    idx_tr = GroupIndex(train.groups)
    idx_va = GroupIndex(valid.groups, lam=idx_tr.lam)
    X_aug, y_aug, w = augment(train.X, train.y, idx_tr)

    def best(use_loss):
        scores = []
        for alpha in (1e0, 1e2, 1e3, 3e3, 1e4, 3e4):
            fit = (
                Ridge(alpha=alpha).fit(X_aug, y_aug, sample_weight=w)
                if use_loss
                else Ridge(alpha=alpha).fit(train.X, train.y)
            )
            pred = fit.predict(valid.X)
            scores.append((power_loss(valid.y, pred, idx_va), rho_between(valid.y, pred, idx_va)))
        return min(scores)

    loss_best, loss_rho = best(True)
    mse_best, mse_rho = best(False)
    assert loss_best < mse_best, "loss should win on held-out loss when capacity binds"
    assert loss_rho > mse_rho, "and on between-group correlation"
