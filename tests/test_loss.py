"""T8 and the pure-numpy loss surface."""

from __future__ import annotations

import numpy as np
import pytest

from hierarchical_mse import (
    GroupIndex,
    init_raw_score,
    power_loss,
    power_loss_parts,
    rho_between,
)


def test_anova_identity(small_data):
    """T8: MSE_total = MSE_within + MSE_between_rows, because the parts are orthogonal."""
    idx = GroupIndex(small_data.groups)
    rng = np.random.default_rng(0)
    for _ in range(5):
        pred = rng.normal(size=small_data.y.size)
        parts = power_loss_parts(small_data.y, pred, idx)
        assert parts.mse_total == pytest.approx(parts.mse_within + parts.mse_between_rows)


def test_parts_sum_to_total(small_data):
    idx = GroupIndex(small_data.groups, lam=7.0)
    pred = np.random.default_rng(1).normal(size=small_data.y.size)
    parts = power_loss_parts(small_data.y, pred, idx)
    assert parts.total == pytest.approx(parts.mse_total + parts.penalty)
    assert parts.total == pytest.approx(power_loss(small_data.y, pred, idx))


def test_penalty_is_lam_times_between_groups(small_data):
    """With uniform weights the penalty is exactly lam * mse_between_groups."""
    idx = GroupIndex(small_data.groups, lam=3.0)
    pred = np.random.default_rng(2).normal(size=small_data.y.size)
    parts = power_loss_parts(small_data.y, pred, idx)
    assert parts.penalty == pytest.approx(3.0 * parts.mse_between_groups)


def test_between_rows_and_groups_differ_when_sizes_differ(small_data):
    """The two between-group conventions coincide only for equal group sizes."""
    idx = GroupIndex(small_data.groups)
    assert idx.cv2 > 0
    parts = power_loss_parts(small_data.y, np.zeros(small_data.y.size), idx)
    assert parts.mse_between_rows != pytest.approx(parts.mse_between_groups)

    equal = GroupIndex(np.repeat([0, 1, 2], 5))
    y = np.arange(15.0)
    p = power_loss_parts(y, np.zeros(15), equal)
    assert p.mse_between_rows == pytest.approx(p.mse_between_groups)


def test_lam_zero_is_plain_mse(small_data):
    """The headline claim: lam = 0 recovers ordinary MSE exactly."""
    idx = GroupIndex(small_data.groups, lam=0.0)
    pred = np.random.default_rng(3).normal(size=small_data.y.size)
    assert power_loss(small_data.y, pred, idx) == pytest.approx(np.mean((small_data.y - pred) ** 2))


def test_loss_is_zero_for_perfect_prediction(small_data):
    idx = GroupIndex(small_data.groups)
    assert power_loss(small_data.y, small_data.y, idx) == pytest.approx(0.0)


def test_larger_lam_penalises_between_group_error_more(small_data):
    """A prediction wrong only at the group level costs more as lam grows."""
    idx_lo = GroupIndex(small_data.groups, lam=1.0)
    idx_hi = GroupIndex(small_data.groups, lam=100.0)
    pred = small_data.y + idx_lo.broadcast(np.ones(idx_lo.n_groups))  # constant per-group offset
    assert power_loss(small_data.y, pred, idx_hi) > power_loss(small_data.y, pred, idx_lo)


def test_within_group_only_error_is_insensitive_to_lam(small_data):
    """An error that averages to zero inside every group is not penalised by lam."""
    idx_lo = GroupIndex(small_data.groups, lam=1.0)
    idx_hi = GroupIndex(small_data.groups, lam=100.0)
    noise = np.random.default_rng(4).normal(size=small_data.y.size)
    noise -= idx_lo.broadcast(idx_lo.group_mean(noise))  # exactly group-demeaned
    pred = small_data.y + noise
    assert power_loss(small_data.y, pred, idx_hi) == pytest.approx(
        power_loss(small_data.y, pred, idx_lo)
    )


def test_init_raw_score_minimises_the_loss(small_data):
    """init_raw_score is the argmin over constant predictions."""
    idx = GroupIndex(small_data.groups, lam=12.0)
    c = init_raw_score(small_data.y, idx)
    best = power_loss(small_data.y, np.full(idx.n, c), idx)
    for delta in (-0.5, -0.05, 0.05, 0.5):
        other = power_loss(small_data.y, np.full(idx.n, c + delta), idx)
        assert other > best


def test_init_raw_score_limits(small_data):
    """lam = 0 gives the grand mean; large lam approaches the mean of group means."""
    y = small_data.y
    assert init_raw_score(y, GroupIndex(small_data.groups, lam=0.0)) == pytest.approx(y.mean())

    idx = GroupIndex(small_data.groups, lam=1e9)
    mean_of_group_means = idx.group_mean(y).mean()
    assert init_raw_score(y, idx) == pytest.approx(mean_of_group_means, rel=1e-6)
    # and the two genuinely differ, so the distinction matters
    assert mean_of_group_means != pytest.approx(y.mean())


def test_rho_between(small_data):
    idx = GroupIndex(small_data.groups)
    assert rho_between(small_data.y, small_data.y, idx) == pytest.approx(1.0)
    assert rho_between(small_data.y, -small_data.y, idx) == pytest.approx(-1.0)
    assert np.isnan(rho_between(small_data.y, np.ones(idx.n), idx))


def test_shape_mismatch_raises(small_data):
    idx = GroupIndex(small_data.groups)
    with pytest.raises(ValueError, match="expected"):
        power_loss(small_data.y[:-1], small_data.y[:-1], idx)
