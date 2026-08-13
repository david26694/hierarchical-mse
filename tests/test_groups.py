"""T9, T10, T11: group geometry, weights and label handling."""

from __future__ import annotations

import numpy as np
import pytest

from hierarchical_mse import GroupIndex


def test_design_constants_equal_sizes():
    """T9: equal group sizes give cv2 = 0, hence lam = nbar."""
    idx = GroupIndex(np.repeat([0, 1, 2, 3], 25))
    assert idx.n == 100
    assert idx.n_groups == 4
    assert idx.nbar == pytest.approx(25.0)
    assert idx.cv2 == pytest.approx(0.0)
    assert idx.lam == pytest.approx(25.0)


def test_design_constants_unequal_sizes():
    """T9: nbar, cv2 and lam match hand computation on a known size vector."""
    sizes = np.array([1, 2, 3, 10])
    idx = GroupIndex(np.repeat(np.arange(4), sizes))
    nbar = sizes.mean()
    cv2 = sizes.var() / nbar**2
    assert idx.sizes.tolist() == sizes.tolist()
    assert idx.nbar == pytest.approx(nbar)
    assert idx.cv2 == pytest.approx(cv2)
    assert idx.lam == pytest.approx(nbar * (1 + cv2))


def test_singleton_groups_have_no_within_variation():
    """T9 (degenerate): with one row per group, all error is between-group."""
    from hierarchical_mse import power_loss_parts

    idx = GroupIndex(np.arange(6))
    y = np.arange(6.0)
    parts = power_loss_parts(y, np.zeros(6), idx)
    assert parts.mse_within == pytest.approx(0.0)
    assert parts.mse_between_rows == pytest.approx(parts.mse_total)


def test_explicit_lam_overrides_default():
    idx = GroupIndex(np.repeat([0, 1], 5), lam=3.5)
    assert idx.lam == pytest.approx(3.5)


def test_lam_zero_is_allowed():
    assert GroupIndex(np.repeat([0, 1], 5), lam=0.0).lam == 0.0


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_lam_rejected(bad):
    with pytest.raises(ValueError, match="lam"):
        GroupIndex(np.repeat([0, 1], 5), lam=bad)


def test_kappa_uniform_weights():
    """T10: with omega = 1, kappa_b = lam * nbar / n_b."""
    sizes = np.array([2, 4, 8])
    idx = GroupIndex(np.repeat(np.arange(3), sizes), lam=6.0)
    np.testing.assert_allclose(idx.kappa, 6.0 * idx.nbar / sizes)


def test_kappa_size_weights_is_constant():
    """T10: with omega proportional to group size, kappa collapses to lam itself."""
    sizes = np.array([2, 4, 8])
    groups = np.repeat(np.arange(3), sizes)
    idx = GroupIndex(groups, lam=6.0, group_weights=sizes)
    np.testing.assert_allclose(idx.kappa, np.full(3, 6.0))


def test_uniform_weights_match_default():
    """T10: passing explicit ones reproduces the default path exactly."""
    groups = np.repeat(np.arange(4), [2, 5, 1, 7])
    a = GroupIndex(groups, lam=2.0)
    b = GroupIndex(groups, lam=2.0, group_weights=np.ones(4))
    np.testing.assert_allclose(a.kappa, b.kappa)


@pytest.mark.parametrize(
    "labels",
    [
        np.array(["b", "b", "a", "a", "c", "c"]),
        np.array([10, 10, 5, 5, 99, 99]),
        np.array([-3, -3, 0, 0, 7, 7]),
        np.array([0, 0, 1, 1, 2, 2]),
    ],
)
def test_label_types_induce_the_same_partition(labels):
    """T11: any labelling of the same partition behaves identically.

    Codes themselves follow sorted label order, so they are not comparable across
    labelling schemes; the partition and everything user-facing must be.
    """
    from hierarchical_mse import power_loss

    reference = GroupIndex(np.array([0, 0, 1, 1, 2, 2]))
    idx = GroupIndex(labels)
    assert idx.n_groups == reference.n_groups

    same_group = idx.codes[:, None] == idx.codes[None, :]
    expected = reference.codes[:, None] == reference.codes[None, :]
    np.testing.assert_array_equal(same_group, expected)

    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    pred = np.zeros(6)
    assert power_loss(y, pred, idx) == pytest.approx(power_loss(y, pred, reference))


def test_group_mean_and_broadcast_roundtrip():
    idx = GroupIndex(np.repeat([0, 1], [3, 2]))
    v = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    np.testing.assert_allclose(idx.group_mean(v), [2.0, 15.0])
    np.testing.assert_allclose(idx.broadcast(idx.group_mean(v)), [2, 2, 2, 15, 15])


def test_shape_validation():
    idx = GroupIndex(np.repeat([0, 1], 3))
    with pytest.raises(ValueError, match="expected shape"):
        idx.group_mean(np.ones(5))
    with pytest.raises(ValueError, match="1-D"):
        GroupIndex(np.ones((4, 2)))
    with pytest.raises(ValueError, match="non-empty"):
        GroupIndex(np.array([]))


def test_bad_group_weights_rejected():
    groups = np.repeat([0, 1], 3)
    with pytest.raises(ValueError, match="length"):
        GroupIndex(groups, group_weights=np.ones(3))
    with pytest.raises(ValueError, match="non-negative"):
        GroupIndex(groups, group_weights=np.array([-1.0, 1.0]))
