"""Shared fixtures: a hierarchical data-generating process with a capacity trade-off."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class GroupedData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]


def make_grouped_data(
    seed: int,
    n_groups: int = 300,
    nbar: float = 40.0,
    cv: float = 1.5,
    s_between: float = 0.25,
    n_within_features: int = 12,
    n_between_features: int = 2,
) -> GroupedData:
    """Two-level DGP: ``y = sqrt(s_between) * M_b + sqrt(1 - s_between) * eps_bi``.

    The feature set is deliberately **many weak within-group predictors plus a few
    group-level ones**. That structure matters: with only a couple of near-orthogonal
    covariates a regularizer shrinks each coefficient independently and never has to
    *allocate* capacity, so no within-versus-between trade-off binds and every loss
    reaches the same between-group fit. The many-weak-features case is where the loss
    earns its keep, and is also the common shape in practice.
    """
    rng = np.random.default_rng(seed)

    sigma = np.sqrt(np.log(1.0 + cv**2))
    rates = np.exp(rng.normal(np.log(nbar) - sigma**2 / 2, sigma, n_groups))
    sizes = np.maximum(rng.poisson(rates), 2)

    groups = np.repeat(np.arange(n_groups), sizes)
    n = groups.size

    between_signal = rng.normal(size=n_groups)
    between_signal /= between_signal.std()
    within_signal = rng.normal(size=n)

    y = np.sqrt(s_between) * between_signal[groups] + np.sqrt(1.0 - s_between) * within_signal

    columns, names = [], []
    for j in range(n_within_features):
        columns.append(0.7 * within_signal + 0.7 * rng.normal(size=n))
        names.append(f"within_{j}")
    for j in range(n_between_features):
        # group-persistent noise, so the group-level features are informative but imperfect
        columns.append(0.8 * between_signal[groups] + 0.6 * rng.normal(size=n_groups)[groups])
        names.append(f"between_{j}")

    return GroupedData(X=np.column_stack(columns), y=y, groups=groups, feature_names=names)


def split_by_group(data: GroupedData, frac: float = 0.5) -> tuple[GroupedData, GroupedData]:
    """Split into two disjoint sets of whole groups (never split a group across sets)."""
    unique = np.unique(data.groups)
    cut = int(len(unique) * frac)
    left_ids, right_ids = set(unique[:cut].tolist()), set(unique[cut:].tolist())
    left = np.array([g in left_ids for g in data.groups])
    right = np.array([g in right_ids for g in data.groups])

    def subset(mask: np.ndarray) -> GroupedData:
        return GroupedData(
            X=data.X[mask],
            y=data.y[mask],
            groups=data.groups[mask],
            feature_names=data.feature_names,
        )

    return subset(left), subset(right)


@pytest.fixture
def small_data() -> GroupedData:
    """Small enough for dense linear algebra and finite differences."""
    return make_grouped_data(seed=0, n_groups=20, nbar=10.0, cv=1.0)


@pytest.fixture
def data() -> GroupedData:
    """Default-size dataset for training tests."""
    return make_grouped_data(seed=1)
