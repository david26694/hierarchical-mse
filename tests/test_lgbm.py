"""T1, T4-T7: the LightGBM adapters, end to end."""

from __future__ import annotations

import numpy as np
import pytest

from hierarchical_mse import (
    GroupIndex,
    init_raw_score,
    lgbm_eval,
    lgbm_objective,
    power_loss,
)

from .conftest import make_grouped_data, split_by_group

lgb = pytest.importorskip("lightgbm")

BASE_PARAMS = {
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "seed": 0,
}


def _train(X, y, idx, *, rounds=30, hessian="bound", learning_rate=0.1, **params):
    """Train with the custom objective, returning (booster, init_score)."""
    init = init_raw_score(y, idx)
    dtrain = lgb.Dataset(X, y, init_score=np.full(len(y), init), free_raw_data=False)
    booster = lgb.train(
        {
            **BASE_PARAMS,
            "objective": lgbm_objective(idx, hessian=hessian),
            "learning_rate": learning_rate,
            **params,
        },
        dtrain,
        num_boost_round=rounds,
    )
    return booster, init


def _predict(booster, init, X, num_iteration=None):
    """Predictions including the init score, which LightGBM does not add back."""
    return init + booster.predict(X, num_iteration=num_iteration)


# --------------------------------------------------------------------------- T1
def test_lam_zero_reproduces_builtin_l2(data):
    """T1: at lam = 0 the objective is bit-for-bit LightGBM's built-in L2.

    This validates the whole harness -- gradient signs, row alignment, init score,
    Dataset plumbing -- against a known-good reference. If anything else in this
    file fails, check that this still passes first.
    """
    X, y = data.X, data.y
    idx = GroupIndex(data.groups, lam=0.0)
    init = float(y.mean())

    common = dict(num_boost_round=25)
    dtrain_custom = lgb.Dataset(X, y, init_score=np.full(len(y), init), free_raw_data=False)
    custom = lgb.train(
        {**BASE_PARAMS, "objective": lgbm_objective(idx), "learning_rate": 0.1},
        dtrain_custom,
        **common,
    )

    dtrain_builtin = lgb.Dataset(X, y, init_score=np.full(len(y), init), free_raw_data=False)
    builtin = lgb.train(
        {**BASE_PARAMS, "objective": "l2", "learning_rate": 0.1, "boost_from_average": False},
        dtrain_builtin,
        **common,
    )

    np.testing.assert_allclose(custom.predict(X), builtin.predict(X), rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- T4
def test_bound_hessian_descends_monotonically(data):
    """T4: at learning_rate=1.0 the bound guarantees the loss never increases."""
    X, y = data.X, data.y
    idx = GroupIndex(data.groups)
    booster, init = _train(X, y, idx, rounds=25, learning_rate=1.0, hessian="bound")

    losses = [power_loss(y, np.full(len(y), init), idx)]
    losses += [
        power_loss(y, _predict(booster, init, X, num_iteration=k), idx)
        for k in range(1, booster.num_trees() + 1)
    ]
    diffs = np.diff(losses)
    assert np.all(diffs <= 1e-8 * np.abs(losses[:-1])), f"increased at {np.argmax(diffs) + 1}"
    assert losses[-1] < losses[0]


# --------------------------------------------------------------------------- T7
def test_exact_diagonal_is_not_monotone(data):
    """T7: the same run with hessian='diag' overshoots. Characterization test.

    Kept loose on purpose -- the point is that the guarantee is absent, not any
    specific magnitude, so it stays valid across LightGBM versions.
    """
    X, y = data.X, data.y
    idx = GroupIndex(data.groups)
    booster, init = _train(X, y, idx, rounds=25, learning_rate=1.0, hessian="diag")

    losses = np.array(
        [power_loss(y, np.full(len(y), init), idx)]
        + [
            power_loss(y, _predict(booster, init, X, num_iteration=k), idx)
            for k in range(1, booster.num_trees() + 1)
        ]
    )
    assert losses.max() > losses[0], "expected the exact diagonal to overshoot at lr=1.0"


# --------------------------------------------------------------------------- T5
def test_row_alignment(data):
    """T5: predictions are invariant to row order when groups are permuted with them."""
    X, y, groups = data.X, data.y, data.groups
    idx = GroupIndex(groups)
    booster, init = _train(X, y, idx, rounds=15)
    base = _predict(booster, init, X)

    perm = np.random.default_rng(0).permutation(len(y))
    idx_perm = GroupIndex(groups[perm])
    booster_p, init_p = _train(X[perm], y[perm], idx_perm, rounds=15)
    permuted = _predict(booster_p, init_p, X[perm])

    np.testing.assert_allclose(base[perm], permuted, rtol=1e-6, atol=1e-6)


def test_scrambled_groups_are_worse(data):
    """T5 negative control: without this, the permutation test could pass vacuously."""
    X, y, groups = data.X, data.y, data.groups
    idx = GroupIndex(groups)
    booster, init = _train(X, y, idx, rounds=30)
    honest = power_loss(y, _predict(booster, init, X), idx)

    scrambled = np.random.default_rng(1).permutation(groups)
    idx_bad = GroupIndex(scrambled, lam=idx.lam)
    booster_b, init_b = _train(X, y, idx_bad, rounds=30)
    # scored against the TRUE grouping -- a model trained on nonsense groups must be worse
    wrong = power_loss(y, _predict(booster_b, init_b, X), idx)

    assert wrong > honest


# --------------------------------------------------------------------------- T6
def test_objective_minimises_its_own_training_loss():
    """T6: the optimizer does what it says -- lower TRAINING loss than l2 at matched capacity.

    This is the defensible half of "does it work". See the companion test below for
    the held-out picture, which is NOT favourable in this regime.
    """
    full = make_grouped_data(seed=7, n_groups=400, nbar=40.0, cv=1.5, s_between=0.15)
    idx = GroupIndex(full.groups)
    capacity = {"num_leaves": 8, "min_data_in_leaf": 100}

    booster, init = _train(full.X, full.y, idx, rounds=200, learning_rate=0.3, **capacity)
    custom = power_loss(full.y, _predict(booster, init, full.X), idx)

    d_l2 = lgb.Dataset(full.X, full.y, free_raw_data=False)
    l2 = lgb.train(
        {**BASE_PARAMS, "objective": "l2", "learning_rate": 0.3, **capacity},
        d_l2,
        num_boost_round=200,
    )
    baseline = power_loss(full.y, l2.predict(full.X), idx)

    assert custom < baseline


def test_between_component_overfits_on_held_out_groups():
    """Characterization of a real limitation -- deliberately asserted, not hidden.

    ``MSE_between`` is supported on only ``B`` groups, so up-weighting it by ``lam``
    concentrates the objective on a small effective sample. With a flexible learner
    (boosting run to convergence) the model drives the *training* group-mean
    residuals toward zero without generalising, and held-out performance degrades
    monotonically in ``lam``.

    Measured on this DGP, best-round held-out loss against a fixed yardstick:

        lam=0 (l2)  3.6846        lam=5.0     3.7112
        lam=0.5     3.6870        lam=20.0    4.0692
        lam=2.0     3.7092        lam=71.0    4.5739

    If a future change makes this test fail, that is good news and worth
    investigating rather than deleting.
    """
    full = make_grouped_data(seed=7, n_groups=400, nbar=40.0, cv=1.5, s_between=0.15)
    train, valid = split_by_group(full, frac=0.6)
    idx_tr = GroupIndex(train.groups)
    idx_va = GroupIndex(valid.groups, lam=idx_tr.lam)

    booster, init = _train(
        train.X, train.y, idx_tr, rounds=300, learning_rate=0.3, num_leaves=8, min_data_in_leaf=100
    )
    train_loss = power_loss(train.y, _predict(booster, init, train.X), idx_tr)
    valid_loss = power_loss(valid.y, _predict(booster, init, valid.X), idx_va)

    assert train_loss < 0.5 * valid_loss, "expected a large train/valid gap on the between term"


# --------------------------------------------------------------------------- adapters
def test_eval_metric_runs_with_valid_sets(data):
    """The feval path works with a mapping and reports the same value as power_loss."""
    train, valid = split_by_group(data, frac=0.7)
    idx_tr = GroupIndex(train.groups)
    idx_va = GroupIndex(valid.groups, lam=idx_tr.lam)
    init = init_raw_score(train.y, idx_tr)

    dtrain = lgb.Dataset(
        train.X, train.y, init_score=np.full(len(train.y), init), free_raw_data=False
    )
    dvalid = lgb.Dataset(
        valid.X,
        valid.y,
        init_score=np.full(len(valid.y), init),
        reference=dtrain,
        free_raw_data=False,
    )

    history: dict = {}
    booster = lgb.train(
        {**BASE_PARAMS, "objective": lgbm_objective(idx_tr), "learning_rate": 0.1},
        dtrain,
        num_boost_round=10,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=lgbm_eval({dtrain: idx_tr, dvalid: idx_va}),
        callbacks=[lgb.record_evaluation(history)],
    )
    reported = history["valid"]["hierarchical_mse"][-1]
    expected = power_loss(valid.y, _predict(booster, init, valid.X), idx_va)
    assert reported == pytest.approx(expected, rel=1e-6)


def test_eval_with_single_index_rejects_wrong_size(data):
    """A single GroupIndex must not silently score a differently-sized eval set."""
    train, valid = split_by_group(data, frac=0.7)
    idx_tr = GroupIndex(train.groups)
    feval = lgbm_eval(idx_tr)
    dvalid = lgb.Dataset(valid.X, valid.y, free_raw_data=False).construct()
    with pytest.raises(ValueError, match="rows"):
        feval(np.zeros(len(valid.y)), dvalid)


def test_objective_rejects_mismatched_index(data):
    idx = GroupIndex(data.groups[:-5])
    objective = lgbm_objective(idx)
    with pytest.raises(ValueError, match="rows"):
        objective(data.y, np.zeros(len(data.y)))


def test_sklearn_api_roundtrip(data):
    """The same objective callable works through LGBMRegressor."""
    idx = GroupIndex(data.groups)
    model = lgb.LGBMRegressor(
        objective=lgbm_objective(idx), n_estimators=10, learning_rate=0.1, **BASE_PARAMS
    )
    model.fit(data.X, data.y, init_score=np.full(len(data.y), init_raw_score(data.y, idx)))
    assert np.isfinite(model.predict(data.X)).all()
