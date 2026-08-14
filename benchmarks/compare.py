"""Reproduce every empirical claim in the README.

    uv run python benchmarks/compare.py

Two data-generating processes, identical except for how the group-level signal is
carried, and two learners. The point of the comparison is that the *DGP* decides the
answer, not the loss:

* ``make_grouped_data``      -- 2 strong clean group-level features. Plain MSE already
  reaches the between-group ceiling, so nothing can help.
* ``make_hard_grouped_data`` -- 20 weak group-level features. Recovering the group
  signal costs real capacity, so a within-versus-between trade-off binds.

Every comparison tunes the regularization strength (Ridge ``alpha``, LightGBM
``learning_rate``) **separately for each method**, and validates on an entirely
independent draw of groups. Benchmarking without those two controls is what produced
two successive wrong conclusions in this repo's history.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hierarchical_mse import (  # noqa: E402
    GroupIndex,
    augment,
    init_raw_score,
    lgbm_objective,
    power_loss,
    rho_between,
)
from tests.conftest import make_grouped_data, make_hard_grouped_data  # noqa: E402

RIDGE_ALPHAS = [1e0, 1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5]
LGB_RATES = [0.05, 0.3, 1.0]
LGB_BASE = {
    "verbose": -1,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "seed": 0,
}


def ridge_best(train, valid, idx_tr, idx_va, use_loss):
    """Best held-out (loss, rho, alpha) over the alpha grid."""
    augmented = augment(train.X, train.y, idx_tr) if use_loss else None
    best = None
    for alpha in RIDGE_ALPHAS:
        fit = (
            Ridge(alpha=alpha).fit(*augmented)
            if use_loss
            else Ridge(alpha=alpha).fit(train.X, train.y)
        )
        pred = fit.predict(valid.X)
        cand = (power_loss(valid.y, pred, idx_va), rho_between(valid.y, pred, idx_va), alpha)
        if best is None or cand[0] < best[0]:
            best = cand
    return best


def lgbm_best(train, valid, idx_tr, idx_va, objective, hessian="bound", rounds=600):
    import lightgbm as lgb

    best = None
    for lr in LGB_RATES if objective == "loss" else [0.05, 0.1]:
        if objective == "l2":
            dtrain = lgb.Dataset(train.X, train.y, free_raw_data=False)
            booster = lgb.train(
                {
                    **LGB_BASE,
                    "objective": "l2",
                    "learning_rate": lr,
                    "num_leaves": 8,
                    "min_data_in_leaf": 100,
                },
                dtrain,
                num_boost_round=rounds,
            )

            def predict(k, b=booster):
                return b.predict(valid.X, num_iteration=k)
        else:
            init = init_raw_score(train.y, idx_tr)
            dtrain = lgb.Dataset(
                train.X, train.y, init_score=np.full(len(train.y), init), free_raw_data=False
            )
            booster = lgb.train(
                {
                    **LGB_BASE,
                    "objective": lgbm_objective(idx_tr, hessian=hessian),
                    "learning_rate": lr,
                    "num_leaves": 8,
                    "min_data_in_leaf": 100,
                },
                dtrain,
                num_boost_round=rounds,
            )

            def predict(k, b=booster, i=init):
                return i + b.predict(valid.X, num_iteration=k)

        for k in range(10, rounds + 1, 10):
            pred = predict(k)
            cand = (power_loss(valid.y, pred, idx_va), rho_between(valid.y, pred, idx_va), lr)
            if best is None or cand[0] < best[0]:
                best = cand
    return best


def run(name, train, valid):
    idx_tr = GroupIndex(train.groups)
    idx_va = GroupIndex(valid.groups, lam=idx_tr.lam)
    print(f"\n{'=' * 78}\n{name}")
    print(
        f"  train n={idx_tr.n} groups={idx_tr.n_groups} lam={idx_tr.lam:.1f}, "
        f"{train.X.shape[1]} features; validation = independent groups"
    )
    print(f"{'=' * 78}")
    print(f"{'learner':>28} | {'held-out loss':>13} {'rho_between':>12} {'tuned':>8}")

    for tag, use_loss in (("Ridge, MSE-trained", False), ("Ridge, loss-trained", True)):
        loss, rho, hp = ridge_best(train, valid, idx_tr, idx_va, use_loss)
        print(f"{tag:>28} | {loss:>13.4f} {rho:>12.4f} {f'a={hp:g}':>8}")

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print(f"{'(lightgbm not installed)':>28} |")
        return
    for tag, obj, hess in (
        ("LightGBM, l2", "l2", None),
        ("LightGBM, loss (bound)", "loss", "bound"),
        ("LightGBM, loss (diag)", "loss", "diag"),
    ):
        loss, rho, hp = lgbm_best(train, valid, idx_tr, idx_va, obj, hessian=hess or "bound")
        print(f"{tag:>28} | {loss:>13.4f} {rho:>12.4f} {f'lr={hp:g}':>8}")


if __name__ == "__main__":
    run(
        "EASY DGP -- group signal in 2 strong clean features (nothing to reallocate)",
        make_grouped_data(seed=0),
        make_grouped_data(seed=1),
    )
    run(
        "HARD DGP -- group signal in 20 weak features (capacity genuinely binds)",
        make_hard_grouped_data(seed=0),
        make_hard_grouped_data(seed=1),
    )
    print("\nExpected: on the EASY DGP nothing beats MSE/l2. On the HARD DGP the loss wins")
    print("for Ridge and loses for LightGBM -- same data, same objective, opposite outcome.")
