# hierarchical-mse

A regression loss for **grouped data where between-group accuracy matters more than per-row accuracy** —
with its gradient, Hessian, and a LightGBM objective.

```
pip install hierarchical-mse[lightgbm]
```

## The problem

Ordinary MSE weights every row equally. But when rows belong to groups and what you actually care about is
how well the model predicts **group aggregates**, plain MSE allocates capacity in the wrong place: most of
the squared-error mass lives in within-group noise, so a capacity- or regularization-constrained model chases
that noise and under-fits the group-level signal. The more rows per group, the worse the mismatch.

Concretely — with a fixed regularization budget, an L1/L2 penalty shrinks group-level features to zero long
before row-level ones, and a tree's split criterion prefers a row-level split reducing squared error across
`N` rows over a group-level split that only moves `B` group means.

## The loss

$$\mathcal{L}(g) \;=\; \underbrace{\frac1N\sum_{b,i}\big(y_{b,i}-g_{b,i}\big)^2}_{\text{MSE}_{\text{total}}} \;+\; \lambda\underbrace{\frac1B\sum_b\big(\bar y_b-\bar g_b\big)^2}_{\text{MSE}_{\text{between}}}$$

Groups `b = 1..B`, `n_b` rows in group `b`. `λ = 0` is ordinary MSE; larger `λ` buys group-aggregate accuracy
at the cost of per-row accuracy. Because the within/between decomposition is orthogonal, `λ` is exactly the
extra weight placed on the between-group component.

## Usage

```python
import lightgbm as lgb
from hierarchical_mse import GroupIndex, init_raw_score, lgbm_eval, lgbm_objective

idx = GroupIndex(groups_train, lam=50.0)  # groups_train: one label per row
idx_valid = GroupIndex(groups_valid, lam=idx.lam)  # same lam for a comparable metric

dtrain = lgb.Dataset(
    X_train, y_train, init_score=np.full(len(y_train), init_raw_score(y_train, idx))
)
dvalid = lgb.Dataset(
    X_valid, y_valid, init_score=np.full(len(y_valid), init_raw_score(y_train, idx))
)

booster = lgb.train(
    {"objective": lgbm_objective(idx), "learning_rate": 0.1, "verbose": -1},
    dtrain,
    num_boost_round=300,
    valid_sets=[dvalid],
    feval=lgbm_eval({dtrain: idx, dvalid: idx_valid}),
    callbacks=[lgb.early_stopping(30)],
)
```

Without LightGBM, the loss is usable directly:

```python
from hierarchical_mse import GroupIndex, grad_hess, power_loss, power_loss_parts, rho_between

idx = GroupIndex(groups)
power_loss(y, pred, idx)  # the value
power_loss_parts(y, pred, idx)  # within / between decomposition
grad_hess(y, pred, idx)  # (grad, hess) of the N/2-rescaled loss
rho_between(y, pred, idx)  # between-group correlation of y and pred
```

## When this helps — and when it doesn't

**Read this before adopting it.** The loss, its gradient and Hessian, and the LightGBM integration are
verified correct (59 tests, including that `λ=0` reproduces LightGBM's built-in `l2` bit-for-bit). Whether
raising `λ` *helps* is a separate question, and the honest answer is: **it depends on whether model capacity
binds, and with a flexible learner it can actively hurt.**

Measured on a synthetic two-level DGP (400 groups, `n̄=40`, `cv=1.5`, boosting to convergence with
group-wise holdout and best-round selection), held-out loss against a fixed yardstick:

| training λ | held-out loss | ρ_between |
|---|---|---|
| **0 (plain l2)** | **3.6846** | **0.8943** |
| 0.5 | 3.6870 | 0.8946 |
| 2.0 | 3.7092 | 0.8938 |
| 5.0 | 3.7112 | 0.8940 |
| 20.0 | 4.0692 | 0.8849 |
| 71.0 (the default) | 4.5739 | 0.8711 |

λ monotonically *hurts* here. The mechanism is overfitting: `MSE_between` is supported on only `B` groups, so
up-weighting it by λ concentrates the objective on a small effective sample, and a boosted ensemble run to
convergence drives the *training* group-mean residuals to near zero (0.13 train vs 5.86 valid) without
generalizing. This is asserted as a test (`test_between_component_overfits_on_held_out_groups`) rather than
hidden.

The loss is designed for learners where **capacity genuinely binds** — the source paper demonstrates it with
Ridge under a fixed penalty, where gains grow from ~0% to ~40% as the penalty tightens. Boosting with enough
rounds is not capacity-limited, which removes the very trade-off the loss exists to rebalance.

So:

- **Try it** when your learner is genuinely constrained (few rounds, strong regularization, small linear
  models), when `B` is large, and when group means are well-estimated (large `n_b`).
- **Be skeptical** with high-capacity models trained to convergence. Validate against `λ=0` on a group-wise
  holdout before believing any gain; treat λ as a hyperparameter to tune downward, not a constant to adopt.
- **The default `λ = n̄(1+cv²)` is aggressive** for general use. It is the variance-optimal value for a
  specific experimental-design problem (below), not a good general prior.

## Three things that will bite you

**1. `init_score` is mandatory.** LightGBM does not apply `boost_from_average` to custom objectives, so the
initial raw score is 0. Pass `init_raw_score(y, idx)`. For large `λ` it is close to the mean of the *group
means*, not the grand mean of `y` — materially different when group sizes are dispersed.

**2. Early stopping must watch this loss, not `l2`.** Otherwise the stopping rule optimizes the objective
you just replaced, and cuts training exactly when the between-group fit starts paying off. That's what
`lgbm_eval` is for.

**3. Split holdouts by group, never by row.** `MSE_between` is supported on `B` groups rather than `N` rows,
so up-weighting it by `λ` concentrates the effective objective on a much smaller sample. A random row split
both leaks (a group straddles folds) and reports between-group gains that don't transfer.

## Choosing λ

`λ` is a knob: it is the weight on the between-group term, and `λ=0` recovers MSE.

If you leave `lam=None`, the default is `λ = n̄(1 + cv²)` where `n̄` is the mean group size and `cv²` the
squared coefficient of variation of group sizes. That value is not arbitrary — it is the variance-optimal
weight derived in [arXiv:2607.27376](https://arxiv.org/abs/2607.27376) for cluster-randomized experiments,
where the groups are randomization units and this loss trains a control variate. In that setting `λ` is a
**design constant, not a hyperparameter**, and should be computed from the group-size distribution of the
data you ultimately care about — which may differ from your training data. Pass it explicitly if so.

Outside that setting, treat the default as a reasonable starting scale and tune it.

## The Hessian

`grad_hess` defaults to `hessian="bound"`, returning `1 + κ_b` rather than the true diagonal
`1 + κ_b/n_b`. This is deliberate and load-bearing: LightGBM's leaf value assumes a diagonal Hessian, but
this loss is not separable, and its off-diagonal mass sits entirely in the "shift the whole group" direction
— where the exact diagonal understates curvature by up to a factor `n_b`, causing leaf values to overshoot by
the same factor on precisely the group-separating splits the loss exists to encourage.

The bound is the block's top eigenvalue, so it majorizes, guaranteeing monotone descent at
`learning_rate=1.0`, and it is *exact* for leaves holding a whole group. See
[hessian-bound.md](hessian-bound.md) for the full argument and measurements.

## License

MIT
