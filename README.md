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

### Linear models — exact, and no LightGBM needed

For any model **linear in the features** the loss reduces to plain weighted least squares on one extra row
per group, because `g(X̄_b) = ḡ_b`. So it trains exactly with any sklearn estimator that takes
`sample_weight`:

```python
from sklearn.linear_model import ElasticNet, Ridge
from hierarchical_mse import GroupIndex, augment

idx = GroupIndex(groups, lam=50.0)
model = Ridge(alpha=1e4).fit(*augment(X, y, idx))  # exact, verified to 2e-16
model = ElasticNet(alpha=0.01).fit(*augment(X, y, idx))
```

This identity is *exact* — the augmented weighted SSE and the loss differ by a single constant factor, and
the fit matches a direct numerical minimizer to 1e-8. It fails for trees (`g(mean(x)) != mean(g(x))`), which
is why the LightGBM path needs a custom gradient.

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

**It depends on the learner, and it depends on whether the group signal is expensive to extract.** Both
findings below are on synthetic data with held-out **groups**, each method tuned to its own best
regularization strength. Every number in this section is reproduced by:

```
uv run python benchmarks/compare.py
```

### The deciding factor: is there anything to reallocate?

The loss reallocates capacity from within-group fit toward between-group fit. That can only help if
recovering the group signal actually *costs* capacity. Two DGPs, identical except for how the group signal
is carried:

| group signal carried by | can the loss help? |
|---|---|
| 2 strong, clean group-level features | **No.** Plain MSE reaches the between-group ceiling for free; there is nothing to reallocate, and no setting of λ helps any learner. |
| 20 weak group-level features | **Yes, for linear models.** Recovering the signal costs real coefficient budget, which must compete with numerous strong row-level features. |

Benchmarking only on the first DGP produced two successive wrong conclusions in this repo's history. If you
evaluate this loss, check first whether plain MSE is already at the ceiling — if it is, the loss cannot help
and the benchmark is uninformative.

### Linear models: it works, when capacity binds

On the second DGP (`make_hard_grouped_data` in the tests), Ridge with each method at its own best `alpha`,
validated on an entirely independent draw of groups:

| | MSE-trained | loss-trained |
|---|---|---|
| held-out loss | 18.26 (alpha=3000) | **16.70** (alpha=3000) — **8.6% better** |
| ρ_between | 0.7373 | **0.7656** |

Both methods select the same `alpha`, and the loss has higher ρ_between at all 12 alphas tested. No
recalibration is needed for this win. Asserted in `test_loss_beats_mse_when_the_group_signal_is_expensive`.

### LightGBM: it does not transfer

On that **same** DGP, with the same λ and per-method learning-rate tuning:

| learner | MSE-trained | loss-trained |
|---|---|---|
| Ridge | 18.26 / ρ 0.737 | **16.70 / ρ 0.766** |
| LightGBM (`hessian="bound"`) | **19.18 / ρ 0.722** | 22.72 / ρ 0.662 |
| LightGBM (`hessian="diag"`) | — | 35.28 / ρ 0.559 |

Same data, same objective, opposite outcome — so this is specific to the tree learner, not the loss.

The likely mechanism is memorization. The group-level features are **constant within group**, so a tree can
split finely on them and identify individual training groups; the λ-weighted between term then rewards
memorizing their means (measured elsewhere: 0.13 train vs 5.86 valid). A linear model is structurally immune,
being unable to isolate individual groups regardless of weighting. This is a plausible mechanism consistent
with the measurements, not something proven.

Note also that `diag` is far worse than `bound`, which rules out the Hessian bound's damping as the cause:
relaxing the damping makes things dramatically worse, not better.

### Practical guidance

- **Linear models with real regularization** are the demonstrated use case.
- **Boosted trees: don't.** No configuration tested beat `λ=0`.
- **Before adopting anything**, benchmark against `λ=0` on a group-wise holdout with the regularization
  strength tuned *separately per method*, and check that plain MSE is not already at the between-group
  ceiling.
- **The default `λ = n̄(1+cv²)` is aggressive** for general use. It is the variance-optimal value for a
  specific experimental-design problem (below), not a general prior.

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
