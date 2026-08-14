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

**Read this before adopting it.** Everything here is measured on a synthetic two-level DGP (400 groups,
`n̄=40`, `cv=1.5`, 12 weak within-group features + 2 group-level features), with held-out **groups** and a
fixed λ yardstick. The headline: **it works for linear models under a binding penalty, and did not help
LightGBM in any regime tested.**

### Linear models: no benefit either, once each method is tuned

An earlier version of this README claimed a +32% improvement for Ridge. **That was an artifact and has been
retracted.** It came from comparing both methods at the *same* `alpha`, deep in an over-regularized regime.
`normalize=True` equalizes total sample weight but **not** effective regularization: the group-mean rows have
much smaller feature variance (within-group noise averages out), so at equal `alpha` the loss-trained fit is
shrunk far harder — 36x compression of predicted group means versus 13x for MSE. The apparent win was "the
loss degrades more gracefully when both models are wrecked," not "the loss is better."

Tuning `alpha` per method on a group-wise holdout removes the confound:

| | MSE-trained | loss-trained |
|---|---|---|
| best raw held-out loss | **2.9450** (alpha=100) | 3.1416 (alpha=100) |
| best after per-level calibration | **2.9510** (alpha=1000) | 3.2029 (alpha=1000) |
| best rho_between | **0.9177** | 0.9099 |

MSE wins on every metric. Note the last row in particular: properly tuned MSE reaches a *higher* between-group
correlation than the loss attains at any `alpha`, so this is not a scale problem that recalibration could fix.

What remains true, and is still asserted in `tests/test_linear.py`: at a **fixed** heavy penalty the
loss-trained model does retain more between-group information (rho 0.866 vs 0.792 at alpha=3e5), and it is
badly miscalibrated by construction (between-group slope 32.6 vs 9.4). Both are real. Neither amounts to a
benefit once each method is allowed its own regularization strength.

### Summary of the evidence

On this DGP, **the loss has not been shown to beat plain MSE for any learner tested**, linear or boosted,
once each method is tuned fairly. The implementation is verified correct -- `lam=0` reproduces LightGBM's
built-in `l2` bit-for-bit, and the linear augmentation matches a direct minimizer to 1e-8 -- but correctness
of the objective is not evidence that the objective is useful here.

This is one synthetic DGP, and its group-level signal sits in two clean group-constant features, so plain MSE
already reaches the between-group ceiling and there is nothing to reallocate. A DGP where the group signal is
genuinely hard to extract remains untested and might behave differently.

### Practical guidance

- **Benchmark against `λ=0` before adopting anything**, on a group-wise holdout, with the regularization
  strength tuned *separately for each method*. Comparing at a shared hyperparameter is what produced the
  retracted result above.
- **Recalibrate per level** before comparing, since the loss produces miscalibrated predictors by
  construction -- but do not expect recalibration alone to create a benefit.
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
