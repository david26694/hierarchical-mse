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
model = Ridge(alpha=1e4).fit(*augment(X, y, idx))       # exact, verified to 2e-16
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

### Linear models: it works, but you must recalibrate

With `Ridge` and a tight penalty (so capacity genuinely binds), the loss raises between-group predictive
quality substantially:

| Ridge alpha | ρ_between (MSE) | ρ_between (loss) | loss after recalibration |
|---|---|---|---|
| 1 000 | 0.9177 | 0.9099 | −8.5% (MSE wins) |
| 30 000 | 0.8645 | **0.8768** | **+8.8%** |
| 100 000 | 0.8163 | **0.8688** | **+26.2%** |
| 300 000 | 0.7921 | **0.8659** | **+32.4%** |

Note the two columns disagree at first: on *raw* squared error the loss looks worse everywhere. That is not a
contradiction — it is the point. The loss deliberately reallocates capacity toward group means, which drives
the between-group regression slope of `y` on `g` away from 1 (measured: **θ_m = 32.6** at alpha=3e5, versus
9.4 for MSE). Raw squared error punishes that scale error even though the prediction carries *more
information* about group means.

So a loss-trained predictor is **miscalibrated by construction**. Fit a slope per level and the extra
information converts into a 32% improvement:

```python
gb = idx.group_mean(g); gw = g - gb[idx.codes]      # within / between components
theta_w = (gw @ yw) / (gw @ gw)                      # fit on training data
theta_m = (gc @ yc) / (gc @ gc)                      # gc, yc = centred group means
calibrated = y.mean() + theta_w * gw + theta_m * (gb[idx.codes] - gb.mean())
```

Skip this step and the loss will look useless. Both effects are asserted in `tests/test_linear.py`.

### LightGBM: no benefit found

Across λ (0 → 71), learning rate, rounds, group weights, `n̄`, and seven capacity budgets with per-method
learning-rate tuning, **`λ=0` was never beaten**. At the default λ, held-out loss degrades from 3.68 to 4.57.
Recalibration does not rescue it: the LightGBM models come out already calibrated (θ_m ≈ 1.0) with *lower*
ρ_between (0.858 vs 0.892), and after optimal per-level rescaling the achievable loss depends on the
predictor only through its correlations — so a lower ρ cannot be recovered.

The mechanism appears to be overfitting: `MSE_between` is supported on only `B` groups, and a converged
boosted ensemble drives the *training* group-mean residuals to near zero (0.13 train vs 5.86 valid). This is
asserted in `test_between_component_overfits_on_held_out_groups` rather than hidden.

**This is a negative result on one DGP, not a proof.** In this data the group-level signal sits in two clean
group-constant features, so plain MSE already reaches the between-group ceiling and there is nothing to
reallocate. A DGP where the group signal is genuinely *hard* to extract might behave differently; that is
untested.

### Practical guidance

- **Linear + real regularization** is the demonstrated use case. Recalibrate per level afterwards.
- **Boosted trees:** validate against `λ=0` on a group-wise holdout before believing any gain. Do not assume
  it transfers.
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
