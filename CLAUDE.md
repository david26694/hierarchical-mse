# CLAUDE.md

Context for future sessions. Read this before changing anything in `src/`.

## What this is

One loss and its derivatives, nothing else:

```
L(g) = MSE_total(g) + lam * MSE_between(g)
```

over groups `b` with `n_b` rows. `lam = 0` is ordinary MSE. Provenance for the default
`lam = nbar * (1 + cv2)` is [arXiv:2607.27376](https://arxiv.org/abs/2607.27376), which derives it as the
variance-optimal weight for cluster-randomized experiments.

## The math (do not re-derive)

Gradients and Hessians are for the loss **rescaled by `N/2`**. With
`kappa_b = lam * N * omega_b / (n_b * sum(omega))`:

| quantity | formula |
|---|---|
| gradient | `grad_i = -e_i - kappa_b * ebar_b` |
| exact Hessian diagonal | `hess_i = 1 + kappa_b / n_b` |
| **majorizing bound (default)** | `hess_i = 1 + kappa_b` |
| optimal init raw score | `c* = (mean(y) + lam * wmean_b(ybar_b)) / (1 + lam)` |

`omega = 1` gives `kappa_b = lam * nbar / n_b`; `omega = n_b` gives `kappa_b = lam` (constant).

The true per-group Hessian block is `H_b = I + (kappa_b/n_b) * J`, eigenvalues `{1 (x n_b-1), 1+kappa_b}`.

## Invariants — do not break silently

1. **The `N/2` rescale.** Un-rescaled hessians are `O(1/N)` and LightGBM's default `min_child_weight=1e-3`
   would then refuse to split. Changing the scale means revisiting that.
2. **`hessian="bound"` stays the default.** The exact diagonal does not majorize `H`, so leaf values overshoot
   by up to `n_b` on group-pure leaves — precisely the splits the loss exists to encourage. See
   `hessian-bound.md`. `reg_lambda` semantics also shift with the bound; do not silently rescale it.
3. **lightgbm is an optional extra.** `groups.py` and `loss.py` must stay pure numpy. `lgbm.py` imports
   lightgbm inside functions only, so `import hierarchical_mse` works without it.
4. **`lam` must stay explicitly settable.** It is a design constant in the experimental setting and a
   hyperparameter elsewhere; never force inference from the training groups.
5. **No experiment vocabulary in the API or docstrings.** This library is about *groups*. The source paper is
   provenance, mentioned in the README and here — nowhere else. Grouping is the caller's business.

## First thing to check when something looks wrong

**`lam = 0` must reproduce LightGBM's built-in `l2` bit-for-bit** (`grad = pred - y`, `hess = 1`).
`test_lam_zero_reproduces_builtin_l2` asserts this end to end. If it fails, the problem is plumbing — row
alignment, gradient sign, init score — not the loss.

## What is and is not demonstrated

**Linear models: the loss works.** With `Ridge` under a binding penalty, `augment()` raises held-out
`rho_between` (0.792 -> 0.866 at `alpha=3e5`) and, after per-level recalibration, cuts the loss by 32%.
Asserted in `tests/test_linear.py`.

**Critically, loss-trained predictors are miscalibrated by construction.** The loss reallocates capacity
toward group means, driving the between-group slope of `y` on `g` away from 1 (measured `theta_m = 32.6` vs
9.4 for MSE). Raw squared error punishes that scale error, so on raw metrics the loss looks *worse* even
when it carries more information. Recalibrating per level is what converts the gain. Anyone evaluating this
library without that step will wrongly conclude it does nothing.

**LightGBM: no benefit found.** Across `lam`, learning rate, rounds, group weights, `nbar`, and seven
capacity budgets with per-method lr tuning, `lam=0` was never beaten. Recalibration does not rescue it --
the boosted models come out already calibrated (`theta_m ~ 1.0`) with *lower* `rho_between`, and the
post-calibration optimum depends on the predictor only through its correlations. Mechanism appears to be
overfitting: `MSE_between` is supported on only `B` groups and a converged ensemble memorizes training group
means (0.13 train vs 5.86 valid). Asserted in `test_between_component_overfits_on_held_out_groups`.

Do not "fix" the LightGBM result by weakening tests or softening the README. If a change makes the
overfitting test fail, that is a real result worth investigating and writing up.

**Open work.** The negative LightGBM result is measured on one DGP whose group-level signal sits in two
clean group-constant features -- plain MSE already reaches the between-group ceiling there, so nothing needs
reallocating. A DGP where the group signal is genuinely hard to extract (a high-order interaction of
group-level features, needing many splits) has not been tried and might behave differently.

## Commands

```
uv sync --all-extras --group dev
uv run pytest -q
uv run pytest -q -k "not lgbm"     # must also pass without the extra
uv run ruff check --fix . && uv run ruff format .
uv build
```

## Scope boundary

Out of scope: covariate pipelines, experiment analysis, per-level adjustment coefficients, and the
row-augmentation trick (which implements this same loss exactly for models linear in the features, since
`g(Xbar_b) = gbar_b` there — an identity that fails for trees, which is why this package exists).
