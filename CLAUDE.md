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

**No learner tested has been shown to benefit from `lam > 0` on the test DGP, once each method is tuned
fairly.** Linear and boosted alike. Correctness of the objective is established; usefulness is not.

**Ridge (retracted claim).** An earlier README reported +32% for the loss. It was an artifact of comparing
both methods at the *same* `alpha`. `normalize=True` equalizes total sample weight but **not** effective
regularization -- group-mean rows have much smaller feature variance, so at equal `alpha` the loss-trained
fit is shrunk far harder (36x vs 13x compression of predicted group means). Tuned per method:
MSE 2.9450 / rho 0.9177 versus loss 3.1416 / rho 0.9099. MSE wins raw, calibrated, and on rho. Asserted in
`test_fixed_penalty_comparison_is_misleading`, which pins *both* the tempting fixed-alpha result and its
refutation.

**LightGBM.** Across `lam`, learning rate, rounds, group weights, `nbar`, and seven capacity budgets with
per-method lr tuning, `lam=0` was never beaten. Per-level recalibration does not rescue it -- the boosted
models come out already calibrated (`theta_m ~ 1.0`) with *lower* `rho_between`. Asserted in
`test_between_component_overfits_on_held_out_groups`.

**Per-level recalibration: measured, and it does not matter.** An earlier version of this file claimed
loss-trained predictors are "miscalibrated by construction". That was overstated. The dramatic
miscalibration (between-group slope 32.6 vs 9.4) appears only at absurd regularization strengths; at each
method's tuned alpha both models sit at `theta_m ~ 1.0`, so recalibration is nearly a no-op. It changes no
verdict on either DGP -- hard DGP loss wins +8.6% raw and +11.2% recalibrated; easy DGP MSE wins either way.
Reported by `benchmarks/compare.py`. Like the retracted Ridge claim, the miscalibration story was a *symptom*
of the fixed-alpha comparison, not an independent finding.

### Rules for any future benchmark here

1. **Tune the regularization strength separately for each method.** Comparing at a shared hyperparameter is
   what produced the retracted result.
2. **Split holdouts by group**, never by row.
3. **Report raw and recalibrated**, not whichever is favourable.
4. Do not weaken a failing negative-result test. If one starts failing, that is a finding -- investigate and
   write it up.

**Open work.** The DGP's group-level signal sits in two clean group-constant features, so plain MSE already
reaches the between-group ceiling and there is nothing to reallocate. A DGP where the group signal is
genuinely hard to extract (a high-order interaction of group-level features) is untested.

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
