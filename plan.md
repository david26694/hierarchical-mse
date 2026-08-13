# hierarchical-mse — build plan

## Context

A regression loss for **grouped data where between-group accuracy matters more than per-row accuracy**.

Ordinary MSE treats every row equally. But when rows belong to groups and what you actually care about is how
well the model predicts **group aggregates**, plain MSE spends the model's capacity in the wrong place: most
of the squared-error mass lives in within-group noise, so a capacity- or regularization-constrained model
chases that noise and under-fits the group-level signal. The more rows per group, the worse the mismatch.

This library packages a loss that fixes the allocation, plus its gradient, Hessian, and a ready-to-use
LightGBM objective:

$$\mathcal{L}(g) \;=\; \underbrace{\frac1N\sum_{b,i}\big(y_{b,i}-g_{b,i}\big)^2}_{\text{MSE}_{\text{total}}} \;+\; \lambda\underbrace{\frac1B\sum_b\big(\bar y_b-\bar g_b\big)^2}_{\text{MSE}_{\text{between}}}$$

Groups `b = 1..B`, `n_b` rows in group `b`, `N = Σn_b`, `n̄ = N/B`, `cv²` the squared coefficient of variation
of group sizes. `ȳ_b`, `ḡ_b` are group means. `λ ≥ 0` sets how much extra weight the between-group error
carries; `λ = 0` is ordinary MSE.

**Origin and the default λ.** The loss comes from *"Power-Optimal Covariate Adjustment for Switchback
Experiments"* (Pankratev, [arXiv:2607.27376](https://arxiv.org/abs/2607.27376)), where the groups are
randomization units and the variance of the treatment-effect estimator is shown to be an affine functional of
`MSE_within` and `MSE_between` with design-determined weights. That derivation pins the optimal value at

$$\lambda_{\text{default}} \;=\; \bar n\,(1+cv^2)$$

which is what the library defaults to. For that use case λ is a **design constant read off the group-size
distribution, not a hyperparameter**. For general grouped regression it is simply a knob, and callers should
be free to set it. The API must support both readings.

Why this needs to be a library rather than a snippet: the loss is **not separable across rows**, so it cannot
be expressed with `sample_weight` on any standard learner, and the naive LightGBM implementation is subtly
wrong in a way that yields a plausible-but-worse model. See [Design decisions](#design-decisions).

**Decisions taken:** package `hierarchical_mse` (name verified free on PyPI — `/simple/hierarchical-mse/`
returns 404); lightgbm as an **optional extra** so the core loss stays pure-numpy; MIT, aimed at public PyPI;
ruff + pytest + GitHub Actions.

---

## The math (this is the spec)

Residual `e = y − g`, group-mean residual `ē_b`.

**Orthogonal decomposition** (the reason the loss is well-posed):

$$\text{MSE}_{\text{total}} \;=\; \text{MSE}_{\text{within}} + \text{MSE}_{\text{between}}, \qquad \text{MSE}_{\text{within}} = \mathbb{E}\big[(e_{b,i}-\bar e_b)^2\big], \quad \text{MSE}_{\text{between}} = \mathbb{E}\big[\bar e_b^{\,2}\big]$$

so `L = MSE_within + (1+λ)·MSE_between` — the loss is a re-weighting of two orthogonal components, and λ is
exactly the extra weight placed on the between-group one.

**Rescale by `N/2`** for the gradient/Hessian. Monotone rescaling, same minimizer, but it puts the Hessian at
`O(1)` — load-bearing, see D3. With `κ_b := λ n̄ / n_b`:

| quantity | formula |
|---|---|
| gradient | $\;\text{grad}_i = -e_i - \kappa_b\,\bar e_b$ |
| exact Hessian diagonal | $\;\text{hess}_i = 1 + \kappa_b/n_b$ |
| **majorizing Hessian (default)** | $\;\text{hess}_i = 1 + \kappa_b$ |
| optimal init raw score | $\;c^\star = \dfrac{\overline{y} + \lambda\,\text{mean}_b(\bar y_b)}{1+\lambda}$ |

The gradient is the whole method in one line: the ordinary residual with the **group-mean residual added
back, inflated by `κ_b ≈ λ`**. Split gains are computed from `Σ grad` within a candidate leaf, so any split
separating groups sees a signal ~λ times larger than under plain MSE.

**Generalized group weights.** Allow per-group weights `ω_b` so the between term becomes
`λ·Σω_b ē_b² / Σω_b`, giving

$$\kappa_b \;=\; \frac{\lambda\,N\,\omega_b}{n_b\,\sum_c \omega_c}$$

`ω_b = 1` recovers the above. `ω_b ∝ n_b` is inverse-variance weighting of `ȳ_b`, which matters because
`Var(ȳ_b) = σ²_between + σ²_within/n_b` — small groups have noisy targets, and the unweighted form
over-trusts them.

**Sanity anchor.** At λ=0 the formulas collapse to `grad = pred − y`, `hess = 1` — *exactly* LightGBM's
built-in L2. This is the single most valuable test in the suite (T1 below).

---

## Repo layout

```
hierarchical-mse/
├── .github/workflows/ci.yml
├── .gitignore
├── .python-version
├── CLAUDE.md
├── LICENSE                     # MIT
├── README.md
├── plan.md                     # this file
├── pyproject.toml
├── ruff.toml
├── uv.lock
├── src/hierarchical_mse/
│   ├── __init__.py             # the entire public API, re-exported
│   ├── groups.py               # GroupIndex: codes, sizes, nbar, cv2, lam, kappa
│   ├── loss.py                 # power_loss, power_loss_parts, grad_hess, init_raw_score (pure numpy)
│   └── lgbm.py                 # LightGBM objective + eval factories (imports lightgbm lazily)
└── tests/
    ├── conftest.py             # grouped-data DGP fixture
    ├── test_groups.py
    ├── test_loss.py
    ├── test_grad_hess.py
    └── test_lgbm.py            # skipped unless lightgbm installed
```

Src layout (not flat) so tests run against the installed package and can't accidentally import from the
working directory.

---

## Public API

Deliberately tiny. Everything below is re-exported from `hierarchical_mse`.

```python
GroupIndex(groups, lam=None, group_weights=None)   # precomputed geometry; .lam, .nbar, .cv2, .kappa, .sizes
GroupIndex.group_mean(v) -> np.ndarray             # length-B group means of a length-N vector

power_loss(y, pred, idx) -> float                  # the value, in reported (unscaled) form
power_loss_parts(y, pred, idx) -> (mse_within, mse_between)
rho_between(y, pred, idx) -> float                 # between-group correlation of y and pred
grad_hess(y, pred, idx, hessian="bound") -> (grad, hess)
init_raw_score(y, idx) -> float

lgbm_objective(idx, hessian="bound") -> callable   # (y_true, y_pred) -> (grad, hess)
lgbm_eval(idx_by_dataset) -> callable              # feval: (preds, dataset) -> (name, value, False)
```

`groups` accepts any array of group labels (ints, strings, tuples) and is factorized internally.
`GroupIndex` exists so the `bincount`s and `κ_b` are computed once, not per boosting round.

`lam=None` defaults to `n̄(1+cv²)` **from the given groups**, but is explicitly settable — both because
general grouped regression treats it as a knob, and because in the experimentation use case the model is
often trained on one dataset while λ should come from another's group-size distribution. The docstring must
state both readings.

`rho_between` is worth exposing rather than leaving to users: it is the quantity the source derivation shows
governs the downstream benefit, so it is the right thing to log when comparing models.

---

## Design decisions

These are the things that are easy to get wrong and expensive to debug later. Each gets a test.

**D1 — Hessian must be the majorizing bound, not the exact diagonal.** The true per-group Hessian block is
`H_b = I + (κ_b/n_b)·J`, with eigenvalues `{1 (×(n_b−1)), 1+κ_b}`. LightGBM's leaf value `−Σg/(Σh+reg_lambda)`
is a Newton step that **assumes `H` is diagonal**. All the off-diagonal mass sits in the "shift every row in
group `b`" direction, where curvature is `1+κ_b` — up to `n_b`× what the diagonal `1+κ_b/n_b` reports.
Supplying the exact diagonal therefore **overshoots precisely on the group-aligned splits the loss exists to
encourage**. Since `diag(1+κ_b) ⪰ H_b` (that value *is* the top eigenvalue), the bound gives a valid
majorize-minimize step and monotone descent at `learning_rate=1.0`. Cost: leaves that are purely within-group
are over-damped by ~λ, so `MSE_within` is learned slowly — acceptable, since that component is the one being
deliberately de-prioritized. Expose `hessian="diag"` for anyone who wants to trade the guarantee for speed.

**D2 — Row alignment is an unchecked assumption.** The gradient is only correct if `y_pred` arrives in the
original training-row order so `codes` lines up. This is LightGBM's documented behaviour, but it's exactly the
kind of thing that silently yields a plausible-but-wrong model. Dedicated test with a negative control (T5).

**D3 — The `N/2` rescale is not cosmetic.** Un-rescaled hessians are `O(1/N)`, and LightGBM's default
`min_child_weight=1e-3` would then silently refuse to split any realistically-sized leaf. Document it, and
note that `reg_lambda` semantics shift too: `Σh` is now up to `(1+κ_b)` per row rather than 1, so a
`reg_lambda` ported from an L2-trained model is effectively ~λ× weaker. Do **not** silently rescale it.

**D4 — Native `lgb.train`, not `LGBMRegressor`, when there is an eval set.** The sklearn wrapper's
`eval_metric` callable receives only `(y_true, y_pred)` — no dataset identity — so with more than one eval set
there's no reliable way to pick the right `GroupIndex` (length-matching breaks when train and valid are the
same size). `feval` receives the `Dataset` itself. `lgbm_objective` still returns the sklearn-compatible
signature so `LGBMRegressor(objective=...)` works for the single-dataset case.

**D5 — Early stopping must watch this loss, not `l2`.** Otherwise the stopping rule optimizes the very
objective the method rejects, and cuts training exactly when the between-group fit starts paying off. This is
why `lgbm_eval` ships alongside the objective rather than being left to the user.

**D6 — `boost_from_average` does not apply to custom objectives.** Initial raw score is 0, so callers must
pass `init_score = init_raw_score(y, idx)`. For large λ this is close to the *unweighted mean of group means*,
not the grand mean — materially different when group sizes are dispersed, which is the whole setting. README
must show this; it's the easiest thing to forget.

**D7 — lightgbm is an optional extra.** `loss.py` and `groups.py` are pure numpy. `lgbm.py` imports lightgbm
inside the functions and raises a clear `ImportError` naming `pip install hierarchical-mse[lightgbm]`.
`__init__.py` must re-export the lgbm names without importing lightgbm at module scope — use a module-level
`__getattr__` (PEP 562) so `import hierarchical_mse` works without lightgbm installed.

**D8 — Grouping is the caller's business.** The library takes an array of group labels and nothing else. It
does not know or care whether groups are randomization units, customers, regions, or time buckets, and must
not import anything experiment-specific. Keep the docstrings in group language; mention the source paper once,
in the README and CLAUDE.md, as provenance for the default λ.

---

## Tests

`tests/conftest.py` provides a hierarchical DGP: `B` groups with lognormal-Poisson sizes (target `n̄`, `cv`),
outcome `y = √S_between·M_b + √S_within·ε_{b,i}`, and a feature set of **many weak within-group predictors
plus a few group-level predictors**. That feature structure matters — an earlier two-covariate attempt showed
no gain at all, because with near-orthogonal features a regularizer shrinks each coefficient independently and
never has to *allocate* capacity, so no trade-off binds. The many-weak-features case is the one where the loss
earns its keep, and it is also the common shape in practice.

| # | test | assertion |
|---|---|---|
| **T1** | **λ=0 ≡ built-in L2** | With λ=0, same seed/params, `init_score=mean(y)`: custom-objective booster matches `objective="l2"` to floating-point. **Run this first** — it validates the whole harness against a known-good reference before anything interesting is asserted. |
| T2 | finite-difference gradient | Perturb `pred` elementwise on a small dataset (~200 rows, ~20 groups); analytic `grad` matches to ~1e-6 relative. |
| T3 | Hessian bound is necessary *and* sufficient | Build `H_b` densely for small groups. Assert `eigmin(diag(h_bound) − H_b) ≥ −1e-10` (valid majorizer) **and** that `diag(h_diag) − H_b` has a strictly negative eigenvalue. The second half is what stops D1 being cargo-cult. |
| T4 | monotone descent | `hessian="bound"`, `learning_rate=1.0`, `subsample=1.0`: loss on train is non-increasing across all boosting rounds. |
| T5 | row-alignment guard | Fit; refit on a permutation of rows with groups permuted identically → predictions match. Then refit with a **scrambled** group array → loss materially worse. Without the negative control this test passes vacuously. |
| T6 | it improves the between-group fit | Capacity-binding regime (small `num_leaves`, large `min_child_samples`), train/holdout split **by group**: custom objective achieves lower held-out `power_loss` **and** higher `rho_between` than `objective="l2"` at identical hyperparameters. |
| T7 | Hessian-mode contrast | `hessian="diag"` at `learning_rate=1.0` shows non-monotone loss where `"bound"` is monotone. Characterization test — assert loosely, keep tolerant across LightGBM versions. |
| T8 | orthogonal decomposition | `power_loss_parts` satisfies `MSE_total = MSE_within + MSE_between` to float tolerance. |
| T9 | design constants | `GroupIndex` recovers known `n̄`, `cv²`, `λ` on hand-built size vectors, including the degenerate equal-size case (`cv²=0 ⇒ λ=n̄`) and singleton groups (`n_b=1 ⇒ MSE_within` contribution 0). |
| T10 | group weights | `ω_b ∝ n_b` changes `κ_b` as derived; `ω_b = 1` reproduces the unweighted path exactly. |
| T11 | label handling | String / tuple / non-contiguous-integer group labels all factorize to the same result as contiguous ints. |

`test_lgbm.py` guards with `pytest.importorskip("lightgbm")` so the suite passes without the extra.

**Split holdouts by group, never by row.** `MSE_between` is supported on `B` groups rather than `N` rows, so
up-weighting it by λ concentrates the effective objective on a much smaller sample. A random row split both
leaks (a group straddles folds) and reports between-group gains that don't transfer. T6 must enforce this,
and the README should say it.

---

## Build steps

1. `git init`; add `.gitignore` (Python + `.venv/`, `dist/`, `.ruff_cache/`, `.pytest_cache/`) and
   `.python-version` (3.11).
2. `uv init --lib --name hierarchical-mse` in the folder, then reshape to the layout above.
3. `pyproject.toml`: `requires-python = ">=3.10"`, `dependencies = ["numpy>=1.20"]`,
   `[project.optional-dependencies] lightgbm = ["lightgbm>=4.0"]`,
   `[dependency-groups] dev = ["pytest", "ruff", "scikit-learn"]` (sklearn is test-only, for the DGP and
   reference fits). MIT license, PyPI classifiers, project URLs.
   Pin `lightgbm>=4.0`: the `fobj` argument to `lgb.train` was removed in 4.0 — the callable now goes in
   `params["objective"]`, and D4 assumes the 4.x API.
4. `uv lock && uv sync --all-extras --group dev` → commit `uv.lock`.
5. Implement `groups.py` → `loss.py` → `lgbm.py`, in that order.
6. Write tests in the order **T1, T9, T8, T2, T3** (pure/cheap and harness-validating) then **T4–T7, T10, T11**.
7. `ruff.toml`; run `uv run ruff check --fix && uv run ruff format`.
8. `.github/workflows/ci.yml`: matrix over Python 3.10–3.13, `uv sync --all-extras --group dev`,
   `uv run ruff check`, `uv run pytest`, plus a `uv build` job.
9. `README.md`: the loss in two lines, a copy-paste LightGBM example **including `init_score` (D6)**, the λ
   explanation (knob in general, derived constant in the experimentation case), and the group-wise holdout
   warning.
10. `CLAUDE.md`: see below.

## CLAUDE.md contents

So a future session doesn't re-derive the math or re-make a settled decision:

- One-paragraph statement of what the loss is, in group language, with the arXiv link as provenance.
- The gradient/Hessian table and the `κ_b` definition, verbatim from this plan.
- **Invariants that must not be broken silently:** the `N/2` scaling (D3), `hessian="bound"` as default (D1),
  lazy lightgbm import (D7), λ settable rather than always inferred, and **no switchback/experiment
  vocabulary in the API or docstrings** (D8).
- The λ=0 ≡ L2 identity, flagged as the first thing to check when anything looks wrong.
- Commands: `uv sync --all-extras --group dev`, `uv run pytest`, `uv run ruff check --fix`, `uv build`.
- Scope boundary: covariate pipelines and any experiment-analysis machinery are **out of scope**.

---

## Verification

```
uv sync --all-extras --group dev
uv run ruff check && uv run pytest -v
uv run pytest -v -k "not lgbm"     # must also pass with lightgbm absent
uv build                            # sdist + wheel
uv run --isolated --with dist/*.whl python -c "import hierarchical_mse as m; print(m.__version__, m.power_loss)"
```

The last two lines are the real check on D7: install **without** the extra and confirm
`import hierarchical_mse` still works and the pure-numpy API is usable.

## Out of scope

Covariate pipelines, experiment analysis, and the row-augmentation trick that implements this same loss for
models linear in the features (appending one group-mean row per group with weight `n̄²(1+cv²)` is *exact*
there, because `g(X̄_b) = ḡ_b`; that identity fails for trees, which is why this package exists).
