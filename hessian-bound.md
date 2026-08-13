# The Hessian bound: what it does and why

Background note for design decision **D1** in [plan.md](plan.md). Explains what
`hess_i = 1 + κ_b` actually does, why the exact diagonal is not the safe choice, and what the bound costs.

---

## 1. Setup

The loss, rescaled by `N/2` (see D3), over groups `b` with `n_b` rows each:

$$\mathcal{L} = \tfrac12\sum_{b,i} e_{b,i}^2 + \tfrac{\lambda\bar n}{2}\sum_b \bar e_b^{\,2}, \qquad e = y - g, \qquad \kappa_b := \frac{\lambda\bar n}{n_b}$$

Differentiating twice with respect to the **prediction vector** gives a Hessian that is block-diagonal by
group, with block

$$H_b = I_{n_b} + \frac{\kappa_b}{n_b}J_{n_b}, \qquad \text{eigenvalues}\;\; \big\{\underbrace{1}_{\times(n_b-1)},\;\; 1+\kappa_b\big\}$$

where `J` is the all-ones matrix. Two facts about this block drive everything below:

- Its **diagonal** entries are `1 + κ_b/n_b`.
- Its **largest eigenvalue** is `1 + κ_b`, attained on the constant vector `1` — the direction "shift every
  row in group `b` by the same amount".

The gap between those two numbers is the entire subject of this document. It is a factor of `n_b`.

---

## 2. What LightGBM needs, versus what it can be told

LightGBM's leaf value is

$$w^\star = \frac{-\sum_{i \in L} \text{grad}_i}{\sum_{i \in L}\text{hess}_i + \texttt{reg\_lambda}}$$

This comes from minimizing `w·Σg + ½w²·Σh`, which is correct only if the curvature of "shift every row in
leaf `L` by `w`" equals `Σh`. The true curvature is the full quadratic form:

$$\mathbf{1}_L^\top H\, \mathbf{1}_L \;=\; \sum_b \Big[\,m_b \;+\; \tfrac{\kappa_b}{n_b}m_b^2\,\Big], \qquad m_b := \big|\,L \cap \text{group } b\,\big|$$

For a **separable** loss `H` is genuinely diagonal, so `1ᵀH1 = Σh` exactly and LightGBM is right. Our loss is
not separable: every pair of rows in the same group contributes an off-diagonal `κ_b/n_b`, and there are
`m_b²` such pairs.

**No per-row `hess` can fix this.** `Σ_{i∈L} h_i` is *linear* in `m_b`; the truth is *quadratic* in `m_b`.
The two can only agree at a single leaf composition. Evaluating the endpoints:

| leaf composition | `1ᵀH1` | per-row `h` that matches it |
|---|---|---|
| `m_b = 1` — one row per group, **scattered** | `Σ_b (1 + κ_b/n_b)` | `1 + κ_b/n_b` ← the exact diagonal |
| `m_b = n_b` — a whole group, **pure** | `n_b(1 + κ_b)` | `1 + κ_b` ← **the bound** |

So the diagonal and the bound are not two heuristics among many. They are precisely the two endpoints of what
the API can express, and every choice is a point on that segment.

---

## 3. What the bound does

Set `hess_i = 1 + κ_b` — the top eigenvalue of the block. Three consequences.

### It makes the surrogate an upper bound

Because `1 + κ_b` is the largest eigenvalue, `diag(1+κ_b) ⪰ H_b` in the positive-semidefinite order. So the
quadratic model LightGBM minimizes sits **above** the true loss everywhere, touching it at the current
prediction. That is a textbook majorize-minimize construction, and its guarantee is that minimizing the
surrogate can never increase the true loss — **monotone descent at `learning_rate = 1.0`**, no tuning.

The exact diagonal has no such guarantee: `diag(1+κ_b/n_b) − H_b` is *not* PSD, so the surrogate dips below
the true loss and "minimizing" it can move uphill.

```
n_b = 180, lambda = 585  ->  kappa_b = 585
  diag  (1 + k/n_b):  eigmin(diag(h) - H) = -581.75   NOT PSD -> can overshoot
  bound (1 + k)    :  eigmin(diag(h) - H) =    0.00   PSD     -> safe, and tight
```

Tight, note — `0.00`, not some large positive slack. The bound is the *smallest* uniform diagonal that
majorizes, so it damps as little as possible while staying safe.

### It is exact where the loss does its work

For a leaf holding one whole group, `Σh = n_b(1+κ_b)` is **precisely** `1ᵀH1`. Not an approximation. The
resulting leaf value is exactly the correct Newton step, which has a pleasing closed form:

$$\text{grad sum} = -n_b\bar e_b(1+\kappa_b) \quad\Rightarrow\quad w^\star = \frac{n_b \bar e_b (1+\kappa_b)}{n_b(1+\kappa_b)} = \bar e_b$$

*Move the leaf by exactly the group's mean residual.* Compare against what the exact diagonal produces, for
`n̄ = 180`, `cv = 1.5` (so `λ = 585`), on a leaf equal to one whole group with `ē_b = 1`:

| `n_b` | `κ_b` | correct value | exact-diagonal value | bound value | overshoot |
|---|---|---|---|---|---|
| 2 | 52650 | 1.0000 | 2.00 | 1.0000 | 2× |
| 10 | 10530 | 1.0000 | 9.99 | 1.0000 | 10× |
| 50 | 2106 | 1.0000 | 48.86 | 1.0000 | 49× |
| 180 | 585 | 1.0000 | 137.88 | 1.0000 | **138×** |
| 500 | 210.6 | 1.0000 | 148.89 | 1.0000 | 149× |

The overshoot factor is `n_b(1+κ_b)/(n_b+κ_b)`, which approaches `n_b` as `κ_b` grows. It is not a constant
you can absorb into `learning_rate`: it scales with group size, so a rate tuned on one dataset breaks on the
next.

### It also prices splits correctly

`Σh` appears in the split-gain formula too, `(Σg)²/(Σh + reg_lambda)`, so the same reasoning applies to which
splits get chosen — not only to leaf values:

- **Group-pure splits:** the bound gives `Σh = 1ᵀH1` exactly, so their gain is priced correctly.
- **Scattered splits:** the bound overstates `Σh`, so their gain is *understated* — they look less attractive
  than they truly are.

Net effect: a conservative bias toward group-separating splits. It happens to point the same direction as the
objective, but be clear that it is a bias, not a correction.

The exact diagonal gets this backwards in the dangerous way: it **understates** `Σh` on group-pure leaves, so
it **overstates** their gain. LightGBM becomes eager to make exactly the splits it will then overshoot on. The
two errors compound.

---

## 4. What the bound costs

On **group-scattered** leaves the bound overstates curvature by up to `1 + κ_b`, so those steps are damped by
roughly λ. Within-group structure is therefore learned ~λ× more slowly than it would be under plain MSE.

This is tolerable by construction — the whole premise of the loss is that within-group error matters less —
but it is a real cost, not a free lunch, and it is why `hessian="diag"` remains available. If you choose it,
`learning_rate` must be below roughly `1/n̄` to avoid the overshoot above, and monotone descent is no longer
guaranteed.

A secondary effect worth knowing: the over-damping acts as **implicit regularization**. If the bound is ever
replaced by something tighter, expect to need more `reg_lambda` or fewer rounds to compensate.

---

## 5. The general recipe

Nothing above is specific to this loss:

> **Exact gradient + any majorizing diagonal ⇒ guaranteed descent, for any non-separable loss in LightGBM.**

The gradient channel is lossless — `∂L/∂g_i` is a well-defined vector whether or not `L` separates, and
everything LightGBM does with `Σg` is correct. All the information loss is in the `hess` channel. So the
useful question is not *"what is the second derivative?"* but *"what diagonal makes the surrogate an upper
bound?"* Treat the `hess` slot as **step-size control**, not as a derivative.

---

## 6. The alternative we are not doing (yet)

A genuine non-diagonal Newton step is reachable, and worth recording so it does not have to be re-derived.

A tree can only express functions constant on its `K` leaves, so the `N×N` Hessian is irrelevant — only its
`K×K` projection matters. With `Z` the leaf-indicator matrix and `A` the loss's PSD matrix, the exact optimal
leaf values are

$$w^\star = (Z^\top A Z)^{-1}Z^\top A e, \qquad e = y - g_{\text{prev}}$$

Both pieces are cheap. With `M[k,b]` the count of group `b`'s rows in leaf `k`:

$$Z^\top A Z = \tfrac1N\operatorname{diag}(\text{leaf sizes}) + \tfrac{\lambda}{B}M\operatorname{diag}(1/n_b^2)M^\top, \qquad (Z^\top A e)_k = \tfrac1N\sum_{i\in k}e_i + \tfrac{\lambda}{B}\sum_b M[k,b]\tfrac{\bar e_b}{n_b}$$

a `K×K` solve (`K = num_leaves`, typically 31). The API supports it: `predict(pred_leaf=True)` yields `Z`, and
`Booster.set_leaf_output(tree_id, leaf_id, value)` (LightGBM ≥ 4.0, inside our pin) writes values back.

**Why it is not v1.** `set_leaf_output` is a raw C call that writes the tree value and almost certainly does
not invalidate LightGBM's cached internal training scores. Continuing through `lgb.train`'s own loop would
then compute the next round's gradients from stale predictions. Driving the loop manually — one tree per
round with `keep_training_booster`, refitting leaves and re-deriving scores each iteration — is a real
complexity jump and version-sensitive in the way that rots.

The bound is already exact on group-pure leaves, so this only recovers the mixed-leaf damping. Let test **T6**
decide whether that is worth the fragility.

---

## 7. Summary

| | exact diagonal `1 + κ_b/n_b` | **bound `1 + κ_b`** |
|---|---|---|
| majorizes `H`? | no (`eigmin = −581.75`) | **yes, tightly (`0.00`)** |
| monotone descent at `lr=1`? | no | **yes** |
| group-pure leaves | up to `n_b`× overshoot (138× measured) | **exact** |
| scattered leaves | exact | over-damped by up to `1+κ_b` |
| split gains | overstates group-pure splits | correct on group-pure, conservative on scattered |
| failure mode | silent; looks like "the loss doesn't work" | slower within-group learning |

The asymmetry is the point. The diagonal is exact where it does not matter and catastrophically wrong where it
does; the bound is exactly the other way round. Since the entire purpose of the loss is to encourage
group-separating splits, the bound is right where it counts.
