# Getting Started

## Key concepts

### Declaring symbols: `v()`, `p()`, and `vars_params`

Use dedicated constructors for each class of symbol:

| Symbol type | Constructor | Declaration list |
|---|---|---|
| Endogenous variable at time `t+lag` | `v(name, lag)` | `vars_dyn` |
| Exogenous variable at time `t` | `v(name, 0)` | `vars_exo` |
| Parameter (time-invariant) | `p(name)` | `vars_params` |

```python
from pyperfectforesight import p, v, process_model

alpha = p("alpha")   # parameter
k_m   = v("k", -1)  # k_{t-1}
k_0   = v("k",  0)  # k_t

vars_params = ["alpha", "beta", "delta"]
model_funcs = process_model(equations, vars_dyn, vars_params=vars_params)
```

**Naming constraint**: a parameter name must not match the pattern `<var>_<int>` when `<var>` is also a declared endogenous, exogenous, or auxiliary variable — e.g. `p("rho_1")` would collide with `v("rho", 1)`. `process_model` raises a `ValueError` if such a clash is detected.

### Dynare lag notation

Equations are written using the `v(name, lag)` helper, which creates a time-indexed SymPy symbol. The lag argument follows Dynare's convention:

| Expression | Meaning |
|---|---|
| `v("k", -1)` | $k_{t-1}$ — lagged value (one period ago) |
| `v("k", 0)` | $k_t$ — current-period value |
| `v("c", 1)` | $c_{t+1}$ — lead value (one period ahead) |

For example, the standard capital accumulation equation $k_t = k_{t-1}^\alpha - c_t$ is written:

```python
eq_kacc = v("k", 0) - v("k", -1)**ALPHA + v("c", 0)
```

Note that `k` appears at lag `-1` — this is Dynare's convention for a stock variable that accumulates from last period.

### `initial_state` semantics

`initial_state` is always the **pre-period-0** value of the stock variable(s) — i.e., $k_{-1}$ in Dynare notation. The period-0 values of *all* variables, including jump variables such as consumption, are determined simultaneously by the model equations.

Do not confuse `initial_state` with $k_0$ (the period-0 value of capital, which is endogenous). If `initial_state` is omitted it defaults to `ss[stock_var_indices]`, meaning the economy starts at the initial steady state.

### Stock variable inference

A variable is classified as a **stock** (predetermined) variable if it appears at any negative lag in the model equations. A variable that only appears at lag 0 or positive lags is a **jump** variable — it is free to respond at $t=0$ and is not pinned by `initial_state`.

`stock_var_indices` is inferred automatically from the lead-lag incidence table computed during `process_model`. You can always pass it explicitly to override the inference:

```python
sol = solve_perfect_foresight(..., stock_var_indices=[1])  # force k (index 1) as stock
```

### BVP (augmented-path) formulation

The solver always uses the **augmented-path BVP formulation**. It builds a `T+2`-row path:

- **Row 0** (`initval`): pre-period-0 boundary — stock variables from `initial_state`, all others from `ss_initial`
- **Rows 1 … T**: the `T` free periods (the unknowns being solved)
- **Row T+1** (`endval`): terminal steady state `ss` (or a user-supplied `endval`)

Residuals are evaluated at $t = 0, \ldots, T-1$ using the full augmented path, so all $T \times n$ unknowns are determined simultaneously. This correctly handles jump variables: pinning `X[0]` directly would over-constrain them and produce a structurally singular Jacobian.

## Model class

The `Model` class is the recommended entry point.  It wraps
`process_model` at construction time so that `model_funcs` and
`vars_dyn` never need to be repeated at solve time, and it exposes
`steady_state()`, `solve()`, `solve_homotopy()`, and
`solve_expectation_errors()` as methods.

```python
from pyperfectforesight import Model

model = Model(equations, vars_dyn, vars_exo=..., vars_params=...)
```

**Attributes** set after construction:

| Attribute | Description |
|---|---|
| `model.vars_dyn` | Endogenous variable names (post-elimination) |
| `model.vars_exo` | Exogenous variable names |
| `model.vars_params` | Parameter names |
| `model.vars_aux` | Auxiliary variable names |
| `model.aux_method` | Auxiliary variable handling method used (`'analytical'`, `'nested'`, or `'dynamic'`) |

**Methods:**

| Method | Description |
|---|---|
| `model.steady_state(params, exog_ss=None, initial_guess=None)` | Compute the model steady state numerically |
| `model.solve(T, params, ss, ...)` | Solve the perfect foresight problem |
| `model.solve_homotopy(T, params, ss, ...)` | Solve via homotopy continuation |
| `model.solve_expectation_errors(T, params, ss, news_shocks, ...)` | Solve with surprise MIT shocks |

The model lazily builds a compiled steady-state bundle the first time it
is needed.  For permanent shocks, `endval` is auto-computed from
`exog_path[-1]` without any extra setup — pass `endval=...` explicitly
to override, or `compiled_ss=None` to disable auto-computation.

## Minimal RBC example

Here is a complete two-variable RBC model — Euler equation and capital
accumulation — solved with a 10% capital shock.

```python
import numpy as np
from pyperfectforesight import v, Model

# Parameters baked in numerically
ALPHA = 0.36
BETA  = 0.99

# Dynare-style equations:
#   Euler:   1/c_t = beta * alpha * k_t^(alpha-1) / c_{t+1}
#   Capital: k_t   = k_{t-1}^alpha - c_t
#
# k appears at lag -1 in the accumulation equation (Dynare convention).
eq_euler = v("c", 0)**(-1) - BETA * ALPHA * v("k", 0)**(ALPHA-1) * v("c", 1)**(-1)
eq_kacc  = v("k", 0) - v("k", -1)**ALPHA + v("c", 0)

model = Model([eq_euler, eq_kacc], ["c", "k"])

# Steady state
K_SS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
C_SS = K_SS**ALPHA - K_SS
ss = np.array([C_SS, K_SS])

# Transition path: k_{-1} starts 10% above steady state
T = 100
k_neg1 = np.array([K_SS * 1.1])   # initial_state = k_{-1} (Dynare convention)

sol = model.solve(T, {}, ss, initial_state=k_neg1, stock_var_indices=[1])
print(f"Converged: {sol.success}")

# Unpack solution
X = sol.x.reshape(T, -1)  # shape (T, 2): columns are [c, k]
c_path = X[:, 0]
k_path = X[:, 1]
```

## RBC model with exogenous TFP shock

When the model has exogenous variables, pass `vars_exo` and supply an
`exog_path` — either a `T × n_exo` array or a `{name: array}` dict:

```python
import sympy as sp
import numpy as np
from pyperfectforesight import v, Model

ALPHA, BETA = 0.36, 0.99

eq_euler = v("c", 0)**(-1) - BETA * ALPHA * v("k", 0)**(ALPHA-1) * v("c", 1)**(-1)
eq_kacc  = v("k", 0) - sp.exp(v("z", 0)) * v("k", -1)**ALPHA + v("c", 0)

model = Model([eq_euler, eq_kacc], ["c", "k"], vars_exo=["z"])

K_SS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
C_SS = K_SS**ALPHA - K_SS
ss = np.array([C_SS, K_SS])

T = 100

# AR(1) TFP shock: 1% on impact, rho=0.9 decay
rho = 0.9
exog = np.zeros((T, 1))
exog[0, 0] = 0.01
for t in range(1, T):
    exog[t, 0] = rho * exog[t-1, 0]

k_neg1 = np.array([K_SS])   # k_{-1} at steady state

sol = model.solve(T, {}, ss, initial_state=k_neg1, stock_var_indices=[1],
                  exog_path={"z": exog[:, 0]})   # dict form; array also accepted
print(f"Converged: {sol.success}")
```

## Permanent shock with auto-computed terminal steady state

For a **permanent** shock the terminal steady state differs from the
initial one.  With the `Model` class this is handled automatically:
`model.solve()` computes `endval` from `exog_path[-1]` using the
model's lazily-built compiled steady-state bundle — no extra setup
needed.

```python
import numpy as np
from pyperfectforesight import p, v, Model

ALPHA = p("alpha")
BETA  = p("beta")
PARAMS = {ALPHA: 0.36, BETA: 0.99}

eq_euler = 1/v("c", 0) - BETA * ALPHA * v("z", 1) * v("k", 0)**(ALPHA - 1) / v("c", 1)
eq_kacc  = v("k", 0) - v("z", 0) * v("k", -1)**ALPHA + v("c", 0)

model = Model([eq_euler, eq_kacc], ["c", "k"],
              vars_exo=["z"], vars_params=["alpha", "beta"])

# Pre-shock steady state: z = 1.0
ss_pre = model.steady_state(PARAMS, exog_ss=np.array([1.0]))

T = 100
# Permanent TFP increase: z jumps from 1.0 to 1.05 at period 0
exog_path = np.full((T, 1), 1.05)

# endval is auto-computed from exog_path[-1] (post-shock SS at z=1.05)
sol = model.solve(T, PARAMS, ss_pre,
                  exog_path=exog_path,
                  initial_state=np.array([ss_pre[1]]))
print(f"Converged: {sol.success}")

X = sol.x.reshape(T, -1)
c_path, k_path = X[:, 0], X[:, 1]
```

Pass `endval=...` explicitly to override the auto-computed terminal
steady state, or `compiled_ss=None` to disable auto-computation
entirely.

## Inequality constraints and the zero lower bound

Encode inequality constraints using SymPy's `sp.Min` or `sp.Max` directly in
the equation list.  The standard `sparse_newton` solver treats them as ordinary
nonlinear equations.

The canonical example is the zero lower bound (ZLB) on the nominal interest
rate.  Write it as the **NCP (Fischer-min) condition**:

```python
import sympy as sp
from pyperfectforesight import v

i_0    = v("i",    0)
xi_2_0 = v("xi_2", 0)
sigma  = 1.0   # or a parameter symbol

# min(xi_2/σ, i) = 0  encodes:
#   xi_2/σ ≥ 0,  i ≥ 0,  (xi_2/σ) · i = 0
eq_zlb = sp.Min(xi_2_0 / sigma, i_0)
```

This single equation encodes all three complementarity conditions at once.
The partial derivatives of `sp.Min(a, b)` are piecewise: with respect to `a`,
the derivative is 1 when `a < b` and 0 when `a > b`; with respect to `b`, it
is 0 when `a < b` and 1 when `a > b`.  These can be written using Heaviside
factors — `∂Min(a,b)/∂a = Heaviside(b − a)` and `∂Min(a,b)/∂b = Heaviside(a − b)`
— which is what SymPy's lambdified Jacobian computes.  The kink at `a = b`
(measure zero) does not prevent Newton convergence in practice.

`Model` (and the underlying `process_model`) still attempts static elimination
via `sp.solve`.  If `sp.solve` raises `NotImplementedError` — which typically
happens for equations involving `sp.Min` or `sp.Max` — it falls back to no
static elimination.  No special flag is needed:

```python
from pyperfectforesight import Model

model = Model(
    equations, vars_dyn,
    vars_exo=vars_exo,
    vars_params=vars_params,
)
```
