# Getting Started

## Key concepts

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

## Minimal RBC example

Here is a complete two-variable RBC model — Euler equation and capital accumulation — solved with a 10% capital shock.

```python
import numpy as np
from pyperfectforesight import v, process_model, solve_perfect_foresight

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

vars_dyn = ["c", "k"]
model_funcs = process_model([eq_euler, eq_kacc], vars_dyn)

# Steady state
K_SS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
C_SS = K_SS**ALPHA - K_SS
ss = np.array([C_SS, K_SS])

# Transition path: k_{-1} starts 10% above steady state
T = 100

# initial_state = k_{-1} (pre-period-0 capital, Dynare convention)
k_neg1 = np.array([K_SS * 1.1])

sol = solve_perfect_foresight(
    T, {}, ss, model_funcs, vars_dyn,
    initial_state=k_neg1,
    stock_var_indices=[1],    # index of k in vars_dyn
)
print(f"Converged: {sol.success}")

# Unpack solution
X = sol.x.reshape(T, -1)  # shape (T, 2): columns are [c, k]
c_path = X[:, 0]
k_path = X[:, 1]
```

## RBC model with exogenous TFP shock

When the model has exogenous variables, pass `vars_exo` to `process_model` and supply a `T × n_exo` array as `exog_path`:

```python
import sympy as sp
import numpy as np
from pyperfectforesight import v, process_model, solve_perfect_foresight

ALPHA, BETA = 0.36, 0.99

# TFP shock z enters the production function
eq_euler = v("c", 0)**(-1) - BETA * ALPHA * v("k", 0)**(ALPHA-1) * v("c", 1)**(-1)
eq_kacc  = v("k", 0) - sp.exp(v("z", 0)) * v("k", -1)**ALPHA + v("c", 0)

vars_dyn = ["c", "k"]
model_funcs = process_model([eq_euler, eq_kacc], vars_dyn, vars_exo=["z"])

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

sol = solve_perfect_foresight(
    T, {}, ss, model_funcs, vars_dyn,
    initial_state=k_neg1,
    stock_var_indices=[1],
    exog_path=exog,
)
print(f"Converged: {sol.success}")
```
