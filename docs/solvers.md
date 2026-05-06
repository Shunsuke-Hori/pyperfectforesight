# Solvers

`pyperfectforesight` provides three solver methods on the `Model` class, each targeting a different use case.

---

## `Model.solve`

The core solver. Given parameters and boundary conditions, it finds the perfect foresight transition path by solving the $T \times n$ BVP system with a sparse Newton method.

### Basic usage

```python
import numpy as np
from pyperfectforesight import Model

m = Model()
m.endog("c k")
m.params("alpha beta")

eq_euler = m.c[0]**(-1) - m.beta * m.alpha * m.k[0]**(m.alpha - 1) * m.c[1]**(-1)
eq_kacc  = m.k[0] - m.k[-1]**m.alpha + m.c[0]
m.build([eq_euler, eq_kacc])

PARAMS = {m.alpha: 0.36, m.beta: 0.99}
ss = m.steady_state(PARAMS)

T = 100
k_neg1 = np.array([ss[1] * 1.1])   # k_{-1}: 10% above steady state

sol = m.solve(T, PARAMS, initial_state=k_neg1, endval=ss)

if sol.success:
    X = sol.x.reshape(T, -1)  # shape (T, n_endo)
```

The return value is a `scipy.optimize.OptimizeResult`-like object with `.success`, `.message`, and `.x` (the flattened `T*n` solution).

### Options

| Parameter | Default | Description |
|---|---|---|
| `exog_path` | `None` | Exogenous variable path — either a `(T, n_exo)` array or a `{str: array}` dict mapping variable names to length-T arrays (e.g. `{"z": np.ones(T)}`). Pass `None` or omit when there are no exogenous shocks. |
| `initial_state` | `None` | Pre-period-0 values of stock variables ($k_{-1}$ in Dynare notation). Mutually exclusive with `ss_initial`. When neither is provided, defaults to `endval[stock_var_indices]`. |
| `ss_initial` | `None` | Full initial steady-state vector for the `initval` boundary row. Use this for an on-SS start when `endval` differs from the initial SS. Mutually exclusive with `initial_state`. |
| `stock_var_indices` | `None` | Column indices (into `vars_dyn`) of stock (predetermined) variables. Inferred automatically from the lead-lag incidence table when not provided. |
| `endval` | *(required keyword)* | Terminal steady state — the fixed right BVP boundary. Must be a valid steady state consistent with the terminal exogenous level. |
| `method` | `'sparse_newton'` | Solver backend. Currently only `'sparse_newton'` is supported. |
| `solver_options` | `None` | Dict of sparse Newton solver options: `maxiter`, `ftol`, `xtol`, `maxfev`. |
| `homotopy_fallback` | `True` | If `True`, automatically retries with homotopy continuation when direct Newton fails. |
| `homotopy_options` | `None` | Dict of options forwarded to the homotopy fallback: `n_steps`, `verbose`, and other `solve_homotopy` options. |

---

## `Model.solve_homotopy`

When direct Newton fails to converge — typically for large shocks far from steady state — homotopy continuation incrementally scales the shock from zero to its full value, using the previous step's solution as a warm start.

### When to use it

- Direct `Model.solve` returns `sol.success = False`
- Initial state is far from steady state (e.g., capital 50% above)
- Large permanent shocks that dramatically change the terminal steady state

### Usage

```python
import numpy as np

# (using the same model m from the previous example)

k_neg1 = np.array([ss[1] * 1.5])   # 50% above steady state

sol = m.solve_homotopy(T, PARAMS,
    initial_state=k_neg1,
    endval=ss,
    n_steps=10,      # number of continuation steps from ss to full shock
    verbose=True,
)
print(f"Converged: {sol.success}")
```

### Additional options

The following options from `solve_perfect_foresight` are supported: `exog_path`, `initial_state`, `ss_initial`, `stock_var_indices`, `endval`, `solver_options`, `method`. The options `X0`, `homotopy_fallback`, and `homotopy_options` are **not** accepted. Additional homotopy-specific options:

| Parameter | Default | Description |
|---|---|---|
| `n_steps` | `10` | Number of homotopy steps. Larger values are more robust but slower. Must be a positive integer. |
| `exog_ss` | `None` | Baseline exogenous path at $\lambda=0$ (no shock). Defaults to all zeros. |
| `verbose` | `False` | Print convergence status at each step. |

Note: `endval` is held **fixed** at its supplied value throughout all homotopy steps. Only `initial_state` and `exog_path` are scaled from their $\lambda=0$ baselines to their $\lambda=1$ targets.

The solver raises `RuntimeError` if any intermediate step fails to converge. In that case, try increasing `n_steps`.

---

## `Model.solve_expectation_errors`

Replicates Dynare's `perfect_foresight_with_expectation_errors_solver`. Agents are surprised at one or more `learnt_in` periods, re-solving from each surprise point. The full path is stitched from the resulting sub-simulations.

This is the standard protocol for "news shocks" or "MIT shocks" with multiple surprise dates.

### `news_shocks` format

`news_shocks` is a list of 2-tuples `(learnt_in, exog_path)` or 3-tuples `(learnt_in, exog_path, endval)`:

- **`learnt_in`**: the period at which agents learn of (and start reacting to) the shock. Period numbering starts at 1.
- **`exog_path`**: the agents' belief about the exogenous path, **indexed from period `learnt_in`**. Either a `(T_sub, n_exo)` array or a `{str: array}` dict mapping variable names to length-T_sub arrays. Row 0 = period `learnt_in`, row 1 = period `learnt_in + 1`, etc. Do **not** pre-offset as if row 0 were period 1; the solver handles alignment internally. Pass `None` for an all-zero path (only correct when the exogenous steady state is zero).
- **`endval`** (3-tuple only): override the terminal steady state for this and all subsequent sub-solves. Use this for permanent shocks that change the long-run equilibrium.

The list must be **sorted by `learnt_in`** and the **first entry must have `learnt_in=1`**.

### `exog_path` row alignment

For a sub-solve starting at `learnt_in=k`, the solver uses rows `0` through `T-k` of the supplied `exog_path` (i.e., `T - k + 1` rows). Passing a full `T`-row array is always safe; extra rows are ignored. When `constant_simulation_length=True` every sub-solve uses all `T` rows.

### `endval` persistence

An `endval` supplied in a 3-tuple applies to that sub-solve and remains the terminal boundary for **all later segments** unless overridden by another 3-tuple further down the list. This mirrors Dynare's `endval(learnt_in=k)` semantics for permanent shocks.

### Usage example

```python
import numpy as np

# (using the same model m with exogenous TFP z as in Getting Started)
T = 100
ss = m.steady_state(PARAMS, exog_ss=np.array([0.0]))
k_neg1 = np.array([ss[1]])   # start at steady state

# Agents initially expect no shock (period 1).
# At period 3 they learn of a permanent 1% TFP shock.
exog_surprise = np.full((T, 1), 0.01)   # permanent shock from period 3 onward

news_shocks = [
    (1, None),               # period 1: baseline, no shock expected
    (3, exog_surprise),      # period 3: agents learn of permanent TFP shock
]

sol = m.solve_expectation_errors(T, PARAMS, news_shocks,
    initial_state=k_neg1,
    endval=ss,
)
print(f"Converged: {sol.success}, message: {sol.message}")

X_full = sol.x.reshape(T, -1)   # (T, n_endo) stitched path
```

### Example with changing terminal steady state

When the shock is permanent and shifts the long-run equilibrium, pass the new steady state explicitly in a 3-tuple:

```python
ss_initial  = m.steady_state(PARAMS, exog_ss=np.array([0.0]))
ss_terminal = m.steady_state(PARAMS, exog_ss=np.array([0.05]))

exog_surprise = np.full((T, 1), 0.05)

news_shocks = [
    (1, None),                              # period 1: no shock yet
    (3, exog_surprise, ss_terminal),        # period 3: permanent shock; endval changes
]

sol = m.solve_expectation_errors(T, PARAMS, news_shocks,
    initial_state=k_neg1,
    endval=ss_initial,       # initial terminal boundary (overridden at learnt_in=3)
)
```

### Options

| Parameter | Default | Description |
|---|---|---|
| `news_shocks` | *(required)* | List of `(learnt_in, exog_path)` or `(learnt_in, exog_path, endval)` tuples. |
| `endval` | *(required keyword)* | Initial terminal steady state. Overridden by any 3-tuple entry in `news_shocks`. |
| `initial_state` | `None` | Same semantics as `solve_perfect_foresight`. |
| `ss_initial` | `None` | Same semantics as `solve_perfect_foresight`. |
| `stock_var_indices` | `None` | Same semantics as `solve_perfect_foresight`. |
| `constant_simulation_length` | `False` | If `False` (Dynare default), each sub-solve uses the shrinking horizon `T - learnt_in + 1`. If `True` (Dynare's `constant_simulation_length` option), every sub-solve uses the full `T` periods. |
| `solver_options` | `None` | Forwarded to each sub-solve. Same keys as `solve_perfect_foresight`. |
| `sub_x0` | `None` | Per-sub-solve initial guesses. A list or tuple of the same length as `news_shocks`; each entry is either `None` (use the automatic warm-start) or an `(T_sub, n_endo)` array to use as the warm-start for that sub-solve. Rows are trimmed or padded to `T_sub` if needed. |

### Supplying per-sub-solve initial guesses (`sub_x0`)

By default each sub-solve is warm-started from the previous sub-solve's tail solution. This works well when sub-solve 1 is non-trivial, but can break down in the common pre-announcement pattern where sub-solve 1 is trivial (agents stay at the initial steady state) and sub-solve 2 must transition to a new steady state. In that case the automatic warm-start for sub-solve 2 (all `ss1`) can be far from the solution.

Use `sub_x0` to inject high-quality initial guesses directly:

```python
from pyperfectforesight import make_initial_guess

# news_shocks has three entries; learnt_in values are read from the list.
news_shocks = [
    (1,  None),           # period 1: baseline, agents expect no shock
    (10, exog_news),      # period 10: agents learn of a news shock
    (25, exog_dis),       # period 25: shock is disappointed
]

T_sub2 = T - news_shocks[1][0] + 1   # T - 10 + 1
T_sub3 = T - news_shocks[2][0] + 1   # T - 25 + 1

sub_x0 = [
    None,                                                                # sub-solve 1: trivial, auto warm-start is fine
    make_initial_guess(T_sub2, ss1_vec, ss2_vec, method='exponential'),  # sub-solve 2: news shock
    make_initial_guess(T_sub3, ss2_vec, ss1_vec, method='exponential'),  # sub-solve 3: disappointment
]

sol = m.solve_expectation_errors(T, PARAMS, news_shocks,
    sub_x0=sub_x0,
    initial_state=k_neg1,
    endval=ss1_vec,
)
```

---

## Terminal steady state

For permanent shocks that shift the long-run equilibrium, the terminal steady state must be consistent with the terminal exogenous level.  `m.steady_state(params, exog_ss=...)` computes it at any exogenous level; the result is a `SteadyState` object that is transparently usable as a numpy array.

### `SteadyState`

```python
import numpy as np

# (using model m with exogenous TFP z)
ss_initial  = m.steady_state(PARAMS, exog_ss=np.array([0.0]))
ss_terminal = m.steady_state(PARAMS, exog_ss=np.array([0.05]))

print(ss_terminal)
# SteadyState(values={c: 2.972, k: 40.999}, params={alpha: 0.36, beta: 0.99}, exog_ss={z: 0.05})

# Access provenance at any time
ss_terminal.values    # endogenous values as ndarray
ss_terminal.params    # {'alpha': 0.36, ...}
ss_terminal.exog_ss   # array([0.05])
ss_terminal.vars_exo  # ['z']
```

`SteadyState` is a drop-in replacement for any plain `ndarray` used as `ss_initial` or `endval`.

### Passing `endval` explicitly

Compute the terminal steady state once and pass it as `endval`:

```python
T = 100
exog_path = np.full((T, 1), 0.05)  # permanent shock

sol = m.solve(T, PARAMS,
    exog_path=exog_path,
    ss_initial=ss_initial,
    endval=ss_terminal,
)
```

For repeated simulations with the same terminal exogenous level, the same `endval` object can be reused across calls without recomputation:

```python
for shock in shock_list:
    sol = m.solve(T, PARAMS,
        exog_path=shock,
        ss_initial=ss_initial,
        endval=ss_terminal,   # pre-computed, reused across calls
    )
```
