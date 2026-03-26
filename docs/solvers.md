# Solvers

`pyperfectforesight` provides three high-level solver functions, each targeting a different use case.

---

## `solve_perfect_foresight`

The core solver. Given a model, steady state, and initial condition, it finds the perfect foresight transition path by solving the $T \times n$ BVP system with a sparse Newton method.

### Basic usage

```python
from pyperfectforesight import solve_perfect_foresight

sol = solve_perfect_foresight(
    T, X0, params_dict, ss, model_funcs, vars_dyn,
    initial_state=k_neg1,
    stock_var_indices=[1],
)

if sol.success:
    X = sol.x.reshape(T, -1)  # shape (T, n_endo)
```

The return value is a `scipy.optimize.OptimizeResult`-like object with `.success`, `.message`, and `.x` (the flattened `T*n` solution).

### Options

| Parameter | Default | Description |
|---|---|---|
| `exog_path` | `None` | Exogenous variable path, shape `(T, n_exo)`. Pass `None` or omit when there are no exogenous shocks. |
| `initial_state` | `None` | Pre-period-0 values of stock variables ($k_{-1}$ in Dynare notation). Defaults to `ss_initial[stock_var_indices]` (economy starts at steady state). |
| `stock_var_indices` | `None` | Column indices (into `vars_dyn`) of stock (predetermined) variables. Inferred automatically from the lead-lag incidence table when not provided. |
| `ss_initial` | `None` | Initial steady-state values used for the `initval` boundary row. Defaults to `ss`. Set this when the model starts from a *different* steady state than `ss`. |
| `endval` | `None` | Terminal steady state (right BVP boundary). Defaults to `ss`. Set this for permanent shocks that shift the long-run equilibrium. |
| `solver_options` | `None` | Dict of sparse Newton solver options: `maxiter`, `ftol`, `xtol`, `maxfev`. |

---

## `solve_perfect_foresight_homotopy`

When direct Newton fails to converge — typically for large shocks far from steady state — homotopy continuation incrementally scales the shock from zero to its full value, using the previous step's solution as a warm start.

### When to use it

- Direct `solve_perfect_foresight` returns `sol.success = False`
- Initial state is far from steady state (e.g., capital 50% above)
- Large permanent shocks that dramatically change the terminal steady state

### Usage

```python
import numpy as np
from pyperfectforesight import solve_perfect_foresight_homotopy

k_neg1 = np.array([K_SS * 1.5])   # 50% above steady state

sol = solve_perfect_foresight_homotopy(
    T, X0, {}, ss, model_funcs, vars_dyn,
    initial_state=k_neg1,
    stock_var_indices=[1],
    n_steps=10,      # number of continuation steps from ss to full shock
    verbose=True,
)
print(f"Converged: {sol.success}")
```

### Additional options

All options from `solve_perfect_foresight` are accepted, plus:

| Parameter | Default | Description |
|---|---|---|
| `n_steps` | `10` | Number of homotopy steps. Larger values are more robust but slower. Must be a positive integer. |
| `exog_ss` | `None` | Baseline exogenous path at $\lambda=0$ (no shock). Defaults to all zeros. |
| `verbose` | `False` | Print convergence status at each step. |

The solver raises `RuntimeError` if any intermediate step fails to converge. In that case, try increasing `n_steps`.

---

## `solve_perfect_foresight_expectation_errors`

Replicates Dynare's `perfect_foresight_with_expectation_errors_solver`. Agents are surprised at one or more `learnt_in` periods, re-solving from each surprise point. The full path is stitched from the resulting sub-simulations.

This is the standard protocol for "news shocks" or "MIT shocks" with multiple surprise dates.

### `news_shocks` format

`news_shocks` is a list of 2-tuples `(learnt_in, exog_path)` or 3-tuples `(learnt_in, exog_path, endval)`:

- **`learnt_in`**: the period at which agents learn of (and start reacting to) the shock. Period numbering starts at 1.
- **`exog_path`**: the agents' belief about the exogenous path, **indexed from period `learnt_in`**. Row 0 = period `learnt_in`, row 1 = period `learnt_in + 1`, etc. Do **not** pre-offset as if row 0 were period 1; the solver handles alignment internally. Pass `None` for an all-zero path.
- **`endval`** (3-tuple only): override the terminal steady state for this and all subsequent sub-solves. Use this for permanent shocks that change the long-run equilibrium.

The list must be **sorted by `learnt_in`** and the **first entry must have `learnt_in=1`**.

### `exog_path` row alignment

For a sub-solve starting at `learnt_in=k`, the solver uses rows `0` through `T-k` of the supplied `exog_path` (i.e., `T - k + 1` rows). Passing a full `T`-row array is always safe; extra rows are ignored. When `constant_simulation_length=True` every sub-solve uses all `T` rows.

### `endval` persistence

An `endval` supplied in a 3-tuple applies to that sub-solve and remains the terminal boundary for **all later segments** unless overridden by another 3-tuple further down the list. This mirrors Dynare's `endval(learnt_in=k)` semantics for permanent shocks.

### Usage example

```python
import numpy as np
from pyperfectforesight import solve_perfect_foresight_expectation_errors

# Same RBC model with exogenous TFP z as in Getting Started.
T = 100
X0 = np.tile(ss, (T, 1))

# Agents initially expect no shock (period 1).
# At period 3 they learn of a permanent 1% TFP shock.
exog_surprise = np.full((T, 1), 0.01)   # permanent shock from period 3 onward

news_shocks = [
    (1, None),               # period 1: baseline, no shock expected
    (3, exog_surprise),      # period 3: agents learn of permanent TFP shock
]

sol = solve_perfect_foresight_expectation_errors(
    T, X0, {}, ss, model_funcs, vars_dyn,
    news_shocks=news_shocks,
    initial_state=k_neg1,
    stock_var_indices=[1],
)
print(f"Converged: {sol.success}, message: {sol.message}")

X_full = sol.x.reshape(T, -1)   # (T, n_endo) stitched path
```

### Example with changing terminal steady state

When the shock is permanent and shifts the long-run equilibrium, pass the new steady state as `endval` in a 3-tuple:

```python
ss_new = np.array([C_SS_NEW, K_SS_NEW])  # new steady state under permanent shock

news_shocks = [
    (1, None),
    (3, exog_surprise, ss_new),   # 3-tuple: endval takes effect from period 3
]

sol = solve_perfect_foresight_expectation_errors(
    T, X0, {}, ss_new, model_funcs, vars_dyn,
    news_shocks=news_shocks,
    initial_state=k_neg1,
    stock_var_indices=[1],
)
```

### Options

| Parameter | Default | Description |
|---|---|---|
| `news_shocks` | *(required)* | List of `(learnt_in, exog_path)` or `(learnt_in, exog_path, endval)` tuples. |
| `initial_state` | `None` | Same semantics as `solve_perfect_foresight`. |
| `ss_initial` | `None` | Same semantics as `solve_perfect_foresight`. |
| `stock_var_indices` | `None` | Same semantics as `solve_perfect_foresight`. |
| `constant_simulation_length` | `False` | If `False` (Dynare default), each sub-solve uses the shrinking horizon `T - learnt_in + 1`. If `True` (Dynare's `constant_simulation_length` option), every sub-solve uses the full `T` periods. |
| `solver_options` | `None` | Forwarded to each sub-solve. Same keys as `solve_perfect_foresight`. |
| `sub_x0` | `None` | Per-sub-solve initial guesses. A list of the same length as `news_shocks`; each entry is either `None` (use the automatic warm-start) or an `(T_sub, n_endo)` array to use as the warm-start for that sub-solve. Rows are trimmed or padded to `T_sub` if needed. |

### Supplying per-sub-solve initial guesses (`sub_x0`)

By default each sub-solve is warm-started from the previous sub-solve's tail solution. This works well when sub-solve 1 is non-trivial, but can break down in the common pre-announcement pattern where sub-solve 1 is trivial (agents stay at the initial steady state) and sub-solve 2 must transition to a new steady state. In that case the automatic warm-start for sub-solve 2 (all `ss1`) can be far from the solution.

Use `sub_x0` to inject high-quality initial guesses directly:

```python
from pyperfectforesight import make_initial_guess

T_sub2 = T - learnt_in2 + 1
T_sub3 = T - learnt_in3 + 1

sub_x0 = [
    None,                                                             # sub-solve 1: trivial, auto warm-start is fine
    make_initial_guess(T_sub2, ss1_vec, ss2_vec, method='exponential'),  # sub-solve 2: news shock
    make_initial_guess(T_sub3, ss2_vec, ss1_vec, method='exponential'),  # sub-solve 3: disappointment
]

sol = solve_perfect_foresight_expectation_errors(
    T, X0, params_dict, ss1_vec, model_funcs, vars_dyn,
    news_shocks=news_shocks,
    sub_x0=sub_x0,
    initial_state=k_neg1,
)
```
