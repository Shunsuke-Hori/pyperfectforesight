# Solvers

`pyperfectforesight` provides three high-level solver functions, each targeting a different use case.

---

## `solve_perfect_foresight`

The core solver. Given a model, steady state, and initial condition, it finds the perfect foresight transition path by solving the $T \times n$ BVP system with a sparse Newton method.

### Basic usage

```python
from pyperfectforesight import solve_perfect_foresight

sol = solve_perfect_foresight(
    T, params_dict, ss, model_funcs, vars_dyn,
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
| `endval` | `None` | Terminal steady state (right BVP boundary). If `None` and `compiled_ss` is provided and `exog_path` is not `None`, automatically computed from `exog_path[-1]`. Otherwise defaults to `ss`. Pass a pre-computed value for repeated simulations to avoid recomputation. |
| `compiled_ss` | `None` | Pre-compiled steady-state bundle from `compile_steady_state_funcs()`. Enables automatic `endval` computation from the terminal exogenous level (see the *Terminal steady state* section below). |
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
    T, {}, ss, model_funcs, vars_dyn,
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
| `compiled_ss` | `None` | Same as `solve_perfect_foresight`: enables auto-computation of `endval` from `exog_path[-1]`. The computed `endval` is then interpolated across homotopy steps from `ss_initial` to the terminal value. |

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

When `compiled_ss` is provided, the same persistence rule applies to auto-computed `endval`s: each segment with an `exog_path` automatically updates the terminal boundary from that path's last row; a segment without an `exog_path` reuses the previous segment's value.

### Usage example

```python
import numpy as np
from pyperfectforesight import solve_perfect_foresight_expectation_errors

# Same RBC model with exogenous TFP z as in Getting Started.
T = 100

# Agents initially expect no shock (period 1).
# At period 3 they learn of a permanent 1% TFP shock.
exog_surprise = np.full((T, 1), 0.01)   # permanent shock from period 3 onward

news_shocks = [
    (1, None),               # period 1: baseline, no shock expected
    (3, exog_surprise),      # period 3: agents learn of permanent TFP shock
]

sol = solve_perfect_foresight_expectation_errors(
    T, {}, ss, model_funcs, vars_dyn, news_shocks,
    initial_state=k_neg1,
    stock_var_indices=[1],
)
print(f"Converged: {sol.success}, message: {sol.message}")

X_full = sol.x.reshape(T, -1)   # (T, n_endo) stitched path
```

### Example with changing terminal steady state

When the shock is permanent and shifts the long-run equilibrium, you can either pass the new steady state explicitly in a 3-tuple, or let `compiled_ss` compute it automatically from `exog_path[-1]`.

**Option A — explicit `endval` in a 3-tuple** (useful when the terminal steady state is pre-computed):

```python
ss_new = np.array([C_SS_NEW, K_SS_NEW])  # new steady state under permanent shock

news_shocks = [
    (1, None),
    (3, exog_surprise, ss_new),   # 3-tuple: endval takes effect from period 3
]

sol = solve_perfect_foresight_expectation_errors(
    T, {}, ss_new, model_funcs, vars_dyn, news_shocks,
    initial_state=k_neg1,
    stock_var_indices=[1],
)
```

**Option B — automatic via `compiled_ss`** (endval computed from `exog_surprise[-1]`):

```python
from pyperfectforesight import compile_steady_state_funcs, solve_steady_state

compiled_ss = compile_steady_state_funcs(equations, vars_dyn, vars_exo=['z'])
ss_initial = solve_steady_state(compiled_ss, params)  # at z=0

news_shocks = [
    (1, None),           # 2-tuple: no exog_path, keeps current endval (ss_initial)
    (3, exog_surprise),  # 2-tuple: endval auto-computed from exog_surprise[-1]
]

sol = solve_perfect_foresight_expectation_errors(
    T, params, ss_initial, model_funcs, vars_dyn, news_shocks,
    initial_state=k_neg1,
    stock_var_indices=[1],
    compiled_ss=compiled_ss,   # enables auto-computation
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
| `sub_x0` | `None` | Per-sub-solve initial guesses. A list or tuple of the same length as `news_shocks`; each entry is either `None` (use the automatic warm-start) or an `(T_sub, n_endo)` array to use as the warm-start for that sub-solve. Rows are trimmed or padded to `T_sub` if needed. |
| `compiled_ss` | `None` | Pre-compiled steady-state bundle from `compile_steady_state_funcs()`. When provided, `endval` for each sub-solve is automatically computed from the last row of that segment's `exog_path`, unless the 3-tuple already supplies an explicit `endval`. Persists across segments. |

### Supplying per-sub-solve initial guesses (`sub_x0`)

By default each sub-solve is warm-started from the previous sub-solve's tail solution. This works well when sub-solve 1 is non-trivial, but can break down in the common pre-announcement pattern where sub-solve 1 is trivial (agents stay at the initial steady state) and sub-solve 2 must transition to a new steady state. In that case the automatic warm-start for sub-solve 2 (all `ss1`) can be far from the solution.

Use `sub_x0` to inject high-quality initial guesses directly:

```python
from pyperfectforesight import make_initial_guess, solve_perfect_foresight_expectation_errors

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

sol = solve_perfect_foresight_expectation_errors(
    T, params_dict, ss1_vec, model_funcs, vars_dyn, news_shocks,
    sub_x0=sub_x0,
    initial_state=k_neg1,
)
```

---

## Terminal steady state

For permanent shocks that shift the long-run equilibrium, the terminal steady state must be consistent with the terminal exogenous level.  `compile_steady_state_funcs` + `solve_steady_state` compute it at any exogenous level; the result is a `SteadyState` object that is transparently usable as a numpy array.

### `SteadyState`

```python
import numpy as np
from pyperfectforesight import compile_steady_state_funcs, solve_steady_state

compiled_ss = compile_steady_state_funcs(equations, vars_dyn, vars_exo=['z'])

ss_initial  = solve_steady_state(compiled_ss, params, exog_ss=np.array([0.0]))
ss_terminal = solve_steady_state(compiled_ss, params, exog_ss=np.array([0.05]))

print(ss_terminal)
# SteadyState(values={c: 2.972, k: 40.999},
#             params={alpha: 0.36, beta: 0.99, delta: 0.025},
#             exog_ss={z: 0.05})

# Access provenance at any time
ss_terminal.values    # endogenous values as ndarray
ss_terminal.params    # {'alpha': 0.36, ...}
ss_terminal.exog_ss   # array([0.05])
ss_terminal.vars_exo  # ['z']
```

`SteadyState` is a drop-in replacement for any plain `ndarray` used as `ss`, `ss_initial`, or `endval`.

### Auto-computing `endval` from `exog_path[-1]`

Pass `compiled_ss` to any solver and omit `endval` — the terminal boundary is computed automatically from the last row of `exog_path`, guaranteeing a valid steady state:

```python
T = 100
exog_path = np.full((T, 1), 0.05)  # permanent shock

sol = solve_perfect_foresight(
    T, params, ss_terminal, model_funcs, vars_dyn,
    exog_path=exog_path, ss_initial=ss_initial,
    compiled_ss=compiled_ss,          # endval auto-computed from exog_path[-1]
)
```

For repeated simulations with the same terminal exogenous level, pass `endval` explicitly to skip the steady-state solve:

```python
for shock in shock_list:
    sol = solve_perfect_foresight(
        T, params, ss_terminal, model_funcs, vars_dyn,
        exog_path=shock, ss_initial=ss_initial,
        endval=ss_terminal,   # already computed, no recomputation needed
    )
```
