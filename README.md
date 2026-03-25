# DynareByPython

A minimal Dynare-style perfect foresight solver in Python. This package provides tools for solving dynamic economic models using perfect foresight methods, inspired by [Dynare](https://www.dynare.org/).

## Features

- **Dynare-style lag notation**: Write equations using `v("k", -1)` for lagged variables, matching Dynare's convention
- **Augmented-path BVP solver**: Stock/jump variable models use a boundary-value problem formulation — `initial_state` is the pre-period-0 value `k_{-1}`, and all period-0 variables are solved simultaneously
- **Symbolic equation processing**: Define models using SymPy symbolic math
- **Automatic differentiation**: Compute Jacobian blocks automatically
- **Sparse Newton solver**: Efficient sparse Jacobian and Newton iterations for large-scale models
- **Homotopy continuation**: `solve_perfect_foresight_homotopy` for large shocks that are hard to solve directly
- **Expectation-errors solver**: `solve_perfect_foresight_expectation_errors` replicates Dynare's `perfect_foresight_with_expectation_errors_solver` — agents are surprised at multiple `learnt_in` periods and the full path is stitched from sub-simulations
- **Generic steady-state solver**: Numerical steady-state computation for any model
- **Auxiliary variable support**: Handle auxiliary (non-dynamic) variables via analytical substitution, dynamic augmentation, or nested numerical solving

## Installation

### From source (development)

1. Clone or download this repository
2. Set up the conda environment:
   ```bash
   bash setup_conda_env.sh
   conda activate pyperfectforesight
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### With pip (when published)

```bash
pip install pyperfectforesight
```

## Quick Start

Here's a simple RBC (Real Business Cycle) model in Dynare lag notation:

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

# Transition path: k_{-1} (pre-period-0 capital) starts 10% above steady state
T = 100
X0 = np.tile(ss, (T, 1))          # warm-start: constant ss path

# initial_state = k_{-1} (pre-period-0 capital, Dynare convention)
k_neg1 = np.array([K_SS * 1.1])

sol = solve_perfect_foresight(
    T, X0, {}, ss, model_funcs, vars_dyn,
    initial_state=k_neg1,
    stock_var_indices=[1],         # index of k in vars_dyn
)
print(f"Converged: {sol.success}")
```

### With Exogenous Variables

```python
import sympy as sp
import numpy as np
from pyperfectforesight import v, process_model, solve_perfect_foresight

ALPHA, BETA = 0.36, 0.99

# TFP shock z enters the capital accumulation equation
eq_euler = v("c", 0)**(-1) - BETA * ALPHA * v("k", 0)**(ALPHA-1) * v("c", 1)**(-1)
eq_kacc  = v("k", 0) - sp.exp(v("z", 0)) * v("k", -1)**ALPHA + v("c", 0)

vars_dyn = ["c", "k"]
model_funcs = process_model([eq_euler, eq_kacc], vars_dyn, vars_exo=["z"])

K_SS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
C_SS = K_SS**ALPHA - K_SS
ss = np.array([C_SS, K_SS])

T = 100
X0 = np.tile(ss, (T, 1))

# AR(1) TFP shock: 1% on impact, rho=0.9 decay
rho = 0.9
exog = np.zeros((T, 1))
exog[0, 0] = 0.01
for t in range(1, T):
    exog[t, 0] = rho * exog[t-1, 0]

k_neg1 = np.array([K_SS])   # k_{-1} starts at steady state

sol = solve_perfect_foresight(
    T, X0, {}, ss, model_funcs, vars_dyn,
    initial_state=k_neg1,
    stock_var_indices=[1],
    exog_path=exog,
)
```

### Homotopy for Large Shocks

When direct Newton fails to converge for large shocks, use homotopy continuation:

```python
from pyperfectforesight import solve_perfect_foresight_homotopy

# Same model setup as above...
k_neg1 = np.array([K_SS * 1.5])   # 50% above steady state

sol = solve_perfect_foresight_homotopy(
    T, X0, {}, ss, model_funcs, vars_dyn,
    initial_state=k_neg1,
    stock_var_indices=[1],
    n_steps=10,       # number of homotopy steps from ss to full shock
    verbose=True,
)
print(f"Converged: {sol.success}")
```

### Multiple Surprise Shocks (Expectation Errors)

Replicates Dynare's `perfect_foresight_with_expectation_errors_solver`. Agents are surprised at each `learnt_in` period and re-solve from that point forward; the full path is stitched from the sub-simulations.

```python
import numpy as np
from pyperfectforesight import solve_perfect_foresight_expectation_errors

# Same RBC model with exogenous TFP z...
# Agents initially expect no shock (period 1), then learn of a
# persistent TFP shock at period 3.
T = 100

# exog_path for learnt_in=3: row 0 = period 3, row 1 = period 4, …
# (T rows is fine; the solver uses only the first T - learnt_in + 1 = T-2 rows)
exog_surprise = np.full((T, 1), 0.01)   # permanent shock from period 3 onward

news_shocks = [
    (1, None),                  # period 1: baseline, no shock expected
    (3, exog_surprise),         # period 3: agents learn of permanent shock
]

sol = solve_perfect_foresight_expectation_errors(
    T, X0, {}, ss, model_funcs, vars_dyn,
    news_shocks=news_shocks,
)
print(f"Converged: {sol.success}, message: {sol.message}")
X_full = sol.x.reshape(T, -1)   # (T, n_endo) stitched path
```

Each entry in `news_shocks` is a 2-tuple `(learnt_in, exog_path)` or a 3-tuple `(learnt_in, exog_path, endval)`. The optional `endval` mirrors Dynare's `endval(learnt_in=k)` block for permanent shocks that change the terminal steady state. An `endval` in a 3-tuple applies to that sub-solve and remains the terminal boundary for all later segments unless overridden again by another 3-tuple. The list must be sorted by `learnt_in` and the first entry must have `learnt_in=1`.

## Stock/Jump Variable Formulation

The solver always uses an **augmented-path BVP (boundary value problem) formulation**:

- An `initval` boundary row is prepended and an `endval` row (terminal steady state) is appended to form a `T+2`-row augmented path. The `initval` row holds pre-period-0 values for stock variables (from `initial_state`) and steady-state values for jump variables (from `ss_initial`); the `t=-1` entries for jump variables are not economically meaningful since jump variables have no negative-lag appearances by definition.
- Residuals are evaluated at periods `t = 0, …, T-1` using the full augmented path, so all `T×n` unknowns (including period-0 jump variables) are determined simultaneously.
- `initial_state` provides `k_{-1}` — the **pre-period-0** value for each **stock** variable, following Dynare's convention. Period-0 values of all variables (including jump variables like `c`) are solved by the model.
- `stock_var_indices` is inferred automatically from the lead-lag incidence table: variables that appear at any negative lag are classified as stock (predetermined); all others are jump variables free to respond at `t=0`. You can also pass it explicitly to override the inference.

This correctly handles jump variables — pinning `X[0]` directly would over-constrain them and produce a structurally singular Jacobian.

## Examples

See the `examples/` directory for complete examples:

- `rbc_demo.py`: Basic RBC model with capital shock
- `rbc_with_government.py`: RBC model with exogenous government spending shocks

Run the examples:
```bash
python examples/rbc_demo.py
python examples/rbc_with_government.py
```

## Package Structure

```
pyperfectforesight/
├── __init__.py       # Package exports
├── __version__.py    # Version information
└── core.py          # Core functionality
```

### Main Functions

- **`v(name, lag)`**: Create a time-indexed symbolic variable (e.g. `v("k", -1)` for `k_{t-1}`)
- **`process_model(equations, vars_dyn, ...)`**: Process and compile model equations
- **`compute_steady_state_numerical(equations, vars_dyn, params_dict, ...)`**: Compute steady state numerically
- **`solve_perfect_foresight(T, X0, params_dict, ss, model_funcs, vars_dyn, ...)`**: Solve perfect foresight transition path
- **`solve_perfect_foresight_homotopy(T, X0, params_dict, ss, model_funcs, vars_dyn, ...)`**: Homotopy continuation for difficult shocks
- **`solve_perfect_foresight_expectation_errors(T, X0, params_dict, ss, model_funcs, vars_dyn, news_shocks, ...)`**: Multiple surprise (MIT) shocks — replicates Dynare's expectation-errors solver

### Low-level Functions

For advanced users who want more control:

- `lead_lag_incidence()`: Detect variable lead/lag structure in equations
- `is_static()`, `eliminate_static()`: Handle static equations
- `local_blocks()`: Compute Jacobian blocks
- `residual()`, `sparse_jacobian()`: Build residuals and Jacobians
- `append_terminal_conditions()` *(legacy)*: Manually append terminal-condition rows; retained for backward compatibility but not needed when using the high-level solvers

## Configuration Options

### `process_model()` options:
- `vars_exo=None`: List of exogenous variable names
- `vars_aux=None`: List of auxiliary (non-dynamic) variable names
- `aux_method='auto'`: How to handle auxiliary variables: `'auto'`, `'analytical'`, `'dynamic'`, `'nested'`
- `eliminate_static_vars=True/False`: Eliminate static variables before solving

### `solve_perfect_foresight()` options:
- `exog_path=None`: Exogenous variable path (`T × n_exo` array)
- `initial_state=None`: Pre-period-0 values of stock variables (`k_{-1}` in Dynare convention); defaults to `ss_initial[stock_var_indices]` (economy starts at steady state)
- `stock_var_indices=None`: Column indices (into `vars_dyn`) of stock (predetermined) variables; inferred from the lead-lag incidence table when not provided
- `ss_initial=None`: Initial steady-state values used for the `initval` boundary row; defaults to `ss`
- `endval=None`: Override the terminal steady state (right BVP boundary); defaults to `ss`. Use this for permanent shocks that shift the long-run equilibrium.
- `solver_options=None`: Sparse Newton solver options (treated as `{}` when `None`; supports `maxiter`, `ftol`, `xtol`, `maxfev`)
- `method` *(deprecated)*: Previously selected the `scipy.optimize.root` backend; now ignored

### `solve_perfect_foresight_homotopy()` options:
- All options from `solve_perfect_foresight()`, plus:
- `n_steps=10`: Number of homotopy steps (must be a positive integer)
- `exog_ss=None`: Baseline exogenous path at `λ=0`; defaults to zero
- `verbose=False`: Print progress at each homotopy step

### `solve_perfect_foresight_expectation_errors()` options:
- `news_shocks`: List of 2-tuples `(learnt_in, exog_path)` or 3-tuples `(learnt_in, exog_path, endval)`. Must be sorted by `learnt_in`; first entry must have `learnt_in=1`. Each `exog_path` is the belief path **indexed from period `learnt_in`**: row 0 = period `learnt_in`, row 1 = period `learnt_in+1`, etc. Do **not** pre-offset it as if row 0 were period 1; the solver handles that alignment internally. When `constant_simulation_length=False` (default), at least `T - learnt_in + 1` rows are required; longer paths (including a full `T`-row array) are accepted and extra rows are ignored. `exog_path=None` passes an all-zero path (only correct when the exogenous steady state is zero).
- `initial_state=None`, `ss_initial=None`, `stock_var_indices=None`: Same semantics as `solve_perfect_foresight()`
- `constant_simulation_length=False`: If `False` (Dynare default), each sub-solve uses the shrinking horizon `T - learnt_in + 1`. If `True` (Dynare's `constant_simulation_length` option), every sub-solve runs for the full `T` periods.
- `solver_options=None`: Forwarded to each sub-solve (same keys as `solve_perfect_foresight()`)

## Requirements

- Python >= 3.9
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- SymPy >= 1.9.0
- Matplotlib >= 3.3.0 (for examples)

## Development

To contribute or modify:

1. Clone the repository
2. Install in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests (all test files are in `tests/`):
   ```bash
   pytest
   ```

## License

MIT License

## Acknowledgments

Inspired by [Dynare](https://www.dynare.org/), the reference platform for solving dynamic economic models.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{pyperfectforesight,
  title={pyperfectforesight: A Minimal Dynare-style Perfect Foresight Solver in Python},
  author={Shunsuke Hori},
  year={2026},
  url={https://github.com/Shunsuke-Hori/pyperfectforesight}
}
```
