# DynareByPython

A minimal Dynare-style perfect foresight solver in Python. This package provides tools for solving dynamic economic models using perfect foresight methods, inspired by [Dynare](https://www.dynare.org/).

## Features

- **Dynare-style lag notation**: Write equations using `v("k", -1)` for lagged variables, matching Dynare's convention
- **Augmented-path BVP solver**: Stock/jump variable models use a boundary-value problem formulation — `initial_state` is the pre-period-0 value `k_{-1}`, and all period-0 variables are solved simultaneously
- **Symbolic equation processing**: Define models using SymPy symbolic math
- **Automatic differentiation**: Compute Jacobian blocks automatically
- **Sparse Newton solver**: Efficient sparse Jacobian and Newton iterations for large-scale models
- **Homotopy continuation**: `solve_perfect_foresight_homotopy` for large shocks that are hard to solve directly
- **Generic steady-state solver**: Numerical steady-state computation for any model
- **Auxiliary variable support**: Handle auxiliary (non-dynamic) variables via analytical substitution, dynamic augmentation, or nested numerical solving

## Installation

### From source (development)

1. Clone or download this repository
2. Set up the conda environment:
   ```bash
   bash setup_conda_env.sh
   conda activate dynare_python
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### With pip (when published)

```bash
pip install dynare-python
```

## Quick Start

Here's a simple RBC (Real Business Cycle) model in Dynare lag notation:

```python
import numpy as np
from dynare_python import v, process_model, solve_perfect_foresight

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

# Transition path: k starts 10% above steady state
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
from dynare_python import v, process_model, solve_perfect_foresight

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
from dynare_python import solve_perfect_foresight_homotopy

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

## Stock/Jump Variable Formulation

When `stock_var_indices` and `initial_state` are provided, the solver uses an **augmented-path BVP formulation**:

- An `initval` row (pre-period-0 values) is prepended and an `endval` row (terminal steady state) is appended to form a `T+2`-row augmented path.
- Residuals are evaluated at periods `t = 0, …, T-1` using the full augmented path, so all `T×n` unknowns (including period-0 jump variables) are determined simultaneously.
- `initial_state` is `k_{-1}` — the **pre-period-0** value of each stock variable, following Dynare's convention. Period-0 values of all variables (including jump variables like `c`) are solved by the model.
- `stock_var_indices` lists the column indices (into `vars_dyn`) of the stock (predetermined) variables.

This contrasts with the simple approach of pinning `X[0]` to given values, which over-constrains jump variables and can produce a structurally singular Jacobian.

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
dynare_python/
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

### Low-level Functions

For advanced users who want more control:

- `lead_lag_incidence()`: Detect variable lead/lag structure in equations
- `is_static()`, `eliminate_static()`: Handle static equations
- `local_blocks()`: Compute Jacobian blocks
- `residual()`, `sparse_jacobian()`: Build residuals and Jacobians (standard mode)
- `append_terminal_conditions()`: Add terminal steady-state constraints

## Configuration Options

### `process_model()` options:
- `vars_exo=None`: List of exogenous variable names
- `vars_aux=None`: List of auxiliary (non-dynamic) variable names
- `aux_method='auto'`: How to handle auxiliary variables: `'auto'`, `'analytical'`, `'dynamic'`, `'nested'`
- `eliminate_static_vars=True/False`: Eliminate static variables before solving

### `solve_perfect_foresight()` options:
- `exog_path=None`: Exogenous variable path (`T × n_exo` array)
- `initial_state=None`: Pre-period-0 values of stock variables (`k_{-1}` in Dynare convention); required when `stock_var_indices` is provided
- `stock_var_indices=None`: Column indices (into `vars_dyn`) of stock (predetermined) variables
- `ss_initial=None`: Steady-state values used for non-stock `initval` entries; defaults to `ss`
- `use_terminal_conditions=True/False`: Enforce terminal steady state (ignored in BVP mode — terminal condition is always enforced via `endval`)
- `solver_options={}`: Sparse Newton solver options (`maxiter`, `ftol`, `xtol`)
- `method` *(deprecated)*: Previously selected the `scipy.optimize.root` backend; now ignored

### `solve_perfect_foresight_homotopy()` options:
- All options from `solve_perfect_foresight()`, plus:
- `n_steps=10`: Number of homotopy steps (must be a positive integer)
- `exog_ss=None`: Baseline exogenous path at `λ=0`; defaults to zero
- `verbose=False`: Print progress at each homotopy step

## Requirements

- Python >= 3.8
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
3. Run tests:
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
@software{dynare_python,
  title={DynareByPython: A Minimal Dynare-style Perfect Foresight Solver in Python},
  author={Shunsuke Hori},
  year={2026},
  url={https://github.com/Shunsuke-Hori/pyperfectforesight}
}
```
