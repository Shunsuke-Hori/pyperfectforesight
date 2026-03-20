# DynareByPython

A minimal Dynare-style perfect foresight solver in Python. This package provides tools for solving dynamic economic models using perfect foresight methods, inspired by [Dynare](https://www.dynare.org/).

## Features

- **Symbolic equation processing**: Define models using SymPy symbolic math
- **Automatic differentiation**: Compute Jacobian blocks automatically
- **Sparse matrix support**: Efficient handling of large-scale models
- **Flexible configuration**: Customizable solver methods and options
- **Generic steady-state solver**: Numerical steady-state computation for any model
- **Clean API**: Simple, intuitive interface for model definition and solving

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

Here's a simple RBC (Real Business Cycle) model example:

```python
import sympy as sp
import numpy as np
from dynare_python import v, process_model, compute_steady_state_numerical, solve_perfect_foresight

# 1. Define model parameters
beta, delta, alpha = sp.symbols("beta delta alpha")
vars_dyn = ["c", "k"]

# 2. Define equations
c_0, c_p = v("c", 0), v("c", 1)
k_0, k_p = v("k", 0), v("k", 1)

eq_euler = 1/c_0 - beta*(alpha*k_p**(alpha-1) + (1-delta))/c_p
eq_kacc = k_p - (1-delta)*k_0 - k_0**alpha + c_0

equations = [eq_euler, eq_kacc]

# 3. Process model
model_funcs = process_model(equations, vars_dyn)

# 4. Compute steady state
params = {beta: 0.96, delta: 0.08, alpha: 0.36}
ss = compute_steady_state_numerical(equations, vars_dyn, params)

# 5. Solve transition path
T = 100
X0 = np.tile(ss, (T, 1))
X0[0, 1] = 0.9 * ss[1]  # 10% capital shock

sol = solve_perfect_foresight(T, X0, params, ss, model_funcs, vars_dyn)
print(f"Converged: {sol.success}")
```

### With Exogenous Variables

You can also specify exogenous variables with predetermined paths:

```python
# Define model with exogenous variables
vars_dyn = ["c", "k"]  # Endogenous
vars_exo = ["g"]       # Exogenous (e.g., government spending)

# Include exogenous variables in equations
g_0 = v("g", 0)
eq_kacc = k_p - (1-delta)*k_0 - k_0**alpha + c_0 + g_0

# Process model
model_funcs = process_model(equations, vars_dyn, vars_exo=vars_exo)

# Define exogenous path (T x n_exo array)
exog_path = np.zeros((T, 1))
exog_path[:, 0] = 0.2  # Baseline government spending
exog_path[10:30, 0] = 0.22  # Temporary increase

# Solve with exogenous path
sol = solve_perfect_foresight(T, X0, params, ss, model_funcs, vars_dyn,
                              exog_path=exog_path)
```

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

- **`v(name, lag)`**: Create time-indexed symbolic variables
- **`process_model(equations, vars_dyn, ...)`**: Process and compile model equations
- **`compute_steady_state_numerical(equations, vars_dyn, params_dict, ...)`**: Compute steady state numerically
- **`solve_perfect_foresight(T, X0, params_dict, ss, model_funcs, vars_dyn, ...)`**: Solve perfect foresight problem

### Low-level Functions

For advanced users who want more control:

- `lead_lag_incidence()`: Detect variable lags in equations
- `is_static()`, `eliminate_static()`: Handle static equations
- `local_blocks()`: Compute Jacobian blocks
- `residual()`, `sparse_jacobian()`: Build residuals and Jacobians
- `append_terminal_conditions()`: Add terminal steady-state constraints

## Configuration Options

### `process_model()` options:
- `vars_exo=None`: List of exogenous variable names
- `eliminate_static_vars=True/False`: Eliminate static variables
- `compiler='lambdify'`: Compilation backend (extensible)

### `solve_perfect_foresight()` options:
- `exog_path=None`: Exogenous variable path (T x n_exo array)
- `use_terminal_conditions=True/False`: Enforce terminal steady state
- `solver_options={}`: Sparse Newton solver options (keys: `maxiter`, `ftol`, `xtol`, `maxfev`)
- `method` *(deprecated)*: Previously selected the `scipy.optimize.root` backend; now ignored — the solver is always the built-in sparse Newton method

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
3. Run tests (when available):
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
