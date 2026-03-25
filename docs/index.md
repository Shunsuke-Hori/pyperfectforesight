# pyperfectforesight

A minimal [Dynare](https://www.dynare.org/)-style perfect foresight solver in Python. `pyperfectforesight` lets you define dynamic economic models with Dynare's familiar lag notation, then solve for the perfect foresight transition path using a sparse Newton BVP solver. It supports stock/jump variable models, homotopy continuation for large shocks, and Dynare's expectation-errors (multiple surprise MIT shocks) protocol — all from pure Python with NumPy, SciPy, and SymPy.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
getting-started
solvers
initial-guess
auxiliary-variables
api-reference
```

## Features

- **Dynare lag notation** — write `v("k", -1)` for $k_{t-1}$ and `v("c", 1)` for $c_{t+1}$, matching Dynare's convention exactly
- **Augmented-path BVP solver** — pre-period-0 `initial_state` (`k_{-1}`) is pinned; all period-0 variables including jump variables are solved simultaneously
- **Automatic stock-variable inference** — lead-lag incidence detects predetermined variables; no manual classification needed
- **Homotopy continuation** — `solve_perfect_foresight_homotopy` for large shocks that defeat direct Newton
- **Expectation-errors solver** — replicates Dynare's `perfect_foresight_with_expectation_errors_solver` for sequences of surprise shocks
- **Auxiliary variable support** — four methods (`auto`, `analytical`, `dynamic`, `nested`) to handle static/auxiliary variables
- **Symbolic processing + automatic differentiation** — models defined via SymPy; sparse Jacobians computed automatically
