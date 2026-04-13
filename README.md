# pyperfectforesight

A minimal Dynare-style perfect foresight solver in Python.

**[Documentation](https://shunsuke-hori.github.io/pyperfectforesight/)**

## Quick Start

Declare the model once with the `Model` class, compute the steady state, solve.

```python
import numpy as np
from pyperfectforesight import p, v, Model

ALPHA = p("alpha")
BETA  = p("beta")
PARAMS = {ALPHA: 0.36, BETA: 0.99}

# Dynare lag notation: v("k", -1) = k_{t-1}, v("c", 1) = c_{t+1}
eq_euler = 1/v("c", 0) - BETA * ALPHA * v("z", 1) * v("k", 0)**(ALPHA-1) / v("c", 1)
eq_kacc  = v("k", 0) - v("z", 0) * v("k", -1)**ALPHA + v("c", 0)

model = Model([eq_euler, eq_kacc], ["c", "k"],
              vars_exo=["z"], vars_params=["alpha", "beta"])

# Steady state at z=1
ss = model.steady_state(PARAMS, exog_ss=np.array([1.0]))

T = 100
exog_path = np.full((T, 1), 1.05)   # permanent 5% TFP increase

# Terminal steady state is auto-computed from exog_path[-1]
sol = model.solve(T, PARAMS, ss,
                  exog_path=exog_path,
                  initial_state=np.array([ss[1]]))

print(f"Converged: {sol.success}")
X = sol.x.reshape(T, -1)   # shape (T, 2): columns are [c, k]
```

For homotopy continuation, expectation-errors (news shocks), the functional API, and full option reference, see the **[documentation](https://shunsuke-hori.github.io/pyperfectforesight/)**.

## Features

- **Object-oriented API**: `Model` class — declare the model once, call `model.solve()`, `model.solve_homotopy()`, `model.solve_expectation_errors()`, and `model.steady_state()` without repeating bookkeeping
- **Dynare-style lag notation**: Write equations using `v("k", -1)` for lagged variables, matching Dynare's convention exactly
- **Automatic terminal steady-state computation**: Pass `compiled_ss` to any solver and omit `endval` — the terminal boundary is computed by solving for the steady state at `exog_path[-1]`
- **Augmented-path BVP solver**: `initial_state` is the pre-period-0 value `k_{-1}`; all period-0 variables including jump variables are solved simultaneously
- **Sparse Newton solver**: Efficient sparse Jacobian and Newton iterations
- **Homotopy continuation**: `solve_perfect_foresight_homotopy` for large shocks that defeat direct Newton
- **Expectation-errors solver**: Replicates Dynare's `perfect_foresight_with_expectation_errors_solver` — multiple surprise MIT shocks, path stitched from sub-simulations
- **Symbolic equation processing + automatic differentiation**: Models defined via SymPy; Jacobians computed automatically
- **Auxiliary variable support**: Handle static/auxiliary variables via analytical substitution, dynamic augmentation, or nested numerical solving

## Performance

pyperfectforesight is **~23–61× faster** than Dynare 6.2 on the same RBC model, measured on the solver step alone (excludes one-time compilation/setup on both sides).

![Benchmark: pyperfectforesight vs Dynare 6.2](docs/benchmark_plot.png)

| Horizon T | Python (ms) | Dynare (ms) | Speedup |
|----------:|------------:|------------:|--------:|
|        50 |        0.86 |       19.55 |  22.7×  |
|       100 |        0.82 |       31.08 |  37.9×  |
|       200 |        1.06 |       50.95 |  48.1×  |
|       500 |        2.10 |       99.75 |  47.4×  |
|      1000 |        3.45 |      211.79 |  61.4×  |

*RBC model, 3 variables, one-time TFP shock. Median of 20 runs each. Solver only.*

To reproduce: install dev extras first (`pip install -e ".[dev]"`), then run `python scripts/benchmark.py --dynare --plot` (requires MATLAB + Dynare 6.2). Omit `--dynare` to use the saved Dynare CSV already in the repo.

## Why pyperfectforesight?

### vs Dynare

Dynare is the reference platform and pyperfectforesight is validated against it (results agree to ~1e-10 on the same models). The reasons to use this package instead:

- **No MATLAB required.** Dynare requires MATLAB (commercial) or Octave. pyperfectforesight is pure Python — `pip install` and go.
- **Python-native workflow.** Equations are SymPy expressions. Results are NumPy arrays. No `.mod` files, no separate toolchain — the model lives in the same script as the analysis.
- **Programmatic.** Parameter sweeps, Monte Carlo, IRF grids: write a loop. In Dynare you would need to script around MATLAB/Octave.
- **Faster.** ~23–61× faster than Dynare on the same RBC model (see [Performance](#performance) above).

### vs dolo

[dolo](https://github.com/EconForge/dolo) is a Python DSGE toolkit with a `deterministic_solve` function. The key differences:

- **dolo has hidden shifts in the exogenous path.** Its `_shocks_to_epsilons` function silently drops `shocks[0]` and maps `epsilons[t] = shocks[t+1]`. YAML `transition` equations then add a second shift for state variables. The shock you supply at index `t` lands at simulation period `t+1` or `t+2` depending on variable declaration — with no warning.
- **pyperfectforesight uses direct timing.** `exog_path[t]` is the exogenous value at period `t`, matching Dynare's convention exactly.
- **Expectation-errors solver.** pyperfectforesight replicates Dynare's `perfect_foresight_with_expectation_errors_solver`. dolo has no equivalent.

## Installation

```bash
pip install pyperfectforesight
```

From source (development):

```bash
git clone https://github.com/Shunsuke-Hori/pyperfectforesight.git
cd pyperfectforesight
pip install -e ".[dev]"
```

## Examples

See the `examples/` directory for complete runnable scripts:

- `rbc_demo.py`: Basic RBC model with capital shock
- `rbc_with_government.py`: RBC with exogenous government spending
- `rbc_with_investment.py`: RBC with auxiliary variables (investment ratio)
- `rbc_taxes.py`: RBC with labor, government spending, and AR(1) TFP
- `gali_2015_zlb.py`: Optimal monetary policy at the ZLB (Gali 2015, Ch. 5.4.2)

## Development

```bash
git clone https://github.com/Shunsuke-Hori/pyperfectforesight.git
cd pyperfectforesight
pip install -e ".[dev]"
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
