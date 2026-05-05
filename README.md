# pyperfectforesight

A minimal Dynare-style perfect foresight solver in Python.

**[Documentation](https://shunsuke-hori.github.io/pyperfectforesight/)**

## Quick Start

Use the builder DSL: declare variables and parameters on a `Model` instance, write equations using attribute access, then call `build()`.

```python
import numpy as np
from pyperfectforesight import Model

m = Model()
m.endog("k c")
m.exog("z")
m.params("alpha beta")

# Dynare lag/lead notation: m.k[-1] = k_{t-1}, m.c[1] = c_{t+1}, m.z[1] = z_{t+1}
eq_euler = 1/m.c[0] - m.beta * m.alpha * m.z[1] * m.k[0]**(m.alpha - 1) / m.c[1]
eq_kacc  = m.k[0] - m.z[0] * m.k[-1]**m.alpha + m.c[0]
m.build([eq_euler, eq_kacc])

PARAMS = {m.alpha: 0.36, m.beta: 0.99}

# Steady state at z=1
ss = m.steady_state(PARAMS, exog_ss=np.array([1.0]))

T = 100
exog_path = np.full((T, 1), 1.05)   # permanent 5% TFP increase

# Compute terminal steady state at new exogenous level
ss_terminal = m.steady_state(PARAMS, exog_ss=np.array([1.05]))

sol = m.solve(T, PARAMS,
              endval=ss_terminal,
              ss_initial=ss,          # on-SS start: economy was at ss before shock
              exog_path=exog_path)

print(f"Converged: {sol.success}")
X = sol.x.reshape(T, -1)   # shape (T, 2): columns are [c, k]
```

For homotopy continuation, expectation-errors (news shocks), the functional API, and full option reference, see the **[documentation](https://shunsuke-hori.github.io/pyperfectforesight/)**.

## Features

- **Object-oriented API**: `Model` class — declare the model once, call `model.solve()`, `model.solve_homotopy()`, `model.solve_expectation_errors()`, and `model.steady_state()` without repeating bookkeeping
- **Dynare-style lag notation**: Write equations using `v("k", -1)` for lagged variables, matching Dynare's convention exactly
- **Explicit terminal steady state**: Pass the pre-computed terminal SS as `endval`; use `solve_steady_state()` to compute it at any exogenous level
- **Augmented-path BVP solver**: `ss_initial` or `initial_state` sets the pre-period-0 boundary; all period-0 variables including jump variables are solved simultaneously
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
git clone https://github.com/Shunsuke-Hori/pyperfectforesight.git
cd pyperfectforesight
pip install -e .
```

Once published to PyPI: `pip install pyperfectforesight`

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
