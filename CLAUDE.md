# CLAUDE.md

Project context and conventions for Claude Code.

## Project overview

`dynare_python` is a minimal Dynare-style perfect foresight solver in Python.
All model logic lives in `dynare_python/core.py`. There are no sub-packages.

## Key conventions

### Dynare lag notation
Equations are written with `v("k", -1)` for `k_{t-1}` (lag) and `v("c", 1)` for `c_{t+1}` (lead).
`v("x", 0)` is the current-period value.
This matches Dynare's convention: a stock variable `k` appears at lag -1 in the capital accumulation equation.

### `initial_state` semantics
The meaning of `initial_state` depends on whether `stock_var_indices` is provided:

- **Standard / overdetermined mode (`stock_var_indices is None`)**: `initial_state` is the full period-0 state vector `X[0]` (all endogenous variables at `t = 0`). The solver pins `X[0]` and finds a least-squares path over `X[1:T-1]`.
- **Stock/jump BVP mode (`stock_var_indices` provided)**: `initial_state` is the **pre-period-0** value of the stock variable(s) — i.e., `k_{-1}` in Dynare notation. Period-0 values of all variables (including jump variables) are solved simultaneously by the model. Do not confuse with `k_0` (the period-0 value, which is endogenous).

### BVP (augmented-path) formulation
The BVP branch is activated when **both** `stock_var_indices` and `initial_state` are provided (passing `stock_var_indices` without `initial_state` raises a `ValueError`). The solver builds a `T+2`-row augmented path:
- Row 0: `initval` (pre-period-0 boundary — stock vars from `initial_state`, others from `ss_initial`)
- Rows 1…T: the `T` free periods (solved)
- Row T+1: `endval` (terminal steady state)

Residuals are evaluated at `t = 0, …, T-1` using index `t + lag + 1`, clamped to `[0, T+1]` so that lags beyond the single boundary row reuse `initval`/`endval` (e.g. `k_{-2} = k_{-1} = initval`).
This gives a square `T×n` system.

Private helpers `_residual_bvp()` and `_jacobian_bvp()` implement this; the public `residual()` and `sparse_jacobian()` are standard-mode only.

### Arbitrary lag/lead support
`residual()`, `sparse_jacobian()`, `_residual_bvp()`, and `_jacobian_bvp()` derive lag sets dynamically from `all_syms` via `_resolve_lag_sets()` / `_compute_lag_sets()`. Models with `|lag| > 1` are supported; in BVP mode a `UserWarning` is emitted because pre-sample values beyond `initval` are clamped.

## Repository layout

```
dynare_python/
├── __init__.py         # Public API exports
├── __version__.py      # Version string
└── core.py             # All model/solver logic

examples/               # Standalone demo scripts
AUXILIARY_VARIABLES.md  # Docs for auxiliary variable handling
README.md               # User-facing documentation
CLAUDE.md               # This file
```

## Testing

Tests live in the repo root (not in a `tests/` directory):

```
test_homotopy.py           # solve_perfect_foresight and solve_perfect_foresight_homotopy tests
test_arbitrary_lags.py     # arbitrary lag/lead support tests
test_auto_to_dynamic.py    # aux variable auto→dynamic fallback
test_dynamic_fallback.py   # aux variable dynamic method
test_methods_comparison.py # comparison script (runs as a script, not pytest)
```

Run all pytest-compatible tests:
```bash
pytest test_homotopy.py test_arbitrary_lags.py test_auto_to_dynamic.py test_dynamic_fallback.py
```

## Branching convention

Use a separate branch per concern:
- `feat/<name>` for new features
- `fix/<name>` for bug fixes
- `docs/<name>` for documentation-only changes

Do not mix documentation changes, fixes, and features on the same branch.
