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
`initial_state` is the **pre-period-0** value of the stock variable(s) — i.e., `k_{-1}` in Dynare notation.
Period-0 values of all variables (including jump variables) are solved simultaneously by the model.
Do not confuse with `k_0` (the period-0 value, which is endogenous).

### BVP (augmented-path) formulation
When `stock_var_indices` is provided, the solver builds a `T+2`-row augmented path:
- Row 0: `initval` (pre-period-0 boundary — stock vars from `initial_state`, others from `ss_initial`)
- Rows 1…T: the `T` free periods (solved)
- Row T+1: `endval` (terminal steady state)

Residuals are evaluated at `t = 0, …, T-1` using exact index `t + lag + 1`, with no boundary clamping.
This gives a square `T×n` system.

Private helpers `_residual_bvp()` and `_jacobian_bvp()` implement this; the public `residual()` and `sparse_jacobian()` are standard-mode only.

### Hard-coded lag limitation (known issue)
`residual()`, `sparse_jacobian()`, `_residual_bvp()`, and `_jacobian_bvp()` all hard-code `for lag in [-1, 0, 1]` when building substitution dicts.
Models with `|lag| > 1` will raise a `KeyError`. Fixing this is tracked in `feat/arbitrary-lags`.

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
test_homotopy.py           # solve_perfect_foresight_homotopy tests
test_auto_to_dynamic.py    # aux variable auto→dynamic fallback
test_dynamic_fallback.py   # aux variable dynamic method
test_methods_comparison.py # comparison script (runs as a script, not pytest)
```

Run all pytest-compatible tests:
```bash
pytest test_homotopy.py test_auto_to_dynamic.py test_dynamic_fallback.py
```

## Branching convention

Use a separate branch per concern:
- `feat/<name>` for new features
- `fix/<name>` for bug fixes
- `docs/<name>` for documentation-only changes

Do not mix documentation changes, fixes, and features on the same branch.

## Active branches / planned work

- `feat/arbitrary-lags` — extend `residual()`, `sparse_jacobian()`, `_residual_bvp()`, `_jacobian_bvp()` to support `|lag| > 1` by deriving lags dynamically from `all_syms` via `_parse_time_symbol()` instead of hard-coding `[-1, 0, 1]`.
