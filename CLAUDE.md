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
`initial_state` is always the **pre-period-0** value of the stock variable(s) — i.e., `k_{-1}` in Dynare notation. Period-0 values of all variables (including jump variables) are solved simultaneously by the model. Do not confuse it with `k_0` (the period-0 value, which is endogenous).

If `initial_state` is omitted it defaults to `ss_initial[stock_var_indices]` (economy starts at the initial steady state).

### Stock variable inference
`stock_var_indices` is inferred automatically via `_infer_stock_var_indices(model_funcs, vars_dyn)` when not provided by the caller. The helper reads `model_funcs['incidence']` and returns the indices of variables that appear at any negative lag. Jump variables (no negative-lag appearances) are free to respond at `t=0`. The caller can always pass `stock_var_indices` explicitly to override the inference.

### BVP (augmented-path) formulation
The solver **always** uses the augmented-path BVP formulation. The solver builds a `T+2`-row augmented path:
- Row 0: `initval` (pre-period-0 boundary — stock vars from `initial_state`, others from `ss_initial`)
- Rows 1…T: the `T` free periods (solved)
- Row T+1: `endval` (terminal steady state = `ss`)

Residuals are evaluated at `t = 0, …, T-1` using index `t + lag + 1`, clamped to `[0, T+1]` so that lags beyond the single boundary row reuse `initval`/`endval` (e.g. `k_{-2} = k_{-1} = initval`).
This gives a square `T×n` system.

Private helpers `_residual_bvp()` and `_jacobian_bvp()` implement this. `use_terminal_conditions` has been removed — the solver always appends a fixed `endval` boundary row, which enforces the terminal condition when leads make it bind.

### Arbitrary lag/lead support
`residual()`, `sparse_jacobian()`, `_residual_bvp()`, and `_jacobian_bvp()` derive lag sets dynamically from `all_syms` via `_resolve_lag_sets()` / `_compute_lag_sets()`. Models with `|lag| > 1` are supported; in BVP mode a `UserWarning` is emitted because pre-sample values beyond `initval` are clamped.

### `endval` parameter
`solve_perfect_foresight()` accepts an `endval` keyword argument to override the terminal BVP boundary (right-hand steady state). Defaults to `ss`. Use for permanent shocks where the long-run equilibrium differs from the initial steady state.

### Expectation-errors solver
`solve_perfect_foresight_expectation_errors()` replicates Dynare's `perfect_foresight_with_expectation_errors_setup` / `perfect_foresight_with_expectation_errors_solver`. It accepts a `news_shocks` list of 2- or 3-tuples `(learnt_in, exog_path[, endval])`. At each `learnt_in` the solver re-solves from that period forward and stitches the pieces together. Key design points:
- `learnt_in` is 1-indexed; the list must be sorted and start with `learnt_in=1`.
- The 3-tuple `endval` overrides the terminal steady state from that segment onward (mirrors Dynare's `endval(learnt_in=k)`).
- `constant_simulation_length=False` (default) uses a shrinking horizon `T - learnt_in + 1`; `True` uses the full `T` every sub-solve.
- The returned `OptimizeResult` includes `success`, `status` (1/0), `message`, `sub_results` (per-segment), `x_aux`, and `vars_aux`.
- `exog_path=None` passes an all-zero path; only correct when the exogenous steady state is zero.

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
test_homotopy.py                    # solve_perfect_foresight and solve_perfect_foresight_homotopy tests
test_arbitrary_lags.py              # arbitrary lag/lead support tests
test_auto_to_dynamic.py             # aux variable auto→dynamic fallback
test_dynamic_fallback.py            # aux variable dynamic method
test_custom_endval.py               # endval / permanent-shock tests
test_dynare_rbc.py                  # regression vs Dynare 6.2 RBC reference output
test_expectation_errors.py          # solve_perfect_foresight_expectation_errors unit tests
test_dynare_expectation_errors.py   # regression vs Dynare 6.2 reference output (3-segment RBC)
test_methods_comparison.py          # comparison script (runs as a script, not pytest)
```

Run all pytest-compatible tests:
```bash
pytest test_homotopy.py test_arbitrary_lags.py test_auto_to_dynamic.py test_dynamic_fallback.py test_custom_endval.py test_dynare_rbc.py test_expectation_errors.py test_dynare_expectation_errors.py
```

## Branching convention

Use a separate branch per concern:
- `feat/<name>` for new features
- `fix/<name>` for bug fixes
- `docs/<name>` for documentation-only changes

Do not mix documentation changes, fixes, and features on the same branch.
