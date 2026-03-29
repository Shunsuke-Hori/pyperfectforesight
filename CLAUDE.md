# CLAUDE.md

Project context and conventions for Claude Code.

## Project overview

`pyperfectforesight` is a minimal Dynare-style perfect foresight solver in Python.
All model logic lives in `pyperfectforesight/core.py`. There are no sub-packages.

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

### `endval` parameter and auto-computation
`solve_perfect_foresight()`, `solve_perfect_foresight_homotopy()`, and `solve_perfect_foresight_expectation_errors()` all accept a `compiled_ss` keyword argument (a bundle from `compile_steady_state_funcs()`). When `compiled_ss` is provided and `endval` is omitted:
- `solve_perfect_foresight` / `solve_perfect_foresight_homotopy`: `endval` is automatically computed by calling `solve_steady_state(compiled_ss, params_dict, exog_ss=exog_path[-1])`.
- `solve_perfect_foresight_expectation_errors`: for each segment, `endval` is auto-computed from the last row of that segment's `exog_path`, unless a 3-tuple already supplies an explicit `endval`.

When `endval` is provided explicitly, `compiled_ss` is ignored for `endval` resolution (the pre-computed value is used as-is). This allows skipping recomputation in repeated simulations.

### `SteadyState` class
`solve_steady_state()` returns a `SteadyState` object (not a plain ndarray). It carries:
- `.values`: endogenous variable values (numpy array)
- `.params`: `{str: float}` dict of parameter values used
- `.exog_ss`: exogenous variable values at which the SS was computed (array or None)
- `.vars_dyn`, `.vars_exo`: variable name lists

`SteadyState` is transparently usable as a numpy array everywhere (`__array__`, `__len__`, `.shape`, `.size`, `__getitem__`, `__iter__` are all implemented). No changes needed in calling code.

### `compile_steady_state_funcs` — exogenous variables as free parameters
Exogenous variables are **not** substituted to zero at compile time. Instead, each exo var `z` gets a symbol `z_exo_ss` that becomes a free argument in the lambdified functions. `solve_steady_state(..., exog_ss=...)` passes those values at solve time. Backward compatibility is maintained: old callers omitting `exog_ss` get zeros.

The compiled bundle now has two extra keys: `exo_ss_syms` (list of SymPy symbols) and `vars_exo` (list of strings).

### Function signatures
All three solvers share the convention: required positional arguments first; `X0` comes immediately after `vars_dyn` (or `news_shocks`), followed by other optional positional-or-keyword parameters and then keyword-only args.

```python
solve_perfect_foresight(
    T, params_dict, ss, model_funcs, vars_dyn,
    X0=None, exog_path=None, initial_state=None, ss_initial=None,
    stock_var_indices=None, ..., *, endval=None, compiled_ss=None, ...)

solve_perfect_foresight_homotopy(
    T, params_dict, ss, model_funcs, vars_dyn,
    X0=None, exog_path=None, initial_state=None, ss_initial=None,
    stock_var_indices=None, *, endval=None, compiled_ss=None, ...)

solve_perfect_foresight_expectation_errors(
    T, params_dict, ss, model_funcs, vars_dyn, news_shocks,
    X0=None, initial_state=None, ss_initial=None,
    stock_var_indices=None, ..., compiled_ss=None)
```

### `X0` default initial guess
`X0` is optional in all three solvers (default `None`). When omitted, the solver constructs the initial guess automatically:

- `solve_perfect_foresight`: tiles `endval` (or `ss` if `endval` is not provided) over all `T` periods.
- `solve_perfect_foresight_homotopy`: always warm-starts from `np.tile(ss_initial, (T, 1))` regardless of `X0`; `X0` is validated for shape if provided but otherwise unused.
- `solve_perfect_foresight_expectation_errors`: tiles the effective terminal steady state for the first segment — the `endval` from the first `news_shocks` 3-tuple if present, else the auto-computed SS from `compiled_ss` at `exog_path[-1]` of the first segment if `compiled_ss` is provided, else `ss` — over all `T` periods. Subsequent sub-solves are warm-started from the previous sub-solve's tail.

### Expectation-errors solver
`solve_perfect_foresight_expectation_errors()` replicates Dynare's `perfect_foresight_with_expectation_errors_setup` / `perfect_foresight_with_expectation_errors_solver`. It accepts a `news_shocks` list of 2- or 3-tuples `(learnt_in, exog_path[, endval])`. At each `learnt_in` the solver re-solves from that period forward and stitches the pieces together. Key design points:
- `learnt_in` is 1-indexed; the list must be sorted and start with `learnt_in=1`.
- The 3-tuple `endval` overrides the terminal steady state from that segment onward (mirrors Dynare's `endval(learnt_in=k)`).
- `constant_simulation_length=False` (default) uses a shrinking horizon `T - learnt_in + 1`; `True` uses the full `T` every sub-solve.
- The returned `OptimizeResult` includes `success`, `status` (1/0), `message`, `sub_results` (per-segment), `x_aux`, and `vars_aux`.
- `exog_path=None` passes an all-zero path; only correct when the exogenous steady state is zero.

## Repository layout

```
pyperfectforesight/
├── __init__.py         # Public API exports
├── __version__.py      # Version string
└── core.py             # All model/solver logic

examples/               # Standalone demo scripts (PNGs saved here)
tests/                  # All pytest-compatible test files and reference data
├── dynare_ref_output/  # Dynare 6.2 reference CSVs (only .csv files are tracked)
├── test_homotopy.py
├── test_arbitrary_lags.py
├── test_custom_endval.py
├── test_dynare_rbc.py
├── test_expectation_errors.py
└── test_dynare_expectation_errors.py
scripts/                # Standalone comparison/demo scripts (not collected by pytest)
├── compare_aux_methods.py
├── compare_auto_to_dynamic.py
└── compare_dynamic_fallback.py
docs/                   # Sphinx documentation source (MyST/Markdown)
├── conf.py
├── index.md
├── installation.md
├── getting-started.md
├── solvers.md
├── initial-guess.md
├── auxiliary-variables.md
└── api-reference.md
README.md               # User-facing documentation
CLAUDE.md               # This file
```

## Testing

Tests live in `tests/`. `pyproject.toml` sets `testpaths = ["tests"]` so a bare `pytest` runs all pytest-compatible tests:

```bash
pytest
```

Individual files (if needed):
```
tests/test_homotopy.py                    # solve_perfect_foresight and solve_perfect_foresight_homotopy tests
tests/test_arbitrary_lags.py              # arbitrary lag/lead support tests
tests/test_custom_endval.py               # endval / permanent-shock tests
tests/test_dynare_rbc.py                  # regression vs Dynare 6.2 RBC reference output
tests/test_expectation_errors.py          # solve_perfect_foresight_expectation_errors unit tests
tests/test_dynare_expectation_errors.py   # regression vs Dynare 6.2 reference output (3-segment RBC)
```

Standalone comparison scripts (run directly, not via pytest) live in `scripts/`.

## Branching convention

Use a separate branch per concern:
- `feat/<name>` for new features
- `fix/<name>` for bug fixes
- `docs/<name>` for documentation-only changes

Do not mix documentation changes, fixes, and features on the same branch.
