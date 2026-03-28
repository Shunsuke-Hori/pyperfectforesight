"""Tests for compile_steady_state_funcs and solve_steady_state."""

import numpy as np
import pytest
import sympy as sp

from pyperfectforesight import (
    v,
    compute_steady_state_numerical,
    compile_steady_state_funcs,
    solve_steady_state,
)

# ---------------------------------------------------------------------------
# Simple two-variable RBC model with symbolic parameters
# ---------------------------------------------------------------------------
# Euler:   alpha * A * k_0^(alpha-1) = 1/beta - (1 - delta)
# Capital: c_0 + k_0 = A * k_{-1}^alpha + (1-delta) * k_{-1}

alpha = sp.Symbol("alpha")
delta = sp.Symbol("delta")
beta  = sp.Symbol("beta")
A     = sp.Symbol("A")

EQUATIONS = [
    v("c", 0) + v("k", 0) - (1 - delta) * v("k", -1) - A * v("k", -1) ** alpha,
    alpha * A * v("k", 0) ** (alpha - 1) - (1 / beta - (1 - delta)),
]
VARS_DYN = ["c", "k"]
PARAMS = {alpha: 0.36, delta: 0.025, beta: 0.99, A: 1.0}
INITIAL_GUESS = np.array([1.0, 10.0])


# ---------------------------------------------------------------------------
# 1. Equivalence with compute_steady_state_numerical
# ---------------------------------------------------------------------------

def test_solve_steady_state_matches_numerical():
    """solve_steady_state must return the same values as compute_steady_state_numerical."""
    ss_ref = compute_steady_state_numerical(EQUATIONS, VARS_DYN, PARAMS, initial_guess=INITIAL_GUESS)
    compiled = compile_steady_state_funcs(EQUATIONS, VARS_DYN)
    ss_new = solve_steady_state(compiled, PARAMS, initial_guess=INITIAL_GUESS)
    np.testing.assert_allclose(ss_new, ss_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. Parameter sweep: compiled bundle reused across calls
# ---------------------------------------------------------------------------

def test_parameter_sweep_reuses_compiled_bundle():
    """Sweeping over beta values with a fixed compiled bundle gives sensible results.

    Higher beta (more patient) → higher steady-state capital.
    """
    compiled = compile_steady_state_funcs(EQUATIONS, VARS_DYN)
    beta_vals = [0.97, 0.98, 0.99]
    k_ss_vals = []
    for b in beta_vals:
        params = {**PARAMS, beta: b}
        ss = solve_steady_state(compiled, params, initial_guess=INITIAL_GUESS)
        k_ss_vals.append(ss[1])  # capital is second variable
    # Higher beta → higher k_ss
    assert k_ss_vals[0] < k_ss_vals[1] < k_ss_vals[2]


# ---------------------------------------------------------------------------
# 3. param_syms detection
# ---------------------------------------------------------------------------

def test_param_syms_detected_correctly():
    """compile_steady_state_funcs must detect exactly the four model parameters."""
    compiled = compile_steady_state_funcs(EQUATIONS, VARS_DYN)
    detected = {s.name for s in compiled["param_syms"]}
    assert detected == {"alpha", "delta", "beta", "A"}


# ---------------------------------------------------------------------------
# 4. Error: missing parameter in params_dict
# ---------------------------------------------------------------------------

def test_solve_steady_state_missing_param_raises():
    """solve_steady_state must raise ValueError when a parameter is missing."""
    compiled = compile_steady_state_funcs(EQUATIONS, VARS_DYN)
    incomplete = {k: v for k, v in PARAMS.items() if k != beta}
    with pytest.raises(ValueError, match="Missing parameter value"):
        solve_steady_state(compiled, incomplete, initial_guess=INITIAL_GUESS)


# ---------------------------------------------------------------------------
# 5. Error: undeclared time-indexed variable (typo) raises at compile time
# ---------------------------------------------------------------------------

def test_compile_raises_on_undeclared_variable():
    """compile_steady_state_funcs must raise ValueError when an equation contains
    a time-indexed symbol whose base name is not in vars_dyn or vars_exo."""
    bad_eq = v("kk", -1) - v("k", -1)  # "kk" is not declared
    equations_with_typo = EQUATIONS + [bad_eq]
    with pytest.raises(ValueError, match="time-indexed symbol"):
        compile_steady_state_funcs(equations_with_typo, VARS_DYN)


# ---------------------------------------------------------------------------
# 6. vars_exo: exogenous variables are zeroed at steady state
# ---------------------------------------------------------------------------

def test_compile_with_vars_exo():
    """compile_steady_state_funcs zeroes exogenous variables at the steady state."""
    g = sp.Symbol("g")
    # Augment capital equation with a government spending shock (zero at ss)
    eqs = [
        v("c", 0) + v("k", 0) + v("g", 0) - (1 - delta) * v("k", -1) - A * v("k", -1) ** alpha,
        alpha * A * v("k", 0) ** (alpha - 1) - (1 / beta - (1 - delta)),
    ]
    compiled = compile_steady_state_funcs(eqs, VARS_DYN, vars_exo=["g"])
    ss = solve_steady_state(compiled, PARAMS, initial_guess=INITIAL_GUESS)
    # With g=0 at steady state the result should match the baseline
    compiled_base = compile_steady_state_funcs(EQUATIONS, VARS_DYN)
    ss_base = solve_steady_state(compiled_base, PARAMS, initial_guess=INITIAL_GUESS)
    np.testing.assert_allclose(ss, ss_base, rtol=1e-6)
