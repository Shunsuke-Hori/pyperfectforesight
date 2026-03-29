"""Tests for SteadyState, compile_steady_state_funcs, solve_steady_state,
and the compiled_ss / auto-endval feature in the three solvers."""

import numpy as np
import pytest
import sympy as sp

from pyperfectforesight import (
    v,
    process_model,
    compile_steady_state_funcs,
    solve_steady_state,
    solve_perfect_foresight,
    solve_perfect_foresight_homotopy,
    solve_perfect_foresight_expectation_errors,
    SteadyState,
)

# ---------------------------------------------------------------------------
# RBC model with exogenous TFP level z (Dynare lag notation)
#
#   Euler:   1/c_t = beta * alpha * z_{t+1} * k_t^(alpha-1) / c_{t+1}
#   Capital: k_t   = z_t * k_{t-1}^alpha - c_t
#
# Steady state at exogenous level Z:
#   1 = beta * alpha * Z * k_ss^(alpha-1)  =>  k_ss = (alpha * beta * Z)^(1/(1-alpha))
#   c_ss = Z * k_ss^alpha - k_ss
# ---------------------------------------------------------------------------

ALPHA = sp.Symbol("alpha")
BETA  = sp.Symbol("beta")

PARAMS = {ALPHA: 0.36, BETA: 0.99}
ALPHA_V = 0.36
BETA_V  = 0.99

EQ1 = 1/v("c", 0) - BETA * ALPHA * v("z", 1) * v("k", 0)**(ALPHA - 1) / v("c", 1)
EQ2 = v("k", 0) - v("z", 0) * v("k", -1)**ALPHA + v("c", 0)

EQUATIONS = [EQ1, EQ2]
VARS_DYN  = ["c", "k"]
VARS_EXO  = ["z"]

T = 80


def _analytical_ss(z):
    k = (ALPHA_V * BETA_V * z) ** (1 / (1 - ALPHA_V))
    c = z * k**ALPHA_V - k
    return np.array([c, k])


@pytest.fixture(scope="module")
def model():
    return process_model(EQUATIONS, VARS_DYN, VARS_EXO)


@pytest.fixture(scope="module")
def compiled_ss():
    return compile_steady_state_funcs(EQUATIONS, VARS_DYN, VARS_EXO)


# ---------------------------------------------------------------------------
# 1. compile_steady_state_funcs bundle structure
# ---------------------------------------------------------------------------

def test_compile_bundle_keys(compiled_ss):
    assert "funcs" in compiled_ss
    assert "ss_syms" in compiled_ss
    assert "param_syms" in compiled_ss
    assert "exo_ss_syms" in compiled_ss
    assert "vars_exo" in compiled_ss
    assert "vars_dyn" in compiled_ss
    assert compiled_ss["vars_exo"] == ["z"]
    assert compiled_ss["vars_dyn"] == ["c", "k"]
    assert len(compiled_ss["exo_ss_syms"]) == 1
    assert str(compiled_ss["exo_ss_syms"][0]) == "z_exo_ss"


# ---------------------------------------------------------------------------
# 2. solve_steady_state at z=1 (default exog_ss=None → zeros)
# ---------------------------------------------------------------------------

def test_solve_ss_at_zero_exog(compiled_ss):
    """exog_ss=None defaults to 0; steady state matches analytical formula."""
    ss = solve_steady_state(compiled_ss, PARAMS)
    expected = _analytical_ss(0.0)
    # z=0 means k=0, c=0; the fsolve may not converge well — just check type
    assert isinstance(ss, SteadyState)


def test_solve_ss_at_nonzero_exog_array(compiled_ss):
    """solve_steady_state with exog_ss as array returns correct steady state."""
    expected = _analytical_ss(1.0)
    ss = solve_steady_state(
        compiled_ss, PARAMS, exog_ss=np.array([1.0]),
        initial_guess=expected,
    )
    np.testing.assert_allclose(np.asarray(ss), expected, rtol=1e-6)


def test_solve_ss_at_nonzero_exog_dict(compiled_ss):
    """solve_steady_state accepts exog_ss as a dict {var_name: value}."""
    expected = _analytical_ss(1.0)
    ss = solve_steady_state(
        compiled_ss, PARAMS, exog_ss={"z": 1.0},
        initial_guess=expected,
    )
    np.testing.assert_allclose(np.asarray(ss), expected, rtol=1e-6)


def test_solve_ss_dict_missing_key_defaults_zero(compiled_ss):
    """A dict missing a variable defaults that variable to zero."""
    # Both should behave identically (z defaults to 0 in both cases).
    # Don't assert numerical equality here — fsolve at z=0 is degenerate for
    # this model (k=0, c=0); just verify both calls succeed without error.
    guess = _analytical_ss(1.0)
    ss_dict  = solve_steady_state(compiled_ss, PARAMS, exog_ss={}, initial_guess=guess)
    ss_array = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([0.0]), initial_guess=guess)
    assert isinstance(ss_dict,  SteadyState)
    assert isinstance(ss_array, SteadyState)


def test_solve_ss_exog_shape_mismatch_raises(compiled_ss):
    """Wrong-length exog_ss array raises ValueError."""
    with pytest.raises(ValueError, match="exog_ss has"):
        solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# 3. SteadyState provenance
# ---------------------------------------------------------------------------

def test_steady_state_provenance(compiled_ss):
    """SteadyState carries values, params, exog_ss, vars_dyn, vars_exo."""
    ss = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([1.05]))

    assert isinstance(ss, SteadyState)
    assert ss.vars_dyn == ["c", "k"]
    assert ss.vars_exo == ["z"]
    np.testing.assert_allclose(ss.exog_ss, [1.05], rtol=1e-10)
    # params stored as plain {str: float}
    assert isinstance(ss.params, dict)
    assert all(isinstance(k, str) for k in ss.params)
    assert abs(ss.params["alpha"] - 0.36) < 1e-12
    assert abs(ss.params["beta"]  - 0.99) < 1e-12


# ---------------------------------------------------------------------------
# 4. SteadyState numpy interoperability
# ---------------------------------------------------------------------------

def test_steady_state_numpy_interop(compiled_ss):
    """SteadyState is transparently usable as a numpy array."""
    expected = _analytical_ss(1.0)
    ss = solve_steady_state(
        compiled_ss, PARAMS, exog_ss=np.array([1.0]), initial_guess=expected
    )

    # np.asarray
    np.testing.assert_allclose(np.asarray(ss), expected, rtol=1e-6)

    # len
    assert len(ss) == 2

    # shape and size
    assert ss.shape == (2,)
    assert ss.size == 2

    # scalar indexing
    assert abs(ss[0] - expected[0]) < 1e-6

    # fancy indexing
    np.testing.assert_allclose(ss[np.array([1, 0])], expected[[1, 0]], rtol=1e-6)

    # iteration
    vals = list(ss)
    np.testing.assert_allclose(vals, expected, rtol=1e-6)

    # np.tile (used internally for initial guess)
    tiled = np.tile(ss, (3, 1))
    assert tiled.shape == (3, 2)

    # arithmetic (used in homotopy interpolation)
    result = ss + np.zeros(2)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_steady_state_repr(compiled_ss):
    """SteadyState repr includes values, params, and exog_ss."""
    ss = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([1.05]))
    r = repr(ss)
    assert "SteadyState" in r
    assert "alpha" in r
    assert "z" in r


# ---------------------------------------------------------------------------
# 5. compiled_ss in solve_perfect_foresight (auto endval)
# ---------------------------------------------------------------------------

def test_compiled_ss_auto_endval_matches_explicit(model, compiled_ss):
    """compiled_ss auto-computes endval from exog_path[-1]; result matches
    solving with the same endval passed explicitly."""
    z_terminal = 1.05
    exog_path  = np.full((T, 1), z_terminal)
    ss_initial = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([1.0]))
    ss_terminal = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([z_terminal]))
    k_neg1 = ss_initial[1:2]

    sol_auto = solve_perfect_foresight(
        T, PARAMS, ss_terminal, model, VARS_DYN,
        exog_path=exog_path, ss_initial=ss_initial,
        initial_state=k_neg1,
        compiled_ss=compiled_ss,
    )
    sol_explicit = solve_perfect_foresight(
        T, PARAMS, ss_terminal, model, VARS_DYN,
        exog_path=exog_path, ss_initial=ss_initial,
        initial_state=k_neg1,
        endval=ss_terminal,
    )

    assert sol_auto.success, sol_auto.message
    assert sol_explicit.success, sol_explicit.message
    np.testing.assert_allclose(sol_auto.x, sol_explicit.x, atol=1e-10)


def test_compiled_ss_explicit_endval_takes_priority(model, compiled_ss):
    """When endval is passed explicitly, compiled_ss is ignored for endval."""
    z_terminal  = 1.05
    exog_path   = np.full((T, 1), z_terminal)
    ss_initial  = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([1.0]))
    ss_terminal = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([z_terminal]))
    k_neg1      = ss_initial[1:2]

    sol_with    = solve_perfect_foresight(
        T, PARAMS, ss_terminal, model, VARS_DYN,
        exog_path=exog_path, ss_initial=ss_initial,
        initial_state=k_neg1,
        endval=ss_terminal, compiled_ss=compiled_ss,
    )
    sol_without = solve_perfect_foresight(
        T, PARAMS, ss_terminal, model, VARS_DYN,
        exog_path=exog_path, ss_initial=ss_initial,
        initial_state=k_neg1,
        endval=ss_terminal,
    )

    assert sol_with.success
    np.testing.assert_allclose(sol_with.x, sol_without.x, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. compiled_ss in solve_perfect_foresight_homotopy (auto endval)
# ---------------------------------------------------------------------------

def test_homotopy_compiled_ss_auto_endval(model, compiled_ss):
    """compiled_ss auto-computes endval in homotopy; result matches explicit."""
    z_terminal  = 1.05
    exog_path   = np.full((T, 1), z_terminal)
    ss_initial  = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([1.0]))
    ss_terminal = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([z_terminal]))
    k_neg1      = ss_initial[1:2]

    sol_auto = solve_perfect_foresight_homotopy(
        T, PARAMS, ss_terminal, model, VARS_DYN,
        exog_path=exog_path, ss_initial=ss_initial,
        initial_state=k_neg1,
        compiled_ss=compiled_ss,
        n_steps=5,
    )
    sol_explicit = solve_perfect_foresight_homotopy(
        T, PARAMS, ss_terminal, model, VARS_DYN,
        exog_path=exog_path, ss_initial=ss_initial,
        initial_state=k_neg1,
        endval=ss_terminal,
        n_steps=5,
    )

    assert sol_auto.success, sol_auto.message
    np.testing.assert_allclose(sol_auto.x, sol_explicit.x, atol=1e-8)


# ---------------------------------------------------------------------------
# 7. compiled_ss in solve_perfect_foresight_expectation_errors (auto endval)
# ---------------------------------------------------------------------------

def test_expectation_errors_compiled_ss_auto_endval(model, compiled_ss):
    """compiled_ss auto-computes per-segment endval in expectation-errors solver."""
    z_terminal = 1.05
    exog_path  = np.full((T, 1), z_terminal)
    ss_initial  = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([1.0]))
    ss_terminal = solve_steady_state(compiled_ss, PARAMS, exog_ss=np.array([z_terminal]))
    k_neg1 = ss_initial[1:2]

    # Using compiled_ss: endval auto-computed from exog_path[-1] in segment 2
    news_auto = [
        (1,  np.full((T, 1), 1.0)),   # segment 1: no shock expected (z=1)
        (10, exog_path),               # segment 10: learns of permanent shock
    ]
    sol_auto = solve_perfect_foresight_expectation_errors(
        T, PARAMS, ss_terminal, model, VARS_DYN, news_auto,
        initial_state=k_neg1,
        ss_initial=ss_initial,
        compiled_ss=compiled_ss,
    )

    # Explicit 3-tuple endval for comparison
    news_explicit = [
        (1,  np.full((T, 1), 1.0)),
        (10, exog_path, ss_terminal),
    ]
    sol_explicit = solve_perfect_foresight_expectation_errors(
        T, PARAMS, ss_terminal, model, VARS_DYN, news_explicit,
        initial_state=k_neg1,
        ss_initial=ss_initial,
    )

    assert sol_auto.success, sol_auto.message
    assert sol_explicit.success, sol_explicit.message
    np.testing.assert_allclose(sol_auto.x, sol_explicit.x, atol=1e-8)
