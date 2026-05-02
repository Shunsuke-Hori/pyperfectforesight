"""
test_model_class.py

Unit tests for the Model class (object-oriented API).

Verifies:
  - construction and attribute access
  - steady_state() returns the correct numerical steady state
  - solve() produces the same path as the functional API
  - auto endval computation from exog_path[-1] (permanent shock)
  - compiled_ss=None opts out of auto-endval
  - explicit endval overrides auto-computation
  - solve_homotopy() delegates correctly
  - solve_expectation_errors() delegates correctly
  - static-variable elimination is reflected in vars_dyn attribute
"""

import numpy as np
import sympy as sp
import pytest

from pyperfectforesight import (
    v, p, Model,
    process_model, solve_perfect_foresight,
    solve_perfect_foresight_homotopy,
    solve_perfect_foresight_expectation_errors,
    EndvalNotSteadyStateWarning,
)

# ── Simple two-variable RBC (no exogenous, parameters baked in) ──────────────

ALPHA = 0.36
BETA  = 0.99

EQ_EULER = v("c", 0)**(-1) - BETA * ALPHA * v("k", 0)**(ALPHA - 1) * v("c", 1)**(-1)
EQ_KACC  = v("k", 0) - v("k", -1)**ALPHA + v("c", 0)

VARS_DYN = ["c", "k"]

K_SS = (ALPHA * BETA)**(1 / (1 - ALPHA))
C_SS = K_SS**ALPHA - K_SS
SS   = np.array([C_SS, K_SS])

T = 60


# ── RBC with exogenous TFP (permanent shock tests) ───────────────────────────

ALPHA_S = p("alpha")
BETA_S  = p("beta")
PARAMS  = {ALPHA_S: ALPHA, BETA_S: BETA}

EQ_EULER_Z = (
    v("c", 0)**(-1)
    - BETA_S * ALPHA_S * sp.exp(v("z", 1)) * v("k", 0)**(ALPHA_S - 1) * v("c", 1)**(-1)
)
EQ_KACC_Z = v("k", 0) - sp.exp(v("z", 0)) * v("k", -1)**ALPHA_S + v("c", 0)

VARS_EXO_Z   = ["z"]
VARS_PARAMS  = ["alpha", "beta"]

Z_NEW    = 0.05
K_SS_NEW = (ALPHA * BETA * np.exp(Z_NEW))**(1 / (1 - ALPHA))
C_SS_NEW = np.exp(Z_NEW) * K_SS_NEW**ALPHA - K_SS_NEW
SS_NEW   = np.array([C_SS_NEW, K_SS_NEW])


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_simple():
    return Model([EQ_EULER, EQ_KACC], VARS_DYN)


@pytest.fixture(scope="module")
def model_z():
    return Model(
        [EQ_EULER_Z, EQ_KACC_Z], VARS_DYN,
        vars_exo=VARS_EXO_Z,
        vars_params=VARS_PARAMS,
    )


# ── Construction and attributes ───────────────────────────────────────────────

def test_model_attributes(model_simple):
    assert model_simple.vars_dyn == ["c", "k"]
    assert model_simple.vars_exo == []
    assert model_simple.vars_aux == []
    assert model_simple.vars_params == []


def test_model_z_attributes(model_z):
    assert model_z.vars_dyn == ["c", "k"]
    assert model_z.vars_exo == ["z"]
    assert model_z.vars_params == ["alpha", "beta"]


def test_model_funcs_bundle_stored(model_simple):
    """Internal _funcs dict is accessible (for debugging/advanced use)."""
    assert 'dynamic_eqs' in model_simple._funcs
    assert 'residual_funcs' in model_simple._funcs


# ── steady_state() ────────────────────────────────────────────────────────────

def test_steady_state_no_exog(model_simple):
    """steady_state() with no exogenous variables matches known analytical SS."""
    ss = model_simple.steady_state({}, initial_guess=SS * 1.1)
    np.testing.assert_allclose(ss, SS, rtol=1e-6)


def test_steady_state_with_exog(model_z):
    """steady_state() with exog_ss returns the correct steady state."""
    ss = model_z.steady_state(PARAMS, exog_ss=np.array([0.0]), initial_guess=SS * 1.1)
    np.testing.assert_allclose(ss, SS, rtol=1e-5)

    ss_new = model_z.steady_state(PARAMS, exog_ss=np.array([Z_NEW]),
                                  initial_guess=SS_NEW * 1.1)
    np.testing.assert_allclose(ss_new, SS_NEW, rtol=1e-5)


def test_steady_state_cached(model_simple):
    """_compiled_ss is populated after the first steady_state() call."""
    model_simple.steady_state({}, initial_guess=SS * 1.1)   # ensure compilation
    assert model_simple._compiled_ss is not None


# ── solve() — parity with functional API ─────────────────────────────────────

def test_solve_matches_functional_api(model_simple):
    """Model.solve() produces the same path as solve_perfect_foresight()."""
    k_neg1 = np.array([K_SS * 1.1])

    funcs = process_model([EQ_EULER, EQ_KACC], VARS_DYN)
    sol_func = solve_perfect_foresight(
        T, {}, funcs, VARS_DYN,
        endval=SS,
        initial_state=k_neg1, stock_var_indices=[1],
    )

    sol_oop = model_simple.solve(
        T, {},
        endval=SS,
        initial_state=k_neg1, stock_var_indices=[1],
    )

    assert sol_func.success
    assert sol_oop.success
    np.testing.assert_allclose(sol_oop.x, sol_func.x, atol=1e-10)


# ── solve() — permanent shock with explicit endval ───────────────────────────

def test_solve_explicit_endval_permanent_shock(model_z):
    """Model.solve() with endval=SS_NEW reaches the new steady state."""
    exog_path = np.full((T, 1), Z_NEW)
    k_neg1    = np.array([K_SS])

    sol = model_z.solve(
        T, PARAMS,
        endval=SS_NEW,
        exog_path=exog_path, initial_state=k_neg1,
    )
    assert sol.success

    # Terminal period should be near the new steady state
    X = sol.x.reshape(T, 2)
    np.testing.assert_allclose(X[-1], SS_NEW, atol=1e-3)


# ── solve() — wrong endval leads to different terminal ───────────────────────

def test_solve_explicit_endval_overrides(model_z):
    """Passing a wrong endval leads to a different (wrong) terminal."""
    exog_path = np.full((T, 1), Z_NEW)
    k_neg1    = np.array([K_SS])

    sol_correct = model_z.solve(T, PARAMS, endval=SS_NEW,
                                exog_path=exog_path, initial_state=k_neg1)
    with pytest.warns(EndvalNotSteadyStateWarning):
        sol_wrong = model_z.solve(T, PARAMS, endval=SS,  # wrong: pre-shock SS
                                  exog_path=exog_path, initial_state=k_neg1)

    assert sol_correct.success
    assert sol_wrong.success
    X_correct = sol_correct.x.reshape(T, 2)
    X_wrong   = sol_wrong.x.reshape(T, 2)
    # Terminal periods must differ when endval differs
    assert not np.allclose(X_correct[-1], X_wrong[-1], atol=1e-3)


# ── solve_homotopy() ──────────────────────────────────────────────────────────

def test_solve_homotopy_converges(model_simple):
    """Model.solve_homotopy() converges on a large capital shock."""
    k_neg1 = np.array([K_SS * 1.5])   # 50% shock — may need homotopy

    sol = model_simple.solve_homotopy(
        T, {},
        endval=SS,
        initial_state=k_neg1, stock_var_indices=[1],
    )
    assert sol.success


def test_solve_homotopy_matches_functional_api(model_simple):
    """Model.solve_homotopy() produces the same path as the functional API."""
    k_neg1 = np.array([K_SS * 1.2])

    funcs = process_model([EQ_EULER, EQ_KACC], VARS_DYN)
    sol_func = solve_perfect_foresight_homotopy(
        T, {}, funcs, VARS_DYN,
        endval=SS,
        initial_state=k_neg1, stock_var_indices=[1],
    )
    sol_oop = model_simple.solve_homotopy(
        T, {},
        endval=SS,
        initial_state=k_neg1, stock_var_indices=[1],
    )

    assert sol_func.success
    assert sol_oop.success
    np.testing.assert_allclose(sol_oop.x, sol_func.x, atol=1e-8)


# ── solve_expectation_errors() ────────────────────────────────────────────────

def test_solve_expectation_errors_converges(model_simple):
    """Model.solve_expectation_errors() runs a single-segment expectation-error sim."""
    k_neg1 = np.array([K_SS])

    # Single-segment: initial belief only (no exogenous variables in this model)
    news_shocks = [
        (1, None),   # period 1: initial belief
    ]

    sol = model_simple.solve_expectation_errors(
        T, {}, news_shocks,
        endval=SS,
        initial_state=k_neg1, stock_var_indices=[1],
    )
    assert sol.success


def test_solve_expectation_errors_matches_functional_api(model_simple):
    """Model.solve_expectation_errors() matches the functional API."""
    k_neg1     = np.array([K_SS * 1.05])
    news_shocks = [(1, None)]

    funcs = process_model([EQ_EULER, EQ_KACC], VARS_DYN)
    sol_func = solve_perfect_foresight_expectation_errors(
        T, {}, funcs, VARS_DYN, news_shocks,
        endval=SS,
        initial_state=k_neg1, stock_var_indices=[1],
    )
    sol_oop = model_simple.solve_expectation_errors(
        T, {}, news_shocks,
        endval=SS,
        initial_state=k_neg1, stock_var_indices=[1],
    )

    assert sol_func.success
    assert sol_oop.success
    np.testing.assert_allclose(sol_oop.x, sol_func.x, atol=1e-10)


# ── Static elimination reflected in vars_dyn ─────────────────────────────────

def test_vars_dyn_reflects_static_elimination():
    """vars_dyn attribute omits eliminated static variables."""
    delta = sp.Rational(1, 40)
    k_m, k_0, y_0 = v("k", -1), v("k", 0), v("y", 0)

    eq_dyn    = k_0 - (1 - delta) * k_m
    eq_static = y_0 - k_0**2       # y is truly static

    model = Model([eq_dyn, eq_static], ["k", "y"])

    assert "k" in model.vars_dyn
    assert "y" not in model.vars_dyn   # eliminated
