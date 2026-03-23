"""Tests for custom endval (permanent shock / different terminal steady state)."""

import numpy as np
import pytest
import sympy as sp

from dynare_python import v, process_model, solve_perfect_foresight, solve_perfect_foresight_homotopy

# ---------------------------------------------------------------------------
# RBC model — same as other test files
# ---------------------------------------------------------------------------
ALPHA = 0.36
BETA = 0.99

EQ1 = v("c", 0) ** (-1) - BETA * ALPHA * v("k", 0) ** (ALPHA - 1) * v("c", 1) ** (-1)
EQ2 = v("k", 0) - v("k", -1) ** ALPHA + v("c", 0)

VARS_DYN = ["c", "k"]

K_SS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
C_SS = K_SS ** ALPHA - K_SS
SS = np.array([C_SS, K_SS])

T = 80


@pytest.fixture(scope="module")
def model():
    return process_model([EQ1, EQ2], VARS_DYN)


# ---------------------------------------------------------------------------
# RBC model with permanent TFP shock (exogenous z)
#
# k_t = exp(z_t) * k_{t-1}^alpha - c_t
# Euler is unchanged (z doesn't enter MPK in this formulation)
#
# New steady state at z = Z_NEW:
#   k_ss_new = K_SS  (Euler condition unchanged)
#   c_ss_new = exp(Z_NEW) * K_SS^alpha - K_SS
# ---------------------------------------------------------------------------
Z_NEW = 0.05   # 5% permanent TFP increase
K_SS_NEW = K_SS
C_SS_NEW = np.exp(Z_NEW) * K_SS_NEW ** ALPHA - K_SS_NEW
SS_NEW = np.array([C_SS_NEW, K_SS_NEW])


@pytest.fixture(scope="module")
def model_z():
    eq1_z = v("c", 0) ** (-1) - BETA * ALPHA * v("k", 0) ** (ALPHA - 1) * v("c", 1) ** (-1)
    eq2_z = v("k", 0) - sp.exp(v("z", 0)) * v("k", -1) ** ALPHA + v("c", 0)
    return process_model([eq1_z, eq2_z], VARS_DYN, vars_exo=["z"])


# ---------------------------------------------------------------------------
# 1. endval=None defaults to ss (backward compatible)
# ---------------------------------------------------------------------------

def test_endval_none_defaults_to_ss(model):
    """Omitting endval is identical to passing endval=ss."""
    k_neg1 = np.array([K_SS * 1.1])
    X0 = np.tile(SS, (T, 1))

    sol_default = solve_perfect_foresight(
        T, X0, {}, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
    )
    sol_explicit = solve_perfect_foresight(
        T, X0, {}, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
        endval=SS,
    )

    assert sol_default.success
    assert sol_explicit.success
    np.testing.assert_allclose(sol_default.x, sol_explicit.x, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Custom endval changes the solution
# ---------------------------------------------------------------------------

def test_custom_endval_changes_solution(model):
    """Passing a different endval produces a different transition path."""
    k_neg1 = np.array([K_SS])
    X0 = np.tile(SS, (T, 1))
    SS_ALT = SS * 1.05   # arbitrary different terminal value

    sol_default = solve_perfect_foresight(
        T, X0, {}, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
    )
    sol_custom = solve_perfect_foresight(
        T, X0, {}, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
        endval=SS_ALT,
    )

    assert sol_default.success
    assert sol_custom.success
    # Solutions must differ — endval has a real effect
    assert not np.allclose(sol_default.x, sol_custom.x, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Permanent TFP shock: path converges to new steady state
# ---------------------------------------------------------------------------

def test_permanent_shock_reaches_new_ss(model_z):
    """With a permanent exog shock and endval=SS_NEW, path[-1] ≈ SS_NEW.

    The economy starts at the old steady state and transitions to a new one
    driven by a permanent 5% TFP increase (z=Z_NEW throughout).  With T=80
    the path is very close to the new steady state by the final period.
    """
    exog_path = np.full((T, 1), Z_NEW)
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS])

    sol = solve_perfect_foresight(
        T, X0, {}, SS, model_z, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
        exog_path=exog_path,
        endval=SS_NEW,
    )

    assert sol.success, sol.message
    assert np.linalg.norm(sol.fun) < 1e-6
    path = sol.x.reshape(T, -1)
    # With T=80 and a stable model, path[-1] should be close to SS_NEW
    np.testing.assert_allclose(path[-1], SS_NEW, atol=1e-3)
    # And clearly not at the old ss
    assert not np.allclose(path[-1], SS, atol=1e-3)


# ---------------------------------------------------------------------------
# 4. endval shape validation
# ---------------------------------------------------------------------------

def test_endval_wrong_shape_raises(model):
    """endval with wrong number of elements raises ValueError."""
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS])

    with pytest.raises(ValueError, match="endval has"):
        solve_perfect_foresight(
            T, X0, {}, SS, model, VARS_DYN,
            initial_state=k_neg1, stock_var_indices=[1],
            endval=np.array([C_SS]),   # only 1 element, needs 2
        )


# ---------------------------------------------------------------------------
# 5. Homotopy with permanent shock: endval scales from ss_initial to SS_NEW
# ---------------------------------------------------------------------------

def test_homotopy_permanent_shock(model_z):
    """Homotopy converges for a permanent TFP shock with custom endval."""
    exog_path = np.full((T, 1), Z_NEW)
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS])

    sol = solve_perfect_foresight_homotopy(
        T, X0, {}, SS, model_z, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
        exog_path=exog_path,
        endval=SS_NEW,
        n_steps=5,
    )

    assert sol.success, sol.message
    assert np.linalg.norm(sol.fun) < 1e-6
    path = sol.x.reshape(T, -1)
    np.testing.assert_allclose(path[-1], SS_NEW, atol=1e-3)


# ---------------------------------------------------------------------------
# 6. Homotopy: endval=None defaults to ss (backward compatible)
# ---------------------------------------------------------------------------

def test_homotopy_endval_none_defaults_to_ss(model):
    """Omitting endval in homotopy is identical to passing endval=ss."""
    k_neg1 = np.array([K_SS * 1.2])
    X0 = np.tile(SS, (T, 1))

    sol_default = solve_perfect_foresight_homotopy(
        T, X0, {}, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
        n_steps=4,
    )
    sol_explicit = solve_perfect_foresight_homotopy(
        T, X0, {}, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
        endval=SS, n_steps=4,
    )

    assert sol_default.success
    assert sol_explicit.success
    np.testing.assert_allclose(sol_default.x, sol_explicit.x, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. Homotopy: endval wrong shape raises ValueError
# ---------------------------------------------------------------------------

def test_homotopy_endval_wrong_shape_raises(model):
    """endval with wrong shape raises ValueError in homotopy."""
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS * 1.1])

    with pytest.raises(ValueError, match="endval has"):
        solve_perfect_foresight_homotopy(
            T, X0, {}, SS, model, VARS_DYN,
            initial_state=k_neg1, stock_var_indices=[1],
            endval=np.array([C_SS]),
            n_steps=3,
        )
