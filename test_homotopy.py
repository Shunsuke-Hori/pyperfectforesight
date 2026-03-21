"""Tests for solve_perfect_foresight_homotopy."""

import numpy as np
import pytest
import sympy as sp

from dynare_python import (
    v,
    process_model,
    solve_perfect_foresight,
    solve_perfect_foresight_homotopy,
)

# ---------------------------------------------------------------------------
# Shared RBC fixture
# ---------------------------------------------------------------------------
# Two-variable RBC:
#   Euler:   1/c_0 = beta * alpha * k_1^(alpha-1) / c_1
#   Capital: k_1   = k_0^alpha - c_0

ALPHA = 0.36
BETA = 0.99

# Parameters are baked in numerically so no SymPy parameter symbols are needed.
EQ1 = v("c", 0) ** (-1) - BETA * ALPHA * v("k", 1) ** (ALPHA - 1) * v("c", 1) ** (-1)
EQ2 = v("k", 1) - v("k", 0) ** ALPHA + v("c", 0)

VARS_DYN = ["c", "k"]
PARAMS = {}

K_SS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
C_SS = K_SS**ALPHA - K_SS
SS = np.array([C_SS, K_SS])

T = 60


@pytest.fixture
def model():
    return process_model([EQ1, EQ2], VARS_DYN)


@pytest.fixture
def X0():
    return np.tile(SS, (T, 1))


# ---------------------------------------------------------------------------
# 1. Error cases
# ---------------------------------------------------------------------------


def test_raises_when_nothing_to_scale(model, X0):
    """Must provide initial_state or exog_path."""
    with pytest.raises(ValueError, match="nothing to homotopy on"):
        solve_perfect_foresight_homotopy(
            T, X0, PARAMS, SS, model, VARS_DYN
        )


def test_raises_on_invalid_n_steps(model, X0):
    """n_steps must be a positive integer."""
    k0 = np.array([K_SS * 1.1])
    for bad in (0, -1, 1.5, "10"):
        with pytest.raises(ValueError, match="n_steps"):
            solve_perfect_foresight_homotopy(
                T, X0, PARAMS, SS, model, VARS_DYN,
                initial_state=k0, stock_var_indices=[1],
                n_steps=bad,
            )


def test_raises_on_initial_state_length_mismatch(model, X0):
    """initial_state length must match stock_var_indices when provided."""
    wrong_initial_state = np.array([K_SS * 1.1, C_SS])  # 2 elements, but only 1 stock var
    with pytest.raises(ValueError, match="initial_state has 2 elements"):
        solve_perfect_foresight_homotopy(
            T, X0, PARAMS, SS, model, VARS_DYN,
            initial_state=wrong_initial_state, stock_var_indices=[1],
        )


# ---------------------------------------------------------------------------
# 2. Stock/jump mode — initial_state only
# ---------------------------------------------------------------------------


def test_stock_mode_initial_state(model, X0):
    """Homotopy with stock_var_indices and initial_state converges."""
    k0 = np.array([K_SS * 1.3])  # 30% above ss
    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k0,
        stock_var_indices=[1],
        n_steps=5,
    )
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-6


def test_stock_mode_matches_direct_solve(model, X0):
    """Homotopy solution matches direct Newton for a moderate shock."""
    k0 = np.array([K_SS * 1.1])

    sol_direct = solve_perfect_foresight(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k0, stock_var_indices=[1],
    )
    sol_hom = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k0, stock_var_indices=[1],
        n_steps=4,
    )

    assert sol_direct.success
    assert sol_hom.success
    np.testing.assert_allclose(
        sol_hom.x.reshape(T, -1),
        sol_direct.x.reshape(T, -1),
        rtol=1e-5,
        atol=1e-8,
    )


def test_large_shock_stock_mode(model, X0):
    """Homotopy succeeds for a large shock (150% of ss) that may be hard
    to solve directly from the steady-state initial guess."""
    k0 = np.array([K_SS * 1.5])
    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k0,
        stock_var_indices=[1],
        n_steps=8,
    )
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-6


# ---------------------------------------------------------------------------
# 3. Exog-path mode
# ---------------------------------------------------------------------------


def test_exog_path_mode(X0):
    """Homotopy with an exogenous shock path converges.

    k is a stock variable (predetermined at t=0); c is a jump variable that
    is free to respond to the shock.  This is the natural IRF setup.
    """
    eq1_z = (
        v("c", 0) ** (-1)
        - BETA * ALPHA * v("k", 1) ** (ALPHA - 1) * v("c", 1) ** (-1)
    )
    eq2_z = v("k", 1) - sp.exp(v("z", 0)) * v("k", 0) ** ALPHA + v("c", 0)

    model_z = process_model([eq1_z, eq2_z], VARS_DYN, vars_exo=["z"])

    # AR(1) technology shock: z_0 = 1%, rho = 0.9
    rho = 0.9
    exog = np.zeros((T, 1))
    exog[0, 0] = 0.01
    for t in range(1, T):
        exog[t, 0] = rho * exog[t - 1, 0]

    # k starts at its steady-state value (stock variable, index 1)
    k0 = np.array([K_SS])

    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model_z, VARS_DYN,
        exog_path=exog,
        initial_state=k0,
        stock_var_indices=[1],
        n_steps=5,
    )
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-6


# ---------------------------------------------------------------------------
# 4. Free initial-state mode (no stock_var_indices)
# ---------------------------------------------------------------------------


def test_initial_state_and_exog_path_combined(X0):
    """Homotopy scales both initial_state and exog_path simultaneously."""
    eq1_z = (
        v("c", 0) ** (-1)
        - BETA * ALPHA * v("k", 1) ** (ALPHA - 1) * v("c", 1) ** (-1)
    )
    eq2_z = v("k", 1) - sp.exp(v("z", 0)) * v("k", 0) ** ALPHA + v("c", 0)

    model_z = process_model([eq1_z, eq2_z], VARS_DYN, vars_exo=["z"])

    # AR(1) shock path
    exog = np.zeros((T, 1))
    exog[0, 0] = 0.01
    for t in range(1, T):
        exog[t, 0] = 0.9 * exog[t - 1, 0]

    # k also starts above steady state
    k0 = np.array([K_SS * 1.2])

    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model_z, VARS_DYN,
        exog_path=exog,
        initial_state=k0,
        stock_var_indices=[1],
        n_steps=6,
    )
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-6


# ---------------------------------------------------------------------------
# 5. n_steps=1 produces same result as a single direct call
# ---------------------------------------------------------------------------


def test_one_step_equals_direct_solve(model, X0):
    """With n_steps=1, homotopy does one solve at lam=1 from the ss warm start."""
    k0 = np.array([K_SS * 1.05])

    sol_hom = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k0, stock_var_indices=[1],
        n_steps=1,
    )
    # Warm start for n_steps=1 is the ss path, same as X0
    sol_direct = solve_perfect_foresight(
        T, np.tile(SS, (T, 1)), PARAMS, SS, model, VARS_DYN,
        initial_state=k0, stock_var_indices=[1],
    )

    assert sol_hom.success
    assert sol_direct.success
    np.testing.assert_allclose(
        sol_hom.x.reshape(T, -1),
        sol_direct.x.reshape(T, -1),
        rtol=1e-5,
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# 6. verbose=True does not raise
# ---------------------------------------------------------------------------


def test_verbose_does_not_raise(model, X0, capsys):
    """verbose=True prints progress without errors."""
    k0 = np.array([K_SS * 1.1])
    solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k0, stock_var_indices=[1],
        n_steps=3, verbose=True,
    )
    out = capsys.readouterr().out
    assert "homotopy step" in out
    assert "converged" in out


# ---------------------------------------------------------------------------
# 7. exog_ss parameter shifts the baseline
# ---------------------------------------------------------------------------


def test_exog_ss_baseline(X0):
    """exog_ss shifts the lam=0 baseline away from zero.

    Homotopy scales the exogenous path from exog_ss (1% constant shock)
    toward exog_path (2%), using the steady-state path as the warm start
    rather than a separately solved lam=0 baseline.
    """
    eq1_z = (
        v("c", 0) ** (-1)
        - BETA * ALPHA * v("k", 1) ** (ALPHA - 1) * v("c", 1) ** (-1)
    )
    eq2_z = v("k", 1) - sp.exp(v("z", 0)) * v("k", 0) ** ALPHA + v("c", 0)

    model_z = process_model([eq1_z, eq2_z], VARS_DYN, vars_exo=["z"])

    exog_target = np.full((T, 1), 0.02)   # target: constant 2% shock
    exog_base   = np.full((T, 1), 0.01)   # baseline: constant 1% shock
    k0 = np.array([K_SS])

    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model_z, VARS_DYN,
        exog_path=exog_target,
        exog_ss=exog_base,
        initial_state=k0,
        stock_var_indices=[1],
        n_steps=4,
    )
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-6
