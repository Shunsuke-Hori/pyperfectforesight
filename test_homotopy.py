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
# Shared RBC fixture — Dynare-style lag notation
# ---------------------------------------------------------------------------
# Two-variable RBC in Dynare notation where k is a stock variable:
#   Euler:   1/c_t = beta * alpha * k_t^(alpha-1) / c_{t+1}
#   Capital: k_t   = k_{t-1}^alpha - c_t
#
# k appears at lag (-1) in the accumulation equation and at current (0) in
# the Euler condition. initial_state supplies k_{-1} (the pre-period-0 value
# of capital); k_0 and c_0 are both determined by the model at t=0.

ALPHA = 0.36
BETA = 0.99

# Parameters are baked in numerically so no SymPy parameter symbols are needed.
EQ1 = v("c", 0) ** (-1) - BETA * ALPHA * v("k", 0) ** (ALPHA - 1) * v("c", 1) ** (-1)
EQ2 = v("k", 0) - v("k", -1) ** ALPHA + v("c", 0)

VARS_DYN = ["c", "k"]
PARAMS = {}

K_SS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
C_SS = K_SS**ALPHA - K_SS
SS = np.array([C_SS, K_SS])

T = 60


@pytest.fixture(scope="module")
def model():
    return process_model([EQ1, EQ2], VARS_DYN)


@pytest.fixture(scope="module")
def X0():
    return np.tile(SS, (T, 1))


# ---------------------------------------------------------------------------
# 1. Error cases
# ---------------------------------------------------------------------------


def test_solve_raises_stock_var_indices_without_initial_state(model, X0):
    """solve_perfect_foresight raises when stock_var_indices is given without initial_state."""
    with pytest.raises(ValueError, match="stock_var_indices"):
        solve_perfect_foresight(
            T, X0, PARAMS, SS, model, VARS_DYN,
            stock_var_indices=[1],
            # initial_state intentionally omitted
        )


def test_raises_when_nothing_to_scale(model, X0):
    """Must provide initial_state or exog_path."""
    with pytest.raises(ValueError, match="nothing to homotopy on"):
        solve_perfect_foresight_homotopy(
            T, X0, PARAMS, SS, model, VARS_DYN
        )


def test_raises_on_invalid_n_steps(model, X0):
    """n_steps must be a positive integer."""
    k_neg1 = np.array([K_SS * 1.1])
    for bad in (0, -1, 1.5, "10"):
        with pytest.raises(ValueError, match="n_steps"):
            solve_perfect_foresight_homotopy(
                T, X0, PARAMS, SS, model, VARS_DYN,
                initial_state=k_neg1, stock_var_indices=[1],
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
# 2. Stock/jump mode — initial_state = k_{-1} (pre-period-0 capital)
# ---------------------------------------------------------------------------


def test_stock_mode_initial_state(model, X0):
    """Homotopy with stock_var_indices and initial_state converges.

    initial_state = K_SS * 1.3 means capital was 30% above steady state
    before period 0 (Dynare convention: k_{-1}).
    """
    k_neg1 = np.array([K_SS * 1.3])
    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k_neg1,
        stock_var_indices=[1],
        n_steps=5,
    )
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-6


def test_stock_mode_matches_direct_solve(model, X0):
    """Homotopy solution matches direct Newton for a moderate shock.

    Both calls use the BVP formulation with initial_state = k_{-1}.
    """
    k_neg1 = np.array([K_SS * 1.1])

    sol_direct = solve_perfect_foresight(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
    )
    sol_hom = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
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
    k_neg1 = np.array([K_SS * 1.5])
    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k_neg1,
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

    k is a stock variable (predetermined at t=-1 via initial_state); c is a
    jump variable that is free to respond to the shock.  This is the natural
    IRF setup.  The exogenous TFP shock z enters multiplicatively in the
    capital accumulation equation (Dynare lag notation).
    """
    eq1_z = (
        v("c", 0) ** (-1)
        - BETA * ALPHA * v("k", 0) ** (ALPHA - 1) * v("c", 1) ** (-1)
    )
    eq2_z = v("k", 0) - sp.exp(v("z", 0)) * v("k", -1) ** ALPHA + v("c", 0)

    model_z = process_model([eq1_z, eq2_z], VARS_DYN, vars_exo=["z"])

    # AR(1) technology shock: z_0 = 1%, rho = 0.9
    rho = 0.9
    exog = np.zeros((T, 1))
    exog[0, 0] = 0.01
    for t in range(1, T):
        exog[t, 0] = rho * exog[t - 1, 0]

    # k_{-1} starts at its steady-state value (no initial capital displacement)
    k_neg1 = np.array([K_SS])

    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model_z, VARS_DYN,
        exog_path=exog,
        initial_state=k_neg1,
        stock_var_indices=[1],
        n_steps=5,
    )
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-6


# ---------------------------------------------------------------------------
# 4. Scaling both initial_state and exog_path simultaneously
# ---------------------------------------------------------------------------


def test_initial_state_and_exog_path_combined(X0):
    """Homotopy scales both initial_state and exog_path simultaneously."""
    eq1_z = (
        v("c", 0) ** (-1)
        - BETA * ALPHA * v("k", 0) ** (ALPHA - 1) * v("c", 1) ** (-1)
    )
    eq2_z = v("k", 0) - sp.exp(v("z", 0)) * v("k", -1) ** ALPHA + v("c", 0)

    model_z = process_model([eq1_z, eq2_z], VARS_DYN, vars_exo=["z"])

    # AR(1) shock path
    exog = np.zeros((T, 1))
    exog[0, 0] = 0.01
    for t in range(1, T):
        exog[t, 0] = 0.9 * exog[t - 1, 0]

    # k_{-1} also above steady state
    k_neg1 = np.array([K_SS * 1.2])

    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model_z, VARS_DYN,
        exog_path=exog,
        initial_state=k_neg1,
        stock_var_indices=[1],
        n_steps=6,
    )
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-6


def test_exog_path_only_no_initial_state(X0):
    """Homotopy with exog_path only and initial_state=None (Case 2).

    When initial_state=None and stock_var_indices=None, solve_perfect_foresight
    uses Case 2: X[0] is pinned to ss_initial for ALL variables.  This is
    valid when all model variables are predetermined (state variables), so
    pinning the full X[0] vector is the correct boundary condition.

    The model below is a linear VAR(1) where both c and k depend only on
    their own lagged values and the exogenous shock z, making them both
    predetermined.  The steady state at z=0 is (C_SS, K_SS), the same as
    the RBC fixture.
    """
    # Linear VAR(1): both c and k are state variables (no jump variables).
    #   c_{t+1} = C_SS + RHO_C * (c_t - C_SS) + z_t
    #   k_{t+1} = K_SS + PHI   * (c_t - C_SS) + PSI * (k_t - K_SS)
    # SS at z=0: c_SS = C_SS, k_SS = K_SS ✓
    RHO_C, PHI, PSI = 0.8, 0.3, 0.9
    eq1_var = v("c", 1) - (C_SS + RHO_C * (v("c", 0) - C_SS) + v("z", 0))
    eq2_var = v("k", 1) - (K_SS + PHI * (v("c", 0) - C_SS) + PSI * (v("k", 0) - K_SS))
    model_var = process_model([eq1_var, eq2_var], VARS_DYN, vars_exo=["z"])

    # AR(1) shock starting at t=0 and decaying so the economy returns to the
    # original SS, making the terminal condition X[T-1]=ss consistent.
    rho_z = 0.9
    exog = np.zeros((T, 1))
    exog[0, 0] = 0.01
    for t in range(1, T):
        exog[t, 0] = rho_z * exog[t - 1, 0]

    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model_var, VARS_DYN,
        exog_path=exog,
        # initial_state intentionally omitted — exercises the exog-only branch
        # through Case 2 (X[0] pinned to ss for all variables).
        # use_terminal_conditions=False keeps the system square (exactly solvable)
        # since Case 2 with terminal conditions is overdetermined.
        use_terminal_conditions=False,
        n_steps=5,
    )
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-6


# ---------------------------------------------------------------------------
# 5. n_steps=1 produces same result as a single direct call
# ---------------------------------------------------------------------------


def test_one_step_equals_direct_solve(model, X0):
    """With n_steps=1, homotopy does one BVP solve at lam=1 from the ss warm start.

    The direct solve and the single homotopy step both start from the ss path
    and solve the same augmented-path BVP, so their results must match.
    """
    k_neg1 = np.array([K_SS * 1.05])

    sol_hom = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
        n_steps=1,
    )
    # Direct BVP solve from the same warm start (ss path)
    sol_direct = solve_perfect_foresight(
        T, np.tile(SS, (T, 1)), PARAMS, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
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
    k_neg1 = np.array([K_SS * 1.1])
    solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=k_neg1, stock_var_indices=[1],
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
        - BETA * ALPHA * v("k", 0) ** (ALPHA - 1) * v("c", 1) ** (-1)
    )
    eq2_z = v("k", 0) - sp.exp(v("z", 0)) * v("k", -1) ** ALPHA + v("c", 0)

    model_z = process_model([eq1_z, eq2_z], VARS_DYN, vars_exo=["z"])

    exog_target = np.full((T, 1), 0.02)   # target: constant 2% shock
    exog_base   = np.full((T, 1), 0.01)   # baseline: constant 1% shock
    k_neg1 = np.array([K_SS])

    sol = solve_perfect_foresight_homotopy(
        T, X0, PARAMS, SS, model_z, VARS_DYN,
        exog_path=exog_target,
        exog_ss=exog_base,
        initial_state=k_neg1,
        stock_var_indices=[1],
        n_steps=4,
    )
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-6


def test_exog_ss_without_exog_path_warns(X0):
    """Passing exog_ss without exog_path emits a UserWarning.

    exog_ss is only meaningful as the lam=0 baseline when scaling toward a
    provided exog_path.  Without exog_path, exog_ss has no effect and the
    homotopy proceeds with a zero exogenous path throughout.
    """
    eq1_z = (
        v("c", 0) ** (-1)
        - BETA * ALPHA * v("k", 0) ** (ALPHA - 1) * v("c", 1) ** (-1)
    )
    eq2_z = v("k", 0) - sp.exp(v("z", 0)) * v("k", -1) ** ALPHA + v("c", 0)
    model_z = process_model([eq1_z, eq2_z], VARS_DYN, vars_exo=["z"])

    exog_base = np.full((T, 1), 0.01)
    k_neg1 = np.array([K_SS * 1.1])

    with pytest.warns(UserWarning, match="exog_ss"):
        solve_perfect_foresight_homotopy(
            T, X0, PARAMS, SS, model_z, VARS_DYN,
            exog_ss=exog_base,        # provided without exog_path
            initial_state=k_neg1,
            stock_var_indices=[1],
            n_steps=3,
        )
