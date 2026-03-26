"""Tests for solve_perfect_foresight_expectation_errors."""

import numpy as np
import pytest

from pyperfectforesight import (
    v,
    process_model,
    solve_perfect_foresight,
    solve_perfect_foresight_expectation_errors,
)

# ---------------------------------------------------------------------------
# Shared RBC fixture — same model as test_homotopy.py
# ---------------------------------------------------------------------------
# Euler:   1/c_t = beta * alpha * k_t^(alpha-1) / c_{t+1}
# Capital: k_t   = k_{t-1}^alpha - c_t

ALPHA = 0.36
BETA = 0.99

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
# 1. Validation errors
# ---------------------------------------------------------------------------


def test_empty_news_shocks_raises(model, X0):
    with pytest.raises(ValueError, match="non-empty"):
        solve_perfect_foresight_expectation_errors(
            T, X0, PARAMS, SS, model, VARS_DYN, news_shocks=[]
        )


def test_first_learnt_in_not_1_raises(model, X0):
    exog = np.zeros((T, 0))
    with pytest.raises(ValueError, match="learnt_in=1"):
        solve_perfect_foresight_expectation_errors(
            T, X0, PARAMS, SS, model, VARS_DYN,
            news_shocks=[(5, exog)],
        )


def test_unsorted_news_shocks_raises(model, X0):
    exog = np.zeros((T, 0))
    with pytest.raises(ValueError, match="sorted"):
        solve_perfect_foresight_expectation_errors(
            T, X0, PARAMS, SS, model, VARS_DYN,
            news_shocks=[(1, exog), (30, exog), (15, exog)],
        )


def test_duplicate_learnt_in_raises(model, X0):
    exog = np.zeros((T, 0))
    with pytest.raises(ValueError, match="duplicate"):
        solve_perfect_foresight_expectation_errors(
            T, X0, PARAMS, SS, model, VARS_DYN,
            news_shocks=[(1, exog), (15, exog), (15, exog)],
        )


# ---------------------------------------------------------------------------
# 2. Single news event == solve_perfect_foresight
# ---------------------------------------------------------------------------


def test_single_shock_matches_direct_solve(model, X0):
    """With one news event at t=1, result must equal solve_perfect_foresight."""
    k_init = K_SS * 0.9
    initial_state = np.array([k_init])

    sol_direct = solve_perfect_foresight(
        T, X0, PARAMS, SS, model, VARS_DYN,
        initial_state=initial_state,
    )

    sol_ee = solve_perfect_foresight_expectation_errors(
        T, X0, PARAMS, SS, model, VARS_DYN,
        news_shocks=[(1, None)],
        initial_state=initial_state,
    )

    assert sol_ee.success
    np.testing.assert_allclose(
        sol_ee.x.reshape(T, -1),
        sol_direct.x.reshape(T, -1),
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# 3. No shock at steady state → path stays at SS
# ---------------------------------------------------------------------------


def test_no_shock_stays_at_ss(model, X0):
    """Starting at SS with no perturbation, every sub-solve stays at SS."""
    sol = solve_perfect_foresight_expectation_errors(
        T, X0, PARAMS, SS, model, VARS_DYN,
        news_shocks=[(1, None), (20, None), (40, None)],
    )
    assert sol.success
    np.testing.assert_allclose(
        sol.x.reshape(T, -1), np.tile(SS, (T, 1)), atol=1e-8
    )


# ---------------------------------------------------------------------------
# 4. Two news events — stitching check
# ---------------------------------------------------------------------------


def test_two_events_stitching(model, X0):
    """Two news events: verify path is correctly stitched and sub_results present."""
    k_init = K_SS * 0.9
    initial_state = np.array([k_init])
    split = 20

    sol = solve_perfect_foresight_expectation_errors(
        T, X0, PARAMS, SS, model, VARS_DYN,
        news_shocks=[(1, None), (split, None)],
        initial_state=initial_state,
    )

    assert sol.success
    assert len(sol.sub_results) == 2

    X_full = sol.x.reshape(T, -1)
    n_keep_1 = split - 1  # rows kept from sub-solve 1 (periods 1..split-1)

    # Stitching: the first n_keep_1 rows of the full path must exactly match
    # the first n_keep_1 rows of sub-solve 1's solution.
    np.testing.assert_array_equal(
        X_full[:n_keep_1],
        sol.sub_results[0].x.reshape(T, -1)[:n_keep_1],
    )

    # Monotone convergence back to SS (capital starts below SS and recovers).
    assert X_full[0, 1] < K_SS  # starts below SS
    assert X_full[-1, 1] == pytest.approx(K_SS, rel=1e-4)  # converges to SS


# ---------------------------------------------------------------------------
# 5. constant_simulation_length=False
# ---------------------------------------------------------------------------


def test_shrinking_window(model, X0):
    """constant_simulation_length=False should also produce a T-row stitched path."""
    k_init = K_SS * 0.9
    initial_state = np.array([k_init])

    sol = solve_perfect_foresight_expectation_errors(
        T, X0, PARAMS, SS, model, VARS_DYN,
        news_shocks=[(1, None), (30, None)],
        initial_state=initial_state,
        constant_simulation_length=False,
    )

    assert sol.success
    assert sol.x.shape == (T * len(VARS_DYN),)
    X_full = sol.x.reshape(T, -1)
    assert X_full[0, 1] < K_SS
    assert X_full[-1, 1] == pytest.approx(K_SS, rel=1e-4)


# ---------------------------------------------------------------------------
# 6. sub_results diagnostic field
# ---------------------------------------------------------------------------


def test_sub_results_length(model, X0):
    """sol.sub_results has one entry per news event."""
    sol = solve_perfect_foresight_expectation_errors(
        T, X0, PARAMS, SS, model, VARS_DYN,
        news_shocks=[(1, None), (15, None), (40, None)],
    )
    assert len(sol.sub_results) == 3
    for sr in sol.sub_results:
        assert hasattr(sr, 'success')


# ---------------------------------------------------------------------------
# 7. sub_x0 — per-sub-solve initial guess
# ---------------------------------------------------------------------------


def test_sub_x0_none_list_same_as_default(model, X0):
    """sub_x0=[None, None] must give the same result as sub_x0=None."""
    k_init = K_SS * 0.9
    initial_state = np.array([k_init])
    news_shocks = [(1, None), (30, None)]

    sol_default = solve_perfect_foresight_expectation_errors(
        T, X0, PARAMS, SS, model, VARS_DYN,
        news_shocks=news_shocks,
        initial_state=initial_state,
    )
    sol_none_list = solve_perfect_foresight_expectation_errors(
        T, X0, PARAMS, SS, model, VARS_DYN,
        news_shocks=news_shocks,
        initial_state=initial_state,
        sub_x0=[None, None],
    )

    assert sol_default.success
    assert sol_none_list.success
    np.testing.assert_allclose(sol_none_list.x, sol_default.x, atol=1e-8)


def test_sub_x0_explicit_guess_converges(model, X0):
    """An explicit sub_x0 array for the first sub-solve still converges."""
    k_init = K_SS * 0.9
    initial_state = np.array([k_init])

    # Use SS as the explicit initial guess for sub-solve 1 (same as X0).
    explicit_x0 = np.tile(SS, (T, 1))
    sol = solve_perfect_foresight_expectation_errors(
        T, X0, PARAMS, SS, model, VARS_DYN,
        news_shocks=[(1, None)],
        initial_state=initial_state,
        sub_x0=[explicit_x0],
    )
    assert sol.success


def test_sub_x0_wrong_length_raises(model, X0):
    """sub_x0 length != len(news_shocks) must raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        solve_perfect_foresight_expectation_errors(
            T, X0, PARAMS, SS, model, VARS_DYN,
            news_shocks=[(1, None), (30, None)],
            sub_x0=[None],  # length 1 vs 2
        )


def test_sub_x0_wrong_type_raises(model, X0):
    """sub_x0 that is not a list or tuple must raise ValueError."""
    with pytest.raises(ValueError, match="list or tuple"):
        solve_perfect_foresight_expectation_errors(
            T, X0, PARAMS, SS, model, VARS_DYN,
            news_shocks=[(1, None)],
            sub_x0=X0,  # ndarray, not a list
        )


def test_sub_x0_wrong_shape_raises(model, X0):
    """An entry in sub_x0 with wrong n_endo must raise ValueError."""
    bad_x0 = np.ones((T, 99))  # wrong n_endo
    with pytest.raises(ValueError, match="shape"):
        solve_perfect_foresight_expectation_errors(
            T, X0, PARAMS, SS, model, VARS_DYN,
            news_shocks=[(1, None)],
            sub_x0=[bad_x0],
        )


def test_sub_x0_empty_rows_raises(model, X0):
    """An entry in sub_x0 with zero rows must raise ValueError."""
    empty_x0 = np.empty((0, len(VARS_DYN)))
    with pytest.raises(ValueError, match="at least one row"):
        solve_perfect_foresight_expectation_errors(
            T, X0, PARAMS, SS, model, VARS_DYN,
            news_shocks=[(1, None)],
            sub_x0=[empty_x0],
        )


def test_sub_x0_override_second_subsolver(model, X0):
    """Supplying an explicit guess for the second sub-solve (None for first)."""
    k_init = K_SS * 0.9
    initial_state = np.array([k_init])
    # Provide explicit guess for sub-solve 2 only; sub-solve 1 uses auto.
    T_sub2 = T - 30 + 1
    explicit_x0_2 = np.tile(SS, (T_sub2, 1))

    sol = solve_perfect_foresight_expectation_errors(
        T, X0, PARAMS, SS, model, VARS_DYN,
        news_shocks=[(1, None), (30, None)],
        initial_state=initial_state,
        sub_x0=[None, explicit_x0_2],
    )
    assert sol.success
    assert sol.x.shape == (T * len(VARS_DYN),)
