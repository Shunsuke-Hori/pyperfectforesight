"""Tests for arbitrary lag/lead support (|lag| > 1) in residual and Jacobian."""

import warnings

import numpy as np

from pyperfectforesight import v, process_model, solve_perfect_foresight


ALPHA = 0.36
BETA  = 0.99

K_SS = (ALPHA * BETA) ** (1 / (1 - ALPHA))
C_SS = K_SS**ALPHA - K_SS
SS   = np.array([C_SS, K_SS])

T = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rbc_model():
    """Standard RBC model (lags -1, 0, +1 only) — baseline sanity check."""
    eq1 = v("c", 0)**(-1) - BETA * ALPHA * v("k", 0)**(ALPHA-1) * v("c", 1)**(-1)
    eq2 = v("k", 0) - v("k", -1)**ALPHA + v("c", 0)
    return process_model([eq1, eq2], ["c", "k"])


def _lag2_model():
    """Variant that adds a lag-2 term to the capital equation.

    k_t = 0.5 * k_{t-1}^alpha + 0.5 * k_{t-2}^alpha - c_t

    Steady state is the same (at SS, 0.5*k_ss^a + 0.5*k_ss^a = k_ss^a = k_ss + c_ss ✓).
    """
    eq1 = v("c", 0)**(-1) - BETA * ALPHA * v("k", 0)**(ALPHA-1) * v("c", 1)**(-1)
    eq2 = (
        v("k", 0)
        - 0.5 * v("k", -1)**ALPHA
        - 0.5 * v("k", -2)**ALPHA
        + v("c", 0)
    )
    return process_model([eq1, eq2], ["c", "k"])


def _lead2_model():
    """Variant where the Euler equation references c at lead 2.

    1/c_t = beta^2 * alpha^2 * k_t^(2*(alpha-1)) / c_{t+2}

    (This is not economically motivated — just a test vehicle for lead > 1.)
    Steady state: 1/c_ss = beta^2 * alpha^2 * k_ss^(2*(alpha-1)) / c_ss
    → 1 = beta^2 * alpha^2 * k_ss^(2*(alpha-1))
    → k_ss = (alpha*beta)^(1/(1-alpha)) = K_SS  ✓
    """
    eq1 = (
        v("c", 0)**(-1)
        - BETA**2 * ALPHA**2 * v("k", 0)**(2*(ALPHA-1)) * v("c", 2)**(-1)
    )
    eq2 = v("k", 0) - v("k", -1)**ALPHA + v("c", 0)
    return process_model([eq1, eq2], ["c", "k"])


# ---------------------------------------------------------------------------
# 1. process_model detects arbitrary lags
# ---------------------------------------------------------------------------

def test_process_model_detects_lag2():
    model = _lag2_model()
    inc = model["incidence"]
    assert -2 in inc["k"], "lag -2 should appear in the incidence table for k"
    assert -1 in inc["k"]
    assert  0 in inc["k"]


def test_process_model_detects_lead2():
    model = _lead2_model()
    inc = model["incidence"]
    assert 2 in inc["c"], "lead +2 should appear in the incidence table for c"
    assert 1 not in inc["c"]   # lead 1 should NOT appear in this model


# ---------------------------------------------------------------------------
# 2. Standard mode: residual() and sparse_jacobian() handle lag > 1
# ---------------------------------------------------------------------------

def test_residual_lag2_at_steady_state():
    """residual() with lag-2 model returns ~0 at the steady-state path."""
    from pyperfectforesight import residual

    model = _lag2_model()
    X_ss = np.tile(SS, (T, 1))
    F = residual(
        X_ss, {}, model["all_syms"], model["residual_funcs"],
        model["vars_dyn"], model["dynamic_eqs"],
    )
    assert np.allclose(F, 0, atol=1e-10), f"||F|| = {np.linalg.norm(F):.2e}"


def test_sparse_jacobian_lag2_shape():
    """sparse_jacobian() returns correctly shaped matrix for lag-2 model."""
    from pyperfectforesight import sparse_jacobian

    model = _lag2_model()
    n = len(model["vars_dyn"])
    neq = len(model["dynamic_eqs"])
    X_ss = np.tile(SS, (T, 1))
    J = sparse_jacobian(
        X_ss, {}, model["all_syms"], model["block_funcs"],
        model["vars_dyn"], model["dynamic_eqs"],
    )
    assert J.shape == (neq * (T - 1), n * T)


def test_sparse_jacobian_lead2_shape():
    """sparse_jacobian() returns correctly shaped matrix for lead-2 model."""
    from pyperfectforesight import sparse_jacobian

    model = _lead2_model()
    n = len(model["vars_dyn"])
    neq = len(model["dynamic_eqs"])
    X_ss = np.tile(SS, (T, 1))
    J = sparse_jacobian(
        X_ss, {}, model["all_syms"], model["block_funcs"],
        model["vars_dyn"], model["dynamic_eqs"],
    )
    assert J.shape == (neq * (T - 1), n * T)


# ---------------------------------------------------------------------------
# 3. BVP mode: _residual_bvp and _jacobian_bvp handle lag > 1
# ---------------------------------------------------------------------------

def test_bvp_residual_lag2_at_steady_state():
    """_residual_bvp() with lag-2 model returns ~0 at the steady-state path."""
    from pyperfectforesight.core import _residual_bvp

    model = _lag2_model()
    X_ss = np.tile(SS, (T, 1))
    initval = SS.copy()   # k_{-1} = k_ss
    endval  = SS.copy()

    F = _residual_bvp(
        X_ss, {}, model["all_syms"], model["residual_funcs"],
        model["vars_dyn"], model["dynamic_eqs"],
        model["vars_exo"], None, initval, endval,
    )
    assert np.allclose(F, 0, atol=1e-10), f"||F|| = {np.linalg.norm(F):.2e}"


def test_bvp_jacobian_lag2_shape():
    """_jacobian_bvp() returns (T*neq, T*n) for lag-2 model."""
    from pyperfectforesight.core import _jacobian_bvp

    model = _lag2_model()
    n   = len(model["vars_dyn"])
    neq = len(model["dynamic_eqs"])
    X_ss = np.tile(SS, (T, 1))
    initval = SS.copy()
    endval  = SS.copy()

    J = _jacobian_bvp(
        X_ss, {}, model["all_syms"], model["block_funcs"],
        model["vars_dyn"], model["dynamic_eqs"],
        model["vars_exo"], None, initval, endval,
    )
    assert J.shape == (neq * T, n * T)


# ---------------------------------------------------------------------------
# 4. Full solve with lag-2 model (BVP mode)
# ---------------------------------------------------------------------------

def test_solve_lag2_bvp_small_shock():
    """solve_perfect_foresight converges for lag-2 model with a small shock.

    The BVP augmented path provides only one pre-sample boundary row (initval),
    so k_{-2} is clamped to k_{-1} = initval.  We interpret the shock as having
    occurred at t=-1: k_{-1} is set slightly above K_SS while earlier pre-sample
    values (t <= -2) are assumed equal to K_SS.  Clamping k_{-2} to k_{-1}
    is therefore intentional.  A UserWarning is expected because |lag| > 1.
    """
    model = _lag2_model()
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS * 1.05])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solve_perfect_foresight(
            T, {}, SS, model, model["vars_dyn"], X0,
            initial_state=k_neg1,
            stock_var_indices=[1],
        )
    assert any("BVP mode" in str(warning.message) for warning in w), \
        "Expected UserWarning about |lag| > 1 in BVP mode"
    assert sol.success, sol.message
    assert np.linalg.norm(sol.fun) < 1e-6


def test_solve_lead2_bvp_small_shock():
    """solve_perfect_foresight converges for lead-2 model with a small shock.

    A UserWarning is expected because the model has lead > 1.
    """
    model = _lead2_model()
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS * 1.05])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solve_perfect_foresight(
            T, {}, SS, model, model["vars_dyn"], X0,
            initial_state=k_neg1,
            stock_var_indices=[1],
        )
    assert any("BVP mode" in str(warning.message) for warning in w), \
        "Expected UserWarning about |lag| > 1 in BVP mode"
    assert sol.success, sol.message
    assert np.linalg.norm(sol.fun) < 1e-6


# ---------------------------------------------------------------------------
# 5. Backward-compatible: standard RBC (lags -1/0/+1) still works correctly
# ---------------------------------------------------------------------------

def test_standard_rbc_still_works():
    """Regression: the standard RBC model (lags in [-1,0,1]) is unaffected."""
    model = _rbc_model()
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS * 1.1])

    sol = solve_perfect_foresight(
        T, {}, SS, model, model["vars_dyn"], X0,
        initial_state=k_neg1,
        stock_var_indices=[1],
    )
    assert sol.success, sol.message
    assert np.linalg.norm(sol.fun) < 1e-6


# ---------------------------------------------------------------------------
# 6. UserWarning for |lag| > 1 in BVP mode
# ---------------------------------------------------------------------------


def test_bvp_warns_when_lag_gt_1():
    """BVP mode emits a UserWarning when the model has |lag| > 1."""
    model = _lag2_model()
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solve_perfect_foresight(
            T, {}, SS, model, model["vars_dyn"], X0,
            initial_state=k_neg1,
            stock_var_indices=[1],
        )

    bvp_warns = [x for x in w if issubclass(x.category, UserWarning) and "BVP mode" in str(x.message)]
    assert len(bvp_warns) == 1, f"Expected exactly one BVP UserWarning, got {len(bvp_warns)}"
    assert "endo_lags" in str(bvp_warns[0].message)
    assert "exo_lags" in str(bvp_warns[0].message)


def test_bvp_no_warning_for_standard_lags():
    """BVP mode does NOT emit a UserWarning for models with |lag| <= 1."""
    model = _rbc_model()
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS * 1.05])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solve_perfect_foresight(
            T, {}, SS, model, model["vars_dyn"], X0,
            initial_state=k_neg1,
            stock_var_indices=[1],
        )

    bvp_warns = [x for x in w if issubclass(x.category, UserWarning) and "BVP mode" in str(x.message)]
    assert len(bvp_warns) == 0, "Unexpected BVP UserWarning for a standard |lag| <= 1 model"


def test_bvp_warning_emitted_without_explicit_stock_var_indices():
    """The |lag| > 1 UserWarning fires even when stock_var_indices is not provided.

    BVP mode is always active; stock_var_indices is inferred from the incidence
    table.  For the lag-2 model, k appears at lag -2 and -1, so k is inferred
    as a stock variable and BVP mode is triggered — the warning must fire.
    """
    model = _lag2_model()
    X0 = np.tile(SS, (T, 1))
    k_neg1 = np.array([K_SS])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # stock_var_indices omitted → inferred; BVP still active → warning fires
        solve_perfect_foresight(
            T, {}, SS, model, model["vars_dyn"], X0,
            initial_state=k_neg1,
        )

    bvp_warns = [x for x in w if issubclass(x.category, UserWarning) and "BVP mode" in str(x.message)]
    assert len(bvp_warns) == 1, f"Expected exactly one BVP UserWarning, got {len(bvp_warns)}"
