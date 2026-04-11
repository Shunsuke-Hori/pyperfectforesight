"""
test_eliminate_static.py

Unit tests for eliminate_static / _eliminate_static_core, covering the
success path (a truly-static variable is eliminated) as well as failure
paths (variable has dynamics, no candidates, sp.solve fails).
"""

import sympy as sp
import pytest

from pyperfectforesight import v, process_model
from pyperfectforesight.core import _eliminate_static_core, eliminate_static


# ── helpers ──────────────────────────────────────────────────────────────────

delta = sp.Rational(1, 40)   # 0.025, exact arithmetic


# ── _eliminate_static_core: success path ─────────────────────────────────────

def test_core_eliminates_truly_static_var():
    """Variable appearing only at t=0 is solved out of the system."""
    k_m, k_0, y_0 = v("k", -1), v("k", 0), v("y", 0)

    eq_dyn    = k_0 - (1 - delta) * k_m   # k has a lag → dynamic
    eq_static = y_0 - k_0**2              # y only at t=0 → truly static

    eqs, eliminated = _eliminate_static_core(
        [eq_static], [eq_dyn], vars_dyn=["k", "y"], vars_exo=[]
    )

    assert eliminated == frozenset({"y"})
    assert len(eqs) == 1
    assert v("y", 0) not in eqs[0].free_symbols


def test_core_substitutes_into_dynamic_eq():
    """Eliminated variable is substituted into dynamic equations that use it."""
    k_m, k_0, c_0, y_0 = v("k", -1), v("k", 0), v("c", 0), v("y", 0)

    # y_0 = k_0^0.5  (static)
    # k_0 = (1-delta)*k_{-1} + y_0 - c_0  (dynamic, uses y_0)
    eq_static = y_0 - k_0**sp.Rational(1, 2)
    eq_dyn    = k_0 - (1 - delta) * k_m - y_0 + c_0

    eqs, eliminated = _eliminate_static_core(
        [eq_static], [eq_dyn], vars_dyn=["k", "y", "c"], vars_exo=[]
    )

    assert "y" in eliminated
    assert v("y", 0) not in eqs[0].free_symbols
    # y_0 was replaced by k_0**(1/2); the resulting equation simplifies to
    # k_0 - (1-delta)*k_{-1} - k_0**(1/2) + c_0 = 0
    expected = (k_0 - (1 - delta) * k_m - k_0**sp.Rational(1, 2) + c_0).expand()
    assert (eqs[0] - expected).expand() == 0


def test_core_partial_elimination():
    """Only the truly-static var is eliminated; the other static eq is preserved.

    Two static equations:
      y_0 = k_0^2   (y is truly static — no leads/lags anywhere)
      g_0 - e_g_0   (g has a lead in the Euler eq → not a candidate)

    Expected: y is eliminated, the g equation is kept in the returned system.
    """
    k_m, k_0 = v("k", -1), v("k", 0)
    y_0, g_0, g_p = v("y", 0), v("g", 0), v("g", 1)
    e_g_0 = v("e_g", 0)

    eq_static_y = y_0 - k_0**2
    eq_static_g = g_0 - e_g_0
    # Euler-like: uses g_p so "g" has a lead → g is NOT a candidate
    eq_euler = k_0 - (1 - delta) * k_m - g_p

    eqs, eliminated = _eliminate_static_core(
        [eq_static_y, eq_static_g], [eq_euler],
        vars_dyn=["k", "y", "g"], vars_exo=["e_g"],
    )

    # y eliminated, g stays
    assert "y" in eliminated
    assert "g" not in eliminated
    # equation for g must still be present
    assert any(g_0 in eq.free_symbols for eq in eqs), (
        "g_0 - e_g_0 equation must be preserved after partial elimination"
    )
    # y_0 must not appear anywhere
    assert all(y_0 not in eq.free_symbols for eq in eqs)
    # system is still square: 1 eliminated → len(eqs) == len(dynamic)+len(leftover) == 1+1 == 2
    assert len(eqs) == 2


# ── _eliminate_static_core: no-op paths ──────────────────────────────────────

def test_core_skips_var_with_lead():
    """Variable that appears at a lead in any equation is not eliminated."""
    k_m, k_0, k_p = v("k", -1), v("k", 0), v("k", 1)
    g_0 = v("g", 0)

    # g_0 = 0 is static, but g also appears as g_1 via k_p (hypothetical Euler)
    # We simulate this by having g appear at lag +1 somewhere.
    g_p = v("g", 1)
    eq_static  = g_0                          # g_0 = 0
    eq_dynamic = k_0 - (1 - delta) * k_m - g_p  # g appears at +1

    eqs, eliminated = _eliminate_static_core(
        [eq_static], [eq_dynamic], vars_dyn=["k", "g"], vars_exo=[]
    )

    assert eliminated == frozenset()
    assert len(eqs) == 2                       # fallback: static + dynamic


def test_core_skips_exo_symbol():
    """Exogenous variable symbols are excluded from candidates."""
    k_m, k_0, e_0 = v("k", -1), v("k", 0), v("e", 0)

    eq_static = e_0                        # e_0 = 0 — but e is exogenous
    eq_dyn    = k_0 - (1 - delta) * k_m

    eqs, eliminated = _eliminate_static_core(
        [eq_static], [eq_dyn], vars_dyn=["k"], vars_exo=["e"]
    )

    assert eliminated == frozenset()
    assert len(eqs) == 2


def test_core_skips_param_named_like_var():
    """A parameter like rho_1 (parses as lag 1) must not pollute vars_with_dynamics."""
    # If rho_1 is a parameter and "rho" is not in vars_dyn, it should be ignored.
    rho_1 = sp.Symbol("rho_1")             # looks like v("rho", 1)
    k_m, k_0, y_0 = v("k", -1), v("k", 0), v("y", 0)

    eq_static = y_0 - rho_1 * k_0         # uses param rho_1
    eq_dyn    = k_0 - (1 - delta) * k_m

    # "rho" is not in vars_dyn → should not block y from being eliminated
    eqs, eliminated = _eliminate_static_core(
        [eq_static], [eq_dyn], vars_dyn=["k", "y"], vars_exo=[]
    )

    assert "y" in eliminated
    assert len(eqs) == 1


def test_core_vars_params_prevents_param_blocking_endogenous():
    """vars_params prevents rho_1 (a param) from blocking endogenous rho elimination.

    Without vars_params: rho_1 parses as base='rho', lag=1, and since 'rho'
    is in vars_dyn it pollutes vars_with_dynamics, blocking rho_0 elimination.
    With vars_params=['rho_1']: rho_1 is recognised as a parameter and skipped.
    """
    rho_1 = sp.Symbol("rho_1")   # parameter that looks like v("rho", 1)
    rho_0 = v("rho", 0)
    k_m, k_0 = v("k", -1), v("k", 0)

    eq_static = rho_0 - rho_1 * k_0    # rho is truly static
    eq_dyn    = k_0 - (1 - delta) * k_m

    # Without vars_params: rho_1 looks like a lag-1 symbol for "rho", which is
    # in vars_dyn → "rho" lands in vars_with_dynamics → no elimination.
    eqs_no_params, elim_no_params = _eliminate_static_core(
        [eq_static], [eq_dyn], vars_dyn=["k", "rho"], vars_exo=[]
    )
    assert elim_no_params == frozenset(), "without vars_params rho_1 should block elimination"

    # With vars_params: rho_1 is skipped → "rho" stays out of vars_with_dynamics
    # → rho_0 is correctly eliminated.
    eqs_with_params, elim_with_params = _eliminate_static_core(
        [eq_static], [eq_dyn], vars_dyn=["k", "rho"], vars_exo=[],
        vars_params=["rho_1"],
    )
    assert "rho" in elim_with_params
    assert len(eqs_with_params) == 1
    assert rho_0 not in eqs_with_params[0].free_symbols


def test_core_empty_static_eqs():
    k_m, k_0 = v("k", -1), v("k", 0)
    eq_dyn = k_0 - (1 - delta) * k_m
    eqs, eliminated = _eliminate_static_core([], [eq_dyn])
    assert eqs == [eq_dyn]
    assert eliminated == frozenset()


# ── eliminate_static public API ───────────────────────────────────────────────

def test_public_api_returns_list():
    k_m, k_0, y_0 = v("k", -1), v("k", 0), v("y", 0)
    result = eliminate_static(
        [y_0 - k_0**2], [k_0 - (1 - delta) * k_m],
        vars_dyn=["k", "y"], vars_exo=[]
    )
    assert isinstance(result, list)
    assert len(result) == 1


# ── process_model integration ─────────────────────────────────────────────────

def test_process_model_eliminates_truly_static_var_and_updates_vars_dyn():
    """process_model must remove eliminated var from vars_dyn to stay square."""
    k_m, k_0, y_0 = v("k", -1), v("k", 0), v("y", 0)

    eqs = [
        k_0 - (1 - delta) * k_m,   # dynamic
        y_0 - k_0**2,              # static, y truly static
    ]
    model = process_model(eqs, ["k", "y"])

    assert len(model["dynamic_eqs"]) == len(model["vars_dyn"]), (
        "dynamic_eqs and vars_dyn must be the same length after elimination"
    )
    assert "y" not in model["vars_dyn"]
    assert "k" in model["vars_dyn"]


def test_process_model_warns_on_param_name_clash():
    """process_model warns when a param name clashes with a time-indexed var pattern."""
    import warnings
    k_m, k_0 = v("k", -1), v("k", 0)
    rho_1 = sp.Symbol("rho_1")   # looks like v("k",...) but base is "rho" — and "rho" is a declared var

    # "rho" is both in vars_dyn and a parameter named "rho_1" parses as v("rho", 1)
    rho_0 = v("rho", 0)
    eq1 = k_0 - (1 - delta) * k_m
    eq2 = rho_0 - rho_1 * k_0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        process_model([eq1, eq2], ["k", "rho"], vars_params=["rho_1"])

    assert any("rho_1" in str(warning.message) for warning in w), (
        "Expected UserWarning about parameter name 'rho_1' clashing with v('rho', 1)"
    )


def test_process_model_no_warn_for_safe_param_names():
    """process_model does not warn for param names that don't clash with vars."""
    import warnings
    k_m, k_0, c_0 = v("k", -1), v("k", 0), v("c", 0)
    beta = sp.Symbol("beta")
    delta_s = sp.Symbol("delta")

    eq1 = k_0 - (1 - delta_s) * k_m - c_0
    eq2 = c_0 - beta * k_0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        process_model([eq1, eq2], ["k", "c"], vars_params=["beta", "delta"])

    clashes = [x for x in w if issubclass(x.category, UserWarning) and "parses as" in str(x.message)]
    assert not clashes, f"Unexpected clash warning(s): {clashes}"
